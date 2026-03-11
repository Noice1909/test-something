"""Reflection Specialist — analyzes failures and recommends retry strategies."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import ReflectionResult, RetryStrategy, SpecialistResult
from src.agents.state import AgentState
from src.agents.utils import extract_text
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

# ── Error-based reflection (query execution failed) ─────────────────────────

_REFLECTION_PROMPT = """\
A graph database query failed. Analyze the failure and recommend a retry \
strategy.

## Original Question
{question}

## Strategy Used
{strategy}

## Generated Query
{query}

## Error
{error}
Category: {error_category}

## Attempt Number
{attempt} of {max_attempts}

## Retry Strategies
- expand_discovery: Search for entities with broader/different terms
- simplify_query: Reduce query complexity (fewer hops, simpler patterns)
- add_traversals: Try additional relationship traversals
- change_schema: Use different node labels / relationship types
- give_up: Return a fallback answer

Return a JSON object with:
- "should_retry": bool
- "retry_strategy": one of the above strategies
- "reasoning": why this strategy was chosen
- "fallback_answer": if giving up, provide a helpful message (else null)

Return ONLY the JSON object:"""

# ── Empty-result reflection (query succeeded but 0 rows) ────────────────────

_EMPTY_RESULT_PROMPT = """\
A graph database query executed successfully but returned 0 rows. Your job is \
to analyze WHY it returned nothing and suggest a concrete next step based on \
the actual database schema.

## User Question
{question}

## Query That Returned 0 Rows
{query}

## Reasoning for That Query
{reasoning}

## Database Schema (actual labels and relationships in the DB)
Labels: {schema_labels}
Relationship Types: {schema_rels}
Relationship Patterns (how labels connect):
{schema_patterns}

## Previous Attempts That Also Returned 0 Rows
{previous_attempts}

## Your Task
Look at the SCHEMA PATTERNS above. Based on the actual relationships that \
exist in the database, reason step by step:

1. Could the wrong property name have been used? (e.g. "name" vs "title")
2. Was the relationship direction reversed?
3. Were the WHERE filters too restrictive?
4. Are there connecting paths through INTERMEDIATE nodes?
   For example, if the schema has (Person)-[ACTED_IN]->(Movie) and \
(Movie)-[IN_GENRE]->(Genre), then Person connects to Genre via Movie.
5. Has every reasonable approach already been tried?

## Rules
- Start simple: suggest a 1-hop approach with corrected labels/properties first
- Only suggest more hops if shorter approaches have already been tried
- Do NOT repeat any query from "Previous Attempts"
- If all reasonable approaches have been exhausted, set should_retry to false

Return a JSON object with:
- "should_retry": bool
- "next_approach": a concrete description of what query to try next \
(e.g. "Match Person via ACTED_IN to Movie, then filter Movie.title")
- "suggested_hops": number (1, 2, or 3)
- "reasoning": step-by-step analysis of why the previous query was empty
- "fallback_answer": if should_retry is false, a user-friendly message \
confirming no matching data exists (else null)

Return ONLY the JSON object:"""


class ReflectionSpecialist:
    """Analyzes failures and recommends retry strategies or fallback answers."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

    # ── Error-based reflection ───────────────────────────────────────────

    async def run(self, state: AgentState, max_attempts: int = 3) -> SpecialistResult:
        t0 = time.time()
        try:
            # If we've exhausted attempts, give up
            if state.attempt_number >= max_attempts:
                reflection = ReflectionResult(
                    should_retry=False,
                    retry_strategy=RetryStrategy.GIVE_UP,
                    reasoning=f"Max attempts ({max_attempts}) reached",
                    fallback_answer=(
                        f"I wasn't able to find a definitive answer for \"{state.question}\". "
                        f"The database query encountered issues after {max_attempts} attempts. "
                        "Please try rephrasing your question or ask about specific entities."
                    ),
                )
                state.reflection = reflection
                dur = (time.time() - t0) * 1000
                state.log_specialist("reflection", success=True, duration_ms=dur, detail="give_up")
                return SpecialistResult(success=True, data=reflection, duration_ms=dur)

            prompt = _REFLECTION_PROMPT.format(
                question=state.question,
                strategy=state.strategy.value,
                query=state.generated_query.query or "None generated",
                error=state.execution_result.error or "Unknown",
                error_category=state.execution_result.error_category or "unknown",
                attempt=state.attempt_number,
                max_attempts=max_attempts,
            )

            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
            reflection = self._parse_response(text)
            state.reflection = reflection

            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "reflection", success=True, duration_ms=dur,
                detail=f"strategy={reflection.retry_strategy.value}, retry={reflection.should_retry}",
            )
            return SpecialistResult(success=True, data=reflection, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("reflection", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    # ── Empty-result reflection (schema-driven) ──────────────────────────

    async def run_empty_result(
        self, state: AgentState, max_empty_retries: int = 2
    ) -> SpecialistResult:
        """Analyze why a query returned 0 rows and suggest next approach."""
        t0 = time.time()
        try:
            # If exhausted empty retries, conclude data is absent
            if state.empty_retries_used >= max_empty_retries:
                reflection = ReflectionResult(
                    should_retry=False,
                    retry_strategy=RetryStrategy.GIVE_UP,
                    reasoning=(
                        f"Tried {state.empty_retries_used} alternative approaches, "
                        "all returned 0 rows — data likely absent"
                    ),
                    fallback_answer=(
                        f"After trying multiple query approaches, no matching data "
                        f"was found in the database for \"{state.question}\". "
                        "The data you're looking for may not exist in the current database."
                    ),
                )
                state.reflection = reflection
                dur = (time.time() - t0) * 1000
                state.log_specialist(
                    "reflection_empty", success=True, duration_ms=dur,
                    detail="exhausted_empty_retries",
                )
                return SpecialistResult(success=True, data=reflection, duration_ms=dur)

            # Fetch live schema for the LLM to reason about
            schema = await self._db.get_schema()

            patterns_text = "\n".join(
                f"  ({p['from']})-[{p['type']}]->({p['to']})"
                for p in schema.get("relationship_patterns", [])[:40]
            ) or "None"

            previous_text = "\n".join(
                f"  {i+1}. {pq.get('query', 'N/A')} → 0 rows"
                f"\n     Reasoning: {pq.get('reasoning', 'N/A')}"
                for i, pq in enumerate(state.previous_empty_queries)
            ) or "None (first attempt)"

            prompt = _EMPTY_RESULT_PROMPT.format(
                question=state.question,
                query=state.generated_query.query,
                reasoning=state.generated_query.reasoning,
                schema_labels=", ".join(schema.get("labels", [])),
                schema_rels=", ".join(schema.get("relationship_types", [])),
                schema_patterns=patterns_text,
                previous_attempts=previous_text,
            )

            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
            reflection = self._parse_empty_response(text)
            state.reflection = reflection

            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "reflection_empty", success=True, duration_ms=dur,
                detail=(
                    f"retry={reflection.should_retry}, "
                    f"next={reflection.next_approach[:60]}"
                ),
            )
            return SpecialistResult(success=True, data=reflection, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "reflection_empty", success=False, duration_ms=dur, detail=str(exc)
            )
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    # ── Parsers ──────────────────────────────────────────────────────────

    def _parse_response(self, text: str) -> ReflectionResult:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return ReflectionResult(
                should_retry=True,
                retry_strategy=RetryStrategy.SIMPLIFY_QUERY,
                reasoning="LLM parse failed, defaulting to simplify",
            )

        strategy_map = {
            "expand_discovery": RetryStrategy.EXPAND_DISCOVERY,
            "simplify_query": RetryStrategy.SIMPLIFY_QUERY,
            "add_traversals": RetryStrategy.ADD_TRAVERSALS,
            "change_schema": RetryStrategy.CHANGE_SCHEMA,
            "give_up": RetryStrategy.GIVE_UP,
        }

        strategy = strategy_map.get(
            data.get("retry_strategy", "simplify_query"), RetryStrategy.SIMPLIFY_QUERY
        )

        return ReflectionResult(
            should_retry=data.get("should_retry", True) and strategy != RetryStrategy.GIVE_UP,
            retry_strategy=strategy,
            reasoning=data.get("reasoning", ""),
            fallback_answer=data.get("fallback_answer"),
        )

    def _parse_empty_response(self, text: str) -> ReflectionResult:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return ReflectionResult(
                should_retry=True,
                retry_strategy=RetryStrategy.CHANGE_SCHEMA,
                reasoning="LLM parse failed, will try different schema approach",
                next_approach="Try a simpler query with fewer constraints",
            )

        should_retry = data.get("should_retry", True)

        return ReflectionResult(
            should_retry=should_retry,
            retry_strategy=RetryStrategy.GIVE_UP if not should_retry else RetryStrategy.CHANGE_SCHEMA,
            reasoning=data.get("reasoning", ""),
            fallback_answer=data.get("fallback_answer"),
            next_approach=data.get("next_approach", ""),
        )

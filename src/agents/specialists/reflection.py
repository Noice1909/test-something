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


class ReflectionSpecialist:
    """Analyzes failures and recommends retry strategies or fallback answers."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

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

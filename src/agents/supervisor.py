"""Supervisor — main orchestrator with hybrid strategy decision-making."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import (
    AgenticResponse,
    RetryStrategy,
    StrategyType,
    SupervisorDecision,
)
from src.agents.state import AgentState
from src.agents.specialists.discovery import DiscoverySpecialist
from src.agents.specialists.schema_planning import SchemaPlanningSpecialist
from src.agents.specialists.schema_reasoning import SchemaReasoningSpecialist
from src.agents.specialists.query_planning import QueryPlanningSpecialist
from src.agents.specialists.query_generation import QueryGenerationSpecialist
from src.agents.specialists.execution import ExecutionSpecialist
from src.agents.specialists.reflection import ReflectionSpecialist
from src.agents.utils import extract_text
from src.agents.conversation import ConversationManager
from src.cache.cache_manager import CacheManager
from src.cache.cache_keys import response_key, strategy_key
from src.database.abstract import AbstractDatabase
from src.security.prompt_guard import detect_injection, sanitize, wrap_user_input
from src.config import settings

logger = logging.getLogger(__name__)

_STRATEGY_PROMPT = """\
You are the supervisor of a graph database query system. Analyze the user's \
question and choose the best strategy.

## Question
<<<USER_INPUT>>>
{question}
<<<END_USER_INPUT>>>

{conversation_context}

## Available Strategies
1. discovery_first — Unknown terms, acronyms, or ambiguous entities that need \
   to be found in the database first.
2. direct_query — Entities are clearly mentioned and easy to match \
   (e.g. "show Application named ABC").
3. schema_exploration — Questions about database structure, relationships, \
   or connectivity (e.g. "what labels exist?").
4. aggregation — Ranking, top-N, most/least queries with ordering \
   (e.g. "top 5 actors by movie count", "which director has the most movies").
5. property_lookup — Looking up a specific entity by a known property value \
   (e.g. "show application named X", "find movie titled Inception").
6. simple_count — Simple count queries with no entity discovery needed \
   (e.g. "how many applications are there?", "total number of movies").
7. label_list — List or enumerate all entities of a type \
   (e.g. "list all domains", "show all applications", "what are the genres?").
8. entity_detail — Describe or get details about a specific entity \
   (e.g. "tell me about Tom Hanks", "describe the CNAPP application").
9. relationship_query — Questions about what connects to or depends on something \
   (e.g. "what depends on Domain X?", "what is linked to Application Y?").
10. path_query — Find how two entities are connected or the path between them \
    (e.g. "how is Person X connected to Movie Y?", "shortest path between A and B").
11. comparison — Compare two or more entities \
    (e.g. "compare Application A and Application B").

Return a JSON object with:
- "strategy": one of the strategy names above (e.g. "discovery_first", "simple_count")
- "reasoning": brief explanation of why

Return ONLY the JSON object:"""


# ── Rule-based strategy classifier ────────────────────────────────────────────

# Ordered most-specific first; checked sequentially
_STRATEGY_RULES: list[tuple[StrategyType, re.Pattern[str]]] = [
    (StrategyType.SCHEMA_EXPLORATION, re.compile(
        r"\b(schema|structure|what labels|what relationships|what types|database structure|what nodes)\b", re.I)),
    (StrategyType.PATH_QUERY, re.compile(
        r"\b(path between|how .{1,30} connected to|shortest path|route from .{1,30} to)\b", re.I)),
    (StrategyType.COMPARISON, re.compile(
        r"\b(compare|versus|vs\.?|difference between|differ from)\b", re.I)),
    (StrategyType.AGGREGATION, re.compile(
        r"\b(most|top \d|average|sum of|highest|lowest|rank|best|worst|least|order by)\b", re.I)),
    (StrategyType.SIMPLE_COUNT, re.compile(
        r"\b(how many|count of|number of|total number)\b", re.I)),
    (StrategyType.PROPERTY_LOOKUP, re.compile(
        r"\b(named|called|with name|titled|where name)\b", re.I)),
    (StrategyType.ENTITY_DETAIL, re.compile(
        r"\b(describe|tell me about|details of|detail of|info about|information about)\b", re.I)),
    (StrategyType.RELATIONSHIP_QUERY, re.compile(
        r"\b(connected to|relates to|linked to|depends on|belongs to|associated with|runs on|part of)\b", re.I)),
    (StrategyType.LABEL_LIST, re.compile(
        r"\b(list all|show all|get all|show me all|what are the|which .{0,20} are)\b", re.I)),
]


def _classify_strategy(question: str) -> tuple[StrategyType, str, str]:
    """Classify question using keyword heuristics.

    Returns ``(strategy, reasoning, confidence)`` where confidence is
    ``"high"`` when exactly one pattern matches, or ``"low"`` when zero
    or multiple conflicting patterns match.
    """
    matches: list[tuple[StrategyType, str]] = []
    for strategy, pattern in _STRATEGY_RULES:
        m = pattern.search(question)
        if m:
            matches.append((strategy, m.group(0)))

    if len(matches) == 1:
        strat, matched = matches[0]
        return strat, f"Matched pattern: '{matched}'", "high"

    if len(matches) > 1:
        # Multiple matches — pick the first (most-specific) but mark low confidence
        strat, matched = matches[0]
        others = ", ".join(s.value for s, _ in matches[1:])
        return strat, f"Multiple matches: '{matched}' (also: {others})", "low"

    # No matches
    return StrategyType.DISCOVERY_FIRST, "No pattern matched", "low"

_ANSWER_PROMPT = """\
You are answering a user's question using graph database query results. \
Your job is to give a DIRECT, clear answer.

## Question
<<<USER_INPUT>>>
{question}
<<<END_USER_INPUT>>>

## Query Results
{results}

## Rules
1. Answer DIRECTLY — state the answer in the very first sentence.
2. NEVER start with "Based on the provided information" or similar hedging.
3. NEVER say "there is not" or decline if the results DO contain data.
4. If results have data, summarize it clearly and concisely.
5. If results are empty, say so simply: "No matching data was found."
6. Use natural language, not technical jargon.
7. Keep the answer concise — 1 to 3 sentences for simple queries.

Answer:"""

# Maps strategy → ordered specialist sequence
_STRATEGY_SEQUENCES: dict[StrategyType, list[str]] = {
    # Original strategies (now using merged "schema_planning")
    StrategyType.DISCOVERY_FIRST: [
        "discovery", "schema_planning", "query_generation", "execution",
    ],
    StrategyType.DIRECT_QUERY: [
        "schema_planning", "query_generation", "execution",
    ],
    StrategyType.SCHEMA_EXPLORATION: [
        "schema_planning", "discovery", "query_generation", "execution",
    ],
    StrategyType.AGGREGATION: [
        "discovery", "schema_planning", "query_generation", "execution",
    ],
    # Narrowed strategies — shorter sequences for specific query types
    StrategyType.PROPERTY_LOOKUP: [
        "discovery", "query_generation", "execution",
    ],
    StrategyType.SIMPLE_COUNT: [
        "schema_planning", "query_generation", "execution",
    ],
    StrategyType.LABEL_LIST: [
        "schema_planning", "query_generation", "execution",
    ],
    StrategyType.ENTITY_DETAIL: [
        "discovery", "query_generation", "execution",
    ],
    StrategyType.RELATIONSHIP_QUERY: [
        "discovery", "schema_planning", "query_generation", "execution",
    ],
    StrategyType.PATH_QUERY: [
        "discovery", "schema_planning", "query_generation", "execution",
    ],
    StrategyType.COMPARISON: [
        "discovery", "schema_planning", "query_generation", "execution",
    ],
}

# For empty-result retries, re-run from merged planning onward
_EMPTY_RETRY_SEQUENCE: list[str] = [
    "schema_planning", "query_generation", "execution",
]


class Supervisor:
    """LLM-based orchestrator that coordinates specialist agents."""

    def __init__(
        self,
        db: AbstractDatabase,
        llm: BaseChatModel,
        tools: dict[str, Any],
        max_attempts: int = 3,
        max_empty_retries: int = 2,
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools
        self._max_attempts = max_attempts
        self._max_empty_retries = max_empty_retries

        # Create specialist instances
        self._specialists: dict[str, Any] = {
            "discovery": DiscoverySpecialist(db, llm, tools),
            "schema_planning": SchemaPlanningSpecialist(db, llm, tools),
            "query_generation": QueryGenerationSpecialist(db, llm, tools),
            "execution": ExecutionSpecialist(db, tools),
            "reflection": ReflectionSpecialist(db, llm, tools),
            # Keep old specialists for backward compat
            "schema_reasoning": SchemaReasoningSpecialist(db, llm, tools),
            "query_planning": QueryPlanningSpecialist(db, llm, tools),
        }

        # Caching and conversation
        self._cache = CacheManager()
        self._conversation = ConversationManager()

        logger.info("Supervisor initialized with %d specialists", len(self._specialists))

    async def process_question(
        self,
        question: str,
        trace_id: str | None = None,
        conversation_id: str | None = None,
    ) -> AgenticResponse:
        """Process a user question through the agentic pipeline."""

        # ── Prompt injection check ──
        is_suspicious, reason = detect_injection(question)
        if is_suspicious:
            logger.warning("Prompt injection detected (reason=%s), sanitizing input", reason)
        clean_question = sanitize(question)

        # ── Response cache check ──
        cache_key = response_key(clean_question)
        cached = await self._cache.get(cache_key)
        if cached and conversation_id is None:
            logger.info("Cache HIT for question")
            return AgenticResponse(**cached, from_cache=True)

        # ── State setup ──
        state = AgentState(question=clean_question)
        if trace_id:
            state.trace_id = trace_id

        # ── Conversation context ──
        conv_id = await self._conversation.prepare_state(state, conversation_id)

        logger.info("[%s] Processing: %s", state.trace_id, clean_question[:100])

        t0 = time.time()

        for attempt in range(1, self._max_attempts + 1):
            state.attempt_number = attempt

            # 1) Decide strategy
            if attempt == 1:
                decision = await self._decide_strategy(clean_question, state)
                state.strategy = decision.strategy
                logger.info("[%s] Strategy: %s (%s)",
                            state.trace_id, decision.strategy.value, decision.reasoning)
            else:
                logger.info("[%s] Retry attempt %d", state.trace_id, attempt)

            # 2) Execute specialist sequence
            sequence = _STRATEGY_SEQUENCES.get(state.strategy, _STRATEGY_SEQUENCES[StrategyType.DISCOVERY_FIRST])
            success = await self._execute_sequence(state, sequence)

            # 3) Check results
            if success and state.execution_result.success:
                if state.execution_result.rows:
                    # Got data — format a clean user-facing answer
                    answer = await self._format_answer(clean_question, state.execution_result.rows)
                    response = AgenticResponse(
                        answer=answer,
                        strategy_used=state.strategy.value,
                        attempts=attempt,
                        success=True,
                        trace_id=state.trace_id,
                        specialist_log=state.history,
                        cypher_attempts=state.cypher_attempts,
                        conversation_id=conv_id,
                    )
                    # Cache + checkpoint
                    await self._cache.set(cache_key, {
                        "answer": answer,
                        "strategy_used": state.strategy.value,
                        "attempts": attempt,
                        "success": True,
                        "trace_id": state.trace_id,
                        "specialist_log": state.history,
                        "cypher_attempts": state.cypher_attempts,
                    }, settings.cache_response_ttl)
                    await self._conversation.save_turn(conv_id, state, answer)
                    return response

                # ── Empty results — try alternative approaches gradually ──
                empty_result = await self._handle_empty_results(state, clean_question)
                if empty_result is not None:
                    empty_result.conversation_id = conv_id
                    await self._conversation.save_turn(conv_id, state, empty_result.answer)
                    return empty_result
                continue

            # 4) Query failed (execution error) — reflect
            await self._specialists["reflection"].run(state, self._max_attempts)

            if not state.reflection.should_retry:
                fallback_answer = state.reflection.fallback_answer or (
                    f"I wasn't able to answer \"{clean_question}\" at this time. "
                    "Please try rephrasing your question."
                )
                response = AgenticResponse(
                    answer=fallback_answer,
                    strategy_used=state.strategy.value,
                    attempts=attempt,
                    success=False,
                    trace_id=state.trace_id,
                    specialist_log=state.history,
                    cypher_attempts=state.cypher_attempts,
                    conversation_id=conv_id,
                )
                await self._conversation.save_turn(conv_id, state, fallback_answer)
                return response

            # Apply retry adjustments
            self._apply_retry_strategy(state)

        # Safety fallback
        fallback_answer = (
            f"I could not determine an answer for \"{clean_question}\" "
            f"after {self._max_attempts} attempts. "
            "Please try a different phrasing."
        )
        response = AgenticResponse(
            answer=fallback_answer,
            strategy_used=state.strategy.value,
            attempts=self._max_attempts,
            success=False,
            trace_id=state.trace_id,
            specialist_log=state.history,
            cypher_attempts=state.cypher_attempts,
            conversation_id=conv_id,
        )
        await self._conversation.save_turn(conv_id, state, fallback_answer)
        return response

    # ── Empty-result handler (gradual retry) ─────────────────────────────

    async def _handle_empty_results(
        self, state: AgentState, question: str,
    ) -> AgenticResponse | None:
        """Handle queries that succeeded but returned 0 rows."""
        logger.info(
            "[%s] Query returned 0 rows (empty retry %d/%d)",
            state.trace_id, state.empty_retries_used + 1, self._max_empty_retries,
        )

        state.previous_empty_queries.append({
            "query": state.generated_query.query,
            "reasoning": state.generated_query.reasoning,
        })

        if state.empty_retries_used >= self._max_empty_retries:
            return AgenticResponse(
                answer=(
                    f"After trying {state.empty_retries_used + 1} different query "
                    f"approaches, no matching data was found for \"{question}\". "
                    "The data you're looking for may not exist in the current database."
                ),
                strategy_used=state.strategy.value,
                attempts=state.attempt_number,
                success=True,
                trace_id=state.trace_id,
                specialist_log=state.history,
                cypher_attempts=state.cypher_attempts,
            )

        state.empty_retries_used += 1

        await self._specialists["reflection"].run_empty_result(
            state, self._max_empty_retries,
        )

        if not state.reflection.should_retry:
            return AgenticResponse(
                answer=state.reflection.fallback_answer or (
                    f"No matching data found for \"{question}\" in the database."
                ),
                strategy_used=state.strategy.value,
                attempts=state.attempt_number,
                success=True,
                trace_id=state.trace_id,
                specialist_log=state.history,
                cypher_attempts=state.cypher_attempts,
            )

        logger.info(
            "[%s] Empty-result retry: %s",
            state.trace_id, state.reflection.next_approach[:80],
        )
        success = await self._execute_sequence(state, _EMPTY_RETRY_SEQUENCE)

        if success and state.execution_result.success and state.execution_result.rows:
            answer = await self._format_answer(question, state.execution_result.rows)
            return AgenticResponse(
                answer=answer,
                strategy_used=state.strategy.value,
                attempts=state.attempt_number,
                success=True,
                trace_id=state.trace_id,
                specialist_log=state.history,
                cypher_attempts=state.cypher_attempts,
            )

        return None

    # ── Strategy decision ────────────────────────────────────────────────

    async def _decide_strategy(self, question: str, state: AgentState) -> SupervisorDecision:
        """Hybrid strategy: rule-based for clear queries, LLM fallback for ambiguous."""
        # 1. Check strategy cache
        s_key = strategy_key(question)
        cached = await self._cache.get(s_key)
        if cached:
            try:
                return SupervisorDecision(
                    strategy=StrategyType(cached["strategy"]),
                    reasoning=cached.get("reasoning", "cached"),
                )
            except Exception as exc:
                logger.debug("Ignoring malformed cached strategy, will recompute: %s", exc)

        # 2. Try rule-based classification (instant, ~0ms)
        strategy, reasoning, confidence = _classify_strategy(question)

        if confidence == "high":
            logger.info("[%s] Rule-based strategy: %s (%s)", state.trace_id, strategy.value, reasoning)
            decision = SupervisorDecision(
                strategy=strategy,
                reasoning=f"[rule] {reasoning}",
                specialist_sequence=list(_STRATEGY_SEQUENCES.get(strategy, [])),
            )
            await self._cache.set(s_key, {
                "strategy": strategy.value,
                "reasoning": decision.reasoning,
            }, settings.cache_strategy_ttl)
            return decision

        # 3. Low confidence — fall back to LLM with all 11 strategies
        logger.info("[%s] Rule-based uncertain (%s), falling back to LLM", state.trace_id, reasoning)
        return await self._llm_decide_strategy(question, state, s_key)

    async def _llm_decide_strategy(
        self, question: str, state: AgentState, cache_key: str,
    ) -> SupervisorDecision:
        """LLM fallback for ambiguous queries — uses full 11-strategy prompt."""
        conv_context = ""
        if state.previous_context:
            conv_context = "## Conversation History\n"
            for turn in state.previous_context:
                conv_context += f"- Q: {turn['question'][:100]}\n  A: {turn['answer'][:100]}\n"

        prompt = _STRATEGY_PROMPT.format(question=question, conversation_context=conv_context)
        try:
            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])

            # Map all 11 strategies
            strategy_str = data.get("strategy", "discovery_first")
            try:
                strategy = StrategyType(strategy_str)
            except ValueError:
                strategy = StrategyType.DISCOVERY_FIRST

            decision = SupervisorDecision(
                strategy=strategy,
                reasoning=f"[llm] {data.get('reasoning', '')}",
                specialist_sequence=list(_STRATEGY_SEQUENCES.get(strategy, [])),
            )

            await self._cache.set(cache_key, {
                "strategy": strategy.value,
                "reasoning": decision.reasoning,
            }, settings.cache_strategy_ttl)

            return decision
        except Exception as exc:
            logger.warning("LLM strategy decision failed: %s — defaulting to discovery_first", exc)
            return SupervisorDecision(
                strategy=StrategyType.DISCOVERY_FIRST,
                reasoning=f"Default (LLM error: {exc})",
            )

    async def _execute_sequence(
        self, state: AgentState, sequence: list[str]
    ) -> bool:
        for specialist_name in sequence:
            specialist = self._specialists.get(specialist_name)
            if not specialist:
                logger.warning("Unknown specialist: %s", specialist_name)
                continue

            result = await specialist.run(state)
            if not result.success:
                logger.warning("[%s] %s failed: %s",
                               state.trace_id, specialist_name, result.error)
                if specialist_name == "discovery":
                    continue
                return False

        return True

    async def _format_answer(self, question: str, rows: list[dict]) -> str:
        results_text = json.dumps(rows[:20], indent=2, default=str)
        if len(results_text) > 3000:
            results_text = results_text[:3000] + "\n... (truncated)"

        prompt = _ANSWER_PROMPT.format(question=question, results=results_text)
        try:
            response = await self._llm.ainvoke(prompt)
            return extract_text(response)
        except Exception as exc:
            logger.warning("Answer formatting failed: %s", exc)
            return f"Found {len(rows)} results:\n" + "\n".join(
                str(row) for row in rows[:10]
            )

    def _apply_retry_strategy(self, state: AgentState) -> None:
        strategy = state.reflection.retry_strategy
        if strategy == RetryStrategy.EXPAND_DISCOVERY:
            state.discoveries.clear()
        elif strategy == RetryStrategy.SIMPLIFY_QUERY:
            state.generated_query.query = ""
        elif strategy == RetryStrategy.CHANGE_SCHEMA:
            state.schema_selection.node_labels.clear()
            state.schema_selection.relationship_types.clear()
        # ADD_TRAVERSALS: keep state as-is, let planner adjust

    async def shutdown(self) -> None:
        """Cleanup cache and conversation stores."""
        await self._cache.close()
        await self._conversation.close()

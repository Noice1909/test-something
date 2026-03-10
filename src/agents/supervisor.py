"""Supervisor — main orchestrator with LLM-based strategy decision-making."""

from __future__ import annotations

import json
import logging
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
from src.agents.specialists.schema_reasoning import SchemaReasoningSpecialist
from src.agents.specialists.query_planning import QueryPlanningSpecialist
from src.agents.specialists.query_generation import QueryGenerationSpecialist
from src.agents.specialists.execution import ExecutionSpecialist
from src.agents.specialists.reflection import ReflectionSpecialist
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_STRATEGY_PROMPT = """\
You are the supervisor of a graph database query system. Analyze the user's \
question and choose the best strategy.

## Question
{question}

## Available Strategies
1. discovery_first — Use when the question contains unknown terms, acronyms, \
   or ambiguous entities that need to be found in the database first.
2. direct_query — Use when entities are clearly mentioned and easy to match \
   (e.g. "show Application named ABC").
3. schema_exploration — Use when asking about database structure, relationships, \
   or connectivity (e.g. "what's connected to Domain nodes?").
4. aggregation — Use for count/sum/avg queries (e.g. "how many applications?").

Return a JSON object with:
- "strategy": one of discovery_first, direct_query, schema_exploration, aggregation
- "reasoning": brief explanation of why

Return ONLY the JSON object:"""

_ANSWER_PROMPT = """\
Based on these query results, provide a concise natural-language answer to the \
user's question.

## Question
{question}

## Results
{results}

Provide a helpful, conversational answer:"""

# Maps strategy → ordered specialist sequence
_STRATEGY_SEQUENCES: dict[StrategyType, list[str]] = {
    StrategyType.DISCOVERY_FIRST: [
        "discovery", "schema_reasoning", "query_planning",
        "query_generation", "execution",
    ],
    StrategyType.DIRECT_QUERY: [
        "schema_reasoning", "query_planning",
        "query_generation", "execution",
    ],
    StrategyType.SCHEMA_EXPLORATION: [
        "schema_reasoning", "discovery",
        "query_planning", "query_generation", "execution",
    ],
    StrategyType.AGGREGATION: [
        "discovery", "schema_reasoning", "query_planning",
        "query_generation", "execution",
    ],
}


class Supervisor:
    """LLM-based orchestrator that coordinates specialist agents."""

    def __init__(
        self,
        db: AbstractDatabase,
        llm: BaseChatModel,
        tools: dict[str, Any],
        max_attempts: int = 3,
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools
        self._max_attempts = max_attempts

        # Create specialist instances
        self._specialists: dict[str, Any] = {
            "discovery": DiscoverySpecialist(db, llm, tools),
            "schema_reasoning": SchemaReasoningSpecialist(db, llm, tools),
            "query_planning": QueryPlanningSpecialist(db, llm, tools),
            "query_generation": QueryGenerationSpecialist(db, llm, tools),
            "execution": ExecutionSpecialist(db, tools),
            "reflection": ReflectionSpecialist(db, llm, tools),
        }
        logger.info("Supervisor initialized with %d specialists", len(self._specialists))

    async def process_question(
        self, question: str, trace_id: str | None = None
    ) -> AgenticResponse:
        """Process a user question through the agentic pipeline."""
        state = AgentState(question=question)
        if trace_id:
            state.trace_id = trace_id

        logger.info("[%s] Processing: %s", state.trace_id, question)

        for attempt in range(1, self._max_attempts + 1):
            state.attempt_number = attempt

            # 1) Decide strategy
            if attempt == 1:
                decision = await self._decide_strategy(question)
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
                # Query succeeded — format answer
                if state.execution_result.rows:
                    answer = await self._format_answer(question, state.execution_result.rows)
                else:
                    answer = f"The query executed successfully but returned no results for \"{question}\"."

                return AgenticResponse(
                    answer=answer,
                    strategy_used=state.strategy.value,
                    attempts=attempt,
                    success=True,
                    trace_id=state.trace_id,
                    specialist_log=state.history,
                )

            # 4) Query failed — reflect
            reflection_result = await self._specialists["reflection"].run(state, self._max_attempts)

            if not state.reflection.should_retry:
                # Give up with fallback
                return AgenticResponse(
                    answer=state.reflection.fallback_answer or f"Unable to answer: {question}",
                    strategy_used=state.strategy.value,
                    attempts=attempt,
                    success=False,
                    trace_id=state.trace_id,
                    specialist_log=state.history,
                )

            # Apply retry adjustments
            self._apply_retry_strategy(state)

        # Should not reach here, but safety fallback
        return AgenticResponse(
            answer=f"I could not determine an answer for \"{question}\" after {self._max_attempts} attempts.",
            strategy_used=state.strategy.value,
            attempts=self._max_attempts,
            success=False,
            trace_id=state.trace_id,
            specialist_log=state.history,
        )

    async def _decide_strategy(self, question: str) -> SupervisorDecision:
        prompt = _STRATEGY_PROMPT.format(question=question)
        try:
            response = await self._llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])

            strategy_map = {
                "discovery_first": StrategyType.DISCOVERY_FIRST,
                "direct_query": StrategyType.DIRECT_QUERY,
                "schema_exploration": StrategyType.SCHEMA_EXPLORATION,
                "aggregation": StrategyType.AGGREGATION,
            }
            strategy = strategy_map.get(data.get("strategy", ""), StrategyType.DISCOVERY_FIRST)
            return SupervisorDecision(
                strategy=strategy,
                reasoning=data.get("reasoning", ""),
                specialist_sequence=list(_STRATEGY_SEQUENCES.get(strategy, [])),
            )
        except Exception as exc:
            logger.warning("Strategy decision failed: %s — defaulting to discovery_first", exc)
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
                # For discovery failures, continue (non-critical)
                if specialist_name == "discovery":
                    continue
                return False

        return True

    async def _format_answer(self, question: str, rows: list[dict]) -> str:
        # Truncate results for LLM context
        results_text = json.dumps(rows[:20], indent=2, default=str)
        if len(results_text) > 3000:
            results_text = results_text[:3000] + "\n... (truncated)"

        prompt = _ANSWER_PROMPT.format(question=question, results=results_text)
        try:
            response = await self._llm.ainvoke(prompt)
            return response.content if hasattr(response, "content") else str(response)
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

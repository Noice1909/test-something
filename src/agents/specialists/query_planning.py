"""Query Planning Specialist — decides query execution strategy."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.utils import extract_text

from langchain_core.language_models import BaseChatModel

from src.agents.base import QueryComplexity, QueryPlan, SpecialistResult
from src.agents.state import AgentState
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_PLANNING_PROMPT = """\
You are a graph query planner. Based on the user question, discovered entities, \
and relevant schema, decide the best query strategy.

## User Question
{question}

## Discoveries
{discoveries}

## Relevant Schema
Node Labels: {labels}
Relationship Types: {rel_types}

## Strategy Options
- DIRECT: Simple match on a single label (e.g. "find all Movies")
- ONE_HOP: One relationship traversal (e.g. "movies directed by X")
- MULTI_HOP: Multiple hops/traversals (e.g. "actors who worked with directors of X")
- AGGREGATION: Count/sum/avg operations (e.g. "how many movies are there")

## Intent Options
- LIST: Return a list of entities
- COUNT: Return a count/number
- FIND: Find a specific entity
- EXPLORE: Explore relationships/connections

Return a JSON object with:
- "strategy": one of DIRECT, ONE_HOP, MULTI_HOP, AGGREGATION
- "intent": one of LIST, COUNT, FIND, EXPLORE
- "reasoning": brief explanation
- "filters": optional dict of filter conditions

Return ONLY the JSON object:"""


class QueryPlanningSpecialist:
    """Decides query complexity and intent based on question analysis."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            disc_text = "\n".join(
                f"- {d.entity_name} (label={d.label})"
                for d in state.discoveries
            ) if state.discoveries else "None"

            prompt = _PLANNING_PROMPT.format(
                question=state.question,
                discoveries=disc_text,
                labels=", ".join(state.schema_selection.node_labels) or "None",
                rel_types=", ".join(state.schema_selection.relationship_types) or "None",
            )

            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
            plan = self._parse_response(text)
            state.query_plan = plan

            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "query_planning", success=True, duration_ms=dur,
                detail=f"strategy={plan.strategy.value}, intent={plan.intent}",
            )
            return SpecialistResult(success=True, data=plan, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("query_planning", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    def _parse_response(self, text: str) -> QueryPlan:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return QueryPlan(strategy=QueryComplexity.DIRECT, intent="LIST", reasoning="Parse failed")

        strategy_map = {
            "DIRECT": QueryComplexity.DIRECT,
            "ONE_HOP": QueryComplexity.ONE_HOP,
            "MULTI_HOP": QueryComplexity.MULTI_HOP,
            "AGGREGATION": QueryComplexity.AGGREGATION,
        }

        return QueryPlan(
            strategy=strategy_map.get(
                data.get("strategy", "DIRECT").upper(), QueryComplexity.DIRECT
            ),
            intent=data.get("intent", "LIST").upper(),
            reasoning=data.get("reasoning", ""),
            filters=data.get("filters", {}),
        )

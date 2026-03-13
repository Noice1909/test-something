"""Schema + Planning Specialist — combined schema selection and query planning in one LLM call.

Replaces the separate SchemaReasoningSpecialist and QueryPlanningSpecialist
with a single specialist that performs both tasks in one LLM round-trip,
saving ~12-15s per query.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import (
    QueryComplexity,
    QueryPlan,
    SchemaSelection,
    SchemaPlan,
    SpecialistResult,
)
from src.agents.state import AgentState
from src.agents.utils import extract_text
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_SCHEMA_PLANNING_PROMPT = """\
You are a graph database expert. Given a user question, discovered entities, \
and the full database schema, perform TWO tasks in a single response:

**Task 1 — Schema Selection**: Select ONLY the node labels and relationship \
types relevant to answering the question.

**Task 2 — Query Planning**: Decide the best query strategy and intent.

## User Question
{question}

## Discovered Entities
{discoveries}

## Database Schema
Labels: {labels}
Relationship Types: {rel_types}
Relationship Patterns:
{patterns}
Label Properties:
{label_properties}

## Strategy Options (Task 2)
- DIRECT: Simple match on a single label (e.g. "find all Movies")
- ONE_HOP: One relationship traversal (e.g. "movies directed by X")
- TWO_HOP: Two traversals through intermediate node \
(e.g. "genres of movies by Tom Hanks" -> Person->Movie->Genre)
- MULTI_HOP: Three or more hops (use sparingly, only when needed)
- AGGREGATION: Count/sum/avg operations (e.g. "how many movies are there")

## Intent Options (Task 2)
LIST, COUNT, FIND, EXPLORE, COMPARE, RANK, EXISTS, PATH, GROUP, FILTER, \
DESCRIBE, SUGGEST, SUMMARIZE, TREND, RELATE, INDIRECT

## Rules
- Prefer simpler strategies (DIRECT, ONE_HOP) when possible
- If the question asks about a specific property or detail of a known entity, \
check whether the answer is in the node's own properties (see Label Properties). \
If so, select DIRECT strategy with no relationship types needed.
- Only include labels/relationships that are actually needed
- The intent should reflect what the user wants to know

Return a JSON object with:
- "node_labels": list of relevant label strings
- "relationship_types": list of relevant relationship type strings
- "schema_reasoning": one-sentence explanation for schema choices
- "strategy": one of DIRECT, ONE_HOP, TWO_HOP, MULTI_HOP, AGGREGATION
- "intent": one of the intents above
- "plan_reasoning": brief explanation for strategy choice
- "filters": optional dict of filter conditions

Return ONLY the JSON object:"""


class SchemaPlanningSpecialist:
    """Combined schema selection + query planning in a single LLM call."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            schema = await self._db.get_schema()

            disc_text = "\n".join(
                f"- {d.entity_name} (label={d.label}, confidence={d.confidence:.2f})"
                for d in state.discoveries
            ) if state.discoveries else "None"

            patterns_text = "\n".join(
                f"  ({p['from']})-[{p['type']}]->({p['to']})"
                + (" [BIDIRECTIONAL]" if p.get("bidirectional") else "")
                for p in schema.get("relationship_patterns", [])[:50]
            ) or "None"

            label_props_text = "\n".join(
                f"  {label}: {', '.join(props)}"
                for label, props in schema.get("label_properties", {}).items()
                if props
            ) or "None"

            prompt = _SCHEMA_PLANNING_PROMPT.format(
                question=state.question,
                discoveries=disc_text,
                labels=", ".join(schema.get("labels", [])),
                rel_types=", ".join(schema.get("relationship_types", [])),
                patterns=patterns_text,
                label_properties=label_props_text,
            )

            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
            result = self._parse_response(text, schema)

            # Populate state.schema_selection (used by query_generation)
            state.schema_selection = SchemaSelection(
                node_labels=result.node_labels,
                relationship_types=result.relationship_types,
                reasoning=result.schema_reasoning,
            )

            # Fallback: if no labels selected, provide ALL labels
            if not state.schema_selection.node_labels:
                all_labels = schema.get("labels", [])
                state.schema_selection.node_labels = all_labels
                state.schema_selection.reasoning += " (fallback: all labels provided)"
                logger.info("Schema planning selected 0 labels — using all %d labels", len(all_labels))

            if not state.schema_selection.relationship_types:
                state.schema_selection.relationship_types = schema.get("relationship_types", [])

            # Populate state.query_plan (used by query_generation)
            state.query_plan = QueryPlan(
                strategy=result.strategy,
                intent=result.intent,
                reasoning=result.plan_reasoning,
                filters=result.filters,
            )

            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "schema_planning", success=True, duration_ms=dur,
                detail=(
                    f"labels={len(result.node_labels)}, "
                    f"rels={len(result.relationship_types)}, "
                    f"strategy={result.strategy.value}, intent={result.intent}"
                ),
            )
            return SpecialistResult(success=True, data=result, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("schema_planning", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    def _parse_response(self, text: str, schema: dict[str, Any]) -> SchemaPlan:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return SchemaPlan(schema_reasoning="LLM parse failed")

        valid_labels = set(schema.get("labels", []))
        valid_rels = set(schema.get("relationship_types", []))

        strategy_map = {
            "DIRECT": QueryComplexity.DIRECT,
            "ONE_HOP": QueryComplexity.ONE_HOP,
            "TWO_HOP": QueryComplexity.TWO_HOP,
            "MULTI_HOP": QueryComplexity.MULTI_HOP,
            "AGGREGATION": QueryComplexity.AGGREGATION,
        }

        return SchemaPlan(
            node_labels=[lb for lb in data.get("node_labels", []) if lb in valid_labels],
            relationship_types=[rt for rt in data.get("relationship_types", []) if rt in valid_rels],
            schema_reasoning=data.get("schema_reasoning", data.get("reasoning", "")),
            strategy=strategy_map.get(
                data.get("strategy", "DIRECT").upper(), QueryComplexity.DIRECT,
            ),
            intent=data.get("intent", "LIST").upper(),
            plan_reasoning=data.get("plan_reasoning", ""),
            filters=data.get("filters", {}),
        )

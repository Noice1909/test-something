"""Schema Reasoning Specialist — selects relevant node/edge types using LLM."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.utils import extract_text

from langchain_core.language_models import BaseChatModel

from src.agents.base import SchemaSelection, SpecialistResult
from src.agents.state import AgentState
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_SCHEMA_REASONING_PROMPT = """\
You are a graph database schema expert. Given a user question, discovered \
entities, and the full database schema, select ONLY the node labels and \
relationship types that are relevant to answering the question.

## User Question
{question}

## Discovered Entities
{discoveries}

## Database Schema
Labels: {labels}
Relationship Types: {rel_types}
Relationship Patterns: {patterns}

## Instructions
Return a JSON object with:
- "node_labels": list of relevant label strings
- "relationship_types": list of relevant relationship type strings
- "reasoning": one-sentence explanation

Return ONLY the JSON object:"""


class SchemaReasoningSpecialist:
    """Uses LLM to select the relevant subset of the schema for a question."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            # 1) Get full schema
            schema = await self._db.get_schema()

            # 2) Format discoveries
            disc_text = "\n".join(
                f"- {d.entity_name} (label={d.label}, confidence={d.confidence:.2f})"
                for d in state.discoveries
            ) if state.discoveries else "None"

            # 3) Format patterns
            patterns_text = "\n".join(
                f"  ({p['from']})-[{p['type']}]->({p['to']})"
                for p in schema.get("relationship_patterns", [])[:30]
            ) or "None"

            # 4) Ask LLM
            prompt = _SCHEMA_REASONING_PROMPT.format(
                question=state.question,
                discoveries=disc_text,
                labels=", ".join(schema.get("labels", [])),
                rel_types=", ".join(schema.get("relationship_types", [])),
                patterns=patterns_text,
            )
            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)

            # 5) Parse JSON
            selection = self._parse_response(text, schema)
            state.schema_selection = selection

            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "schema_reasoning", success=True, duration_ms=dur,
                detail=f"Selected {len(selection.node_labels)} labels, "
                       f"{len(selection.relationship_types)} rel types",
            )
            return SpecialistResult(success=True, data=selection, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("schema_reasoning", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    def _parse_response(self, text: str, schema: dict[str, Any]) -> SchemaSelection:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Fallback: use all labels from discoveries
            labels = list({d.label for d in [] if d.label})
            return SchemaSelection(node_labels=labels, reasoning="LLM parse failed, using discovery labels")

        valid_labels = set(schema.get("labels", []))
        valid_rels = set(schema.get("relationship_types", []))

        return SchemaSelection(
            node_labels=[lb for lb in data.get("node_labels", []) if lb in valid_labels],
            relationship_types=[rt for rt in data.get("relationship_types", []) if rt in valid_rels],
            reasoning=data.get("reasoning", ""),
        )

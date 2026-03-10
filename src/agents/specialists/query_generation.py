"""Query Generation Specialist — creates database-specific queries."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import GeneratedQuery, SpecialistResult
from src.agents.state import AgentState
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = """\
You are a Neo4j Cypher query expert. Generate a READ-ONLY Cypher query to \
answer the user's question.

## User Question
{question}

## Query Plan
Strategy: {strategy}
Intent: {intent}
Reasoning: {plan_reasoning}

## Relevant Schema
Node Labels: {labels}
Relationship Types: {rel_types}
Label Properties: {label_props}
Relationship Patterns: {patterns}

## Discovered Entities
{discoveries}

## Rules
1. ONLY use labels and relationship types from the schema above
2. The query MUST be read-only (no CREATE, MERGE, DELETE, SET, REMOVE)
3. Always include a LIMIT clause (default 25)
4. Use parameterized queries with $param syntax where possible
5. Return meaningful properties, not just node references

Return a JSON object with:
- "query": the Cypher query string
- "parameters": dict of query parameters (or empty dict)

Return ONLY the JSON object:"""


class QueryGenerationSpecialist:
    """Generates Cypher queries from the plan and schema."""

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

            # Format label properties
            label_props = ""
            for label in state.schema_selection.node_labels:
                props = schema.get("label_properties", {}).get(label, [])
                if props:
                    label_props += f"\n  {label}: {', '.join(props)}"

            # Format patterns
            patterns = "\n".join(
                f"  ({p['from']})-[{p['type']}]->({p['to']})"
                for p in schema.get("relationship_patterns", [])
                if p["from"] in state.schema_selection.node_labels
                or p["to"] in state.schema_selection.node_labels
            )[:500] or "None"

            # Format discoveries
            disc_text = "\n".join(
                f"- {d.entity_name} (label={d.label}, id={d.node_id}, props={d.properties})"
                for d in state.discoveries[:10]
            ) if state.discoveries else "None"

            prompt = _GENERATION_PROMPT.format(
                question=state.question,
                strategy=state.query_plan.strategy.value,
                intent=state.query_plan.intent,
                plan_reasoning=state.query_plan.reasoning,
                labels=", ".join(state.schema_selection.node_labels) or "None",
                rel_types=", ".join(state.schema_selection.relationship_types) or "None",
                label_props=label_props or "None",
                patterns=patterns,
                discoveries=disc_text,
            )

            response = await self._llm.ainvoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            generated = self._parse_response(text)

            # Validate read-only
            if not generated.is_read_only:
                return SpecialistResult(
                    success=False,
                    error="Generated query contains write operations",
                    duration_ms=(time.time() - t0) * 1000,
                )

            state.generated_query = generated
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "query_generation", success=True, duration_ms=dur,
                detail=f"Cypher: {generated.query[:100]}…",
            )
            return SpecialistResult(success=True, data=generated, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("query_generation", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    def _parse_response(self, text: str) -> GeneratedQuery:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Try to extract just the cypher query
            return GeneratedQuery(query=text.strip(), is_read_only=self._check_read_only(text))

        query = data.get("query", "")
        params = data.get("parameters", {})
        return GeneratedQuery(
            query=query,
            language="cypher",
            parameters=params,
            is_read_only=self._check_read_only(query),
        )

    @staticmethod
    def _check_read_only(query: str) -> bool:
        upper = query.upper()
        write_keywords = ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP"]
        return not any(kw in upper for kw in write_keywords)

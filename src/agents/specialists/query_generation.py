"""Query Generation Specialist — creates database-specific queries.

Loads the Cypher syntax skill from .agent/skills/cypher_syntax/SKILL.md and
injects the reference into the LLM prompt so it can generate syntactically
correct queries for all common patterns (aggregation, traversal, filtering, etc.).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import GeneratedQuery, SpecialistResult
from src.agents.state import AgentState
from src.agents.utils import extract_text
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

# ── Load Cypher syntax skill ─────────────────────────────────────────────────

_SKILL_PATH = Path(__file__).resolve().parents[2] / ".." / ".agent" / "skills" / "cypher_syntax" / "SKILL.md"


def _load_cypher_skill() -> str:
    """Read the Cypher syntax SKILL.md and strip the YAML front-matter."""
    try:
        path = _SKILL_PATH.resolve()
        text = path.read_text(encoding="utf-8")
        # Strip YAML front-matter (between --- markers)
        if text.startswith("---"):
            end = text.index("---", 3)
            text = text[end + 3:].strip()
        logger.info("Cypher syntax skill loaded from %s", path)
        return text
    except Exception as exc:
        logger.warning("Could not load Cypher syntax skill: %s", exc)
        return ""


_CYPHER_REFERENCE = _load_cypher_skill()

# ── Prompt ────────────────────────────────────────────────────────────────────

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

{cypher_reference}

## CRITICAL Rules — You MUST follow these
1. ONLY use node labels listed in "Node Labels" above. Do NOT invent labels.
   - Example: if the schema has Person and Movie but NOT Director, then to find
     directors use: MATCH (p:Person)-[:DIRECTED]->(m:Movie)  — NOT (d:Director)
   - Roles like Director, Actor, Producer are expressed via RELATIONSHIPS, not labels.
2. ONLY use relationship types listed in "Relationship Types" above.
3. The query MUST be read-only (no CREATE, MERGE, DELETE, SET, REMOVE).
4. The query MUST end with a RETURN clause (never end with WITH).
5. Always include a LIMIT clause (default 25).
6. Return meaningful properties (e.g. n.name, n.title), NOT raw node references.
7. For "most" / "top" questions use ORDER BY aggregate DESC LIMIT N.
8. Always alias aggregations (e.g. count(m) AS movie_count).
9. Use the "Relationship Patterns" above to determine traversal direction.

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
                cypher_reference=_CYPHER_REFERENCE,
            )

            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
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

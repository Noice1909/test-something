"""Schema + Query Combined Specialist — schema selection, planning, AND Cypher
generation in a single LLM call.

Merges SchemaPlanningSpecialist and QueryGenerationSpecialist to eliminate one
full LLM round-trip (~12-18 s), cutting the happy-path pipeline from 4 LLM
calls to 3.

Enable via ``combine_schema_query = true`` in settings (default: true).
When disabled, the supervisor falls back to the original two-step sequence.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import (
    GeneratedQuery,
    QueryComplexity,
    QueryPlan,
    SchemaSelection,
    SchemaPlan,
    SpecialistResult,
    StrategyType,
)
from src.agents.state import AgentState
from src.agents.utils import extract_text
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

# ── Load Cypher syntax skill (same path as query_generation) ─────────────────

_SKILL_PATH = Path(__file__).resolve().parents[2] / ".." / "skills" / "cypher_syntax" / "SKILL.md"


def _load_cypher_skill() -> str:
    try:
        path = _SKILL_PATH.resolve()
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            end = text.index("---", 3)
            text = text[end + 3:].strip()
        return text
    except Exception as exc:
        logger.warning("Could not load Cypher syntax skill: %s", exc)
        return ""


_CYPHER_REFERENCE = _load_cypher_skill()

# ── Write-keyword guard ──────────────────────────────────────────────────────

_WRITE_KEYWORDS = ["CREATE", "MERGE", "DELETE", "SET ", "REMOVE", "DROP"]

# ── Combined prompt ──────────────────────────────────────────────────────────

_COMBINED_PROMPT = """\
You are a Neo4j Cypher query expert. Given a user question, discovered \
entities, and the full database schema, perform THREE tasks in a single response:

**Task 1 — Schema Selection**: Select ONLY the node labels and relationship \
types relevant to answering the question.

**Task 2 — Query Planning**: Choose the best query strategy and intent.

**Task 3 — Cypher Generation**: Generate a READ-ONLY Cypher query.

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

{cypher_reference}

## CRITICAL Rules — You MUST follow these
1. ONLY use node labels listed in "Labels" above. Do NOT invent labels.
   - Example: if the schema has Person and Movie but NOT Director, then to find \
directors use: MATCH (p:Person)-[:DIRECTED]->(m:Movie) — NOT (d:Director)
   - Roles like Director, Actor, Producer are expressed via RELATIONSHIPS, not labels.
2. ONLY use relationship types listed in "Relationship Types" above.
3. The query MUST be read-only (no CREATE, MERGE, DELETE, SET, REMOVE).
4. The query MUST end with a RETURN clause (never end with WITH).
5. Always include a LIMIT clause (default 25).
6. Return meaningful properties (e.g. n.name, n.title), NOT raw node references.
7. For "most" / "top" questions use ORDER BY aggregate DESC LIMIT N.
8. Always alias aggregations (e.g. count(m) AS movie_count).
9. Use the "Relationship Patterns" above to determine traversal direction.
   - Patterns marked [BIDIRECTIONAL] can be traversed either way.
   - For single-direction patterns, you MUST use the exact direction shown.
10. For PROPERTY_LOOKUP or ENTITY_DETAIL questions: prefer a simple MATCH with \
WHERE on the entity's known identifier, and RETURN the specific property. Check \
"Label Properties" above — the answer may be in the node's own properties \
without any relationship traversal.
11. When "Label Properties" lists the property the user is asking about, generate \
a simple query like: MATCH (n:Label) WHERE n.name = $name RETURN n.property \
LIMIT 1. Do NOT add unnecessary relationship traversals.
12. When a discovered entity has an elementId, you MAY use it for an exact lookup: \
MATCH (n) WHERE elementId(n) = $nodeId RETURN ... — this is the fastest possible query. \
Use this when the question asks about a SPECIFIC discovered entity.
13. Prefer simpler strategies (DIRECT, ONE_HOP) when possible.
14. Only include labels/relationships that are actually needed.

Return a JSON object with:
- "node_labels": list of relevant label strings (Task 1)
- "relationship_types": list of relevant relationship type strings (Task 1)
- "schema_reasoning": one-sentence explanation for schema choices (Task 1)
- "strategy": one of DIRECT, ONE_HOP, TWO_HOP, MULTI_HOP, AGGREGATION (Task 2)
- "intent": one of the intents above (Task 2)
- "query": the Cypher query string (Task 3)
- "parameters": dict of query parameters or empty dict (Task 3)
- "reasoning": brief explanation of WHY you chose this query structure (Task 3)

Return ONLY the JSON object:"""


class SchemaQuerySpecialist:
    """Combined schema selection + query planning + Cypher generation in one LLM call."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            schema = state.schema or await self._db.get_schema()

            # ── Format discoveries ────────────────────────────────────────
            sorted_disc = sorted(
                state.discoveries, key=lambda d: d.confidence, reverse=True
            )[:10] if state.discoveries else []
            disc_text = "\n".join(
                f"- {d.entity_name} (label={d.label}, confidence={d.confidence:.2f}, "
                f"elementId={d.node_id}, props={d.properties})"
                for d in sorted_disc
            ) if sorted_disc else "None"

            # ── Format schema context ─────────────────────────────────────
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

            prompt = _COMBINED_PROMPT.format(
                question=state.question,
                discoveries=disc_text,
                labels=", ".join(schema.get("labels", [])),
                rel_types=", ".join(schema.get("relationship_types", [])),
                patterns=patterns_text,
                label_properties=label_props_text,
                cypher_reference=_CYPHER_REFERENCE,
            )

            # ── Retry context (empty-result retries) ──────────────────────
            if state.previous_empty_queries:
                prev_lines = "\n".join(
                    f"  {i+1}. {pq.get('query', 'N/A')} -> returned 0 rows\n"
                    f"     Why: {pq.get('reasoning', 'N/A')}"
                    for i, pq in enumerate(state.previous_empty_queries)
                )
                next_hint = state.reflection.next_approach or ""
                prompt += (
                    f"\n\n## Previous queries returned 0 rows — "
                    f"use a DIFFERENT approach:\n{prev_lines}"
                )
                if next_hint:
                    prompt += (
                        f"\n\n## Suggested next approach:\n{next_hint}\n\n"
                        "Generate a DIFFERENT query based on the suggestion above. "
                        "Consider different properties, reversed directions, or "
                        "connecting through intermediate nodes."
                    )

            # ── Single LLM call ───────────────────────────────────────────
            response = await self._llm.ainvoke(prompt)
            text = extract_text(response)
            parsed = self._parse_response(text, schema)

            # Validate read-only
            if not parsed["is_read_only"]:
                err = "Generated query contains write operations"
                dur = (time.time() - t0) * 1000
                state.log_specialist("schema_query", success=False, duration_ms=dur, detail=err)
                return SpecialistResult(success=False, error=err, duration_ms=dur)

            # ── Populate state.schema_selection ────────────────────────────
            state.schema_selection = SchemaSelection(
                node_labels=parsed["node_labels"],
                relationship_types=parsed["relationship_types"],
                reasoning=parsed["schema_reasoning"],
            )

            # Fallback: if no labels selected, provide ALL labels
            if not state.schema_selection.node_labels:
                all_labels = schema.get("labels", [])
                state.schema_selection.node_labels = all_labels
                state.schema_selection.reasoning += " (fallback: all labels provided)"

            if not state.schema_selection.relationship_types:
                state.schema_selection.relationship_types = schema.get("relationship_types", [])

            # ── Populate state.query_plan ──────────────────────────────────
            state.query_plan = QueryPlan(
                strategy=parsed["strategy"],
                intent=parsed["intent"],
                reasoning=parsed["reasoning"],
            )

            # ── Populate state.generated_query ────────────────────────────
            state.generated_query = GeneratedQuery(
                query=parsed["query"],
                language="cypher",
                parameters=parsed["parameters"],
                is_read_only=parsed["is_read_only"],
                reasoning=parsed["reasoning"],
            )

            # Log cypher attempt
            state.log_cypher_attempt(
                query=parsed["query"],
                parameters=parsed["parameters"],
                reasoning=parsed["reasoning"],
            )

            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "schema_query", success=True, duration_ms=dur,
                detail=(
                    f"labels={len(parsed['node_labels'])}, "
                    f"strategy={parsed['strategy'].value}, "
                    f"cypher={parsed['query'][:80]}..."
                ),
            )
            return SpecialistResult(success=True, data=parsed, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("schema_query", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    def _parse_response(self, text: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Parse the combined JSON response into all three outputs."""
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Attempt to extract just a Cypher query from raw text
            return {
                "node_labels": schema.get("labels", []),
                "relationship_types": schema.get("relationship_types", []),
                "schema_reasoning": "LLM parse failed — using full schema",
                "strategy": QueryComplexity.DIRECT,
                "intent": "LIST",
                "query": text.strip(),
                "parameters": {},
                "reasoning": "Parse fallback",
                "is_read_only": self._check_read_only(text),
            }

        valid_labels = set(schema.get("labels", []))
        valid_rels = set(schema.get("relationship_types", []))

        strategy_map = {
            "DIRECT": QueryComplexity.DIRECT,
            "ONE_HOP": QueryComplexity.ONE_HOP,
            "TWO_HOP": QueryComplexity.TWO_HOP,
            "MULTI_HOP": QueryComplexity.MULTI_HOP,
            "AGGREGATION": QueryComplexity.AGGREGATION,
        }

        query = data.get("query", "")

        return {
            "node_labels": [lb for lb in data.get("node_labels", []) if lb in valid_labels],
            "relationship_types": [rt for rt in data.get("relationship_types", []) if rt in valid_rels],
            "schema_reasoning": data.get("schema_reasoning", data.get("reasoning", "")),
            "strategy": strategy_map.get(
                data.get("strategy", "DIRECT").upper(), QueryComplexity.DIRECT,
            ),
            "intent": data.get("intent", "LIST").upper(),
            "query": query,
            "parameters": data.get("parameters", {}),
            "reasoning": data.get("reasoning", ""),
            "is_read_only": self._check_read_only(query),
        }

    @staticmethod
    def _check_read_only(query: str) -> bool:
        upper = query.upper()
        return not any(kw in upper for kw in _WRITE_KEYWORDS)

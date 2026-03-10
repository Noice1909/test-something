"""Category 6 — Query & Execution Tools.

Used for advanced agent operations: running arbitrary read-only Cypher,
explaining/profiling queries, validation, and cost estimation.
"""

from __future__ import annotations

import re
from typing import Any

from src.database.abstract import AbstractDatabase

_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|DROP|CALL\s*\{|FOREACH)\b",
    re.IGNORECASE,
)


async def run_read_only_cypher(
    db: AbstractDatabase, *, query: str, parameters: dict[str, Any] | None = None, **_: Any,
) -> list[dict]:
    """Execute an arbitrary Cypher query, but only if it is read-only."""
    if _WRITE_PATTERN.search(query):
        return [{"error": "Query rejected: contains write operations"}]
    return await db.execute_read(query, parameters)


async def explain_cypher_query(
    db: AbstractDatabase, *, query: str, **_: Any
) -> list[dict]:
    return await db.execute_read(f"EXPLAIN {query}")


async def profile_cypher_query(
    db: AbstractDatabase, *, query: str, parameters: dict[str, Any] | None = None, **_: Any,
) -> list[dict]:
    if _WRITE_PATTERN.search(query):
        return [{"error": "Profile rejected: contains write operations"}]
    return await db.execute_read(f"PROFILE {query}", parameters)


async def validate_cypher_query(
    db: AbstractDatabase, *, query: str, **_: Any
) -> dict[str, Any]:
    """Check if a Cypher query is syntactically valid using EXPLAIN."""
    try:
        await db.execute_read(f"EXPLAIN {query}")
        return {"valid": True, "error": None}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


async def estimate_query_cost(
    db: AbstractDatabase, *, query: str, **_: Any
) -> dict[str, Any]:
    """Use EXPLAIN to estimate rows and db-hits."""
    try:
        rows = await db.execute_read(f"EXPLAIN {query}")
        return {"estimated": True, "plan": rows}
    except Exception as exc:
        return {"estimated": False, "error": str(exc)}


# ── registry ──

QUERY_EXECUTION_TOOLS: dict = {
    "run_read_only_cypher": run_read_only_cypher,
    "explain_cypher_query": explain_cypher_query,
    "profile_cypher_query": profile_cypher_query,
    "validate_cypher_query": validate_cypher_query,
    "estimate_query_cost": estimate_query_cost,
}

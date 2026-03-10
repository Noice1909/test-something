"""Category 10 — APOC Graph Query Helpers.

Dynamically construct and run queries. Only read-safe variants exposed.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def apoc_cypher_run(
    db: AbstractDatabase, *, query: str, params: dict | None = None,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Run a Cypher query via APOC (read-only)."""
    if not AbstractDatabase.is_read_only(query):  # type: ignore[arg-type]
        return [{"error": "Only read-only queries allowed"}]
    return await db.execute_read(query, params or {})


async def apoc_cypher_runFirstColumn(
    db: AbstractDatabase, *, query: str, params: dict | None = None, **_: Any,
) -> list[dict]:
    """Run query and return first column only."""
    if not AbstractDatabase.is_read_only(query):  # type: ignore[arg-type]
        return [{"error": "Only read-only queries allowed"}]
    rows = await db.execute_read(query, params or {})
    if rows and isinstance(rows[0], dict):
        first_key = next(iter(rows[0]))
        return [{"value": r[first_key]} for r in rows]
    return rows


async def apoc_cypher_runFirstColumnSingle(
    db: AbstractDatabase, *, query: str, params: dict | None = None, **_: Any,
) -> dict:
    """Run query and return single value from first column."""
    results = await apoc_cypher_runFirstColumn(db, query=query, params=params)
    return results[0] if results else {}


async def apoc_cypher_runTimeboxed(
    db: AbstractDatabase, *, query: str, timeout_ms: int = 5000,
    params: dict | None = None, **_: Any,
) -> list[dict]:
    """Run query with a timeout (fallback: normal execution)."""
    if not AbstractDatabase.is_read_only(query):  # type: ignore[arg-type]
        return [{"error": "Only read-only queries allowed"}]
    return await db.execute_read(query, params or {})


async def apoc_cypher_explain(
    db: AbstractDatabase, *, query: str, **_: Any,
) -> list[dict]:
    """Return EXPLAIN plan for a query."""
    return await db.execute_read(f"EXPLAIN {query}")


async def apoc_cypher_profile(
    db: AbstractDatabase, *, query: str, **_: Any,
) -> list[dict]:
    """Return PROFILE plan for a query."""
    return await db.execute_read(f"PROFILE {query}")


async def apoc_cypher_validate(
    db: AbstractDatabase, *, query: str, **_: Any,
) -> dict:
    """Validate Cypher syntax without executing."""
    try:
        await db.execute_read(f"EXPLAIN {query}")
        return {"valid": True, "error": None}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


async def apoc_cypher_runSchema(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Run schema-related queries."""
    return await db.execute_read("CALL db.schema.visualization()")


async def apoc_cypher_runManyReadOnly(
    db: AbstractDatabase, *, queries: list[str], **_: Any,
) -> list[dict]:
    """Run multiple read-only queries and aggregate results."""
    results: list[dict] = []
    for q in queries:
        if AbstractDatabase.is_read_only(q):  # type: ignore[arg-type]
            rows = await db.execute_read(q)
            results.append({"query": q, "rows": rows, "count": len(rows)})
        else:
            results.append({"query": q, "error": "Write query blocked"})
    return results


async def apoc_cypher_parallel2(
    db: AbstractDatabase, *, queries: list[str], **_: Any,
) -> list[dict]:
    """Run multiple queries (sequentially, read-only)."""
    return await apoc_cypher_runManyReadOnly(db, queries=queries)


APOC_QUERY_TOOLS: dict = {
    "apoc_cypher_run": apoc_cypher_run,
    "apoc_cypher_runFirstColumn": apoc_cypher_runFirstColumn,
    "apoc_cypher_runFirstColumnSingle": apoc_cypher_runFirstColumnSingle,
    "apoc_cypher_runTimeboxed": apoc_cypher_runTimeboxed,
    "apoc_cypher_explain": apoc_cypher_explain,
    "apoc_cypher_profile": apoc_cypher_profile,
    "apoc_cypher_validate": apoc_cypher_validate,
    "apoc_cypher_runSchema": apoc_cypher_runSchema,
    "apoc_cypher_runManyReadOnly": apoc_cypher_runManyReadOnly,
    "apoc_cypher_parallel2": apoc_cypher_parallel2,
}

"""Category 5 — Graph Data Inspection Tools.

Used for debugging, sampling, or exploring query results in detail.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def sample_nodes(
    db: AbstractDatabase, *, label: str | None = None, limit: int = 10, **_: Any
) -> list[dict]:
    if label:
        q = f"MATCH (n:`{label}`) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props LIMIT {int(limit)}"
    else:
        q = f"MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props LIMIT {int(limit)}"
    return await db.execute_read(q)


async def sample_relationships(
    db: AbstractDatabase, *, rel_type: str | None = None, limit: int = 10, **_: Any
) -> list[dict]:
    if rel_type:
        q = (
            f"MATCH (a)-[r:`{rel_type}`]->(b) "
            "RETURN elementId(r) AS id, type(r) AS type, "
            f"properties(r) AS props, labels(a)[0] AS from_label, labels(b)[0] AS to_label LIMIT {int(limit)}"
        )
    else:
        q = (
            "MATCH (a)-[r]->(b) "
            "RETURN elementId(r) AS id, type(r) AS type, "
            f"properties(r) AS props, labels(a)[0] AS from_label, labels(b)[0] AS to_label LIMIT {int(limit)}"
        )
    return await db.execute_read(q)


async def preview_subgraph(
    db: AbstractDatabase, *, node_limit: int = 20, rel_limit: int = 30, **_: Any
) -> dict:
    nodes = await db.execute_read(
        f"MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props LIMIT {int(node_limit)}"
    )
    rels = await db.execute_read(
        "MATCH (a)-[r]->(b) "
        f"RETURN elementId(r) AS id, type(r) AS type, elementId(a) AS from_id, elementId(b) AS to_id LIMIT {int(rel_limit)}"
    )
    return {"nodes": nodes, "relationships": rels}


async def get_node_properties(db: AbstractDatabase, *, node_id: int, **_: Any) -> dict:
    rows = await db.execute_read(
        "MATCH (n) WHERE elementId(n) = $id RETURN properties(n) AS props",
        {"id": str(node_id)},
    )
    return rows[0]["props"] if rows else {}


async def get_relationship_properties(db: AbstractDatabase, *, rel_id: int, **_: Any) -> dict:
    rows = await db.execute_read(
        "MATCH ()-[r]->() WHERE elementId(r) = $id RETURN properties(r) AS props",
        {"id": str(rel_id)},
    )
    return rows[0]["props"] if rows else {}


async def get_property_statistics(
    db: AbstractDatabase, *, label: str, key: str, **_: Any
) -> dict:
    rows = await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] IS NOT NULL "
        f"RETURN count(n['{key}']) AS total, "
        f"min(n['{key}']) AS min_val, max(n['{key}']) AS max_val, "
        f"avg(CASE WHEN n['{key}'] IS :: INTEGER OR n['{key}'] IS :: FLOAT THEN n['{key}'] ELSE null END) AS avg_val"
    )
    return rows[0] if rows else {}


async def get_distinct_property_values(
    db: AbstractDatabase, *, label: str, key: str, limit: int = 50, **_: Any
) -> list:
    rows = await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] IS NOT NULL "
        f"RETURN DISTINCT n['{key}'] AS val, count(*) AS cnt "
        f"ORDER BY cnt DESC LIMIT {int(limit)}"
    )
    return rows


async def get_missing_property_nodes(
    db: AbstractDatabase, *, label: str, key: str, limit: int = 25, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] IS NULL "
        f"RETURN elementId(n) AS id, properties(n) AS props LIMIT {int(limit)}"
    )


async def get_nodes_created_recently(
    db: AbstractDatabase, *, limit: int = 20, **_: Any
) -> list[dict]:
    """Return nodes with a `created_at` or `createdAt` timestamp, most recent first."""
    return await db.execute_read(
        "MATCH (n) WHERE n.created_at IS NOT NULL OR n.createdAt IS NOT NULL "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        "ORDER BY coalesce(n.created_at, n.createdAt) DESC "
        f"LIMIT {int(limit)}"
    )


async def get_relationships_created_recently(
    db: AbstractDatabase, *, limit: int = 20, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH ()-[r]->() WHERE r.created_at IS NOT NULL OR r.createdAt IS NOT NULL "
        "RETURN elementId(r) AS id, type(r) AS type, properties(r) AS props "
        "ORDER BY coalesce(r.created_at, r.createdAt) DESC "
        f"LIMIT {int(limit)}"
    )


# ── registry ──

DATA_INSPECTION_TOOLS: dict = {
    "sample_nodes": sample_nodes,
    "sample_relationships": sample_relationships,
    "preview_subgraph": preview_subgraph,
    "get_node_properties": get_node_properties,
    "get_relationship_properties": get_relationship_properties,
    "get_property_statistics": get_property_statistics,
    "get_distinct_property_values": get_distinct_property_values,
    "get_missing_property_nodes": get_missing_property_nodes,
    "get_nodes_created_recently": get_nodes_created_recently,
    "get_relationships_created_recently": get_relationships_created_recently,
}

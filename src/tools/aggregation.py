"""Category 4 — Aggregation & Analytics Tools.

Agents often need counts, distributions, and summaries to understand the data.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def count_nodes(db: AbstractDatabase, **_: Any) -> int:
    rows = await db.execute_read("MATCH (n) RETURN count(n) AS cnt")
    return rows[0]["cnt"] if rows else 0


async def count_nodes_by_label(db: AbstractDatabase, *, label: str, **_: Any) -> int:
    rows = await db.execute_read(
        f"MATCH (n:`{label}`) RETURN count(n) AS cnt"
    )
    return rows[0]["cnt"] if rows else 0


async def count_relationships(db: AbstractDatabase, **_: Any) -> int:
    rows = await db.execute_read("MATCH ()-[r]-() RETURN count(r)/2 AS cnt")
    return int(rows[0]["cnt"]) if rows else 0


async def count_relationships_by_type(db: AbstractDatabase, *, rel_type: str, **_: Any) -> int:
    rows = await db.execute_read(
        f"MATCH ()-[r:`{rel_type}`]-() RETURN count(r)/2 AS cnt"
    )
    return int(rows[0]["cnt"]) if rows else 0


async def count_nodes_with_property(
    db: AbstractDatabase, *, label: str, key: str, **_: Any
) -> int:
    rows = await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] IS NOT NULL RETURN count(n) AS cnt"
    )
    return rows[0]["cnt"] if rows else 0


async def get_most_connected_nodes(
    db: AbstractDatabase, *, limit: int = 10, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (n)-[r]-() "
        "RETURN elementId(n) AS id, labels(n) AS labels, count(r) AS degree "
        f"ORDER BY degree DESC LIMIT {int(limit)}"
    )


async def get_least_connected_nodes(
    db: AbstractDatabase, *, limit: int = 10, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (n)-[r]-() "
        "RETURN elementId(n) AS id, labels(n) AS labels, count(r) AS degree "
        f"ORDER BY degree ASC LIMIT {int(limit)}"
    )


async def get_node_degree_distribution(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (n) "
        "OPTIONAL MATCH (n)-[r]-() "
        "WITH n, count(r) AS degree "
        "RETURN degree, count(*) AS node_count ORDER BY degree"
    )


async def get_relationship_distribution(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH ()-[r]-() "
        "RETURN type(r) AS rel_type, count(*)/2 AS cnt ORDER BY cnt DESC"
    )


async def get_top_labels_by_node_count(
    db: AbstractDatabase, *, limit: int = 20, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "CALL db.labels() YIELD label "
        "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
        f"RETURN label, cnt ORDER BY cnt DESC LIMIT {int(limit)}"
    )


# ── registry ──

AGGREGATION_TOOLS: dict = {
    "count_nodes": count_nodes,
    "count_nodes_by_label": count_nodes_by_label,
    "count_relationships": count_relationships,
    "count_relationships_by_type": count_relationships_by_type,
    "count_nodes_with_property": count_nodes_with_property,
    "get_most_connected_nodes": get_most_connected_nodes,
    "get_least_connected_nodes": get_least_connected_nodes,
    "get_node_degree_distribution": get_node_degree_distribution,
    "get_relationship_distribution": get_relationship_distribution,
    "get_top_labels_by_node_count": get_top_labels_by_node_count,
}

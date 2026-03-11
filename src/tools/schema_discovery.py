"""Category 1 — Schema Discovery Tools.

These help the agent understand the database structure: labels, relationship
types, property keys, indexes, constraints, and sample subgraphs.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


# ── helpers ──

async def get_all_labels(db: AbstractDatabase, **_: Any) -> list[str]:
    return await db.get_labels()


async def get_all_relationship_types(db: AbstractDatabase, **_: Any) -> list[str]:
    return await db.get_relationship_types()


async def get_all_property_keys(db: AbstractDatabase, **_: Any) -> list[str]:
    return await db.get_property_keys()


async def get_database_info(db: AbstractDatabase, **_: Any) -> dict[str, Any]:
    rows = await db.execute_read(
        "CALL dbms.components() YIELD name, versions, edition "
        "RETURN name, versions, edition"
    )
    return rows[0] if rows else {}


async def get_schema_overview(db: AbstractDatabase, **_: Any) -> dict[str, Any]:
    return await db.get_schema()


async def get_node_labels_with_counts(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "CALL db.labels() YIELD label "
        "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
        "RETURN label, cnt ORDER BY cnt DESC"
    )


async def get_relationship_types_with_counts(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "CALL db.relationshipTypes() YIELD relationshipType AS type "
        "CALL { WITH type MATCH ()-[r]-() WHERE type(r) = type RETURN count(r)/2 AS cnt } "
        "RETURN type, toInteger(cnt) AS cnt ORDER BY cnt DESC"
    )


async def get_node_properties_for_label(db: AbstractDatabase, *, label: str, **_: Any) -> list[str]:
    rows = await db.execute_read(
        f"MATCH (n:`{label}`) UNWIND keys(n) AS k RETURN DISTINCT k ORDER BY k LIMIT 100"
    )
    return [r["k"] for r in rows]


async def get_relationship_properties_for_type(
    db: AbstractDatabase, *, rel_type: str, **_: Any
) -> list[str]:
    rows = await db.execute_read(
        f"MATCH ()-[r:`{rel_type}`]->() UNWIND keys(r) AS k RETURN DISTINCT k ORDER BY k LIMIT 100"
    )
    return [r["k"] for r in rows]


async def get_labels_and_relationship_matrix(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "RETURN labels(a)[0] AS from_label, type(r) AS rel, labels(b)[0] AS to_label, count(*) AS cnt "
        "ORDER BY cnt DESC LIMIT 200"
    )


async def get_possible_relationships_between_labels(
    db: AbstractDatabase, *, label_a: str, label_b: str, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}`)-[r]->(b:`{label_b}`) "
        "RETURN DISTINCT type(r) AS rel_type, count(*) AS cnt ORDER BY cnt DESC"
    )


async def get_constraints(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read("SHOW CONSTRAINTS")


async def get_indexes(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read("SHOW INDEXES")


async def get_fulltext_indexes(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW INDEXES YIELD * WHERE type = 'FULLTEXT' RETURN *"
    )


async def get_vector_indexes(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW INDEXES YIELD * WHERE type = 'VECTOR' RETURN *"
    )


async def get_range_indexes(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW INDEXES YIELD * WHERE type = 'RANGE' RETURN *"
    )


async def get_lookup_indexes(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW INDEXES YIELD * WHERE type = 'LOOKUP' RETURN *"
    )


async def get_index_details(db: AbstractDatabase, *, index_name: str, **_: Any) -> list[dict]:
    return await db.execute_read(
        f"SHOW INDEXES YIELD * WHERE name = $name RETURN *",
        {"name": index_name},
    )


async def get_schema_sample_graph(
    db: AbstractDatabase, *, limit: int = 25, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "RETURN labels(a) AS from_labels, type(r) AS rel, labels(b) AS to_labels, "
        "properties(a) AS a_props, properties(b) AS b_props "
        f"LIMIT {int(limit)}"
    )


async def get_property_types_for_label(
    db: AbstractDatabase, *, label: str, **_: Any
) -> list[dict]:
    rows = await db.execute_read(
        f"MATCH (n:`{label}`) WITH n LIMIT 100 "
        "UNWIND keys(n) AS k "
        "RETURN DISTINCT k AS property, "
        "apoc.meta.cypher.type(n[k]) AS type "
        "ORDER BY k"
    )
    return rows


# ── registry ──

SCHEMA_DISCOVERY_TOOLS: dict = {
    "get_all_labels": get_all_labels,
    "get_all_relationship_types": get_all_relationship_types,
    "get_all_property_keys": get_all_property_keys,
    "get_database_info": get_database_info,
    "get_schema_overview": get_schema_overview,
    "get_node_labels_with_counts": get_node_labels_with_counts,
    "get_relationship_types_with_counts": get_relationship_types_with_counts,
    "get_node_properties_for_label": get_node_properties_for_label,
    "get_relationship_properties_for_type": get_relationship_properties_for_type,
    "get_labels_and_relationship_matrix": get_labels_and_relationship_matrix,
    "get_possible_relationships_between_labels": get_possible_relationships_between_labels,
    "get_constraints": get_constraints,
    "get_indexes": get_indexes,
    "get_fulltext_indexes": get_fulltext_indexes,
    "get_vector_indexes": get_vector_indexes,
    "get_range_indexes": get_range_indexes,
    "get_lookup_indexes": get_lookup_indexes,
    "get_index_details": get_index_details,
    "get_schema_sample_graph": get_schema_sample_graph,
    "get_property_types_for_label": get_property_types_for_label,
}

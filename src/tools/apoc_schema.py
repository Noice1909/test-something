"""Category 8 — APOC Schema & Metadata Tools.

These help agents inspect schema using APOC procedures.
Note: Requires APOC plugin installed on the Neo4j server.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def apoc_meta_schema(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return full schema metadata via APOC."""
    return await db.execute_read("CALL apoc.meta.schema() YIELD value RETURN value")


async def apoc_meta_graph(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return a meta-graph of label-to-label relationships."""
    return await db.execute_read(
        "CALL apoc.meta.graph() YIELD nodes, relationships "
        "RETURN nodes, relationships"
    )


async def apoc_meta_graph_sample(
    db: AbstractDatabase, *, sample_size: int = 100, **_: Any
) -> list[dict]:
    """Return a sampled meta-graph."""
    return await db.execute_read(
        f"CALL apoc.meta.graph({{sample: {int(sample_size)}}}) "
        "YIELD nodes, relationships RETURN nodes, relationships"
    )


async def apoc_meta_nodeTypeProperties(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return property types per node label."""
    return await db.execute_read(
        "CALL apoc.meta.nodeTypeProperties() "
        "YIELD nodeType, propertyName, propertyTypes "
        "RETURN nodeType, propertyName, propertyTypes"
    )


async def apoc_meta_relTypeProperties(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return property types per relationship type."""
    return await db.execute_read(
        "CALL apoc.meta.relTypeProperties() "
        "YIELD relType, propertyName, propertyTypes "
        "RETURN relType, propertyName, propertyTypes"
    )


async def apoc_meta_stats(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return database statistics via APOC."""
    return await db.execute_read("CALL apoc.meta.stats() YIELD value RETURN value")


async def apoc_meta_data(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return metadata about all data in the database."""
    return await db.execute_read(
        "CALL apoc.meta.data() YIELD label, property, type, elementType "
        "RETURN label, property, type, elementType"
    )


async def apoc_meta_types(
    db: AbstractDatabase, *, label: str = "", **_: Any
) -> list[dict]:
    """Return the types of properties for nodes with given label."""
    if label:
        return await db.execute_read(
            f"MATCH (n:`{label}`) WITH n LIMIT 100 "
            "RETURN apoc.meta.types(n) AS types"
        )
    return await db.execute_read(
        "MATCH (n) WITH n LIMIT 100 RETURN labels(n) AS labels, apoc.meta.types(n) AS types"
    )


async def apoc_meta_type(
    db: AbstractDatabase, *, value: str, **_: Any
) -> list[dict]:
    """Return the Cypher type of a given value expression."""
    return await db.execute_read(f"RETURN apoc.meta.type({value}) AS type")


async def apoc_meta_isType(
    db: AbstractDatabase, *, value: str, type_name: str, **_: Any
) -> list[dict]:
    """Check if a value matches the given type."""
    return await db.execute_read(
        f"RETURN apoc.meta.isType({value}, $type) AS matches",
        {"type": type_name},
    )


async def apoc_meta_cypher_types(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return all known Cypher types."""
    return await db.execute_read(
        "MATCH (n) WITH n LIMIT 50 UNWIND keys(n) AS k "
        "RETURN DISTINCT k AS property, apoc.meta.cypher.type(n[k]) AS type"
    )


async def apoc_meta_relTypeProperties_sample(
    db: AbstractDatabase, *, sample: int = 100, **_: Any
) -> list[dict]:
    """Sampled relationship type properties."""
    return await db.execute_read(
        f"CALL apoc.meta.relTypeProperties({{sample: {int(sample)}}}) "
        "YIELD relType, propertyName, propertyTypes "
        "RETURN relType, propertyName, propertyTypes"
    )


async def apoc_meta_nodeTypeProperties_sample(
    db: AbstractDatabase, *, sample: int = 100, **_: Any
) -> list[dict]:
    """Sampled node type properties."""
    return await db.execute_read(
        f"CALL apoc.meta.nodeTypeProperties({{sample: {int(sample)}}}) "
        "YIELD nodeType, propertyName, propertyTypes "
        "RETURN nodeType, propertyName, propertyTypes"
    )


async def apoc_meta_schema_map(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return schema as a map structure."""
    return await db.execute_read("CALL apoc.meta.schema() YIELD value RETURN value")


async def apoc_meta_subGraph(
    db: AbstractDatabase, *, labels: str = "", rels: str = "", **_: Any
) -> list[dict]:
    """Return a meta-subgraph for specified labels/relationships."""
    config: dict[str, Any] = {}
    if labels:
        config["labels"] = labels.split(",")
    if rels:
        config["rels"] = rels.split(",")
    return await db.execute_read(
        "CALL apoc.meta.subGraph($config) YIELD nodes, relationships "
        "RETURN nodes, relationships",
        {"config": config},
    )


APOC_SCHEMA_TOOLS: dict = {
    "apoc_meta_schema": apoc_meta_schema,
    "apoc_meta_graph": apoc_meta_graph,
    "apoc_meta_graph_sample": apoc_meta_graph_sample,
    "apoc_meta_nodeTypeProperties": apoc_meta_nodeTypeProperties,
    "apoc_meta_relTypeProperties": apoc_meta_relTypeProperties,
    "apoc_meta_stats": apoc_meta_stats,
    "apoc_meta_data": apoc_meta_data,
    "apoc_meta_types": apoc_meta_types,
    "apoc_meta_type": apoc_meta_type,
    "apoc_meta_isType": apoc_meta_isType,
    "apoc_meta_cypher_types": apoc_meta_cypher_types,
    "apoc_meta_relTypeProperties_sample": apoc_meta_relTypeProperties_sample,
    "apoc_meta_nodeTypeProperties_sample": apoc_meta_nodeTypeProperties_sample,
    "apoc_meta_schema_map": apoc_meta_schema_map,
    "apoc_meta_subGraph": apoc_meta_subGraph,
}

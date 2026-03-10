"""Category 3 — Graph Search Tools.

Help agents retrieve relevant nodes via text search, property matching,
fulltext indexes, and vector similarity search.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def search_nodes_by_text(
    db: AbstractDatabase, *, text: str, limit: int = 20, **_: Any
) -> list[dict]:
    """Brute-force text search across all string properties."""
    return await db.execute_read(
        "MATCH (n) WHERE ANY(k IN keys(n) WHERE toString(n[k]) CONTAINS $text) "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        f"LIMIT {int(limit)}",
        {"text": text},
    )


async def search_relationships_by_text(
    db: AbstractDatabase, *, text: str, limit: int = 20, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH ()-[r]->() WHERE ANY(k IN keys(r) WHERE toString(r[k]) CONTAINS $text) "
        "RETURN elementId(r) AS id, type(r) AS type, properties(r) AS props "
        f"LIMIT {int(limit)}",
        {"text": text},
    )


async def search_nodes_by_property(
    db: AbstractDatabase, *, label: str, key: str, value: Any, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] = $val "
        "RETURN elementId(n) AS id, properties(n) AS props LIMIT 25",
        {"val": value},
    )


async def search_nodes_by_property_range(
    db: AbstractDatabase, *, label: str, key: str,
    min_val: Any = None, max_val: Any = None, limit: int = 25, **_: Any,
) -> list[dict]:
    conditions = []
    params: dict[str, Any] = {}
    if min_val is not None:
        conditions.append(f"n['{key}'] >= $min_val")
        params["min_val"] = min_val
    if max_val is not None:
        conditions.append(f"n['{key}'] <= $max_val")
        params["max_val"] = max_val
    where = " AND ".join(conditions) if conditions else "TRUE"
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE {where} "
        f"RETURN elementId(n) AS id, properties(n) AS props LIMIT {int(limit)}",
        params,
    )


async def search_nodes_by_multiple_properties(
    db: AbstractDatabase, *, label: str, filters: dict[str, Any], limit: int = 25, **_: Any,
) -> list[dict]:
    conditions = []
    params: dict[str, Any] = {}
    for i, (k, v) in enumerate(filters.items()):
        param_name = f"v{i}"
        conditions.append(f"n['{k}'] = ${param_name}")
        params[param_name] = v
    where = " AND ".join(conditions) if conditions else "TRUE"
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE {where} "
        f"RETURN elementId(n) AS id, properties(n) AS props LIMIT {int(limit)}",
        params,
    )


async def search_nodes_using_fulltext_index(
    db: AbstractDatabase, *, index_name: str, query: str, limit: int = 20, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"CALL db.index.fulltext.queryNodes($idx, $q) YIELD node, score "
        f"RETURN elementId(node) AS id, labels(node) AS labels, "
        f"properties(node) AS props, score LIMIT {int(limit)}",
        {"idx": index_name, "q": query},
    )


async def search_relationships_using_fulltext_index(
    db: AbstractDatabase, *, index_name: str, query: str, limit: int = 20, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"CALL db.index.fulltext.queryRelationships($idx, $q) YIELD relationship, score "
        f"RETURN elementId(relationship) AS id, type(relationship) AS type, "
        f"properties(relationship) AS props, score LIMIT {int(limit)}",
        {"idx": index_name, "q": query},
    )


async def search_nodes_by_embedding(
    db: AbstractDatabase, *, label: str, property_key: str,
    embedding: list[float], limit: int = 10, **_: Any,
) -> list[dict]:
    """Cosine similarity on a stored embedding property (no vector index)."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{property_key}'] IS NOT NULL "
        "WITH n, gds.similarity.cosine(n[$prop], $emb) AS score "
        f"RETURN elementId(n) AS id, properties(n) AS props, score "
        f"ORDER BY score DESC LIMIT {int(limit)}",
        {"prop": property_key, "emb": embedding},
    )


async def vector_similarity_search_nodes(
    db: AbstractDatabase, *, index_name: str,
    query_vector: list[float], limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node, score "
        "RETURN elementId(node) AS id, labels(node) AS labels, "
        "properties(node) AS props, score",
        {"idx": index_name, "k": int(limit), "vec": query_vector},
    )


async def vector_similarity_search_relationships(
    db: AbstractDatabase, *, index_name: str,
    query_vector: list[float], limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "CALL db.index.vector.queryRelationships($idx, $k, $vec) YIELD relationship, score "
        "RETURN elementId(relationship) AS id, type(relationship) AS type, "
        "properties(relationship) AS props, score",
        {"idx": index_name, "k": int(limit), "vec": query_vector},
    )


# ── registry ──

GRAPH_SEARCH_TOOLS: dict = {
    "search_nodes_by_text": search_nodes_by_text,
    "search_relationships_by_text": search_relationships_by_text,
    "search_nodes_by_property": search_nodes_by_property,
    "search_nodes_by_property_range": search_nodes_by_property_range,
    "search_nodes_by_multiple_properties": search_nodes_by_multiple_properties,
    "search_nodes_using_fulltext_index": search_nodes_using_fulltext_index,
    "search_relationships_using_fulltext_index": search_relationships_using_fulltext_index,
    "search_nodes_by_embedding": search_nodes_by_embedding,
    "vector_similarity_search_nodes": vector_similarity_search_nodes,
    "vector_similarity_search_relationships": vector_similarity_search_relationships,
}

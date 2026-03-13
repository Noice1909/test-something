"""Category 3 — Graph Search Tools.

Help agents retrieve relevant nodes via text search, property matching,
fulltext indexes, Levenshtein fuzzy search, and vector similarity search.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase

# Properties to exclude from broad text searches (embeddings, vectors, raw data)
_EXCLUDED_PROPS: list[str] = [
    "embedding", "embedding_vector", "vector",
    "raw_embedding", "raw_text", "features",
]


# ── Text search tools ────────────────────────────────────────────────────────


async def search_nodes_by_text(
    db: AbstractDatabase, *, text: str, limit: int = 20, **_: Any
) -> list[dict]:
    """Case-insensitive text search across all string properties,
    excluding embedding/vector properties for speed and accuracy."""
    return await db.execute_read(
        "MATCH (n) WHERE ANY(k IN keys(n) "
        "WHERE NOT k IN $excluded AND n[k] IS NOT NULL "
        "AND toLower(toString(n[k])) CONTAINS toLower($text)) "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        f"LIMIT {int(limit)}",
        {"text": text, "excluded": _EXCLUDED_PROPS},
    )


async def search_relationships_by_text(
    db: AbstractDatabase, *, text: str, limit: int = 20, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH ()-[r]->() WHERE ANY(k IN keys(r) "
        "WHERE NOT k IN $excluded AND r[k] IS NOT NULL "
        "AND toLower(toString(r[k])) CONTAINS $text) "
        "RETURN elementId(r) AS id, type(r) AS type, properties(r) AS props "
        f"LIMIT {int(limit)}",
        {"text": text, "excluded": _EXCLUDED_PROPS},
    )


# ── Levenshtein fuzzy search tools ───────────────────────────────────────────


async def fuzzy_search_all_properties(
    db: AbstractDatabase, *, text: str, threshold: int = 3,
    limit: int = 20, **_: Any,
) -> list[dict]:
    """Global fuzzy search — Levenshtein distance across all string properties,
    excluding embeddings. Returns nodes sorted by closest match (minDist).

    Ideal for short terms (e.g. "HK") and typo-tolerant search where CONTAINS
    is too loose. threshold=2 means up to 2 edits away from an exact match.
    """
    return await db.execute_read(
        "WITH $text AS query, $threshold AS threshold "
        "MATCH (n) "
        "WITH n, query, threshold, "
        "  [k IN keys(n) "
        "    WHERE NOT k IN $excluded AND n[k] IS NOT NULL "
        "    | toString(n[k])] AS vals "
        "UNWIND vals AS v "
        "WITH n, query, threshold, toLower(v) AS s "
        "WITH n, query, threshold, "
        "  collect(apoc.text.levenshteinDistance(s, toLower(query))) AS dists "
        "WITH n, apoc.coll.min(dists) AS minDist "
        "WHERE minDist IS NOT NULL AND minDist <= $threshold "
        "RETURN elementId(n) AS id, labels(n) AS labels, "
        "  properties(n) AS props, minDist "
        f"ORDER BY minDist LIMIT {int(limit)}",
        {"text": text, "threshold": threshold, "excluded": _EXCLUDED_PROPS},
    )


async def fuzzy_search_label_properties(
    db: AbstractDatabase, *, label: str, text: str, threshold: int = 3,
    limit: int = 20, **_: Any,
) -> list[dict]:
    """Label-scoped fuzzy search — Levenshtein across all non-embedding
    properties of a single label. Much faster than the global variant."""
    return await db.execute_read(
        "WITH $text AS query, $threshold AS threshold "
        f"MATCH (n:`{label}`) "
        "WITH n, query, threshold, "
        "  [k IN keys(n) "
        "    WHERE NOT k IN $excluded AND n[k] IS NOT NULL "
        "    | toString(n[k])] AS vals "
        "UNWIND vals AS v "
        "WITH n, query, threshold, toLower(v) AS s "
        "WITH n, query, threshold, "
        "  collect(apoc.text.levenshteinDistance(s, toLower(query))) AS dists "
        "WITH n, apoc.coll.min(dists) AS minDist "
        "WHERE minDist IS NOT NULL AND minDist <= $threshold "
        "RETURN elementId(n) AS id, labels(n) AS labels, "
        "  properties(n) AS props, minDist "
        f"ORDER BY minDist LIMIT {int(limit)}",
        {"text": text, "threshold": threshold, "excluded": _EXCLUDED_PROPS},
    )


# ── Property-targeted search tools ───────────────────────────────────────────


async def search_nodes_by_property(
    db: AbstractDatabase, *, label: str, key: str, value: Any, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] = $val "
        "RETURN elementId(n) AS id, properties(n) AS props LIMIT 25",
        {"val": value},
    )


async def search_nodes_by_property_case_insensitive(
    db: AbstractDatabase, *, label: str, key: str, value: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Case-insensitive exact match on a specific label+property.
    Much faster than global text search — only scans one label."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE toLower(n['{key}']) = toLower($val) "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        f"LIMIT {int(limit)}",
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


# ── Prefix / Regex search tools ──────────────────────────────────────────────


async def search_nodes_by_text_prefix(
    db: AbstractDatabase, *, label: str, key: str, prefix: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Prefix search — uses range index when available for fast lookups."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] STARTS WITH $prefix "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        f"LIMIT {int(limit)}",
        {"prefix": prefix},
    )


async def search_nodes_by_regex(
    db: AbstractDatabase, *, label: str, key: str, pattern: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Regex search on a specific property using Neo4j's =~ operator."""
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] =~ $pattern "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        f"LIMIT {int(limit)}",
        {"pattern": pattern},
    )


# ── Fulltext index search tools ──────────────────────────────────────────────


async def search_nodes_using_fulltext_index(
    db: AbstractDatabase, *, index_name: str, query: str, limit: int = 20, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"CALL db.index.fulltext.queryNodes($idx, $q) YIELD node, score "
        f"RETURN elementId(node) AS id, labels(node) AS labels, "
        f"properties(node) AS props, score LIMIT {int(limit)}",
        {"idx": index_name, "q": query},
    )


async def search_nodes_using_fulltext_enhanced(
    db: AbstractDatabase, *, index_name: str, query: str,
    fuzzy: bool = False, wildcard: bool = False,
    limit: int = 20, **_: Any,
) -> list[dict]:
    """Fulltext search with optional Lucene fuzzy (~) or wildcard (*) modifiers.

    fuzzy=True  → "term" becomes "term~"  (catches typos/near-misses)
    wildcard=True → "term" becomes "term*" (catches partial matches)
    """
    terms = query.strip().split()
    enhanced_parts: list[str] = []
    for term in terms:
        if fuzzy and not term.endswith("~"):
            enhanced_parts.append(f"{term}~")
        elif wildcard and not term.endswith("*"):
            enhanced_parts.append(f"{term}*")
        else:
            enhanced_parts.append(term)
    enhanced_query = " ".join(enhanced_parts)

    return await db.execute_read(
        "CALL db.index.fulltext.queryNodes($idx, $q) YIELD node, score "
        "RETURN elementId(node) AS id, labels(node) AS labels, "
        "properties(node) AS props, score "
        f"ORDER BY score DESC LIMIT {int(limit)}",
        {"idx": index_name, "q": enhanced_query},
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


# ── Vector similarity search tools ───────────────────────────────────────────


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
    # Text search (case-insensitive, excludes embeddings)
    "search_nodes_by_text": search_nodes_by_text,
    "search_relationships_by_text": search_relationships_by_text,
    # Levenshtein fuzzy search (distance-scored, excludes embeddings)
    "fuzzy_search_all_properties": fuzzy_search_all_properties,
    "fuzzy_search_label_properties": fuzzy_search_label_properties,
    # Property-targeted search
    "search_nodes_by_property": search_nodes_by_property,
    "search_nodes_by_property_case_insensitive": search_nodes_by_property_case_insensitive,
    "search_nodes_by_property_range": search_nodes_by_property_range,
    "search_nodes_by_multiple_properties": search_nodes_by_multiple_properties,
    # Prefix / Regex
    "search_nodes_by_text_prefix": search_nodes_by_text_prefix,
    "search_nodes_by_regex": search_nodes_by_regex,
    # Fulltext index (Lucene)
    "search_nodes_using_fulltext_index": search_nodes_using_fulltext_index,
    "search_nodes_using_fulltext_enhanced": search_nodes_using_fulltext_enhanced,
    "search_relationships_using_fulltext_index": search_relationships_using_fulltext_index,
    # Vector similarity
    "search_nodes_by_embedding": search_nodes_by_embedding,
    "vector_similarity_search_nodes": vector_similarity_search_nodes,
    "vector_similarity_search_relationships": vector_similarity_search_relationships,
}

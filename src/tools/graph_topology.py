"""Categories 20-22 — Graph Topology, Relationship Patterns, Property Analysis."""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


# ── Category 20: Graph Topology Analysis ─────────────────────────────────────

async def find_hubs_in_graph(
    db: AbstractDatabase, *, label: str = "", min_degree: int = 5,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Find hub nodes with high degree."""
    label_clause = f":`{label}`" if label else ""
    return await db.execute_read(
        f"MATCH (n{label_clause}) "
        f"WITH n, size((n)--()) AS degree WHERE degree >= {min_degree} "
        f"RETURN labels(n) AS labels, properties(n) AS props, degree "
        f"ORDER BY degree DESC LIMIT {int(limit)}"
    )


async def find_leaf_nodes(
    db: AbstractDatabase, *, label: str = "", limit: int = 50, **_: Any,
) -> list[dict]:
    """Find nodes with exactly one relationship."""
    label_clause = f":`{label}`" if label else ""
    return await db.execute_read(
        f"MATCH (n{label_clause}) "
        "WITH n, size((n)--()) AS degree WHERE degree = 1 "
        f"RETURN labels(n) AS labels, properties(n) AS props LIMIT {int(limit)}"
    )


async def find_bridge_nodes(
    db: AbstractDatabase, *, label: str = "", limit: int = 25, **_: Any,
) -> list[dict]:
    """Find nodes connecting otherwise disconnected parts."""
    label_clause = f":`{label}`" if label else ""
    return await db.execute_read(
        f"MATCH (n{label_clause})-[r1]-(a), (n)-[r2]-(b) "
        "WHERE NOT (a)--(b) AND id(a) < id(b) "
        "WITH n, count(DISTINCT a) + count(DISTINCT b) AS bridge_score "
        f"RETURN labels(n) AS labels, properties(n) AS props, bridge_score "
        f"ORDER BY bridge_score DESC LIMIT {int(limit)}"
    )


async def find_triangle_relationships(
    db: AbstractDatabase, *, label: str = "", limit: int = 25, **_: Any,
) -> list[dict]:
    """Find triangle patterns in the graph."""
    label_clause = f":`{label}`" if label else ""
    return await db.execute_read(
        f"MATCH (a{label_clause})--(b)--(c)--(a) "
        "WHERE id(a) < id(b) AND id(b) < id(c) "
        f"RETURN labels(a)[0] AS l1, labels(b)[0] AS l2, labels(c)[0] AS l3, "
        f"count(*) AS triangles LIMIT {int(limit)}"
    )


async def find_cycles(
    db: AbstractDatabase, *, max_length: int = 4, limit: int = 10, **_: Any,
) -> list[dict]:
    """Find cyclic paths."""
    return await db.execute_read(
        f"MATCH path = (n)-[*2..{max_length}]-(n) "
        "WHERE ALL(r IN relationships(path) WHERE startNode(r) <> endNode(r)) "
        f"RETURN [x IN nodes(path) | labels(x)[0]] AS cycle_labels, length(path) AS len "
        f"LIMIT {int(limit)}"
    )


async def find_short_cycles(db: AbstractDatabase, *, limit: int = 25, **_: Any) -> list[dict]:
    return await find_cycles(db, max_length=3, limit=limit)


async def detect_bipartite_patterns(
    db: AbstractDatabase, *, label_a: str, label_b: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}`)-[r]->(b:`{label_b}`) "
        "RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC "
        f"LIMIT {int(limit)}"
    )


async def detect_cliques(
    db: AbstractDatabase, *, label: str = "", limit: int = 10, **_: Any,
) -> list[dict]:
    label_clause = f":`{label}`" if label else ""
    return await db.execute_read(
        f"MATCH (a{label_clause})--(b)--(c)--(a) "
        "WHERE id(a) < id(b) AND id(b) < id(c) "
        "RETURN count(*) AS clique_count"
    )


async def detect_star_patterns(
    db: AbstractDatabase, *, min_degree: int = 5, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "MATCH (center)-[r]-(leaf) "
        "WITH center, count(DISTINCT leaf) AS degree WHERE degree >= $min "
        "RETURN labels(center) AS labels, properties(center) AS props, degree "
        f"ORDER BY degree DESC LIMIT {int(limit)}",
        {"min": min_degree},
    )


async def detect_common_neighbors(
    db: AbstractDatabase, *, label_a: str, val_a: str,
    label_b: str, val_b: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}` {{name: $va}})--(n)--(b:`{label_b}` {{name: $vb}}) "
        "RETURN labels(n) AS labels, properties(n) AS props, count(*) AS paths",
        {"va": val_a, "vb": val_b},
    )


# ── Category 21: Relationship Pattern Discovery ─────────────────────────────

async def discover_relationship_patterns(db: AbstractDatabase, *, limit: int = 50, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "RETURN labels(a)[0] AS from_label, type(r) AS rel, labels(b)[0] AS to_label, count(*) AS cnt "
        f"ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def discover_common_relationship_paths(
    db: AbstractDatabase, *, max_hops: int = 3, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (a)-[*1..{max_hops}]->(b) "
        "WITH [r IN relationships(path) | type(r)] AS pattern, count(*) AS freq "
        f"RETURN pattern, freq ORDER BY freq DESC LIMIT {int(limit)}"
    )


async def get_frequent_relationship_sequences(
    db: AbstractDatabase, *, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r1]->(b)-[r2]->(c) "
        "RETURN type(r1) AS rel1, type(r2) AS rel2, count(*) AS freq "
        f"ORDER BY freq DESC LIMIT {int(limit)}"
    )


async def get_relationship_cooccurrence(db: AbstractDatabase, *, limit: int = 25, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (n)-[r1]->(), (n)-[r2]->() WHERE type(r1) < type(r2) "
        "RETURN type(r1) AS rel1, type(r2) AS rel2, count(DISTINCT n) AS co_occur "
        f"ORDER BY co_occur DESC LIMIT {int(limit)}"
    )


async def get_top_relationship_pairs(db: AbstractDatabase, *, limit: int = 25, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "RETURN labels(a)[0] AS from_l, type(r) AS rel, labels(b)[0] AS to_l, count(*) AS cnt "
        f"ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def get_relationship_direction_distribution(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS outgoing "
        "ORDER BY outgoing DESC LIMIT 50"
    )


async def get_relationship_chain_patterns(
    db: AbstractDatabase, *, depth: int = 3, limit: int = 15, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = ()-[*{depth}]->() "
        "WITH [r IN relationships(path) | type(r)] AS chain, count(*) AS freq "
        f"RETURN chain, freq ORDER BY freq DESC LIMIT {int(limit)}"
    )


async def get_meta_relationship_graph(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt, count(*) AS cnt "
        "RETURN src, rel, tgt, cnt ORDER BY cnt DESC"
    )


# ── Category 22: Property Analysis ───────────────────────────────────────────

async def get_property_value_distribution(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN n.`{prop}` AS value, count(*) AS cnt "
        f"ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def get_numeric_property_statistics(
    db: AbstractDatabase, *, label: str, prop: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"RETURN min(n.`{prop}`) AS min_val, max(n.`{prop}`) AS max_val, "
        f"avg(n.`{prop}`) AS avg_val, count(n.`{prop}`) AS cnt, "
        f"stDev(n.`{prop}`) AS std_dev"
    )


async def get_string_property_statistics(
    db: AbstractDatabase, *, label: str, prop: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"RETURN min(size(n.`{prop}`)) AS min_len, max(size(n.`{prop}`)) AS max_len, "
        f"avg(size(n.`{prop}`)) AS avg_len, count(n.`{prop}`) AS cnt"
    )


async def get_property_null_ratio(
    db: AbstractDatabase, *, label: str, prop: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) "
        f"RETURN count(n) AS total, "
        f"count(CASE WHEN n.`{prop}` IS NULL THEN 1 END) AS nulls, "
        f"toFloat(count(CASE WHEN n.`{prop}` IS NULL THEN 1 END)) / count(n) AS null_ratio"
    )


async def get_property_cardinality(
    db: AbstractDatabase, *, label: str, prop: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"RETURN count(DISTINCT n.`{prop}`) AS cardinality, count(n.`{prop}`) AS total"
    )


async def get_property_entropy(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 100, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"WITH n.`{prop}` AS val, count(*) AS freq, "
        f"toFloat(count(*)) / $lim AS prob "
        "RETURN count(DISTINCT val) AS distinct_values, count(*) AS total",
        {"lim": limit},
    )


async def get_property_value_samples(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"RETURN DISTINCT n.`{prop}` AS sample LIMIT {int(limit)}"
    )


async def find_outlier_property_values(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NOT NULL "
        f"WITH n.`{prop}` AS val, count(*) AS freq "
        f"WHERE freq = 1 RETURN val AS outlier LIMIT {int(limit)}"
    )


async def find_duplicate_property_values(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WITH n.`{prop}` AS val, count(*) AS cnt "
        f"WHERE cnt > 1 RETURN val, cnt ORDER BY cnt DESC LIMIT {int(limit)}"
    )


async def find_nodes_missing_required_properties(
    db: AbstractDatabase, *, label: str, prop: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n.`{prop}` IS NULL "
        f"RETURN properties(n) AS props LIMIT {int(limit)}"
    )


# ── Registries ───────────────────────────────────────────────────────────────

TOPOLOGY_TOOLS: dict = {
    "find_hubs_in_graph": find_hubs_in_graph,
    "find_leaf_nodes": find_leaf_nodes,
    "find_bridge_nodes": find_bridge_nodes,
    "find_triangle_relationships": find_triangle_relationships,
    "find_cycles": find_cycles,
    "find_short_cycles": find_short_cycles,
    "detect_bipartite_patterns": detect_bipartite_patterns,
    "detect_cliques": detect_cliques,
    "detect_star_patterns": detect_star_patterns,
    "detect_common_neighbors": detect_common_neighbors,
}

RELATIONSHIP_PATTERN_TOOLS: dict = {
    "discover_relationship_patterns": discover_relationship_patterns,
    "discover_common_relationship_paths": discover_common_relationship_paths,
    "get_frequent_relationship_sequences": get_frequent_relationship_sequences,
    "get_relationship_cooccurrence": get_relationship_cooccurrence,
    "get_top_relationship_pairs": get_top_relationship_pairs,
    "get_relationship_direction_distribution": get_relationship_direction_distribution,
    "get_relationship_chain_patterns": get_relationship_chain_patterns,
    "get_meta_relationship_graph": get_meta_relationship_graph,
}

PROPERTY_ANALYSIS_TOOLS: dict = {
    "get_property_value_distribution": get_property_value_distribution,
    "get_numeric_property_statistics": get_numeric_property_statistics,
    "get_string_property_statistics": get_string_property_statistics,
    "get_property_null_ratio": get_property_null_ratio,
    "get_property_cardinality": get_property_cardinality,
    "get_property_entropy": get_property_entropy,
    "get_property_value_samples": get_property_value_samples,
    "find_outlier_property_values": find_outlier_property_values,
    "find_duplicate_property_values": find_duplicate_property_values,
    "find_nodes_missing_required_properties": find_nodes_missing_required_properties,
}

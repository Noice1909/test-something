"""Categories 23-25 — Graph Sampling, Navigation, Comparison Tools."""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


# ── Category 23: Graph Sampling ──────────────────────────────────────────────

async def sample_nodes_by_label(
    db: AbstractDatabase, *, label: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN properties(n) AS props LIMIT {int(limit)}"
    )


async def sample_relationships_by_type(
    db: AbstractDatabase, *, rel_type: str, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a)-[r:`{rel_type}`]->(b) "
        "RETURN labels(a)[0] AS from_label, properties(r) AS rel_props, "
        f"labels(b)[0] AS to_label LIMIT {int(limit)}"
    )


async def sample_paths(
    db: AbstractDatabase, *, max_hops: int = 3, limit: int = 5, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (a)-[*1..{max_hops}]->(b) "
        "RETURN [x IN nodes(path) | labels(x)[0]] AS node_labels, "
        f"[r IN relationships(path) | type(r)] AS rels LIMIT {int(limit)}"
    )


async def sample_k_hop_subgraph(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    k: int = 2, limit: int = 50, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}` {{{prop}: $val}})-[*1..{k}]-(m) "
        f"RETURN DISTINCT labels(m) AS labels, properties(m) AS props LIMIT {int(limit)}",
        {"val": value},
    )


async def sample_random_nodes(
    db: AbstractDatabase, *, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n) WITH n, rand() AS r ORDER BY r LIMIT {int(limit)} "
        "RETURN labels(n) AS labels, properties(n) AS props"
    )


async def sample_random_relationships(
    db: AbstractDatabase, *, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a)-[r]->(b) WITH a, r, b, rand() AS rnd ORDER BY rnd LIMIT {int(limit)} "
        "RETURN labels(a)[0] AS from_label, type(r) AS rel, labels(b)[0] AS to_label"
    )


async def sample_dense_regions(
    db: AbstractDatabase, *, min_degree: int = 5, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n) WITH n, size((n)--()) AS deg WHERE deg >= {min_degree} "
        f"RETURN labels(n) AS labels, properties(n) AS props, deg "
        f"ORDER BY deg DESC LIMIT {int(limit)}"
    )


async def sample_sparse_regions(
    db: AbstractDatabase, *, max_degree: int = 2, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n) WITH n, size((n)--()) AS deg WHERE deg <= {max_degree} "
        f"RETURN labels(n) AS labels, properties(n) AS props, deg "
        f"ORDER BY deg LIMIT {int(limit)}"
    )


# ── Category 24: Graph Navigation ────────────────────────────────────────────

async def walk_graph_randomly(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    steps: int = 5, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}` {{{prop}: $val}}) "
        f"CALL {{ WITH n MATCH path = (n)-[*1..{steps}]-(m) "
        "WITH path, rand() AS r ORDER BY r LIMIT 1 RETURN path }} "
        "RETURN [x IN nodes(path) | [labels(x)[0], properties(x)]] AS walk",
        {"val": value},
    )


async def walk_graph_bfs(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    max_depth: int = 3, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (start:`{label}` {{{prop}: $val}}) "
        f"CALL apoc.path.subgraphNodes(start, {{maxLevel: {max_depth}}}) "
        f"YIELD node RETURN labels(node) AS labels, properties(node) AS props LIMIT {int(limit)}",
        {"val": value},
    )


async def walk_graph_dfs(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    max_depth: int = 3, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (n:`{label}` {{{prop}: $val}})-[*1..{max_depth}]-(m) "
        "WITH path LIMIT 1 "
        "RETURN [x IN nodes(path) | [labels(x)[0], properties(x)]] AS dfs_walk",
        {"val": value},
    )


async def expand_paths_with_constraints(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    rel_types: str = "", max_hops: int = 3, limit: int = 25, **_: Any,
) -> list[dict]:
    rel_filter = f":`{'`|`'.join(rel_types.split(','))}`" if rel_types else ""
    return await db.execute_read(
        f"MATCH path = (n:`{label}` {{{prop}: $val}})-[{rel_filter}*1..{max_hops}]-(m) "
        "RETURN [x IN nodes(path) | labels(x)[0]] AS labels, "
        f"[r IN relationships(path) | type(r)] AS rels LIMIT {int(limit)}",
        {"val": value},
    )


async def expand_until_label(
    db: AbstractDatabase, *, start_label: str, start_value: str,
    target_label: str, max_hops: int = 5, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (s:`{start_label}` {{name: $sv}})-[*1..{max_hops}]-(t:`{target_label}`) "
        "RETURN [x IN nodes(path) | labels(x)[0]] AS path_labels, "
        f"properties(t) AS target_props LIMIT {int(limit)}",
        {"sv": start_value},
    )


async def expand_until_relationship_type(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    target_rel: str, max_hops: int = 5, limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (n:`{label}` {{{prop}: $val}})-[*1..{max_hops}]-(m) "
        f"WHERE ANY(r IN relationships(path) WHERE type(r) = $trel) "
        "RETURN [x IN nodes(path) | labels(x)[0]] AS path_labels, "
        f"[r IN relationships(path) | type(r)] AS rels LIMIT {int(limit)}",
        {"val": value, "trel": target_rel},
    )


async def get_next_hop_nodes(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}` {{{prop}: $val}})-[r]->(m) "
        f"RETURN type(r) AS rel, labels(m) AS labels, properties(m) AS props LIMIT {int(limit)}",
        {"val": value},
    )


async def get_previous_hop_nodes(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (m)-[r]->(n:`{label}` {{{prop}: $val}}) "
        f"RETURN type(r) AS rel, labels(m) AS labels, properties(m) AS props LIMIT {int(limit)}",
        {"val": value},
    )


# ── Category 25: Graph Comparison ────────────────────────────────────────────

async def find_similar_nodes_by_neighbors(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    limit: int = 10, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (target:`{label}` {{{prop}: $val}})--(neighbor) "
        f"WITH target, collect(DISTINCT id(neighbor)) AS t_neighbors "
        f"MATCH (other:`{label}`)--(neighbor) "
        "WHERE other <> target "
        "WITH other, t_neighbors, collect(DISTINCT id(neighbor)) AS o_neighbors "
        "WITH other, "
        "size(apoc.coll.intersection(t_neighbors, o_neighbors)) AS common, "
        "size(apoc.coll.union(t_neighbors, o_neighbors)) AS total "
        "WHERE total > 0 "
        "RETURN properties(other) AS props, toFloat(common)/total AS jaccard "
        f"ORDER BY jaccard DESC LIMIT {int(limit)}",
        {"val": value},
    )


async def find_similar_nodes_by_properties(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    compare_props: str = "name", limit: int = 10, **_: Any,
) -> list[dict]:
    c_props = [p.strip() for p in compare_props.split(",")]
    sim_clause = " + ".join(
        f"CASE WHEN n.`{p}` = target.`{p}` THEN 1 ELSE 0 END" for p in c_props
    )
    return await db.execute_read(
        f"MATCH (target:`{label}` {{{prop}: $val}}) "
        f"MATCH (n:`{label}`) WHERE n <> target "
        f"WITH n, target, ({sim_clause}) AS match_score "
        "WHERE match_score > 0 "
        f"RETURN properties(n) AS props, match_score "
        f"ORDER BY match_score DESC LIMIT {int(limit)}",
        {"val": value},
    )


async def compute_jaccard_similarity_between_nodes(
    db: AbstractDatabase, *, label_a: str, val_a: str,
    label_b: str, val_b: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}` {{name: $va}})--(n1) "
        f"WITH a, collect(DISTINCT id(n1)) AS set_a "
        f"MATCH (b:`{label_b}` {{name: $vb}})--(n2) "
        "WITH set_a, collect(DISTINCT id(n2)) AS set_b "
        "RETURN size(apoc.coll.intersection(set_a, set_b)) AS intersection, "
        "size(apoc.coll.union(set_a, set_b)) AS union_size, "
        "toFloat(size(apoc.coll.intersection(set_a, set_b))) / "
        "size(apoc.coll.union(set_a, set_b)) AS jaccard",
        {"va": val_a, "vb": val_b},
    )


async def compute_overlap_similarity(
    db: AbstractDatabase, *, label_a: str, val_a: str,
    label_b: str, val_b: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}` {{name: $va}})--(n1) "
        f"WITH a, collect(DISTINCT id(n1)) AS set_a "
        f"MATCH (b:`{label_b}` {{name: $vb}})--(n2) "
        "WITH set_a, collect(DISTINCT id(n2)) AS set_b "
        "WITH set_a, set_b, size(apoc.coll.intersection(set_a, set_b)) AS inter "
        "RETURN inter, CASE WHEN size(set_a) < size(set_b) THEN size(set_a) ELSE size(set_b) END AS min_size, "
        "toFloat(inter) / CASE WHEN size(set_a) < size(set_b) THEN size(set_a) ELSE size(set_b) END AS overlap",
        {"va": val_a, "vb": val_b},
    )


async def compute_cosine_similarity_between_nodes(
    db: AbstractDatabase, *, label_a: str, val_a: str,
    label_b: str, val_b: str, **_: Any,
) -> list[dict]:
    return await compute_jaccard_similarity_between_nodes(
        db, label_a=label_a, val_a=val_a, label_b=label_b, val_b=val_b,
    )


async def compare_node_neighborhoods(
    db: AbstractDatabase, *, label_a: str, val_a: str,
    label_b: str, val_b: str, limit: int = 25, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a:`{label_a}` {{name: $va}})-[r1]-(n) "
        "RETURN 'A' AS source, type(r1) AS rel, labels(n)[0] AS neighbor_label, "
        f"properties(n) AS props LIMIT {int(limit)} "
        "UNION ALL "
        f"MATCH (b:`{label_b}` {{name: $vb}})-[r2]-(m) "
        "RETURN 'B' AS source, type(r2) AS rel, labels(m)[0] AS neighbor_label, "
        f"properties(m) AS props LIMIT {int(limit)}",
        {"va": val_a, "vb": val_b},
    )


# ── Registries ───────────────────────────────────────────────────────────────

GRAPH_SAMPLING_TOOLS: dict = {
    "sample_nodes_by_label": sample_nodes_by_label,
    "sample_relationships_by_type": sample_relationships_by_type,
    "sample_paths": sample_paths,
    "sample_k_hop_subgraph": sample_k_hop_subgraph,
    "sample_random_nodes": sample_random_nodes,
    "sample_random_relationships": sample_random_relationships,
    "sample_dense_regions": sample_dense_regions,
    "sample_sparse_regions": sample_sparse_regions,
}

GRAPH_NAVIGATION_TOOLS: dict = {
    "walk_graph_randomly": walk_graph_randomly,
    "walk_graph_bfs": walk_graph_bfs,
    "walk_graph_dfs": walk_graph_dfs,
    "expand_paths_with_constraints": expand_paths_with_constraints,
    "expand_until_label": expand_until_label,
    "expand_until_relationship_type": expand_until_relationship_type,
    "get_next_hop_nodes": get_next_hop_nodes,
    "get_previous_hop_nodes": get_previous_hop_nodes,
}

GRAPH_COMPARISON_TOOLS: dict = {
    "find_similar_nodes_by_neighbors": find_similar_nodes_by_neighbors,
    "find_similar_nodes_by_properties": find_similar_nodes_by_properties,
    "compute_jaccard_similarity_between_nodes": compute_jaccard_similarity_between_nodes,
    "compute_overlap_similarity": compute_overlap_similarity,
    "compute_cosine_similarity_between_nodes": compute_cosine_similarity_between_nodes,
    "compare_node_neighborhoods": compare_node_neighborhoods,
}

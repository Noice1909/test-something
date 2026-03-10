"""Category 9 — APOC Graph Exploration Tools.

Allow agents to traverse graphs without writing complex Cypher.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def apoc_path_expand(
    db: AbstractDatabase, *, start_label: str, start_prop: str,
    start_value: str, rel_filter: str = "", label_filter: str = "",
    min_level: int = 1, max_level: int = 3, limit: int = 25, **_: Any,
) -> list[dict]:
    """Expand paths from a starting node."""
    return await db.execute_read(
        f"MATCH (n:`{start_label}` {{{start_prop}: $val}}) "
        f"CALL apoc.path.expand(n, $relFilter, $labelFilter, {min_level}, {max_level}) "
        f"YIELD path RETURN path LIMIT {int(limit)}",
        {"val": start_value, "relFilter": rel_filter, "labelFilter": label_filter},
    )


async def apoc_path_expandConfig(
    db: AbstractDatabase, *, start_label: str, start_prop: str,
    start_value: str, config: dict | None = None, limit: int = 25, **_: Any,
) -> list[dict]:
    """Expand paths with full config map."""
    cfg = config or {"maxLevel": 3}
    return await db.execute_read(
        f"MATCH (n:`{start_label}` {{{start_prop}: $val}}) "
        "CALL apoc.path.expandConfig(n, $config) "
        f"YIELD path RETURN path LIMIT {int(limit)}",
        {"val": start_value, "config": cfg},
    )


async def apoc_path_subgraphNodes(
    db: AbstractDatabase, *, start_label: str, start_prop: str,
    start_value: str, max_level: int = 3, limit: int = 50, **_: Any,
) -> list[dict]:
    """Get all nodes in a subgraph from a starting node."""
    return await db.execute_read(
        f"MATCH (n:`{start_label}` {{{start_prop}: $val}}) "
        f"CALL apoc.path.subgraphNodes(n, {{maxLevel: {max_level}}}) "
        f"YIELD node RETURN labels(node) AS labels, properties(node) AS props LIMIT {int(limit)}",
        {"val": start_value},
    )


async def apoc_path_subgraphAll(
    db: AbstractDatabase, *, start_label: str, start_prop: str,
    start_value: str, max_level: int = 3, limit: int = 50, **_: Any,
) -> list[dict]:
    """Get all nodes and relationships in a subgraph."""
    return await db.execute_read(
        f"MATCH (n:`{start_label}` {{{start_prop}: $val}}) "
        f"CALL apoc.path.subgraphAll(n, {{maxLevel: {max_level}}}) "
        f"YIELD nodes, relationships RETURN nodes, relationships LIMIT {int(limit)}",
        {"val": start_value},
    )


async def apoc_path_spanningTree(
    db: AbstractDatabase, *, start_label: str, start_prop: str,
    start_value: str, max_level: int = 3, limit: int = 50, **_: Any,
) -> list[dict]:
    """Get spanning tree from a starting node."""
    return await db.execute_read(
        f"MATCH (n:`{start_label}` {{{start_prop}: $val}}) "
        f"CALL apoc.path.spanningTree(n, {{maxLevel: {max_level}}}) "
        f"YIELD path RETURN path LIMIT {int(limit)}",
        {"val": start_value},
    )


async def apoc_path_neighbors(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    limit: int = 25, **_: Any,
) -> list[dict]:
    """Get immediate neighbors of a node."""
    return await db.execute_read(
        f"MATCH (n:`{label}` {{{prop}: $val}})-[r]-(m) "
        f"RETURN labels(m) AS labels, type(r) AS rel, properties(m) AS props LIMIT {int(limit)}",
        {"val": value},
    )


async def apoc_path_neighborsConfig(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    config: dict | None = None, limit: int = 25, **_: Any,
) -> list[dict]:
    """Get neighbors with config options."""
    cfg = config or {"maxLevel": 1}
    return await db.execute_read(
        f"MATCH (n:`{label}` {{{prop}: $val}}) "
        "CALL apoc.path.subgraphNodes(n, $config) "
        f"YIELD node RETURN labels(node) AS labels, properties(node) AS props LIMIT {int(limit)}",
        {"val": value, "config": cfg},
    )


async def apoc_path_expand_to(
    db: AbstractDatabase, *, start_label: str, start_value: str,
    end_label: str, max_level: int = 5, limit: int = 25, **_: Any,
) -> list[dict]:
    """Expand paths from start to a target label."""
    return await db.execute_read(
        f"MATCH (s:`{start_label}` {{name: $sval}}) "
        f"CALL apoc.path.expandConfig(s, {{labelFilter: '+{end_label}', maxLevel: {max_level}}}) "
        f"YIELD path RETURN path LIMIT {int(limit)}",
        {"sval": start_value},
    )


async def apoc_path_slice(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    offset: int = 0, length: int = 3, **_: Any,
) -> list[dict]:
    """Get a slice of a path."""
    return await db.execute_read(
        f"MATCH path = (n:`{label}` {{{prop}: $val}})-[*1..5]-(m) "
        f"WITH path LIMIT 1 "
        f"RETURN apoc.path.slice(path, {offset}, {length}) AS sliced",
        {"val": value},
    )


async def apoc_path_elements(
    db: AbstractDatabase, *, label: str, prop: str, value: str,
    max_hops: int = 3, limit: int = 10, **_: Any,
) -> list[dict]:
    """Get elements of paths from a node."""
    return await db.execute_read(
        f"MATCH path = (n:`{label}` {{{prop}: $val}})-[*1..{max_hops}]-(m) "
        f"WITH path LIMIT {int(limit)} "
        "RETURN [x IN nodes(path) | properties(x)] AS node_props, "
        "[r IN relationships(path) | type(r)] AS rel_types",
        {"val": value},
    )


APOC_EXPLORATION_TOOLS: dict = {
    "apoc_path_expand": apoc_path_expand,
    "apoc_path_expandConfig": apoc_path_expandConfig,
    "apoc_path_subgraphNodes": apoc_path_subgraphNodes,
    "apoc_path_subgraphAll": apoc_path_subgraphAll,
    "apoc_path_spanningTree": apoc_path_spanningTree,
    "apoc_path_neighbors": apoc_path_neighbors,
    "apoc_path_neighborsConfig": apoc_path_neighborsConfig,
    "apoc_path_expand_to": apoc_path_expand_to,
    "apoc_path_slice": apoc_path_slice,
    "apoc_path_elements": apoc_path_elements,
}

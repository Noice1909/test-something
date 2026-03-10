"""Category 2 — Graph Structure Exploration Tools.

Useful for agents to understand graph connectivity: neighbours, paths,
shortest paths, k-hop neighbours, connected components, etc.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def get_neighbors_of_node(
    db: AbstractDatabase, *, node_id: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (n)-[r]-(m) WHERE elementId(n) = $id "
        "RETURN elementId(m) AS neighbor_id, labels(m) AS labels, type(r) AS rel_type, "
        "properties(m) AS props LIMIT 50",
        {"id": str(node_id)},
    )


async def get_incoming_relationships(
    db: AbstractDatabase, *, node_id: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (m)-[r]->(n) WHERE elementId(n) = $id "
        "RETURN elementId(m) AS from_id, labels(m) AS from_labels, type(r) AS rel_type, "
        "properties(r) AS rel_props LIMIT 50",
        {"id": str(node_id)},
    )


async def get_outgoing_relationships(
    db: AbstractDatabase, *, node_id: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (n)-[r]->(m) WHERE elementId(n) = $id "
        "RETURN elementId(m) AS to_id, labels(m) AS to_labels, type(r) AS rel_type, "
        "properties(r) AS rel_props LIMIT 50",
        {"id": str(node_id)},
    )


async def get_all_relationships_of_node(
    db: AbstractDatabase, *, node_id: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (n)-[r]-(m) WHERE elementId(n) = $id "
        "RETURN elementId(m) AS other_id, labels(m) AS other_labels, "
        "type(r) AS rel_type, startNode(r) = n AS outgoing, "
        "properties(r) AS rel_props LIMIT 100",
        {"id": str(node_id)},
    )


async def get_node_by_id(db: AbstractDatabase, *, node_id: int, **_: Any) -> dict | None:
    rows = await db.execute_read(
        "MATCH (n) WHERE elementId(n) = $id "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props",
        {"id": str(node_id)},
    )
    return rows[0] if rows else None


async def get_node_by_property(
    db: AbstractDatabase, *, label: str, key: str, value: Any, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) WHERE n['{key}'] = $val "
        "RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props LIMIT 20",
        {"val": value},
    )


async def get_nodes_by_label(
    db: AbstractDatabase, *, label: str, limit: int = 25, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) "
        f"RETURN elementId(n) AS id, properties(n) AS props LIMIT {int(limit)}"
    )


async def get_relationships_by_type(
    db: AbstractDatabase, *, rel_type: str, limit: int = 25, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (a)-[r:`{rel_type}`]->(b) "
        "RETURN elementId(a) AS from_id, labels(a) AS from_labels, "
        "elementId(b) AS to_id, labels(b) AS to_labels, "
        f"properties(r) AS props LIMIT {int(limit)}"
    )


async def get_subgraph_by_node(
    db: AbstractDatabase, *, node_id: int, depth: int = 2, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (n)-[*1..{int(depth)}]-(m) WHERE elementId(n) = $id "
        "UNWIND nodes(path) AS node "
        "RETURN DISTINCT elementId(node) AS id, labels(node) AS labels, "
        "properties(node) AS props LIMIT 100",
        {"id": str(node_id)},
    )


async def get_k_hop_neighbors(
    db: AbstractDatabase, *, node_id: int, k: int = 2, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n)-[*1..{int(k)}]-(m) WHERE elementId(n) = $id "
        "RETURN DISTINCT elementId(m) AS id, labels(m) AS labels, "
        "properties(m) AS props LIMIT 100",
        {"id": str(node_id)},
    )


async def get_nodes_connected_between(
    db: AbstractDatabase, *, id_a: int, id_b: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[*1..3]-(n)-[*1..3]-(b) "
        "WHERE elementId(a) = $a AND elementId(b) = $b AND n <> a AND n <> b "
        "RETURN DISTINCT elementId(n) AS id, labels(n) AS labels, "
        "properties(n) AS props LIMIT 50",
        {"a": str(id_a), "b": str(id_b)},
    )


async def get_relationship_path_between_nodes(
    db: AbstractDatabase, *, id_a: int, id_b: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH path = shortestPath((a)-[*..10]-(b)) "
        "WHERE elementId(a) = $a AND elementId(b) = $b "
        "RETURN [n IN nodes(path) | {id: elementId(n), labels: labels(n)}] AS nodes, "
        "[r IN relationships(path) | {type: type(r), from: elementId(startNode(r)), "
        "to: elementId(endNode(r))}] AS rels",
        {"a": str(id_a), "b": str(id_b)},
    )


async def get_shortest_path_between_nodes(
    db: AbstractDatabase, *, id_a: int, id_b: int, **_: Any
) -> list[dict]:
    return await db.execute_read(
        "MATCH path = shortestPath((a)-[*..15]-(b)) "
        "WHERE elementId(a) = $a AND elementId(b) = $b "
        "RETURN length(path) AS hops, "
        "[n IN nodes(path) | {id: elementId(n), labels: labels(n), props: properties(n)}] AS nodes, "
        "[r IN relationships(path) | {type: type(r)}] AS rels",
        {"a": str(id_a), "b": str(id_b)},
    )


async def get_all_paths_between_nodes(
    db: AbstractDatabase, *, id_a: int, id_b: int, max_depth: int = 5, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH path = (a)-[*1..{int(max_depth)}]-(b) "
        "WHERE elementId(a) = $a AND elementId(b) = $b "
        "RETURN length(path) AS hops, "
        "[n IN nodes(path) | elementId(n)] AS node_ids LIMIT 20",
        {"a": str(id_a), "b": str(id_b)},
    )


async def get_nodes_within_depth(
    db: AbstractDatabase, *, node_id: int, depth: int = 3, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n)-[*1..{int(depth)}]-(m) WHERE elementId(n) = $id "
        "RETURN DISTINCT elementId(m) AS id, labels(m) AS labels LIMIT 200",
        {"id": str(node_id)},
    )


async def get_connected_component_of_node(
    db: AbstractDatabase, *, node_id: int, limit: int = 100, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n)-[*]-(m) WHERE elementId(n) = $id "
        f"RETURN DISTINCT elementId(m) AS id, labels(m) AS labels LIMIT {int(limit)}",
        {"id": str(node_id)},
    )


async def get_isolated_nodes(
    db: AbstractDatabase, *, limit: int = 50, **_: Any
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n) WHERE NOT (n)--() "
        f"RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props "
        f"LIMIT {int(limit)}"
    )


async def get_node_degree(db: AbstractDatabase, *, node_id: int, **_: Any) -> dict:
    rows = await db.execute_read(
        "MATCH (n) WHERE elementId(n) = $id "
        "OPTIONAL MATCH (n)-[r_out]->() "
        "OPTIONAL MATCH (n)<-[r_in]-() "
        "RETURN elementId(n) AS id, count(DISTINCT r_out) AS out_degree, "
        "count(DISTINCT r_in) AS in_degree",
        {"id": str(node_id)},
    )
    return rows[0] if rows else {}


async def get_nodes_by_multiple_labels(
    db: AbstractDatabase, *, labels: list[str], limit: int = 25, **_: Any
) -> list[dict]:
    label_clause = ":".join(f"`{lb}`" for lb in labels)
    return await db.execute_read(
        f"MATCH (n:{label_clause}) "
        f"RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props LIMIT {int(limit)}"
    )


async def get_relationship_direction_stats(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "RETURN type(r) AS rel_type, "
        "labels(a)[0] AS from_label, labels(b)[0] AS to_label, count(*) AS cnt "
        "ORDER BY cnt DESC LIMIT 100"
    )


# ── registry ──

GRAPH_EXPLORATION_TOOLS: dict = {
    "get_neighbors_of_node": get_neighbors_of_node,
    "get_incoming_relationships": get_incoming_relationships,
    "get_outgoing_relationships": get_outgoing_relationships,
    "get_all_relationships_of_node": get_all_relationships_of_node,
    "get_node_by_id": get_node_by_id,
    "get_node_by_property": get_node_by_property,
    "get_nodes_by_label": get_nodes_by_label,
    "get_relationships_by_type": get_relationships_by_type,
    "get_subgraph_by_node": get_subgraph_by_node,
    "get_k_hop_neighbors": get_k_hop_neighbors,
    "get_nodes_connected_between": get_nodes_connected_between,
    "get_relationship_path_between_nodes": get_relationship_path_between_nodes,
    "get_shortest_path_between_nodes": get_shortest_path_between_nodes,
    "get_all_paths_between_nodes": get_all_paths_between_nodes,
    "get_nodes_within_depth": get_nodes_within_depth,
    "get_connected_component_of_node": get_connected_component_of_node,
    "get_isolated_nodes": get_isolated_nodes,
    "get_node_degree": get_node_degree,
    "get_nodes_by_multiple_labels": get_nodes_by_multiple_labels,
    "get_relationship_direction_stats": get_relationship_direction_stats,
}

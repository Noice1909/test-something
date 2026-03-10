"""Category 26 — Graph Data Science (Read-Only) + Category 27 — DB Monitoring."""

from __future__ import annotations

import logging
from typing import Any

from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)


# ── Category 26: Graph Data Science (read-only stream) ───────────────────────
# These require the Neo4j Graph Data Science plugin.
# All use .stream mode (no writes, no graph mutation).


async def _gds_stream(db: AbstractDatabase, algo: str, config: dict) -> list[dict]:
    """Generic helper for GDS stream calls."""
    try:
        return await db.execute_read(
            f"CALL gds.{algo}.stream($config) YIELD *",
            {"config": config},
        )
    except Exception as exc:
        logger.debug("GDS %s unavailable: %s", algo, exc)
        return [{"error": f"GDS {algo} not available: {exc}"}]


async def gds_degree_centrality_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.degree.stream($g) YIELD nodeId, score "
            f"RETURN gds.util.asNode(nodeId).name AS name, score "
            f"ORDER BY score DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS degree centrality not available: {exc}"}]


async def gds_page_rank_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.pageRank.stream($g) YIELD nodeId, score "
            f"RETURN gds.util.asNode(nodeId).name AS name, score "
            f"ORDER BY score DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS pageRank not available: {exc}"}]


async def gds_betweenness_centrality_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.betweenness.stream($g) YIELD nodeId, score "
            f"RETURN gds.util.asNode(nodeId).name AS name, score "
            f"ORDER BY score DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS betweenness not available: {exc}"}]


async def gds_closeness_centrality_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.closeness.stream($g) YIELD nodeId, score "
            f"RETURN gds.util.asNode(nodeId).name AS name, score "
            f"ORDER BY score DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS closeness not available: {exc}"}]


async def gds_triangle_count_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.triangleCount.stream($g) YIELD nodeId, triangleCount "
            f"RETURN gds.util.asNode(nodeId).name AS name, triangleCount "
            f"ORDER BY triangleCount DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS triangleCount not available: {exc}"}]


async def gds_node_similarity_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.nodeSimilarity.stream($g) YIELD node1, node2, similarity "
            f"RETURN gds.util.asNode(node1).name AS name1, "
            f"gds.util.asNode(node2).name AS name2, similarity "
            f"ORDER BY similarity DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS nodeSimilarity not available: {exc}"}]


async def gds_louvain_community_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.louvain.stream($g) YIELD nodeId, communityId "
            f"RETURN gds.util.asNode(nodeId).name AS name, communityId "
            f"ORDER BY communityId LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS louvain not available: {exc}"}]


async def gds_label_propagation_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.labelPropagation.stream($g) YIELD nodeId, communityId "
            f"RETURN gds.util.asNode(nodeId).name AS name, communityId "
            f"ORDER BY communityId LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS labelPropagation not available: {exc}"}]


async def gds_shortest_path_dijkstra_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph",
    source_label: str = "", source_val: str = "",
    target_label: str = "", target_val: str = "", **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"MATCH (s:`{source_label}` {{name: $sv}}), (t:`{target_label}` {{name: $tv}}) "
            f"CALL gds.shortestPath.dijkstra.stream($g, {{sourceNode: s, targetNode: t}}) "
            "YIELD index, sourceNode, targetNode, totalCost, path "
            "RETURN totalCost, [n IN nodes(path) | n.name] AS path_names",
            {"g": graph_name, "sv": source_val, "tv": target_val},
        )
    except Exception as exc:
        return [{"error": f"GDS dijkstra not available: {exc}"}]


async def gds_k_nearest_neighbors_stream(
    db: AbstractDatabase, *, graph_name: str = "myGraph", limit: int = 25, **_: Any,
) -> list[dict]:
    try:
        return await db.execute_read(
            f"CALL gds.knn.stream($g) YIELD node1, node2, similarity "
            f"RETURN gds.util.asNode(node1).name AS name1, "
            f"gds.util.asNode(node2).name AS name2, similarity "
            f"ORDER BY similarity DESC LIMIT {int(limit)}",
            {"g": graph_name},
        )
    except Exception as exc:
        return [{"error": f"GDS knn not available: {exc}"}]


# ── Category 27: Database Monitoring ─────────────────────────────────────────

async def get_active_queries(db: AbstractDatabase, **_: Any) -> list[dict]:
    try:
        return await db.execute_read("SHOW TRANSACTIONS YIELD * RETURN *")
    except Exception:
        return [{"note": "SHOW TRANSACTIONS not available"}]


async def get_transaction_statistics(db: AbstractDatabase, **_: Any) -> list[dict]:
    try:
        return await db.execute_read(
            "SHOW TRANSACTIONS YIELD currentQueryId, status, elapsedTime "
            "RETURN currentQueryId, status, elapsedTime"
        )
    except Exception:
        return [{"note": "Transaction stats not available"}]


async def get_memory_usage(db: AbstractDatabase, **_: Any) -> dict:
    try:
        rows = await db.execute_read(
            "CALL dbms.queryJmx('java.lang:type=Memory') YIELD attributes RETURN attributes"
        )
        return rows[0] if rows else {"note": "Memory info not available"}
    except Exception:
        return {"note": "Memory usage not available on this edition"}


async def get_page_cache_statistics(db: AbstractDatabase, **_: Any) -> dict:
    try:
        rows = await db.execute_read(
            "CALL dbms.queryJmx('org.neo4j:name=Page cache,*') YIELD attributes RETURN attributes"
        )
        return rows[0] if rows else {"note": "Page cache stats not available"}
    except Exception:
        return {"note": "Page cache stats not available on this edition"}


async def get_connection_statistics(db: AbstractDatabase, **_: Any) -> dict:
    try:
        rows = await db.execute_read(
            "CALL dbms.queryJmx('org.neo4j:name=Bolt,*') YIELD attributes RETURN attributes"
        )
        return rows[0] if rows else {"note": "Connection stats not available"}
    except Exception:
        return {"note": "Connection stats not available on this edition"}


async def get_database_uptime(db: AbstractDatabase, **_: Any) -> dict:
    try:
        rows = await db.execute_read(
            "CALL dbms.queryJmx('java.lang:type=Runtime') "
            "YIELD attributes RETURN attributes.Uptime.value AS uptime_ms"
        )
        return rows[0] if rows else {"note": "Uptime not available"}
    except Exception:
        return {"note": "Uptime not available on this edition"}


async def get_thread_pool_statistics(db: AbstractDatabase, **_: Any) -> dict:
    try:
        rows = await db.execute_read(
            "CALL dbms.queryJmx('java.lang:type=Threading') "
            "YIELD attributes RETURN attributes"
        )
        return rows[0] if rows else {"note": "Thread stats not available"}
    except Exception:
        return {"note": "Thread stats not available on this edition"}


# ── Registries ───────────────────────────────────────────────────────────────

GDS_TOOLS: dict = {
    "gds_degree_centrality_stream": gds_degree_centrality_stream,
    "gds_page_rank_stream": gds_page_rank_stream,
    "gds_betweenness_centrality_stream": gds_betweenness_centrality_stream,
    "gds_closeness_centrality_stream": gds_closeness_centrality_stream,
    "gds_triangle_count_stream": gds_triangle_count_stream,
    "gds_node_similarity_stream": gds_node_similarity_stream,
    "gds_louvain_community_stream": gds_louvain_community_stream,
    "gds_label_propagation_stream": gds_label_propagation_stream,
    "gds_shortest_path_dijkstra_stream": gds_shortest_path_dijkstra_stream,
    "gds_k_nearest_neighbors_stream": gds_k_nearest_neighbors_stream,
}

MONITORING_TOOLS: dict = {
    "get_active_queries": get_active_queries,
    "get_transaction_statistics": get_transaction_statistics,
    "get_memory_usage": get_memory_usage,
    "get_page_cache_statistics": get_page_cache_statistics,
    "get_connection_statistics": get_connection_statistics,
    "get_database_uptime": get_database_uptime,
    "get_thread_pool_statistics": get_thread_pool_statistics,
}

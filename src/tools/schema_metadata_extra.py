"""Categories 18-19 — Schema/Metadata & Query Planning/Performance Tools."""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


# ── Category 18: Additional Schema & Metadata ────────────────────────────────

async def get_label_property_existence_constraints(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW CONSTRAINTS YIELD * WHERE type = 'NODE_PROPERTY_EXISTENCE' RETURN *"
    )


async def get_relationship_property_existence_constraints(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW CONSTRAINTS YIELD * WHERE type = 'RELATIONSHIP_PROPERTY_EXISTENCE' RETURN *"
    )


async def get_unique_constraints(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW CONSTRAINTS YIELD * WHERE type = 'UNIQUENESS' RETURN *"
    )


async def get_node_key_constraints(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW CONSTRAINTS YIELD * WHERE type = 'NODE_KEY' RETURN *"
    )


async def get_relationship_key_constraints(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW CONSTRAINTS YIELD * WHERE type = 'RELATIONSHIP_KEY' RETURN *"
    )


async def get_database_version(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
    )


async def get_database_status(db: AbstractDatabase, **_: Any) -> list[dict]:
    try:
        return await db.execute_read("CALL dbms.listConfig('dbms.default_database') YIELD value RETURN value")
    except Exception:
        return [{"status": "available"}]


async def get_default_database(db: AbstractDatabase, **_: Any) -> list[dict]:
    try:
        return await db.execute_read("CALL dbms.listConfig('dbms.default_database') YIELD value RETURN value")
    except Exception:
        return [{"database": "neo4j"}]


async def get_all_databases(db: AbstractDatabase, **_: Any) -> list[dict]:
    try:
        return await db.execute_read("SHOW DATABASES YIELD name, currentStatus RETURN name, currentStatus")
    except Exception:
        return [{"name": "neo4j", "currentStatus": "online"}]


async def get_database_size(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "CALL apoc.meta.stats() YIELD labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount "
        "RETURN labelCount, relTypeCount, propertyKeyCount, nodeCount, relCount"
    )


async def get_label_statistics(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "CALL db.labels() YIELD label "
        "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
        "RETURN label, cnt ORDER BY cnt DESC"
    )


async def get_relationship_type_statistics(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "CALL db.relationshipTypes() YIELD relationshipType AS type "
        "CALL { WITH type MATCH ()-[r]->() WHERE type(r) = type RETURN count(r) AS cnt } "
        "RETURN type, cnt ORDER BY cnt DESC"
    )


async def get_property_usage_statistics(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (n) UNWIND keys(n) AS k "
        "RETURN k AS property, count(*) AS usage ORDER BY usage DESC LIMIT 50"
    )


async def get_schema_relationship_patterns(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (a)-[r]->(b) "
        "RETURN DISTINCT labels(a)[0] AS from_label, type(r) AS rel, labels(b)[0] AS to_label, count(*) AS cnt "
        "ORDER BY cnt DESC LIMIT 100"
    )


async def get_node_label_combinations(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "MATCH (n) RETURN labels(n) AS label_combo, count(n) AS cnt ORDER BY cnt DESC LIMIT 50"
    )


# ── Category 19: Query Planning & Performance ────────────────────────────────

async def explain_cypher_query_plan(db: AbstractDatabase, *, query: str, **_: Any) -> list[dict]:
    """Return EXPLAIN plan for a query."""
    return await db.execute_read(f"EXPLAIN {query}")


async def profile_cypher_query_plan(db: AbstractDatabase, *, query: str, **_: Any) -> list[dict]:
    """Return PROFILE plan for a query (actually executes it)."""
    return await db.execute_read(f"PROFILE {query}")


async def estimate_result_size(
    db: AbstractDatabase, *, label: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(f"MATCH (n:`{label}`) RETURN count(n) AS estimated_size")


async def get_query_execution_time_estimate(
    db: AbstractDatabase, *, query: str, **_: Any,
) -> dict:
    """Estimate query execution time by running EXPLAIN."""
    try:
        await db.execute_read(f"EXPLAIN {query}")
        return {"estimate": "available", "note": "See EXPLAIN output for estimated rows"}
    except Exception as exc:
        return {"estimate": "error", "error": str(exc)}


async def get_query_memory_estimate(
    db: AbstractDatabase, *, query: str, **_: Any,
) -> dict:
    try:
        await db.execute_read(f"EXPLAIN {query}")
        return {"note": "Check EXPLAIN plan for memory estimates"}
    except Exception as exc:
        return {"error": str(exc)}


async def get_query_plan_tree(db: AbstractDatabase, *, query: str, **_: Any) -> list[dict]:
    return await db.execute_read(f"EXPLAIN {query}")


async def get_query_operator_statistics(db: AbstractDatabase, *, query: str, **_: Any) -> list[dict]:
    return await db.execute_read(f"PROFILE {query}")


async def get_index_usage_for_query(db: AbstractDatabase, *, query: str, **_: Any) -> list[dict]:
    return await db.execute_read(f"EXPLAIN {query}")


async def get_cardinality_estimates(
    db: AbstractDatabase, *, label: str, **_: Any,
) -> list[dict]:
    return await db.execute_read(
        f"MATCH (n:`{label}`) RETURN count(n) AS cardinality"
    )


async def get_query_cache_statistics(db: AbstractDatabase, **_: Any) -> dict:
    try:
        rows = await db.execute_read(
            "CALL dbms.queryJmx('org.neo4j:name=Query Cache,*') YIELD attributes RETURN attributes"
        )
        return rows[0] if rows else {"note": "Query cache stats not available"}
    except Exception:
        return {"note": "Query cache stats not available on this edition"}


# ── Registries ───────────────────────────────────────────────────────────────

SCHEMA_METADATA_EXTRA_TOOLS: dict = {
    "get_label_property_existence_constraints": get_label_property_existence_constraints,
    "get_relationship_property_existence_constraints": get_relationship_property_existence_constraints,
    "get_unique_constraints": get_unique_constraints,
    "get_node_key_constraints": get_node_key_constraints,
    "get_relationship_key_constraints": get_relationship_key_constraints,
    "get_database_version": get_database_version,
    "get_database_status": get_database_status,
    "get_default_database": get_default_database,
    "get_all_databases": get_all_databases,
    "get_database_size": get_database_size,
    "get_label_statistics": get_label_statistics,
    "get_relationship_type_statistics": get_relationship_type_statistics,
    "get_property_usage_statistics": get_property_usage_statistics,
    "get_schema_relationship_patterns": get_schema_relationship_patterns,
    "get_node_label_combinations": get_node_label_combinations,
}

QUERY_PERFORMANCE_TOOLS: dict = {
    "explain_cypher_query_plan": explain_cypher_query_plan,
    "profile_cypher_query_plan": profile_cypher_query_plan,
    "estimate_result_size": estimate_result_size,
    "get_query_execution_time_estimate": get_query_execution_time_estimate,
    "get_query_memory_estimate": get_query_memory_estimate,
    "get_query_plan_tree": get_query_plan_tree,
    "get_query_operator_statistics": get_query_operator_statistics,
    "get_index_usage_for_query": get_index_usage_for_query,
    "get_cardinality_estimates": get_cardinality_estimates,
    "get_query_cache_statistics": get_query_cache_statistics,
}

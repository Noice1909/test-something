"""Built-in Neo4j tools — each is a LangChain BaseTool subclass.

All tools are **read-only** and take the Neo4j async driver as a dependency
injected at registration time via the ``driver`` field.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from neo4j import AsyncDriver
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Input schemas (Pydantic v2)
# ─────────────────────────────────────────────────────────────────────────────


class RunCypherInput(BaseModel):
    query: str = Field(description="A read-only Cypher query to execute")
    params: dict[str, Any] = Field(default_factory=dict, description="Query parameters")


class GetSchemaInput(BaseModel):
    pass  # no inputs needed


class SearchNodesInput(BaseModel):
    search_term: str = Field(description="The text to search for across node properties")
    label: str | None = Field(default=None, description="Optional label to filter by")
    limit: int = Field(default=25, description="Maximum number of results")


class FuzzySearchInput(BaseModel):
    search_term: str = Field(description="The approximate text to match (typo-tolerant)")
    label: str | None = Field(default=None, description="Optional label to scope the search")
    threshold: int = Field(default=2, description="Max Levenshtein edit distance")
    limit: int = Field(default=25, description="Maximum number of results")


class GetNodeByIdInput(BaseModel):
    element_id: str = Field(description="The Neo4j elementId of the node")


class GetNeighborsInput(BaseModel):
    element_id: str = Field(description="The Neo4j elementId of the starting node")
    limit: int = Field(default=50, description="Maximum neighbors to return")


class CountNodesInput(BaseModel):
    label: str | None = Field(default=None, description="Optional label to count (all if omitted)")


class GetRelationshipPatternsInput(BaseModel):
    pass  # no inputs needed


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(records: list[dict]) -> str:
    """Serialise Neo4j records to a JSON string, handling Neo4j types."""

    def _convert(obj: Any) -> Any:
        if hasattr(obj, "__dict__") and hasattr(obj, "element_id"):
            # Node or Relationship
            d: dict[str, Any] = {"elementId": obj.element_id}
            if hasattr(obj, "labels"):
                d["labels"] = list(obj.labels)
            d["properties"] = dict(obj)
            return d
        if hasattr(obj, "nodes") and hasattr(obj, "relationships"):
            # Path
            return {
                "nodes": [_convert(n) for n in obj.nodes],
                "relationships": [_convert(r) for r in obj.relationships],
            }
        return obj

    converted = [{k: _convert(v) for k, v in rec.items()} for rec in records]
    return json.dumps(converted, indent=2, default=str)


import re as _re

# Every Neo4j Cypher clause / command that mutates data, schema, or admin state.
# Organised by category for auditability.
_WRITE_KEYWORDS = (
    # ── Data Manipulation (DML) ──────────────────────────────────────────
    "CREATE ",          # CREATE (n), CREATE (a)-[r]->(b)
    "MERGE ",           # MERGE (n), MERGE (a)-[r]->(b)
    "DELETE ",          # DELETE n
    "DETACH DELETE ",   # DETACH DELETE n  (delete node + all rels)
    "DETACH ",          # catch-all for DETACH
    "SET ",             # SET n.prop = val, SET n:Label
    "REMOVE ",          # REMOVE n.prop, REMOVE n:Label
    "FOREACH ",         # FOREACH (x IN list | CREATE ...)
    "FOREACH(",         # FOREACH(x IN list | ...)

    # ── Schema / Index / Constraint ──────────────────────────────────────
    "CREATE INDEX ",          # CREATE INDEX FOR (n:Label) ON (n.prop)
    "CREATE INDEX(",          # CREATE INDEX ...
    "DROP INDEX ",            # DROP INDEX name
    "CREATE CONSTRAINT ",     # CREATE CONSTRAINT FOR (n:Label) ...
    "DROP CONSTRAINT ",       # DROP CONSTRAINT name
    "CREATE FULLTEXT INDEX ", # CREATE FULLTEXT INDEX ...
    "CREATE LOOKUP INDEX ",   # CREATE LOOKUP INDEX ...
    "CREATE POINT INDEX ",    # CREATE POINT INDEX ...
    "CREATE RANGE INDEX ",    # CREATE RANGE INDEX ...
    "CREATE TEXT INDEX ",     # CREATE TEXT INDEX ...
    "CREATE OR REPLACE ",     # CREATE OR REPLACE alias/index

    # ── Admin / Database ─────────────────────────────────────────────────
    "CREATE DATABASE ",       # CREATE DATABASE name
    "DROP DATABASE ",         # DROP DATABASE name
    "ALTER DATABASE ",        # ALTER DATABASE name SET ...
    "START DATABASE ",        # START DATABASE name
    "STOP DATABASE ",         # STOP DATABASE name
    "CREATE COMPOSITE DATABASE ", # CREATE COMPOSITE DATABASE ...
    "CREATE ALIAS ",          # CREATE ALIAS name FOR DATABASE ...
    "DROP ALIAS ",            # DROP ALIAS name
    "ALTER ALIAS ",           # ALTER ALIAS name ...

    # ── Security / Users / Roles ─────────────────────────────────────────
    "CREATE USER ",           # CREATE USER name ...
    "DROP USER ",             # DROP USER name
    "ALTER USER ",            # ALTER USER name SET PASSWORD ...
    "CREATE ROLE ",           # CREATE ROLE name
    "DROP ROLE ",             # DROP ROLE name
    "RENAME ",                # RENAME ROLE/USER/DATABASE
    "GRANT ",                 # GRANT privilege TO role
    "DENY ",                  # DENY privilege TO role
    "REVOKE ",                # REVOKE privilege FROM role

    # ── Server / Cluster ─────────────────────────────────────────────────
    "ALTER SERVER ",          # ALTER SERVER name SET ...
    "ENABLE SERVER ",         # ENABLE SERVER name
    "DEALLOCATE ",            # DEALLOCATE DATABASES FROM SERVER
    "REALLOCATE ",            # REALLOCATE DATABASES
    "TERMINATE ",             # TERMINATE TRANSACTION(S)

    # ── Bulk Import ──────────────────────────────────────────────────────
    "LOAD CSV ",              # LOAD CSV WITH HEADERS FROM ...
    "LOAD CSV(",              # LOAD CSV(...)

    # ── Subquery writes / batching ───────────────────────────────────────
    "CALL {",                 # CALL { CREATE ... }  (subquery writes)
    " IN TRANSACTIONS",       # ... IN TRANSACTIONS  (batched writes)
)


def _normalise_query(query: str) -> str:
    """Upper-case + collapse all whitespace to single spaces.

    Prevents bypasses like ``CREATE\\n(n)`` where a newline replaces a space.
    """
    return _re.sub(r"\s+", " ", query.upper().strip())


def _assert_readonly(query: str) -> None:
    """Raise ValueError if the query contains any write operation."""
    normalised = _normalise_query(query)
    for kw in _WRITE_KEYWORDS:
        if kw in normalised:
            raise ValueError(
                f"BLOCKED: Write operation '{kw.strip()}' detected. "
                f"This system is strictly READ-ONLY — no data modifications allowed."
            )


async def _run_readonly(driver: AsyncDriver, query: str, params: dict | None = None, database: str = "neo4j") -> list[dict]:
    """Execute a read-only Cypher query and return records as dicts.

    Enforced at TWO levels:
      1. Keyword scanning — rejects queries containing write operations
      2. Neo4j driver execute_read() — the server itself rejects writes
    """
    _assert_readonly(query)

    async with driver.session(database=database) as session:

        async def _read_tx(tx):
            result = await tx.run(query, params or {})
            return [dict(record) async for record in result]

        return await session.execute_read(_read_tx)


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────


class RunCypherTool(BaseTool):
    name: str = "RunCypherTool"
    description: str = (
        "Execute a READ-ONLY Cypher query against the Neo4j graph and return "
        "results as JSON. Write operations (CREATE, MERGE, DELETE, SET) are blocked."
    )
    args_schema: Type[BaseModel] = RunCypherInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self, query: str, params: dict[str, Any] | None = None) -> str:
        records = await _run_readonly(self.driver, query, params or {}, self.database)
        if not records:
            return "Query returned 0 results."
        return _serialise(records)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class GetSchemaTool(BaseTool):
    name: str = "GetSchemaTool"
    description: str = (
        "Get the graph schema — all node labels with their properties, "
        "and all relationship types. Returns a structured overview."
    )
    args_schema: Type[BaseModel] = GetSchemaInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self) -> str:
        # Node labels + properties
        labels_q = """
        CALL db.labels() YIELD label
        CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes
        WITH label, nodeLabels, propertyName, propertyTypes
        WHERE label IN nodeLabels
        RETURN label, collect(DISTINCT {property: propertyName, types: propertyTypes}) AS properties
        """
        # Fallback for simpler Neo4j versions
        try:
            labels = await _run_readonly(self.driver, labels_q, database=self.database)
        except Exception:
            labels = await _run_readonly(
                self.driver,
                "CALL db.labels() YIELD label RETURN label",
                database=self.database,
            )

        # Relationship types
        rels = await _run_readonly(
            self.driver,
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
            database=self.database,
        )

        # Node counts per label
        counts = []
        for rec in labels:
            lbl = rec.get("label", rec.get("relationshipType", ""))
            if lbl:
                cnt = await _run_readonly(
                    self.driver,
                    f"MATCH (n:`{lbl}`) RETURN count(n) AS count",
                    database=self.database,
                )
                counts.append({"label": lbl, "count": cnt[0]["count"] if cnt else 0})

        result = {
            "labels": labels,
            "relationship_types": [r["relationshipType"] for r in rels],
            "node_counts": counts,
        }
        return json.dumps(result, indent=2, default=str)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class SearchNodesTool(BaseTool):
    name: str = "SearchNodesTool"
    description: str = (
        "Full-text search across node properties. Returns matching nodes "
        "with their labels, properties, and elementIds."
    )
    args_schema: Type[BaseModel] = SearchNodesInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self, search_term: str, label: str | None = None, limit: int = 25) -> str:
        # Try fulltext index first
        try:
            # Get available fulltext indexes
            indexes = await _run_readonly(
                self.driver,
                "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' RETURN name",
                database=self.database,
            )
            if indexes:
                idx_name = indexes[0]["name"]
                records = await _run_readonly(
                    self.driver,
                    f'CALL db.index.fulltext.queryNodes("{idx_name}", $term) '
                    f"YIELD node, score "
                    f"RETURN node, score ORDER BY score DESC LIMIT $limit",
                    {"term": search_term, "limit": limit},
                    self.database,
                )
                if records:
                    return _serialise(records)
        except Exception:
            pass

        # Fallback: case-insensitive CONTAINS across string properties
        if label:
            q = (
                f"MATCH (n:`{label}`) WHERE any(k IN keys(n) WHERE "
                f"n[k] IS :: STRING AND toLower(n[k]) CONTAINS toLower($term)) "
                f"RETURN n LIMIT $limit"
            )
        else:
            q = (
                "MATCH (n) WHERE any(k IN keys(n) WHERE "
                "n[k] IS :: STRING AND toLower(n[k]) CONTAINS toLower($term)) "
                "RETURN n LIMIT $limit"
            )
        records = await _run_readonly(self.driver, q, {"term": search_term, "limit": limit}, self.database)
        return _serialise(records) if records else "No results found."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class FuzzySearchTool(BaseTool):
    name: str = "FuzzySearchTool"
    description: str = (
        "Fuzzy search using Levenshtein distance — tolerant of typos and "
        "approximate spellings. Returns matching nodes ranked by edit distance."
    )
    args_schema: Type[BaseModel] = FuzzySearchInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(
        self,
        search_term: str,
        label: str | None = None,
        threshold: int = 2,
        limit: int = 25,
    ) -> str:
        # Try fulltext index with Lucene fuzzy syntax first
        try:
            indexes = await _run_readonly(
                self.driver,
                "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' RETURN name",
                database=self.database,
            )
            if indexes:
                idx_name = indexes[0]["name"]
                fuzzy_term = f"{search_term}~{threshold}"
                records = await _run_readonly(
                    self.driver,
                    f'CALL db.index.fulltext.queryNodes("{idx_name}", $term) '
                    f"YIELD node, score RETURN node, score ORDER BY score DESC LIMIT $limit",
                    {"term": fuzzy_term, "limit": limit},
                    self.database,
                )
                if records:
                    return _serialise(records)
        except Exception:
            pass

        # Fallback: apoc.text.levenshteinDistance if APOC available
        try:
            match_clause = f"MATCH (n:`{label}`)" if label else "MATCH (n)"
            q = (
                f"{match_clause} "
                f"UNWIND keys(n) AS k "
                f"WITH n, k WHERE n[k] IS :: STRING "
                f"WITH n, k, apoc.text.levenshteinDistance(toLower(n[k]), toLower($term)) AS dist "
                f"WHERE dist <= $threshold "
                f"RETURN DISTINCT n, min(dist) AS distance ORDER BY distance LIMIT $limit"
            )
            records = await _run_readonly(
                self.driver,
                q,
                {"term": search_term, "threshold": threshold, "limit": limit},
                self.database,
            )
            return _serialise(records) if records else "No fuzzy matches found."
        except Exception as exc:
            return f"Fuzzy search unavailable (requires APOC or fulltext index): {exc}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class GetNodeByIdTool(BaseTool):
    name: str = "GetNodeByIdTool"
    description: str = (
        "Fetch a single node by its Neo4j elementId. Returns all labels "
        "and properties."
    )
    args_schema: Type[BaseModel] = GetNodeByIdInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self, element_id: str) -> str:
        records = await _run_readonly(
            self.driver,
            "MATCH (n) WHERE elementId(n) = $eid RETURN n",
            {"eid": element_id},
            self.database,
        )
        return _serialise(records) if records else f"No node found with elementId: {element_id}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class GetNeighborsTool(BaseTool):
    name: str = "GetNeighborsTool"
    description: str = (
        "Get all nodes connected to a given node — neighbors, relationship "
        "types, and directions (incoming/outgoing)."
    )
    args_schema: Type[BaseModel] = GetNeighborsInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self, element_id: str, limit: int = 50) -> str:
        records = await _run_readonly(
            self.driver,
            "MATCH (n)-[r]-(neighbor) WHERE elementId(n) = $eid "
            "RETURN type(r) AS rel_type, "
            "CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction, "
            "neighbor LIMIT $limit",
            {"eid": element_id, "limit": limit},
            self.database,
        )
        return _serialise(records) if records else "No neighbors found."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class CountNodesTool(BaseTool):
    name: str = "CountNodesTool"
    description: str = (
        "Count nodes, optionally filtered by label. Returns the count."
    )
    args_schema: Type[BaseModel] = CountNodesInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self, label: str | None = None) -> str:
        if label:
            records = await _run_readonly(
                self.driver,
                f"MATCH (n:`{label}`) RETURN count(n) AS count",
                database=self.database,
            )
        else:
            records = await _run_readonly(
                self.driver,
                "MATCH (n) RETURN count(n) AS count",
                database=self.database,
            )
        count = records[0]["count"] if records else 0
        return json.dumps({"count": count, "label": label or "all"})

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class GetRelationshipPatternsTool(BaseTool):
    name: str = "GetRelationshipPatternsTool"
    description: str = (
        "List all (SourceLabel)-[RELATIONSHIP_TYPE]->(TargetLabel) patterns "
        "with counts and direction information. Essential for writing correct "
        "Cypher queries — wrong direction = 0 results."
    )
    args_schema: Type[BaseModel] = GetRelationshipPatternsInput
    read_only: bool = True
    driver: Any = None
    database: str = "neo4j"

    async def _arun(self) -> str:
        records = await _run_readonly(
            self.driver,
            "MATCH (a)-[r]->(b) "
            "WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt, count(*) AS cnt "
            "RETURN src, rel, tgt, cnt ORDER BY cnt DESC",
            database=self.database,
        )
        if not records:
            return "No relationship patterns found."

        lines = []
        for rec in records:
            lines.append(
                f"(:{rec['src']})-[:{rec['rel']}]->(:{rec['tgt']})  — {rec['cnt']} relationships"
            )
        return "\n".join(lines)

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_neo4j_tools(driver: AsyncDriver, database: str = "neo4j") -> list[BaseTool]:
    """Create all built-in Neo4j tools with the given driver."""
    kwargs: dict[str, Any] = {"driver": driver, "database": database}
    return [
        RunCypherTool(**kwargs),
        GetSchemaTool(**kwargs),
        SearchNodesTool(**kwargs),
        FuzzySearchTool(**kwargs),
        GetNodeByIdTool(**kwargs),
        GetNeighborsTool(**kwargs),
        CountNodesTool(**kwargs),
        GetRelationshipPatternsTool(**kwargs),
    ]

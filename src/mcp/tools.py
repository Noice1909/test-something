from __future__ import annotations

import json
import re
from typing import Any

import structlog

from src.services.concept_service import ConceptService
from src.services.cypher_validator import CypherValidator
from src.services.index_service import IndexService
from src.services.neo4j_service import Neo4jService
from src.services.schema_service import SchemaService

logger = structlog.get_logger()

WRITE_KEYWORDS_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP)\b", re.IGNORECASE
)


class MCPTools:
    """MCP tool implementations for the Neo4j NL Agent."""

    def __init__(
        self,
        neo4j_svc: Neo4jService,
        schema_svc: SchemaService,
        concept_svc: ConceptService,
        index_svc: IndexService,
        cypher_validator: CypherValidator,
    ) -> None:
        self.neo4j_svc = neo4j_svc
        self.schema_svc = schema_svc
        self.concept_svc = concept_svc
        self.index_svc = index_svc
        self.cypher_validator = cypher_validator

    def get_schema(self, filter_labels: list[str] | None = None) -> dict[str, Any]:
        """Tool 1: Return full or filtered graph schema."""
        if filter_labels:
            text = self.schema_svc.get_filtered_schema(filter_labels)
        else:
            text = self.schema_svc.get_full_schema_text()

        return {
            "schema_text": text,
            "labels": self.schema_svc.labels,
            "relationship_types": self.schema_svc.relationship_types,
        }

    def search_concepts(self, query: str) -> list[dict[str, Any]]:
        """Tool 2: Search :Concept nodes by nlp_terms."""
        if not self.concept_svc.available:
            return [{"error": "Concept nodes not available in this database"}]

        concepts = self.concept_svc.match_concepts(query)
        return [
            {
                "name": c.name,
                "nlp_terms": c.nlp_terms,
                "description": c.description,
            }
            for c in concepts
        ]

    def fuzzy_search_global(self, term: str, limit: int = 5) -> list[dict[str, Any]]:
        """Tool 3: Full-text search using global name index."""
        index_name = self.index_svc.get_global_index("name")
        if not index_name:
            return [{"error": "No global name index available"}]

        return self.neo4j_svc.execute_read(
            "CALL db.index.fulltext.queryNodes($index, $term) "
            "YIELD node, score WHERE score > 0.3 "
            "RETURN node.name AS name, node.id AS id, "
            "labels(node)[0] AS label, score "
            "ORDER BY score DESC LIMIT $limit",
            {"index": index_name, "term": f"{term}~", "limit": limit},
        )

    def fuzzy_search_by_label(
        self, label: str, term: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Tool 4: Full-text search using per-label index."""
        index_name = self.index_svc.get_label_index(label)
        if not index_name:
            return [{"error": f"No fulltext index for label '{label}'"}]

        return self.neo4j_svc.execute_read(
            "CALL db.index.fulltext.queryNodes($index, $term) "
            "YIELD node, score WHERE score > 0.3 "
            "RETURN node.name AS name, node.id AS id, "
            "labels(node)[0] AS label, score "
            "ORDER BY score DESC LIMIT $limit",
            {"index": index_name, "term": f"{term}~", "limit": limit},
        )

    def execute_cypher(self, cypher: str) -> dict[str, Any]:
        """Tool 5: Execute read-only Cypher query (write-blocked)."""
        if WRITE_KEYWORDS_PATTERN.search(cypher):
            return {"error": "Write operations are not allowed", "results": []}

        try:
            results = self.neo4j_svc.execute_read(cypher)
            return {"results": results, "count": len(results)}
        except Exception as exc:
            return {"error": str(exc), "results": []}

    def validate_cypher(self, cypher: str) -> dict[str, Any]:
        """Tool 6: Validate Cypher against discovered schema."""
        result = self.cypher_validator.validate(cypher)
        return {
            "valid": result.valid,
            "cypher": result.cypher,
            "errors": result.errors,
        }

    def list_indexes(self) -> dict[str, Any]:
        """Tool 7: List all discovered FULLTEXT indexes."""
        return self.index_svc.to_dict()

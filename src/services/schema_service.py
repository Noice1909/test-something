from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.services.neo4j_service import Neo4jService

logger = structlog.get_logger()


@dataclass
class SchemaCache:
    data: dict[str, Any] | None = None
    formatted: str = ""
    fetched_at: float = 0.0

    def is_stale(self, ttl: int) -> bool:
        return self.data is None or (time.time() - self.fetched_at) > ttl


class SchemaService:
    """Dynamically discovers graph schema via apoc.meta.schema()."""

    def __init__(self, neo4j_svc: Neo4jService, cache_ttl: int = 300) -> None:
        self.neo4j_svc = neo4j_svc
        self.cache_ttl = cache_ttl
        self._cache = SchemaCache()
        self._labels: list[str] = []
        self._relationship_types: list[str] = []
        self._schema_detail: dict[str, Any] = {}

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def relationship_types(self) -> list[str]:
        return self._relationship_types

    @property
    def schema_detail(self) -> dict[str, Any]:
        return self._schema_detail

    async def discover(self) -> None:
        """Fetch full schema from Neo4j and cache it."""
        # Labels
        label_rows = self.neo4j_svc.execute_read(
            "CALL db.labels() YIELD label RETURN label ORDER BY label"
        )
        self._labels = [r["label"] for r in label_rows]

        # Relationship types
        rel_rows = self.neo4j_svc.execute_read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
        )
        self._relationship_types = [r["relationshipType"] for r in rel_rows]

        # Rich schema via APOC
        try:
            meta_rows = self.neo4j_svc.execute_read(
                "CALL apoc.meta.schema() YIELD value RETURN value"
            )
            if meta_rows:
                self._schema_detail = meta_rows[0]["value"]
        except Exception as exc:
            logger.warning("apoc_meta_schema_failed", error=str(exc))
            self._schema_detail = {}

        # Build and cache formatted schema
        formatted = self._format_full_schema()
        self._cache = SchemaCache(
            data=self._schema_detail,
            formatted=formatted,
            fetched_at=time.time(),
        )

        logger.info(
            "schema_discovery_complete",
            labels=len(self._labels),
            relationships=len(self._relationship_types),
        )

    def get_full_schema_text(self) -> str:
        """Return the full formatted schema string."""
        if self._cache.is_stale(self.cache_ttl):
            # Synchronous re-fetch (called from within the pipeline)
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # If already in async context, schedule as task
                loop.create_task(self.discover())
            except RuntimeError:
                asyncio.run(self.discover())
        return self._cache.formatted

    def get_filtered_schema(self, relevant_labels: list[str]) -> str:
        """Return schema text filtered to only relevant labels + their neighbors."""
        if not relevant_labels:
            return self.get_full_schema_text()

        relevant_set = set(relevant_labels)

        # Add 1-hop neighbor labels from relationship patterns
        for label in list(relevant_set):
            label_meta = self._schema_detail.get(label, {})
            if isinstance(label_meta, dict):
                for rel_type, rel_info in label_meta.items():
                    if isinstance(rel_info, dict) and "labels" in rel_info:
                        for target_label in rel_info["labels"]:
                            relevant_set.add(target_label)

        return self._format_schema_for_labels(relevant_set)

    def get_pruned_schema(self, question: str) -> str:
        """Pruned-by-Exact-Match: filter schema by question tokens."""
        tokens = set(question.lower().split())

        matched_labels = []
        for label in self._labels:
            if label.lower() in tokens or any(t in label.lower() for t in tokens):
                matched_labels.append(label)

        if not matched_labels:
            return self.get_full_schema_text()

        return self.get_filtered_schema(matched_labels)

    def label_exists(self, label: str) -> bool:
        return label in self._labels

    def relationship_type_exists(self, rel_type: str) -> bool:
        return rel_type in self._relationship_types

    def get_properties_for_label(self, label: str) -> dict[str, str]:
        """Return {property_name: type} for a label."""
        label_meta = self._schema_detail.get(label, {})
        props = {}
        if isinstance(label_meta, dict):
            for key, val in label_meta.items():
                if isinstance(val, dict) and val.get("type") in (
                    "STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE",
                    "DATETIME", "POINT", "LIST", "LONG", "DOUBLE",
                ):
                    props[key] = val["type"]
                elif isinstance(val, str) and key not in ("type", "count"):
                    props[key] = val
        return props

    def _format_full_schema(self) -> str:
        return self._format_schema_for_labels(set(self._labels))

    def _format_schema_for_labels(self, label_set: set[str]) -> str:
        lines: list[str] = []

        lines.append("=== NODE LABELS ===")
        for label in sorted(label_set):
            meta = self._schema_detail.get(label, {})
            count = meta.get("count", "?") if isinstance(meta, dict) else "?"
            lines.append(f"{label} ({count} nodes)")

            # Properties
            props = self.get_properties_for_label(label)
            if props:
                prop_str = ", ".join(f"{k} ({v})" for k, v in sorted(props.items()))
                lines.append(f"  Properties: {prop_str}")

            # Relationships out
            if isinstance(meta, dict):
                for key, val in meta.items():
                    if isinstance(val, dict) and "direction" in val:
                        direction = val.get("direction", "out")
                        target_labels = val.get("labels", [])
                        if target_labels:
                            targets = "|".join(sorted(target_labels))
                            if direction == "out":
                                lines.append(f"  -[:{key}]->({targets})")
                            else:
                                lines.append(f"  <-[:{key}]-({targets})")

            lines.append("")

        # Relationship patterns summary
        lines.append("=== RELATIONSHIP PATTERNS ===")
        seen_patterns: set[str] = set()
        for label in sorted(label_set):
            meta = self._schema_detail.get(label, {})
            if not isinstance(meta, dict):
                continue
            for key, val in meta.items():
                if isinstance(val, dict) and "direction" in val:
                    for target in val.get("labels", []):
                        if val.get("direction") == "out":
                            pattern = f"(:{label})-[:{key}]->(:{target})"
                        else:
                            pattern = f"(:{target})-[:{key}]->(:{label})"
                        if pattern not in seen_patterns:
                            seen_patterns.add(pattern)
                            lines.append(pattern)
        lines.append("")

        # Property reference per label
        lines.append("=== PROPERTY NAMES PER LABEL ===")
        for label in sorted(label_set):
            props = self.get_properties_for_label(label)
            if props:
                lines.append(f"{label}: {', '.join(sorted(props.keys()))}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "labels": self._labels,
            "relationship_types": self._relationship_types,
            "label_count": len(self._labels),
            "relationship_type_count": len(self._relationship_types),
        }

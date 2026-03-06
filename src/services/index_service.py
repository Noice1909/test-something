from __future__ import annotations

import structlog

from src.services.neo4j_service import Neo4jService

logger = structlog.get_logger()


class IndexService:
    """Dynamically discovers all indexes from the database on startup."""

    def __init__(self, neo4j_svc: Neo4jService) -> None:
        self.neo4j_svc = neo4j_svc
        self.fulltext_by_label: dict[str, str] = {}
        self.global_indexes: dict[str, str] = {}
        self.special_indexes: dict[str, str] = {}
        self._all_labels: set[str] = set()

    @property
    def count(self) -> int:
        return len(self.fulltext_by_label) + len(self.global_indexes) + len(self.special_indexes)

    async def discover(self) -> None:
        """Query SHOW INDEXES and categorize all FULLTEXT indexes."""
        rows = self.neo4j_svc.execute_read(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties "
            "WHERE type = 'FULLTEXT' "
            "RETURN name, labelsOrTypes, properties"
        )

        # Also fetch all labels for detecting global indexes
        label_rows = self.neo4j_svc.execute_read(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        if label_rows:
            self._all_labels = set(label_rows[0]["labels"])

        for row in rows:
            name: str = row["name"]
            labels_or_types: list[str] = row.get("labelsOrTypes") or []
            properties: list[str] = row.get("properties") or []

            # Detect global indexes (cover all or most labels)
            if len(labels_or_types) >= len(self._all_labels) * 0.8 or len(labels_or_types) > 20:
                for prop in properties:
                    self.global_indexes[prop] = name
                logger.debug("global_index_found", name=name, properties=properties)
                continue

            # Multi-property indexes (special)
            if len(properties) > 1:
                key = f"{'+'.join(labels_or_types)}:{'+'.join(properties)}"
                self.special_indexes[key] = name
                logger.debug("special_index_found", name=name, key=key)
                continue

            # Per-label single-property FULLTEXT indexes
            for label in labels_or_types:
                self.fulltext_by_label[label] = name
                logger.debug("label_index_found", label=label, name=name)

        logger.info(
            "index_discovery_complete",
            per_label=len(self.fulltext_by_label),
            global_count=len(self.global_indexes),
            special=len(self.special_indexes),
        )

    def has_label_index(self, label: str) -> bool:
        return label in self.fulltext_by_label

    def get_label_index(self, label: str) -> str | None:
        return self.fulltext_by_label.get(label)

    def has_global_index(self, property_name: str) -> bool:
        return property_name in self.global_indexes

    def get_global_index(self, property_name: str) -> str | None:
        return self.global_indexes.get(property_name)

    def get_best_index(self, label: str | None, property_name: str) -> str | None:
        """Return the most specific index available, or None."""
        if label and self.has_label_index(label):
            return self.get_label_index(label)
        if self.has_global_index(property_name):
            return self.get_global_index(property_name)
        return None

    def get_special_index(self, key: str) -> str | None:
        return self.special_indexes.get(key)

    def to_dict(self) -> dict:
        return {
            "fulltext_by_label": self.fulltext_by_label,
            "global_indexes": self.global_indexes,
            "special_indexes": self.special_indexes,
        }

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pybreaker
import structlog

from src.config import Settings
from src.core.exceptions import OllamaUnavailableError
from src.prompts.entity_prompt import ENTITY_EXTRACTION_PROMPT
from src.services.concept_service import ConceptService
from src.services.index_service import IndexService
from src.services.neo4j_service import Neo4jService

logger = structlog.get_logger()


@dataclass
class ExtractedEntity:
    value: str
    entity_type: str  # "name" | "id" | "keyword"
    likely_label: str | None = None


@dataclass
class MappedEntity:
    original_value: str
    resolved_value: str
    label: str
    score: float
    property_name: str


class EntityExtractor:
    def __init__(
        self,
        neo4j_svc: Neo4jService,
        index_svc: IndexService,
        concept_svc: ConceptService,
        ollama_breaker: pybreaker.CircuitBreaker,
        settings: Settings,
        llm: Any,
    ) -> None:
        self.neo4j_svc = neo4j_svc
        self.index_svc = index_svc
        self.concept_svc = concept_svc
        self.ollama_breaker = ollama_breaker
        self.settings = settings
        self.llm = llm

    def extract_entities(self, question: str) -> list[ExtractedEntity]:
        """Use LLM to extract named entities from the question."""
        label_descriptions = self.concept_svc.get_label_descriptions()

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            label_descriptions=label_descriptions or "No concept metadata available.",
            question=question,
        )

        try:
            @self.ollama_breaker
            def _call_llm() -> str:
                response = self.llm.invoke(prompt)
                return response.content

            raw = _call_llm()
        except pybreaker.CircuitBreakerError as exc:
            raise OllamaUnavailableError("Ollama circuit breaker is open") from exc

        return self._parse_entities(raw)

    def map_entities(
        self,
        entities: list[ExtractedEntity],
        matched_concept_names: list[str],
    ) -> list[MappedEntity]:
        """Fuzzy-match extracted entities against actual DB values using fulltext indexes."""
        mapped: list[MappedEntity] = []

        for entity in entities:
            results = self._search_entity(entity, matched_concept_names)
            if results:
                best = results[0]
                mapped.append(
                    MappedEntity(
                        original_value=entity.value,
                        resolved_value=best.get("name") or best.get("id") or entity.value,
                        label=best.get("label", ""),
                        score=best.get("score", 0.0),
                        property_name=best.get("property", "name"),
                    )
                )
                logger.info(
                    "entity_mapped_detail",
                    original=entity.value,
                    resolved=mapped[-1].resolved_value,
                    label=mapped[-1].label,
                    score=mapped[-1].score,
                    best_result=best,
                )

        return mapped

    def _search_entity(
        self,
        entity: ExtractedEntity,
        matched_concept_names: list[str],
    ) -> list[dict[str, Any]]:
        """Search for an entity using the best available index."""
        # Determine which index to use
        label = entity.likely_label
        if not label and matched_concept_names:
            label = matched_concept_names[0]

        prop = "id" if entity.entity_type == "id" else "name"
        index_name = self.index_svc.get_best_index(label, prop)

        if index_name:
            return self._fulltext_search(index_name, entity.value, prop)
        else:
            return self._fallback_search(entity.value, label)

    def _fulltext_search(
        self, index_name: str, term: str, prop: str
    ) -> list[dict[str, Any]]:
        """Search using a FULLTEXT index with fuzzy matching."""
        search_term = f"{term}~"
        try:
            results = self.neo4j_svc.execute_read(
                "CALL db.index.fulltext.queryNodes($index_name, $search_term) "
                "YIELD node, score "
                "WHERE score > 0.3 "
                "RETURN node.name AS name, node.id AS id, "
                "labels(node)[0] AS label, score, $prop AS property "
                "ORDER BY score DESC LIMIT 5",
                {
                    "index_name": index_name,
                    "search_term": search_term,
                    "prop": prop,
                },
            )
            return results
        except Exception as exc:
            logger.warning("fulltext_search_failed", index=index_name, term=term, error=str(exc))
            return []

    def _fallback_search(
        self, term: str, label: str | None
    ) -> list[dict[str, Any]]:
        """Fallback: CONTAINS search when no fulltext index is available."""
        if label:
            cypher = (
                f"MATCH (n:{label}) "
                "WHERE toLower(n.name) CONTAINS toLower($term) "
                "RETURN n.name AS name, n.id AS id, labels(n)[0] AS label, "
                "1.0 AS score, 'name' AS property "
                "LIMIT 5"
            )
        else:
            cypher = (
                "MATCH (n) "
                "WHERE toLower(n.name) CONTAINS toLower($term) "
                "RETURN n.name AS name, n.id AS id, labels(n)[0] AS label, "
                "1.0 AS score, 'name' AS property "
                "LIMIT 5"
            )

        try:
            return self.neo4j_svc.execute_read(cypher, {"term": term})
        except Exception as exc:
            logger.warning("fallback_search_failed", term=term, error=str(exc))
            return []

    def _parse_entities(self, raw: str) -> list[ExtractedEntity]:
        """Parse LLM JSON output into ExtractedEntity list."""
        # Try to find JSON in the response
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("entity_extraction_no_json", raw=text[:200])
            return []

        try:
            data = json.loads(text[start:end])
            entities = data.get("entities", [])
            return [
                ExtractedEntity(
                    value=e.get("value", ""),
                    entity_type=e.get("type", "keyword"),
                    likely_label=e.get("likely_label"),
                )
                for e in entities
                if e.get("value")
            ]
        except json.JSONDecodeError as exc:
            logger.warning("entity_extraction_json_parse_failed", error=str(exc), raw=text[:200])
            return []

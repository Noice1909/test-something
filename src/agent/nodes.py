from __future__ import annotations

from dataclasses import asdict
from typing import Any

import structlog

from src.agent.state import AgentState
from src.services.concept_service import ConceptService
from src.services.cypher_generator import CypherGenerator
from src.services.cypher_validator import CypherValidator
from src.services.entity_extractor import EntityExtractor
from src.services.few_shot_service import FewShotService
from src.services.neo4j_service import Neo4jService
from src.services.response_generator import ResponseGenerator
from src.services.schema_service import SchemaService

logger = structlog.get_logger()


class AgentNodes:
    """All LangGraph node functions. Each returns a partial state update dict."""

    def __init__(
        self,
        concept_svc: ConceptService,
        entity_extractor: EntityExtractor,
        schema_svc: SchemaService,
        few_shot_svc: FewShotService,
        cypher_generator: CypherGenerator,
        cypher_validator: CypherValidator,
        neo4j_svc: Neo4jService,
        response_generator: ResponseGenerator,
    ) -> None:
        self.concept_svc = concept_svc
        self.entity_extractor = entity_extractor
        self.schema_svc = schema_svc
        self.few_shot_svc = few_shot_svc
        self.cypher_generator = cypher_generator
        self.cypher_validator = cypher_validator
        self.neo4j_svc = neo4j_svc
        self.response_generator = response_generator

    def match_concepts(self, state: AgentState) -> dict[str, Any]:
        """Step 1: Match question tokens against :Concept.nlp_terms."""
        question = state["question"]
        concepts = self.concept_svc.match_concepts(question)
        concept_dicts = [asdict(c) for c in concepts]
        logger.info("concepts_matched", count=len(concepts), names=[c.name for c in concepts])
        return {"matched_concepts": concept_dicts}

    def extract_entities(self, state: AgentState) -> dict[str, Any]:
        """Step 2: LLM extracts named entities from the question."""
        question = state["question"]
        try:
            entities = self.entity_extractor.extract_entities(question)
            entity_dicts = [
                {"value": e.value, "type": e.entity_type, "likely_label": e.likely_label}
                for e in entities
            ]
            logger.info("entities_extracted", count=len(entities))
            return {"extracted_entities": entity_dicts}
        except Exception as exc:
            logger.warning("entity_extraction_failed", error=str(exc))
            return {"extracted_entities": []}

    def map_entities(self, state: AgentState) -> dict[str, Any]:
        """Step 3: Fuzzy-match entities against DB values via fulltext indexes."""
        from src.services.entity_extractor import ExtractedEntity

        raw_entities = state.get("extracted_entities", [])
        entities = [
            ExtractedEntity(
                value=e["value"],
                entity_type=e["type"],
                likely_label=e.get("likely_label"),
            )
            for e in raw_entities
        ]

        concept_names = [c["name"] for c in state.get("matched_concepts", [])]

        try:
            mapped = self.entity_extractor.map_entities(entities, concept_names)
            mapped_dicts = [
                {
                    "original": m.original_value,
                    "resolved": m.resolved_value,
                    "label": m.label,
                    "score": m.score,
                    "property": m.property_name,
                }
                for m in mapped
            ]
            logger.info("entities_mapped", count=len(mapped))
            return {"mapped_entities": mapped_dicts}
        except Exception as exc:
            logger.warning("entity_mapping_failed", error=str(exc))
            return {"mapped_entities": []}

    def filter_schema(self, state: AgentState) -> dict[str, Any]:
        """Step 4: Filter schema to relevant labels based on matched concepts."""
        concept_names = [c["name"] for c in state.get("matched_concepts", [])]
        mapped_labels = [m["label"] for m in state.get("mapped_entities", []) if m.get("label")]

        relevant_labels = list(set(concept_names + mapped_labels))

        if relevant_labels:
            filtered = self.schema_svc.get_filtered_schema(relevant_labels)
        else:
            filtered = self.schema_svc.get_pruned_schema(state["question"])

        return {"filtered_schema": filtered}

    def retrieve_examples(self, state: AgentState) -> dict[str, Any]:
        """Step 5: Retrieve similar few-shot examples from ChromaDB."""
        question = state["question"]
        examples_text = self.few_shot_svc.retrieve(question)
        return {"few_shot_examples": examples_text}

    def generate_cypher(self, state: AgentState) -> dict[str, Any]:
        """Step 6: LLM generates Cypher using schema + entities + examples."""
        question = state["question"]
        filtered_schema = state.get("filtered_schema", "")
        few_shot = state.get("few_shot_examples", "")

        # Format mapped entities for the prompt AND rewrite the question
        mapped = state.get("mapped_entities", [])
        logger.info("generate_cypher_mapped_entities", mapped=mapped)

        # Rewrite question to use resolved entity values
        rewritten_question = question
        if mapped:
            lines = []
            for m in mapped:
                lines.append(
                    f"- \"{m['original']}\" resolved to \"{m['resolved']}\" "
                    f"(label: {m['label']}, property: {m['property']})"
                )
                # Replace original entity with resolved entity in the question
                # Try case-sensitive first, then case-insensitive
                if m['original'] in rewritten_question:
                    rewritten_question = rewritten_question.replace(m['original'], m['resolved'])
                else:
                    # Case-insensitive replacement for when LLM corrects typos during extraction
                    import re
                    pattern = re.compile(re.escape(m['original']), re.IGNORECASE)
                    rewritten_question = pattern.sub(m['resolved'], rewritten_question)
            entities_text = "\n".join(lines)
        else:
            entities_text = ""

        logger.info("generate_cypher_entities_text", entities_text=entities_text,
                   original_question=question, rewritten_question=rewritten_question)
        cypher = self.cypher_generator.generate(
            question=rewritten_question,  # Use rewritten question with resolved entities
            filtered_schema=filtered_schema,
            mapped_entities_text=entities_text,
            few_shot_text=few_shot,
        )

        return {"cypher": cypher, "validation_errors": [], "retry_count": state.get("retry_count", 0)}

    def validate_cypher(self, state: AgentState) -> dict[str, Any]:
        """Step 7: Validate the generated Cypher query."""
        cypher = state.get("cypher", "")

        # Check for UNABLE_TO_GENERATE signal
        if cypher.startswith("UNABLE_TO_GENERATE"):
            reason = cypher.split("\n", 1)[1] if "\n" in cypher else "Could not generate query"
            return {"validation_errors": [reason], "retry_count": state.get("retry_count", 0) + 999}

        result = self.cypher_validator.validate(cypher)

        if result.valid:
            return {"cypher": result.cypher, "validation_errors": []}
        else:
            logger.warning("cypher_validation_failed", errors=result.errors)
            return {
                "cypher": result.cypher,
                "validation_errors": result.errors,
                "retry_count": state.get("retry_count", 0) + 1,
            }

    def correct_cypher(self, state: AgentState) -> dict[str, Any]:
        """Self-correction: feed error back to LLM for a fix."""
        errors = state.get("validation_errors", [])
        error_text = "; ".join(errors)

        # Reconstruct mapped entities text (same format as in generate_cypher node)
        mapped = state.get("mapped_entities", [])
        entities_text = ""
        if mapped:
            lines = []
            for m in mapped:
                lines.append(
                    f"- \"{m['original']}\" resolved to \"{m['resolved']}\" "
                    f"(label: {m['label']}, property: {m['property']})"
                )
            entities_text = "\n".join(lines)

        corrected = self.cypher_generator.correct(
            question=state["question"],
            failed_cypher=state.get("cypher", ""),
            error_message=error_text,
            filtered_schema=state.get("filtered_schema", ""),
            mapped_entities_text=entities_text,
        )

        logger.info("cypher_correction_attempt", retry=state.get("retry_count", 0))
        return {"cypher": corrected, "validation_errors": []}

    def execute_query(self, state: AgentState) -> dict[str, Any]:
        """Step 8: Execute the validated Cypher against Neo4j."""
        cypher = state.get("cypher", "")
        try:
            results = self.neo4j_svc.execute_read(cypher)
            logger.info("query_executed", result_count=len(results))
            return {"query_results": results}
        except Exception as exc:
            logger.error("query_execution_failed", error=str(exc))
            return {"query_results": [], "error": str(exc)}

    def generate_response(self, state: AgentState) -> dict[str, Any]:
        """Step 9: Convert results to natural language answer."""
        # If we have fatal errors and no results, return error message
        if state.get("error") and not state.get("query_results"):
            return {
                "answer": "I had trouble looking that up. Could you try rephrasing your question?"
            }

        # If validation exhausted retries, return friendly message
        if state.get("retry_count", 0) >= 999:
            return {
                "answer": "I wasn't able to understand that question well enough to look it up. "
                "Could you try asking in a different way?"
            }

        results = state.get("query_results", [])
        question = state["question"]

        try:
            answer = self.response_generator.generate(question, results)
            return {"answer": answer}
        except Exception as exc:
            logger.error("response_generation_failed", error=str(exc))
            if results:
                return {"answer": "I found some information but had trouble formatting the answer. Please try again."}
            return {"answer": "Something went wrong. Please try again."}

    @staticmethod
    def validation_router(state: AgentState) -> str:
        """Route after validation: pass, fail (retry), or fatal (give up)."""
        errors = state.get("validation_errors", [])
        retry_count = state.get("retry_count", 0)

        if not errors:
            return "pass"

        from src.config import settings
        if retry_count >= settings.MAX_CYPHER_RETRIES:
            return "fatal"

        return "fail"

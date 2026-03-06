from __future__ import annotations

from typing import Any

import pybreaker
import structlog

from src.config import Settings
from src.core.exceptions import OllamaUnavailableError
from src.prompts.cypher_prompt import CYPHER_SYSTEM_PROMPT
from src.services.schema_service import SchemaService

logger = structlog.get_logger()


class CypherGenerator:
    def __init__(
        self,
        schema_svc: SchemaService,
        ollama_breaker: pybreaker.CircuitBreaker,
        settings: Settings,
        llm: Any,
    ) -> None:
        self.schema_svc = schema_svc
        self.ollama_breaker = ollama_breaker
        self.settings = settings
        self.llm = llm

    def generate(
        self,
        question: str,
        filtered_schema: str,
        mapped_entities_text: str,
        few_shot_text: str,
    ) -> str:
        """Generate a Cypher query from a natural language question."""
        logger.info("cypher_gen_inputs", question=question, mapped_entities_text=mapped_entities_text)
        system_prompt = CYPHER_SYSTEM_PROMPT.format(
            filtered_schema=filtered_schema,
            mapped_entities=mapped_entities_text or "No specific entities identified.",
            few_shot_examples=few_shot_text or "No similar examples available.",
            question=question,
        )

        try:
            @self.ollama_breaker
            def _call() -> str:
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ])
                return response.content.strip()

            raw = _call()
        except pybreaker.CircuitBreakerError as exc:
            raise OllamaUnavailableError("Ollama circuit breaker is open") from exc

        # Clean up: strip markdown fences if present
        cypher = self._clean_cypher(raw)
        logger.info("cypher_generated", cypher=cypher[:200])
        return cypher

    def correct(
        self,
        question: str,
        failed_cypher: str,
        error_message: str,
        filtered_schema: str,
        mapped_entities_text: str = "",
    ) -> str:
        """Ask the LLM to fix a failed Cypher query."""
        from src.prompts.correction_prompt import CORRECTION_PROMPT

        prompt = CORRECTION_PROMPT.format(
            failed_cypher=failed_cypher,
            error_message=error_message,
            question=question,
            mapped_entities=mapped_entities_text or "No specific entities identified.",
            filtered_schema=filtered_schema,
        )

        try:
            @self.ollama_breaker
            def _call() -> str:
                response = self.llm.invoke(prompt)
                return response.content.strip()

            raw = _call()
        except pybreaker.CircuitBreakerError as exc:
            raise OllamaUnavailableError("Ollama circuit breaker is open") from exc

        cypher = self._clean_cypher(raw)
        logger.info("cypher_corrected", cypher=cypher[:200])
        return cypher

    def _clean_cypher(self, raw: str) -> str:
        """Remove markdown fences and extra whitespace."""
        text = raw.strip()

        # Remove ```cypher ... ``` or ``` ... ```
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```cypher or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Remove trailing semicolons
        text = text.rstrip(";").strip()

        return text

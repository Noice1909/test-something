from __future__ import annotations

import json
from typing import Any

import pybreaker
import structlog

from src.config import Settings
from src.core.exceptions import OllamaUnavailableError
from src.prompts.response_prompt import RESPONSE_SYSTEM_PROMPT

logger = structlog.get_logger()


class ResponseGenerator:
    def __init__(
        self,
        ollama_breaker: pybreaker.CircuitBreaker,
        settings: Settings,
        llm: Any,
    ) -> None:
        self.ollama_breaker = ollama_breaker
        self.settings = settings
        self.llm = llm

    def generate(self, question: str, results: list[dict[str, Any]]) -> str:
        """Convert raw Neo4j results into a natural language answer."""
        if not results:
            return self._empty_response()

        # Truncate results for LLM context
        display_results = results[:25]
        results_text = json.dumps(display_results, indent=2, default=str)

        user_prompt = (
            f"User question: {question}\n\n"
            f"Data ({len(results)} total results, showing {len(display_results)}):\n"
            f"{results_text}"
        )

        try:
            @self.ollama_breaker
            def _call() -> str:
                response = self.llm.invoke([
                    {"role": "system", "content": RESPONSE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ])
                return response.content.strip()

            return _call()
        except pybreaker.CircuitBreakerError as exc:
            raise OllamaUnavailableError("Ollama circuit breaker is open") from exc

    def _empty_response(self) -> str:
        return (
            "I wasn't able to find any information matching your question. "
            "Could you try rephrasing it or being more specific?"
        )

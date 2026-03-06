from __future__ import annotations

import pybreaker

from src.config import Settings


def create_circuit_breakers(settings: Settings) -> dict[str, pybreaker.CircuitBreaker]:
    """Create circuit breakers for external dependencies."""

    neo4j_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CB_FAIL_MAX,
        reset_timeout=settings.CB_RESET_TIMEOUT,
        name="neo4j",
    )

    ollama_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CB_FAIL_MAX,
        reset_timeout=settings.CB_RESET_TIMEOUT,
        name="ollama",
    )

    return {
        "neo4j": neo4j_breaker,
        "ollama": ollama_breaker,
    }


def get_breaker_states(breakers: dict[str, pybreaker.CircuitBreaker]) -> dict[str, str]:
    """Return human-readable state of each circuit breaker."""
    return {
        name: breaker.current_state
        for name, breaker in breakers.items()
    }

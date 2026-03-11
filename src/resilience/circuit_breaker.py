"""Circuit breaker instances for Neo4j and LLM services."""

from __future__ import annotations

import logging
from typing import Any

import pybreaker

from src.config import settings

logger = logging.getLogger(__name__)


# ── Custom exception ──


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is OPEN and calls are being rejected."""

    def __init__(self, service: str, retry_after: int) -> None:
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker OPEN for {service}. Retry after {retry_after}s.")


# ── Listener ──


class _LoggingListener(pybreaker.CircuitBreakerListener):
    """Logs circuit breaker state transitions."""

    def state_change(self, cb: pybreaker.CircuitBreaker, old_state: Any, new_state: Any) -> None:
        logger.warning(
            "Circuit breaker '%s': %s -> %s",
            cb.name,
            old_state.name if hasattr(old_state, "name") else old_state,
            new_state.name if hasattr(new_state, "name") else new_state,
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: Exception) -> None:
        logger.debug("Circuit breaker '%s' recorded failure: %s", cb.name, exc)


_listener = _LoggingListener()

# ── Breaker instances ──

neo4j_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.cb_neo4j_fail_max,
    reset_timeout=settings.cb_neo4j_reset_timeout,
    name="neo4j",
    listeners=[_listener],
)

llm_breaker = pybreaker.CircuitBreaker(
    fail_max=settings.cb_llm_fail_max,
    reset_timeout=settings.cb_llm_reset_timeout,
    name="llm",
    listeners=[_listener],
)


def get_breaker_states() -> dict[str, str]:
    """Return a snapshot of all circuit breaker states."""
    return {
        "neo4j": neo4j_breaker.current_state,
        "llm": llm_breaker.current_state,
    }

"""Tests for circuit breaker behavior."""

from __future__ import annotations

from src.resilience.circuit_breaker import (
    CircuitOpenError,
    get_breaker_states,
    neo4j_breaker,
    llm_breaker,
)


class TestCircuitBreakerStates:

    def test_initial_state_is_closed(self):
        states = get_breaker_states()
        assert states["neo4j"] == "closed"
        assert states["llm"] == "closed"

    def test_circuit_open_error_has_attributes(self):
        err = CircuitOpenError("test_service", 30)
        assert err.service == "test_service"
        assert err.retry_after == 30
        assert "test_service" in str(err)

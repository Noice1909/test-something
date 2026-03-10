"""Unit tests for AgentState."""

from src.agents.state import AgentState
from src.agents.base import StrategyType


class TestAgentState:
    def test_defaults(self):
        s = AgentState(question="test question")
        assert s.question == "test question"
        assert s.attempt_number == 1
        assert s.strategy == StrategyType.DISCOVERY_FIRST
        assert s.discoveries == []
        assert s.history == []
        assert s.trace_id.startswith("agentic-")

    def test_log_specialist(self):
        s = AgentState(question="q")
        s.log_specialist("discovery", success=True, duration_ms=42.0, detail="found 3")
        assert len(s.history) == 1
        assert s.history[0]["specialist"] == "discovery"
        assert s.history[0]["success"] is True
        assert s.history[0]["duration_ms"] == 42.0

    def test_elapsed_ms(self):
        s = AgentState(question="q")
        # elapsed_ms should be >= 0
        assert s.elapsed_ms >= 0

    def test_custom_trace_id(self):
        s = AgentState(question="q", trace_id="custom-123")
        assert s.trace_id == "custom-123"

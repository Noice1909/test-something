"""Unit tests for base data structures."""

from src.agents.base import (
    StrategyType,
    QueryComplexity,
    RetryStrategy,
    SpecialistResult,
    DiscoveryResult,
    SchemaSelection,
    QueryPlan,
    GeneratedQuery,
    ExecutionResult,
    ReflectionResult,
    SupervisorDecision,
    AgenticResponse,
)


class TestEnums:
    def test_strategy_type_values(self):
        assert StrategyType.DISCOVERY_FIRST.value == "discovery_first"
        assert StrategyType.DIRECT_QUERY.value == "direct_query"
        assert StrategyType.SCHEMA_EXPLORATION.value == "schema_exploration"
        assert StrategyType.AGGREGATION.value == "aggregation"

    def test_query_complexity_values(self):
        assert QueryComplexity.DIRECT.value == "DIRECT"
        assert QueryComplexity.ONE_HOP.value == "ONE_HOP"
        assert QueryComplexity.MULTI_HOP.value == "MULTI_HOP"
        assert QueryComplexity.AGGREGATION.value == "AGGREGATION"

    def test_retry_strategy_values(self):
        assert RetryStrategy.EXPAND_DISCOVERY.value == "expand_discovery"
        assert RetryStrategy.GIVE_UP.value == "give_up"


class TestDataclasses:
    def test_specialist_result_defaults(self):
        r = SpecialistResult(success=True)
        assert r.success is True
        assert r.data is None
        assert r.error is None
        assert r.duration_ms == 0.0

    def test_discovery_result(self):
        d = DiscoveryResult(entity_name="Test", label="Movie", confidence=0.95)
        assert d.entity_name == "Test"
        assert d.confidence == 0.95
        assert d.properties == {}

    def test_schema_selection_defaults(self):
        s = SchemaSelection()
        assert s.node_labels == []
        assert s.relationship_types == []

    def test_query_plan_defaults(self):
        p = QueryPlan()
        assert p.strategy == QueryComplexity.DIRECT
        assert p.intent == ""

    def test_generated_query_defaults(self):
        g = GeneratedQuery()
        assert g.query == ""
        assert g.language == "cypher"
        assert g.is_read_only is True

    def test_execution_result_defaults(self):
        e = ExecutionResult()
        assert e.success is False
        assert e.rows == []

    def test_reflection_result_defaults(self):
        r = ReflectionResult()
        assert r.should_retry is False
        assert r.retry_strategy == RetryStrategy.GIVE_UP

    def test_supervisor_decision(self):
        d = SupervisorDecision(
            strategy=StrategyType.AGGREGATION,
            reasoning="Count query detected",
        )
        assert d.strategy == StrategyType.AGGREGATION

    def test_agentic_response(self):
        r = AgenticResponse(
            answer="42 movies found",
            strategy_used="aggregation",
            attempts=1,
            success=True,
            trace_id="test-123",
        )
        assert r.success is True
        assert r.specialist_log == []

"""Tests for pipeline optimizations: combined SchemaQuerySpecialist,
schema caching in AgentState, optional answer formatting, and combined
strategy sequences."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import (
    DiscoveryResult,
    GeneratedQuery,
    QueryComplexity,
    SchemaSelection,
    SpecialistResult,
    StrategyType,
)
from src.agents.specialists.schema_query import SchemaQuerySpecialist
from src.agents.state import AgentState


# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_schema() -> dict:
    return {
        "labels": ["Application", "Domain", "SubDomain", "Person"],
        "relationship_types": ["HAS_SUBDOMAIN", "BELONGS_TO", "RUNS_ON"],
        "relationship_patterns": [
            {"from": "Domain", "type": "HAS_SUBDOMAIN", "to": "SubDomain"},
            {"from": "Application", "type": "BELONGS_TO", "to": "Domain"},
            {"from": "Application", "type": "RUNS_ON", "to": "SubDomain"},
        ],
        "label_properties": {
            "Application": ["name", "id", "status"],
            "Domain": ["name"],
            "SubDomain": ["name"],
            "Person": ["name", "age"],
        },
    }


def _make_mock_db(schema: dict | None = None) -> AsyncMock:
    db = AsyncMock()
    db.get_schema = AsyncMock(return_value=schema or _make_schema())
    db.execute_read = AsyncMock(return_value=[])
    return db


def _make_combined_response(
    node_labels: list[str] | None = None,
    relationship_types: list[str] | None = None,
    strategy: str = "ONE_HOP",
    intent: str = "LIST",
    query: str = "MATCH (n:Application) RETURN n.name LIMIT 25",
    parameters: dict | None = None,
) -> str:
    return json.dumps({
        "node_labels": node_labels or ["Application", "Domain"],
        "relationship_types": relationship_types or ["BELONGS_TO"],
        "schema_reasoning": "App belongs to domain",
        "strategy": strategy,
        "intent": intent,
        "query": query,
        "parameters": parameters or {},
        "reasoning": "Simple traversal",
    })


# ══════════════════════════════════════════════════════════════════════════════
# 1. SchemaQuerySpecialist — combined specialist
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaQueryParsing:
    """Test _parse_response for the combined specialist."""

    def test_valid_combined_response(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        text = _make_combined_response()
        result = specialist._parse_response(text, schema)

        assert result["node_labels"] == ["Application", "Domain"]
        assert result["relationship_types"] == ["BELONGS_TO"]
        assert result["strategy"] == QueryComplexity.ONE_HOP
        assert result["intent"] == "LIST"
        assert "MATCH" in result["query"]
        assert result["is_read_only"] is True

    def test_filters_invalid_labels(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        text = _make_combined_response(
            node_labels=["Application", "FakeLabel"],
            relationship_types=["BELONGS_TO", "FAKE_REL"],
        )
        result = specialist._parse_response(text, schema)

        assert "FakeLabel" not in result["node_labels"]
        assert "Application" in result["node_labels"]
        assert "FAKE_REL" not in result["relationship_types"]

    def test_invalid_json_returns_fallback(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        result = specialist._parse_response("not json at all", schema)

        assert result["node_labels"] == schema["labels"]
        assert "parse failed" in result["schema_reasoning"].lower()

    def test_json_embedded_in_markdown(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        text = "```json\n" + _make_combined_response() + "\n```"
        result = specialist._parse_response(text, schema)

        assert result["node_labels"] == ["Application", "Domain"]
        assert "MATCH" in result["query"]

    def test_write_query_detected(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        text = _make_combined_response(
            query="CREATE (n:Application {name: 'test'}) RETURN n"
        )
        result = specialist._parse_response(text, schema)

        assert result["is_read_only"] is False

    def test_unknown_strategy_defaults_direct(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        text = _make_combined_response(strategy="UNKNOWN_STRATEGY")
        result = specialist._parse_response(text, schema)

        assert result["strategy"] == QueryComplexity.DIRECT

    def test_all_strategy_values(self):
        schema = _make_schema()
        specialist = SchemaQuerySpecialist.__new__(SchemaQuerySpecialist)
        for name, expected in [
            ("DIRECT", QueryComplexity.DIRECT),
            ("ONE_HOP", QueryComplexity.ONE_HOP),
            ("TWO_HOP", QueryComplexity.TWO_HOP),
            ("MULTI_HOP", QueryComplexity.MULTI_HOP),
            ("AGGREGATION", QueryComplexity.AGGREGATION),
        ]:
            text = _make_combined_response(strategy=name)
            result = specialist._parse_response(text, schema)
            assert result["strategy"] == expected, f"Failed for {name}"


class TestSchemaQueryRun:
    """Test the full run() method of SchemaQuerySpecialist."""

    @pytest.mark.asyncio
    async def test_populates_all_state_fields(self):
        """run() should populate schema_selection, query_plan, AND generated_query."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="list applications in domain X")
        state.discoveries = [
            DiscoveryResult(entity_name="X", label="Domain", confidence=0.9),
        ]

        result = await specialist.run(state)

        assert result.success is True
        # Schema selection
        assert "Application" in state.schema_selection.node_labels
        assert "BELONGS_TO" in state.schema_selection.relationship_types
        # Query plan
        assert state.query_plan.strategy == QueryComplexity.ONE_HOP
        assert state.query_plan.intent == "LIST"
        # Generated query
        assert "MATCH" in state.generated_query.query
        assert state.generated_query.is_read_only is True

    @pytest.mark.asyncio
    async def test_single_llm_call(self):
        """Combined specialist must make exactly ONE LLM call."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="list all applications")
        await specialist.run(state)

        assert llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_uses_cached_schema_from_state(self):
        """Should use state.schema instead of calling db.get_schema."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()

        await specialist.run(state)

        db.get_schema.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_db_schema_when_state_empty(self):
        """Should fetch from DB when state.schema is None."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="test")
        assert state.schema is None

        await specialist.run(state)

        db.get_schema.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_write_query(self):
        """Should reject queries containing write operations."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response(
                query="CREATE (n:Application {name: 'x'}) RETURN n"
            )
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="create app")
        result = await specialist.run(state)

        assert result.success is False
        assert "write" in result.error.lower()

    @pytest.mark.asyncio
    async def test_fallback_all_labels_when_none_selected(self):
        """When LLM selects 0 valid labels, should fallback to ALL labels."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response(
                node_labels=["NonExistentLabel"],
                relationship_types=[],
            )
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="something weird")
        state.schema = _make_schema()

        result = await specialist.run(state)

        assert result.success is True
        assert set(state.schema_selection.node_labels) == set(_make_schema()["labels"])
        assert "fallback" in state.schema_selection.reasoning.lower()

    @pytest.mark.asyncio
    async def test_llm_error_returns_failure(self):
        """When LLM call raises, should return failure."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()

        result = await specialist.run(state)

        assert result.success is False
        assert "LLM timeout" in result.error

    @pytest.mark.asyncio
    async def test_logs_to_state_history(self):
        """run() should log to state.history."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()
        await specialist.run(state)

        assert len(state.history) == 1
        assert state.history[0]["specialist"] == "schema_query"
        assert state.history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_logs_cypher_attempt(self):
        """run() should log a cypher attempt on success."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()
        await specialist.run(state)

        assert len(state.cypher_attempts) == 1
        assert "MATCH" in state.cypher_attempts[0]["cypher"]

    @pytest.mark.asyncio
    async def test_retry_context_included(self):
        """When previous_empty_queries exist, prompt should include retry context."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=_make_combined_response()
        ))

        specialist = SchemaQuerySpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()
        state.previous_empty_queries = [
            {"query": "MATCH (n) RETURN n LIMIT 10", "reasoning": "first try"},
        ]

        await specialist.run(state)

        # Verify prompt included retry context
        prompt_text = llm.ainvoke.call_args[0][0]
        assert "returned 0 rows" in prompt_text
        assert "DIFFERENT approach" in prompt_text


# ══════════════════════════════════════════════════════════════════════════════
# 2. AgentState — schema caching
# ══════════════════════════════════════════════════════════════════════════════


class TestAgentStateSchema:

    def test_schema_defaults_to_none(self):
        state = AgentState(question="test")
        assert state.schema is None

    def test_schema_can_be_set(self):
        state = AgentState(question="test")
        state.schema = _make_schema()
        assert state.schema is not None
        assert "labels" in state.schema


# ══════════════════════════════════════════════════════════════════════════════
# 3. Combined strategy sequences
# ══════════════════════════════════════════════════════════════════════════════


class TestCombinedSequences:

    def test_all_strategies_have_combined_sequences(self):
        from src.agents.supervisor import _COMBINED_STRATEGY_SEQUENCES
        for strategy in StrategyType:
            assert strategy in _COMBINED_STRATEGY_SEQUENCES, (
                f"Missing combined sequence for {strategy}"
            )

    def test_combined_sequences_shorter_than_original(self):
        from src.agents.supervisor import (
            _COMBINED_STRATEGY_SEQUENCES,
            _STRATEGY_SEQUENCES,
        )
        for strategy in StrategyType:
            combined = _COMBINED_STRATEGY_SEQUENCES[strategy]
            original = _STRATEGY_SEQUENCES[strategy]
            assert len(combined) <= len(original), (
                f"Combined sequence for {strategy} should be <= original"
            )

    def test_combined_sequences_use_schema_query(self):
        from src.agents.supervisor import _COMBINED_STRATEGY_SEQUENCES
        for strategy, seq in _COMBINED_STRATEGY_SEQUENCES.items():
            assert "schema_query" in seq, (
                f"{strategy} should use schema_query"
            )
            assert "schema_planning" not in seq
            assert "query_generation" not in seq

    def test_combined_empty_retry_uses_schema_query(self):
        from src.agents.supervisor import _COMBINED_EMPTY_RETRY_SEQUENCE
        assert "schema_query" in _COMBINED_EMPTY_RETRY_SEQUENCE
        assert "execution" in _COMBINED_EMPTY_RETRY_SEQUENCE
        assert len(_COMBINED_EMPTY_RETRY_SEQUENCE) == 2


# ══════════════════════════════════════════════════════════════════════════════
# 4. Optional answer formatting
# ══════════════════════════════════════════════════════════════════════════════


class TestOptionalAnswerFormatting:

    def test_format_answer_raw_with_results(self):
        from src.agents.supervisor import Supervisor
        rows = [{"name": "App1"}, {"name": "App2"}]
        result = Supervisor._format_answer_raw(rows)
        assert "Found 2 result(s)" in result
        assert "App1" in result
        assert "App2" in result

    def test_format_answer_raw_empty(self):
        from src.agents.supervisor import Supervisor
        result = Supervisor._format_answer_raw([])
        assert "No matching data" in result

    def test_format_answer_raw_truncates_long_output(self):
        from src.agents.supervisor import Supervisor
        rows = [{"data": "x" * 200} for _ in range(50)]
        result = Supervisor._format_answer_raw(rows)
        assert len(result) < 4000  # should be truncated

    @pytest.mark.asyncio
    async def test_format_answer_skips_llm_when_disabled(self):
        """When format_answer_with_llm=False, should NOT call LLM."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="formatted"))

        from src.agents.supervisor import Supervisor
        supervisor = Supervisor.__new__(Supervisor)
        supervisor._llm = llm

        rows = [{"name": "App1"}]
        with patch("src.agents.supervisor.settings") as mock_settings:
            mock_settings.format_answer_with_llm = False
            result = await supervisor._format_answer("test?", rows)

        llm.ainvoke.assert_not_called()
        assert "App1" in result

    @pytest.mark.asyncio
    async def test_format_answer_calls_llm_when_enabled(self):
        """When format_answer_with_llm=True, should call LLM."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Nice answer"))

        from src.agents.supervisor import Supervisor
        supervisor = Supervisor.__new__(Supervisor)
        supervisor._llm = llm

        rows = [{"name": "App1"}]
        with patch("src.agents.supervisor.settings") as mock_settings:
            mock_settings.format_answer_with_llm = True
            result = await supervisor._format_answer("test?", rows)

        llm.ainvoke.assert_called_once()
        assert result == "Nice answer"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Schema caching in specialists
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaCachingInSpecialists:

    @pytest.mark.asyncio
    async def test_schema_planning_uses_cached_schema(self):
        from src.agents.specialists.schema_planning import SchemaPlanningSpecialist
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "node_labels": ["Application"],
                "relationship_types": ["BELONGS_TO"],
                "schema_reasoning": "test",
                "strategy": "DIRECT",
                "intent": "LIST",
            })
        ))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()

        await specialist.run(state)

        db.get_schema.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_generation_uses_cached_schema(self):
        from src.agents.specialists.query_generation import QueryGenerationSpecialist
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "query": "MATCH (n:Application) RETURN n LIMIT 25",
                "parameters": {},
                "reasoning": "test",
            })
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()
        state.schema_selection = SchemaSelection(
            node_labels=["Application"],
            relationship_types=["BELONGS_TO"],
        )

        await specialist.run(state)

        db.get_schema.assert_not_called()

    @pytest.mark.asyncio
    async def test_reflection_uses_cached_schema(self):
        from src.agents.specialists.reflection import ReflectionSpecialist
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "should_retry": True,
                "next_approach": "try different query",
                "suggested_hops": 1,
                "reasoning": "wrong property",
            })
        ))

        specialist = ReflectionSpecialist(db, llm, {})
        state = AgentState(question="test")
        state.schema = _make_schema()
        state.generated_query = GeneratedQuery(
            query="MATCH (n) RETURN n", reasoning="test"
        )

        await specialist.run_empty_result(state)

        db.get_schema.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Config settings
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigSettings:

    def test_combine_schema_query_default_true(self):
        from src.config import Settings
        s = Settings()
        assert s.combine_schema_query is True

    def test_format_answer_with_llm_default_true(self):
        from src.config import Settings
        s = Settings()
        assert s.format_answer_with_llm is True

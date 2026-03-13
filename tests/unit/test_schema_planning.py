"""Tests for the merged SchemaPlanningSpecialist."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import (
    DiscoveryResult,
    QueryComplexity,
    SchemaPlan,
    SpecialistResult,
)
from src.agents.specialists.schema_planning import SchemaPlanningSpecialist
from src.agents.state import AgentState


def _make_schema() -> dict:
    """Build a realistic mock schema dict."""
    return {
        "labels": ["Application", "Domain", "SubDomain", "Person"],
        "relationship_types": ["HAS_SUBDOMAIN", "BELONGS_TO", "RUNS_ON"],
        "relationship_patterns": [
            {"from": "Domain", "type": "HAS_SUBDOMAIN", "to": "SubDomain", "bidirectional": False},
            {"from": "Application", "type": "BELONGS_TO", "to": "Domain", "bidirectional": False},
            {"from": "Application", "type": "RUNS_ON", "to": "SubDomain", "bidirectional": False},
        ],
        "label_properties": {
            "Application": ["name", "id"],
            "Domain": ["name"],
            "SubDomain": ["name"],
            "Person": ["name", "age"],
        },
    }


class TestSchemaPlanningParsing:
    """Test the _parse_response method and JSON handling."""

    def test_valid_combined_response(self):
        """Correct JSON should produce a full SchemaPlan."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        text = json.dumps({
            "node_labels": ["Application", "Domain"],
            "relationship_types": ["BELONGS_TO"],
            "schema_reasoning": "App belongs to domain",
            "strategy": "ONE_HOP",
            "intent": "LIST",
            "plan_reasoning": "Simple traversal needed",
            "filters": {"name": "CNAPP"},
        })

        result = specialist._parse_response(text, schema)

        assert isinstance(result, SchemaPlan)
        assert result.node_labels == ["Application", "Domain"]
        assert result.relationship_types == ["BELONGS_TO"]
        assert result.schema_reasoning == "App belongs to domain"
        assert result.strategy == QueryComplexity.ONE_HOP
        assert result.intent == "LIST"
        assert result.plan_reasoning == "Simple traversal needed"
        assert result.filters == {"name": "CNAPP"}

    def test_filters_invalid_labels(self):
        """Labels not in schema should be filtered out."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        text = json.dumps({
            "node_labels": ["Application", "FakeLabel", "Domain"],
            "relationship_types": ["BELONGS_TO", "FAKE_REL"],
            "schema_reasoning": "test",
            "strategy": "DIRECT",
            "intent": "FIND",
        })

        result = specialist._parse_response(text, schema)

        assert "FakeLabel" not in result.node_labels
        assert "Application" in result.node_labels
        assert "Domain" in result.node_labels
        assert "FAKE_REL" not in result.relationship_types
        assert "BELONGS_TO" in result.relationship_types

    def test_invalid_json_returns_defaults(self):
        """Invalid JSON should return a SchemaPlan with default values."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        result = specialist._parse_response("not json at all", schema)

        assert isinstance(result, SchemaPlan)
        assert result.node_labels == []
        assert result.relationship_types == []
        assert "parse failed" in result.schema_reasoning.lower()

    def test_json_embedded_in_text(self):
        """JSON embedded in markdown code blocks should be extracted."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        text = "Here is my analysis:\n```json\n" + json.dumps({
            "node_labels": ["Person"],
            "relationship_types": [],
            "schema_reasoning": "looking for person",
            "strategy": "DIRECT",
            "intent": "DESCRIBE",
        }) + "\n```"

        result = specialist._parse_response(text, schema)
        assert result.node_labels == ["Person"]
        assert result.intent == "DESCRIBE"

    def test_unknown_strategy_defaults_to_direct(self):
        """Unknown strategy string should default to DIRECT."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        text = json.dumps({
            "node_labels": ["Application"],
            "strategy": "UNKNOWN_STRATEGY",
            "intent": "LIST",
        })

        result = specialist._parse_response(text, schema)
        assert result.strategy == QueryComplexity.DIRECT

    def test_all_strategy_values(self):
        """All valid strategy strings should map correctly."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        for name, expected in [
            ("DIRECT", QueryComplexity.DIRECT),
            ("ONE_HOP", QueryComplexity.ONE_HOP),
            ("TWO_HOP", QueryComplexity.TWO_HOP),
            ("MULTI_HOP", QueryComplexity.MULTI_HOP),
            ("AGGREGATION", QueryComplexity.AGGREGATION),
        ]:
            text = json.dumps({"strategy": name, "intent": "LIST"})
            result = specialist._parse_response(text, schema)
            assert result.strategy == expected, f"Failed for strategy {name}"

    def test_case_insensitive_strategy(self):
        """Strategy matching should be case-insensitive."""
        schema = _make_schema()
        specialist = SchemaPlanningSpecialist.__new__(SchemaPlanningSpecialist)

        text = json.dumps({"strategy": "one_hop", "intent": "list"})
        result = specialist._parse_response(text, schema)
        assert result.strategy == QueryComplexity.ONE_HOP
        assert result.intent == "LIST"


class TestSchemaPlanningRun:
    """Test the full run method of SchemaPlanningSpecialist."""

    @pytest.mark.asyncio
    async def test_populates_both_state_fields(self):
        """run() should populate both state.schema_selection and state.query_plan."""
        schema = _make_schema()
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=schema)
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "node_labels": ["Application", "Domain"],
                "relationship_types": ["BELONGS_TO"],
                "schema_reasoning": "App-Domain relationship",
                "strategy": "ONE_HOP",
                "intent": "LIST",
                "plan_reasoning": "Single traversal",
                "filters": {},
            })
        ))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="list applications in domain X")
        state.discoveries = [
            DiscoveryResult(entity_name="X", label="Domain", confidence=0.9),
        ]

        result = await specialist.run(state)

        assert result.success is True

        # Schema selection populated
        assert "Application" in state.schema_selection.node_labels
        assert "Domain" in state.schema_selection.node_labels
        assert "BELONGS_TO" in state.schema_selection.relationship_types

        # Query plan populated
        assert state.query_plan.strategy == QueryComplexity.ONE_HOP
        assert state.query_plan.intent == "LIST"

    @pytest.mark.asyncio
    async def test_fallback_when_zero_labels_selected(self):
        """When LLM selects 0 valid labels, should fallback to ALL labels."""
        schema = _make_schema()
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=schema)
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "node_labels": ["NonExistentLabel"],
                "relationship_types": [],
                "schema_reasoning": "wrong",
                "strategy": "DIRECT",
                "intent": "FIND",
            })
        ))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="something weird")

        result = await specialist.run(state)

        assert result.success is True
        # Fallback: all labels from schema
        assert set(state.schema_selection.node_labels) == set(schema["labels"])
        assert "fallback" in state.schema_selection.reasoning.lower()

    @pytest.mark.asyncio
    async def test_fallback_when_zero_rels_selected(self):
        """When LLM selects 0 valid relationship types, should fallback to ALL."""
        schema = _make_schema()
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=schema)
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "node_labels": ["Application"],
                "relationship_types": ["NONEXISTENT_REL"],
                "schema_reasoning": "test",
                "strategy": "DIRECT",
                "intent": "FIND",
            })
        ))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="test")

        result = await specialist.run(state)

        assert result.success is True
        assert set(state.schema_selection.relationship_types) == set(schema["relationship_types"])

    @pytest.mark.asyncio
    async def test_llm_error_returns_failure(self):
        """When LLM call raises, should return failure SpecialistResult."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="test query")

        result = await specialist.run(state)

        assert result.success is False
        assert "LLM timeout" in result.error

    @pytest.mark.asyncio
    async def test_db_schema_error_returns_failure(self):
        """When db.get_schema raises, should return failure SpecialistResult."""
        db = AsyncMock()
        db.get_schema = AsyncMock(side_effect=RuntimeError("DB unreachable"))
        llm = AsyncMock()

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="test query")

        result = await specialist.run(state)

        assert result.success is False
        assert "DB unreachable" in result.error

    @pytest.mark.asyncio
    async def test_single_llm_call_only(self):
        """The merged specialist should make exactly ONE LLM call (not two)."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
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
        state = AgentState(question="list all applications")

        await specialist.run(state)

        assert llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_specialist_logs_to_state(self):
        """run() should log to state.history."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "node_labels": ["Application"],
                "relationship_types": [],
                "schema_reasoning": "test",
                "strategy": "DIRECT",
                "intent": "LIST",
            })
        ))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="test")

        await specialist.run(state)

        assert len(state.history) == 1
        assert state.history[0]["specialist"] == "schema_planning"
        assert state.history[0]["success"] is True
        assert state.history[0]["duration_ms"] > 0

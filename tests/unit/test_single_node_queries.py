"""Tests for single-node property query fixes.

Verifies that when schema_planning is skipped (PROPERTY_LOOKUP, ENTITY_DETAIL),
query_generation auto-populates schema context from discoveries so the LLM
has full visibility into node properties.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import (
    DiscoveryResult,
    QueryComplexity,
    SpecialistResult,
    StrategyType,
)
from src.agents.specialists.query_generation import QueryGenerationSpecialist
from src.agents.specialists.schema_planning import SchemaPlanningSpecialist
from src.agents.state import AgentState


def _make_schema() -> dict:
    """Build a realistic mock schema dict."""
    return {
        "labels": ["Application", "Domain", "SubDomain"],
        "relationship_types": ["BELONGS_TO", "HAS_SUBDOMAIN", "RUNS_ON"],
        "relationship_patterns": [
            {"from": "Application", "type": "BELONGS_TO", "to": "Domain", "bidirectional": False},
            {"from": "Domain", "type": "HAS_SUBDOMAIN", "to": "SubDomain", "bidirectional": False},
            {"from": "Application", "type": "RUNS_ON", "to": "SubDomain", "bidirectional": False},
        ],
        "label_properties": {
            "Application": ["name", "risk_level", "description", "status"],
            "Domain": ["name"],
            "SubDomain": ["name"],
        },
    }


def _mock_llm_response(query: str, reasoning: str = "test") -> MagicMock:
    """Create a mock LLM response returning a Cypher query JSON."""
    return MagicMock(content=json.dumps({
        "query": query,
        "parameters": {},
        "reasoning": reasoning,
    }))


# ── Fix 1: Auto-populate schema_selection from discoveries ───────────────────


class TestAutoPopulateSchema:

    @pytest.mark.asyncio
    async def test_populates_schema_from_discoveries(self):
        """When schema_planning is skipped, labels should be derived from discoveries."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application) WHERE a.name = 'CNAPP' RETURN a.risk_level LIMIT 1"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="what is the risk_level of application CNAPP?")
        state.discoveries = [
            DiscoveryResult(
                entity_name="CNAPP", label="Application",
                node_id="42", confidence=0.95,
                properties={"name": "CNAPP"},
            ),
        ]
        # schema_selection is empty (schema_planning was skipped)
        assert state.schema_selection.node_labels == []

        result = await specialist.run(state)

        assert result.success is True
        assert "Application" in state.schema_selection.node_labels
        # Relationship types involving Application should be populated
        assert "BELONGS_TO" in state.schema_selection.relationship_types
        assert "RUNS_ON" in state.schema_selection.relationship_types

    @pytest.mark.asyncio
    async def test_preserves_existing_schema_selection(self):
        """When schema_planning already ran, auto-populate should NOT override."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (d:Domain) RETURN d.name LIMIT 25"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="list all domains")
        state.discoveries = [
            DiscoveryResult(
                entity_name="Finance", label="Domain",
                node_id="10", confidence=0.9,
            ),
        ]
        # Pre-populate as if schema_planning ran
        state.schema_selection.node_labels = ["Domain"]
        state.schema_selection.relationship_types = ["HAS_SUBDOMAIN"]

        await specialist.run(state)

        # Should NOT have been overridden
        assert state.schema_selection.node_labels == ["Domain"]
        assert state.schema_selection.relationship_types == ["HAS_SUBDOMAIN"]

    @pytest.mark.asyncio
    async def test_filters_invalid_discovery_labels(self):
        """Discovery labels not in schema should be excluded."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application) RETURN a.name LIMIT 25"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="find FakeEntity")
        state.discoveries = [
            DiscoveryResult(
                entity_name="X", label="FakeLabel",
                node_id="1", confidence=0.8,
            ),
            DiscoveryResult(
                entity_name="Y", label="Application",
                node_id="2", confidence=0.7,
            ),
        ]

        await specialist.run(state)

        assert "FakeLabel" not in state.schema_selection.node_labels
        assert "Application" in state.schema_selection.node_labels

    @pytest.mark.asyncio
    async def test_skips_unknown_labels_from_discoveries(self):
        """Discoveries with label='Unknown' should be excluded."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application) RETURN a.name LIMIT 25"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="find something")
        state.discoveries = [
            DiscoveryResult(
                entity_name="mystery", label="Unknown",
                node_id="99", confidence=0.3,
            ),
        ]

        await specialist.run(state)

        assert "Unknown" not in state.schema_selection.node_labels

    @pytest.mark.asyncio
    async def test_no_discoveries_leaves_schema_empty(self):
        """When there are no discoveries and no schema_planning, schema stays empty."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (n) RETURN n.name LIMIT 25"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="show me everything")
        # No discoveries, no schema_planning

        await specialist.run(state)

        assert state.schema_selection.node_labels == []


# ── Fix 2: Auto-populate query_plan intent ───────────────────────────────────


class TestAutoPopulateIntent:

    @pytest.mark.asyncio
    async def test_property_lookup_gets_find_intent(self):
        """PROPERTY_LOOKUP strategy should auto-populate intent='FIND'."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application {name: 'X'}) RETURN a LIMIT 1"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="find application named X")
        state.strategy = StrategyType.PROPERTY_LOOKUP
        state.discoveries = [
            DiscoveryResult(entity_name="X", label="Application", confidence=0.9),
        ]
        assert state.query_plan.intent == ""

        await specialist.run(state)

        assert state.query_plan.intent == "FIND"

    @pytest.mark.asyncio
    async def test_entity_detail_gets_describe_intent(self):
        """ENTITY_DETAIL strategy should auto-populate intent='DESCRIBE'."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application {name: 'CNAPP'}) RETURN a LIMIT 1"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="describe application CNAPP")
        state.strategy = StrategyType.ENTITY_DETAIL
        state.discoveries = [
            DiscoveryResult(entity_name="CNAPP", label="Application", confidence=0.9),
        ]

        await specialist.run(state)

        assert state.query_plan.intent == "DESCRIBE"

    @pytest.mark.asyncio
    async def test_preserves_existing_intent(self):
        """When schema_planning already set an intent, it should NOT be overridden."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application) RETURN a.name LIMIT 25"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="list all applications")
        state.schema_selection.node_labels = ["Application"]
        state.query_plan.intent = "LIST"
        state.query_plan.strategy = QueryComplexity.DIRECT
        state.discoveries = [
            DiscoveryResult(entity_name="X", label="Application", confidence=0.9),
        ]

        await specialist.run(state)

        assert state.query_plan.intent == "LIST"


# ── Fix 3: Prompt includes label properties ──────────────────────────────────


class TestPromptContainsProperties:

    @pytest.mark.asyncio
    async def test_prompt_includes_label_properties_after_auto_populate(self):
        """The LLM prompt should contain the label's properties when auto-populated."""
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=_make_schema())
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=_mock_llm_response(
            "MATCH (a:Application {name: 'CNAPP'}) RETURN a.risk_level LIMIT 1"
        ))

        specialist = QueryGenerationSpecialist(db, llm, {})
        state = AgentState(question="what is the risk_level of CNAPP?")
        state.strategy = StrategyType.PROPERTY_LOOKUP
        state.discoveries = [
            DiscoveryResult(
                entity_name="CNAPP", label="Application",
                node_id="42", confidence=0.95,
            ),
        ]

        await specialist.run(state)

        # Capture the prompt sent to the LLM
        prompt = llm.ainvoke.call_args[0][0]
        # Should contain Application's properties
        assert "risk_level" in prompt
        assert "description" in prompt
        assert "status" in prompt
        # Should contain the label
        assert "Application" in prompt


# ── Fix 4: Schema planning prompt includes label_properties ──────────────────


class TestSchemaPlanningLabelProperties:

    @pytest.mark.asyncio
    async def test_schema_planning_prompt_includes_label_properties(self):
        """The schema planning LLM prompt should now include label_properties."""
        schema = _make_schema()
        db = AsyncMock()
        db.get_schema = AsyncMock(return_value=schema)
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "node_labels": ["Application"],
                "relationship_types": [],
                "schema_reasoning": "Property exists on node",
                "strategy": "DIRECT",
                "intent": "FIND",
                "plan_reasoning": "Single node lookup",
            })
        ))

        specialist = SchemaPlanningSpecialist(db, llm, {})
        state = AgentState(question="what is the risk_level of CNAPP?")

        await specialist.run(state)

        # Capture the prompt sent to the LLM
        prompt = llm.ainvoke.call_args[0][0]
        # Should contain label properties section
        assert "Application: name, risk_level, description, status" in prompt
        assert "Domain: name" in prompt

"""Tests for the hybrid strategy classifier (rule-based + LLM fallback)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import StrategyType, SupervisorDecision
from src.agents.state import AgentState
from src.agents.supervisor import _classify_strategy


class TestClassifyStrategy:
    """Tests for the rule-based _classify_strategy function."""

    # ── Schema / structure questions → SCHEMA_EXPLORATION ──

    @pytest.mark.parametrize("question", [
        "what labels exist in the database?",
        "show me the schema",
        "what relationships are available?",
        "what types of nodes are there?",
        "what is the database structure?",
    ])
    def test_schema_exploration(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.SCHEMA_EXPLORATION
        assert confidence == "high"

    # ── Path queries → PATH_QUERY ──

    @pytest.mark.parametrize("question", [
        "find the path between Application A and Domain B",
        "how is Tom connected to that movie?",
        "shortest path from user1 to user2",
    ])
    def test_path_query(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.PATH_QUERY
        assert confidence == "high"

    # ── Comparison → COMPARISON ──

    @pytest.mark.parametrize("question", [
        "compare Application A and Application B",
        "what is the difference between Domain X and Domain Y?",
        "Application A vs Application B",
    ])
    def test_comparison(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.COMPARISON
        assert confidence == "high"

    # ── Aggregation → AGGREGATION ──

    @pytest.mark.parametrize("question", [
        "which actor has the most movies?",
        "top 5 applications by connections",
        "highest rated movies",
        "rank domains by subdomain count",
    ])
    def test_aggregation(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.AGGREGATION
        assert confidence == "high"

    # ── Simple count → SIMPLE_COUNT ──

    @pytest.mark.parametrize("question", [
        "how many applications are there?",
        "count of movies in the database",
        "total number of domains",
    ])
    def test_simple_count(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.SIMPLE_COUNT
        assert confidence == "high"

    # ── Property lookup → PROPERTY_LOOKUP ──

    @pytest.mark.parametrize("question", [
        "show the application named CNAPP",
        "find the movie called Inception",
        "entity with name FooBar",
    ])
    def test_property_lookup(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.PROPERTY_LOOKUP
        assert confidence == "high"

    # ── Entity detail → ENTITY_DETAIL ──

    @pytest.mark.parametrize("question", [
        "tell me about Tom Hanks",
        "describe the CNAPP application",
        "details of Domain ABC",
        "info about the Finance domain",
    ])
    def test_entity_detail(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.ENTITY_DETAIL
        assert confidence == "high"

    # ── Relationship query → RELATIONSHIP_QUERY ──

    @pytest.mark.parametrize("question", [
        "what depends on Domain X?",
        "which applications are linked to this component?",
        "what is associated with the Finance domain?",
    ])
    def test_relationship_query(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.RELATIONSHIP_QUERY
        assert confidence == "high"

    # ── Label list → LABEL_LIST ──

    @pytest.mark.parametrize("question", [
        "list all domains",
        "show all applications",
        "what are the genres?",
        "show me all users",
    ])
    def test_label_list(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.LABEL_LIST
        assert confidence == "high"

    # ── No pattern → DISCOVERY_FIRST with low confidence ──

    @pytest.mark.parametrize("question", [
        "CNAPP",
        "something random",
        "xyzzy 12345",
    ])
    def test_no_match_returns_discovery_first_low(self, question: str):
        strategy, reasoning, confidence = _classify_strategy(question)
        assert strategy == StrategyType.DISCOVERY_FIRST
        assert confidence == "low"
        assert "No pattern matched" in reasoning

    # ── Multiple matches → first wins, low confidence ──

    def test_multiple_matches_returns_low_confidence(self):
        # "how many applications are connected to Domain X?"
        # Matches: SIMPLE_COUNT ("how many") and RELATIONSHIP_QUERY ("connected to")
        strategy, reasoning, confidence = _classify_strategy(
            "how many applications are connected to Domain X?"
        )
        assert confidence == "low"
        assert "Multiple matches" in reasoning

    # ── Empty question ──

    def test_empty_question(self):
        strategy, reasoning, confidence = _classify_strategy("")
        assert strategy == StrategyType.DISCOVERY_FIRST
        assert confidence == "low"

    # ── Very long question ──

    def test_long_question(self):
        question = "tell me about " + "word " * 200 + "the application"
        strategy, reasoning, confidence = _classify_strategy(question)
        # Should still work — "tell me about" matches ENTITY_DETAIL
        assert strategy == StrategyType.ENTITY_DETAIL


class TestHybridDecideStrategy:
    """Tests for the full _decide_strategy method on the Supervisor."""

    @pytest.mark.asyncio
    async def test_high_confidence_skips_llm(self):
        """When rule-based returns high confidence, LLM should NOT be called."""
        from src.agents.supervisor import Supervisor

        db = AsyncMock()
        llm = AsyncMock()
        tools = {}

        with patch.object(Supervisor, "__init__", lambda self, *a, **k: None):
            sup = Supervisor.__new__(Supervisor)
            sup._llm = llm
            sup._cache = AsyncMock()
            sup._cache.get = AsyncMock(return_value=None)
            sup._cache.set = AsyncMock()

            state = AgentState(question="how many movies are there?")
            decision = await sup._decide_strategy("how many movies are there?", state)

        assert decision.strategy == StrategyType.SIMPLE_COUNT
        assert "[rule]" in decision.reasoning
        # LLM should NOT have been invoked
        llm.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_confidence_calls_llm(self):
        """When rule-based returns low confidence, LLM should be called."""
        from src.agents.supervisor import Supervisor

        db = AsyncMock()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({"strategy": "discovery_first", "reasoning": "ambiguous"})
        ))
        tools = {}

        with patch.object(Supervisor, "__init__", lambda self, *a, **k: None):
            sup = Supervisor.__new__(Supervisor)
            sup._llm = llm
            sup._cache = AsyncMock()
            sup._cache.get = AsyncMock(return_value=None)
            sup._cache.set = AsyncMock()

            state = AgentState(question="CNAPP stuff")
            decision = await sup._decide_strategy("CNAPP stuff", state)

        assert "[llm]" in decision.reasoning
        llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_strategy_returned(self):
        """When strategy is in cache, return immediately without rules or LLM."""
        from src.agents.supervisor import Supervisor

        llm = AsyncMock()

        with patch.object(Supervisor, "__init__", lambda self, *a, **k: None):
            sup = Supervisor.__new__(Supervisor)
            sup._llm = llm
            sup._cache = AsyncMock()
            sup._cache.get = AsyncMock(return_value={
                "strategy": "aggregation",
                "reasoning": "cached decision",
            })

            state = AgentState(question="top 5 movies")
            decision = await sup._decide_strategy("top 5 movies", state)

        assert decision.strategy == StrategyType.AGGREGATION
        assert decision.reasoning == "cached decision"
        llm.ainvoke.assert_not_called()


class TestStrategySequences:
    """Verify that each strategy maps to a valid specialist sequence."""

    def test_all_strategies_have_sequences(self):
        from src.agents.supervisor import _STRATEGY_SEQUENCES
        for strat in StrategyType:
            assert strat in _STRATEGY_SEQUENCES, f"Missing sequence for {strat}"

    def test_all_sequences_end_with_execution(self):
        from src.agents.supervisor import _STRATEGY_SEQUENCES
        for strat, seq in _STRATEGY_SEQUENCES.items():
            assert seq[-1] == "execution", f"{strat} doesn't end with execution"

    def test_narrowed_strategies_are_shorter(self):
        from src.agents.supervisor import _STRATEGY_SEQUENCES
        full_len = len(_STRATEGY_SEQUENCES[StrategyType.DISCOVERY_FIRST])
        # These should skip either discovery or schema_planning
        assert len(_STRATEGY_SEQUENCES[StrategyType.SIMPLE_COUNT]) < full_len
        assert len(_STRATEGY_SEQUENCES[StrategyType.LABEL_LIST]) < full_len
        assert len(_STRATEGY_SEQUENCES[StrategyType.PROPERTY_LOOKUP]) < full_len
        assert len(_STRATEGY_SEQUENCES[StrategyType.ENTITY_DETAIL]) < full_len

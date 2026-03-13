"""Tests for the optimized Discovery specialist (deterministic + parallel + single LLM)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import SpecialistResult
from src.agents.specialists.discovery import (
    DiscoverySpecialist,
    _build_deep_fuzzy_plan,
    _build_fuzzy_plan,
    _build_tool_plan,
    _extract_search_terms,
    _levenshtein_threshold,
    _score_result,
    _smart_truncate,
)
from src.agents.state import AgentState


# ── _extract_search_terms ─────────────────────────────────────────────────────


class TestExtractSearchTerms:

    def test_basic_extraction(self):
        terms = _extract_search_terms("what movies did Tom Hanks act in?")
        assert "Tom" in terms
        assert "Hanks" in terms
        assert "movies" in terms

    def test_stop_words_removed(self):
        terms = _extract_search_terms("what is the name of the application?")
        for w in ("what", "is", "the", "of"):
            assert w not in terms
        assert "name" in terms
        assert "application" in terms

    def test_quoted_strings_preserved(self):
        terms = _extract_search_terms('find "Tom Hanks" in movies')
        assert "Tom Hanks" in terms  # quoted phrase kept intact
        assert terms[0] == "Tom Hanks"  # quoted terms come first

    def test_single_char_words_removed(self):
        terms = _extract_search_terms("a b c application")
        assert "a" not in terms
        assert "b" not in terms
        assert "c" not in terms
        assert "application" in terms

    def test_empty_question(self):
        terms = _extract_search_terms("")
        assert terms == []

    def test_only_stop_words(self):
        terms = _extract_search_terms("what is the")
        assert terms == []

    def test_mixed_case_stop_words(self):
        terms = _extract_search_terms("What IS the Application?")
        # "What" and "IS" should be recognized as stop words (case insensitive)
        assert "Application" in terms
        assert len(terms) == 1


# ── _build_tool_plan ──────────────────────────────────────────────────────────


class TestBuildToolPlan:

    def test_includes_text_search(self):
        tools = {"search_nodes_by_text": AsyncMock(), "get_all_labels": AsyncMock()}
        plan = _build_tool_plan(["CNAPP", "application"], tools, [])
        names = [name for name, _ in plan]
        assert "search_nodes_by_text" in names

    def test_includes_fulltext_index_searches(self):
        tools = {
            "search_nodes_by_text": AsyncMock(),
            "search_nodes_using_fulltext_index": AsyncMock(),
            "get_all_labels": AsyncMock(),
        }
        indexes = [
            {"name": "idx_app_name", "labelsOrTypes": ["Application"], "properties": ["name"]},
            {"name": "idx_domain_name", "labelsOrTypes": ["Domain"], "properties": ["name"]},
        ]
        plan = _build_tool_plan(["CNAPP"], tools, indexes)
        ft_calls = [name for name, _ in plan if name == "search_nodes_using_fulltext_index"]
        assert len(ft_calls) == 2  # one per index

    def test_limits_fulltext_indexes_to_3(self):
        tools = {
            "search_nodes_using_fulltext_index": AsyncMock(),
            "get_all_labels": AsyncMock(),
        }
        indexes = [
            {"name": f"idx_{i}", "labelsOrTypes": [f"Label{i}"], "properties": ["name"]}
            for i in range(5)
        ]
        plan = _build_tool_plan(["term"], tools, indexes)
        ft_calls = [name for name, _ in plan if name == "search_nodes_using_fulltext_index"]
        assert len(ft_calls) == 3  # capped at 3

    def test_always_includes_get_all_labels(self):
        tools = {"get_all_labels": AsyncMock()}
        plan = _build_tool_plan(["term"], tools, [])
        names = [name for name, _ in plan]
        assert "get_all_labels" in names

    def test_empty_terms_no_text_search(self):
        tools = {"search_nodes_by_text": AsyncMock(), "get_all_labels": AsyncMock()}
        plan = _build_tool_plan([], tools, [])
        names = [name for name, _ in plan]
        # search_nodes_by_text should NOT be included with empty text
        assert "search_nodes_by_text" not in names

    def test_missing_tools_gracefully_skipped(self):
        # No tools available at all
        plan = _build_tool_plan(["term"], {}, [])
        assert plan == []

    def test_property_targeted_when_label_in_terms(self):
        """When a term matches a label name, property-targeted search is added."""
        schema = {
            "labels": ["Application", "Domain"],
            "label_properties": {"Application": ["name", "id", "description"]},
        }
        tools = {
            "search_nodes_by_property_case_insensitive": AsyncMock(),
            "search_nodes_by_text": AsyncMock(),
            "get_all_labels": AsyncMock(),
        }
        plan = _build_tool_plan(["Application", "CNAPP"], tools, [], schema=schema)
        names = [name for name, _ in plan]
        assert "search_nodes_by_property_case_insensitive" in names
        # Verify the args target the Application label with name property
        prop_calls = [(n, kw) for n, kw in plan if n == "search_nodes_by_property_case_insensitive"]
        assert prop_calls[0][1]["label"] == "Application"
        assert prop_calls[0][1]["key"] == "name"
        assert prop_calls[0][1]["value"] == "CNAPP"

    def test_case_insensitive_text_search_always_used(self):
        """search_nodes_by_text (now case-insensitive) is always included as fallback."""
        tools = {"search_nodes_by_text": AsyncMock(), "get_all_labels": AsyncMock()}
        plan = _build_tool_plan(["CNAPP"], tools, [])
        names = [name for name, _ in plan]
        assert "search_nodes_by_text" in names

    def test_schema_none_backward_compat(self):
        """schema=None should still produce a valid plan."""
        tools = {"search_nodes_by_text": AsyncMock(), "get_all_labels": AsyncMock()}
        plan = _build_tool_plan(["term"], tools, [], schema=None)
        names = [name for name, _ in plan]
        assert "search_nodes_by_text" in names
        assert "get_all_labels" in names

    def test_fulltext_enhanced_fuzzy_added_for_long_terms(self):
        """Enhanced fulltext with Lucene fuzzy is added for terms >= 4 chars."""
        tools = {
            "search_nodes_using_fulltext_index": AsyncMock(),
            "search_nodes_using_fulltext_enhanced": AsyncMock(),
            "get_all_labels": AsyncMock(),
        }
        indexes = [{"name": "idx1"}]
        plan = _build_tool_plan(["CNAPP"], tools, indexes)  # 5 chars
        names = [name for name, _ in plan]
        assert "search_nodes_using_fulltext_enhanced" in names

    def test_fulltext_enhanced_not_added_for_short_terms(self):
        """Enhanced fulltext NOT added when all terms < 4 chars."""
        tools = {
            "search_nodes_using_fulltext_index": AsyncMock(),
            "search_nodes_using_fulltext_enhanced": AsyncMock(),
            "get_all_labels": AsyncMock(),
        }
        indexes = [{"name": "idx1"}]
        plan = _build_tool_plan(["HK"], tools, indexes)  # 2 chars
        names = [name for name, _ in plan]
        assert "search_nodes_using_fulltext_enhanced" not in names


# ── _build_fuzzy_plan ─────────────────────────────────────────────────────────


class TestBuildFuzzyPlan:

    def test_fuzzy_for_capitalized_terms(self):
        tools = {"apoc_text_fuzzyMatch": AsyncMock()}
        plan = _build_fuzzy_plan(["CNAPP", "domain"], ["Application", "Domain"], tools)
        assert len(plan) > 0
        assert plan[0][0] == "apoc_text_fuzzyMatch"

    def test_no_fuzzy_without_tool(self):
        plan = _build_fuzzy_plan(["CNAPP"], ["Application"], {})
        assert plan == []

    def test_no_fuzzy_for_short_lowercase_terms(self):
        tools = {"apoc_text_fuzzyMatch": AsyncMock()}
        plan = _build_fuzzy_plan(["app", "db"], ["Application", "Database"], tools)
        # "app" and "db" are <=4 chars and lowercase → no entity terms
        assert plan == []

    def test_fuzzy_matches_label_by_term(self):
        tools = {"apoc_text_fuzzyMatch": AsyncMock()}
        plan = _build_fuzzy_plan(
            ["Application", "Finance"],
            ["Application", "Domain", "SubDomain"],
            tools,
        )
        # "Application" in terms matches "Application" label
        labels_used = [kwargs["label"] for _, kwargs in plan]
        assert "Application" in labels_used


# ── _levenshtein_threshold ───────────────────────────────────────────────────


class TestLevenshteinThreshold:

    def test_short_term_threshold_equals_length(self):
        assert _levenshtein_threshold("HK") == 2
        assert _levenshtein_threshold("AB") == 2
        assert _levenshtein_threshold("X") == 1
        assert _levenshtein_threshold("ABC") == 3

    def test_medium_term_threshold_is_2(self):
        assert _levenshtein_threshold("CNAPP") == 2
        assert _levenshtein_threshold("domain") == 2
        assert _levenshtein_threshold("Finance") == 2

    def test_long_term_threshold_is_3(self):
        assert _levenshtein_threshold("Application") == 3
        assert _levenshtein_threshold("something_very_long") == 3


# ── _build_deep_fuzzy_plan ───────────────────────────────────────────────────


class TestBuildDeepFuzzyPlan:

    def test_uses_label_scoped_when_labels_available(self):
        tools = {"fuzzy_search_label_properties": AsyncMock()}
        plan = _build_deep_fuzzy_plan(["HK"], tools, labels=["Application", "Domain"])
        assert len(plan) > 0
        assert all(n == "fuzzy_search_label_properties" for n, _ in plan)

    def test_falls_back_to_global_when_no_labels(self):
        tools = {"fuzzy_search_all_properties": AsyncMock()}
        plan = _build_deep_fuzzy_plan(["HK"], tools, labels=None)
        assert len(plan) == 1
        assert plan[0][0] == "fuzzy_search_all_properties"

    def test_falls_back_to_global_when_empty_labels(self):
        tools = {"fuzzy_search_all_properties": AsyncMock()}
        plan = _build_deep_fuzzy_plan(["HK"], tools, labels=[])
        assert len(plan) == 1
        assert plan[0][0] == "fuzzy_search_all_properties"

    def test_adaptive_threshold_short_term(self):
        tools = {"fuzzy_search_all_properties": AsyncMock()}
        plan = _build_deep_fuzzy_plan(["HK"], tools)
        assert plan[0][1]["threshold"] == 2  # len("HK") = 2

    def test_adaptive_threshold_long_term(self):
        tools = {"fuzzy_search_all_properties": AsyncMock()}
        plan = _build_deep_fuzzy_plan(["Application"], tools)
        assert plan[0][1]["threshold"] == 3  # long term

    def test_empty_when_no_tools(self):
        plan = _build_deep_fuzzy_plan(["HK"], {})
        assert plan == []

    def test_limits_to_2_terms(self):
        tools = {"fuzzy_search_all_properties": AsyncMock()}
        plan = _build_deep_fuzzy_plan(["A", "B", "C", "D"], tools)
        assert len(plan) == 2  # only first 2 terms


# ── _score_result ─────────────────────────────────────────────────────────────


class TestScoreResult:

    def test_fulltext_score(self):
        result = {"id": "1", "score": 0.95}
        assert _score_result(result, "search_nodes_using_fulltext_index") == 0.95

    def test_fuzzy_sim(self):
        result = {"id": "1", "sim": 0.8}
        assert _score_result(result, "apoc_text_fuzzyMatch") == 0.8

    def test_text_search_default(self):
        result = {"id": "1", "name": "Test"}
        assert _score_result(result, "search_nodes_by_text") == 0.5

    def test_unknown_tool_default(self):
        result = {"id": "1"}
        assert _score_result(result, "get_all_labels") == 0.3

    def test_levenshtein_distance_0_gives_1(self):
        result = {"id": "1", "minDist": 0}
        assert _score_result(result, "fuzzy_search_all_properties") == 1.0

    def test_levenshtein_distance_1_gives_0_8(self):
        result = {"id": "1", "minDist": 1}
        assert _score_result(result, "fuzzy_search_all_properties") == 0.8

    def test_levenshtein_distance_2_gives_0_6(self):
        result = {"id": "1", "minDist": 2}
        assert _score_result(result, "fuzzy_search_label_properties") == 0.6

    def test_levenshtein_distance_high_floors_at_0_1(self):
        result = {"id": "1", "minDist": 10}
        assert _score_result(result, "fuzzy_search_all_properties") == 0.1

    def test_property_targeted_score(self):
        result = {"id": "1"}
        assert _score_result(result, "search_nodes_by_property_case_insensitive") == 0.85
        assert _score_result(result, "search_nodes_by_property") == 0.85

    def test_prefix_search_score(self):
        result = {"id": "1"}
        assert _score_result(result, "search_nodes_by_text_prefix") == 0.7


# ── _smart_truncate ───────────────────────────────────────────────────────────


class TestSmartTruncate:

    def test_deduplicates_by_id(self):
        tool_results = {
            "search_1": [
                {"id": "1", "labels": ["App"], "name": "CNAPP", "score": 0.9},
                {"id": "2", "labels": ["App"], "name": "Other", "score": 0.5},
            ],
            "search_2": [
                {"id": "1", "labels": ["App"], "name": "CNAPP", "score": 0.8},  # duplicate
            ],
        }
        result = _smart_truncate(tool_results)
        # id "1" should appear only once
        assert result.count('"id": "1"') <= 1

    def test_scores_ranked_highest_first(self):
        tool_results = {
            "search": [
                {"id": "low", "name": "Low"},
                {"id": "high", "name": "High", "score": 0.99},
            ],
        }
        result = _smart_truncate(tool_results)
        # High-scored result should come before low-scored
        pos_high = result.find("High")
        pos_low = result.find("Low")
        assert pos_high < pos_low

    def test_empty_results(self):
        result = _smart_truncate({})
        assert "No results found" in result

    def test_non_dict_results_skipped(self):
        tool_results = {
            "tool1": ["string_result", 42, None],
        }
        result = _smart_truncate(tool_results)
        assert "No results found" in result

    def test_budget_respected(self):
        """Results should not exceed the total budget."""
        # Create many results
        tool_results = {
            "search": [
                {"id": str(i), "name": f"Entity{i}", "labels": ["Test"],
                 "description": "x" * 200}
                for i in range(100)
            ],
        }
        result = _smart_truncate(tool_results)
        # Should be within budget (with some margin for formatting)
        from src.agents.specialists.discovery import _TOTAL_BUDGET
        assert len(result) <= _TOTAL_BUDGET + 500  # small margin for last entry

    def test_cross_tool_boost_same_entity(self):
        """Entity found by multiple tools gets boosted score."""
        tool_results = {
            "tool_a": [{"id": "1", "labels": ["App"], "name": "X", "score": 0.7}],
            "tool_b": [{"id": "1", "labels": ["App"], "name": "X", "score": 0.6}],
            "tool_c": [{"id": "1", "labels": ["App"], "name": "X", "score": 0.5}],
        }
        result = _smart_truncate(tool_results)
        # Node "1" found by 3 tools → boosted score = 0.7 + 0.1*(3-1) = 0.9
        assert "score=0.90" in result

    def test_boost_capped_at_1(self):
        """Boosted score should never exceed 1.0."""
        tool_results = {
            f"tool_{i}": [{"id": "1", "labels": ["App"], "name": "X", "score": 0.95}]
            for i in range(5)
        }
        result = _smart_truncate(tool_results)
        # 0.95 + 0.1*4 = 1.35 → capped at 1.0
        assert "score=1.00" in result

    def test_single_tool_no_boost(self):
        """Entity from only 1 tool gets no boost."""
        tool_results = {
            "tool_a": [{"id": "1", "labels": ["App"], "name": "X", "score": 0.7}],
        }
        result = _smart_truncate(tool_results)
        # 0.7 + 0.1*(1-1) = 0.7 — no boost
        assert "score=0.70" in result


# ── DiscoverySpecialist ───────────────────────────────────────────────────────


def _make_mock_db() -> AsyncMock:
    """Create a mock DB with get_schema returning a basic schema."""
    db = AsyncMock()
    db.get_schema = AsyncMock(return_value={
        "labels": ["Application", "Domain"],
        "label_properties": {
            "Application": ["name", "id", "description"],
            "Domain": ["name", "type"],
        },
        "relationship_types": ["BELONGS_TO"],
        "relationship_patterns": [],
    })
    return db


class TestDiscoverySpecialist:

    @pytest.mark.asyncio
    async def test_fulltext_index_caching(self):
        """Fulltext indexes should be fetched once and cached."""
        db = _make_mock_db()
        llm = AsyncMock()
        call_count = 0

        async def mock_get_indexes(db_arg):
            nonlocal call_count
            call_count += 1
            return [{"name": "idx1", "labelsOrTypes": ["App"], "properties": ["name"]}]

        tools = {"get_fulltext_indexes": mock_get_indexes}
        specialist = DiscoverySpecialist(db, llm, tools)

        # First call should fetch
        result1 = await specialist._get_fulltext_indexes()
        assert call_count == 1
        assert len(result1) == 1

        # Second call should use cache
        result2 = await specialist._get_fulltext_indexes()
        assert call_count == 1  # NOT incremented
        assert result2 == result1

    @pytest.mark.asyncio
    async def test_fulltext_index_missing_tool(self):
        """When get_fulltext_indexes tool is not available, return empty list."""
        db = _make_mock_db()
        llm = AsyncMock()
        specialist = DiscoverySpecialist(db, llm, {})

        result = await specialist._get_fulltext_indexes()
        assert result == []

    @pytest.mark.asyncio
    async def test_fulltext_index_error_returns_empty(self):
        """When get_fulltext_indexes raises, cache empty list and move on."""
        db = _make_mock_db()
        llm = AsyncMock()

        async def mock_fail(db_arg):
            raise RuntimeError("Connection failed")

        tools = {"get_fulltext_indexes": mock_fail}
        specialist = DiscoverySpecialist(db, llm, tools)

        result = await specialist._get_fulltext_indexes()
        assert result == []
        # Subsequent calls should return cached empty list
        assert specialist._fulltext_indexes == []

    @pytest.mark.asyncio
    async def test_parallel_execution_merges_results(self):
        """_execute_parallel should run all tools and merge results."""
        db = _make_mock_db()
        llm = AsyncMock()

        async def mock_search(db_arg, **kwargs):
            return [{"id": "1", "name": "Result1"}]

        async def mock_labels(db_arg, **kwargs):
            return ["Application", "Domain"]

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
        }
        specialist = DiscoverySpecialist(db, llm, tools)

        results = await specialist._execute_parallel([
            ("search_nodes_by_text", {"text": "test", "limit": 10}),
            ("get_all_labels", {}),
        ])

        assert "search_nodes_by_text" in results
        assert "get_all_labels" in results
        assert results["search_nodes_by_text"] == [{"id": "1", "name": "Result1"}]
        assert results["get_all_labels"] == ["Application", "Domain"]

    @pytest.mark.asyncio
    async def test_parallel_execution_handles_tool_failure(self):
        """Failed tools should return empty list, not crash the pipeline."""
        db = _make_mock_db()
        llm = AsyncMock()

        async def mock_fail(db_arg, **kwargs):
            raise RuntimeError("Tool broken")

        async def mock_labels(db_arg, **kwargs):
            return ["App"]

        tools = {
            "search_nodes_by_text": mock_fail,
            "get_all_labels": mock_labels,
        }
        specialist = DiscoverySpecialist(db, llm, tools)

        results = await specialist._execute_parallel([
            ("search_nodes_by_text", {"text": "test", "limit": 10}),
            ("get_all_labels", {}),
        ])

        # Failed tool returns empty list, others still work
        assert results["search_nodes_by_text"] == []
        assert results["get_all_labels"] == ["App"]

    @pytest.mark.asyncio
    async def test_run_end_to_end(self):
        """Full run should: extract terms → select tools → parallel exec → LLM extract."""
        db = _make_mock_db()
        llm = AsyncMock()

        # LLM returns entity extraction JSON
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "entities": [
                    {
                        "name": "CNAPP",
                        "label": "Application",
                        "node_id": "42",
                        "confidence": 0.95,
                        "properties": {"name": "CNAPP"},
                    }
                ]
            })
        ))

        async def mock_search(db_arg, **kwargs):
            return [{"id": "42", "labels": ["Application"], "name": "CNAPP"}]

        async def mock_labels(db_arg, **kwargs):
            return ["Application", "Domain"]

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
        }
        specialist = DiscoverySpecialist(db, llm, tools)
        state = AgentState(question="find the application named CNAPP")

        result = await specialist.run(state)

        assert result.success is True
        assert len(state.discoveries) == 1
        assert state.discoveries[0].entity_name == "CNAPP"
        assert state.discoveries[0].label == "Application"
        assert state.discoveries[0].node_id == "42"
        # LLM should have been called exactly once (entity extraction)
        assert llm.ainvoke.call_count == 1

    @pytest.mark.asyncio
    async def test_run_with_no_results(self):
        """When tools return nothing, run should still succeed with empty discoveries."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({"entities": []})
        ))

        async def mock_search(db_arg, **kwargs):
            return []

        async def mock_labels(db_arg, **kwargs):
            return []

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
        }
        specialist = DiscoverySpecialist(db, llm, tools)
        state = AgentState(question="find something nonexistent")

        result = await specialist.run(state)
        assert result.success is True
        assert state.discoveries == []

    @pytest.mark.asyncio
    async def test_run_deduplicates_entities(self):
        """Entities with the same node_id should be deduplicated."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "entities": [
                    {"name": "CNAPP", "label": "App", "node_id": "1", "confidence": 0.9, "properties": {}},
                    {"name": "CNAPP App", "label": "App", "node_id": "1", "confidence": 0.7, "properties": {}},
                ]
            })
        ))

        async def mock_search(db_arg, **kwargs):
            return [{"id": "1", "name": "CNAPP"}]

        async def mock_labels(db_arg, **kwargs):
            return []

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
        }
        specialist = DiscoverySpecialist(db, llm, tools)
        state = AgentState(question="CNAPP application")

        await specialist.run(state)
        # Should have only 1 entity (deduplicated by node_id)
        assert len(state.discoveries) == 1
        # Higher confidence one should be kept
        assert state.discoveries[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_run_handles_llm_parse_failure(self):
        """When LLM returns invalid JSON, run should succeed with empty discoveries."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="not valid json at all"))

        async def mock_search(db_arg, **kwargs):
            return [{"id": "1", "name": "Test"}]

        async def mock_labels(db_arg, **kwargs):
            return []

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
        }
        specialist = DiscoverySpecialist(db, llm, tools)
        state = AgentState(question="test query")

        result = await specialist.run(state)
        assert result.success is True
        assert state.discoveries == []

    @pytest.mark.asyncio
    async def test_deep_fuzzy_triggered_when_sparse_results(self):
        """Phase 2 deep fuzzy should trigger when Phase 1 has < 3 results."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({"entities": [
                {"name": "HK", "label": "Application", "node_id": "210",
                 "confidence": 0.9, "properties": {"name": "HK"}}
            ]})
        ))

        # Phase 1 text search returns only 1 result (< 3 threshold)
        async def mock_search(db_arg, **kwargs):
            return [{"id": "210", "labels": ["Application"], "props": {"name": "HK"}}]

        async def mock_labels(db_arg, **kwargs):
            return ["Application"]

        deep_fuzzy_called = False

        async def mock_fuzzy_label(db_arg, **kwargs):
            nonlocal deep_fuzzy_called
            deep_fuzzy_called = True
            return [{"id": "210", "labels": ["Application"],
                     "props": {"name": "HK"}, "minDist": 0}]

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
            "fuzzy_search_label_properties": mock_fuzzy_label,
        }
        specialist = DiscoverySpecialist(db, llm, tools)
        state = AgentState(question="find HK")

        result = await specialist.run(state)
        assert result.success is True
        assert deep_fuzzy_called, "Deep fuzzy should have been triggered"

    @pytest.mark.asyncio
    async def test_deep_fuzzy_skipped_when_enough_results(self):
        """Phase 2 deep fuzzy should NOT trigger when Phase 1 has >= 3 results."""
        db = _make_mock_db()
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content=json.dumps({"entities": []})
        ))

        async def mock_search(db_arg, **kwargs):
            # Return 5 results — more than the threshold
            return [{"id": str(i), "labels": ["App"], "name": f"E{i}"} for i in range(5)]

        async def mock_labels(db_arg, **kwargs):
            return ["Application"]

        deep_fuzzy_called = False

        async def mock_fuzzy(db_arg, **kwargs):
            nonlocal deep_fuzzy_called
            deep_fuzzy_called = True
            return []

        tools = {
            "search_nodes_by_text": mock_search,
            "get_all_labels": mock_labels,
            "fuzzy_search_label_properties": mock_fuzzy,
        }
        specialist = DiscoverySpecialist(db, llm, tools)
        state = AgentState(question="find applications")

        await specialist.run(state)
        assert not deep_fuzzy_called, "Deep fuzzy should NOT have been triggered"

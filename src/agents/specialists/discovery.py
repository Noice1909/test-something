"""Discovery Specialist — deterministic tool selection + parallel execution + single LLM extraction.

Replaces the original LLM-driven tool-selection loop (6 LLM calls) with:
1. Deterministic, priority-based tool selection (schema-aware)
2. Parallel execution of all tools via asyncio.gather
3. Two-phase strategy: fast targeted search → deep Levenshtein fuzzy fallback
4. Cross-tool confidence boosting for deduplication
5. Single LLM call to extract entities from combined results
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import DiscoveryResult, SpecialistResult
from src.agents.state import AgentState
from src.agents.utils import extract_text
from src.config import settings
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

# ── Configuration (from settings) ─────────────────────────────────────────────

_MAX_RESULTS_PER_TOOL: int = settings.discovery_max_results_per_tool
_TOP_N_FULL: int = settings.discovery_smart_truncation_top_n
_SUMMARY_N: int = settings.discovery_smart_truncation_summary_n
_TOTAL_BUDGET: int = settings.discovery_total_budget_chars
_FUZZY_THRESHOLD: float = settings.discovery_fuzzy_threshold
_CROSS_TOOL_BOOST: float = settings.discovery_cross_tool_boost
_DEEP_FUZZY_MIN: int = settings.discovery_deep_fuzzy_min_results
_LEVENSHTEIN_SHORT_MAX: int = settings.discovery_levenshtein_short_max

# ── Stop words for search term extraction ─────────────────────────────────────

_STOP_WORDS: set[str] = {
    "what", "which", "who", "where", "when", "how", "is", "are", "was",
    "were", "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "and", "or", "not", "do", "does", "did", "can", "could",
    "should", "would", "will", "shall", "may", "might", "must", "have",
    "has", "had", "be", "been", "being", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "me", "my", "all", "any", "many", "much", "some", "no", "show",
    "list", "find", "get", "tell", "about", "between", "most", "top",
    "best", "least", "currently", "also", "there", "here", "just",
}

# Common property names that typically hold identifying text
_NAME_KEYS: tuple[str, ...] = ("name", "title", "id", "code")

# ── Entity extraction prompt (single LLM call) ───────────────────────────────

_ENTITY_EXTRACTION_PROMPT = """\
You are analyzing graph database search results to find entities relevant to \
the user's question.

## User Question
{question}

## Search Results from Multiple Tools
{results}

## Instructions
Extract ALL entities from the results that are relevant to answering the question.
For each entity provide: name, label, node_id, confidence (0-1), and key properties.
Prioritize entities that directly match the user's question terms.
If no relevant entities are found, return an empty list.

Return a JSON object with:
- "entities": list of objects with "name", "label", "node_id", "confidence", "properties"

Return ONLY the JSON object:"""


# ── Helper functions ──────────────────────────────────────────────────────────


def _extract_search_terms(question: str) -> list[str]:
    """Extract meaningful search terms from the question."""
    # Preserve quoted strings as exact terms
    quoted = re.findall(r'"([^"]+)"', question)

    # Remove quoted parts and split remaining into words
    clean = re.sub(r'"[^"]*"', "", question)
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", clean)

    terms: list[str] = []
    for w in words:
        if w.lower() not in _STOP_WORDS and len(w) > 1:
            terms.append(w)

    # Quoted strings get highest priority (inserted at front)
    return quoted + terms


# ── Tool plan builders ────────────────────────────────────────────────────────


def _add_property_targeted_searches(
    plan: list[tuple[str, dict[str, Any]]],
    terms: list[str],
    available_tools: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Add property-targeted searches when a term matches a known label."""
    ci_tool = "search_nodes_by_property_case_insensitive"
    exact_tool = "search_nodes_by_property"
    tool_name = ci_tool if ci_tool in available_tools else (
        exact_tool if exact_tool in available_tools else None
    )
    if not tool_name:
        return

    labels = schema.get("labels", [])
    label_props = schema.get("label_properties", {})
    terms_lower = [t.lower() for t in terms]

    # Find terms that match a label name
    matched_labels: list[str] = []
    for label in labels:
        if label.lower() in terms_lower:
            matched_labels.append(label)

    # For each matched label, search its name/title property with remaining terms
    for label in matched_labels[:2]:
        props = label_props.get(label, [])
        target_key = next((k for k in _NAME_KEYS if k in props), None)
        if not target_key:
            continue
        value_terms = [t for t in terms if t.lower() != label.lower()]
        if value_terms:
            plan.append((tool_name, {
                "label": label, "key": target_key, "value": value_terms[0],
                "limit": _MAX_RESULTS_PER_TOOL,
            }))


def _add_fulltext_searches(
    plan: list[tuple[str, dict[str, Any]]],
    terms: list[str],
    fulltext_indexes: list[dict],
    available_tools: dict[str, Any],
) -> None:
    """Add fulltext index searches, including enhanced fuzzy variants."""
    query_str = " ".join(terms[:3])
    if not query_str:
        return

    std_tool = "search_nodes_using_fulltext_index"
    enh_tool = "search_nodes_using_fulltext_enhanced"

    for idx in fulltext_indexes[:3]:
        idx_name = idx.get("name", "")
        if not idx_name:
            continue

        # Standard fulltext search
        if std_tool in available_tools:
            plan.append((std_tool, {
                "index_name": idx_name, "query": query_str,
                "limit": _MAX_RESULTS_PER_TOOL,
            }))

        # Enhanced fuzzy variant for terms >= 4 chars (catches typos)
        if enh_tool in available_tools and any(len(t) >= 4 for t in terms):
            plan.append((enh_tool, {
                "index_name": idx_name, "query": query_str,
                "fuzzy": True, "limit": _MAX_RESULTS_PER_TOOL,
            }))


def _build_tool_plan(
    terms: list[str],
    available_tools: dict[str, Any],
    fulltext_indexes: list[dict],
    schema: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Deterministically select which tools to run and with what arguments.

    Priority order:
    1. Property-targeted search (fastest, most accurate)
    2. Fulltext index search + enhanced fuzzy (fast, scored)
    3. Case-insensitive broad text search (fallback)
    4. get_all_labels (needed for fuzzy phase)
    """
    plan: list[tuple[str, dict[str, Any]]] = []
    search_text = " ".join(terms[:3])

    # 1. Property-targeted search (when schema available and terms match labels)
    if schema and search_text:
        _add_property_targeted_searches(plan, terms, available_tools, schema)

    # 2. Fulltext index searches (standard + enhanced fuzzy)
    if search_text:
        _add_fulltext_searches(plan, terms, fulltext_indexes, available_tools)

    # 3. Broad text search (case-insensitive, embedding-excluded)
    if "search_nodes_by_text" in available_tools and search_text:
        plan.append(("search_nodes_by_text", {
            "text": search_text, "limit": _MAX_RESULTS_PER_TOOL,
        }))

    # 4. Always: get all labels (lightweight, needed for fuzzy phase)
    if "get_all_labels" in available_tools:
        plan.append(("get_all_labels", {}))

    return plan


def _build_fuzzy_plan(
    terms: list[str],
    labels: list[str],
    available_tools: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build fuzzy-match tool calls based on discovered labels."""
    if "apoc_text_fuzzyMatch" not in available_tools:
        return []

    # Only fuzzy-match terms that look like entity names (capitalized or long)
    entity_terms = [t for t in terms if (t[0:1].isupper() or len(t) > 4)][:2]
    if not entity_terms:
        return []

    # Find labels whose names partially overlap with search terms
    relevant_labels: list[str] = []
    for label in labels:
        label_lower = label.lower() if isinstance(label, str) else ""
        for term in terms:
            if term.lower() in label_lower or label_lower in term.lower():
                relevant_labels.append(label)
                break

    # If no label matched terms, try first 2 labels
    if not relevant_labels and labels:
        relevant_labels = [lb for lb in labels[:2] if isinstance(lb, str)]

    plan: list[tuple[str, dict[str, Any]]] = []
    for label in relevant_labels[:2]:
        for term in entity_terms[:1]:
            plan.append(("apoc_text_fuzzyMatch", {
                "label": label,
                "prop": "name",
                "search": term,
                "threshold": _FUZZY_THRESHOLD,
                "limit": 5,
            }))

    return plan


def _levenshtein_threshold(term: str) -> int:
    """Compute adaptive Levenshtein distance threshold based on term length.

    Short terms (1-3 chars): threshold = len(term) — e.g. "HK" → 2
    Medium terms (4-7 chars): threshold = 2
    Long terms (8+ chars): threshold = 3
    """
    if len(term) <= _LEVENSHTEIN_SHORT_MAX:
        return len(term)
    if len(term) <= 7:
        return 2
    return 3


def _build_deep_fuzzy_plan(
    terms: list[str],
    available_tools: dict[str, Any],
    labels: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Build Levenshtein-based deep fuzzy search (Phase 2).

    Only called when Phase 1 produced sparse results.
    Uses label-scoped variant when labels are known (faster).
    """
    plan: list[tuple[str, dict[str, Any]]] = []

    for term in terms[:2]:
        threshold = _levenshtein_threshold(term)

        if labels and "fuzzy_search_label_properties" in available_tools:
            for label in labels[:3]:
                if not isinstance(label, str):
                    continue
                plan.append(("fuzzy_search_label_properties", {
                    "label": label, "text": term,
                    "threshold": threshold, "limit": 10,
                }))
        elif "fuzzy_search_all_properties" in available_tools:
            plan.append(("fuzzy_search_all_properties", {
                "text": term, "threshold": threshold, "limit": 10,
            }))

    return plan


# ── Scoring and ranking ──────────────────────────────────────────────────────


def _score_result(result: dict[str, Any], tool_name: str) -> float:
    """Assign a relevance score to a result based on its source."""
    # Fulltext results have a Neo4j-provided score
    if "score" in result:
        return min(1.0, float(result["score"]))
    # Fuzzy match results have a similarity score
    if "sim" in result:
        return float(result["sim"])
    # Levenshtein distance results — convert distance to 0-1 score
    if "minDist" in result:
        dist = int(result["minDist"])
        return max(0.1, 1.0 - (dist * 0.2))
    # Property-targeted searches are high confidence
    if tool_name in ("search_nodes_by_property", "search_nodes_by_property_case_insensitive"):
        return 0.85
    # Prefix matches
    if tool_name == "search_nodes_by_text_prefix":
        return 0.7
    # Generic text search
    if tool_name.startswith("search_nodes") or tool_name.startswith("fuzzy_search"):
        return 0.5
    return 0.3


def _smart_truncate(
    tool_results: dict[str, list[Any]],
) -> str:
    """Deduplicate with cross-tool boosting, score, rank, and format within budget.

    When the same entity (by node_id) is found by multiple tools, its score
    is boosted: final_score = max_score + boost * (num_tools - 1), capped at 1.0.
    Top-N results keep full properties; the rest are summarised.
    """
    # Step 1: Collect all results, tracking per-node scores from each tool
    node_entries: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}

    for tool_name, results in tool_results.items():
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id", item.get("node_id", "")))
            if not node_id:
                node_id = f"_anon_{id(item)}"

            score = _score_result(item, tool_name)
            if node_id not in node_entries:
                node_entries[node_id] = []
            node_entries[node_id].append((score, tool_name, item))

    # Step 2: Compute boosted score per unique node
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for entries in node_entries.values():
        max_score = max(s for s, _, _ in entries)
        tool_count = len({tn for _, tn, _ in entries})
        boosted = min(1.0, max_score + _CROSS_TOOL_BOOST * (tool_count - 1))
        # Keep the item from the highest-scoring tool
        best_entry = max(entries, key=lambda x: x[0])
        scored.append((boosted, best_entry[1], best_entry[2]))

    # Step 3: Sort by boosted score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Step 4: Format within budget
    sections: list[str] = []
    total_chars = 0

    # Top-N: full properties
    for score, tool_name, item in scored[:_TOP_N_FULL]:
        entry = json.dumps(item, indent=1, default=str)
        line = f"[{tool_name}, score={score:.2f}] {entry}"
        if total_chars + len(line) > _TOTAL_BUDGET:
            break
        sections.append(line)
        total_chars += len(line) + 1

    # Next-N: summary only (id + labels + name)
    for score, tool_name, item in scored[_TOP_N_FULL:_TOP_N_FULL + _SUMMARY_N]:
        labels = item.get("labels", [])
        props = item.get("props", item.get("properties", {}))
        name = props.get("name", props.get("title", "?"))
        node_id = item.get("id", "?")
        line = f"[{tool_name}, score={score:.2f}] id={node_id}, labels={labels}, name={name}"
        if total_chars + len(line) > _TOTAL_BUDGET:
            break
        sections.append(line)
        total_chars += len(line) + 1

    if not sections:
        return "No results found from any search tool."

    return "\n".join(sections)


# ── Specialist ────────────────────────────────────────────────────────────────


class DiscoverySpecialist:
    """Deterministic tool selection + parallel execution + single LLM entity extraction."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools
        self._fulltext_indexes: list[dict] | None = None  # lazy-cached

    async def _get_fulltext_indexes(self) -> list[dict[str, Any]]:
        """Fetch and cache fulltext indexes (they don't change at runtime)."""
        if self._fulltext_indexes is not None:
            return self._fulltext_indexes
        fn = self._tools.get("get_fulltext_indexes")
        if fn:
            try:
                raw = await fn(self._db)
                self._fulltext_indexes = list(raw) if raw else []
            except Exception as exc:
                logger.debug("Could not fetch fulltext indexes: %s", exc)
                self._fulltext_indexes = []
        else:
            self._fulltext_indexes = []
        return self._fulltext_indexes

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            # Phase 1: Extract search terms deterministically
            terms = _extract_search_terms(state.question)
            if not terms:
                terms = [state.question[:50]]

            # Fetch fulltext indexes (cached) and schema (cached)
            ft_indexes = await self._get_fulltext_indexes()
            schema = await self._db.get_schema()

            # Phase 2: Build tool plan (schema-aware, priority-based)
            tool_plan = _build_tool_plan(terms, self._tools, ft_indexes, schema=schema)

            # Phase 3: Execute all Phase 1 tools in parallel
            phase1_results = await self._execute_parallel(tool_plan)

            # Phase 4: Optional APOC fuzzy-match phase (needs labels from Phase 1)
            labels_result = phase1_results.get("get_all_labels", [])
            if isinstance(labels_result, list) and labels_result:
                fuzzy_plan = _build_fuzzy_plan(terms, labels_result, self._tools)
                if fuzzy_plan:
                    fuzzy_results = await self._execute_parallel(fuzzy_plan)
                    phase1_results.update(fuzzy_results)

            # Phase 5: Deep Levenshtein fuzzy (only if Phase 1 results are sparse)
            non_label_results = {
                k: v for k, v in phase1_results.items() if k != "get_all_labels"
            }
            total_hits = sum(
                len(v) for v in non_label_results.values() if isinstance(v, list)
            )
            if total_hits < _DEEP_FUZZY_MIN:
                deep_plan = _build_deep_fuzzy_plan(
                    terms, self._tools,
                    labels=labels_result if isinstance(labels_result, list) else None,
                )
                if deep_plan:
                    deep_results = await self._execute_parallel(deep_plan)
                    phase1_results.update(deep_results)
                    logger.info(
                        "Deep fuzzy triggered: Phase 1 had %d hits, ran %d deep tools",
                        total_hits, len(deep_plan),
                    )

            # Phase 6: Smart truncation + cross-tool boosting + formatting
            formatted = _smart_truncate(phase1_results)

            # Phase 7: Single LLM call to extract entities
            entities = await self._extract_entities_batch(formatted, state.question)

            state.discoveries = entities
            total_tools = len(tool_plan) + len(
                _build_fuzzy_plan(terms, labels_result if isinstance(labels_result, list) else [], self._tools)
            )
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "discovery", success=True, duration_ms=dur,
                detail=f"{len(entities)} entities via {total_tools} parallel tools + 1 LLM call",
            )
            return SpecialistResult(success=True, data=entities, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "discovery", success=False, duration_ms=dur, detail=str(exc),
            )
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    # ── Parallel execution ────────────────────────────────────────────────

    async def _execute_parallel(
        self, tool_plan: list[tuple[str, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Execute all tool calls in parallel, return {tool_name: result}."""

        async def _run_one(name: str, kwargs: dict[str, Any]) -> tuple[str, Any]:
            try:
                fn = self._tools[name]
                result = await fn(self._db, **kwargs)
                return name, result
            except Exception as exc:
                logger.debug("Discovery tool %s failed: %s", name, exc)
                return name, []

        tasks = [_run_one(name, kwargs) for name, kwargs in tool_plan]
        results = await asyncio.gather(*tasks)

        # Merge results (a tool may appear multiple times, e.g. multiple fulltext indexes)
        merged: dict[str, Any] = {}
        for name, result in results:
            if name in merged:
                if isinstance(merged[name], list) and isinstance(result, list):
                    merged[name].extend(result)
                else:
                    merged[f"{name}_{len(merged)}"] = result
            else:
                merged[name] = result

        return merged

    # ── Single LLM call for entity extraction ─────────────────────────────

    async def _extract_entities_batch(
        self, formatted_results: str, question: str,
    ) -> list[DiscoveryResult]:
        """Single LLM call to extract all entities from combined results."""
        prompt = _ENTITY_EXTRACTION_PROMPT.format(
            question=question,
            results=formatted_results,
        )

        response = await self._llm.ainvoke(prompt)
        text = extract_text(response)

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return []

        entities: list[DiscoveryResult] = []
        for ent in data.get("entities", []):
            entities.append(DiscoveryResult(
                entity_name=ent.get("name", ""),
                label=ent.get("label", "Unknown"),
                node_id=str(ent.get("node_id", "")),
                confidence=float(ent.get("confidence", 0.5)),
                match_type="parallel_discovery",
                properties=ent.get("properties", {}),
            ))

        # Deduplicate by node_id, sorted by confidence
        seen: set[str] = set()
        unique: list[DiscoveryResult] = []
        for d in sorted(entities, key=lambda x: x.confidence, reverse=True):
            key = d.node_id or f"{d.label}:{d.entity_name}"
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique

"""Discovery Specialist — deterministic tool selection + parallel execution + single LLM extraction.

Replaces the original LLM-driven tool-selection loop (6 LLM calls) with:
1. Deterministic tool selection based on question keywords
2. Parallel execution of all tools via asyncio.gather
3. Single LLM call to extract entities from combined results

Expected savings: ~25-30s per query (from ~41s to ~12-18s).
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


def _build_tool_plan(
    terms: list[str],
    available_tools: dict[str, Any],
    fulltext_indexes: list[dict],
) -> list[tuple[str, dict[str, Any]]]:
    """Deterministically select which tools to run and with what arguments."""
    plan: list[tuple[str, dict[str, Any]]] = []

    # 1. Always: broad text search with top terms
    search_text = " ".join(terms[:3])
    if "search_nodes_by_text" in available_tools and search_text:
        plan.append(("search_nodes_by_text", {
            "text": search_text, "limit": _MAX_RESULTS_PER_TOOL,
        }))

    # 2. Fulltext index searches (one per index, max 3)
    if "search_nodes_using_fulltext_index" in available_tools:
        query_str = " ".join(terms[:3])
        for idx in fulltext_indexes[:3]:
            idx_name = idx.get("name", "")
            if idx_name and query_str:
                plan.append(("search_nodes_using_fulltext_index", {
                    "index_name": idx_name,
                    "query": query_str,
                    "limit": _MAX_RESULTS_PER_TOOL,
                }))

    # 3. Always: get all labels (lightweight, needed for fuzzy phase)
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


def _score_result(result: dict[str, Any], tool_name: str) -> float:
    """Assign a relevance score to a result based on its source."""
    # Fulltext results have a Neo4j-provided score
    if "score" in result:
        return float(result["score"])
    # Fuzzy match results have a similarity score
    if "sim" in result:
        return float(result["sim"])
    # Text search results get a default score
    if tool_name.startswith("search_nodes"):
        return 0.5
    return 0.3


def _smart_truncate(
    tool_results: dict[str, list[Any]],
) -> str:
    """Deduplicate, score, rank, and format results within budget.

    Top-N results keep full properties; the rest are summarised.
    """
    # Flatten all results with scores
    scored: list[tuple[float, str, dict[str, Any]]] = []
    seen_ids: set[str] = set()

    for tool_name, results in tool_results.items():
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            # Deduplicate by node id
            node_id = str(item.get("id", item.get("node_id", "")))
            if node_id and node_id in seen_ids:
                continue
            if node_id:
                seen_ids.add(node_id)

            score = _score_result(item, tool_name)
            scored.append((score, tool_name, item))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    sections: list[str] = []
    total_chars = 0

    # Top-N: full properties
    for i, (score, tool_name, item) in enumerate(scored[:_TOP_N_FULL]):
        entry = json.dumps(item, indent=1, default=str)
        line = f"[{tool_name}, score={score:.2f}] {entry}"
        if total_chars + len(line) > _TOTAL_BUDGET:
            break
        sections.append(line)
        total_chars += len(line) + 1

    # Next-N: summary only (id + labels + name)
    for i, (score, tool_name, item) in enumerate(scored[_TOP_N_FULL:_TOP_N_FULL + _SUMMARY_N]):
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

            # Fetch fulltext indexes (cached after first call)
            ft_indexes = await self._get_fulltext_indexes()

            # Phase 2: Build tool plan deterministically
            tool_plan = _build_tool_plan(terms, self._tools, ft_indexes)

            # Phase 3: Execute all Phase 1 tools in parallel
            phase1_results = await self._execute_parallel(tool_plan)

            # Phase 4: Optional fuzzy-match phase (needs labels from Phase 1)
            labels_result = phase1_results.get("get_all_labels", [])
            if isinstance(labels_result, list) and labels_result:
                fuzzy_plan = _build_fuzzy_plan(terms, labels_result, self._tools)
                if fuzzy_plan:
                    fuzzy_results = await self._execute_parallel(fuzzy_plan)
                    phase1_results.update(fuzzy_results)

            # Phase 5: Smart truncation + formatting
            formatted = _smart_truncate(phase1_results)

            # Phase 6: Single LLM call to extract entities
            entities = await self._extract_entities_batch(formatted, state.question)

            state.discoveries = entities
            tool_count = len(tool_plan) + len(
                _build_fuzzy_plan(terms, labels_result if isinstance(labels_result, list) else [], self._tools)
            )
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "discovery", success=True, duration_ms=dur,
                detail=f"{len(entities)} entities via {tool_count} parallel tools + 1 LLM call",
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

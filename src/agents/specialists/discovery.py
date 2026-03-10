"""Discovery Specialist — actively searches the database for entities."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import DiscoveryResult, SpecialistResult
from src.agents.state import AgentState
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_TERM_EXTRACTION_PROMPT = """\
Extract the key search terms from this question that should be looked up in a \
graph database. Return a JSON array of strings. Only include meaningful terms \
(entity names, acronyms, specific nouns). Omit generic words like "show", "find", \
"list".

Question: {question}

Return ONLY a JSON array, e.g. ["CNAPP", "cloud security"]:"""

_DISCOVERY_STRATEGIES = ["exact_match", "fuzzy_match", "label_match"]


class DiscoverySpecialist:
    """Finds entities in the database through multi-strategy search."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            # 1) Extract search terms via LLM
            terms = await self._extract_terms(state.question)
            if not terms:
                return SpecialistResult(success=False, error="No search terms extracted")

            # 2) Search with multiple strategies
            all_discoveries: list[DiscoveryResult] = []
            for term in terms:
                results = await self._search_term(term)
                all_discoveries.extend(results)

            # 3) De-duplicate by node_id
            seen: set[str] = set()
            unique: list[DiscoveryResult] = []
            for d in sorted(all_discoveries, key=lambda x: x.confidence, reverse=True):
                key = d.node_id or f"{d.label}:{d.entity_name}"
                if key not in seen:
                    seen.add(key)
                    unique.append(d)

            state.discoveries = unique
            dur = (time.time() - t0) * 1000
            state.log_specialist("discovery", success=True, duration_ms=dur,
                                 detail=f"{len(unique)} entities found for terms {terms}")
            return SpecialistResult(success=True, data=unique, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist("discovery", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    # ── internals ──

    async def _extract_terms(self, question: str) -> list[str]:
        prompt = _TERM_EXTRACTION_PROMPT.format(question=question)
        response = await self._llm.ainvoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        try:
            # Try to extract JSON array from the response
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            # Fallback: split on commas
            return [t.strip().strip('"\'') for t in text.split(",") if t.strip()]

    async def _search_term(self, term: str) -> list[DiscoveryResult]:
        discoveries: list[DiscoveryResult] = []

        # Strategy 1: exact text search
        if "search_nodes_by_text" in self._tools:
            try:
                rows = await self._tools["search_nodes_by_text"](self._db, text=term, limit=10)
                for row in rows:
                    discoveries.append(DiscoveryResult(
                        entity_name=term,
                        label=row.get("labels", ["Unknown"])[0] if row.get("labels") else "Unknown",
                        node_id=str(row.get("id", "")),
                        confidence=0.9,
                        match_type="exact_match",
                        properties=row.get("props", {}),
                    ))
            except Exception as exc:
                logger.debug("exact search failed for '%s': %s", term, exc)

        # Strategy 2: fulltext index search (fuzzy)
        if "search_nodes_using_fulltext_index" in self._tools:
            try:
                # Try default index names
                for idx_name in ["node_fulltext", "nodeFulltext", "fulltext"]:
                    try:
                        rows = await self._tools["search_nodes_using_fulltext_index"](
                            self._db, index_name=idx_name, query=f"{term}~", limit=10,
                        )
                        for row in rows:
                            discoveries.append(DiscoveryResult(
                                entity_name=term,
                                label=row.get("labels", ["Unknown"])[0] if row.get("labels") else "Unknown",
                                node_id=str(row.get("id", "")),
                                confidence=row.get("score", 0.7),
                                match_type="fuzzy_match",
                                properties=row.get("props", {}),
                            ))
                        if rows:
                            break
                    except Exception:
                        continue
            except Exception as exc:
                logger.debug("fulltext search failed for '%s': %s", term, exc)

        # Strategy 3: label match — check if term matches a label name
        if "get_all_labels" in self._tools:
            try:
                labels = await self._tools["get_all_labels"](self._db)
                term_upper = term.upper()
                for label in labels:
                    if term_upper in label.upper() or label.upper() in term_upper:
                        # Get sample nodes for this label
                        if "get_nodes_by_label" in self._tools:
                            nodes = await self._tools["get_nodes_by_label"](
                                self._db, label=label, limit=5,
                            )
                            for node in nodes:
                                discoveries.append(DiscoveryResult(
                                    entity_name=label,
                                    label=label,
                                    node_id=str(node.get("id", "")),
                                    confidence=0.8,
                                    match_type="label_match",
                                    properties=node.get("props", {}),
                                ))
            except Exception as exc:
                logger.debug("label match failed for '%s': %s", term, exc)

        return discoveries

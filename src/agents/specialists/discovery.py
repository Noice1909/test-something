"""Discovery Specialist — LLM-driven tool selection for entity search.

Instead of hardcoded search strategies, the LLM autonomously picks
from the available MCP tools to discover entities in the graph database.
"""

from __future__ import annotations

import inspect
import json
import logging
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agents.base import DiscoveryResult, SpecialistResult
from src.agents.state import AgentState
from src.agents.utils import extract_text
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

# ── Tool catalog — only discovery-relevant tools are shown to the LLM ────────

_DISCOVERY_TOOL_NAMES: list[str] = [
    # Text search
    "search_nodes_by_text",
    "search_relationships_by_text",
    "search_nodes_by_property",
    "search_nodes_by_multiple_properties",
    "search_nodes_using_fulltext_index",
    # Labels & schema
    "get_all_labels",
    "get_nodes_by_label",
    "get_all_relationship_types",
    # APOC text (fuzzy)
    "apoc_text_fuzzyMatch",
    "apoc_text_similarity",
    "apoc_text_levenshteinDistance",
    "apoc_text_distance",
    # Sampling
    "sample_random_nodes",
    # Fulltext
    "get_fulltext_indexes",
]

_MAX_TOOL_CALLS = 3

# ── Prompts ──────────────────────────────────────────────────────────────────

_TOOL_SELECTION_PROMPT = """\
You are a graph database entity discovery agent. Your job is to find relevant \
entities in the database for the user's question by calling available tools.

## User Question
{question}

## Available Tools
{tool_catalog}

## Previous Tool Calls & Results (this session)
{history}

## Instructions — follow this order of priority:
1. FIRST try search_nodes_by_text with the key terms from the question. \
This searches all property values across all nodes.
2. If text search returns nothing, try get_all_labels to see what labels \
exist. Then look for labels that partially match the user's terms.
3. If the user's term might be misspelled or abbreviated, use fuzzy tools:
   - apoc_text_fuzzyMatch to find similar property values
   - search_nodes_by_text with shorter substrings (e.g. "CNAP" → try "CNA")
4. Once you find matching labels, use get_nodes_by_label to get sample nodes.
5. Do NOT assume label names from the question — always verify they exist first.
6. If previous calls returned nothing useful, try DIFFERENT search terms \
or DIFFERENT tools. Never repeat the same call.

Return a JSON object with:
- "tool_name": name of the tool to call
- "arguments": dict of arguments (excluding 'db')
- "reasoning": why you chose this tool

If you have found enough entities (or no more tools would help), return:
- "done": true
- "reasoning": why you're stopping

Return ONLY the JSON object:"""

_RESULT_ANALYSIS_PROMPT = """\
You called tool "{tool_name}" and got these results:

{results}

Based on the user's question "{question}", extract any discovered entities.

Return a JSON object with:
- "entities": list of objects with "name", "label", "node_id", "confidence" \
(0-1), "properties" (dict)
- If no useful entities found, return empty list

Return ONLY the JSON object:"""


# ── Specialist ───────────────────────────────────────────────────────────────


class DiscoverySpecialist:
    """Finds entities using LLM-driven tool selection from the MCP registry."""

    def __init__(
        self, db: AbstractDatabase, llm: BaseChatModel, tools: dict[str, Any]
    ) -> None:
        self._db = db
        self._llm = llm
        self._tools = tools
        self._tool_catalog = self._build_tool_catalog()

    def _build_tool_catalog(self) -> str:
        """Build a human-readable catalog of available discovery tools."""
        lines: list[str] = []
        for name in _DISCOVERY_TOOL_NAMES:
            fn = self._tools.get(name)
            if fn is None:
                continue

            sig = inspect.signature(fn)
            params = []
            for p in sig.parameters.values():
                if p.name in ("db", "_", "kwargs"):
                    continue
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    continue
                annotation = ""
                if p.annotation != inspect.Parameter.empty:
                    annotation = (
                        f": {p.annotation.__name__}"
                        if hasattr(p.annotation, "__name__")
                        else f": {p.annotation}"
                    )
                default = ""
                if p.default != inspect.Parameter.empty:
                    default = f" = {p.default!r}"
                params.append(f"{p.name}{annotation}{default}")

            doc = (fn.__doc__ or "No description").strip().split("\n")[0]
            params_str = ", ".join(params)
            lines.append(f"- {name}({params_str})\n    {doc}")

        return "\n".join(lines)

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        try:
            all_discoveries: list[DiscoveryResult] = []
            call_history: list[str] = []

            for step in range(1, _MAX_TOOL_CALLS + 1):
                # Ask LLM which tool to call
                tool_decision = await self._ask_llm_for_tool(
                    state.question, call_history,
                )

                if tool_decision.get("done"):
                    logger.info(
                        "[%s] Discovery done after %d tool calls: %s",
                        state.trace_id, step - 1, tool_decision.get("reasoning", ""),
                    )
                    break

                tool_name = tool_decision.get("tool_name", "")
                arguments = tool_decision.get("arguments", {})
                reasoning = tool_decision.get("reasoning", "")

                if tool_name not in self._tools:
                    call_history.append(
                        f"Step {step}: Tried {tool_name} — tool not available"
                    )
                    continue

                # Execute the tool
                logger.info(
                    "[%s] Discovery step %d: %s(%s)",
                    state.trace_id, step, tool_name, arguments,
                )
                try:
                    results = await self._tools[tool_name](self._db, **arguments)
                    results_str = json.dumps(
                        results[:15] if isinstance(results, list) else results,
                        indent=2, default=str,
                    )
                    # Truncate very long results
                    if len(results_str) > 2000:
                        results_str = results_str[:2000] + "\n... (truncated)"

                    call_history.append(
                        f"Step {step}: Called {tool_name}({arguments})\n"
                        f"  Reasoning: {reasoning}\n"
                        f"  Result: {results_str}"
                    )

                    # Extract entities from results
                    entities = await self._extract_entities(
                        tool_name, results_str, state.question,
                    )
                    all_discoveries.extend(entities)

                except Exception as exc:
                    call_history.append(
                        f"Step {step}: Called {tool_name}({arguments}) — "
                        f"ERROR: {exc}"
                    )
                    logger.debug("Tool %s failed: %s", tool_name, exc)

            # De-duplicate by node_id
            seen: set[str] = set()
            unique: list[DiscoveryResult] = []
            for d in sorted(all_discoveries, key=lambda x: x.confidence, reverse=True):
                key = d.node_id or f"{d.label}:{d.entity_name}"
                if key not in seen:
                    seen.add(key)
                    unique.append(d)

            state.discoveries = unique
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "discovery", success=True, duration_ms=dur,
                detail=f"{len(unique)} entities found via {len(call_history)} tool calls",
            )
            return SpecialistResult(success=True, data=unique, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            state.log_specialist(
                "discovery", success=False, duration_ms=dur, detail=str(exc),
            )
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    # ── LLM interactions ─────────────────────────────────────────────────

    async def _ask_llm_for_tool(
        self, question: str, history: list[str],
    ) -> dict[str, Any]:
        """Ask the LLM which tool to call next."""
        history_text = "\n\n".join(history) if history else "None (first call)"

        prompt = _TOOL_SELECTION_PROMPT.format(
            question=question,
            tool_catalog=self._tool_catalog,
            history=history_text,
        )

        response = await self._llm.ainvoke(prompt)
        text = extract_text(response)

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"done": True, "reasoning": "Could not parse LLM response"}

    async def _extract_entities(
        self, tool_name: str, results_str: str, question: str,
    ) -> list[DiscoveryResult]:
        """Ask the LLM to extract entities from tool results."""
        prompt = _RESULT_ANALYSIS_PROMPT.format(
            tool_name=tool_name,
            results=results_str,
            question=question,
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
                match_type=f"tool:{tool_name}",
                properties=ent.get("properties", {}),
            ))

        return entities

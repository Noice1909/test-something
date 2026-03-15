"""Supervisor orchestrator — builds system prompt, routes queries, manages sessions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from config import Settings
from core.context import ConversationContext, SessionStore
from core.loop import EventCallback, run_agent_loop
from discovery.registry import CapabilityRegistry
from discovery.skill_loader import process_dynamic_context, substitute_arguments
from tools.manager import ToolManager

logger = logging.getLogger(__name__)

_BASE_SYSTEM_PROMPT = """\
You are an intelligent AI assistant for querying and analyzing a Neo4j knowledge graph.

You have access to built-in Neo4j tools for running Cypher queries, searching nodes,
exploring the schema, and more.  You also have two special orchestration tools:

- **invoke_skill(name, arguments)** — Activate a registered skill when the user's
  request matches its description.
- **delegate_to_agent(name, prompt)** — Delegate complex, multi-step work to a
  specialized sub-agent.

## How to decide:
1. If the question is simple and you can answer directly with a tool → do it yourself.
2. If the request clearly matches a skill's description → use invoke_skill.
3. If the task is complex and matches an agent's description → use delegate_to_agent.
4. When uncertain, prefer handling it yourself with the available tools.

IMPORTANT:
- All Cypher queries MUST be read-only (no CREATE, MERGE, DELETE, SET, REMOVE).
- Always check relationship directions before writing queries — wrong direction = 0 results.
"""


class Orchestrator:
    """Main supervisor that wires together the TAOR loop, tools, and discovery."""

    def __init__(
        self,
        llm_factory: Callable[..., BaseChatModel],
        registry: CapabilityRegistry,
        tool_manager: ToolManager,
        config: Settings,
    ) -> None:
        self._llm_factory = llm_factory
        self._registry = registry
        self._tool_manager = tool_manager
        self._config = config
        self._sessions = SessionStore()

    # ── System prompt ────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt with dynamic capability descriptions."""
        parts: list[str] = [_BASE_SYSTEM_PROMPT]

        # Inject skill/agent descriptions (~100 tokens each)
        descriptions = self._registry.get_descriptions_for_prompt()
        if descriptions:
            parts.append(descriptions)

        # Inject project-level instructions if present
        project_md = Path("CLAUDE.md")
        if project_md.exists():
            parts.append(
                f"## Project Instructions\n{project_md.read_text(encoding='utf-8')}"
            )

        return "\n\n".join(parts)

    # ── Query handling ───────────────────────────────────────────────────

    async def handle_query(
        self,
        user_input: str,
        session_id: str | None = None,
        on_event: EventCallback = None,
    ) -> dict[str, Any]:
        """Process a user query and return the response.

        Returns a dict with ``response``, ``session_id``, and ``turns_used``.
        """
        # ── Slash command shortcut ───────────────────────────────────────
        resolved = self._registry.resolve_slash_command(user_input)
        if resolved:
            skill, args = resolved
            response = await self._execute_slash_skill(skill, args, on_event)
            return {
                "response": response,
                "session_id": session_id or "ephemeral",
                "turns_used": 1,
                "tools_called": ["invoke_skill"],
            }

        # ── Normal TAOR flow ─────────────────────────────────────────────
        context = self._sessions.get_or_create(session_id)
        context.add_user(user_input)

        # Compact if nearing context limit
        llm = self._llm_factory()
        await context.compact(llm, self._config.max_context_tokens, self._config.compact_threshold)

        # Run the agentic loop
        system_prompt = self._build_system_prompt()
        tools = self._tool_manager.get_all_tools()

        turn_counter = {"count": 0}
        tools_called: list[str] = []

        async def _track_event(event: dict) -> None:
            if event.get("type") == "tool_result":
                tools_called.append(event.get("tool", ""))
            turn_counter["count"] = event.get("turn", 0) + 1
            if on_event:
                await on_event(event)

        result = await run_agent_loop(
            llm=llm,
            system_prompt=system_prompt,
            messages=context.messages,
            tools=tools,
            max_turns=self._config.max_turns,
            on_event=_track_event,
        )

        return {
            "response": result,
            "session_id": context.session_id,
            "turns_used": turn_counter["count"],
            "tools_called": tools_called,
        }

    async def _execute_slash_skill(
        self,
        skill: Any,
        args: str,
        on_event: EventCallback,
    ) -> str:
        """Execute a /slash-command skill directly (bypass LLM routing)."""
        from agents.spawner import SubAgentSpawner

        instructions = substitute_arguments(skill.instructions, args)
        instructions = await process_dynamic_context(instructions)

        if skill.context == "fork":
            spawner = SubAgentSpawner(
                self._llm_factory, self._tool_manager, self._registry
            )
            return await spawner.spawn_from_skill(skill, instructions)
        else:
            # Run a single-turn TAOR with the skill instructions as context
            llm = self._llm_factory(model=skill.model)
            tools = self._tool_manager.filter_tools(allowed=skill.allowed_tools)
            messages = [HumanMessage(content=instructions)]
            return await run_agent_loop(
                llm=llm,
                system_prompt=instructions,
                messages=messages,
                tools=tools,
                max_turns=30,
                on_event=on_event,
            )

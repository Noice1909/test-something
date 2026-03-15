"""Supervisor orchestrator — builds system prompt, routes queries, manages sessions.

Includes:
  - Prompt caching (in-memory hash + Anthropic cache_control support)
  - Hook manager integration (PreToolUse / PostToolUse / Stop)
  - Token-by-token streaming via async generator
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, AsyncGenerator, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from config import Settings
from core.context import ConversationContext, SessionStore
from core.hooks import HookManager
from core.loop import EventCallback, run_agent_loop, run_agent_loop_streaming
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

## STRICT READ-ONLY MODE — MANDATORY
This system is configured for READ-ONLY access to the Neo4j database.
You MUST NEVER generate or attempt any write/mutate operations.

FORBIDDEN operations (will be blocked and rejected):
  Data:   CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE, FOREACH
  Schema: CREATE INDEX, DROP INDEX, CREATE CONSTRAINT, DROP CONSTRAINT
  Admin:  CREATE/DROP/ALTER DATABASE, START/STOP DATABASE, CREATE ALIAS
  Auth:   CREATE/DROP/ALTER USER, CREATE/DROP ROLE, GRANT, DENY, REVOKE, RENAME
  Import: LOAD CSV
  Other:  CALL { ... } (subquery writes), IN TRANSACTIONS, TERMINATE

If a user asks you to create, update, delete, or modify any data:
  → Politely decline and explain that this system is read-only.
  → Do NOT attempt the operation even if the user insists.
  → Suggest they use a direct Neo4j client if they need write access.

Only use MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT, SKIP, UNION,
UNWIND, OPTIONAL MATCH, CALL (read procedures), SHOW, and aggregations.

Always check relationship directions before writing queries — wrong direction = 0 results.
"""


class Orchestrator:
    """Main supervisor that wires together the TAOR loop, tools, and discovery."""

    def __init__(
        self,
        llm_factory: Callable[..., BaseChatModel],
        registry: CapabilityRegistry,
        tool_manager: ToolManager,
        config: Settings,
        hook_manager: HookManager | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._registry = registry
        self._tool_manager = tool_manager
        self._config = config
        self._hook_manager = hook_manager
        self._sessions = SessionStore()

        # Prompt cache state
        self._cached_system_prompt: str | None = None
        self._prompt_cache_hash: str | None = None

    # ── System prompt (with caching) ─────────────────────────────────────

    def _compute_prompt_hash(self) -> str:
        """Hash the inputs that determine the system prompt content."""
        parts: list[str] = []
        parts.append(str(len(self._registry.skills)))
        parts.append(str(len(self._registry.agents)))
        for name in sorted(self._registry.skills):
            parts.append(name)
        for name in sorted(self._registry.agents):
            parts.append(name)
        # Include CLAUDE.md mtime if it exists
        project_md = Path("CLAUDE.md")
        if project_md.exists():
            parts.append(str(os.path.getmtime(project_md)))
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _build_system_prompt(self) -> str:
        """Compose the full system prompt with dynamic capability descriptions.

        Uses an in-memory cache keyed by a hash of registry state + CLAUDE.md mtime.
        """
        current_hash = self._compute_prompt_hash()
        if (
            self._config.enable_prompt_cache
            and self._cached_system_prompt is not None
            and self._prompt_cache_hash == current_hash
        ):
            logger.debug("Using cached system prompt (hash=%s)", current_hash[:8])
            return self._cached_system_prompt

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

        prompt = "\n\n".join(parts)

        # Cache for next call
        self._cached_system_prompt = prompt
        self._prompt_cache_hash = current_hash
        logger.info("Built system prompt (%d chars, hash=%s)", len(prompt), current_hash[:8])

        return prompt

    def invalidate_prompt_cache(self) -> None:
        """Force the system prompt to be rebuilt on next query."""
        self._cached_system_prompt = None
        self._prompt_cache_hash = None

    # ── Query handling (non-streaming) ───────────────────────────────────

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
            hook_manager=self._hook_manager,
            enable_prompt_cache=self._config.enable_prompt_cache,
        )

        # ── Stop hook ────────────────────────────────────────────────────
        if self._hook_manager:
            stop = await self._hook_manager.run_stop(result, tools_called, turn_counter["count"])
            if stop.action == "modify" and stop.modified_response:
                result = stop.modified_response

        return {
            "response": result,
            "session_id": context.session_id,
            "turns_used": turn_counter["count"],
            "tools_called": tools_called,
        }

    # ── Query handling (token-by-token streaming) ────────────────────────

    async def handle_query_streaming(
        self,
        user_input: str,
        session_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator that yields token-level SSE events."""
        # ── Slash command shortcut ───────────────────────────────────────
        resolved = self._registry.resolve_slash_command(user_input)
        if resolved:
            skill, args = resolved
            response = await self._execute_slash_skill(skill, args, on_event=None)
            yield {"type": "done", "content": response, "session_id": session_id or "ephemeral", "turns_used": 1}
            return

        # ── Normal streaming TAOR flow ───────────────────────────────────
        context = self._sessions.get_or_create(session_id)
        context.add_user(user_input)

        llm = self._llm_factory()
        await context.compact(llm, self._config.max_context_tokens, self._config.compact_threshold)

        system_prompt = self._build_system_prompt()
        tools = self._tool_manager.get_all_tools()
        tools_called: list[str] = []
        final_turn = 0

        async for event in run_agent_loop_streaming(
            llm=llm,
            system_prompt=system_prompt,
            messages=context.messages,
            tools=tools,
            max_turns=self._config.max_turns,
            hook_manager=self._hook_manager,
            enable_prompt_cache=self._config.enable_prompt_cache,
        ):
            # Track tools for the Stop hook
            if event.get("type") == "tool_end":
                tools_called.append(event.get("tool", ""))
            if "turn" in event:
                final_turn = event["turn"]

            # Enrich "done" events with session info
            if event.get("type") == "done":
                content = event.get("content", "")

                # ── Stop hook ────────────────────────────────────────────
                if self._hook_manager:
                    stop = await self._hook_manager.run_stop(content, tools_called, final_turn)
                    if stop.action == "modify" and stop.modified_response:
                        content = stop.modified_response

                yield {
                    "type": "done",
                    "content": content,
                    "session_id": context.session_id,
                    "turns_used": final_turn + 1,
                    "tools_called": tools_called,
                }
                return

            yield event

    # ── Slash skill execution ────────────────────────────────────────────

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
                self._llm_factory, self._tool_manager, self._registry,
                hook_manager=self._hook_manager,
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
                hook_manager=self._hook_manager,
                enable_prompt_cache=self._config.enable_prompt_cache,
            )

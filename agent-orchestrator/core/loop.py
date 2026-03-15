"""TAOR agentic loop — Think → Act → Observe → Repeat.

This is the core engine.  All intelligence lives in the LLM; this module
is a thin sampling loop that:
  1. Sends messages to the LLM (with tool definitions)
  2. If the LLM returns tool calls → executes them → feeds results back
  3. If the LLM returns plain text → done

Includes:
  - Hook integration (PreToolUse / PostToolUse)
  - Token-by-token streaming variant
  - Prompt-cache-aware system message construction
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Callable, Awaitable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Type alias for the optional event callback
EventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None

# Avoid hard import — hooks are optional
try:
    from core.hooks import HookManager
except ImportError:
    HookManager = None  # type: ignore[assignment,misc]


def _make_system_message(system_prompt: str, enable_cache: bool = False) -> SystemMessage:
    """Build the SystemMessage, optionally marking it for prompt caching."""
    if enable_cache:
        return SystemMessage(
            content=system_prompt,
            additional_kwargs={"cache_control": {"type": "ephemeral"}},
        )
    return SystemMessage(content=system_prompt)


# ─────────────────────────────────────────────────────────────────────────────
# Standard (non-streaming) TAOR loop
# ─────────────────────────────────────────────────────────────────────────────


async def run_agent_loop(
    llm: BaseChatModel,
    system_prompt: str,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_turns: int = 50,
    on_event: EventCallback = None,
    hook_manager: HookManager | None = None,
    enable_prompt_cache: bool = False,
) -> str:
    """Run the TAOR loop until the LLM stops calling tools or we hit *max_turns*.

    Parameters
    ----------
    llm:
        A LangChain chat model (any provider).
    system_prompt:
        Injected as the first ``SystemMessage`` on every call.
    messages:
        **Mutable** conversation history — new messages are appended in place.
    tools:
        LangChain tools available to the LLM.
    max_turns:
        Safety cap on the number of LLM round-trips.
    on_event:
        Optional async callback for SSE events.
    hook_manager:
        Optional HookManager for PreToolUse / PostToolUse lifecycle hooks.
    enable_prompt_cache:
        If True, adds cache_control to the system message (Anthropic prompt caching).

    Returns
    -------
    str
        The LLM's final textual response.
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    sys_msg = _make_system_message(system_prompt, enable_prompt_cache)

    for turn in range(max_turns):
        # ── THINK ────────────────────────────────────────────────────────
        response: AIMessage = await llm_with_tools.ainvoke([sys_msg] + messages)
        messages.append(response)

        if on_event:
            await on_event(
                {"type": "assistant", "content": response.content, "turn": turn}
            )

        # No tool calls → the LLM is done
        if not response.tool_calls:
            return response.content if isinstance(response.content, str) else str(response.content)

        # ── ACT + OBSERVE ────────────────────────────────────────────────
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # ── PreToolUse hook ──────────────────────────────────────────
            if hook_manager:
                pre = await hook_manager.run_pre_tool(tool_name, dict(tool_args))
                if pre.action == "skip":
                    messages.append(
                        ToolMessage(
                            content=f"Skipped by hook: {pre.reason or 'no reason'}",
                            tool_call_id=tool_id,
                        )
                    )
                    if on_event:
                        await on_event({"type": "hook_skip", "tool": tool_name, "reason": pre.reason, "turn": turn})
                    continue
                if pre.action == "block":
                    messages.append(
                        ToolMessage(
                            content=f"Blocked by hook: {pre.reason or 'no reason'}",
                            tool_call_id=tool_id,
                            status="error",
                        )
                    )
                    if on_event:
                        await on_event({"type": "hook_block", "tool": tool_name, "reason": pre.reason, "turn": turn})
                    continue
                if pre.action == "modify" and pre.modified_args:
                    tool_args = pre.modified_args

            # ── Find and execute tool ────────────────────────────────────
            tool = _find_tool(tools, tool_name)
            if tool is None:
                logger.warning("LLM requested unknown tool: %s", tool_name)
                messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{tool_name}'",
                        tool_call_id=tool_id,
                        status="error",
                    )
                )
                continue

            try:
                result = await tool.ainvoke(tool_args)
                result_str = str(result) if result is not None else ""
                # Truncate very large results to avoid blowing context
                if len(result_str) > 50_000:
                    result_str = result_str[:50_000] + "\n... [truncated]"
            except Exception as exc:
                logger.exception("Tool %s raised: %s", tool_name, exc)
                result_str = f"Error executing {tool_name}: {exc}"
                messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_id, status="error")
                )
                if on_event:
                    await on_event({"type": "tool_result", "tool": tool_name, "turn": turn})
                continue

            # ── PostToolUse hook ─────────────────────────────────────────
            if hook_manager:
                post = await hook_manager.run_post_tool(tool_name, tool_args, result_str)
                if post.action == "modify" and post.modified_result is not None:
                    result_str = post.modified_result

            messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))

            if on_event:
                await on_event(
                    {"type": "tool_result", "tool": tool_name, "turn": turn}
                )

    return "Max turns reached."


# ─────────────────────────────────────────────────────────────────────────────
# Token-by-token streaming TAOR loop
# ─────────────────────────────────────────────────────────────────────────────


async def run_agent_loop_streaming(
    llm: BaseChatModel,
    system_prompt: str,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_turns: int = 50,
    hook_manager: HookManager | None = None,
    enable_prompt_cache: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming TAOR loop — yields events including individual tokens.

    Event types yielded:
      - ``{"type": "token", "content": "...", "turn": N}``
      - ``{"type": "tool_start", "tool": "...", "args": {...}, "turn": N}``
      - ``{"type": "tool_end", "tool": "...", "result": "...", "turn": N}``
      - ``{"type": "hook_skip", "tool": "...", "reason": "...", "turn": N}``
      - ``{"type": "hook_block", "tool": "...", "reason": "...", "turn": N}``
      - ``{"type": "done", "content": "...", "turn": N}``
      - ``{"type": "max_turns"}``
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    sys_msg = _make_system_message(system_prompt, enable_prompt_cache)

    for turn in range(max_turns):
        # ── THINK (streaming) ────────────────────────────────────────────
        full_response: AIMessageChunk | None = None

        async for chunk in llm_with_tools.astream([sys_msg] + messages):
            # Yield each text token
            if chunk.content:
                yield {"type": "token", "content": chunk.content, "turn": turn}
            # Accumulate chunks into the full response
            if full_response is None:
                full_response = chunk
            else:
                full_response = full_response + chunk

        if full_response is None:
            yield {"type": "done", "content": "", "turn": turn}
            return

        # Convert accumulated chunk to AIMessage for history
        ai_msg = AIMessage(
            content=full_response.content,
            tool_calls=full_response.tool_calls if full_response.tool_calls else [],
            additional_kwargs=full_response.additional_kwargs,
        )
        messages.append(ai_msg)

        # No tool calls → done
        if not ai_msg.tool_calls:
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            yield {"type": "done", "content": content, "turn": turn}
            return

        # ── ACT + OBSERVE ────────────────────────────────────────────────
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # ── PreToolUse hook ──────────────────────────────────────────
            if hook_manager:
                pre = await hook_manager.run_pre_tool(tool_name, dict(tool_args))
                if pre.action == "skip":
                    messages.append(
                        ToolMessage(
                            content=f"Skipped by hook: {pre.reason or 'no reason'}",
                            tool_call_id=tool_id,
                        )
                    )
                    yield {"type": "hook_skip", "tool": tool_name, "reason": pre.reason, "turn": turn}
                    continue
                if pre.action == "block":
                    messages.append(
                        ToolMessage(
                            content=f"Blocked by hook: {pre.reason or 'no reason'}",
                            tool_call_id=tool_id,
                            status="error",
                        )
                    )
                    yield {"type": "hook_block", "tool": tool_name, "reason": pre.reason, "turn": turn}
                    continue
                if pre.action == "modify" and pre.modified_args:
                    tool_args = pre.modified_args

            # ── Find and execute tool ────────────────────────────────────
            tool = _find_tool(tools, tool_name)
            if tool is None:
                messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool '{tool_name}'",
                        tool_call_id=tool_id,
                        status="error",
                    )
                )
                continue

            yield {"type": "tool_start", "tool": tool_name, "args": tool_args, "turn": turn}

            try:
                result = await tool.ainvoke(tool_args)
                result_str = str(result) if result is not None else ""
                if len(result_str) > 50_000:
                    result_str = result_str[:50_000] + "\n... [truncated]"
            except Exception as exc:
                logger.exception("Tool %s raised: %s", tool_name, exc)
                result_str = f"Error executing {tool_name}: {exc}"
                messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_id, status="error")
                )
                yield {"type": "tool_end", "tool": tool_name, "result": result_str[:500], "turn": turn, "error": True}
                continue

            # ── PostToolUse hook ─────────────────────────────────────────
            if hook_manager:
                post = await hook_manager.run_post_tool(tool_name, tool_args, result_str)
                if post.action == "modify" and post.modified_result is not None:
                    result_str = post.modified_result

            messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
            yield {"type": "tool_end", "tool": tool_name, "result": result_str[:500], "turn": turn}

    yield {"type": "max_turns"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_tool(tools: list[BaseTool], name: str) -> BaseTool | None:
    """Look up a tool by name (case-sensitive)."""
    for t in tools:
        if t.name == name:
            return t
    return None

"""TAOR agentic loop — Think → Act → Observe → Repeat.

This is the core engine (~60 lines of real logic).  All intelligence
lives in the LLM; this module is a thin sampling loop that:
  1. Sends messages to the LLM (with tool definitions)
  2. If the LLM returns tool calls → executes them → feeds results back
  3. If the LLM returns plain text → done
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Type alias for the optional event callback
EventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


async def run_agent_loop(
    llm: BaseChatModel,
    system_prompt: str,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_turns: int = 50,
    on_event: EventCallback = None,
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
        Optional async callback for streaming / SSE events.

    Returns
    -------
    str
        The LLM's final textual response.
    """
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    sys_msg = SystemMessage(content=system_prompt)

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
                messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
            except Exception as exc:
                logger.exception("Tool %s raised: %s", tool_name, exc)
                messages.append(
                    ToolMessage(
                        content=f"Error executing {tool_name}: {exc}",
                        tool_call_id=tool_id,
                        status="error",
                    )
                )

            if on_event:
                await on_event(
                    {"type": "tool_result", "tool": tool_name, "turn": turn}
                )

    return "Max turns reached."


def _find_tool(tools: list[BaseTool], name: str) -> BaseTool | None:
    """Look up a tool by name (case-sensitive)."""
    for t in tools:
        if t.name == name:
            return t
    return None

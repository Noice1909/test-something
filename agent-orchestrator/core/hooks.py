"""Lifecycle hook system — PreToolUse, PostToolUse, Stop.

Hooks intercept the TAOR loop at three points, allowing callers to
modify arguments, filter results, skip/block tool calls, or post-process
the final response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Literal

logger = logging.getLogger(__name__)


# ── Hook result ──────────────────────────────────────────────────────────────


@dataclass
class HookResult:
    """Returned by every hook callback to tell the loop what to do."""

    action: Literal["allow", "skip", "block", "modify"] = "allow"
    modified_args: dict[str, Any] | None = None       # PreToolUse only
    modified_result: str | None = None                 # PostToolUse only
    modified_response: str | None = None               # Stop only
    reason: str | None = None                          # explanation (skip/block)


# ── Hook types ───────────────────────────────────────────────────────────────


class HookType(str, Enum):
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    STOP = "stop"


# Callback signatures (for documentation — not enforced at runtime):
#   PreToolUse:  async (tool_name: str, tool_args: dict) -> HookResult
#   PostToolUse: async (tool_name: str, tool_args: dict, result: str) -> HookResult
#   Stop:        async (response: str, tools_called: list[str], turns: int) -> HookResult

PreToolHook = Callable[[str, dict[str, Any]], Awaitable[HookResult]]
PostToolHook = Callable[[str, dict[str, Any], str], Awaitable[HookResult]]
StopHook = Callable[[str, list[str], int], Awaitable[HookResult]]


# ── Hook entry (internal) ───────────────────────────────────────────────────


@dataclass(order=True)
class _HookEntry:
    priority: int
    callback: Callable = field(compare=False)
    name: str = field(default="", compare=False)


# ── Hook manager ─────────────────────────────────────────────────────────────


class HookManager:
    """Registry for lifecycle hooks.  Hooks run in priority order (lowest first)."""

    def __init__(self) -> None:
        self._hooks: dict[HookType, list[_HookEntry]] = {ht: [] for ht in HookType}

    # ── Registration ─────────────────────────────────────────────────────

    def register(
        self,
        hook_type: HookType,
        callback: Callable,
        priority: int = 0,
        name: str = "",
    ) -> None:
        entry = _HookEntry(priority=priority, callback=callback, name=name or callback.__name__)
        self._hooks[hook_type].append(entry)
        self._hooks[hook_type].sort()
        logger.info("Registered %s hook: %s (priority=%d)", hook_type.value, entry.name, priority)

    def unregister(self, hook_type: HookType, callback: Callable) -> None:
        self._hooks[hook_type] = [e for e in self._hooks[hook_type] if e.callback is not callback]

    def clear(self, hook_type: HookType | None = None) -> None:
        if hook_type:
            self._hooks[hook_type].clear()
        else:
            for ht in HookType:
                self._hooks[ht].clear()

    # ── Execution ────────────────────────────────────────────────────────

    async def run_pre_tool(self, tool_name: str, tool_args: dict[str, Any]) -> HookResult:
        """Run all PreToolUse hooks.  First non-allow result wins."""
        for entry in self._hooks[HookType.PRE_TOOL_USE]:
            try:
                result = await entry.callback(tool_name, tool_args)
                if result.action != "allow":
                    logger.info(
                        "PreToolUse hook '%s' → %s tool '%s': %s",
                        entry.name, result.action, tool_name, result.reason,
                    )
                    return result
                # "modify" with new args — apply and continue chain
                if result.modified_args is not None:
                    tool_args.update(result.modified_args)
            except Exception:
                logger.exception("PreToolUse hook '%s' failed", entry.name)
        return HookResult(action="allow")

    async def run_post_tool(
        self, tool_name: str, tool_args: dict[str, Any], result: str,
    ) -> HookResult:
        """Run all PostToolUse hooks.  First modify/skip/block wins."""
        for entry in self._hooks[HookType.POST_TOOL_USE]:
            try:
                hook_result = await entry.callback(tool_name, tool_args, result)
                if hook_result.action == "modify" and hook_result.modified_result is not None:
                    result = hook_result.modified_result  # chain modifications
                elif hook_result.action in ("skip", "block"):
                    return hook_result
            except Exception:
                logger.exception("PostToolUse hook '%s' failed", entry.name)
        return HookResult(action="allow", modified_result=result)

    async def run_stop(
        self, response: str, tools_called: list[str], turns: int,
    ) -> HookResult:
        """Run all Stop hooks.  Modifications are chained."""
        current_response = response
        for entry in self._hooks[HookType.STOP]:
            try:
                hook_result = await entry.callback(current_response, tools_called, turns)
                if hook_result.action == "modify" and hook_result.modified_response is not None:
                    current_response = hook_result.modified_response
            except Exception:
                logger.exception("Stop hook '%s' failed", entry.name)
        if current_response != response:
            return HookResult(action="modify", modified_response=current_response)
        return HookResult(action="allow")


# ── Built-in hooks ───────────────────────────────────────────────────────────


import re as _re

# Mirrors the full list in tools/builtin.py — keep in sync.
_WRITE_OPS = (
    # DML
    "CREATE ", "MERGE ", "DELETE ", "DETACH DELETE ", "DETACH ", "SET ", "REMOVE ",
    "FOREACH ", "FOREACH(",
    # Schema / Index / Constraint
    "CREATE INDEX ", "CREATE INDEX(", "DROP INDEX ",
    "CREATE CONSTRAINT ", "DROP CONSTRAINT ",
    "CREATE FULLTEXT INDEX ", "CREATE LOOKUP INDEX ",
    "CREATE POINT INDEX ", "CREATE RANGE INDEX ", "CREATE TEXT INDEX ",
    "CREATE OR REPLACE ",
    # Admin / Database
    "CREATE DATABASE ", "DROP DATABASE ", "ALTER DATABASE ",
    "START DATABASE ", "STOP DATABASE ", "CREATE COMPOSITE DATABASE ",
    "CREATE ALIAS ", "DROP ALIAS ", "ALTER ALIAS ",
    # Security / Users / Roles
    "CREATE USER ", "DROP USER ", "ALTER USER ",
    "CREATE ROLE ", "DROP ROLE ", "RENAME ",
    "GRANT ", "DENY ", "REVOKE ",
    # Server / Cluster
    "ALTER SERVER ", "ENABLE SERVER ", "DEALLOCATE ", "REALLOCATE ", "TERMINATE ",
    # Bulk import
    "LOAD CSV ", "LOAD CSV(",
    # Subquery writes / batching
    "CALL {", " IN TRANSACTIONS",
)


async def safety_write_blocker(tool_name: str, tool_args: dict[str, Any]) -> HookResult:
    """Built-in PreToolUse hook — blocks ANY write/mutate operation.

    Catches write attempts before they even reach the Neo4j driver.
    This is the first line of defense (the driver's execute_read is the second).
    """
    if tool_name == "RunCypherTool":
        raw = str(tool_args.get("query", ""))
        query = _re.sub(r"\s+", " ", raw.upper().strip())
        for op in _WRITE_OPS:
            if op in query:
                return HookResult(
                    action="block",
                    reason=f"BLOCKED: '{op.strip()}' — this system is strictly READ-ONLY",
                )
    return HookResult(action="allow")


async def tool_call_logger(tool_name: str, tool_args: dict[str, Any]) -> HookResult:
    """Built-in PreToolUse hook — logs every tool call."""
    logger.info("Hook[tool_call_logger] → %s(%s)", tool_name, list(tool_args.keys()))
    return HookResult(action="allow")

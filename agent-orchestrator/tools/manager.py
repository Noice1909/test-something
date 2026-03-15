"""Unified tool registry — registers, filters, and dispatches tools."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from neo4j import AsyncDriver

from tools.builtin import create_neo4j_tools

logger = logging.getLogger(__name__)


class ToolManager:
    """Central registry of all available tools (built-in + MCP + orchestration)."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self.mcp_bridge: Any = None  # set later if MCP is enabled

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a single tool by its ``name``."""
        if tool.name in self._tools:
            logger.warning("Overwriting tool: %s", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def register_neo4j_tools(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        """Register all built-in Neo4j tools."""
        for tool in create_neo4j_tools(driver, database):
            self.register(tool)

    async def register_mcp_servers(self, configs: dict[str, dict]) -> None:
        """Start MCP servers and register their tools."""
        if not configs:
            return
        # Lazy import to avoid hard dependency
        from tools.mcp_bridge import MCPBridge

        self.mcp_bridge = MCPBridge()
        for name, config in configs.items():
            try:
                mcp_tools = await self.mcp_bridge.connect(name, config)
                for tool in mcp_tools:
                    self.register(tool)
                logger.info("MCP server '%s' registered %d tools", name, len(mcp_tools))
            except Exception as exc:
                logger.error("Failed to connect MCP server '%s': %s", name, exc)

    # ── Retrieval ────────────────────────────────────────────────────────

    def get_all_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> BaseTool | None:
        """Look up a single tool by name."""
        return self._tools.get(name)

    def filter_tools(
        self,
        allowed: list[str] | None = None,
        disallowed: list[str] | None = None,
    ) -> list[BaseTool]:
        """Return a filtered subset of tools.

        Parameters
        ----------
        allowed:
            If not None, only include tools with these names.
        disallowed:
            Exclude tools with these names (applied after ``allowed``).
        """
        if allowed is not None:
            tools = [self._tools[n] for n in allowed if n in self._tools]
        else:
            tools = list(self._tools.values())

        if disallowed:
            exclude = set(disallowed)
            tools = [t for t in tools if t.name not in exclude]

        return tools

    @property
    def tool_names(self) -> list[str]:
        """Names of all registered tools."""
        return list(self._tools.keys())

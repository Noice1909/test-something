"""MCP (Model Context Protocol) bridge — connect external MCP servers and
wrap their tools as LangChain ``BaseTool`` instances.

Uses stdio transport: starts each MCP server as a subprocess and
communicates via JSON-RPC over stdin/stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_REQUEST_ID = 0


def _next_id() -> int:
    global _REQUEST_ID
    _REQUEST_ID += 1
    return _REQUEST_ID


class _MCPServerConnection:
    """A single MCP server subprocess connection."""

    def __init__(self, name: str, process: asyncio.subprocess.Process) -> None:
        self.name = name
        self.process = process
        self._lock = asyncio.Lock()

    async def send_request(self, method: str, params: dict | None = None) -> Any:
        """Send a JSON-RPC request and wait for the response."""
        assert self.process.stdin is not None
        assert self.process.stdout is not None

        request = {
            "jsonrpc": "2.0",
            "id": _next_id(),
            "method": method,
            "params": params or {},
        }

        async with self._lock:
            self.process.stdin.write(
                (json.dumps(request) + "\n").encode("utf-8")
            )
            await self.process.stdin.drain()

            # Read until we get a complete JSON-RPC response
            line = await asyncio.wait_for(
                self.process.stdout.readline(), timeout=30
            )
            response = json.loads(line.decode("utf-8"))

        if "error" in response:
            raise RuntimeError(
                f"MCP error from '{self.name}': {response['error']}"
            )
        return response.get("result")

    async def close(self) -> None:
        try:
            self.process.terminate()
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except Exception:
            self.process.kill()


class _MCPToolWrapper(BaseTool):
    """LangChain tool that wraps a single MCP tool."""

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] | None = None
    read_only: bool = True  # assume MCP tools are read-only by default

    _connection: Any = None  # _MCPServerConnection
    _mcp_tool_name: str = ""

    class Config:
        arbitrary_types_allowed = True

    async def _arun(self, **kwargs: Any) -> str:
        try:
            result = await self._connection.send_request(
                "tools/call",
                {"name": self._mcp_tool_name, "arguments": kwargs},
            )
            # MCP returns content array
            if isinstance(result, dict) and "content" in result:
                parts = []
                for item in result["content"]:
                    if item.get("type") == "text":
                        parts.append(item["text"])
                    else:
                        parts.append(json.dumps(item))
                return "\n".join(parts)
            return json.dumps(result, default=str)
        except Exception as exc:
            return f"MCP tool error: {exc}"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


def _build_pydantic_model(tool_name: str, input_schema: dict) -> Type[BaseModel]:
    """Dynamically build a Pydantic model from an MCP tool's inputSchema."""
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_def in properties.items():
        prop_type = prop_def.get("type", "string")
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "object": dict,
            "array": list,
        }
        py_type = type_map.get(prop_type, str)
        desc = prop_def.get("description", "")

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=desc))
        else:
            default = prop_def.get("default")
            fields[prop_name] = (
                py_type | None,
                Field(default=default, description=desc),
            )

    # Create model dynamically
    model = type(
        f"MCP_{tool_name}_Input",
        (BaseModel,),
        {"__annotations__": {k: v[0] for k, v in fields.items()},
         **{k: v[1] for k, v in fields.items()}},
    )
    return model  # type: ignore[return-value]


class MCPBridge:
    """Connect to external MCP servers and expose their tools."""

    def __init__(self) -> None:
        self._connections: dict[str, _MCPServerConnection] = {}

    async def connect(self, name: str, config: dict) -> list[BaseTool]:
        """Start an MCP server subprocess and discover its tools.

        Config format::

            {"command": "python", "args": ["mcp_server.py"], "env": {}}
        """
        command = config["command"]
        args = config.get("args", [])
        env = config.get("env")

        logger.info("Starting MCP server '%s': %s %s", name, command, args)

        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        conn = _MCPServerConnection(name, process)
        self._connections[name] = conn

        # Initialize
        await conn.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent-orchestrator", "version": "1.0.0"},
        })

        # Discover tools
        tools_result = await conn.send_request("tools/list")
        mcp_tools = tools_result.get("tools", []) if isinstance(tools_result, dict) else []

        # Wrap each MCP tool as a LangChain BaseTool
        langchain_tools: list[BaseTool] = []
        for tool_def in mcp_tools:
            mcp_name = tool_def["name"]
            lc_name = f"mcp__{name}__{mcp_name}"

            # Build Pydantic input schema
            input_schema = tool_def.get("inputSchema", {})
            args_model = _build_pydantic_model(mcp_name, input_schema)

            wrapper = _MCPToolWrapper(
                name=lc_name,
                description=tool_def.get("description", ""),
                args_schema=args_model,
            )
            wrapper._connection = conn
            wrapper._mcp_tool_name = mcp_name
            langchain_tools.append(wrapper)

            logger.info("  MCP tool: %s → %s", mcp_name, lc_name)

        return langchain_tools

    async def disconnect_all(self) -> None:
        """Gracefully shutdown all MCP server subprocesses."""
        for name, conn in self._connections.items():
            logger.info("Shutting down MCP server '%s'", name)
            await conn.close()
        self._connections.clear()

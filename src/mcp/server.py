from __future__ import annotations

import json
from typing import Any

import structlog
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.mcp.tools import MCPTools

logger = structlog.get_logger()


def create_mcp_server(tools: MCPTools) -> Server:
    """Create an MCP server exposing Neo4j agent tools."""

    server = Server("neo4j-nl-agent")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_schema",
                description="Returns full or filtered graph schema (labels, relationships, properties, directions)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filter_labels": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of labels to filter schema to",
                        }
                    },
                },
            ),
            Tool(
                name="search_concepts",
                description="Search :Concept nodes by nlp_terms to find relevant labels",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search term"}
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="fuzzy_search_global",
                description="Full-text fuzzy search across all node names",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "term": {"type": "string", "description": "Search term"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["term"],
                },
            ),
            Tool(
                name="fuzzy_search_by_label",
                description="Full-text fuzzy search within a specific label",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Node label"},
                        "term": {"type": "string", "description": "Search term"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["label", "term"],
                },
            ),
            Tool(
                name="execute_cypher",
                description="Execute a read-only Cypher query (write operations blocked)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cypher": {"type": "string", "description": "Cypher query"}
                    },
                    "required": ["cypher"],
                },
            ),
            Tool(
                name="validate_cypher",
                description="Validate Cypher syntax and schema compliance",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "cypher": {"type": "string", "description": "Cypher query to validate"}
                    },
                    "required": ["cypher"],
                },
            ),
            Tool(
                name="list_indexes",
                description="List all discovered FULLTEXT indexes",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        dispatch = {
            "get_schema": lambda: tools.get_schema(arguments.get("filter_labels")),
            "search_concepts": lambda: tools.search_concepts(arguments["query"]),
            "fuzzy_search_global": lambda: tools.fuzzy_search_global(
                arguments["term"], arguments.get("limit", 5)
            ),
            "fuzzy_search_by_label": lambda: tools.fuzzy_search_by_label(
                arguments["label"], arguments["term"], arguments.get("limit", 5)
            ),
            "execute_cypher": lambda: tools.execute_cypher(arguments["cypher"]),
            "validate_cypher": lambda: tools.validate_cypher(arguments["cypher"]),
            "list_indexes": lambda: tools.list_indexes(),
        }

        handler = dispatch.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        try:
            result = handler()
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        except Exception as exc:
            logger.error("mcp_tool_error", tool=name, error=str(exc))
            return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]

    return server


async def run_mcp_stdio(tools: MCPTools) -> None:
    """Run the MCP server over STDIO."""
    server = create_mcp_server(tools)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

---
name: find-paths
description: Find shortest paths or all paths between two entities in the graph. Use when the user asks "how are X and Y connected?" or "what's the path from A to B?"
allowed-tools: RunCypherTool SearchNodesTool GetSchemaTool
context: fork
argument-hint: "[entity-A] [entity-B]"
---

You are a path finding specialist.

1. Find both entities using SearchNodesTool
2. Check schema for possible connecting patterns
3. Try shortestPath first:
   `MATCH p = shortestPath((a)-[*..10]-(b)) WHERE elementId(a) = $id1 AND elementId(b) = $id2 RETURN p`
4. If found, describe each hop in the path
5. If no path exists, say so and explain why (disconnected subgraphs)

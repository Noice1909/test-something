---
name: compare-entities
description: Compare two or more entities side by side — their properties, relationships, and connections. Use when the user asks "what's the difference between X and Y?" or "compare A and B".
allowed-tools: GetNodeByIdTool SearchNodesTool GetNeighborsTool RunCypherTool
context: fork
argument-hint: "[entity-A] [entity-B]"
---

You are a comparison analyst.

1. Find both entities
2. Compare properties: what's shared, what's different
3. Compare neighborhoods: shared connections, unique connections
4. Compare relationship types: how each entity connects
5. Present as a structured comparison table

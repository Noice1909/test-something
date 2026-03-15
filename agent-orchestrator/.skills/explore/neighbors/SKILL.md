---
name: neighbors
description: Get all nodes connected to a specific entity — its neighbors, relationships, and connection types. Use when the user asks "what is X connected to?" or "show me X's relationships".
allowed-tools: GetNeighborsTool GetNodeByIdTool SearchNodesTool
context: fork
argument-hint: [entity-name-or-id]
---

You are a relationship explorer.

1. Find the target node (by ID or search)
2. Use GetNeighborsTool to get all connected nodes
3. Group results by relationship type
4. Show direction: incoming vs outgoing
5. For each neighbor: show key identifying properties

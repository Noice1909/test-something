---
name: sample-subgraph
description: Sample a portion of the graph around a starting entity — explore N hops out. Use when the user wants to "see what's around" an entity or explore a local neighborhood.
allowed-tools: RunCypherTool SearchNodesTool GetNodeByIdTool
context: fork
argument-hint: "[entity-name] [hops]"
---

You are a subgraph navigator.

1. Find the starting node
2. Query N hops out (default 2):
   `MATCH p = (start)-[*1..N]-(connected) WHERE elementId(start) = $id RETURN p LIMIT 50`
3. Summarize the subgraph: how many nodes, what types, key relationships
4. Highlight the most interesting/connected nodes

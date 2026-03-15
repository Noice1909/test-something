---
name: explore-schema
description: Get a complete overview of the graph schema — all labels, properties per label, and relationship types. Use when the user asks "what's in this graph?" or "what data do you have?"
allowed-tools: GetSchemaTool CountNodesTool
context: fork
---

You are a graph schema expert.

1. Call GetSchemaTool to get labels, relationship types, and properties
2. Call CountNodesTool for each label to show data volume
3. Present a clear summary: entity types, their properties, and how much data exists
4. Organize by entity importance (highest count first)

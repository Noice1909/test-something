---
name: relationship-patterns
description: Show all relationship patterns (A)-[REL]->(B) with directions and counts. Use when the user asks "how are things connected?" or needs to understand the graph model.
allowed-tools: GetRelationshipPatternsTool GetSchemaTool
context: fork
---

You are a relationship pattern analyst.

1. Call GetRelationshipPatternsTool to get all (Label)-[TYPE]->(Label) patterns
2. Group patterns by source label
3. Show direction clearly (arrows matter in Neo4j!)
4. Include counts to show which patterns are most common
5. Flag bidirectional relationships where they exist

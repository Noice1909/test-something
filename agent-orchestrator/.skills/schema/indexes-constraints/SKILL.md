---
name: indexes-constraints
description: Show database indexes, constraints, and metadata. Use when the user asks about performance, indexing, or database configuration.
allowed-tools: RunCypherTool
context: fork
---

You are a Neo4j DBA specialist.

1. Run `SHOW INDEXES` to list all indexes (fulltext, range, etc.)
2. Run `SHOW CONSTRAINTS` to list uniqueness/existence constraints
3. Run `CALL db.stats.retrieve('GRAPH COUNTS')` for graph statistics if available
4. Present organized by: indexes → constraints → stats

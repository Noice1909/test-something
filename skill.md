---
name: Agentic Graph Query System
description: Complete skill documentation for the autonomous multi-specialist Neo4j query system
---

# Agentic Graph Query System — Skill Reference

## Overview

The system uses an LLM-powered Supervisor to analyze user questions and
coordinate 6 specialist agents that interact with Neo4j through 80+ tools.

## Specialist Skills

### 1. Discovery Specialist
- **Input**: User question
- **Process**: LLM extracts search terms → multi-strategy DB search
- **Strategies**: exact_match, fuzzy_match (fulltext index), label_match
- **Output**: `DiscoveryResult[]` with entity names, labels, node-ids, confidence scores

### 2. Schema Reasoning Specialist
- **Input**: Question + discoveries + full schema
- **Process**: LLM evaluates which node labels and relationship types are relevant
- **Output**: `SchemaSelection` with filtered labels, rel types, and reasoning

### 3. Query Planning Specialist
- **Input**: Question + discoveries + schema selection
- **Process**: LLM determines complexity (DIRECT/ONE_HOP/MULTI_HOP/AGGREGATION) and intent (LIST/COUNT/FIND/EXPLORE)
- **Output**: `QueryPlan`

### 4. Query Generation Specialist
- **Input**: Question + plan + schema + discoveries
- **Process**: LLM generates parameterized Cypher query, validates read-only safety
- **Output**: `GeneratedQuery`

### 5. Execution Specialist
- **Input**: Generated query
- **Process**: Safety validation → execute on Neo4j → categorize errors
- **Output**: `ExecutionResult` with rows or categorized error

### 6. Reflection Specialist
- **Input**: Failed state (question, strategy, query, error)
- **Process**: LLM analyzes failure → recommends retry strategy
- **Strategies**: expand_discovery, simplify_query, add_traversals, change_schema, give_up
- **Output**: `ReflectionResult`

## Tool Categories (80+ tools)

1. **Schema Discovery** (20): Labels, relationships, properties, indexes, constraints
2. **Graph Exploration** (20): Neighbors, paths, shortest path, k-hop, components
3. **Graph Search** (10): Text search, property matching, fulltext, vector similarity
4. **Aggregation** (10): Counts, distributions, most/least connected
5. **Data Inspection** (10): Sampling, statistics, distinct values
6. **Query Execution** (5): Run Cypher, explain, profile, validate
7. **Metadata** (5): Procedures, functions, settings, health check

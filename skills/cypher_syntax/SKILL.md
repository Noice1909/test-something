---
name: Cypher Query Syntax Reference
description: Comprehensive Neo4j Cypher syntax patterns for generating correct, production-quality graph queries. Load this skill when generating or debugging Cypher queries.
---

# Cypher Query Syntax Reference

Use this reference whenever you need to **generate**, **fix**, or **debug** Neo4j Cypher queries. It covers every common pattern with correct syntax and highlights frequent mistakes.

---

## Basic Node Matching

```cypher
-- Match all nodes with a label
MATCH (n:Movie) RETURN n.title, n.year LIMIT 25

-- Match by property
MATCH (n:Movie {title: "Inception"}) RETURN n.title, n.year

-- Match with WHERE clause
MATCH (n:Movie) WHERE n.year >= 2010 RETURN n.title, n.year LIMIT 25

-- Match with parameter
MATCH (n:Movie {title: $title}) RETURN n.title, n.year
```

---

## Relationship Traversal

```cypher
-- Outgoing relationship
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name, m.title

-- Incoming relationship
MATCH (m:Movie)<-[:ACTED_IN]-(p:Person)
RETURN m.title, p.name

-- Either direction
MATCH (a:Person)-[:KNOWS]-(b:Person)
RETURN a.name, b.name

-- Multi-hop traversal
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Person)
RETURN p.name AS actor, m.title AS movie, d.name AS director

-- Variable-length path (1 to 3 hops)
MATCH (a:Person)-[*1..3]-(b:Person)
RETURN DISTINCT a.name, b.name LIMIT 25
```

---

## Counting & Aggregation

### Simple count
```cypher
MATCH (n:Movie) RETURN count(n) AS total
```

### Count with grouping
```cypher
-- Count movies per genre
MATCH (n:Movie)
RETURN n.genre AS genre, count(n) AS count
ORDER BY count DESC
LIMIT 10
```

### Count relationships per node
```cypher
-- Top 5 actors by movie count
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, count(m) AS movie_count
ORDER BY movie_count DESC
LIMIT 5
```

### Finding the MAX (who has the most)
```cypher
-- Director with the most movies
MATCH (p:Person)-[:DIRECTED]->(m:Movie)
RETURN p.name AS director, count(m) AS movie_count
ORDER BY movie_count DESC
LIMIT 1
```

### Sum / Average
```cypher
MATCH (m:Movie) WHERE m.revenue IS NOT NULL
RETURN avg(m.revenue) AS avg_revenue, sum(m.revenue) AS total_revenue

MATCH (m:Movie)
RETURN m.genre, avg(m.rating) AS avg_rating
ORDER BY avg_rating DESC
```

### collect() — aggregate into a list
```cypher
-- List all movies per actor
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
RETURN p.name AS actor, collect(m.title) AS movies
ORDER BY size(collect(m.title)) DESC
LIMIT 10
```

---

## Filtering

```cypher
-- String contains (case-insensitive)
MATCH (n:Movie)
WHERE toLower(n.title) CONTAINS toLower($term)
RETURN n.title LIMIT 25

-- Starts with
MATCH (n:Person)
WHERE n.name STARTS WITH $prefix
RETURN n.name

-- Regular expression
MATCH (n:Person)
WHERE n.name =~ '(?i).*tom.*'
RETURN n.name

-- IN list
MATCH (m:Movie) WHERE m.genre IN ["Action", "Sci-Fi"]
RETURN m.title, m.genre

-- NOT / negation
MATCH (p:Person)
WHERE NOT (p)-[:DIRECTED]->(:Movie)
RETURN p.name

-- IS NOT NULL
MATCH (m:Movie)
WHERE m.rating IS NOT NULL
RETURN m.title, m.rating
ORDER BY m.rating DESC LIMIT 10
```

---

## WITH Clause (chaining & sub-queries)

```cypher
-- Filter after aggregation
MATCH (p:Person)-[:ACTED_IN]->(m:Movie)
WITH p, count(m) AS movie_count
WHERE movie_count > 2
RETURN p.name, movie_count
ORDER BY movie_count DESC

-- Pass computed values downstream
MATCH (m:Movie)
WITH avg(m.rating) AS avgRating
MATCH (top:Movie)
WHERE top.rating > avgRating
RETURN top.title, top.rating
ORDER BY top.rating DESC
```

---

## OPTIONAL MATCH (left join)

```cypher
-- Return person even if they haven't directed anything
MATCH (p:Person {name: $name})
OPTIONAL MATCH (p)-[:DIRECTED]->(m:Movie)
RETURN p.name, collect(m.title) AS directed_movies
```

---

## DISTINCT

```cypher
MATCH (m:Movie)
RETURN DISTINCT m.genre AS genre
ORDER BY genre

MATCH (a)-[r]->(b)
RETURN DISTINCT labels(a)[0] AS from, type(r) AS rel, labels(b)[0] AS to
```

---

## UNWIND

```cypher
-- Process a list of values
UNWIND $names AS name
MATCH (p:Person {name: name})
RETURN p.name, p.born
```

---

## Shortest Path

```cypher
MATCH path = shortestPath(
  (a:Person {name: $from})-[*..15]-(b:Person {name: $to})
)
RETURN [n IN nodes(path) | coalesce(n.name, n.title)] AS route
```

---

## Schema Introspection

```cypher
CALL db.labels()
CALL db.relationshipTypes()
CALL db.propertyKeys()
CALL db.schema.visualization()
```

---

## Returning Properties (not raw nodes)

> **IMPORTANT**: Always return specific properties, never raw node references.

```cypher
-- ✅ CORRECT
MATCH (m:Movie) RETURN m.title, m.year, m.genre LIMIT 25

-- ❌ WRONG — returns opaque node objects
MATCH (m:Movie) RETURN m LIMIT 25
```

---

## Common Mistakes to AVOID

| ❌ Mistake | ✅ Correct |
|---|---|
| `RETURN n` (raw node) | `RETURN n.title, n.year` |
| Missing `LIMIT` | Always add `LIMIT 25` (or appropriate) |
| Wrong direction `(a)-[:R]->(b)` when schema says `(b)-[:R]->(a)` | Check schema patterns first |
| Using labels not in schema | Only use labels from the schema |
| `WHERE n.prop = null` | `WHERE n.prop IS NULL` |
| `count(n.name)` to count nodes | `count(n)` to count nodes |
| Forgetting `ORDER BY` with aggregation | Add `ORDER BY aggregate DESC` |
| `MATCH (n) WHERE n:Label` | `MATCH (n:Label)` is cleaner |

---

## Strategy-Specific Tips

### DIRECT queries
Keep it simple — single `MATCH` with direct property lookup. No aggregation needed.

### ONE_HOP queries
Use one relationship traversal: `(a)-[:REL]->(b)`.

### MULTI_HOP queries
Chain multiple relationships: `(a)-[:R1]->(b)-[:R2]->(c)`. Use `OPTIONAL MATCH` when a hop may not exist.

### AGGREGATION queries
Use `count()`, `collect()`, `avg()`, `sum()` with grouping in `RETURN`. Always add `ORDER BY aggregate DESC` and `LIMIT`. For "who has the most" questions, use `ORDER BY count DESC LIMIT 1`.

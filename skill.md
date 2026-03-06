# Neo4j NL Agent — Skill Guide

## Overview

This agent converts natural language questions into Cypher queries, executes them against a Neo4j database, and returns answers in plain English. It is **fully generic** — it dynamically discovers schema, indexes, and metadata at startup, so it works with any Neo4j database without modification.

**Target accuracy:** 80-90% on well-covered domains.

---

## How It Works

### The 7-Step Pipeline

Every question flows through a LangGraph state machine with 10 nodes and a validation-correction loop:

```
Question (+ thread_id for conversation memory)
     │
     ▼
┌────────────────────────┐
│ 1. CONCEPT MATCHING    │  Match question words against :Concept.nlp_terms
│                        │  (auto-detected; skipped if no Concept nodes exist)
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 2. ENTITY EXTRACTION   │  LLM extracts named entities from the question
│                        │  using discovered schema as context
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 3. ENTITY MAPPING      │  Fuzzy-match entities against actual DB values
│                        │  using dynamically discovered FULLTEXT indexes
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 4. SCHEMA FILTERING    │  Filter full schema to only relevant labels,
│                        │  relationships, and properties
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 5. FEW-SHOT RETRIEVAL  │  Retrieve top-K similar example queries from
│                        │  ChromaDB vector store
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 6. CYPHER GENERATION   │  LLM generates Cypher using: filtered schema +
│                        │  resolved entities + few-shot examples
└───────────┬────────────┘
            ▼
┌────────────────────────┐         ┌───────────────────┐
│ 7. VALIDATION          │──FAIL──▶│ SELF-CORRECTION   │
│   • Write-block        │         │ Feed error + schema│
│   • Schema check       │◀─RETRY──│ back to LLM       │
│   • EXPLAIN test       │         │ (max 3 retries)   │
└───────────┬────────────┘         └───────────────────┘
          PASS
            ▼
┌────────────────────────┐
│ 8. EXECUTE QUERY       │  Run validated Cypher against Neo4j
└───────────┬────────────┘
            ▼
┌────────────────────────┐
│ 9. GENERATE RESPONSE   │  Convert raw results into a natural
│                        │  language answer
└────────────────────────┘
```

### Why Each Step Matters

| Step | Accuracy Impact | What It Prevents |
|------|----------------|------------------|
| Concept Matching | +15-20% | Label hallucination — LLM inventing labels that don't exist |
| Entity Extraction | +5% | Missing or wrong entity references in WHERE clauses |
| Entity Mapping (fulltext fuzzy) | +10-15% | Correct Cypher but wrong entity names (the #1 failure mode) |
| Schema Filtering | +5-10% | Token waste and confusion from irrelevant schema elements |
| Few-Shot Examples | +15-23% | Wrong query patterns — LLM learns correct patterns from examples |
| Validation + Retry | +10-15% | Syntax errors, wrong labels/rels, missing LIMIT |

---

## Dynamic Discovery

Everything is discovered at startup. Nothing is hardcoded.

### Schema Discovery

On startup, `SchemaService` runs:
- `CALL apoc.meta.schema()` — gets all labels, relationships, properties, and their types
- `CALL db.labels()` — confirms available labels
- `CALL db.relationshipTypes()` — confirms available relationship types
- Samples data per label to detect actual property usage

The schema is cached with a configurable TTL (`SCHEMA_CACHE_TTL`, default: 300s). Refresh it anytime via `POST /api/schema/refresh`.

### Index Discovery

`IndexService` runs `SHOW INDEXES WHERE type='FULLTEXT'` and categorizes:
- **Per-label indexes**: cover a single label (e.g., `SORObjectNameIndex` covers `SORObject.name`)
- **Global indexes**: cover many labels (e.g., `globalNameIndex` covers the `name` property across 80%+ of labels)
- **Special indexes**: multi-property indexes (e.g., `faqTextIndex` covers `FAQ.question` and `FAQ.answer`)

The agent automatically selects the most specific index for each entity lookup.

### Concept Discovery

If the database has `:Concept` label nodes, `ConceptService` loads them and builds an in-memory lookup from `nlp_terms`. If no `:Concept` nodes exist, concept matching is silently skipped.

Concept nodes should have:
- `name`: the label name this concept maps to
- `nlp_terms`: comma-separated natural language terms (e.g., "SOR, source, source data, data source")
- `description`: what this concept represents

---

## API Endpoints

### POST /api/ask — Ask a question

The main endpoint. Sends a natural language question through the full pipeline.

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What applications are in the Finance domain?"}'
```

**Request body:**
```json
{
  "question": "What applications are in the Finance domain?",
  "thread_id": null
}
```

- `question` (required): natural language question (1-2000 chars)
- `thread_id` (optional): conversation thread ID for memory. If `null`, a new thread is created. Pass the returned `thread_id` in follow-up questions for conversation context.

**Response:**
```json
{
  "answer": "There are 12 applications in the Finance domain, including...",
  "thread_id": "a1b2c3d4-...",
  "success": true
}
```

### GET /api/schema — View discovered schema

Returns the dynamically discovered schema.

```bash
curl http://localhost:8000/api/schema
```

### POST /api/schema/refresh — Force schema re-discovery

Triggers re-discovery of schema, indexes, and concepts. Use after database schema changes.

```bash
curl -X POST http://localhost:8000/api/schema/refresh
```

### GET /api/health — Health check

Returns system health including Neo4j connectivity, Ollama availability, circuit breaker states, and checkpointer type.

```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "neo4j_connected": true,
  "ollama_available": true,
  "environment": "local",
  "checkpointer": "sqlite",
  "circuit_breakers": {
    "neo4j": "closed",
    "ollama": "closed"
  }
}
```

---

## Conversation Memory

The agent uses LangGraph checkpointer for conversation memory:

- **Local**: SQLite at `data/checkpoints/agent.db`
- **Deployed**: Redis via `REDIS_URL`

**How it works:**
1. First request: omit `thread_id` or send `null` — a new UUID is generated
2. Response includes `thread_id`
3. Follow-up: send the same `thread_id` — the agent has full context of prior Q&A

```bash
# First question
curl -X POST http://localhost:8000/api/ask \
  -d '{"question": "Show me all SOR objects"}'
# Response: {"thread_id": "abc-123", ...}

# Follow-up using same thread
curl -X POST http://localhost:8000/api/ask \
  -d '{"question": "Which of those are in the Finance domain?", "thread_id": "abc-123"}'
```

---

## MCP Tools

Seven tools are available via the MCP (Model Context Protocol) server for external agents:

### 1. get_schema
Returns the full or filtered graph schema.

**Parameters:**
- `filter_labels` (optional, array of strings): labels to filter schema to

**Returns:** `schema_text`, `labels`, `relationship_types`

### 2. search_concepts
Search `:Concept` nodes by natural language terms.

**Parameters:**
- `query` (string): search terms

**Returns:** List of matched concepts with `name`, `nlp_terms`, `description`

### 3. fuzzy_search_global
Full-text search across all labels using the global name index.

**Parameters:**
- `term` (string): search term (fuzzy matching with `~` is automatic)
- `limit` (int, default 5): max results

**Returns:** Matching nodes with `name`, `id`, `label`, `score`

### 4. fuzzy_search_by_label
Full-text search scoped to a specific label using its per-label index.

**Parameters:**
- `label` (string): node label to search within
- `term` (string): search term
- `limit` (int, default 5): max results

**Returns:** Matching nodes with `name`, `id`, `label`, `score`

### 5. execute_cypher
Execute a read-only Cypher query. Write operations are blocked.

**Parameters:**
- `cypher` (string): the Cypher query

**Returns:** `results` (list of records), `count`

### 6. validate_cypher
Validate a Cypher query against the discovered schema without executing.

**Parameters:**
- `cypher` (string): the Cypher query to validate

**Returns:** `valid` (bool), `cypher` (potentially modified), `errors` (list)

### 7. list_indexes
List all discovered FULLTEXT indexes.

**Returns:** `fulltext_by_label`, `global_indexes`, `special_indexes`

### Running the MCP Server

The MCP server runs in STDIO mode for integration with external tools:

```bash
python -m src.mcp.server
```

---

## Configuration

All configuration is via `.env` file (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `local` | `local` (SQLite checkpointer) or `deployed` (Redis) |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | — | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OLLAMA_TEMPERATURE` | `0.0` | Temperature for Cypher generation |
| `SCHEMA_CACHE_TTL` | `300` | Schema cache duration in seconds |
| `FEW_SHOT_K` | `5` | Number of similar examples to retrieve |
| `MAX_CYPHER_RETRIES` | `3` | Max validation-correction cycles |
| `RATE_LIMIT_ASK` | `10/minute` | Rate limit for /api/ask |
| `RATE_LIMIT_DEFAULT` | `30/minute` | Rate limit for other endpoints |
| `CB_FAIL_MAX` | `5` | Circuit breaker: failures before opening |
| `CB_RESET_TIMEOUT` | `30` | Circuit breaker: seconds before retry |
| `SQLITE_CHECKPOINT_PATH` | `data/checkpoints/agent.db` | SQLite DB path (local mode) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis URL (deployed mode) |

---

## Few-Shot Examples

The accuracy booster with the highest impact. Located in `few_shot_examples.yml`.

### How They Work

1. On startup, all examples are embedded into ChromaDB using cosine similarity
2. For each incoming question, the top-K most similar examples are retrieved
3. These examples are injected into the Cypher generation prompt so the LLM sees correct patterns

### Writing Good Examples

```yaml
examples:
  - question: "What applications are in the Finance domain?"
    cypher: |
      MATCH (a:Application)-[:BELONGS_TO]->(d:Domain)
      WHERE toLower(d.name) CONTAINS toLower("finance")
      RETURN a.name AS application, d.name AS domain
      LIMIT 25

  - question: "How many SOR objects are there?"
    cypher: |
      MATCH (s:SORObject)
      RETURN count(s) AS total

  - question: "Show me the path from app X to its data sources"
    cypher: |
      MATCH path = (a:Application)-[:USES*1..3]->(s:SORObject)
      WHERE toLower(a.name) CONTAINS toLower($entity)
      RETURN [n IN nodes(path) | n.name] AS path_names
      LIMIT 25
```

### Tips for Maximum Accuracy

1. **Cover each query pattern 2-3 times** with variations:
   - Simple lookups (MATCH ... WHERE ... RETURN)
   - Aggregations (COUNT, SUM, AVG)
   - Multi-hop traversals (variable-length paths)
   - Reverse traversals (find parents, upstream)
   - Top-N queries (ORDER BY ... DESC LIMIT)

2. **Use exact property names** from your schema — different labels may use different property names

3. **Use `$entity` as placeholder** for user-specified values

4. **Always include LIMIT** — the validator will add one if missing, but explicit is better

5. **Include relationship directions** — this is the most common LLM mistake

6. **Cover fuzzy search patterns** — show `toLower(x) CONTAINS toLower(y)` usage

---

## Production Features

### Circuit Breakers

Two independent circuit breakers protect against cascading failures:

- **Neo4j breaker**: opens after `CB_FAIL_MAX` consecutive Neo4j failures
- **Ollama breaker**: opens after `CB_FAIL_MAX` consecutive Ollama failures

When a circuit is open:
- Calls fail immediately (no waiting for timeout)
- Health endpoint reports the open state
- After `CB_RESET_TIMEOUT` seconds, one probe request is allowed
- If the probe succeeds, the circuit closes

### Rate Limiting

SlowAPI rate limiting per client IP:
- `/api/ask`: `RATE_LIMIT_ASK` (default: 10/minute)
- Other endpoints: `RATE_LIMIT_DEFAULT` (default: 30/minute)
- Returns HTTP 429 with a friendly JSON message when exceeded

### Structured Logging

Uses `structlog` for structured, contextual logging:
- **Local**: human-readable console output
- **Deployed**: JSON format for log aggregation
- Every request gets a unique `X-Request-ID` header (pass your own or one is generated)
- All pipeline steps log their input/output for debugging

### Request Tracing

Every request is tagged with a `request_id` via middleware:
- Injected into all log entries via structlog contextvars
- Returned in the `X-Request-ID` response header
- Pass `X-Request-ID` in request headers to use your own trace ID

---

## Validation Pipeline

The 3-layer validation catches most errors before execution:

### Layer 1: Deterministic Checks (instant, no DB call)
- **Write-block**: rejects any query containing CREATE, MERGE, DELETE, SET, REMOVE, DROP
- **Valid start**: query must start with MATCH, OPTIONAL, WITH, CALL, UNWIND, or RETURN
- **RETURN exists**: ensures the query returns data
- **Label check**: every label in the query must exist in the discovered schema
- **Relationship check**: every relationship type must exist in the discovered schema

### Layer 2: EXPLAIN Test (requires DB call)
- Runs `EXPLAIN <cypher>` against Neo4j
- Catches syntax errors, invalid references, and planning failures
- Does not execute the query

### Layer 3: Self-Correction (LLM-powered, up to MAX_CYPHER_RETRIES)
- When validation fails, the error message + original query + schema are sent back to the LLM
- The LLM attempts to fix the query
- The corrected query goes through validation again
- After `MAX_CYPHER_RETRIES` failures, the agent returns a friendly "could not generate query" message

### Auto-Injected LIMIT
If the generated query has no LIMIT clause, `LIMIT 25` is automatically appended.

---

## Project Structure

```
src/
├── main.py                  # FastAPI app factory + lifespan
├── config.py                # Pydantic Settings from .env
├── models.py                # Request/response models
│
├── core/
│   ├── lifespan.py          # Startup/shutdown orchestration
│   ├── rate_limiter.py      # SlowAPI setup
│   ├── circuit_breaker.py   # pybreaker instances
│   ├── checkpointer.py      # SQLite/Redis factory
│   ├── logging.py           # structlog configuration
│   ├── middleware.py         # Request ID + timing
│   └── exceptions.py        # Custom exceptions + handlers
│
├── agent/
│   ├── graph.py             # LangGraph StateGraph definition
│   ├── state.py             # AgentState TypedDict
│   └── nodes.py             # All graph node functions
│
├── services/
│   ├── neo4j_service.py     # Neo4j driver wrapper
│   ├── schema_service.py    # Dynamic schema discovery
│   ├── concept_service.py   # :Concept node loader
│   ├── index_service.py     # FULLTEXT index discovery
│   ├── entity_extractor.py  # Entity extraction + mapping
│   ├── cypher_generator.py  # LLM Cypher generation
│   ├── cypher_validator.py  # 3-layer validation
│   ├── few_shot_service.py  # ChromaDB few-shot retrieval
│   └── response_generator.py # Results → NL answer
│
├── mcp/
│   ├── server.py            # MCP server setup
│   └── tools.py             # 7 MCP tool definitions
│
├── prompts/
│   ├── cypher_prompt.py     # Cypher generation prompt
│   ├── entity_prompt.py     # Entity extraction prompt
│   ├── correction_prompt.py # Error self-correction prompt
│   └── response_prompt.py   # Answer generation prompt
│
└── routers/
    ├── query.py             # POST /api/ask
    ├── schema.py            # GET /api/schema
    └── health.py            # GET /api/health
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Neo4j credentials and Ollama model

# 3. Ensure Ollama is running
ollama serve
ollama pull llama3.1    # or your chosen model

# 4. Start the server
uvicorn src.main:app --reload --port 8000

# 5. Check health
curl http://localhost:8000/api/health

# 6. Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What nodes are in this database?"}'
```

---

## Tuning Guide

### Improving Accuracy

1. **Add more few-shot examples** — this has the highest ROI. Cover every query pattern you expect users to ask. Aim for 20-50 examples.

2. **Add :Concept nodes** — if your users use domain-specific language (e.g., "SOR" means `SORObject`), create Concept nodes with `nlp_terms` that map natural language to labels.

3. **Create FULLTEXT indexes** — the entity mapper uses them for fuzzy matching. At minimum, create:
   - A global name index: `CREATE FULLTEXT INDEX globalNameIndex FOR (n:Label1|Label2|...) ON EACH [n.name]`
   - Per-label indexes for labels that users frequently search

4. **Increase MAX_CYPHER_RETRIES** — more retries = more chances to self-correct, but also more latency. Default of 3 is a good balance.

5. **Use a better Ollama model** — larger models generate better Cypher. Try `llama3.1:70b` or `codellama:34b` if you have the hardware.

6. **Lower OLLAMA_TEMPERATURE** — 0.0 is best for Cypher generation (deterministic). The response generator uses 0.3 for more natural language.

### Reducing Latency

1. **Decrease FEW_SHOT_K** — fewer examples = faster prompt, but potentially less accurate. Try K=3 for speed.

2. **Decrease SCHEMA_CACHE_TTL** only if schema changes frequently. Otherwise keep it high (300-600s).

3. **Use a faster Ollama model** — smaller models (7B) are faster but less accurate. Find your balance.

4. **Decrease MAX_CYPHER_RETRIES** — set to 1 or 2 for faster failure.

### Debugging

1. Check startup logs — they show what was discovered (indexes, labels, concepts)
2. Check request logs — each step logs its output
3. Use `X-Request-ID` header to trace a request through logs
4. Use the MCP tools directly to test individual components:
   - `get_schema` — see what the agent sees
   - `search_concepts` — test concept matching
   - `fuzzy_search_global` — test entity resolution
   - `validate_cypher` — test a specific query

---

## Failure Modes and Recovery

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "I wasn't able to find any information" | Entity not resolved, or wrong label used | Add few-shot examples covering this pattern |
| Wrong entity matched | Fuzzy search returning wrong result | Create a more specific per-label FULLTEXT index |
| Wrong relationship direction | LLM guessed direction | Add few-shot example with correct direction |
| "I had trouble looking that up" | Cypher execution error (e.g., property doesn't exist on label) | Check logs, add property-per-label examples |
| Circuit breaker open | Neo4j or Ollama down | Check service health, wait for `CB_RESET_TIMEOUT` |
| 429 Too Many Requests | Rate limit exceeded | Wait, or adjust `RATE_LIMIT_ASK` |
| Labels/rels not in schema | Schema cache stale | Call `POST /api/schema/refresh` |
| Wrong label chosen | No Concept nodes for this domain term | Add a `:Concept` node with `nlp_terms` |

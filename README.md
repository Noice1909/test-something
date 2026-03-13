# Agentic Graph Query System

> Autonomous multi-specialist AI system for querying Neo4j graph databases.

## Architecture

> [!NOTE]
> Read the detailed technical architecture, agent workflow, and system autonomy evaluation in [ARCHITECTURE.md](ARCHITECTURE.md) along with visual diagrams.

```
User Question → Supervisor (LLM) → Specialist Agents → Neo4j → Answer
```

The **Supervisor** analyzes each question, picks a strategy, coordinates
specialist agents, and retries on failure — all driven by LLM reasoning.


### Specialists

| Specialist | Role |
|---|---|
| **Discovery** | Search DB for entities (exact, fuzzy, label match) |
| **Schema Reasoning** | LLM selects relevant node/edge types |
| **Query Planning** | Decide DIRECT / ONE_HOP / MULTI_HOP / AGGREGATION |
| **Query Generation** | LLM creates parameterized Cypher — guided by [Cypher Syntax Skill](#cypher-syntax-skill) |
| **Execution** | Run query with safety checks + error categorization |
| **Reflection** | Analyze failures, recommend retry strategy |

### Strategies (auto-selected by LLM)

- `discovery_first` — unknown terms / acronyms → search DB first
- `direct_query` — clear entity references → skip discovery
- `schema_exploration` — structure / relationship questions
- `aggregation` — count / sum / avg queries

## Quick Start

### 1. Configure

```bash
cp .env.example .env
# Edit .env with your Neo4j and LLM settings
```

Key environment variables:

| Variable | Description | Default |
|---|---|---|
| `NEO4J_URI` | Neo4j connection URI | `neo4j+s://localhost:7687` |
| `NEO4J_USER` | Username | `neo4j` |
| `NEO4J_PASSWORD` | Password | — |
| `NEO4J_DATABASE` | Database name | `neo4j` |
| `NEO4J_SKIP_TLS_VERIFY` | Skip TLS cert verification (for AuraDB) | `False` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama3.2` |
| `OPENAI_API_KEY` | OpenAI API key (uses OpenAI if set) | — |

### 2. Install

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### 3. Run

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Test

```bash
# Unit tests
pytest tests/ -v

# Health check
curl http://localhost:8001/api/v1/agentic/health

# Ask a question
curl -X POST http://localhost:8001/api/v1/agentic/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many movies are in the database?"}'

# Batch test (10 questions covering all strategies)
python test_agentic_batch.py
```

### 5. Type Checking

```bash
pip install pyright
pyright src/    # Should return 0 errors
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/agentic/chat` | Submit a question |
| `GET` | `/api/v1/agentic/health` | Health check |
| `GET` | `/health` | Service health |

### Chat Request

```json
{
  "question": "Which director has directed the most movies?"
}
```

### Chat Response

```json
{
  "answer": "Christopher Nolan has directed the most movies with 5 films.",
  "strategy_used": "aggregation",
  "attempts": 1,
  "success": true,
  "trace_id": "agentic-abc12345",
  "specialist_log": [...]
}
```

## Cypher Syntax Skill

The query generation specialist loads a comprehensive Cypher reference
from `.agent/skills/cypher_syntax/SKILL.md` at startup. This teaches the
LLM correct query patterns for:

- Counting & aggregation (`count`, `ORDER BY ... DESC LIMIT 1`)
- Relationship traversal (single-hop, multi-hop)
- Filtering (`WHERE`, `CONTAINS`, `IN`, `IS NOT NULL`)
- `WITH` clause chaining
- `OPTIONAL MATCH`, `DISTINCT`, `collect()`

The skill also enforces critical rules like:
- **Never invent labels** — only use labels from the actual schema
- **Roles like Director/Actor are relationships, not labels** (use `(p:Person)-[:DIRECTED]->`)
- **Always end with RETURN** (never end a query with `WITH`)

## File Structure

```
src/
├── config.py                    # Pydantic settings
├── main.py                      # FastAPI app
├── agents/
│   ├── base.py                  # Enums + data structures
│   ├── state.py                 # Agent state management
│   ├── supervisor.py            # LLM orchestrator
│   ├── supervisor_factory.py    # Singleton factory
│   ├── utils.py                 # extract_text() helper
│   └── specialists/
│       ├── discovery.py
│       ├── schema_reasoning.py
│       ├── query_planning.py
│       ├── query_generation.py  # Loads Cypher syntax skill
│       ├── execution.py
│       └── reflection.py
├── database/
│   ├── abstract.py              # DB interface
│   └── neo4j.py                 # Neo4j async driver (TLS, retry, reconnect)
├── llm/
│   ├── __init__.py              # Re-exports create_llm, get_llm
│   └── provider.py             # Centralised LLM factory (Ollama / OpenAI)
├── tools/                       # ~80 Neo4j tools
│   ├── schema_discovery.py
│   ├── graph_exploration.py
│   ├── graph_search.py
│   ├── aggregation.py
│   ├── data_inspection.py
│   ├── query_execution.py
│   └── metadata.py
└── api/routes/
    └── agentic.py               # FastAPI routes

.agent/skills/cypher_syntax/
    └── SKILL.md                 # Cypher syntax reference (loaded at runtime)

tests/
├── test_base.py
└── test_state.py
```

## Tech Stack

**FastAPI** · **LangChain** · **Ollama/OpenAI** · **Neo4j** · **Pydantic** · **Pyright**

## License

MIT

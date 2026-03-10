# Agentic Graph Query System

> Autonomous multi-specialist AI system for querying Neo4j graph databases.

## Architecture

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
| **Query Generation** | LLM creates parameterized Cypher query |
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
# Edit .env with your Neo4j and Ollama settings
```

### 2. Install

```bash
pip install -e ".[dev]"
```

### 3. Run

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Test

```bash
# Health check
curl http://localhost:8001/api/v1/agentic/health

# Ask a question
curl -X POST http://localhost:8001/api/v1/agentic/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "show CNAPP solutions"}'

# Batch test
python test_agentic_batch.py
```

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/agentic/chat` | Submit a question |
| `GET` | `/api/v1/agentic/health` | Health check |
| `GET` | `/health` | Service health |

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
│   └── specialists/
│       ├── discovery.py
│       ├── schema_reasoning.py
│       ├── query_planning.py
│       ├── query_generation.py
│       ├── execution.py
│       └── reflection.py
├── database/
│   ├── abstract.py              # DB interface
│   └── neo4j.py                 # Neo4j driver
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

tests/
├── test_base.py
└── test_state.py
```

## Tech Stack

**FastAPI** · **LangChain** · **Ollama/OpenAI** · **Neo4j** · **Pydantic**

## License

MIT

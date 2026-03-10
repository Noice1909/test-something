# CLAUDE.md — Project Conventions

## Architecture

- **Supervisor pattern**: LLM-based orchestrator that selects strategies and coordinates specialist agents
- **Specialist agents**: Each specialist has a single responsibility (`run(state) → SpecialistResult`)
- **Tool registry**: All 80+ Neo4j tools are registered in `src/tools/__init__.py` as `TOOL_REGISTRY`
- **State threading**: `AgentState` is created per-question and passed through all specialists

## Code Conventions

- Python 3.11+, fully typed with `from __future__ import annotations`
- Async-first — all DB and LLM calls are `async`
- Pydantic for settings (`src/config.py`) and API models
- Dataclasses for internal data structures (`src/agents/base.py`)

## Key Rules

1. **Read-only**: All generated Cypher queries are validated for read-only operations before execution
2. **Max attempts**: Queries retry up to `AGENTIC_MAX_ATTEMPTS` times with reflection-based adjustments
3. **LLM responses**: Always parsed with JSON extraction and fallback logic
4. **Neo4j tools**: Plain async functions taking `(db, **kwargs)` — no OOP overhead

## File Organization

- `src/agents/` — Core agent logic (supervisor, specialists, state)
- `src/database/` — Database abstraction (ABC + Neo4j impl)
- `src/tools/` — All Neo4j interaction tools grouped by category
- `src/api/` — FastAPI routes
- `tests/` — pytest unit tests

# Agent Orchestrator

A dynamic multi-agent orchestrator that replicates Claude Code's architecture. Pre-configured for Neo4j knowledge graphs, but fully extensible to any domain via dynamic SKILL.md and AGENT.md discovery.

## Features

- **TAOR Agentic Loop** (Think-Act-Observe-Repeat) — ~60 lines of core logic, all intelligence in the LLM
- **Dynamic Skill Discovery** — drop any SKILL.md into `.skills/` and it's auto-discovered
- **Sub-Agent Spawning** — isolated contexts, filtered tools, LLM-based routing
- **LangChain Multi-Provider** — works with Anthropic, OpenAI, Ollama, or any LangChain-supported LLM
- **MCP Support** — connect external Model Context Protocol servers as tools
- **FastAPI Server** — REST + SSE streaming endpoints

## Pre-configured Neo4j Skills (17)

Skills are organized into 6 categories:

- **schema/** — explore-schema, relationship-patterns, indexes-constraints
- **search/** — fulltext-search, fuzzy-search, property-search, regex-search
- **explore/** — node-details, neighbors, find-paths, sample-subgraph
- **query/** — cypher-read, aggregation, pattern-match
- **analysis/** — topology-analysis, compare-entities, graph-statistics
- **utility/** — explain-results, validate-query

## Pre-configured Agents (3)

- **discovery-agent** — Multi-step entity search + relationship mapping
- **query-agent** — Complex Cypher generation with schema awareness
- **analyst-agent** — Data analysis pipeline with insights

## Quick Start

### 1. Install Dependencies

```bash
cd agent-orchestrator
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env`:

```env
# LLM
AGENT_LLM_PROVIDER=anthropic
AGENT_LLM_MODEL=claude-sonnet-4-5-20250929
AGENT_LLM_API_KEY=your-api-key-here

# Neo4j
AGENT_NEO4J_URI=bolt://localhost:7687
AGENT_NEO4J_USER=neo4j
AGENT_NEO4J_PASSWORD=your-password
AGENT_NEO4J_DATABASE=neo4j
```

### 3. Run the Server

```bash
python app.py
```

Server starts at `http://localhost:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /query` | POST | Send a message, get response |
| `POST /query/stream` | POST | SSE streaming (real-time events) |
| `GET /skills` | GET | List all discovered skills |
| `GET /agents` | GET | List all sub-agents |
| `POST /skills/reload` | POST | Hot-reload skills + agents from filesystem |
| `GET /health` | GET | Health check (includes Neo4j connectivity) |

## Usage Examples

### Simple Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What labels exist in the graph?"}'
```

### Slash Command

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "/explore-schema"}'
```

### Session Continuation

```bash
# First message
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Find companies"}' \
  | jq -r '.session_id' > session.txt

# Follow-up in same session
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Show me their employees\", \"session_id\": \"$(cat session.txt)\"}"
```

## Adding Custom Skills

1. Create a new directory in `.skills/`:

```bash
mkdir -p .skills/custom/my-skill
```

2. Create `SKILL.md`:

```yaml
---
name: my-skill
description: What this skill does and when to use it
allowed-tools: RunCypherTool GetSchemaTool
context: fork
argument-hint: [optional-args]
---

You are a specialist in...

Instructions for the LLM when this skill is invoked.
```

3. Reload skills:

```bash
curl -X POST http://localhost:8000/skills/reload
```

The skill is now available! The orchestrator's LLM will automatically invoke it when a user request matches the description.

## Adding Custom Agents

Same pattern as skills, but in `.agents/`:

```yaml
---
name: my-agent
description: What this agent does
tools: RunCypherTool, SearchNodesTool
max-turns: 20
---

You are a specialized agent that...

System prompt for this agent.
```

## Dynamic Context Injection (DCI)

Skills support Dynamic Context Injection — shell commands that run before the LLM sees the skill:

```markdown
Current timestamp: !`date`
Git branch: !`git rev-parse --abbrev-ref HEAD`
```

When the skill is invoked, `!`command`` is replaced with the command's stdout.

## Argument Substitution

Skills support argument substitution:

```markdown
You are searching for: $ARGUMENTS
First argument: $0
Second argument: $1
```

When invoked with `/my-skill foo bar`, these are replaced with actual values.

## Architecture

```
User Query
    ↓
Orchestrator (builds system prompt with skill/agent descriptions)
    ↓
TAOR Loop: LLM → Tool Calls → Results → Repeat
    ↓
LLM decides: handle directly | invoke_skill | delegate_to_agent
    ↓
If delegate → SubAgentSpawner (fresh context, filtered tools)
    ↓
Sub-agent runs own TAOR loop
    ↓
Returns final text to parent
```

**Key insight**: All routing is LLM-based reasoning. No classifiers, no embeddings, no rigid rules. The LLM reads skill/agent descriptions in the system prompt and naturally decides what to invoke.

## MCP Server Integration

Add external MCP servers in `.env`:

```env
AGENT_MCP_SERVERS='{"my-server": {"command": "python", "args": ["mcp_server.py"]}}'
```

Their tools are auto-discovered and registered with the prefix `mcp__my-server__toolname`.

## License

MIT

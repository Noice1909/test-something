"""FastAPI application with lifespan — connects Neo4j, discovers skills/agents,
registers tools, and wires up the orchestrator.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is on sys.path (needed for uvicorn reload subprocess)
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import uvicorn
from fastapi import FastAPI
from neo4j import AsyncGraphDatabase

from api.routes import router
from config import Settings
from core.hooks import HookManager, HookType, safety_write_blocker, tool_call_logger
from core.orchestrator import Orchestrator
from discovery.registry import CapabilityRegistry
from tools.manager import ToolManager
from tools.orchestration import DelegateToAgentTool, InvokeSkillTool
from agents.spawner import SubAgentSpawner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM factory — creates LangChain ChatModel for any provider
# ─────────────────────────────────────────────────────────────────────────────


def _import_chat_class(provider: str):
    """Import the LangChain chat class for the given provider (once)."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def create_llm_factory(config: Settings):
    """Return a callable ``(model=None) -> BaseChatModel``."""
    # Import the class eagerly at factory creation (during startup)
    ChatClass = _import_chat_class(config.llm_provider)

    def factory(model: str | None = None):
        model_name = model or config.llm_model
        kwargs = {"temperature": config.llm_temperature, "max_tokens": config.max_tokens}

        if config.llm_provider == "ollama":
            return ChatClass(model=model_name, base_url=config.llm_base_url, **kwargs)  # type: ignore[arg-type]
        elif config.llm_provider == "anthropic" and config.enable_prompt_cache:
            # Enable Anthropic prompt caching via beta header
            kwargs["model_kwargs"] = {
                "extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}
            }
            return ChatClass(model=model_name, api_key=config.llm_api_key, **kwargs)  # type: ignore[arg-type]
        else:
            return ChatClass(model=model_name, api_key=config.llm_api_key, **kwargs)  # type: ignore[arg-type]

    return factory


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Settings()
    logger.info("Starting agent-orchestrator (provider=%s, model=%s)", config.llm_provider, config.llm_model)

    # ── Connect Neo4j ────────────────────────────────────────────────────
    neo4j_uri = config.neo4j_uri
    if config.neo4j_skip_tls_verify:
        # Rewrite scheme to skip TLS cert verification (same as test-neo project)
        if neo4j_uri.startswith("neo4j+s://"):
            neo4j_uri = "neo4j+ssc://" + neo4j_uri[len("neo4j+s://"):]
            logger.info("TLS cert verification DISABLED (neo4j+ssc://)")
        elif neo4j_uri.startswith("bolt+s://"):
            neo4j_uri = "bolt+ssc://" + neo4j_uri[len("bolt+s://"):]
            logger.info("TLS cert verification DISABLED (bolt+ssc://)")
    driver = AsyncGraphDatabase.driver(
        neo4j_uri, auth=(config.neo4j_user, config.neo4j_password),
    )
    logger.info("Connected to Neo4j: %s", config.neo4j_uri)

    # ── Discovery — scan filesystem for skills & agents ──────────────────
    registry = CapabilityRegistry()
    await registry.load_all(config.skills_dir, config.agents_dir)
    logger.info("Discovered %d skills, %d agents", len(registry.skills), len(registry.agents))

    # ── Tools — register Neo4j builtins + MCP servers ────────────────────
    tool_manager = ToolManager()
    tool_manager.register_neo4j_tools(driver, config.neo4j_database)
    await tool_manager.register_mcp_servers(config.mcp_servers)

    # ── LLM factory ────────────────────────────────────────────────────────
    llm_factory = create_llm_factory(config)

    # ── Hook manager ─────────────────────────────────────────────────────
    hook_manager = HookManager()
    hook_manager.register(HookType.PRE_TOOL_USE, safety_write_blocker, priority=0, name="safety_write_blocker")
    hook_manager.register(HookType.PRE_TOOL_USE, tool_call_logger, priority=10, name="tool_call_logger")
    logger.info("Registered %d hooks", sum(len(v) for v in hook_manager._hooks.values()))

    # ── Sub-agent spawner (with hooks) ───────────────────────────────────
    spawner = SubAgentSpawner(llm_factory, tool_manager, registry, hook_manager=hook_manager)

    # ── Register orchestration meta-tools ────────────────────────────────
    tool_manager.register(InvokeSkillTool(registry=registry, spawner=spawner))
    tool_manager.register(DelegateToAgentTool(registry=registry, spawner=spawner))

    logger.info("Registered %d tools total", len(tool_manager.tool_names))

    # ── Create orchestrator (with hooks) ─────────────────────────────────
    orchestrator = Orchestrator(llm_factory, registry, tool_manager, config, hook_manager=hook_manager)

    # Store in app state for routes to access
    app.state.orchestrator = orchestrator
    app.state.registry = registry
    app.state.driver = driver
    app.state.config = config
    app.state.hook_manager = hook_manager

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Shutting down...")
    await driver.close()
    if tool_manager.mcp_bridge:
        await tool_manager.mcp_bridge.disconnect_all()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agent Orchestrator",
    description=(
        "Dynamic multi-agent orchestrator with SKILL.md/AGENT.md discovery. "
        "Pre-configured for Neo4j knowledge graphs, extensible to any domain."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

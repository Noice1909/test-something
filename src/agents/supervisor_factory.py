"""Singleton factory for the Supervisor — used by FastAPI lifespan."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.supervisor import Supervisor
from src.config import settings
from src.database.neo4j import Neo4jDatabase
from src.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

_instance: Supervisor | None = None


class SupervisorFactory:
    """Creates and caches a single Supervisor instance."""

    @staticmethod
    async def create() -> Supervisor:
        global _instance
        if _instance is not None:
            return _instance

        # 1) Database
        db = Neo4jDatabase()
        await db.connect()

        # 2) LLM
        llm = _create_llm()

        # 3) Tools
        tools = TOOL_REGISTRY

        # 4) Supervisor
        supervisor = Supervisor(
            db=db,
            llm=llm,
            tools=tools,
            max_attempts=settings.agentic_max_attempts,
        )

        _instance = supervisor
        logger.info(
            "✓ Agentic system initialized (max_attempts=%d, tools=%d)",
            settings.agentic_max_attempts,
            len(tools),
        )
        return supervisor

    @staticmethod
    def get() -> Supervisor | None:
        """Return the cached instance, or None if not yet created."""
        return _instance

    @staticmethod
    async def shutdown() -> None:
        """Disconnect the database on app shutdown."""
        global _instance
        if _instance is not None:
            await _instance._db.disconnect()
            _instance = None
            logger.info("Agentic system shut down")


def _create_llm() -> Any:
    """Build a LangChain chat model from settings."""
    if settings.use_openai:
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI model: %s", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama

        logger.info("Using Ollama model: %s @ %s", settings.ollama_model, settings.ollama_base_url)
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

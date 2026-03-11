"""Singleton factory for the Supervisor — used by FastAPI lifespan."""

from __future__ import annotations

import logging

from src.agents.supervisor import Supervisor
from src.config import settings
from src.database.neo4j import Neo4jDatabase
from src.llm import create_llm
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

        # 2) LLM — created via centralised provider
        llm = create_llm()

        # 3) Tools
        tools = TOOL_REGISTRY

        # 4) Supervisor
        supervisor = Supervisor(
            db=db,
            llm=llm,
            tools=tools,
            max_attempts=settings.agentic_max_attempts,
            max_empty_retries=settings.agentic_max_empty_retries,
        )

        _instance = supervisor
        logger.info(
            "✓ Agentic system initialized (max_attempts=%d, max_empty_retries=%d, tools=%d)",
            settings.agentic_max_attempts,
            settings.agentic_max_empty_retries,
            len(tools),
        )
        return supervisor

    @staticmethod
    def get() -> Supervisor | None:
        """Return the cached instance, or None if not yet created."""
        return _instance

    @staticmethod
    async def shutdown() -> None:
        """Cleanup caches, conversation store, and disconnect database."""
        global _instance
        if _instance is not None:
            await _instance.shutdown()
            await _instance._db.disconnect()
            _instance = None
            logger.info("Agentic system shut down")

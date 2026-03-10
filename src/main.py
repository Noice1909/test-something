"""FastAPI application — entry point for the Agentic Graph Query System."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.agents.supervisor_factory import SupervisorFactory
from src.api.routes.agentic import router as agentic_router

# ── Logging ──

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # Startup
    logger.info("[🚀] Starting Agentic Graph Query System …")
    if settings.agentic_system_enabled:
        logger.info("[6b] Initializing Agentic Supervisor …")
        try:
            await SupervisorFactory.create()
        except Exception as exc:
            logger.error("[6b] Supervisor init FAILED: %s", exc)
    else:
        logger.info("Agentic system disabled (AGENTIC_SYSTEM_ENABLED=false)")

    logger.info("Agentic system API routes registered: POST /api/v1/agentic/chat")
    yield

    # Shutdown
    logger.info("[🛑] Shutting down …")
    await SupervisorFactory.shutdown()


# ── App ──


app = FastAPI(
    title="Agentic Graph Query System",
    description="Autonomous multi-specialist AI system for querying Neo4j graph databases",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(agentic_router)


@app.get("/health")
async def root_health():
    return {"status": "ok", "service": "agentic-graph-query"}

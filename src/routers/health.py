from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

from src.core.circuit_breaker import get_breaker_states
from src.models import HealthResponse

logger = structlog.get_logger()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check: Neo4j connectivity, Ollama availability, circuit breaker states."""
    settings = request.app.state.settings
    breakers = request.app.state.breakers
    llm = request.app.state.llm

    # Neo4j check
    neo4j_ok = False
    try:
        neo4j_svc = request.app.state.neo4j_svc
        neo4j_ok = neo4j_svc.verify_connectivity()
    except Exception:
        pass

    # Ollama check
    ollama_ok = False
    try:
        llm.invoke("ping")
        ollama_ok = True
    except Exception:
        pass

    status = "healthy" if (neo4j_ok and ollama_ok) else "degraded"

    return HealthResponse(
        status=status,
        neo4j_connected=neo4j_ok,
        ollama_available=ollama_ok,
        environment=settings.ENVIRONMENT,
        checkpointer="sqlite" if settings.is_local else "redis",
        circuit_breakers=get_breaker_states(breakers),
    )

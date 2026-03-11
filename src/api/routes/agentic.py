"""Agentic system API routes."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import settings
from src.agents.supervisor_factory import SupervisorFactory
from src.api.middleware.auth import require_auth
from src.api.middleware.rate_limit import limiter
from src.api.models.auth import AuthContext
from src.logging_config import current_trace_id
from src.resilience.circuit_breaker import get_breaker_states

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agentic", tags=["agentic"])


# ── Request / Response models ──


class AgenticChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    trace_id: str | None = None
    conversation_id: str | None = None


class AgenticChatResponse(BaseModel):
    answer: str
    strategy_used: str
    attempts: int
    success: bool
    trace_id: str
    conversation_id: str | None = None
    specialist_log: list[dict] = []
    cypher_attempts: list[dict] = []
    from_cache: bool = False


class HealthComponentStatus(BaseModel):
    status: str
    latency_ms: float | None = None
    detail: str | None = None


class AgenticHealthResponse(BaseModel):
    status: str
    supervisor_initialized: bool
    message: str
    components: dict[str, HealthComponentStatus] = {}
    circuit_breakers: dict[str, str] = {}
    version: str = "0.1.0"
    uptime_seconds: float = 0.0


# ── Endpoints ──


@router.post("/chat", response_model=AgenticChatResponse)
@limiter.limit(settings.rate_limit_default)
async def agentic_chat(
    request: Request,
    body: AgenticChatRequest,
    auth: AuthContext = Depends(require_auth),
) -> AgenticChatResponse:
    """Submit a question to the agentic system."""
    # Store auth on request.state for rate-limiter key function
    request.state.auth = auth

    supervisor = SupervisorFactory.get()
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Agentic system not initialized")

    # Set trace context for structured logging
    trace_token = None
    if body.trace_id:
        trace_token = current_trace_id.set(body.trace_id)

    try:
        result = await supervisor.process_question(
            question=body.question,
            trace_id=body.trace_id,
            conversation_id=body.conversation_id,
        )
    finally:
        if trace_token is not None:
            current_trace_id.reset(trace_token)

    return AgenticChatResponse(
        answer=result.answer,
        strategy_used=result.strategy_used,
        attempts=result.attempts,
        success=result.success,
        trace_id=result.trace_id,
        conversation_id=getattr(result, "conversation_id", None),
        specialist_log=result.specialist_log,
        cypher_attempts=result.cypher_attempts,
    )


@router.get("/health", response_model=AgenticHealthResponse)
async def agentic_health() -> AgenticHealthResponse:
    """Check if the agentic system is initialized and healthy."""
    from src.main import get_uptime

    supervisor = SupervisorFactory.get()
    components: dict[str, HealthComponentStatus] = {}

    # Neo4j health
    if supervisor is not None:
        t0 = time.time()
        try:
            db_health = await supervisor._db.health_check()
            latency = round((time.time() - t0) * 1000, 2)
            components["neo4j"] = HealthComponentStatus(
                status="healthy" if db_health.get("healthy") else "unhealthy",
                latency_ms=latency,
            )
        except Exception as exc:
            components["neo4j"] = HealthComponentStatus(
                status="unhealthy", detail=str(exc),
            )
    else:
        components["neo4j"] = HealthComponentStatus(status="unknown", detail="Supervisor not initialized")

    # Circuit breaker states
    breaker_states = get_breaker_states()

    if supervisor is None:
        return AgenticHealthResponse(
            status="unhealthy",
            supervisor_initialized=False,
            message="Agentic system not initialized",
            components=components,
            circuit_breakers=breaker_states,
            uptime_seconds=round(get_uptime(), 1),
        )

    overall = "healthy" if all(
        c.status == "healthy" for c in components.values()
    ) else "degraded"

    return AgenticHealthResponse(
        status=overall,
        supervisor_initialized=True,
        message="Agentic system ready",
        components=components,
        circuit_breakers=breaker_states,
        uptime_seconds=round(get_uptime(), 1),
    )

"""Agentic system API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.supervisor_factory import SupervisorFactory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agentic", tags=["agentic"])


# ── Request / Response models ──


class AgenticChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    trace_id: str | None = None


class AgenticChatResponse(BaseModel):
    answer: str
    strategy_used: str
    attempts: int
    success: bool
    trace_id: str
    specialist_log: list[dict] = []
    cypher_attempts: list[dict] = []


class AgenticHealthResponse(BaseModel):
    status: str
    supervisor_initialized: bool
    message: str


# ── Endpoints ──


@router.post("/chat", response_model=AgenticChatResponse)
async def agentic_chat(request: AgenticChatRequest) -> AgenticChatResponse:
    """Submit a question to the agentic system."""
    supervisor = SupervisorFactory.get()
    if supervisor is None:
        raise HTTPException(status_code=503, detail="Agentic system not initialized")

    result = await supervisor.process_question(
        question=request.question,
        trace_id=request.trace_id,
    )

    return AgenticChatResponse(
        answer=result.answer,
        strategy_used=result.strategy_used,
        attempts=result.attempts,
        success=result.success,
        trace_id=result.trace_id,
        specialist_log=result.specialist_log,
        cypher_attempts=result.cypher_attempts,
    )


@router.get("/health", response_model=AgenticHealthResponse)
async def agentic_health() -> AgenticHealthResponse:
    """Check if the agentic system is initialized and healthy."""
    supervisor = SupervisorFactory.get()
    if supervisor is None:
        return AgenticHealthResponse(
            status="unhealthy",
            supervisor_initialized=False,
            message="Agentic system not initialized",
        )
    return AgenticHealthResponse(
        status="healthy",
        supervisor_initialized=True,
        message="Agentic system ready",
    )

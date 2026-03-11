"""Async job-based query endpoint — submit and poll for results."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.agents.supervisor_factory import SupervisorFactory
from src.api.middleware.auth import require_auth
from src.api.models.auth import AuthContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agentic", tags=["agentic-async"])

# In-memory job store (swap for Redis in production)
_jobs: dict[str, dict[str, Any]] = {}


class AsyncQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    trace_id: str | None = None


class AsyncQuerySubmitResponse(BaseModel):
    job_id: str
    status: str


class AsyncQueryStatusResponse(BaseModel):
    job_id: str
    status: str  # pending | processing | completed | failed
    result: dict | None = None
    submitted_at: float
    completed_at: float | None = None


async def _process_job(job_id: str, question: str, trace_id: str | None) -> None:
    """Background task to process a question."""
    _jobs[job_id]["status"] = "processing"
    try:
        supervisor = SupervisorFactory.get()
        if supervisor is None:
            _jobs[job_id].update(status="failed", result={"error": "System not initialized"})
            return

        result = await supervisor.process_question(question=question, trace_id=trace_id)
        _jobs[job_id].update(
            status="completed",
            result={
                "answer": result.answer,
                "strategy_used": result.strategy_used,
                "attempts": result.attempts,
                "success": result.success,
                "trace_id": result.trace_id,
            },
            completed_at=time.time(),
        )
    except Exception as exc:
        logger.error("Async job %s failed: %s", job_id, exc)
        _jobs[job_id].update(status="failed", result={"error": str(exc)}, completed_at=time.time())


@router.post("/query", response_model=AsyncQuerySubmitResponse)
async def submit_query(
    request: Request,
    body: AsyncQueryRequest,
    auth: AuthContext = Depends(require_auth),
) -> AsyncQuerySubmitResponse:
    """Submit a question for async processing. Returns immediately with a job ID."""
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    _jobs[job_id] = {
        "status": "pending",
        "question": body.question,
        "submitted_at": time.time(),
        "result": None,
        "completed_at": None,
    }

    # Fire and forget
    asyncio.create_task(_process_job(job_id, body.question, body.trace_id))

    return AsyncQuerySubmitResponse(job_id=job_id, status="pending")


@router.get("/query/{job_id}", response_model=AsyncQueryStatusResponse)
async def get_query_status(job_id: str) -> AsyncQueryStatusResponse:
    """Poll for the result of an async query."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return AsyncQueryStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        submitted_at=job["submitted_at"],
        completed_at=job.get("completed_at"),
    )

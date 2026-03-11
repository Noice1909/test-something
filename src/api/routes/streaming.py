"""Server-Sent Events (SSE) streaming endpoint for progressive updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.agents.supervisor_factory import SupervisorFactory
from src.api.middleware.auth import require_auth
from src.api.models.auth import AuthContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agentic", tags=["agentic-stream"])


async def _event_stream(question: str, trace_id: str | None) -> AsyncGenerator[str, None]:
    """Generate SSE events as the question is processed."""
    supervisor = SupervisorFactory.get()
    if supervisor is None:
        yield _sse_event("error", {"code": "SERVICE_UNAVAILABLE", "message": "System not initialized"})
        return

    yield _sse_event("processing", {"status": "started", "question": question[:200]})

    try:
        result = await supervisor.process_question(question=question, trace_id=trace_id)

        # Send specialist log entries as progress events
        for entry in result.specialist_log:
            yield _sse_event("specialist_completed", entry)
            await asyncio.sleep(0)  # Yield control

        # Final answer
        yield _sse_event("answer", {
            "answer": result.answer,
            "strategy_used": result.strategy_used,
            "attempts": result.attempts,
            "success": result.success,
            "trace_id": result.trace_id,
        })

    except Exception as exc:
        logger.error("SSE stream error: %s", exc)
        yield _sse_event("error", {"code": "INTERNAL_ERROR", "message": str(exc)})

    yield _sse_event("done", {"status": "completed"})


def _sse_event(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get("/stream")
async def stream_chat(
    request: Request,
    question: str = Query(..., min_length=1, max_length=4096),
    trace_id: str | None = Query(None),
    auth: AuthContext = Depends(require_auth),
):
    """Stream progressive updates as a question is processed."""
    from starlette.responses import StreamingResponse

    return StreamingResponse(
        _event_stream(question, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

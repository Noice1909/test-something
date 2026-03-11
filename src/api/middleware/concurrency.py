"""Concurrency limiting middleware — prevents resource exhaustion."""

from __future__ import annotations

import asyncio
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class ConcurrencyLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests with 429 when too many are being processed concurrently."""

    def __init__(self, app, *, max_concurrent: int = 50) -> None:  # noqa: ANN001
        super().__init__(app)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Allow health checks to pass through
        if request.url.path.endswith("/health"):
            return await call_next(request)

        if self._semaphore.locked():
            logger.warning(
                "Concurrency limit reached (%d). Rejecting request.", self._max,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "TOO_MANY_REQUESTS",
                        "message": f"Server is handling maximum {self._max} concurrent requests. Retry shortly.",
                    }
                },
                headers={"Retry-After": "5"},
            )

        async with self._semaphore:
            return await call_next(request)


# ── LLM-level semaphore (used by ResilientLLM) ──

_llm_semaphore: asyncio.Semaphore | None = None


def get_llm_semaphore(max_concurrent: int = 10) -> asyncio.Semaphore:
    """Return the shared LLM concurrency semaphore (lazy-init)."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(max_concurrent)
    return _llm_semaphore

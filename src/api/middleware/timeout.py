"""Request timeout middleware — cancels slow requests."""

from __future__ import annotations

import asyncio
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.logging_config import current_trace_id

logger = logging.getLogger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Return 504 if the downstream handler exceeds *timeout* seconds."""

    def __init__(self, app, *, timeout: float = 120.0) -> None:  # noqa: ANN001
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            trace = current_trace_id.get("-")
            logger.warning("Request timed out after %.1fs [%s]", self.timeout, trace)
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "code": "REQUEST_TIMEOUT",
                        "message": f"Request timed out after {self.timeout}s",
                        "trace_id": trace,
                    }
                },
            )

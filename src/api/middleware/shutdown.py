"""Graceful shutdown — tracks in-flight requests and drains before exit."""

from __future__ import annotations

import asyncio
import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)


class InFlightTracker(BaseHTTPMiddleware):
    """Counts active requests.  During shutdown, rejects new ones with 503."""

    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self._in_flight = 0
        self._draining = False
        self._drained = asyncio.Event()
        self._drained.set()  # initially no requests → already drained

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def start_drain(self) -> None:
        """Signal that we are shutting down — reject new requests."""
        self._draining = True
        if self._in_flight == 0:
            self._drained.set()

    async def wait_for_drain(self, timeout: float = 30.0) -> None:
        """Block until all in-flight requests finish, or *timeout* expires."""
        self.start_drain()
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout)
            logger.info("All in-flight requests drained.")
        except asyncio.TimeoutError:
            logger.warning(
                "Drain timeout after %.1fs — %d requests still in flight.",
                timeout, self._in_flight,
            )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self._draining:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "SERVICE_UNAVAILABLE",
                        "message": "Server is shutting down. Please retry shortly.",
                    }
                },
                headers={"Retry-After": "5"},
            )

        self._in_flight += 1
        self._drained.clear()
        try:
            return await call_next(request)
        finally:
            self._in_flight -= 1
            if self._draining and self._in_flight == 0:
                self._drained.set()

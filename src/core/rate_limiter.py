from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def create_limiter() -> Limiter:
    return Limiter(
        key_func=get_remote_address,
        default_limits=[],
        storage_uri="memory://",
    )


def rate_limit_exceeded_handler(_request: Request, _exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "answer": "Too many requests. Please wait before trying again.",
            "thread_id": None,
            "success": False,
        },
    )

"""Global exception handlers — returns a consistent error envelope."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.logging_config import current_trace_id

logger = logging.getLogger(__name__)


def _envelope(code: str, message: str, status: int, *, retry_after: int | None = None) -> JSONResponse:
    body: dict = {
        "error": {
            "code": code,
            "message": message,
            "trace_id": current_trace_id.get("-"),
        }
    }
    headers: dict[str, str] = {}
    if retry_after is not None:
        body["error"]["retry_after"] = retry_after
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(status_code=status, content=body, headers=headers or None)


async def _http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code_map: dict[int, str] = {
        401: "AUTH_REQUIRED",
        403: "AUTH_INVALID",
        404: "NOT_FOUND",
        429: "RATE_LIMIT_EXCEEDED",
        503: "SERVICE_UNAVAILABLE",
        504: "REQUEST_TIMEOUT",
    }
    error_code = code_map.get(exc.status_code, "HTTP_ERROR")
    return _envelope(error_code, str(exc.detail), exc.status_code)


async def _validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    msg = "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in errors) if errors else "Validation error"
    return _envelope("VALIDATION_ERROR", msg, 422)


async def _generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return _envelope("INTERNAL_ERROR", "An internal error occurred.", 500)


def register_error_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI *app*."""
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(Exception, _generic_exception_handler)

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import structlog

logger = structlog.get_logger()


class AgentError(Exception):
    """Base agent exception."""


class Neo4jUnavailableError(AgentError):
    """Neo4j circuit breaker is open or connection failed."""


class OllamaUnavailableError(AgentError):
    """Ollama circuit breaker is open or connection failed."""


class CypherGenerationError(AgentError):
    """LLM failed to generate valid Cypher after all retries."""


class CypherValidationError(AgentError):
    """Generated Cypher failed all validation layers."""


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"answer": message, "thread_id": None, "success": False},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Neo4jUnavailableError)
    async def neo4j_unavailable(_: Request, exc: Neo4jUnavailableError) -> JSONResponse:
        logger.error("neo4j_unavailable", error=str(exc))
        return _error_response(503, "The data service is temporarily unavailable. Please try again shortly.")

    @app.exception_handler(OllamaUnavailableError)
    async def ollama_unavailable(_: Request, exc: OllamaUnavailableError) -> JSONResponse:
        logger.error("ollama_unavailable", error=str(exc))
        return _error_response(503, "The AI service is temporarily unavailable. Please try again shortly.")

    @app.exception_handler(CypherGenerationError)
    async def cypher_generation_error(_: Request, exc: CypherGenerationError) -> JSONResponse:
        logger.warning("cypher_generation_failed", error=str(exc))
        return _error_response(422, "I wasn't able to understand that question well enough. Could you try rephrasing it?")

    @app.exception_handler(CypherValidationError)
    async def cypher_validation_error(_: Request, exc: CypherValidationError) -> JSONResponse:
        logger.warning("cypher_validation_failed", error=str(exc))
        return _error_response(422, "I had trouble processing that query. Could you try a simpler question?")

    @app.exception_handler(Exception)
    async def global_exception(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", error=str(exc))
        return _error_response(500, "Something went wrong. Please try again.")

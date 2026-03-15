"""FastAPI routes — the HTTP interface to the orchestrator."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from api.models import (
    AgentInfo,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReloadResponse,
    SkillInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_orchestrator(request: Request) -> Any:
    """Retrieve the orchestrator from app state."""
    return request.app.state.orchestrator


def _get_registry(request: Request) -> Any:
    return request.app.state.registry


def _get_driver(request: Request) -> Any:
    return request.app.state.driver


# ── POST /query ──────────────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Send a message to the orchestrator and get a response."""
    orchestrator = _get_orchestrator(request)
    try:
        result = await orchestrator.handle_query(
            user_input=body.message,
            session_id=body.session_id,
        )
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        response=result["response"],
        session_id=result["session_id"],
        turns_used=result.get("turns_used", 0),
        tools_called=result.get("tools_called", []),
    )


# ── POST /query/stream ──────────────────────────────────────────────────────


@router.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request) -> EventSourceResponse:
    """SSE token-by-token streaming — yields events as the agent works.

    Event types:
      - token:      individual LLM output token
      - tool_start: before a tool executes
      - tool_end:   after a tool completes
      - hook_skip:  a hook skipped a tool call
      - hook_block: a hook blocked a tool call
      - done:       final response with session info
      - error:      something went wrong
    """
    orchestrator = _get_orchestrator(request)

    async def event_generator():
        try:
            async for event in orchestrator.handle_query_streaming(
                user_input=body.message,
                session_id=body.session_id,
            ):
                yield {"data": json.dumps(event)}
        except Exception as exc:
            logger.exception("Streaming query failed")
            yield {"data": json.dumps({"type": "error", "message": str(exc)})}

    return EventSourceResponse(event_generator())


# ── GET /skills ──────────────────────────────────────────────────────────────


@router.get("/skills", response_model=list[SkillInfo])
async def list_skills(request: Request) -> list[SkillInfo]:
    """List all discovered skills."""
    registry = _get_registry(request)
    return [
        SkillInfo(
            name=s.name,
            description=s.description,
            user_invocable=s.user_invocable,
            argument_hint=s.argument_hint,
            context=s.context,
        )
        for s in registry.skills.values()
    ]


# ── GET /agents ──────────────────────────────────────────────────────────────


@router.get("/agents", response_model=list[AgentInfo])
async def list_agents(request: Request) -> list[AgentInfo]:
    """List all discovered sub-agents."""
    registry = _get_registry(request)
    return [
        AgentInfo(
            name=a.name,
            description=a.description,
            tools=a.tools,
            max_turns=a.max_turns,
        )
        for a in registry.agents.values()
    ]


# ── POST /skills/reload ─────────────────────────────────────────────────────


@router.post("/skills/reload", response_model=ReloadResponse)
async def reload_skills(request: Request) -> ReloadResponse:
    """Hot-reload skills and agents from the filesystem."""
    registry = _get_registry(request)
    await registry.reload()
    # Invalidate the orchestrator's prompt cache so new skills show up
    orchestrator = _get_orchestrator(request)
    orchestrator.invalidate_prompt_cache()
    return ReloadResponse(
        skills_loaded=len(registry.skills),
        agents_loaded=len(registry.agents),
        message="Reloaded successfully",
    )


# ── GET /health ──────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check — includes Neo4j connectivity."""
    registry = _get_registry(request)
    driver = _get_driver(request)

    neo4j_ok = False
    try:
        async with driver.session() as session:
            await session.run("RETURN 1")
        neo4j_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        skills_loaded=len(registry.skills),
        agents_loaded=len(registry.agents),
    )

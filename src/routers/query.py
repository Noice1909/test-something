from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Request
from slowapi import Limiter

from src.config import settings
from src.models import AskRequest, AskResponse

logger = structlog.get_logger()
router = APIRouter()


def _get_limiter(request: Request) -> Limiter:
    return request.app.state.limiter


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """Main endpoint: accepts a natural language question, returns an NL answer."""
    limiter = _get_limiter(request)
    # Rate limiting is handled by SlowAPI middleware via decorator on app level

    graph = request.app.state.graph
    thread_id = body.thread_id or str(uuid.uuid4())

    logger.info("ask_received", question=body.question[:100], thread_id=thread_id)

    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "question": body.question,
        "matched_concepts": [],
        "extracted_entities": [],
        "mapped_entities": [],
        "filtered_schema": "",
        "few_shot_examples": "",
        "cypher": "",
        "validation_errors": [],
        "retry_count": 0,
        "query_results": [],
        "answer": "",
        "error": None,
    }

    result = await graph.ainvoke(initial_state, config)

    answer = result.get("answer", "Something went wrong. Please try again.")
    success = bool(result.get("query_results")) or "couldn't find" not in answer.lower()

    logger.info(
        "ask_completed",
        thread_id=thread_id,
        success=success,
        answer_length=len(answer),
    )

    return AskResponse(
        answer=answer,
        thread_id=thread_id,
        success=success,
    )

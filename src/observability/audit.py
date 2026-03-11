"""Audit logging — records all user interactions for compliance."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("audit")


def log_request(
    *,
    user_id: str,
    question: str,
    trace_id: str,
    ip_address: str = "",
    conversation_id: str | None = None,
) -> None:
    """Log an incoming request to the audit trail."""
    logger.info(
        json.dumps({
            "event": "request",
            "user_id": user_id,
            "question": question[:500],
            "trace_id": trace_id,
            "ip_address": ip_address,
            "conversation_id": conversation_id,
            "timestamp": time.time(),
        }, default=str)
    )


def log_response(
    *,
    trace_id: str,
    success: bool,
    strategy: str,
    attempts: int,
    duration_ms: float,
    cypher_query: str = "",
    row_count: int = 0,
    from_cache: bool = False,
) -> None:
    """Log a completed response to the audit trail."""
    logger.info(
        json.dumps({
            "event": "response",
            "trace_id": trace_id,
            "success": success,
            "strategy": strategy,
            "attempts": attempts,
            "duration_ms": round(duration_ms, 2),
            "cypher_query": cypher_query[:500],
            "row_count": row_count,
            "from_cache": from_cache,
            "timestamp": time.time(),
        }, default=str)
    )

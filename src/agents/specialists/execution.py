"""Execution Specialist — runs queries on the database safely."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from src.agents.base import ExecutionResult, SpecialistResult
from src.agents.state import AgentState
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET\b|REMOVE|DROP|CALL\s*\{|FOREACH)\b",
    re.IGNORECASE,
)


class ExecutionSpecialist:
    """Executes the generated query with safety checks and error categorisation."""

    def __init__(
        self, db: AbstractDatabase, tools: dict[str, Any] | None = None, **_: Any
    ) -> None:
        self._db = db
        self._tools = tools or {}

    async def run(self, state: AgentState) -> SpecialistResult:
        t0 = time.time()
        query = state.generated_query.query
        params = state.generated_query.parameters

        # Safety gate
        if _WRITE_PATTERN.search(query):
            err = "Query rejected: contains write operations"
            state.execution_result = ExecutionResult(
                success=False, error=err, error_category="permission",
            )
            dur = (time.time() - t0) * 1000
            state.log_specialist("execution", success=False, duration_ms=dur, detail=err)
            return SpecialistResult(success=False, error=err, duration_ms=dur)

        try:
            rows = await self._db.execute_read(query, params)
            dur = (time.time() - t0) * 1000

            state.execution_result = ExecutionResult(
                success=True,
                rows=rows,
                row_count=len(rows),
                duration_ms=dur,
            )
            state.log_specialist(
                "execution", success=True, duration_ms=dur,
                detail=f"{len(rows)} rows in {dur:.0f}ms",
            )
            return SpecialistResult(success=True, data=rows, duration_ms=dur)

        except Exception as exc:
            dur = (time.time() - t0) * 1000
            category = self._categorise_error(str(exc))
            state.execution_result = ExecutionResult(
                success=False, error=str(exc), error_category=category, duration_ms=dur,
            )
            state.log_specialist("execution", success=False, duration_ms=dur, detail=str(exc))
            return SpecialistResult(success=False, error=str(exc), duration_ms=dur)

    @staticmethod
    def _categorise_error(error: str) -> str:
        err_lower = error.lower()
        if "syntax" in err_lower or "unexpected" in err_lower:
            return "syntax"
        if "timeout" in err_lower or "timed out" in err_lower:
            return "timeout"
        if "permission" in err_lower or "unauthorized" in err_lower:
            return "permission"
        if "connection" in err_lower or "unavailable" in err_lower:
            return "connection"
        return "unknown"

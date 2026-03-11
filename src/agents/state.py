"""Mutable state object threaded through a single query lifecycle."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import (
    DiscoveryResult,
    ExecutionResult,
    GeneratedQuery,
    QueryPlan,
    ReflectionResult,
    SchemaSelection,
    StrategyType,
)


@dataclass
class AgentState:
    """Accumulates data as specialists execute in sequence.

    One ``AgentState`` per user question.  The supervisor creates it, passes
    it to each specialist, and reads the final state to produce the response.
    """

    # ── input ──
    question: str = ""
    trace_id: str = field(default_factory=lambda: f"agentic-{uuid.uuid4().hex[:8]}")

    # ── supervisor ──
    strategy: StrategyType = StrategyType.DISCOVERY_FIRST
    attempt_number: int = 1

    # ── specialist outputs ──
    discoveries: list[DiscoveryResult] = field(default_factory=list)
    schema_selection: SchemaSelection = field(default_factory=SchemaSelection)
    query_plan: QueryPlan = field(default_factory=QueryPlan)
    generated_query: GeneratedQuery = field(default_factory=GeneratedQuery)
    execution_result: ExecutionResult = field(default_factory=ExecutionResult)
    reflection: ReflectionResult = field(default_factory=ReflectionResult)

    # ── observability ──
    history: list[dict[str, Any]] = field(default_factory=list)
    cypher_attempts: list[dict[str, Any]] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)

    # ── empty-result retry tracking ──
    empty_retries_used: int = 0
    previous_empty_queries: list[dict[str, str]] = field(default_factory=list)

    # ── helpers ──

    def log_specialist(
        self, name: str, *, success: bool, duration_ms: float, detail: str = ""
    ) -> None:
        """Append an entry to the specialist execution log."""
        self.history.append({
            "specialist": name,
            "attempt": self.attempt_number,
            "success": success,
            "duration_ms": round(duration_ms, 2),
            "detail": detail,
        })

    def log_cypher_attempt(
        self,
        *,
        query: str,
        parameters: dict[str, Any] | None = None,
        reasoning: str = "",
        success: bool = False,
        error: str | None = None,
        row_count: int = 0,
        attempt: int | None = None,
    ) -> None:
        """Record a Cypher query execution attempt."""
        self.cypher_attempts.append({
            "attempt": attempt or self.attempt_number,
            "cypher": query,
            "parameters": parameters or {},
            "reasoning": reasoning,
            "success": success,
            "error": error,
            "row_count": row_count,
        })

    @property
    def elapsed_ms(self) -> float:
        return round((time.time() - self._start_time) * 1000, 2)


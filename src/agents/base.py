"""Base data structures for the agentic system."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ── Enumerations ──


class StrategyType(str, enum.Enum):
    """Strategy the supervisor selects for a question."""

    DISCOVERY_FIRST = "discovery_first"
    DIRECT_QUERY = "direct_query"
    SCHEMA_EXPLORATION = "schema_exploration"
    AGGREGATION = "aggregation"


class QueryComplexity(str, enum.Enum):
    """How complex the generated query should be."""

    DIRECT = "DIRECT"
    ONE_HOP = "ONE_HOP"
    TWO_HOP = "TWO_HOP"
    MULTI_HOP = "MULTI_HOP"
    AGGREGATION = "AGGREGATION"


class RetryStrategy(str, enum.Enum):
    """What the Reflection specialist recommends on failure."""

    EXPAND_DISCOVERY = "expand_discovery"
    SIMPLIFY_QUERY = "simplify_query"
    ADD_TRAVERSALS = "add_traversals"
    CHANGE_SCHEMA = "change_schema"
    GIVE_UP = "give_up"


# ── Result Dataclasses ──


@dataclass
class SpecialistResult:
    """Generic result wrapper returned by every specialist."""

    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class DiscoveryResult:
    """An entity discovered in the database."""

    entity_name: str
    label: str
    node_id: str | None = None
    confidence: float = 0.0
    match_type: str = ""  # exact_match, fuzzy_match, etc.
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaSelection:
    """Subset of the schema that the LLM deemed relevant."""

    node_labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class QueryPlan:
    """Execution plan produced by the planning specialist."""

    strategy: QueryComplexity = QueryComplexity.DIRECT
    intent: str = ""  # LIST, COUNT, FIND, EXPLORE, COMPARE, RANK, etc.
    reasoning: str = ""
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedQuery:
    """A database query ready for execution."""

    query: str = ""
    language: str = "cypher"
    parameters: dict[str, Any] = field(default_factory=dict)
    is_read_only: bool = True
    reasoning: str = ""


@dataclass
class ExecutionResult:
    """Raw result of executing a query."""

    success: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    error_category: str | None = None  # syntax, timeout, permission, connection


@dataclass
class ReflectionResult:
    """Recommendation from the Reflection specialist."""

    should_retry: bool = False
    retry_strategy: RetryStrategy = RetryStrategy.GIVE_UP
    modified_parameters: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    fallback_answer: str | None = None
    next_approach: str = ""  # schema-driven suggestion for next query


@dataclass
class SupervisorDecision:
    """The supervisor's initial decision for a question."""

    strategy: StrategyType = StrategyType.DISCOVERY_FIRST
    reasoning: str = ""
    specialist_sequence: list[str] = field(default_factory=list)


@dataclass
class AgenticResponse:
    """Final response sent to the API caller."""

    answer: str
    strategy_used: str
    attempts: int
    success: bool
    trace_id: str
    specialist_log: list[dict[str, Any]] = field(default_factory=list)
    cypher_attempts: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    from_cache: bool = False

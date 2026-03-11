"""Prometheus metrics for the agentic system."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Request-level ──

REQUEST_DURATION = Histogram(
    "agentic_request_duration_seconds",
    "Request processing duration",
    ["method", "endpoint", "status"],
)

REQUESTS_IN_FLIGHT = Gauge(
    "agentic_requests_in_flight",
    "Number of requests currently being processed",
)

# ── Supervisor-level ──

QUESTIONS_TOTAL = Counter(
    "agentic_questions_total",
    "Total questions processed",
    ["strategy", "success"],
)

QUESTION_DURATION = Histogram(
    "agentic_question_duration_seconds",
    "Question processing duration",
    ["strategy"],
)

ATTEMPTS_TOTAL = Counter(
    "agentic_attempts_total",
    "Total execution attempts",
    ["attempt_number"],
)

RETRIES_TOTAL = Counter(
    "agentic_retries_total",
    "Total retries by strategy",
    ["retry_strategy"],
)

# ── Specialist-level ──

SPECIALIST_DURATION = Histogram(
    "agentic_specialist_duration_seconds",
    "Specialist execution duration",
    ["specialist", "success"],
)

SPECIALIST_CALLS = Counter(
    "agentic_specialist_calls_total",
    "Total specialist invocations",
    ["specialist", "success"],
)

# ── Cache ──

CACHE_HITS = Counter(
    "agentic_cache_hits_total",
    "Cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "agentic_cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

# ── Circuit breaker ──

CIRCUIT_BREAKER_STATE = Gauge(
    "agentic_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half-open)",
    ["service"],
)

CIRCUIT_BREAKER_TRIPS = Counter(
    "agentic_circuit_breaker_trips_total",
    "Circuit breaker trip count",
    ["service"],
)

# ── Database ──

NEO4J_QUERY_DURATION = Histogram(
    "agentic_neo4j_query_duration_seconds",
    "Neo4j query execution duration",
)

NEO4J_QUERIES = Counter(
    "agentic_neo4j_queries_total",
    "Total Neo4j queries",
    ["success"],
)

# ── LLM ──

LLM_CALL_DURATION = Histogram(
    "agentic_llm_call_duration_seconds",
    "LLM call duration",
    ["provider", "model"],
)

LLM_CALLS = Counter(
    "agentic_llm_calls_total",
    "Total LLM invocations",
    ["provider", "model", "success"],
)

"""FastAPI application — entry point for the Agentic Graph Query System."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.config import settings
from src.logging_config import setup_logging
from src.agents.supervisor_factory import SupervisorFactory
from src.api.routes.agentic import router as agentic_router
from src.api.routes.streaming import router as streaming_router
from src.api.routes.async_query import router as async_query_router
from src.api.middleware.error_handler import register_error_handlers
from src.api.middleware.rate_limit import limiter
from src.api.middleware.timeout import TimeoutMiddleware
from src.api.middleware.concurrency import ConcurrencyLimitMiddleware
from src.api.middleware.shutdown import InFlightTracker
from src.resilience.circuit_breaker import CircuitOpenError

# ── Logging ──

setup_logging(level=settings.log_level, fmt=settings.log_format)
logger = logging.getLogger(__name__)

# ── Shutdown tracker & DB health monitor (module-level so lifespan can access) ──

_in_flight_tracker: InFlightTracker | None = None
_db_monitor = None
_start_time: float = time.time()


def get_uptime() -> float:
    return time.time() - _start_time


# ── Lifespan ──


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _db_monitor

    # Startup
    logger.info("Starting Agentic Graph Query System")
    if settings.agentic_system_enabled:
        logger.info("Initializing Agentic Supervisor")
        try:
            await SupervisorFactory.create()

            # Start database health monitor
            supervisor = SupervisorFactory.get()
            if supervisor is not None:
                from src.database.health_monitor import DatabaseHealthMonitor
                _db_monitor = DatabaseHealthMonitor(supervisor._db)
                _db_monitor.start()

        except Exception as exc:
            logger.error("Supervisor init FAILED: %s", exc)
    else:
        logger.info("Agentic system disabled (AGENTIC_SYSTEM_ENABLED=false)")

    logger.info("Agentic system API routes registered: POST /api/v1/agentic/chat")
    yield

    # Shutdown
    logger.info("Shutting down")

    # Drain in-flight requests
    if _in_flight_tracker is not None:
        await _in_flight_tracker.wait_for_drain(timeout=30.0)

    # Stop health monitor
    if _db_monitor is not None:
        _db_monitor.stop()
        await _db_monitor.wait()

    await SupervisorFactory.shutdown()


# ── App ──


app = FastAPI(
    title="Agentic Graph Query System",
    description="Autonomous multi-specialist AI system for querying Neo4j graph databases",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Prometheus metrics ──

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# ── Error handlers ──

register_error_handlers(app)


# Circuit-breaker specific handler
async def _circuit_open_handler(_request, exc: CircuitOpenError):  # noqa: ANN001
    from src.api.middleware.error_handler import _envelope
    return _envelope(
        "CIRCUIT_OPEN",
        f"Service '{exc.service}' is temporarily unavailable. Retry after {exc.retry_after}s.",
        503,
        retry_after=exc.retry_after,
    )


app.add_exception_handler(CircuitOpenError, _circuit_open_handler)

# ── Rate limiting ──

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Middleware (order matters — last added = outermost) ──

# 1. CORS (outermost)
_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True if _origins != ["*"] else False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 2. Timeout
app.add_middleware(TimeoutMiddleware, timeout=float(settings.request_timeout_seconds))

# 3. Concurrency limiter
app.add_middleware(ConcurrencyLimitMiddleware, max_concurrent=settings.max_concurrent_requests)

# 4. In-flight tracker (for graceful shutdown)
_in_flight_tracker = InFlightTracker(app)

# ── Routes ──

app.include_router(agentic_router)
app.include_router(streaming_router)
app.include_router(async_query_router)


@app.get("/health")
async def root_health():
    return {"status": "ok", "service": "agentic-graph-query"}

from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.core.exceptions import register_exception_handlers
from src.core.lifespan import lifespan
from src.core.middleware import RequestContextMiddleware
from src.core.rate_limiter import create_limiter, rate_limit_exceeded_handler
from src.routers import health, query, schema


def create_app() -> FastAPI:
    app = FastAPI(
        title="Neo4j NL Agent",
        version="1.0.0",
        description="Natural language interface to Neo4j graph databases",
        lifespan=lifespan,
    )

    # Rate limiter
    limiter = create_limiter()
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # Request context middleware
    app.add_middleware(RequestContextMiddleware)

    # Custom exception handlers
    register_exception_handlers(app)

    # Routers
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(schema.router, prefix="/api", tags=["schema"])
    app.include_router(query.router, prefix="/api", tags=["query"])

    return app


app = create_app()

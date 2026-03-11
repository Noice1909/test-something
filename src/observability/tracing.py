"""OpenTelemetry distributed tracing setup.

Call ``setup_tracing()`` during app startup to instrument FastAPI and httpx.
Requires: opentelemetry-api, opentelemetry-sdk,
          opentelemetry-instrumentation-fastapi, opentelemetry-exporter-otlp
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def setup_tracing(
    service_name: str = "agentic-graph-query",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry tracing (no-op if libraries are missing)."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing initialized (service=%s)", service_name)
    except ImportError:
        logger.info("OpenTelemetry libraries not installed — tracing disabled")


def instrument_fastapi(app) -> None:  # noqa: ANN001
    """Instrument a FastAPI app with OpenTelemetry (no-op if unavailable)."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
    except ImportError:
        pass

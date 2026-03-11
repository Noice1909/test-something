"""Structured logging with sensitive data masking.

Provides JSON-formatted logs with automatic redaction of passwords,
API keys, and other sensitive patterns.  A ``contextvars``-based
request context injects ``trace_id`` into every log record.
"""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar

from pythonjsonlogger.json import JsonFormatter

# ── Request-scoped context ──

current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="-")

# ── Masking patterns ──

_MASK = "***REDACTED***"

_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    # API keys (OpenAI, generic sk- prefixed)
    re.compile(r"sk-[A-Za-z0-9]{20,}", re.IGNORECASE),
    # Generic password fields in key=value or JSON (handles "key": "value" and key=value)
    re.compile(r'(?i)"?(password|passwd|secret|token|api_key|apikey)"?\s*[=:]\s*"?[^\s",}]+'),
    # Neo4j URIs with embedded credentials  user:pass@host
    re.compile(r"://[^@/\s]+:[^@/\s]+@"),
    # Bearer tokens
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
]


def _mask_sensitive(text: str) -> str:
    """Replace sensitive patterns in *text* with a redaction placeholder."""
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(_MASK, text)
    return text


# ── Filters ──


class SensitiveMaskingFilter(logging.Filter):
    """Redacts sensitive data from log messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        if isinstance(record.msg, str):
            record.msg = _mask_sensitive(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: _mask_sensitive(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    _mask_sensitive(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
        return True


class RequestContextFilter(logging.Filter):
    """Injects the current ``trace_id`` into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        record.trace_id = current_trace_id.get("-")  # type: ignore[attr-defined]
        return True


# ── Formatter ──

_JSON_FMT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_JSON_RENAME = {
    "asctime": "timestamp",
    "levelname": "level",
    "name": "logger",
}


# ── Setup ──


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the root logger with structured output and masking.

    Parameters
    ----------
    level:
        Log level name (``DEBUG``, ``INFO``, ``WARNING``, …).
    fmt:
        ``"json"`` for structured JSON lines, ``"text"`` for human-readable.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates on reload
    root.handlers.clear()

    handler = logging.StreamHandler()

    if fmt == "json":
        formatter = JsonFormatter(
            fmt=_JSON_FMT,
            rename_fields=_JSON_RENAME,
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(trace_id)s): %(message)s",
        )

    handler.setFormatter(formatter)
    handler.addFilter(SensitiveMaskingFilter())
    handler.addFilter(RequestContextFilter())
    root.addHandler(handler)

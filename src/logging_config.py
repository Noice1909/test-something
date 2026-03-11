"""
Structured logging configuration using structlog.

Produces coloured, column-aligned log lines::

    17:39:40 │ main.py              │  INFO   │ Starting Agentic Graph Query System

Columns: Time (cyan) │ File (magenta) │ Level (colour-coded) │ Message.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog


# ── ANSI colour helpers ──────────────────────────────────────────────────

_RESET = "\033[0m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_BOLD_RED = "\033[1;31m"
_WHITE = "\033[37m"

_LEVEL_COLOURS: dict[str, str] = {
    "debug": _DIM,
    "info": _GREEN,
    "warning": _YELLOW,
    "error": _RED,
    "critical": _BOLD_RED,
}

_FILE_WIDTH = 20      # pad / truncate the file column
_LEVEL_WIDTH = 8      # pad the level column
_SEP = f"{_DIM}│{_RESET}"


# ── Masking patterns ──────────────────────────────────────────────────────

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
    if not isinstance(text, str):
        return text
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(_MASK, text)
    return text


def _module_to_filename(logger_name: str) -> str:
    """Convert a dotted module path to a short filename, e.g. ``src.main`` → ``main.py``."""
    if not logger_name:
        return "<unknown>"
    last = logger_name.rsplit(".", 1)[-1]
    return f"{last}.py"


# ── Structlog processors ──────────────────────────────────────────────────


def mask_sensitive_processor(
    logger: Any,
    method_name: str,
    event_dict: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """Structlog processor that masks sensitive data in the event message and all values."""
    # Mask the main event message
    if "event" in event_dict and isinstance(event_dict["event"], str):
        event_dict["event"] = _mask_sensitive(event_dict["event"])

    # Mask all string values in the event dict
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = _mask_sensitive(value)

    return event_dict


class ColumnRenderer:
    """structlog processor that renders each event as a fixed-width, coloured row.

    Intended as the **final** processor in a ``ProcessorFormatter`` pipeline.
    """

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: MutableMapping[str, Any],
    ) -> str:
        # ── Extract standard fields ──────────────────────────────────────
        timestamp: str = event_dict.pop("timestamp", "")
        # Keep only HH:MM:SS from an ISO timestamp (or the raw value)
        if "T" in timestamp:
            timestamp = timestamp.split("T", 1)[1][:8]

        level: str = event_dict.pop("level", method_name).upper()
        logger_name: str = event_dict.pop("logger", "")
        event: str = event_dict.pop("event", "")

        filename = _module_to_filename(logger_name)

        # ── Colour each column ───────────────────────────────────────────
        level_colour = _LEVEL_COLOURS.get(level.lower(), _WHITE)

        col_time = f"{_CYAN}{timestamp}{_RESET}"
        col_file = f"{_MAGENTA}{filename:<{_FILE_WIDTH}}{_RESET}"
        col_level = f"{level_colour}{level:^{_LEVEL_WIDTH}}{_RESET}"
        col_event = f"{_WHITE}{event}{_RESET}"

        # ── Extra context key=value pairs (structlog bindings) ───────────
        extras = ""
        # Remove internal keys that shouldn't be displayed
        for key in ("_record", "_from_structlog", "trace_id"):
            event_dict.pop(key, None)
        if event_dict:
            pairs = " ".join(f"{k}={v}" for k, v in event_dict.items())
            extras = f"  {_DIM}{pairs}{_RESET}"

        return f"{col_time} {_SEP} {col_file} {_SEP} {col_level} {_SEP} {col_event}{extras}"


# ── Public API ───────────────────────────────────────────────────────────


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structured, coloured column logging for the entire application.

    Parameters
    ----------
    level:
        Root log level string (e.g. "DEBUG", "INFO", "WARNING").
    fmt:
        Format type: "json" for JSON lines, "text" for colored columns.
    """
    # Enable ANSI escape processing on Windows 10+ consoles
    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # Enable ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x0004)
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            # Fallback: try to enable via simple subprocess if ctypes fails
            import subprocess
            subprocess.run("", shell=True, check=False)  # nosec B602, B604 - Windows ANSI fallback

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        mask_sensitive_processor,  # Mask sensitive data
    ]

    if fmt == "json":
        # JSON format for production
        final_processors: list[structlog.types.Processor] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Colored column format for development
        final_processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            ColumnRenderer(),
        ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=final_processors,
    )

    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Suppress noisy third-party loggers
    for name in ("httpx", "httpcore", "neo4j", "urllib3", "asyncio", "aiosqlite"):
        logging.getLogger(name).setLevel(logging.WARNING)


# ── Request-scoped context (for trace_id injection) ──────────────────────
# Keep the ContextVar for backward compatibility with existing middleware

from contextvars import ContextVar

current_trace_id: ContextVar[str] = ContextVar("current_trace_id", default="-")

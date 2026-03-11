"""Cache key generation with normalization."""

from __future__ import annotations

import hashlib
import re

_STRIP_RE = re.compile(r"[^\w\s]")


def normalize_question(question: str) -> str:
    """Lower-case, strip punctuation/whitespace, collapse spaces."""
    q = question.lower().strip()
    q = _STRIP_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def response_key(question: str) -> str:
    """Key for the full response cache."""
    digest = hashlib.sha256(normalize_question(question).encode()).hexdigest()
    return f"response:{digest}"


def strategy_key(question: str) -> str:
    """Key for strategy-decision cache."""
    digest = hashlib.sha256(normalize_question(question).encode()).hexdigest()
    return f"strategy:{digest}"


def discovery_key(question: str) -> str:
    """Key for discovery-result cache."""
    digest = hashlib.sha256(normalize_question(question).encode()).hexdigest()
    return f"discovery:{digest}"

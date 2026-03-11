"""Prompt injection detection and input sanitization."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that indicate a prompt injection attempt
_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I), "override_instructions"),
    (re.compile(r"ignore\s+(all\s+)?above", re.I), "override_instructions"),
    (re.compile(r"disregard\s+(all\s+)?previous", re.I), "override_instructions"),
    (re.compile(r"forget\s+(all\s+)?(your\s+)?instructions", re.I), "override_instructions"),
    (re.compile(r"you\s+are\s+now\s+", re.I), "role_override"),
    (re.compile(r"act\s+as\s+(a\s+)?", re.I), "role_override"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I), "role_override"),
    (re.compile(r"system\s*:\s*", re.I), "system_prompt"),
    (re.compile(r"<\s*system\s*>", re.I), "system_prompt"),
    (re.compile(r"\b(CREATE|MERGE|DELETE|DROP|SET|REMOVE)\b.*\b(RETURN|SET|WHERE)\b", re.I), "cypher_write"),
    (re.compile(r"CALL\s*\{", re.I), "cypher_subquery"),
]


def detect_injection(text: str) -> tuple[bool, str]:
    """Check *text* for prompt injection patterns.

    Returns ``(is_suspicious, reason)`` — reason is empty if not suspicious.
    """
    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning("Prompt injection detected: pattern=%s input=%.100s", reason, text)
            return True, reason
    return False, ""


def sanitize(text: str) -> str:
    """Strip the most dangerous injection patterns while preserving the question."""
    # Remove attempts to inject system/assistant role markers
    text = re.sub(r"(?i)(system|assistant|human)\s*:\s*", "", text)
    # Remove XML-like tag injections
    text = re.sub(r"<\s*/?\s*(system|prompt|instruction)\s*>", "", text, flags=re.I)
    return text.strip()


def wrap_user_input(question: str) -> str:
    """Wrap user input in delimiters for safe prompt inclusion."""
    return f"<<<USER_INPUT>>>\n{question}\n<<<END_USER_INPUT>>>"

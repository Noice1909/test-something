"""Context window management — prevents prompts from exceeding model limits."""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Estimate token count.

    Uses tiktoken for OpenAI models; falls back to a ``len/4`` heuristic.
    """
    if settings.use_openai:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(settings.openai_model)
            return len(enc.encode(text))
        except Exception as exc:
            # tiktoken not installed or model not found, fallback to heuristic
            import logging
            logging.getLogger(__name__).debug("tiktoken encoding failed, using heuristic: %s", exc)
    # Rough heuristic: ~4 chars per token for English text
    return len(text) // 4


def get_max_prompt_tokens() -> int:
    """Max tokens available for the prompt (context window minus response reserve)."""
    window = settings.llm_context_window
    reserve = settings.llm_response_reserve
    return int(window * (1 - reserve))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* so it fits within *max_tokens*."""
    current = count_tokens(text)
    if current <= max_tokens:
        return text
    # Approximate character budget
    ratio = max_tokens / max(current, 1)
    char_limit = int(len(text) * ratio * 0.95)  # small safety margin
    return text[:char_limit] + "\n... (truncated)"


def fit_to_budget(
    sections: dict[str, str],
    budgets: dict[str, float] | None = None,
) -> dict[str, str]:
    """Proportionally truncate *sections* to fit within the context window.

    *budgets* maps section name → fraction of total budget (0.0–1.0).
    If not provided, uses default allocation.
    """
    if budgets is None:
        budgets = {
            "system": 0.30,
            "schema": 0.25,
            "discoveries": 0.15,
            "history": 0.15,
            "reference": 0.15,
        }

    max_tokens = get_max_prompt_tokens()
    result: dict[str, str] = {}

    for name, text in sections.items():
        fraction = budgets.get(name, 0.10)
        token_budget = int(max_tokens * fraction)
        result[name] = truncate_to_tokens(text, token_budget)

    total = sum(count_tokens(v) for v in result.values())
    if total > max_tokens:
        logger.warning(
            "Prompt still over budget after truncation: %d / %d tokens",
            total, max_tokens,
        )

    return result

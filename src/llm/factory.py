"""
LLM provider factory — plug-and-play.

The entire application references LLMs only through get_llm_from_settings().
To swap from Ollama to your internal provider, update this file alone.
"""
from __future__ import annotations

import structlog
from functools import lru_cache

from langchain_ollama import ChatOllama

from src.config import Settings

logger = structlog.get_logger()


@lru_cache(maxsize=1)
def get_llm(
    base_url: str,
    model: str,
    temperature: float,
) -> ChatOllama:
    """
    Return a cached ChatOllama instance.

    Parameters are explicit (not a Settings object) so @lru_cache can hash them correctly.

    Parameters
    ----------
    base_url:
        Ollama server base URL, e.g. ``http://localhost:11434``.
    model:
        Model tag, e.g. ``qwen2.5:latest``.
    temperature:
        Sampling temperature — ``0.0`` is deterministic (best for Cypher generation).
    """
    logger.info("llm_factory_creating", model=model, base_url=base_url, temperature=temperature)
    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def get_llm_from_settings(settings: Settings) -> ChatOllama:
    """Return an LLM instance configured from application settings."""
    return get_llm(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
        temperature=settings.OLLAMA_TEMPERATURE,
    )

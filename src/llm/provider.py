"""LLM provider — single source of truth for creating and caching LLM clients.

Usage::

    from src.llm import create_llm, get_llm

    # During startup — create and cache the LLM client
    llm = create_llm()

    # Anywhere else — retrieve the cached instance
    llm = get_llm()
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.config import settings

logger = logging.getLogger(__name__)

# Module-level cached instance
_llm: BaseChatModel | None = None


def create_llm(
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Build a LangChain chat model and cache it as the singleton.

    Parameters
    ----------
    provider:
        ``"openai"`` or ``"ollama"``.  Defaults to auto-detect from settings
        (OpenAI if ``OPENAI_API_KEY`` is set, Ollama otherwise).
    model:
        Model name override.  Defaults to the value in settings.
    temperature:
        Sampling temperature.  Defaults to 0 (deterministic).

    Returns
    -------
    BaseChatModel
        The configured LangChain chat model.
    """
    global _llm

    use_openai = provider == "openai" if provider else settings.use_openai

    if use_openai:
        from langchain_openai import ChatOpenAI

        model_name = model or settings.openai_model
        logger.info("LLM provider: OpenAI  model: %s", model_name)
        _llm = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
            temperature=temperature,
        )
    else:
        from langchain_ollama import ChatOllama

        model_name = model or settings.ollama_model
        base_url = settings.ollama_base_url
        logger.info("LLM provider: Ollama  model: %s @ %s", model_name, base_url)
        _llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    return _llm


def get_llm() -> BaseChatModel:
    """Return the cached LLM singleton.  Raises if ``create_llm`` was not called."""
    if _llm is None:
        raise RuntimeError(
            "LLM has not been initialised. "
            "Call `create_llm()` during application startup."
        )
    return _llm

"""LLM wrapper that applies circuit breaker protection."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.resilience.circuit_breaker import CircuitOpenError, llm_breaker


class ResilientLLM:
    """Thin proxy around a ``BaseChatModel`` that trips the LLM circuit breaker."""

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def ainvoke(self, input: Any, **kwargs: Any) -> BaseMessage:  # noqa: A002
        try:
            return await llm_breaker.call(self._llm.ainvoke, input, **kwargs)
        except Exception as exc:
            if "circuit breaker" in str(exc).lower() or isinstance(exc, Exception) and type(exc).__name__ == "CircuitBreakerError":
                raise CircuitOpenError("llm", llm_breaker.reset_timeout) from exc
            raise

    # Passthrough for non-async and attribute access
    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)

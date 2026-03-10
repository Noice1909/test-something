"""LLM provider package.

Centralizes LLM client creation so every module uses the same instance
and configuration.  Import ``create_llm`` or ``get_llm`` from here.
"""

from src.llm.provider import create_llm, get_llm

__all__ = ["create_llm", "get_llm"]

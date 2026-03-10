"""Shared utilities for agent specialists."""

from __future__ import annotations

from typing import Any


def extract_text(response: Any) -> str:
    """Safely extract a plain string from an LLM response.

    LangChain's ``BaseChatModel.ainvoke`` returns a message whose ``.content``
    can be ``str | list[str | dict]``.  This helper normalises it to a plain
    ``str`` so downstream JSON parsing code works without type errors.
    """
    content: Any = getattr(response, "content", None)
    if content is None:
        return str(response)
    if isinstance(content, str):
        return content
    # content is a list of content blocks (multimodal)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            parts.append(item.get("text", str(item)))
        else:
            parts.append(str(item))
    return "".join(parts)

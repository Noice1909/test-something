"""Base tool class — LangChain-compatible with ``read_only`` metadata."""

from __future__ import annotations

from langchain_core.tools import BaseTool as _LCBaseTool


class AgentTool(_LCBaseTool):
    """All project tools extend this.

    The ``read_only`` flag tells the tool manager whether a tool is safe
    for parallel execution (True) or must be run sequentially (False).
    """

    read_only: bool = False

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.config import Settings

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


async def create_checkpointer(settings: Settings) -> BaseCheckpointSaver:
    """
    Return the appropriate LangGraph checkpointer based on environment.
    - local:    MemorySaver  (in-memory, zero-config, no threading issues)
    - deployed: AsyncRedisSaver   (shared across instances)
    """
    if settings.is_local:
        from langgraph.checkpoint.memory import MemorySaver

        # MemorySaver is simple, thread-safe, and perfect for local development
        # For production, use Redis (AsyncRedisSaver below)
        return MemorySaver()
    else:
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        saver = AsyncRedisSaver.from_conn_string(settings.REDIS_URL)
        instance = await saver.__aenter__()
        return instance

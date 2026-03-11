"""Checkpoint store for multi-turn conversation state."""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Snapshot of a single conversation turn."""

    conversation_id: str
    turn_number: int
    question: str
    answer: str
    strategy: str
    discoveries: list[dict] = field(default_factory=list)
    schema_selection: dict = field(default_factory=dict)
    generated_query: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Abstract store ──


class CheckpointStore(ABC):

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    async def load(self, conversation_id: str) -> list[Checkpoint]: ...

    @abstractmethod
    async def latest(self, conversation_id: str) -> Checkpoint | None: ...

    @abstractmethod
    async def delete(self, conversation_id: str) -> None: ...

    async def close(self) -> None:
        """Cleanup hook."""


# ── In-memory store ──


class InMemoryCheckpointStore(CheckpointStore):
    """LRU-evicting in-memory store."""

    def __init__(self, max_conversations: int = 10000) -> None:
        self._max = max_conversations
        self._store: OrderedDict[str, list[Checkpoint]] = OrderedDict()

    async def save(self, checkpoint: Checkpoint) -> None:
        cid = checkpoint.conversation_id
        if cid in self._store:
            self._store.move_to_end(cid)
        else:
            self._store[cid] = []
        self._store[cid].append(checkpoint)
        # Evict oldest if over limit
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    async def load(self, conversation_id: str) -> list[Checkpoint]:
        return list(self._store.get(conversation_id, []))

    async def latest(self, conversation_id: str) -> Checkpoint | None:
        turns = self._store.get(conversation_id, [])
        return turns[-1] if turns else None

    async def delete(self, conversation_id: str) -> None:
        self._store.pop(conversation_id, None)


# ── SQLite store ──


class SQLiteCheckpointStore(CheckpointStore):
    """Persistent SQLite checkpoint store."""

    def __init__(self, db_path: str = "checkpoints.db") -> None:
        self._db_path = db_path
        self._db: Any = None

    async def _ensure_db(self) -> Any:
        if self._db is None:
            import aiosqlite
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints "
                "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "conversation_id TEXT NOT NULL, turn_number INTEGER, "
                "data TEXT NOT NULL, timestamp REAL)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_cp_conv ON checkpoints(conversation_id)"
            )
            await self._db.commit()
        return self._db

    async def save(self, checkpoint: Checkpoint) -> None:
        db = await self._ensure_db()
        await db.execute(
            "INSERT INTO checkpoints (conversation_id, turn_number, data, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (checkpoint.conversation_id, checkpoint.turn_number,
             json.dumps(checkpoint.to_dict(), default=str), checkpoint.timestamp),
        )
        await db.commit()

    async def load(self, conversation_id: str) -> list[Checkpoint]:
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT data FROM checkpoints WHERE conversation_id = ? ORDER BY turn_number",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
        return [Checkpoint.from_dict(json.loads(row[0])) for row in rows]

    async def latest(self, conversation_id: str) -> Checkpoint | None:
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT data FROM checkpoints WHERE conversation_id = ? ORDER BY turn_number DESC LIMIT 1",
            (conversation_id,),
        )
        row = await cursor.fetchone()
        return Checkpoint.from_dict(json.loads(row[0])) if row else None

    async def delete(self, conversation_id: str) -> None:
        db = await self._ensure_db()
        await db.execute("DELETE FROM checkpoints WHERE conversation_id = ?", (conversation_id,))
        await db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None


# ── Redis store ──


class RedisCheckpointStore(CheckpointStore):
    """Redis-backed checkpoint store."""

    def __init__(self, url: str = "redis://localhost:6379/0", ttl: int = 1800) -> None:
        self._url = url
        self._ttl = ttl
        self._redis: Any = None

    async def _ensure_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    async def save(self, checkpoint: Checkpoint) -> None:
        r = await self._ensure_redis()
        key = f"checkpoint:{checkpoint.conversation_id}"
        await r.rpush(key, json.dumps(checkpoint.to_dict(), default=str))
        await r.expire(key, self._ttl)

    async def load(self, conversation_id: str) -> list[Checkpoint]:
        r = await self._ensure_redis()
        key = f"checkpoint:{conversation_id}"
        raw = await r.lrange(key, 0, -1)
        return [Checkpoint.from_dict(json.loads(item)) for item in raw]

    async def latest(self, conversation_id: str) -> Checkpoint | None:
        r = await self._ensure_redis()
        key = f"checkpoint:{conversation_id}"
        raw = await r.lindex(key, -1)
        return Checkpoint.from_dict(json.loads(raw)) if raw else None

    async def delete(self, conversation_id: str) -> None:
        r = await self._ensure_redis()
        await r.delete(f"checkpoint:{conversation_id}")

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


# ── Factory ──


def create_checkpoint_store() -> CheckpointStore:
    backend = settings.checkpoint_backend.lower()
    if backend == "redis":
        return RedisCheckpointStore(url=settings.cache_redis_url, ttl=settings.checkpoint_ttl)
    if backend == "sqlite":
        return SQLiteCheckpointStore(db_path="checkpoints.db")
    return InMemoryCheckpointStore(max_conversations=settings.checkpoint_max_conversations)


def generate_conversation_id() -> str:
    return f"conv-{uuid.uuid4().hex[:12]}"

"""Unified cache manager with pluggable backends (memory / sqlite / redis)."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


# ── Backend protocol ──


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    async def close(self) -> None:
        """Cleanup hook (override for Redis / SQLite)."""


# ── In-memory backend ──


class InMemoryCacheBackend(CacheBackend):
    """TTL-aware dict cache (single-process only)."""

    def __init__(self, max_size: int = 1000) -> None:
        from cachetools import TTLCache
        self._caches: dict[int, TTLCache] = {}
        self._max_size = max_size
        self._default_ttl = 300

    def _get_cache(self, ttl: int) -> Any:
        if ttl not in self._caches:
            from cachetools import TTLCache
            self._caches[ttl] = TTLCache(maxsize=self._max_size, ttl=ttl)
        return self._caches[ttl]

    async def get(self, key: str) -> Any | None:
        for cache in self._caches.values():
            if key in cache:
                return cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: int) -> None:
        cache = self._get_cache(ttl)
        cache[key] = value

    async def delete(self, key: str) -> None:
        for cache in self._caches.values():
            cache.pop(key, None)

    async def clear(self) -> None:
        for cache in self._caches.values():
            cache.clear()


# ── SQLite backend ──


class SQLiteCacheBackend(CacheBackend):
    """Persistent SQLite cache with TTL expiration."""

    def __init__(self, db_path: str = "cache.db") -> None:
        self._db_path = db_path
        self._db: Any = None

    async def _ensure_db(self) -> Any:
        if self._db is None:
            import aiosqlite
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value TEXT, expires_at REAL)"
            )
            await self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)"
            )
            await self._db.commit()
        return self._db

    async def get(self, key: str) -> Any | None:
        db = await self._ensure_db()
        now = time.time()
        cursor = await db.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, now),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    async def set(self, key: str, value: Any, ttl: int) -> None:
        db = await self._ensure_db()
        expires_at = time.time() + ttl
        await db.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, json.dumps(value, default=str), expires_at),
        )
        await db.commit()

    async def delete(self, key: str) -> None:
        db = await self._ensure_db()
        await db.execute("DELETE FROM cache WHERE key = ?", (key,))
        await db.commit()

    async def clear(self) -> None:
        db = await self._ensure_db()
        await db.execute("DELETE FROM cache")
        await db.commit()

    async def close(self) -> None:
        if self._db is not None:
            # Cleanup expired entries before closing
            now = time.time()
            await self._db.execute("DELETE FROM cache WHERE expires_at <= ?", (now,))
            await self._db.commit()
            await self._db.close()
            self._db = None


# ── Redis backend ──


class RedisCacheBackend(CacheBackend):
    """Redis-backed cache for multi-instance deployments."""

    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._redis: Any = None

    async def _ensure_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._url, decode_responses=True)
        return self._redis

    async def get(self, key: str) -> Any | None:
        r = await self._ensure_redis()
        raw = await r.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        r = await self._ensure_redis()
        await r.setex(key, ttl, json.dumps(value, default=str))

    async def delete(self, key: str) -> None:
        r = await self._ensure_redis()
        await r.delete(key)

    async def clear(self) -> None:
        r = await self._ensure_redis()
        await r.flushdb()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


# ── Factory ──


def create_cache_backend() -> CacheBackend:
    """Create a cache backend based on settings."""
    backend = settings.cache_backend.lower()
    if backend == "redis":
        logger.info("Cache backend: Redis (%s)", settings.cache_redis_url)
        return RedisCacheBackend(url=settings.cache_redis_url)
    if backend == "sqlite":
        logger.info("Cache backend: SQLite (%s)", settings.cache_sqlite_path)
        return SQLiteCacheBackend(db_path=settings.cache_sqlite_path)
    logger.info("Cache backend: In-memory (max_size=%d)", settings.cache_max_size)
    return InMemoryCacheBackend(max_size=settings.cache_max_size)


# ── CacheManager (high-level API) ──


class CacheManager:
    """Application-level cache with hit/miss tracking."""

    def __init__(self, backend: CacheBackend | None = None) -> None:
        self._backend = backend or create_cache_backend()
        self.hits = 0
        self.misses = 0

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    async def get(self, key: str) -> Any | None:
        value = await self._backend.get(key)
        if value is not None:
            self.hits += 1
            logger.debug("Cache HIT: %s", key)
        else:
            self.misses += 1
            logger.debug("Cache MISS: %s", key)
        return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self._backend.set(key, value, ttl or settings.cache_response_ttl)

    async def delete(self, key: str) -> None:
        await self._backend.delete(key)

    async def clear(self) -> None:
        await self._backend.clear()

    async def close(self) -> None:
        await self._backend.close()

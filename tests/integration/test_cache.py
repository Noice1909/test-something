"""Tests for the caching layer."""

from __future__ import annotations

import pytest

from src.cache.cache_manager import CacheManager, InMemoryCacheBackend
from src.cache.cache_keys import response_key, strategy_key, normalize_question


class TestCacheKeys:

    def test_normalize_strips_punctuation(self):
        assert normalize_question("Hello, World!") == "hello world"

    def test_normalize_collapses_whitespace(self):
        assert normalize_question("  hello   world  ") == "hello world"

    def test_response_key_deterministic(self):
        k1 = response_key("hello world")
        k2 = response_key("hello world")
        assert k1 == k2

    def test_response_key_different_for_different_questions(self):
        k1 = response_key("question one")
        k2 = response_key("question two")
        assert k1 != k2

    def test_strategy_key_prefix(self):
        key = strategy_key("test")
        assert key.startswith("strategy:")


@pytest.mark.asyncio
class TestInMemoryCache:

    async def test_set_and_get(self):
        cache = CacheManager(InMemoryCacheBackend())
        await cache.set("key1", {"data": "value"}, ttl=60)
        result = await cache.get("key1")
        assert result == {"data": "value"}

    async def test_miss_returns_none(self):
        cache = CacheManager(InMemoryCacheBackend())
        result = await cache.get("nonexistent")
        assert result is None

    async def test_hit_miss_tracking(self):
        cache = CacheManager(InMemoryCacheBackend())
        await cache.get("miss1")
        await cache.set("hit1", "val", ttl=60)
        await cache.get("hit1")
        assert cache.hits == 1
        assert cache.misses == 1

    async def test_delete(self):
        cache = CacheManager(InMemoryCacheBackend())
        await cache.set("key1", "val", ttl=60)
        await cache.delete("key1")
        result = await cache.get("key1")
        assert result is None

    async def test_clear(self):
        cache = CacheManager(InMemoryCacheBackend())
        await cache.set("a", 1, ttl=60)
        await cache.set("b", 2, ttl=60)
        await cache.clear()
        assert await cache.get("a") is None
        assert await cache.get("b") is None

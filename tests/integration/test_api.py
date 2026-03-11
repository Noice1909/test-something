"""Integration tests for API endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_root_health(self, test_client: AsyncClient):
        resp = await test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    async def test_agentic_health(self, test_client: AsyncClient):
        resp = await test_client.get("/api/v1/agentic/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["supervisor_initialized"] is True
        assert "components" in data
        assert "circuit_breakers" in data


@pytest.mark.asyncio
class TestChatEndpoint:

    async def test_chat_success(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/v1/agentic/chat",
            json={"question": "How many applications?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["answer"] == "Test answer"

    async def test_chat_with_trace_id(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/v1/agentic/chat",
            json={"question": "List all nodes", "trace_id": "custom-trace"},
        )
        assert resp.status_code == 200

    async def test_chat_empty_question_rejected(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/v1/agentic/chat",
            json={"question": ""},
        )
        assert resp.status_code == 422

    async def test_chat_too_long_question_rejected(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/v1/agentic/chat",
            json={"question": "x" * 5000},
        )
        assert resp.status_code == 422

    async def test_chat_with_conversation_id(self, test_client: AsyncClient):
        resp = await test_client.post(
            "/api/v1/agentic/chat",
            json={"question": "test", "conversation_id": "conv-test123"},
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
class TestAuthEndpoint:

    async def test_auth_disabled_allows_request(self, test_client: AsyncClient):
        """When auth_api_key is empty, requests pass through."""
        resp = await test_client.post(
            "/api/v1/agentic/chat",
            json={"question": "test question"},
        )
        assert resp.status_code == 200

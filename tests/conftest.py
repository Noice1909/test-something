"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.agents.base import AgenticResponse


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db() -> AsyncMock:
    """A mock AbstractDatabase."""
    db = AsyncMock()
    db.execute_read.return_value = [{"name": "TestNode", "id": "1"}]
    db.health_check.return_value = {"healthy": True}
    db.get_schema.return_value = {
        "labels": ["Application", "Domain"],
        "relationship_types": ["INTERACTS_WITH"],
        "property_keys": ["name", "id"],
        "label_properties": {"Application": ["name", "id"]},
        "relationship_patterns": [{"from": "Application", "type": "INTERACTS_WITH", "to": "Domain"}],
    }
    return db


@pytest.fixture
def mock_llm() -> MagicMock:
    """A mock LLM that returns canned responses."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content='{"strategy": "direct_query", "reasoning": "test"}'))
    return llm


@pytest.fixture
def mock_supervisor(mock_db: AsyncMock) -> MagicMock:
    """A mock Supervisor."""
    supervisor = MagicMock()
    supervisor._db = mock_db
    supervisor.process_question = AsyncMock(
        return_value=AgenticResponse(
            answer="Test answer",
            strategy_used="direct_query",
            attempts=1,
            success=True,
            trace_id="test-trace",
            specialist_log=[],
            cypher_attempts=[],
        )
    )
    return supervisor


@pytest_asyncio.fixture
async def test_client(mock_supervisor: MagicMock) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with mocked supervisor."""
    from src.agents.supervisor_factory import SupervisorFactory

    # Inject mock
    import src.agents.supervisor_factory as sf
    sf._instance = mock_supervisor

    from src.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    sf._instance = None

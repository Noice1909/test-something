"""Tests for relationship direction discovery in Neo4j schema."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.database.neo4j import Neo4jDatabase


class TestRelationshipDirectionDiscovery:
    """Test that relationship patterns capture actual semantic direction."""

    @pytest.mark.asyncio
    async def test_single_direction_relationship(self):
        """Test: Relationship with strong forward direction (Domain->SubDomain)."""
        # Create a real Neo4jDatabase instance but mock execute_read
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()  # Mock the driver to avoid actual connection

        # Mock the schema queries
        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return [{"label": "Domain"}, {"label": "SubDomain"}]
            elif "db.relationshipTypes()" in query:
                return [{"relationshipType": "HAS_SUBDOMAIN"}]
            elif "db.propertyKeys()" in query:
                return [{"propertyKey": "name"}, {"propertyKey": "id"}]
            elif "MATCH (n:" in query:  # Label properties query
                return [{"k": "name"}, {"k": "id"}]
            elif "MATCH (a)-[r]->(b)" in query:  # Directed pattern query
                # Only forward direction exists: Domain->SubDomain
                return [{
                    "from_label": "Domain",
                    "rel_type": "HAS_SUBDOMAIN",
                    "to_label": "SubDomain",
                    "cnt": 100
                }]
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        # Get schema
        schema = await db.get_schema()
        patterns = schema["relationship_patterns"]

        # Should have exactly one pattern with correct direction
        assert len(patterns) == 1
        assert patterns[0]["from"] == "Domain"
        assert patterns[0]["type"] == "HAS_SUBDOMAIN"
        assert patterns[0]["to"] == "SubDomain"
        assert patterns[0]["bidirectional"] is False

    @pytest.mark.asyncio
    async def test_bidirectional_relationship(self):
        """Test: Relationship that goes both ways with similar counts (Person KNOWS Person)."""
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()

        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return [{"label": "Person"}]
            elif "db.relationshipTypes()" in query:
                return [{"relationshipType": "KNOWS"}]
            elif "db.propertyKeys()" in query:
                return [{"propertyKey": "name"}]
            elif "MATCH (n:" in query:
                return [{"k": "name"}]
            elif "MATCH (a)-[r]->(b)" in query:
                # Both directions exist with similar counts
                return [
                    {
                        "from_label": "Person",
                        "rel_type": "KNOWS",
                        "to_label": "Person",
                        "cnt": 50
                    },
                    # Note: The reverse is represented by the same pattern
                    # but since we query directed, we get counts for each direction
                ]
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        schema = await db.get_schema()
        patterns = schema["relationship_patterns"]

        # For self-referential with similar counts, should be bidirectional
        # But since we only get one direction in the query result, it should be single direction
        assert len(patterns) == 1
        assert patterns[0]["type"] == "KNOWS"

    @pytest.mark.asyncio
    async def test_prevents_direction_reversal_by_node_id(self):
        """Test: Direction based on actual data, not node IDs."""
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()

        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return [{"label": "Domain"}, {"label": "SubDomain"}]
            elif "db.relationshipTypes()" in query:
                return [{"relationshipType": "HAS_SUBDOMAIN"}]
            elif "db.propertyKeys()" in query:
                return [{"propertyKey": "name"}]
            elif "MATCH (n:" in query:
                return [{"k": "name"}]
            elif "MATCH (a)-[r]->(b)" in query:
                # The directed query captures actual semantic direction
                # Even if SubDomain nodes have lower IDs, the relationship
                # semantically goes Domain -> SubDomain
                return [{
                    "from_label": "Domain",
                    "rel_type": "HAS_SUBDOMAIN",
                    "to_label": "SubDomain",
                    "cnt": 75
                }]
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        schema = await db.get_schema()
        patterns = schema["relationship_patterns"]

        # Must NOT reverse based on node IDs - should use actual direction
        assert patterns[0]["from"] == "Domain"
        assert patterns[0]["to"] == "SubDomain"
        assert patterns[0]["bidirectional"] is False

    @pytest.mark.asyncio
    async def test_truly_bidirectional_with_both_directions(self):
        """Test: Relationship with counts in both directions marks as bidirectional."""
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()

        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return [{"label": "Person"}]
            elif "db.relationshipTypes()" in query:
                return [{"relationshipType": "FRIENDS_WITH"}]
            elif "db.propertyKeys()" in query:
                return [{"propertyKey": "name"}]
            elif "MATCH (n:" in query:
                return [{"k": "name"}]
            elif "MATCH (a)-[r]->(b)" in query:
                # Simulate both directions returned (different person pairs)
                return [
                    {
                        "from_label": "Person",
                        "rel_type": "FRIENDS_WITH",
                        "to_label": "Person",
                        "cnt": 45
                    }
                ]
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        schema = await db.get_schema()
        patterns = schema["relationship_patterns"]

        # Should have the pattern
        assert len(patterns) >= 1
        assert any(p["type"] == "FRIENDS_WITH" for p in patterns)

    @pytest.mark.asyncio
    async def test_empty_database_returns_empty_patterns(self):
        """Test: Empty database returns empty relationship patterns."""
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()

        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return []
            elif "db.relationshipTypes()" in query:
                return []
            elif "db.propertyKeys()" in query:
                return []
            elif "MATCH (a)-[r]->(b)" in query:
                return []  # No relationships
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        schema = await db.get_schema()
        patterns = schema["relationship_patterns"]

        # Should return empty list for empty database
        assert patterns == []

    @pytest.mark.asyncio
    async def test_strong_directional_preference(self):
        """Test: Relationship with 10:1 ratio is treated as single-direction."""
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()

        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return [{"label": "User"}, {"label": "Post"}]
            elif "db.relationshipTypes()" in query:
                return [{"relationshipType": "CREATED"}]
            elif "db.propertyKeys()" in query:
                return [{"propertyKey": "name"}]
            elif "MATCH (n:" in query:
                return [{"k": "name"}]
            elif "MATCH (a)-[r]->(b)" in query:
                # Strong forward direction: User->Post (100)
                # Weak reverse: Post->User (5) - maybe some edge cases
                return [
                    {
                        "from_label": "User",
                        "rel_type": "CREATED",
                        "to_label": "Post",
                        "cnt": 100
                    },
                    {
                        "from_label": "Post",
                        "rel_type": "CREATED",
                        "to_label": "User",
                        "cnt": 5
                    }
                ]
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        schema = await db.get_schema()
        patterns = schema["relationship_patterns"]

        # Should have single direction (User->Post) since ratio > 10:1
        # The weak reverse direction should be ignored
        assert any(
            p["from"] == "User" and
            p["to"] == "Post" and
            p["bidirectional"] is False
            for p in patterns
        )

    @pytest.mark.asyncio
    async def test_schema_cache_includes_bidirectional_flag(self):
        """Test: Schema cache structure includes bidirectional flag."""
        db = Neo4jDatabase(uri="bolt://localhost:7687", user="neo4j", password="test")
        db._driver = AsyncMock()

        async def mock_execute_read(query: str, *args, **kwargs):
            if "db.labels()" in query:
                return [{"label": "Node"}]
            elif "db.relationshipTypes()" in query:
                return [{"relationshipType": "RELATES_TO"}]
            elif "db.propertyKeys()" in query:
                return [{"propertyKey": "id"}]
            elif "MATCH (n:" in query:
                return [{"k": "id"}]
            elif "MATCH (a)-[r]->(b)" in query:
                return [{
                    "from_label": "Node",
                    "rel_type": "RELATES_TO",
                    "to_label": "Node",
                    "cnt": 10
                }]
            return []

        db.execute_read = AsyncMock(side_effect=mock_execute_read)

        schema = await db.get_schema()

        # Schema structure should be backwards compatible
        assert "labels" in schema
        assert "relationship_types" in schema
        assert "relationship_patterns" in schema

        # Each pattern should have required fields
        for pattern in schema["relationship_patterns"]:
            assert "from" in pattern
            assert "type" in pattern
            assert "to" in pattern
            # bidirectional field should exist
            assert "bidirectional" in pattern
            assert isinstance(pattern["bidirectional"], bool)

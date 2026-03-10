"""Database abstraction layer."""

from src.database.abstract import AbstractDatabase
from src.database.neo4j import Neo4jDatabase

__all__ = ["AbstractDatabase", "Neo4jDatabase"]

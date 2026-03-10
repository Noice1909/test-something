"""Abstract database interface for graph databases."""

from __future__ import annotations

import abc
from typing import Any


class AbstractDatabase(abc.ABC):
    """Database-agnostic interface for graph query operations.

    Concrete subclasses implement this for Neo4j, ArangoDB, Neptune, etc.
    """

    # ── lifecycle ──

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the database connection."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the database connection."""

    @abc.abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Return a health-check dict with at least ``{"healthy": bool}``."""

    # ── schema introspection ──

    @abc.abstractmethod
    async def get_schema(self) -> dict[str, Any]:
        """Return the full graph schema.

        Expected keys (at minimum):
        - ``labels``: list of node label strings
        - ``relationship_types``: list of relationship type strings
        - ``property_keys``: list of property key strings
        """

    # ── query execution ──

    @abc.abstractmethod
    async def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a **read-only** query and return result rows as dicts."""

    @abc.abstractmethod
    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a **write** query and return result rows as dicts."""

    # ── convenience ──

    @abc.abstractmethod
    async def get_labels(self) -> list[str]:
        """Return all node labels in the database."""

    @abc.abstractmethod
    async def get_relationship_types(self) -> list[str]:
        """Return all relationship types in the database."""

    @abc.abstractmethod
    async def get_property_keys(self) -> list[str]:
        """Return all property keys used across the database."""

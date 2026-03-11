"""Neo4j concrete implementation of AbstractDatabase.

Features:
  • URI scheme rewriting — ``neo4j+s://`` → ``neo4j+ssc://`` when
    ``NEO4J_SKIP_TLS_VERIFY=true`` (required for AuraDB)
  • Startup retry loop with exponential back-off
  • ``ensure_connected()`` — verifies driver is alive, auto-reconnects
  • Schema caching with configurable TTL
  • Read-only safety validation
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from src.config import settings
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)

_WRITE_KEYWORDS = frozenset({
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP",
    "CALL {", "FOREACH",
})

# URI scheme constants
_NEO4J_TLS_SCHEME = "neo4j+s://"
_NEO4J_TLS_UNVERIFIED_SCHEME = "neo4j+ssc://"
_BOLT_TLS_SCHEME = "bolt+s://"
_BOLT_TLS_UNVERIFIED_SCHEME = "bolt+ssc://"


def _resolve_uri(uri: str, skip_tls_verify: bool) -> str:
    """Apply optional TLS-verification rewrite and return the final URI."""
    if skip_tls_verify:
        if uri.startswith(_NEO4J_TLS_SCHEME):
            uri = _NEO4J_TLS_UNVERIFIED_SCHEME + uri[len(_NEO4J_TLS_SCHEME):]
            logger.warning(
                "TLS cert verification DISABLED (neo4j+ssc://). "
                "Only use this for managed cloud instances (AuraDB)."
            )
        elif uri.startswith(_BOLT_TLS_SCHEME):
            uri = _BOLT_TLS_UNVERIFIED_SCHEME + uri[len(_BOLT_TLS_SCHEME):]
            logger.warning(
                "TLS cert verification DISABLED (bolt+ssc://). "
                "Only use this for managed cloud instances (AuraDB)."
            )
    else:
        if uri.startswith((_NEO4J_TLS_SCHEME, _BOLT_TLS_SCHEME)):
            logger.info(
                "TLS with full cert verification enabled. "
                "Set NEO4J_SKIP_TLS_VERIFY=true if you encounter certificate "
                "errors with a managed cloud instance (AuraDB)."
            )
    return uri


class Neo4jDatabase(AbstractDatabase):
    """Async Neo4j driver wrapper with schema caching and safety checks."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        schema_ttl: float = 300.0,
        skip_tls_verify: bool | None = None,
    ) -> None:
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._database = database or settings.neo4j_database
        self._schema_ttl = schema_ttl
        self._skip_tls_verify = (
            skip_tls_verify if skip_tls_verify is not None
            else settings.neo4j_skip_tls_verify
        )

        self._driver: AsyncDriver | None = None
        self._schema_cache: dict[str, Any] | None = None
        self._schema_ts: float = 0.0

    # ── lifecycle ──

    async def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> None:
        """Connect to Neo4j with retries and exponential back-off."""
        resolved_uri = _resolve_uri(self._uri, self._skip_tls_verify)

        logger.info(
            "Connecting to Neo4j: uri=%s user=%s database=%s",
            resolved_uri, self._user, self._database,
        )

        for attempt in range(1, max_retries + 1):
            try:
                self._driver = AsyncGraphDatabase.driver(
                    resolved_uri,
                    auth=(self._user, self._password),
                    max_connection_pool_size=settings.neo4j_pool_max_size,
                    connection_acquisition_timeout=settings.neo4j_pool_acquisition_timeout,
                    max_connection_lifetime=settings.neo4j_max_connection_lifetime,
                )
                # Verify connectivity
                await self._driver.verify_connectivity()
                logger.info(
                    "Neo4j connection established (attempt %d, database=%s)",
                    attempt, self._database,
                )
                return
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(
                        "Neo4j connection failed after %d attempts — giving up.",
                        max_retries,
                    )
                    raise
                backoff = min(retry_delay * (2 ** (attempt - 1)), 30)
                logger.warning(
                    "Neo4j attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt, max_retries, exc, backoff,
                )
                await asyncio.sleep(backoff)

    async def disconnect(self) -> None:
        """Gracefully close the Neo4j driver."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed.")

    async def ensure_connected(self) -> None:
        """Verify driver is alive; reconnect if it is not."""
        if self._driver is None:
            await self.connect()
            return
        try:
            async with self._driver.session(database=self._database) as session:
                await session.run("RETURN 1 AS ping")
        except Exception as exc:
            logger.warning("Neo4j connectivity check failed: %s — reconnecting.", exc)
            await self.reconnect()

    async def reconnect(self) -> None:
        """Tear down old driver and create a fresh connection."""
        if self._driver:
            try:
                await self._driver.close()
            except Exception:
                logger.debug("Old Neo4j driver cleanup error (ignored)", exc_info=True)
            self._driver = None
        await self.connect()
        logger.info("Neo4j reconnected successfully.")

    async def health_check(self) -> dict[str, Any]:
        """Check database health WITHOUT going through circuit breaker.

        This method bypasses the circuit breaker to test the actual
        database connection, not the circuit breaker state.
        """
        try:
            # Direct query without circuit breaker
            session = await self._get_session()
            try:
                result = await session.run("RETURN 1 AS ok")  # type: ignore[arg-type]
                data = await result.data()
                return {"healthy": True, "result": data}
            finally:
                await session.close()
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    # ── schema introspection ──

    async def get_schema(self) -> dict[str, Any]:
        now = time.time()
        if self._schema_cache and (now - self._schema_ts) < self._schema_ttl:
            return self._schema_cache

        labels = await self.get_labels()
        rel_types = await self.get_relationship_types()
        prop_keys = await self.get_property_keys()

        # Gather per-label property info
        label_properties: dict[str, list[str]] = {}
        for label in labels:
            rows = await self.execute_read(
                f"MATCH (n:`{label}`) UNWIND keys(n) AS k "
                "RETURN DISTINCT k ORDER BY k LIMIT 50"
            )
            label_properties[label] = [r["k"] for r in rows]

        # Gather relationship patterns
        rel_patterns: list[dict[str, str]] = []
        rows = await self.execute_read(
            "MATCH (a)-[r]->(b) "
            "RETURN DISTINCT labels(a)[0] AS from_label, type(r) AS rel_type, "
            "labels(b)[0] AS to_label LIMIT 200"
        )
        for row in rows:
            rel_patterns.append({
                "from": row["from_label"],
                "type": row["rel_type"],
                "to": row["to_label"],
            })

        self._schema_cache = {
            "labels": labels,
            "relationship_types": rel_types,
            "property_keys": prop_keys,
            "label_properties": label_properties,
            "relationship_patterns": rel_patterns,
        }
        self._schema_ts = now
        logger.info(
            "Schema refreshed: %d labels, %d rel types, %d props",
            len(labels), len(rel_types), len(prop_keys),
        )
        return self._schema_cache

    # ── query execution ──

    async def execute_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from src.resilience.circuit_breaker import neo4j_breaker, CircuitOpenError

        async def _run() -> list[dict[str, Any]]:
            session = await self._get_session()
            try:
                result = await session.run(query, parameters or {})  # type: ignore[arg-type]
                return await result.data()
            finally:
                await session.close()

        try:
            return await neo4j_breaker.call(_run)
        except Exception as exc:
            if "circuit breaker" in str(exc).lower() or type(exc).__name__ == "CircuitBreakerError":
                raise CircuitOpenError("neo4j", neo4j_breaker.reset_timeout) from exc
            raise

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from src.resilience.circuit_breaker import neo4j_breaker, CircuitOpenError

        async def _run() -> list[dict[str, Any]]:
            session = await self._get_session()
            try:
                result = await session.run(query, parameters or {})  # type: ignore[arg-type]
                return await result.data()
            finally:
                await session.close()

        try:
            return await neo4j_breaker.call(_run)
        except Exception as exc:
            if "circuit breaker" in str(exc).lower() or type(exc).__name__ == "CircuitBreakerError":
                raise CircuitOpenError("neo4j", neo4j_breaker.reset_timeout) from exc
            raise

    # ── convenience ──

    async def get_labels(self) -> list[str]:
        rows = await self.execute_read("CALL db.labels()")
        return [r["label"] for r in rows]

    async def get_relationship_types(self) -> list[str]:
        rows = await self.execute_read("CALL db.relationshipTypes()")
        return [r["relationshipType"] for r in rows]

    async def get_property_keys(self) -> list[str]:
        rows = await self.execute_read("CALL db.propertyKeys()")
        return [r["propertyKey"] for r in rows]

    # ── safety ──

    @staticmethod
    def is_read_only(query: str) -> bool:
        """Return True if the query appears to be read-only."""
        upper = query.upper()
        return not any(kw in upper for kw in _WRITE_KEYWORDS)

    # ── internals ──

    async def _get_session(self) -> AsyncSession:
        if not self._driver:
            raise RuntimeError("Neo4j driver not connected — call connect() first")
        return self._driver.session(database=self._database)

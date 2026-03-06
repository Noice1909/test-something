from __future__ import annotations

from typing import Any

import pybreaker
import structlog
from neo4j import Driver, RoutingControl

from src.core.exceptions import Neo4jUnavailableError

logger = structlog.get_logger()


class Neo4jService:
    def __init__(self, driver: Driver, database: str, breaker: pybreaker.CircuitBreaker) -> None:
        self.driver = driver
        self.database = database
        self.breaker = breaker

    def execute_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a read-only Cypher query. Circuit-breaker protected."""
        try:
            @self.breaker
            def _run() -> list[dict[str, Any]]:
                records, _, _ = self.driver.execute_query(
                    cypher,
                    parameters_=params or {},
                    database_=self.database,
                    routing_=RoutingControl.READ,
                )
                return [r.data() for r in records]

            return _run()
        except pybreaker.CircuitBreakerError as exc:
            raise Neo4jUnavailableError("Neo4j circuit breaker is open") from exc

    def explain(self, cypher: str) -> bool:
        """Run EXPLAIN on a query to validate without executing. Returns True if valid."""
        try:
            self.execute_read(f"EXPLAIN {cypher}")
            return True
        except Neo4jUnavailableError:
            raise
        except Exception:
            return False

    def verify_connectivity(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            logger.warning("neo4j_connectivity_check_failed")
            return False

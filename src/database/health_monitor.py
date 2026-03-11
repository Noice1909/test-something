"""Background database health monitor — auto-recovers crashed connections."""

from __future__ import annotations

import asyncio
import logging

from src.config import settings
from src.database.abstract import AbstractDatabase

logger = logging.getLogger(__name__)


class DatabaseHealthMonitor:
    """Periodically pings the database and reconnects on failure."""

    def __init__(
        self,
        db: AbstractDatabase,
        *,
        interval: int | None = None,
        max_failures: int | None = None,
        backoff_max: int | None = None,
    ) -> None:
        self._db = db
        self._interval = interval or settings.db_health_check_interval
        self._max_failures = max_failures or settings.db_max_consecutive_failures
        self._backoff_max = backoff_max or settings.db_reconnect_backoff_max

        self._consecutive_failures = 0
        self._is_healthy = True
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def is_healthy(self) -> bool:
        return self._is_healthy

    def start(self) -> None:
        """Launch the monitor as a background task."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="db-health-monitor")
        logger.info("Database health monitor started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Signal the monitor to stop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()

    async def wait(self) -> None:
        """Wait for the monitor task to finish after stop()."""
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                await self._check()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Health monitor unexpected error: %s", exc)

    async def _check(self) -> None:
        try:
            result = await self._db.health_check()
            if result.get("healthy"):
                if not self._is_healthy:
                    logger.info("Database recovered after %d failures", self._consecutive_failures)
                self._consecutive_failures = 0
                self._is_healthy = True
                return
        except Exception as exc:
            logger.warning("Database health check exception: %s", exc)

        # Failed
        self._consecutive_failures += 1
        logger.warning(
            "Database health check failed (%d/%d)",
            self._consecutive_failures, self._max_failures,
        )

        if self._consecutive_failures >= self._max_failures:
            self._is_healthy = False
            await self._attempt_reconnect()

    async def _attempt_reconnect(self) -> None:
        backoff = min(
            self._interval * (2 ** (self._consecutive_failures - self._max_failures)),
            self._backoff_max,
        )
        logger.info("Attempting database reconnect (backoff=%.1fs)", backoff)
        await asyncio.sleep(backoff)

        try:
            await self._db.reconnect()
            self._consecutive_failures = 0
            self._is_healthy = True
            logger.info("Database reconnected by health monitor")
        except Exception as exc:
            logger.error("Reconnect failed: %s — will retry on next check", exc)

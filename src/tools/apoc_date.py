"""Category 15 — APOC Date & Time Utilities.

Useful when querying time-based graph data.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def apoc_date_currentTimestamp(db: AbstractDatabase, **_: Any) -> list[dict]:
    """Return current timestamp in milliseconds."""
    return await db.execute_read("RETURN timestamp() AS ts")


async def apoc_date_format(
    db: AbstractDatabase, *, timestamp: int, format_str: str = "yyyy-MM-dd HH:mm:ss", **_: Any,
) -> list[dict]:
    """Format a timestamp to string."""
    return await db.execute_read(
        "RETURN apoc.date.format($ts, 'ms', $fmt) AS formatted",
        {"ts": timestamp, "fmt": format_str},
    )


async def apoc_date_parse(
    db: AbstractDatabase, *, date_str: str, format_str: str = "yyyy-MM-dd", **_: Any,
) -> list[dict]:
    """Parse a date string to timestamp."""
    return await db.execute_read(
        "RETURN apoc.date.parse($ds, 'ms', $fmt) AS timestamp",
        {"ds": date_str, "fmt": format_str},
    )


async def apoc_date_convert(
    db: AbstractDatabase, *, value: int, from_unit: str = "ms", to_unit: str = "s", **_: Any,
) -> list[dict]:
    """Convert time value between units."""
    return await db.execute_read(
        "RETURN apoc.date.convert($val, $from, $to) AS converted",
        {"val": value, "from": from_unit, "to": to_unit},
    )


async def apoc_date_field(
    db: AbstractDatabase, *, timestamp: int, field: str = "year", **_: Any,
) -> list[dict]:
    """Extract a field (year, month, day…) from a timestamp."""
    return await db.execute_read(
        "RETURN apoc.date.field($ts, $field) AS value",
        {"ts": timestamp, "field": field},
    )


async def apoc_date_add(
    db: AbstractDatabase, *, timestamp: int, offset: int, unit: str = "d", **_: Any,
) -> list[dict]:
    """Add an offset to a timestamp."""
    return await db.execute_read(
        "RETURN apoc.date.add($ts, 'ms', $offset, $unit) AS result",
        {"ts": timestamp, "offset": offset, "unit": unit},
    )


async def apoc_date_diff(
    db: AbstractDatabase, *, ts1: int, ts2: int, unit: str = "d", **_: Any,
) -> list[dict]:
    """Compute difference between two timestamps."""
    return await db.execute_read(
        "RETURN apoc.date.diff($t1, $t2, $unit) AS diff",
        {"t1": ts1, "t2": ts2, "unit": unit},
    )


APOC_DATE_TOOLS: dict = {
    "apoc_date_currentTimestamp": apoc_date_currentTimestamp,
    "apoc_date_format": apoc_date_format,
    "apoc_date_parse": apoc_date_parse,
    "apoc_date_convert": apoc_date_convert,
    "apoc_date_field": apoc_date_field,
    "apoc_date_add": apoc_date_add,
    "apoc_date_diff": apoc_date_diff,
}

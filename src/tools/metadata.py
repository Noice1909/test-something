"""Category 7 — Database Metadata Tools.

System-level understanding: available procedures, functions, config, health.
"""

from __future__ import annotations

from typing import Any

from src.database.abstract import AbstractDatabase


async def list_procedures(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW PROCEDURES YIELD name, description, signature "
        "RETURN name, description, signature ORDER BY name"
    )


async def list_functions(db: AbstractDatabase, **_: Any) -> list[dict]:
    return await db.execute_read(
        "SHOW FUNCTIONS YIELD name, description, signature "
        "RETURN name, description, signature ORDER BY name"
    )


async def list_settings(db: AbstractDatabase, **_: Any) -> list[dict]:
    try:
        return await db.execute_read(
            "CALL dbms.listConfig() YIELD name, value, description "
            "RETURN name, value, description ORDER BY name LIMIT 100"
        )
    except Exception:
        # Fallback for Neo4j 5+ which uses SHOW SETTINGS
        return await db.execute_read(
            "SHOW SETTINGS YIELD * RETURN name, value, description ORDER BY name LIMIT 100"
        )


async def check_database_health(db: AbstractDatabase, **_: Any) -> dict[str, Any]:
    return await db.health_check()


async def ping_database(db: AbstractDatabase, **_: Any) -> dict[str, bool]:
    try:
        await db.execute_read("RETURN 1")
        return {"reachable": True}
    except Exception:
        return {"reachable": False}


# ── registry ──

METADATA_TOOLS: dict = {
    "list_procedures": list_procedures,
    "list_functions": list_functions,
    "list_settings": list_settings,
    "check_database_health": check_database_health,
    "ping_database": ping_database,
}

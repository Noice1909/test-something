"""API-key authentication dependency for FastAPI."""

from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import Header, HTTPException

from src.config import settings
from src.api.models.auth import AuthContext

logger = logging.getLogger(__name__)


async def require_auth(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
) -> AuthContext:
    """Validate the ``X-API-Key`` header.

    When ``settings.auth_api_key`` is empty, auth is **disabled** and all
    requests pass through as anonymous.  When configured, the header
    must match the key exactly (constant-time comparison).
    """
    configured_key = settings.auth_api_key

    # Auth disabled — dev / local mode
    if not configured_key:
        return AuthContext(user_id="anonymous", authenticated=False)

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(
        hashlib.sha256(x_api_key.encode()).hexdigest(),
        hashlib.sha256(configured_key.encode()).hexdigest(),
    ):
        logger.warning("Invalid API key attempt")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return AuthContext(user_id=f"apikey-{x_api_key[:8]}", authenticated=True)

"""Rate limiting via SlowAPI."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from src.config import settings


def _key_func(request: Request) -> str:
    """Use authenticated user ID when available, else fall back to IP."""
    # AuthContext is injected by the auth dependency and stored on request.state
    auth = getattr(request.state, "auth", None)
    if auth and auth.authenticated:
        return auth.user_id
    return get_remote_address(request)


limiter = Limiter(
    key_func=_key_func,
    default_limits=[settings.rate_limit_default],
    storage_uri=(
        settings.rate_limit_storage
        if settings.rate_limit_storage != "memory"
        else "memory://"
    ),
)

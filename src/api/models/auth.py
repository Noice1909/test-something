"""Authentication context returned by the auth dependency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthContext:
    """Propagated through the request after authentication."""

    user_id: str = "anonymous"
    authenticated: bool = False

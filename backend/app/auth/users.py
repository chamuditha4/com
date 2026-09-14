"""User directory (Option A: hardcoded demo users).

Only PBKDF2 hashes are stored. The demo passwords are documented in the README because this is a
POC. Replacing this module with an OIDC provider (Keycloak) changes nothing downstream:
the rest of the system consumes `Principal` only.
"""

from __future__ import annotations

import asyncio

from app.auth.models import Principal, Role, UserRecord
from app.core.security import verify_password

_USERS: tuple[UserRecord, ...] = (
    UserRecord(
        user_id="u-1001",
        username="viewer",
        display_name="Vera Viewer",
        role=Role.VIEWER,
        department="customer-service",
        password_hash="pbkdf2_sha256$240000$igyuZ+bJ8GzJU40X3I7Hcg==$ekTNWYpQ27vJheRNAcp2hDPw1gVO7qPk7l6L3yojMso=",
    ),
    UserRecord(
        user_id="u-2001",
        username="analyst",
        display_name="Arjun Analyst",
        role=Role.ANALYST,
        department="payments",
        password_hash="pbkdf2_sha256$240000$+v2H8N8hPjtXVp01X5V15w==$huRje3+CfDboK+TTu0iKo4Y4rh1H7vLN2cU2UByxIp0=",
    ),
    UserRecord(
        user_id="u-3001",
        username="admin",
        display_name="Amara Admin",
        role=Role.ADMINISTRATOR,
        department="technology",
        password_hash="pbkdf2_sha256$240000$r7UY3YzGLUc/qQUaNf5h9A==$GZiNd13MFouOz/N/0uHr5PWoNuejKABegSQnl/e2OSs=",
    ),
)

# A valid hash to compare against when the username does not exist, so the response time
# does not reveal which usernames are valid.
_DUMMY_HASH = _USERS[0].password_hash


class UserDirectory:
    def __init__(self, users: tuple[UserRecord, ...] = _USERS) -> None:
        self._by_id = {u.user_id: u for u in users}
        self._by_username = {u.username: u for u in users}

    async def authenticate(self, username: str, password: str) -> Principal | None:
        user = self._by_username.get(username)
        # PBKDF2 is deliberately CPU-expensive: run it off the event loop.
        ok = await asyncio.to_thread(
            verify_password, password, user.password_hash if user else _DUMMY_HASH
        )
        if user is None or not ok or not user.active:
            return None
        return user.to_principal()

    def get_active(self, user_id: str) -> Principal | None:
        user = self._by_id.get(user_id)
        return user.to_principal() if user and user.active else None

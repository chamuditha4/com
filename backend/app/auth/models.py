"""Identity and authorization primitives.

Two independent dimensions of authorization:

* **Permissions** gate *capabilities* (chat, search, analytics tools, MCP tools, admin tools).
* **Clearance** gates *data*: which document `access_level`s a principal may retrieve.

Both are derived from the role in one place (`ROLE_PERMISSIONS`, `ROLE_CLEARANCE`), so the RBAC
matrix in CLAUDE.md §8 maps one-to-one onto code and is covered by tests.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Role(StrEnum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMINISTRATOR = "administrator"


class Permission(StrEnum):
    CHAT = "chat"
    SEARCH = "search"
    ANALYTICS = "analytics"
    MCP = "mcp"
    ADMIN = "admin"


class AccessLevel(StrEnum):
    """Document classification, ordered from least to most sensitive."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({Permission.CHAT, Permission.SEARCH}),
    Role.ANALYST: frozenset({Permission.CHAT, Permission.SEARCH, Permission.ANALYTICS, Permission.MCP}),
    Role.ADMINISTRATOR: frozenset(Permission),
}

ROLE_CLEARANCE: dict[Role, AccessLevel] = {
    Role.VIEWER: AccessLevel.INTERNAL,
    Role.ANALYST: AccessLevel.CONFIDENTIAL,
    Role.ADMINISTRATOR: AccessLevel.RESTRICTED,
}

_LEVEL_ORDER = list(AccessLevel)


def access_levels_up_to(ceiling: AccessLevel) -> tuple[AccessLevel, ...]:
    return tuple(_LEVEL_ORDER[: _LEVEL_ORDER.index(ceiling) + 1])


class Principal(BaseModel):
    """The authenticated caller. Built server-side from a verified token, never from input."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    username: str
    display_name: str
    role: Role
    department: str | None = None

    @property
    def permissions(self) -> frozenset[Permission]:
        return ROLE_PERMISSIONS[self.role]

    @property
    def allowed_access_levels(self) -> tuple[AccessLevel, ...]:
        return access_levels_up_to(ROLE_CLEARANCE[self.role])

    def has(self, permission: Permission) -> bool:
        return permission in self.permissions

    def can_read(self, access_level: str) -> bool:
        return access_level in {level.value for level in self.allowed_access_levels}

    def public_view(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role.value,
            "permissions": sorted(p.value for p in self.permissions),
            "clearance": [level.value for level in self.allowed_access_levels],
        }


class UserRecord(BaseModel):
    user_id: str
    username: str
    display_name: str
    role: Role
    department: str | None = None
    password_hash: str = Field(repr=False)
    active: bool = True

    def to_principal(self) -> Principal:
        return Principal(
            user_id=self.user_id,
            username=self.username,
            display_name=self.display_name,
            role=self.role,
            department=self.department,
        )

"""FastAPI dependencies: container access, authentication, RBAC and rate limiting.

Every protected route resolves the principal from a verified token *on each request*, then
checks permissions server-side. The UI hiding a button is never the control.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.models import Permission, Principal
from app.container import Container
from app.core.exceptions import AuthenticationError, PermissionDeniedError, RateLimitedError
from app.core.logging import user_id_var
from app.core.security import decode_access_token

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    return request.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]


async def get_principal(
    container: ContainerDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    claims = decode_access_token(credentials.credentials, container.settings)
    principal = container.users.get_active(str(claims["sub"]))
    if principal is None:
        raise AuthenticationError("This account is not active.")
    if claims.get("role") != principal.role.value:
        # Role changed since the token was issued: force re-authentication.
        raise AuthenticationError("Your permissions have changed. Please sign in again.")
    user_id_var.set(principal.user_id)
    return principal


def require(permission: Permission) -> Callable[..., Awaitable[Principal]]:
    async def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
        if not principal.has(permission):
            raise PermissionDeniedError(details={"required_permission": permission.value})
        return principal

    dependency.__name__ = f"require_{permission.value}"
    return dependency


async def rate_limited_chat(
    container: ContainerDep,
    principal: Annotated[Principal, Depends(require(Permission.CHAT))],
) -> Principal:
    if container.settings.rate_limit_enabled:
        decision = await container.rate_limiter.acquire(f"user:{principal.user_id}")
        if not decision.allowed:
            retry_after = max(1, math.ceil(decision.retry_after_seconds))
            raise RateLimitedError(
                f"Rate limit reached. Please wait {retry_after} seconds before sending another message.",
                details={"retry_after_seconds": retry_after, "capacity": decision.capacity},
                headers={"Retry-After": str(retry_after)},
            )
    return principal


ChatPrincipal = Annotated[Principal, Depends(rate_limited_chat)]

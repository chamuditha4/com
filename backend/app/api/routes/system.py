"""Auth, health and admin endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.api.deps import ContainerDep, get_principal, require
from app.api.schemas import TokenRequest, TokenResponse
from app.auth.models import AccessLevel, Permission, Principal
from app.core.exceptions import AuthenticationError, RateLimitedError
from app.core.logging import get_logger
from app.core.security import create_access_token
from app.tools.audit import AuditEvent

logger = get_logger(__name__)

auth_router = APIRouter(prefix="/auth", tags=["auth"])
health_router = APIRouter(prefix="/health", tags=["health"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


# --- auth ------------------------------------------------------------------------------------


@auth_router.post("/token", summary="Exchange demo credentials for a bearer token")
async def issue_token(body: TokenRequest, container: ContainerDep) -> TokenResponse:
    # Brute-force protection: login attempts are rate limited per username.
    attempt = await container.rate_limiter.acquire(f"login:{body.username.lower()}")
    if not attempt.allowed:
        raise RateLimitedError("Too many sign-in attempts. Please wait and try again.")

    principal = await container.users.authenticate(body.username, body.password)
    if principal is None:
        await container.audit.record(
            AuditEvent(
                user_id=f"unknown:{body.username[:64]}", role="none", action="login", target="auth", outcome="failed"
            )
        )
        raise AuthenticationError("Invalid username or password.")

    token, expires_in = create_access_token(
        subject=principal.user_id, role=principal.role.value, settings=container.settings
    )
    await container.audit.record(
        AuditEvent(
            user_id=principal.user_id, role=principal.role.value, action="login", target="auth", outcome="succeeded"
        )
    )
    return TokenResponse(access_token=token, expires_in=expires_in, user=principal.public_view())


@auth_router.get("/me", summary="The authenticated principal, permissions and clearance")
async def me(principal: Annotated[Principal, Depends(get_principal)]) -> dict[str, object]:
    return principal.public_view()


# --- health ----------------------------------------------------------------------------------


@health_router.get("/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@health_router.get("/ready", summary="Readiness probe (vector store, Redis, LLM configuration)")
async def ready(container: ContainerDep) -> JSONResponse:
    services = container.services
    checks: dict[str, Any] = {"vector_store": await services.retriever.store.ping()}
    if container.redis is not None:
        try:
            checks["redis"] = bool(await container.redis.ping())
        except Exception:
            checks["redis"] = False
    else:
        checks["redis"] = "not_configured"
    checks["llm"] = {"available": services.llm.available, **services.llm.describe()}  # degraded, not down

    healthy = checks["vector_store"] is True and checks["redis"] in (True, "not_configured")
    return JSONResponse(
        status_code=200 if healthy else 503, content={"status": "ok" if healthy else "unavailable", "checks": checks}
    )


# --- admin -----------------------------------------------------------------------------------

AdminPrincipal = Annotated[Principal, Depends(require(Permission.ADMIN))]
_SENSITIVE = ("key", "secret", "password", "token")


@admin_router.get("/system", summary="Effective (non-secret) configuration and index status")
async def system(_: AdminPrincipal, container: ContainerDep) -> dict[str, Any]:
    settings = container.settings.model_dump(mode="json")
    services = container.services
    return {
        "config": {k: v for k, v in settings.items() if not any(s in k for s in _SENSITIVE)},
        "llm": services.llm.describe(),
        "catalog": services.retriever.catalog.overview(tuple(AccessLevel)).model_dump(mode="json"),
        "namespaces": await services.retriever.store.namespace_counts(),
    }


@admin_router.get("/audit", summary="Recent audit events")
async def audit(
    _: AdminPrincipal, container: ContainerDep, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[AuditEvent]:
    return await container.audit.recent(limit)

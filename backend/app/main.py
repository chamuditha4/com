"""FastAPI application factory.

Run:  uvicorn app.main:create_app --factory --app-dir backend --port 8000

Workers are stateless: conversation checkpoints, long-term memory, rate-limit buckets and the
audit log live in Redis when `REDIS_URL` is set, so any number of replicas can sit behind a
load balancer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import TraceIdMiddleware, register_exception_handlers
from app.api.routes.chat import router as chat_router
from app.api.routes.system import admin_router, auth_router, health_router
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.observability.tracing import configure_tracing

ContainerFactory = Callable[[Settings, AsyncExitStack], Awaitable[Container]]


def create_app(settings: Settings | None = None, *, container_factory: ContainerFactory | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.langsmith_tracing = configure_tracing(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            app.state.container = await (container_factory or build_container)(settings, stack)
            yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Agentic RAG assistant: LangGraph multi-agent orchestration, RLM research, hybrid retrieval, RBAC.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Trace-Id"],
    )
    app.add_middleware(TraceIdMiddleware)
    register_exception_handlers(app)

    for router in (auth_router, chat_router, health_router, admin_router):
        app.include_router(router)
    return app

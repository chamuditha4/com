"""Trace-id middleware and structured error handlers.

The middleware is pure ASGI (not `BaseHTTPMiddleware`) so the trace-id ContextVar stays set for
the whole response, including the body of a Server-Sent Events stream.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.exceptions import AppError
from app.core.logging import get_logger, trace_id_var

logger = get_logger("app.http")


class TraceIdMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Always server-generated: a client-supplied id could be used for log injection or to
        # correlate with someone else's trace. It doubles as the LangSmith root run id.
        trace_id = str(uuid.uuid4())
        scope.setdefault("state", {})["trace_id"] = trace_id
        token = trace_id_var.set(trace_id)
        started = time.perf_counter()
        status = {"code": 500}

        async def send_with_trace(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append((b"x-trace-id", trace_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_trace)
        finally:
            logger.info(
                "request",
                extra={
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status": status["code"],
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            trace_id_var.reset(token)


def _error(
    status: int, code: str, message: str, details: object = None, headers: dict[str, str] | None = None
) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "trace_id": trace_id_var.get()}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError) -> JSONResponse:
        log = logger.warning if exc.status_code < 500 else logger.error
        log("app error", extra={"code": exc.code, "status": exc.status_code})
        return _error(exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Echo field locations and messages, never the submitted values.
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", "")} for e in exc.errors()]
        return _error(422, "invalid_request", "The request was invalid.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error", exc_info=exc)
        return _error(500, "internal_error", "Something went wrong. Please try again later.")

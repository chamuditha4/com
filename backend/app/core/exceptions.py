"""Application error taxonomy.

Every error that can reach a client is an `AppError` with a stable machine-readable
`code` and a user-safe `message`. Internal details go to the logs (with the trace id),
never to the response body.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "Something went wrong. Please try again later."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details or {}
        self.headers = headers or {}
        super().__init__(self.message)


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"
    message = "Authentication required."


class PermissionDeniedError(AppError):
    status_code = 403
    code = "permission_denied"
    message = "Your role does not permit this action."


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "You are sending requests too quickly. Please wait and try again."


class InvalidRequestError(AppError):
    status_code = 400
    code = "invalid_request"
    message = "The request was invalid."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "The requested resource was not found."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "The request conflicts with the current state of the resource."


class RetrievalUnavailableError(AppError):
    status_code = 503
    code = "retrieval_unavailable"
    message = "The knowledge base is temporarily unavailable."


class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_unavailable"
    message = "The language model is temporarily unavailable."


class ToolExecutionError(AppError):
    status_code = 502
    code = "tool_failed"
    message = "A tool failed to execute."

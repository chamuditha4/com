"""Request/response models for the HTTP API. Every boundary is validated here."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app.agents.state import FinalAnswer, ResearchPlan, ResearchReport, RouteDecision, ToolCallRecord, ValidationReport
from app.guardrails.injection import normalize_text

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
MAX_MESSAGE_CHARS = 4000


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in: int
    user: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str = Field(pattern=SESSION_ID_PATTERN)

    @field_validator("message")
    @classmethod
    def _clean(cls, value: str) -> str:
        cleaned = normalize_text(value).strip()
        if not cleaned:
            raise ValueError("message must contain visible text")
        return cleaned


class ResumeRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    approved: bool


class TraceInfo(BaseModel):
    trace_id: str
    langsmith_enabled: bool
    langsmith_project: str


class TurnResult(BaseModel):
    session_id: str
    status: Literal["completed", "awaiting_approval"]
    trace: TraceInfo
    answer: FinalAnswer | None = None
    approval: dict[str, Any] | None = None
    route: RouteDecision | None = None
    research_plan: ResearchPlan | None = None
    research_report: ResearchReport | None = None
    validation: ValidationReport | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionHistory(BaseModel):
    session_id: str
    awaiting_approval: bool
    messages: list[HistoryMessage]

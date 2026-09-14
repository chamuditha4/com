"""Agent Activity events: the real-time mirror of the LangSmith trace.

Every node reports what it is doing through `emit()`. Each event is:
* streamed to the client through LangGraph's custom stream (`stream_mode="custom"`), which feeds
  the Streamlit Agent Activity Panel;
* logged as structured JSON carrying the request's trace id.

LangSmith gets the full-fidelity trace automatically (nodes, LLM calls, `@traceable` retrieval
and RLM slices), and the trace id joins all three views.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger("agent.activity")

EventKind = Literal[
    "node", "decision", "memory", "guardrail", "retrieval", "plan", "batch", "recursion",
    "aggregate", "tool_call", "tool_result", "approval", "generation", "validation", "final",
    "warning", "error",
]


class ActivityEvent(BaseModel):
    type: Literal["activity"] = "activity"
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    trace_id: str
    node: str
    kind: EventKind
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


def emit(runtime: Any, node: str, kind: EventKind, message: str, **data: Any) -> None:
    """Publish an activity event. `runtime` is the LangGraph `Runtime[AgentContext]`."""
    event = ActivityEvent(trace_id=runtime.context.trace_id, node=node, kind=kind, message=message, data=data)
    payload = event.model_dump(mode="json")
    logger.info(message, extra={"event": {k: v for k, v in payload.items() if k != "message"}})
    try:
        runtime.stream_writer(payload)
    except Exception:  # streaming is best-effort; never fail a node because a client went away
        logger.debug("activity stream write failed", exc_info=True)

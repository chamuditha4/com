"""Chat endpoints: streaming (SSE) and non-streaming turns, HITL resume, session history.

Thread ids are derived server-side as `{user_id}:{session_id}`, so a user can only ever read or
continue their own conversations, whatever session id they send.

SSE events: `start` → `activity`* → (`final` | `approval_required` | `error`) → `done`.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agents.context import AgentContext
from app.api.deps import ChatPrincipal, ContainerDep, require
from app.api.schemas import (
    SESSION_ID_PATTERN,
    ChatRequest,
    HistoryMessage,
    ResumeRequest,
    SessionHistory,
    TraceInfo,
    TurnResult,
)
from app.auth.models import Permission, Principal
from app.container import Container
from app.core.exceptions import AppError, ConflictError
from app.core.logging import get_logger, trace_id_var

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def thread_config(
    principal: Principal, session_id: str, *, trace_id: str | None = None, resumed: bool = False
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "configurable": {"thread_id": f"{principal.user_id}:{session_id}"},
        "recursion_limit": 60,
    }
    if trace_id:
        config |= {
            "run_id": uuid.UUID(trace_id),  # LangSmith root run id == API trace id == log trace id
            "run_name": "chat_turn_resume" if resumed else "chat_turn",
            "tags": [f"role:{principal.role.value}"],
            "metadata": {
                "trace_id": trace_id,
                "user_id": principal.user_id,
                "role": principal.role.value,
                "session_id": session_id,
                "resumed": resumed,
            },
        }
    return config


def _context(container: Container, principal: Principal, session_id: str, trace_id: str) -> AgentContext:
    return AgentContext(principal=principal, trace_id=trace_id, session_id=session_id, services=container.services)


async def _pending_approval(container: Container, principal: Principal, session_id: str) -> bool:
    snapshot = await container.graph.aget_state(thread_config(principal, session_id))
    return bool(snapshot.interrupts)


async def _turn_result(container: Container, principal: Principal, session_id: str, trace_id: str) -> TurnResult:
    snapshot = await container.graph.aget_state(thread_config(principal, session_id))
    values = snapshot.values or {}
    trace = TraceInfo(
        trace_id=trace_id,
        langsmith_enabled=container.settings.langsmith_tracing,
        langsmith_project=container.settings.langsmith_project,
    )
    common = {
        "session_id": session_id,
        "trace": trace,
        "route": values.get("route"),
        "research_plan": values.get("research_plan"),
        "research_report": values.get("research_report"),
        "tool_calls": values.get("tool_calls", []),
    }
    if snapshot.interrupts:
        return TurnResult(status="awaiting_approval", approval=snapshot.interrupts[0].value, **common)
    return TurnResult(status="completed", answer=values.get("final"), validation=values.get("validation"), **common)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _stream_graph(
    container: Container, principal: Principal, session_id: str, graph_input: Any, *, resumed: bool
) -> AsyncIterator[str]:
    trace_id = trace_id_var.get() or str(uuid.uuid4())
    yield _sse("start", {"trace_id": trace_id, "session_id": session_id})
    try:
        async with asyncio.timeout(container.settings.chat_timeout_seconds):
            async for _, mode, chunk in container.graph.astream(
                graph_input,
                thread_config(principal, session_id, trace_id=trace_id, resumed=resumed),
                context=_context(container, principal, session_id, trace_id),
                stream_mode=["custom"],
                subgraphs=True,
            ):
                if mode == "custom":
                    yield _sse("activity", chunk)
        result = await _turn_result(container, principal, session_id, trace_id)
        event = "final" if result.status == "completed" else "approval_required"
        yield _sse(event, result.model_dump(mode="json"))
    except TimeoutError:
        logger.error("chat turn timed out", extra={"session_id": session_id})
        yield _sse(
            "error",
            {
                "code": "timeout",
                "message": "The request took too long. Please try a narrower question.",
                "trace_id": trace_id,
            },
        )
    except AppError as exc:
        yield _sse("error", {"code": exc.code, "message": exc.message, "trace_id": trace_id})
    except Exception:
        logger.exception("chat stream failed")
        yield _sse(
            "error",
            {"code": "internal_error", "message": "Something went wrong. Please try again.", "trace_id": trace_id},
        )
    yield _sse("done", {"trace_id": trace_id})


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.post("/stream", summary="Send a message and stream agent activity (SSE)")
async def chat_stream(body: ChatRequest, principal: ChatPrincipal, container: ContainerDep) -> StreamingResponse:
    if await _pending_approval(container, principal, body.session_id):
        raise ConflictError("This session is waiting for an approval decision. Approve or reject it first.")
    graph_input = {"messages": [HumanMessage(content=body.message)]}
    return StreamingResponse(
        _stream_graph(container, principal, body.session_id, graph_input, resumed=False),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("", summary="Send a message and wait for the final answer")
async def chat(body: ChatRequest, principal: ChatPrincipal, container: ContainerDep) -> TurnResult:
    if await _pending_approval(container, principal, body.session_id):
        raise ConflictError("This session is waiting for an approval decision. Approve or reject it first.")
    trace_id = trace_id_var.get() or str(uuid.uuid4())
    async with asyncio.timeout(container.settings.chat_timeout_seconds):
        await container.graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            thread_config(principal, body.session_id, trace_id=trace_id),
            context=_context(container, principal, body.session_id, trace_id),
        )
    return await _turn_result(container, principal, body.session_id, trace_id)


@router.post("/resume/stream", summary="Approve or reject a pending sensitive tool call (SSE)")
async def resume_stream(body: ResumeRequest, principal: ChatPrincipal, container: ContainerDep) -> StreamingResponse:
    if not await _pending_approval(container, principal, body.session_id):
        raise ConflictError("There is no pending approval for this session.")
    return StreamingResponse(
        _stream_graph(container, principal, body.session_id, Command(resume={"approved": body.approved}), resumed=True),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/resume", summary="Approve or reject a pending sensitive tool call")
async def resume(body: ResumeRequest, principal: ChatPrincipal, container: ContainerDep) -> TurnResult:
    if not await _pending_approval(container, principal, body.session_id):
        raise ConflictError("There is no pending approval for this session.")
    trace_id = trace_id_var.get() or str(uuid.uuid4())
    await container.graph.ainvoke(
        Command(resume={"approved": body.approved}),
        thread_config(principal, body.session_id, trace_id=trace_id, resumed=True),
        context=_context(container, principal, body.session_id, trace_id),
    )
    return await _turn_result(container, principal, body.session_id, trace_id)


@router.get("/sessions/{session_id}/history", summary="Conversation history for one of your sessions")
async def history(
    session_id: Annotated[str, Path(pattern=SESSION_ID_PATTERN)],
    principal: Annotated[Principal, Depends(require(Permission.CHAT))],
    container: ContainerDep,
) -> SessionHistory:
    snapshot = await container.graph.aget_state(thread_config(principal, session_id))
    messages = (snapshot.values or {}).get("messages", [])
    return SessionHistory(
        session_id=session_id,
        awaiting_approval=bool(snapshot.interrupts),
        messages=[
            HistoryMessage(role="user" if isinstance(m, HumanMessage) else "assistant", content=m.text)
            for m in messages
            if m.type in ("human", "ai")
        ],
    )


@router.delete("/memory", summary="Forget everything the assistant remembers about you")
async def forget_me(
    principal: Annotated[Principal, Depends(require(Permission.CHAT))], container: ContainerDep
) -> dict[str, int]:
    memory = container.services.memory
    return {"deleted": await memory.forget_all(principal.user_id) if memory else 0}

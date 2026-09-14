"""Test helpers: a scripted LLM and a one-call way to run a conversation turn through the graph."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command

from app.agents.context import AgentContext
from app.auth.models import Principal
from app.container import Container
from app.core.exceptions import LLMUnavailableError

TODAY = date(2026, 9, 14)


class ScriptedLLM:
    """Deterministic stand-in for `LLMClient`. Unscripted calls raise `LLMUnavailableError`,
    which also exercises each node's fallback path."""

    available = True

    def __init__(
        self,
        *,
        generate: Callable[[str, list[BaseMessage]], str] | None = None,
        structured: Callable[[type, list[BaseMessage]], Any] | None = None,
    ) -> None:
        self._generate = generate
        self._structured = structured
        self.calls: list[str] = []

    def describe(self) -> dict[str, str]:
        return {"provider": "scripted"}

    async def generate(
        self, messages: list[BaseMessage], *, tier: str = "reasoning", run_name: str = "generate"
    ) -> str:
        self.calls.append(run_name)
        if self._generate is None:
            raise LLMUnavailableError()
        return self._generate(run_name, list(messages))

    async def structured(
        self, messages: list[BaseMessage], schema: type, *, tier: str = "fast", run_name: str = "structured"
    ) -> Any:
        self.calls.append(run_name)
        result = self._structured(schema, list(messages)) if self._structured else None
        if result is None:
            raise LLMUnavailableError()
        return result

    async def invoke_with_tools(self, messages: list[BaseMessage], tools: list[Any], **_: Any) -> AIMessage:
        self.calls.append("tool_planner")
        raise LLMUnavailableError()


def context_for(container: Container, principal: Principal, session_id: str = "s1") -> AgentContext:
    return AgentContext(
        principal=principal,
        trace_id=str(uuid.uuid4()),
        session_id=session_id,
        services=container.services,
        today=TODAY,
    )


def thread_config(principal: Principal, session_id: str = "s1") -> dict[str, Any]:
    return {"configurable": {"thread_id": f"{principal.user_id}:{session_id}"}}


async def run_turn(container: Container, principal: Principal, question: str, session_id: str = "s1") -> dict[str, Any]:
    return await container.graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        thread_config(principal, session_id),
        context=context_for(container, principal, session_id),
    )


async def resume_turn(
    container: Container, principal: Principal, approved: bool, session_id: str = "s1"
) -> dict[str, Any]:
    return await container.graph.ainvoke(
        Command(resume={"approved": approved}),
        thread_config(principal, session_id),
        context=context_for(container, principal, session_id),
    )


async def stream_events(
    container: Container, principal: Principal, question: str, session_id: str = "s1"
) -> list[dict[str, Any]]:
    events = []
    async for _, mode, chunk in container.graph.astream(
        {"messages": [HumanMessage(content=question)]},
        thread_config(principal, session_id),
        context=context_for(container, principal, session_id),
        stream_mode=["custom", "updates"],
        subgraphs=True,
    ):
        if mode == "custom":
            events.append(chunk)
    return events

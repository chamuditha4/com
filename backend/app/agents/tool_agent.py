"""Tool Agent: plan → (human approval) → execute, as three explicit graph nodes.

Why three nodes? LangGraph re-runs an interrupted node from its beginning when it resumes. Keeping
the LLM call (`tool_planner`) separate from the interrupt (`tool_approval`) means approving a
tool never triggers a second, possibly different, LLM decision.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agents.context import AgentContext
from app.agents.prompts import history_messages, tool_planner_messages
from app.agents.state import AgentState, Evidence, ToolCallRecord
from app.core.exceptions import LLMUnavailableError
from app.observability.events import emit
from app.tools.registry import ToolSpec


def heuristic_tool_calls(question: str, specs: list[ToolSpec]) -> list[ToolCallRecord]:
    """Keyword-based single tool call, used when the LLM is unavailable."""
    available = {s.name: s for s in specs}
    q = question.lower()
    team = next(
        (t for key, t in (("security", "platform-security"), ("sre", "sre"), ("devops", "devops")) if key in q),
        "payments-engineering",
    )
    candidates: list[tuple[bool, str, dict[str, Any]]] = [
        (bool(re.search(r"re-?index", q)), "admin_reindex_knowledge_base", {"reason": question[:300]}),
        ("audit" in q, "admin_audit_log", {"limit": 20}),
        ("ticket" in q, "mcp_search_incident_tickets", {"status": "open" if "open" in q else "any"}),
        (bool(re.search(r"on-?call", q)), "mcp_get_oncall_roster", {"team": team}),
        ("status" in q, "mcp_get_service_status", {}),
        (bool(re.search(r"metric|volume", q)), "mcp_get_payment_metrics", {}),
    ]
    for matched, name, args in candidates:
        if matched and name in available:
            return [
                ToolCallRecord(
                    call_id="heuristic-0",
                    tool=name,
                    args=args,
                    requires_approval=available[name].requires_approval,
                    origin="heuristic",
                )
            ]
    return [
        ToolCallRecord(
            call_id="heuristic-0", tool="knowledge_search", args={"query": question[:500]}, origin="heuristic"
        )
    ]


async def tool_planner(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    services = ctx.services
    specs = await services.tools.tools_for(ctx.principal)
    spec_by_name = {s.name: s for s in specs}
    scratchpad: list[AnyMessage] = list(state.get("tool_messages", []))
    iteration = state.get("tool_iterations", 0)

    new_calls: list[ToolCallRecord] = []
    degraded: list[str] = []
    if services.llm.available:
        try:
            ai = await services.llm.invoke_with_tools(
                tool_planner_messages(
                    question=state.get("question", ""),
                    history=history_messages(state.get("messages", []), services.settings.agent_history_window),
                    today=ctx.today,
                    scratchpad=scratchpad,
                    canary=ctx.canary,
                ),
                [s.to_llm_tool() for s in specs],  # only the tools this role may use
                tier="fast",
                run_name="tool_planner",
            )
            for tc in ai.tool_calls:
                spec = spec_by_name.get(tc["name"])
                new_calls.append(
                    ToolCallRecord(
                        call_id=tc["id"],
                        tool=tc["name"],
                        args=tc.get("args") or {},
                        requires_approval=bool(spec and spec.requires_approval),
                    )
                )
            if ai.tool_calls:
                scratchpad.append(ai)
        except LLMUnavailableError:
            if iteration == 0:
                new_calls, degraded = heuristic_tool_calls(state.get("question", ""), specs), ["tool_planner"]
    elif iteration == 0:
        new_calls, degraded = heuristic_tool_calls(state.get("question", ""), specs), ["tool_planner"]

    for call in new_calls:
        emit(
            runtime,
            "tool_planner",
            "tool_call",
            f"Planned tool call: {call.tool}",
            tool=call.tool,
            args=call.args,
            requires_approval=call.requires_approval,
            origin=call.origin,
            available_tools=sorted(spec_by_name),
        )
    if not new_calls:
        emit(runtime, "tool_planner", "decision", "No further tool calls needed")
    return {
        "tool_calls": [*state.get("tool_calls", []), *new_calls],
        "tool_messages": scratchpad,
        "degraded": degraded,
    }


def route_after_planner(state: AgentState) -> str:
    pending = [c for c in state.get("tool_calls", []) if c.status == "pending"]
    return "tool_approval" if pending else "response_agent"


async def tool_approval(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    calls = state.get("tool_calls", [])
    needing = [c for c in calls if c.status == "pending" and c.requires_approval]
    if not needing:
        return {}

    emit(
        runtime,
        "tool_approval",
        "approval",
        f"Human approval required for {', '.join(c.tool for c in needing)}",
        calls=[c.model_dump() for c in needing],
    )
    decision = interrupt(
        {
            "type": "approval_required",
            "message": "The assistant wants to run a sensitive tool. Approve?",
            "calls": [{"call_id": c.call_id, "tool": c.tool, "args": c.args} for c in needing],
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    needing_ids = {c.call_id for c in needing}
    updated = [
        c.model_copy(
            update={"status": "approved"}
            if approved
            else {"status": "rejected", "error": "Rejected by the human approver."}
        )
        if c.call_id in needing_ids
        else c
        for c in calls
    ]
    emit(
        runtime, "tool_approval", "approval", "Approved by user" if approved else "Rejected by user", approved=approved
    )
    return {"tool_calls": updated}


async def tool_executor(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    services = ctx.services
    calls = state.get("tool_calls", [])
    runnable = [c for c in calls if c.status in ("pending", "approved")]

    results = await asyncio.gather(
        *(
            services.tools.execute(
                ctx.principal, c.tool, c.args, trace_id=ctx.trace_id, approved=c.status == "approved"
            )
            for c in runnable
        )
    )
    outcomes = {c.call_id: r for c, r in zip(runnable, results, strict=True)}

    evidence = list(state.get("evidence", []))
    seen_chunks = {e.chunk_id for e in evidence if e.chunk_id}
    llm_call_ids = {tc["id"] for m in state.get("tool_messages", []) for tc in getattr(m, "tool_calls", [])}
    tool_messages: list[AnyMessage] = []
    updated: list[ToolCallRecord] = []

    for call in calls:
        result = outcomes.get(call.call_id)
        if result is not None:
            call = call.model_copy(
                update={
                    "status": "succeeded" if result.ok else "failed",
                    "output": result.output[:2000] if result.ok else None,
                    "error": result.error,
                    "latency_ms": result.latency_ms,
                }
            )
            emit(
                runtime,
                "tool_executor",
                "tool_result" if result.ok else "error",
                f"{call.tool} {'succeeded' if result.ok else 'failed'} in {result.latency_ms} ms",
                tool=call.tool,
                ok=result.ok,
                error=result.error,
                preview=(result.output or "")[:500],
            )
            if result.ok and result.documents:
                for doc in result.documents:
                    if doc.chunk.chunk_id not in seen_chunks:
                        seen_chunks.add(doc.chunk.chunk_id)
                        evidence.append(Evidence.from_chunk(len(evidence) + 1, doc))
            elif result.ok:
                evidence.append(Evidence(id=len(evidence) + 1, source_type="tool", title=call.tool, text=result.output))
        # Every tool_use the LLM emitted must be answered with a tool_result, including rejections.
        if call.call_id in llm_call_ids and (result is not None or call.status == "rejected"):
            content = (call.output or "") if call.status == "succeeded" else f"ERROR: {call.error}"
            tool_messages.append(ToolMessage(content=content, tool_call_id=call.call_id))
        updated.append(call)

    iterations = state.get("tool_iterations", 0) + 1
    loop = services.llm.available and iterations < services.settings.agent_max_tool_iterations
    return {
        "tool_calls": updated,
        "evidence": evidence,
        "tool_messages": [*state.get("tool_messages", []), *tool_messages],
        "tool_iterations": iterations,
        "tool_loop_continue": loop,
        "retrieval_status": "ok" if evidence else "empty",
    }


def route_after_executor(state: AgentState) -> str:
    return "tool_planner" if state.get("tool_loop_continue") else "response_agent"

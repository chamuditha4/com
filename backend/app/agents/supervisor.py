"""Input guard and Supervisor: screen the request, then decide how to answer it."""

from __future__ import annotations

from datetime import date
from typing import Any

from langgraph.runtime import Runtime

from app.agents.context import AgentContext
from app.agents.heuristics import can_use_tools, route_heuristically
from app.agents.prompts import history_messages, supervisor_messages
from app.agents.state import AgentState, GuardVerdict, RouteDecision
from app.auth.models import Principal
from app.core.exceptions import LLMUnavailableError
from app.guardrails.injection import assess_injection
from app.observability.events import emit

STRATEGY_TO_NODE = {
    "retrieval": "retrieval_agent",
    "research": "research_agent",
    "tools": "tool_planner",
    "direct": "response_agent",
}


async def input_guard(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    assessment = assess_injection(state.get("question", ""))
    verdict = GuardVerdict(
        allowed=not assessment.should_block,
        score=assessment.score,
        signals=assessment.signals,
        reason="prompt-injection / exfiltration attempt" if assessment.should_block else None,
    )
    if assessment.should_block:
        emit(
            runtime,
            "input_guard",
            "guardrail",
            "Blocked request: prompt-injection attempt detected",
            score=assessment.score,
            signals=assessment.signals,
        )
    elif assessment.flagged:
        emit(
            runtime,
            "input_guard",
            "guardrail",
            "Suspicious phrasing flagged; continuing with hardened prompts",
            score=assessment.score,
            signals=assessment.signals,
        )
    else:
        emit(runtime, "input_guard", "guardrail", "Input passed guardrails", score=assessment.score)
    return {"input_guard": verdict}


def route_after_guard(state: AgentState) -> str:
    guard = state.get("input_guard")
    return "supervisor" if guard is None or guard.allowed else "finalize"


def enforce_route_policy(
    decision: RouteDecision, *, question: str, principal: Principal, known_departments: list[str], today: date
) -> tuple[RouteDecision, list[str]]:
    """Server-side policy over the (untrusted) routing decision."""
    notes: list[str] = []
    updates: dict[str, Any] = {}
    if decision.strategy == "tools" and not can_use_tools(principal):
        updates["strategy"] = "retrieval"
        notes.append(f"role '{principal.role.value}' may not use tools; routed to retrieval")
    departments = [d.strip().lower() for d in decision.departments]
    if unknown := [d for d in departments if d not in known_departments]:
        notes.append(f"ignored unknown departments {unknown}")
    updates["departments"] = [d for d in departments if d in known_departments]
    if decision.date_from and decision.date_from > today:
        updates["date_from"] = None
        notes.append("ignored a start date in the future")
    if decision.date_from and decision.date_to and decision.date_from > decision.date_to:
        updates["date_from"], updates["date_to"] = None, None
        notes.append("ignored an inverted date range")
    if not decision.search_query.strip():
        updates["search_query"] = question[:500]
    return decision.model_copy(update=updates), notes


async def supervisor(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    services = ctx.services
    question = state.get("question", "")
    departments = services.retriever.catalog.departments

    decision: RouteDecision | None = None
    decided_by = "llm"
    if services.llm.available:
        try:
            decision = await services.llm.structured(
                supervisor_messages(
                    question=question,
                    history=history_messages(state.get("messages", []), services.settings.agent_history_window),
                    principal=ctx.principal,
                    departments=departments,
                    today=ctx.today,
                    tools_allowed=can_use_tools(ctx.principal),
                    memories=state.get("recalled_memories", []),
                ),
                RouteDecision,
                tier="fast",
                run_name="supervisor_route",
            )
        except LLMUnavailableError:
            emit(runtime, "supervisor", "warning", "LLM routing failed; using heuristic router")
    if decision is None:
        decision = route_heuristically(question, ctx.principal, departments, ctx.today)
        decided_by = "heuristic"

    decision, notes = enforce_route_policy(
        decision, question=question, principal=ctx.principal, known_departments=departments, today=ctx.today
    )
    filters = decision.filters()
    emit(
        runtime,
        "supervisor",
        "decision",
        f"Routing to {STRATEGY_TO_NODE[decision.strategy]} ({decision.strategy}) — {decision.rationale}",
        strategy=decision.strategy,
        intent=decision.intent,
        search_query=decision.search_query,
        filters=filters.describe() if filters else "none",
        decided_by=decided_by,
        policy_notes=notes,
    )
    return {"route": decision, "degraded": ["supervisor"] if decided_by == "heuristic" else []}


def route_by_strategy(state: AgentState) -> str:
    route = state.get("route")
    return STRATEGY_TO_NODE[route.strategy] if route else "response_agent"

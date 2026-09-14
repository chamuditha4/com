"""Response Agent: grounded answer synthesis with inline citations."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agents.answers import RETRIEVAL_UNAVAILABLE, direct_answer, extractive_answer, insufficient_evidence
from app.agents.context import AgentContext
from app.agents.prompts import history_messages, response_messages
from app.agents.state import AgentState
from app.core.exceptions import LLMUnavailableError
from app.observability.events import emit

GROUNDED_STRATEGIES = ("retrieval", "research", "tools")


async def response_agent(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    services = ctx.services
    route = state.get("route")
    strategy = route.strategy if route else "direct"
    evidence = state.get("evidence", [])
    status = state.get("retrieval_status", "not_run")
    attempts = state.get("attempts", 0)
    validation = state.get("validation")
    feedback = [i.message for i in validation.issues] if attempts and validation else []

    # Deterministic short-circuits: no LLM call, so nothing can be hallucinated.
    if strategy in GROUNDED_STRATEGIES and status == "unavailable":
        emit(
            runtime,
            "response_agent",
            "generation",
            "Retrieval unavailable; returning a safe notice instead of guessing",
        )
        return {"draft": RETRIEVAL_UNAVAILABLE}
    if strategy in GROUNDED_STRATEGIES and not evidence:
        emit(
            runtime,
            "response_agent",
            "generation",
            "No evidence available to this role; declining to answer from general knowledge",
        )
        return {"draft": insufficient_evidence()}

    if services.llm.available:
        try:
            draft = await services.llm.generate(
                response_messages(
                    question=state.get("question", ""),
                    history=history_messages(state.get("messages", []), services.settings.agent_history_window),
                    evidence=evidence,
                    memories=state.get("recalled_memories", []),
                    route=route,
                    report=state.get("research_report"),
                    tool_calls=state.get("tool_calls", []),
                    feedback=feedback,
                    canary=ctx.canary,
                ),
                tier="reasoning",
                run_name="response_synthesis" if not feedback else "response_synthesis_retry",
            )
            emit(
                runtime,
                "response_agent",
                "generation",
                f"Generated answer from {len(evidence)} evidence items"
                + (" (retry with validator feedback)" if feedback else ""),
                attempt=attempts + 1,
                chars=len(draft),
            )
            return {"draft": draft}
        except LLMUnavailableError:
            emit(runtime, "response_agent", "warning", "LLM unavailable; falling back to an extractive answer")

    draft = (
        direct_answer(ctx.principal)
        if strategy == "direct"
        else extractive_answer(evidence, state.get("research_report"))
    )
    emit(runtime, "response_agent", "generation", "Composed deterministic answer (degraded mode)", strategy=strategy)
    return {"draft": draft, "degraded": ["response_agent"]}

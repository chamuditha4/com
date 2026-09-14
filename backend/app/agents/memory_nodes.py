"""Memory nodes: `prepare_turn` (load context, reset turn state) and `memory_writer`."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agents.context import AgentContext
from app.agents.prompts import memory_extraction_messages
from app.agents.state import RESET, AgentState
from app.core.exceptions import LLMUnavailableError
from app.memory.long_term import has_memory_cue, heuristic_facts
from app.observability.events import emit


class MemoryExtraction(BaseModel):
    facts: list[str] = Field(default_factory=list, max_length=3)


async def prepare_turn(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    settings = ctx.services.settings
    messages = state.get("messages", [])
    question = next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), "")

    # Short-term memory: keep a bounded window of the conversation in the checkpoint.
    window = settings.agent_history_window
    removals = [RemoveMessage(id=m.id) for m in messages[:-window]] if len(messages) > window else []

    memories: list[str] = []
    degraded: list[str] = []
    if ctx.services.memory and settings.long_term_memory_enabled:
        try:
            memories = await ctx.services.memory.recall(ctx.principal.user_id, question)
        except Exception:
            degraded.append("long_term_memory")
            emit(runtime, "prepare_turn", "warning", "Long-term memory unavailable; continuing without it")

    emit(
        runtime,
        "prepare_turn",
        "memory",
        f"Loaded {len(messages) - 1 - len(removals)} prior messages and recalled {len(memories)} long-term memories",
        trimmed=len(removals),
        memories=memories,
        session_id=ctx.session_id,
    )
    # Reset every turn-scoped field so nothing from the previous question leaks into this one.
    return {
        "messages": removals,
        "question": question,
        "recalled_memories": memories,
        "input_guard": None,
        "route": None,
        "retrieval_status": "not_run",
        "evidence": [],
        "exploration": None,
        "research_plan": None,
        "research_report": None,
        "tool_calls": [],
        "tool_messages": [],
        "tool_iterations": 0,
        "tool_loop_continue": False,
        "draft": None,
        "attempts": 0,
        "validation": None,
        "final": None,
        "degraded": [RESET, *degraded],  # reset turn-scoped degradations, keep this node's
    }


async def memory_writer(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    memory = ctx.services.memory
    guard = state.get("input_guard")
    question = state.get("question", "")
    if (
        memory is None
        or not ctx.services.settings.long_term_memory_enabled
        or (guard is not None and not guard.allowed)
        or not has_memory_cue(question)
    ):
        return {}

    facts: list[str] = []
    extracted_by = "heuristic"
    if ctx.services.llm.available:
        try:
            result = await ctx.services.llm.structured(
                memory_extraction_messages(question), MemoryExtraction, tier="fast", run_name="memory_extraction"
            )
            facts, extracted_by = result.facts, "llm"
        except LLMUnavailableError:
            facts = heuristic_facts(question)
    else:
        facts = heuristic_facts(question)

    try:
        stored = await memory.remember(ctx.principal.user_id, facts)
    except Exception:
        emit(runtime, "memory_writer", "warning", "Could not persist long-term memory")
        return {}
    if stored:
        emit(
            runtime,
            "memory_writer",
            "memory",
            f"Stored {len(stored)} long-term memories",
            facts=stored,
            extracted_by=extracted_by,
        )
    return {}

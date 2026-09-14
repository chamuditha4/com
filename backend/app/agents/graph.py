"""LangGraph wiring for the Enterprise AI Assistant.

    START → prepare_turn → input_guard ─┬─(blocked)──────────────────────────────┐
                                        └─► supervisor ─┬─► retrieval_agent ──┐   │
                                                        ├─► research_agent ───┤   │  (RLM subgraph)
                                                        ├─► tool_planner ⇄ tool_approval → tool_executor
                                                        └─► (direct) ─────────┤   │
                                                                              ▼   │
                                              response_agent ⇄ validator ─► finalize → memory_writer → END

Every node is a plain async function of `(state, runtime)`. Dependencies arrive through
`runtime.context` (see `context.py`), so the graph can be exercised end to end in tests with
offline adapters.
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from app.agents.context import AgentContext
from app.agents.memory_nodes import memory_writer, prepare_turn
from app.agents.research_agent import build_research_graph
from app.agents.response_agent import response_agent
from app.agents.retrieval_agent import retrieval_agent
from app.agents.state import AgentState
from app.agents.supervisor import input_guard, route_after_guard, route_by_strategy, supervisor
from app.agents.tool_agent import route_after_executor, route_after_planner, tool_approval, tool_executor, tool_planner
from app.agents.validator import finalize, route_after_validation, validator


def build_agent_graph(
    *, checkpointer: BaseCheckpointSaver | None = None, store: BaseStore | None = None
) -> CompiledStateGraph:
    graph = StateGraph(AgentState, context_schema=AgentContext)

    graph.add_node("prepare_turn", prepare_turn)
    graph.add_node("input_guard", input_guard)
    graph.add_node("supervisor", supervisor)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("research_agent", build_research_graph())
    graph.add_node("tool_planner", tool_planner)
    graph.add_node("tool_approval", tool_approval)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("response_agent", response_agent)
    graph.add_node("validator", validator)
    graph.add_node("finalize", finalize)
    graph.add_node("memory_writer", memory_writer)

    graph.add_edge(START, "prepare_turn")
    graph.add_edge("prepare_turn", "input_guard")
    graph.add_conditional_edges("input_guard", route_after_guard, ["supervisor", "finalize"])
    graph.add_conditional_edges(
        "supervisor", route_by_strategy, ["retrieval_agent", "research_agent", "tool_planner", "response_agent"]
    )
    graph.add_edge("retrieval_agent", "response_agent")
    graph.add_edge("research_agent", "response_agent")
    graph.add_conditional_edges("tool_planner", route_after_planner, ["tool_approval", "response_agent"])
    graph.add_edge("tool_approval", "tool_executor")
    graph.add_conditional_edges("tool_executor", route_after_executor, ["tool_planner", "response_agent"])
    graph.add_edge("response_agent", "validator")
    graph.add_conditional_edges("validator", route_after_validation, ["response_agent", "finalize"])
    graph.add_edge("finalize", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer, store=store, name="enterprise_assistant")

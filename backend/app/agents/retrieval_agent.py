"""Retrieval Agent: a single hybrid-retrieval pass for focused questions."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from app.agents.context import AgentContext
from app.agents.state import AgentState, Evidence
from app.core.exceptions import RetrievalUnavailableError
from app.observability.events import emit


async def retrieval_agent(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    retriever = ctx.services.retriever
    route = state["route"]
    assert route is not None
    filters = route.filters()

    emit(
        runtime,
        "retrieval_agent",
        "retrieval",
        "Running hybrid search (dense + BM25)",
        query=route.search_query,
        filters=filters.describe() if filters else "none",
    )
    try:
        result = await retriever.search(route.search_query, principal=ctx.principal, filters=filters)
        if not result.chunks and filters is not None:
            emit(runtime, "retrieval_agent", "decision", "No results with filters; relaxing to an unfiltered search")
            result = await retriever.search(route.search_query, principal=ctx.principal)
    except RetrievalUnavailableError:
        emit(runtime, "retrieval_agent", "error", "Knowledge base unavailable (both retrieval legs failed)")
        return {"evidence": [], "retrieval_status": "unavailable", "degraded": ["retrieval"]}

    evidence = [Evidence.from_chunk(i, chunk) for i, chunk in enumerate(result.chunks, start=1)]
    d = result.diagnostics
    emit(
        runtime,
        "retrieval_agent",
        "retrieval",
        f"Retrieved {len(evidence)} passages ({d.dense_hits} dense, {d.sparse_hits} sparse → {d.fused_candidates} fused, reranked by {d.reranker}) in {d.latency_ms} ms",
        diagnostics=d.model_dump(mode="json"),
        results=[
            {
                "id": e.id,
                "source": e.label,
                "access_level": e.access_level,
                "score": e.score,
                "dense_rank": c.dense_rank,
                "sparse_rank": c.sparse_rank,
            }
            for e, c in zip(evidence, result.chunks, strict=True)
        ],
    )
    if d.flagged_injection:
        emit(
            runtime,
            "retrieval_agent",
            "guardrail",
            f"{d.flagged_injection} retrieved passage(s) contain instruction-like text; marked as untrusted",
        )
    return {
        "evidence": evidence,
        "retrieval_status": "ok" if evidence else "empty",
        "degraded": [f"retrieval:{leg}" for leg in d.degraded_legs],
    }

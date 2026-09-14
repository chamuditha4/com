"""Research Agent: Recursive Language Model (RLM) exploration as a LangGraph subgraph.

    explore ──► plan ──► Send(analyze_batch) × N  ──► aggregate
    (metadata)  (Python   │  each batch: analyze_slice(depth=0)
                 plan)    │     too many docs? split by date → analyze_slice(depth+1) …
                          │     leaf: targeted retrieval → sub-agent extraction
                          ▼
                     batch_findings (reducer merges parallel results)

1. **Explore:** read catalog metadata (counts by department/type/date), never full text.
2. **Plan:** the LLM writes a Python plan that is interpreted by an AST whitelist (`rlm_plan.py`).
3. **Decompose:** each `batch(...)` becomes a parallel `Send` to a batch worker.
4. **Retrieve targeted sections:** a leaf slice searches only its own documents (`doc_ids` filter).
5. **Recurse:** a slice holding more documents than `RLM_MAX_DOCS_PER_SLICE` splits at its median
   date and analyzes both halves, down to `RLM_MAX_DEPTH`. Each call is its own LangSmith run,
   so the recursion tree is visible in the trace.
6. **Aggregate:** deduplicate incidents, tally root causes in plain Python (deterministic and
   auditable), and number the evidence for citation.

Bounded fan-out: at most `RLM_MAX_BATCHES` batches × 2^`RLM_MAX_DEPTH` leaves per query, and
a worker-wide semaphore caps concurrent slice analysis.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from langsmith import traceable
from pydantic import BaseModel, Field

from app.agents.context import AgentContext
from app.agents.heuristics import heuristic_incident_findings, normalize_root_cause
from app.agents.prompts import extraction_messages, plan_messages
from app.agents.rlm_plan import PlanValidationError, heuristic_plan, parse_plan, render_plan
from app.agents.state import (
    BatchFinding,
    Evidence,
    IncidentFinding,
    ResearchBatch,
    ResearchOutput,
    ResearchPlan,
    ResearchReport,
    ResearchState,
    RootCauseTally,
)
from app.core.exceptions import LLMUnavailableError, RetrievalUnavailableError
from app.observability.events import emit
from app.retrieval.models import DocumentMetadata, RetrievedChunk

MAX_RESEARCH_EVIDENCE = 30


# --- 1. explore ------------------------------------------------------------------------------


async def explore(state: ResearchState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    catalog = ctx.services.retriever.catalog
    route = state["route"]
    assert route is not None
    clearance = ctx.principal.allowed_access_levels

    overview = catalog.overview(clearance)
    scoped = catalog.overview(clearance, route.filters())
    emit(
        runtime,
        "research.explore",
        "plan",
        f"Explored catalog metadata: {overview.total_documents} documents visible to this role, {scoped.total_documents} match the supervisor's scope",
        visible=overview.model_dump(mode="json"),
        in_scope=scoped.model_dump(mode="json"),
    )
    return {"exploration": overview, "batch_findings": None}


# --- 2. plan ---------------------------------------------------------------------------------


async def plan(state: ResearchState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    settings = ctx.services.settings
    route, overview = state["route"], state["exploration"]
    assert route is not None and overview is not None
    departments = ctx.services.retriever.catalog.departments

    research_plan: ResearchPlan | None = None
    if ctx.services.llm.available:
        try:
            source = await ctx.services.llm.generate(
                plan_messages(
                    question=state["question"],
                    route=route,
                    overview=overview,
                    today=ctx.today,
                    max_batches=settings.rlm_max_batches,
                ),
                tier="reasoning",
                run_name="rlm_generate_python_plan",
            )
            batches, warnings = parse_plan(source, known_departments=departments, max_batches=settings.rlm_max_batches)
            research_plan = ResearchPlan(
                source=source.strip().strip("`"), batches=batches, planned_by="llm", warnings=warnings
            )
        except (LLMUnavailableError, PlanValidationError) as exc:
            emit(runtime, "research.plan", "warning", f"LLM plan rejected ({exc}); using deterministic planner")

    if research_plan is None:
        batches = heuristic_plan(
            question=state["question"],
            route=route,
            overview=overview,
            known_departments=departments,
            today=ctx.today,
            max_batches=settings.rlm_max_batches,
        )
        research_plan = ResearchPlan(
            source=render_plan(
                batches, comment="Deterministic plan: resolved time window split into calendar quarters"
            ),
            batches=batches,
            planned_by="heuristic",
        )

    emit(
        runtime,
        "research.plan",
        "plan",
        f"Generated Python search plan with {len(research_plan.batches)} batches ({research_plan.planned_by})",
        source=research_plan.source,
        batches=[{"label": b.label, "filters": b.filters.describe()} for b in research_plan.batches],
        warnings=research_plan.warnings,
    )
    return {
        "research_plan": research_plan,
        "degraded": ["research_planner"] if research_plan.planned_by == "heuristic" else [],
    }


def dispatch_batches(state: ResearchState) -> list[Send] | str:
    research_plan = state.get("research_plan")
    if not research_plan or not research_plan.batches:
        return "aggregate"
    return [
        Send("analyze_batch", {"batch": b, "question": state["question"], "route": state["route"]})
        for b in research_plan.batches
    ]


# --- 3-5. decompose, retrieve targeted sections, recurse -------------------------------------


class ExtractedIncident(BaseModel):
    doc_id: str
    relevant: bool = Field(
        description="True only if this incident matches the subject of the question (e.g. it affected payments when the question is about payment failures)."
    )
    relevance_reason: str = Field(description="One short sentence justifying the relevance decision.")
    title: str = ""
    date: str = ""
    root_cause_category: str
    root_cause_summary: str
    impact_summary: str = ""
    chunk_ids: list[str] = Field(default_factory=list)


class SliceExtraction(BaseModel):
    incidents: list[ExtractedIncident] = Field(default_factory=list)
    summary: str = ""


def ground_findings(
    extraction: SliceExtraction, chunks: list[RetrievedChunk]
) -> tuple[list[IncidentFinding], int, list[str]]:
    """Keep only relevant findings that point at documents and chunks actually retrieved for this slice.

    Returns (findings, dropped_hallucinated, excluded_as_irrelevant). Titles and dates come from
    document metadata, not from the model. Relevance exclusions are returned (not silently
    discarded) so they can be shown in the Activity Panel and trace."""
    by_doc = {c.chunk.doc_id: c for c in chunks}
    by_chunk = {c.chunk.chunk_id: c for c in chunks}
    findings: list[IncidentFinding] = []
    dropped, excluded = 0, []
    for inc in extraction.incidents:
        if inc.doc_id not in by_doc:
            dropped += 1
            continue
        if not inc.relevant:
            excluded.append(f"{inc.doc_id}: {inc.relevance_reason[:200]}")
            continue
        md = by_doc[inc.doc_id].chunk.metadata
        chunk_ids = [cid for cid in inc.chunk_ids if cid in by_chunk and by_chunk[cid].chunk.doc_id == inc.doc_id]
        if not chunk_ids:
            chunk_ids = [c.chunk.chunk_id for c in chunks if c.chunk.doc_id == inc.doc_id][:2]
        findings.append(
            IncidentFinding(
                doc_id=inc.doc_id,
                title=md.title,
                date=md.created_date.isoformat(),
                root_cause_category=normalize_root_cause(inc.root_cause_category, inc.root_cause_summary),
                root_cause_summary=inc.root_cause_summary[:600],
                impact_summary=inc.impact_summary[:400],
                chunk_ids=chunk_ids,
            )
        )
    return findings, dropped, excluded


def split_batch(batch: ResearchBatch, docs: list[DocumentMetadata]) -> list[ResearchBatch] | None:
    """Split a slice at its median document date into two non-overlapping date ranges."""
    dates = sorted(d.created_date for d in docs)
    mid = len(dates) // 2
    # Move the split point so the halves are separated by a real date boundary.
    while 0 < mid < len(dates) and dates[mid - 1] == dates[mid]:
        mid += 1
    if mid <= 0 or mid >= len(dates):
        return None
    left_end = dates[mid - 1]
    f = batch.filters
    left = f.model_copy(update={"date_from": f.date_from or dates[0], "date_to": left_end})
    right = f.model_copy(update={"date_from": left_end + timedelta(days=1), "date_to": f.date_to or dates[-1]})
    return [
        batch.model_copy(update={"label": f"{batch.label}.a", "filters": left}),
        batch.model_copy(update={"label": f"{batch.label}.b", "filters": right}),
    ]


def diversify_by_document(chunks: list[RetrievedChunk], *, per_doc: int) -> list[RetrievedChunk]:
    """Keep the best `per_doc` passages of every document, preserving relevance order."""
    taken: Counter[str] = Counter()
    selected = []
    for chunk in chunks:
        if taken[chunk.chunk.doc_id] < per_doc:
            taken[chunk.chunk.doc_id] += 1
            selected.append(chunk)
    return selected


def _slice_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    batch = inputs["batch"]
    return {"label": batch.label, "filters": batch.filters.describe(), "depth": inputs["depth"]}


def _slice_outputs(outputs: Any) -> dict[str, Any]:
    return (
        {"findings": [f.model_dump(exclude={"chunks"}) for f in outputs]}
        if isinstance(outputs, list)
        else {"output": outputs}
    )


@traceable(name="rlm_analyze_slice", run_type="chain", process_inputs=_slice_inputs, process_outputs=_slice_outputs)
async def analyze_slice(
    batch: ResearchBatch, depth: int, question: str, runtime: Runtime[AgentContext]
) -> list[BatchFinding]:
    ctx = runtime.context
    services = ctx.services
    settings = services.settings
    docs = services.retriever.catalog.visible(ctx.principal.allowed_access_levels, batch.filters)
    base = {"label": batch.label, "depth": depth, "filters": batch.filters.describe(), "documents_in_scope": len(docs)}

    if not docs:
        emit(runtime, "research.analyze_batch", "batch", f"Slice {batch.label}: no documents in scope", **base)
        return [BatchFinding(**base, chunks_analyzed=0, summary="No documents in scope.")]

    if len(docs) > settings.rlm_max_docs_per_slice and depth < settings.rlm_max_depth:
        halves = split_batch(batch, docs)
        if halves:
            emit(
                runtime,
                "research.analyze_batch",
                "recursion",
                f"Slice {batch.label} holds {len(docs)} documents (> {settings.rlm_max_docs_per_slice}); recursing into {len(halves)} sub-slices at depth {depth + 1}",
                **base,
                children=[h.label for h in halves],
            )
            results = await asyncio.gather(*(analyze_slice(h, depth + 1, question, runtime) for h in halves))
            return [finding for group in results for finding in group]

    # Leaf: retrieve only the targeted sections of this slice's documents, with a per-document
    # quota so one lexically dominant document cannot crowd the others out of the slice.
    filters = batch.filters.model_copy(update={"doc_ids": [d.doc_id for d in docs][:50]})
    per_doc = max(2, settings.rlm_chunks_per_slice // len(docs))
    async with services.research_semaphore:
        try:
            result = await services.retriever.search(
                f"root cause and customer impact: {batch.query}",
                principal=ctx.principal,
                filters=filters,
                top_n=min(60, 8 * len(docs)),
            )
        except RetrievalUnavailableError:
            emit(runtime, "research.analyze_batch", "error", f"Slice {batch.label}: retrieval unavailable", **base)
            return [BatchFinding(**base, chunks_analyzed=0, error="retrieval unavailable")]
        chunks = diversify_by_document(result.chunks, per_doc=per_doc)

        incidents: list[IncidentFinding] = []

        excluded: list[str] = []
        analyzed_by = "heuristic"
        summary = ""
        if services.llm.available and chunks:
            try:
                extraction = await services.llm.structured(
                    extraction_messages(question=question, batch=batch, chunks=chunks),
                    SliceExtraction,
                    tier="fast",
                    run_name=f"rlm_subagent_extract[{batch.label}]",
                )
                incidents, dropped, excluded = ground_findings(extraction, chunks)
                analyzed_by, summary = "llm", extraction.summary[:500]
                if dropped:
                    emit(
                        runtime,
                        "research.analyze_batch",
                        "guardrail",
                        f"Slice {batch.label}: dropped {dropped} finding(s) citing documents that were not retrieved",
                    )
            except LLMUnavailableError:
                pass
        if excluded:
            emit(
                runtime,
                "research.analyze_batch",
                "decision",
                f"Slice {batch.label}: excluded {len(excluded)} incident(s) as not relevant to the question",
                excluded=excluded,
            )
        if analyzed_by == "heuristic":
            incidents = heuristic_incident_findings(chunks)

    emit(
        runtime,
        "research.analyze_batch",
        "batch",
        f"Slice {batch.label} (depth {depth}): {len(docs)} documents, {len(chunks)} targeted passages → {len(incidents)} incidents ({analyzed_by})",
        **base,
        incidents=[f"{i.doc_id}: {i.root_cause_category}" for i in incidents],
    )
    return [
        BatchFinding(
            **base,
            chunks_analyzed=len(chunks),
            incidents=incidents,
            summary=summary,
            analyzed_by=analyzed_by,
            excluded=excluded,
            chunks=chunks,
        )
    ]


async def analyze_batch(state: ResearchState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    findings = await analyze_slice(state["batch"], 0, state["question"], runtime)
    return {"batch_findings": findings}


# --- 6. aggregate ----------------------------------------------------------------------------


def aggregate_findings(question: str, findings: list[BatchFinding]) -> tuple[ResearchReport, list[Evidence]]:
    """Pure aggregation: dedupe incidents, number evidence, tally root causes."""
    chunks: dict[str, RetrievedChunk] = {c.chunk.chunk_id: c for f in findings for c in f.chunks}

    incidents_by_doc: dict[str, IncidentFinding] = {}
    for finding in findings:
        for inc in finding.incidents:
            if inc.doc_id in incidents_by_doc:
                existing = incidents_by_doc[inc.doc_id]
                merged = existing.chunk_ids + [c for c in inc.chunk_ids if c not in existing.chunk_ids]
                incidents_by_doc[inc.doc_id] = existing.model_copy(update={"chunk_ids": merged})
            else:
                incidents_by_doc[inc.doc_id] = inc
    incidents = sorted(incidents_by_doc.values(), key=lambda i: (i.date, i.doc_id))

    # Round-robin allocation: every incident gets its best passage before any incident gets a
    # second one, so the evidence cap can never starve later incidents of citable evidence.
    evidence: list[Evidence] = []
    evidence_id: dict[str, int] = {}
    queues = [[cid for cid in inc.chunk_ids if cid in chunks] for inc in incidents]
    depth = 0
    while len(evidence) < MAX_RESEARCH_EVIDENCE and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and queue[depth] not in evidence_id and len(evidence) < MAX_RESEARCH_EVIDENCE:
                cid = queue[depth]
                evidence_id[cid] = len(evidence) + 1
                evidence.append(Evidence.from_chunk(evidence_id[cid], chunks[cid]))
        depth += 1
    if not incidents:  # non-incident research: pass the best passages through for synthesis
        for c in sorted(chunks.values(), key=lambda c: -c.score)[:12]:
            evidence.append(Evidence.from_chunk(len(evidence) + 1, c))

    counts = Counter(inc.root_cause_category for inc in incidents)
    tallies = [
        RootCauseTally(
            category=category,
            count=count,
            doc_ids=[i.doc_id for i in incidents if i.root_cause_category == category],
            evidence_ids=[
                evidence_id[c]
                for i in incidents
                if i.root_cause_category == category
                for c in i.chunk_ids
                if c in evidence_id
            ],
        )
        for category, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    report = ResearchReport(
        question=question,
        batches_analyzed=len(findings),
        max_depth_reached=max((f.depth for f in findings), default=0),
        documents_in_scope=sum(f.documents_in_scope for f in findings),
        incidents=incidents,
        root_causes=tallies,
        failed_batches=[f.label for f in findings if f.error],
    )
    return report, evidence


async def aggregate(state: ResearchState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    findings = state.get("batch_findings") or []
    report, evidence = aggregate_findings(state["question"], findings)

    emit(
        runtime,
        "research.aggregate",
        "aggregate",
        f"Aggregated {len(findings)} slice results: {len(report.incidents)} incidents, {len(report.recurring)} recurring root causes",
        root_causes=[t.model_dump() for t in report.root_causes],
        failed_batches=report.failed_batches,
        max_depth=report.max_depth_reached,
    )
    all_failed = bool(findings) and all(f.error for f in findings)
    degraded = ["research_extraction"] if any(f.analyzed_by == "heuristic" for f in findings) else []
    if report.failed_batches:
        degraded.append("research_partial_coverage")
    return {
        "research_report": report,
        "evidence": evidence,
        "retrieval_status": "unavailable" if all_failed else ("ok" if evidence else "empty"),
        "degraded": degraded,
    }


def build_research_graph():
    graph = StateGraph(ResearchState, context_schema=AgentContext, output_schema=ResearchOutput)
    graph.add_node("explore", explore)
    graph.add_node("plan", plan)
    graph.add_node("analyze_batch", analyze_batch)
    graph.add_node("aggregate", aggregate)
    graph.add_edge(START, "explore")
    graph.add_edge("explore", "plan")
    graph.add_conditional_edges("plan", dispatch_batches, ["analyze_batch", "aggregate"])
    graph.add_edge("analyze_batch", "aggregate")
    graph.add_edge("aggregate", END)
    return graph.compile(name="research_agent")

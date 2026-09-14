"""Typed agent state: the single source of truth for a conversation turn.

Two lifetimes live in one state object:

* **Conversation scope** (`messages`): persisted by the checkpointer across turns, keyed by
  thread id = `{user_id}:{session_id}`. This is short-term memory.
* **Turn scope** (everything else): reset by `prepare_turn` at the start of each turn, so
  evidence or validation results from a previous question can never leak into the next answer.

Nodes return partial updates; they never mutate state in place. Fields written by parallel
branches (RLM batch workers) or by several nodes use explicit reducers.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.retrieval.catalog import CatalogOverview
from app.retrieval.models import DocumentType, RetrievedChunk, SearchFilters

Strategy = Literal["retrieval", "research", "tools", "direct"]


# --- reducers ------------------------------------------------------------------------------


def append_or_reset(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    """Concatenate updates from parallel branches; a `None` update resets the list (new turn)."""
    if right is None:
        return []
    return [*(left or []), *right]


RESET = "__reset__"


def union_or_reset(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Order-preserving set union. `None` resets; `[RESET, *items]` resets then adds items.

    Union (not concatenation) is required because a subgraph returns the parent's values
    alongside its own additions."""
    if right is None:
        return []
    if right and right[0] == RESET:
        left, right = [], right[1:]
    merged = list(left or [])
    merged.extend(item for item in right if item not in merged)
    return merged


# --- turn models -----------------------------------------------------------------------------


class GuardVerdict(BaseModel):
    allowed: bool
    score: float
    signals: list[str] = Field(default_factory=list)
    reason: str | None = None


class RouteDecision(BaseModel):
    """Supervisor output. Also the structured-output schema the LLM fills in."""

    strategy: Strategy = Field(
        description=(
            "retrieval: a focused question answerable from a few passages. "
            "research: broad/aggregative questions over many documents (summaries, trends, recurring causes, time ranges). "
            "tools: needs live operational data (tickets, service status, on-call, metrics) or calculations. "
            "direct: greetings, thanks, or questions about the assistant itself."
        )
    )
    intent: str = Field(max_length=200, description="One-line description of what the user wants.")
    search_query: str = Field(
        max_length=500,
        description="A standalone search query that resolves pronouns and references using the conversation history.",
    )
    departments: list[str] = Field(
        default_factory=list, description="Department slugs to scope the search, if clearly implied."
    )
    document_types: list[DocumentType] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    rationale: str = Field(max_length=500, description="Why this strategy was chosen.")

    def filters(self) -> SearchFilters | None:
        filters = SearchFilters(
            departments=self.departments or None,
            document_types=self.document_types or None,
            date_from=self.date_from,
            date_to=self.date_to,
        )
        return filters if filters.describe() != "none" else None


class Evidence(BaseModel):
    """A citable unit shown to the response model as `[id]`."""

    id: int
    source_type: Literal["document", "tool"]
    title: str
    text: str
    chunk_id: str | None = None
    doc_id: str | None = None
    section: str | None = None
    department: str | None = None
    document_type: str | None = None
    access_level: str | None = None
    created_date: str | None = None
    score: float | None = None
    injection_signals: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        if self.source_type == "tool":
            return f"Tool · {self.title}"
        return f"{self.doc_id} · {self.title} · {self.section}"

    @classmethod
    def from_chunk(cls, id_: int, item: RetrievedChunk) -> Evidence:
        md = item.chunk.metadata
        return cls(
            id=id_,
            source_type="document",
            title=md.title,
            text=item.chunk.text,
            chunk_id=item.chunk.chunk_id,
            doc_id=md.doc_id,
            section=item.chunk.section,
            department=md.department,
            document_type=md.document_type.value,
            access_level=md.access_level.value,
            created_date=md.created_date.isoformat(),
            score=round(item.score, 4),
            injection_signals=item.injection_signals,
        )


# --- research (RLM) models -------------------------------------------------------------------


class ResearchBatch(BaseModel):
    label: str = Field(max_length=80)
    query: str = Field(max_length=300)
    filters: SearchFilters


class ResearchPlan(BaseModel):
    source: str  # the Python plan, exactly as executed by the plan interpreter
    batches: list[ResearchBatch]
    planned_by: Literal["llm", "heuristic"]
    warnings: list[str] = Field(default_factory=list)


class IncidentFinding(BaseModel):
    doc_id: str
    title: str
    date: str
    root_cause_category: str
    root_cause_summary: str = Field(max_length=600)
    impact_summary: str = Field(default="", max_length=400)
    chunk_ids: list[str] = Field(default_factory=list)


class BatchFinding(BaseModel):
    label: str
    depth: int
    filters: str
    documents_in_scope: int
    chunks_analyzed: int
    incidents: list[IncidentFinding] = Field(default_factory=list)
    summary: str = ""
    analyzed_by: Literal["llm", "heuristic", "none"] = "none"
    error: str | None = None
    excluded: list[str] = Field(default_factory=list)  # "DOC-ID: reason" judged irrelevant
    chunks: list[RetrievedChunk] = Field(default_factory=list)


class RootCauseTally(BaseModel):
    category: str
    count: int
    doc_ids: list[str]
    evidence_ids: list[int]


class ResearchReport(BaseModel):
    question: str
    batches_analyzed: int
    max_depth_reached: int
    documents_in_scope: int
    incidents: list[IncidentFinding]
    root_causes: list[RootCauseTally]
    failed_batches: list[str] = Field(default_factory=list)

    @property
    def recurring(self) -> list[RootCauseTally]:
        return [t for t in self.root_causes if t.count >= 2]


# --- tools -----------------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    call_id: str
    tool: str
    args: dict[str, Any]
    status: Literal["pending", "approved", "rejected", "succeeded", "failed"] = "pending"
    requires_approval: bool = False
    origin: Literal["llm", "heuristic"] = "llm"
    output: str | None = None
    error: str | None = None
    latency_ms: float | None = None


# --- validation & final answer ---------------------------------------------------------------


class ValidationIssue(BaseModel):
    code: str
    message: str
    retryable: bool


class ValidationReport(BaseModel):
    passed: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list)
    redactions: list[str] = Field(default_factory=list)
    will_retry: bool = False

    @property
    def retryable(self) -> bool:
        return bool(self.issues) and all(i.retryable for i in self.issues)


class FinalAnswer(BaseModel):
    answer: str = Field(max_length=16_000)
    strategy: Strategy | Literal["blocked"]
    citations: list[Evidence] = Field(default_factory=list)
    validation_passed: bool
    degraded: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


RetrievalStatus = Literal["not_run", "ok", "empty", "unavailable"]


class AgentState(TypedDict, total=False):
    # Conversation scope (checkpointed)
    messages: Annotated[list[AnyMessage], add_messages]

    # Turn scope (reset by prepare_turn)
    question: str
    recalled_memories: list[str]
    input_guard: GuardVerdict | None
    route: RouteDecision | None
    retrieval_status: RetrievalStatus
    evidence: list[Evidence]
    exploration: CatalogOverview | None
    research_plan: ResearchPlan | None
    research_report: ResearchReport | None
    tool_calls: list[ToolCallRecord]
    tool_messages: list[AnyMessage]
    tool_iterations: int
    tool_loop_continue: bool
    draft: str | None
    attempts: int
    validation: ValidationReport | None
    final: FinalAnswer | None
    degraded: Annotated[list[str], union_or_reset]


class ResearchState(TypedDict, total=False):
    """State of the RLM research subgraph. Keys shared with AgentState map by name."""

    question: str
    route: RouteDecision | None
    exploration: CatalogOverview | None
    research_plan: ResearchPlan | None
    batch_findings: Annotated[list[BatchFinding], append_or_reset]
    batch: ResearchBatch  # payload of a Send() to a batch worker
    research_report: ResearchReport | None
    evidence: list[Evidence]
    retrieval_status: RetrievalStatus
    degraded: Annotated[list[str], union_or_reset]


class ResearchOutput(TypedDict, total=False):
    exploration: CatalogOverview | None
    research_plan: ResearchPlan | None
    research_report: ResearchReport | None
    evidence: list[Evidence]
    retrieval_status: RetrievalStatus
    degraded: Annotated[list[str], union_or_reset]


# Registered with the checkpoint serializer so these types round-trip safely (LangGraph blocks
# deserialization of unregistered classes).
CHECKPOINT_TYPES: tuple[type[BaseModel], ...] = (
    GuardVerdict,
    RouteDecision,
    Evidence,
    ResearchBatch,
    ResearchPlan,
    IncidentFinding,
    BatchFinding,
    RootCauseTally,
    ResearchReport,
    ToolCallRecord,
    ValidationIssue,
    ValidationReport,
    FinalAnswer,
    CatalogOverview,
    SearchFilters,
)

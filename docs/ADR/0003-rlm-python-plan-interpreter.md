# ADR-0003: RLM research as a LangGraph subgraph with an AST-interpreted Python plan

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

CLAUDE.md §5 requires the Research Agent to explore the collection, *generate a Python-based
search plan*, decompose into batches, retrieve targeted sections, recurse with sub-agents and
aggregate. Two risks:

1. Executing model-written Python is remote code execution by design.
2. Unbounded recursion or fan-out can exhaust a worker and the LLM rate limits.

## Options for the plan

| Option | Verdict |
|---|---|
| Execute the plan in the Python sandbox | Rejected. Adds an execution path for model output on every research query. The sandbox is for analytics, not control flow. |
| JSON plan via structured output | Safe, but loses the "Python-based" requirement and is less natural for models to write |
| **Python source interpreted by an AST whitelist** | Accepted. The model writes readable Python. We accept exactly `PLAN = [batch(k=literal, ...)]` and interpret it as data |

## Decision

- `rlm_plan.parse_plan` accepts a single `PLAN` assignment of `batch(...)` calls with literal
  keyword arguments (`ast.literal_eval`). It validates filters with Pydantic, drops unknown
  departments with a warning, and truncates to `RLM_MAX_BATCHES`. Anything else raises
  `PlanValidationError`, and the planner falls back to `heuristic_plan` (calendar quarters over the
  resolved window).
- The plan source is emitted to the Activity Panel and LangSmith exactly as interpreted.
- Topology: `explore → plan → Send(analyze_batch) × N → aggregate`, as a compiled subgraph with an
  explicit output schema.
- Recursion happens inside `analyze_slice` (a `@traceable` async function): split at the median
  document date when a slice exceeds `RLM_MAX_DOCS_PER_SLICE`, bounded by `RLM_MAX_DEPTH`.
  Recursion is driven by catalog metadata (cheap) rather than by retrieving and counting text.
- Leaves retrieve with a `doc_ids` filter and a per-document passage quota. Without the quota,
  one lexically dominant document crowded another document's Root Cause section out of a slice;
  an end-to-end test caught this.
- Aggregation is deterministic Python. The LLM extracts per-slice findings, and the code counts.

## Consequences

- The recursion tree and plan are fully visible and auditable. The tally is reproducible.
- `Send` gives parallel batches, and recursion inside a node gives bounded depth without
  graph-level cycles. The trade-off: sub-slices appear as nested LangSmith runs rather than as
  separate graph nodes.
- Extraction is shaped for incident reports (the canonical use case). Other research questions
  still work: when no incidents are extracted, the best passages flow to synthesis. A
  document-type-specific extraction schema is a natural extension.

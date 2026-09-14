# ADR-0004: Graph topology, runtime-context injection and interrupt-based approvals

- **Status:** Accepted
- **Date:** 2026-09-14

## Decisions

### 1. Principal and services via `runtime.context`, not state

State is checkpointed and replayed. If the principal lived in state, a resumed or replayed run
would authorize with a stale identity, and tests would need to mock module globals. With LangGraph's
`context_schema`, the API passes `AgentContext(principal, trace_id, services)` on every
invocation, including `Command(resume=...)`.

### 2. Tool loop as three nodes: `tool_planner → tool_approval → tool_executor`

LangGraph re-executes an interrupted node from its start on resume. Putting `interrupt()` in the
same node as the LLM tool-planning call would re-run the LLM on approval and could plan a
*different* call than the one the human approved. Separate nodes make approval apply to exactly
the recorded `ToolCallRecord`s.

### 3. Validator loop with bounded retry, then containment

`response_agent ⇄ validator` retries at most `AGENT_MAX_VALIDATION_RETRIES` times, and only for
retryable issues when an LLM is available. `finalize` is the only exit. It substitutes an
extractive answer, re-validated, when validation still fails. A bad generation cannot reach the user.

### 4. Conditional edges read flags, nodes decide

Edge functions receive state only. Decisions that need configuration or services
(`will_retry`, `tool_loop_continue`) are computed inside nodes and written to state, so routing
stays pure and testable.

### 5. Turn-scoped reset in `prepare_turn`

All non-conversation fields reset each turn. `degraded` uses a union reducer with a `RESET`
marker, because subgraph outputs echo parent values.

## Consequences

- Every node is a plain `async def node(state, runtime)`, unit-testable with offline adapters.
- HITL works identically over HTTP (`/chat/resume`) and in tests (`Command(resume=...)`).
- The API rejects a new message with `409` while an approval is pending, so an interrupted run
  cannot be silently abandoned.

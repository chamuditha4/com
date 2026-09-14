# Memory Design

## Short-term (session) memory

| Aspect | Design |
|---|---|
| Mechanism | LangGraph checkpointer; `AgentState.messages` with the `add_messages` reducer |
| Key | `thread_id = "{user_id}:{session_id}"`, built server-side from the verified principal |
| Stored | User questions and final (validated, redacted) answers; the last turn's scoped state (route, evidence, plan, validation) for inspection and HITL resume |
| Window | `prepare_turn` removes messages beyond `AGENT_HISTORY_WINDOW` (default 10) with `RemoveMessage`, so checkpoints stay bounded |
| Injection | Prior turns become chat messages for supervisor, tool planner and response agent. User turns are wrapped in `<history>`, and old citation markers are stripped |
| Retention | Redis checkpointer TTL `SESSION_TTL_MINUTES` (default 24h), refreshed on read |
| Backend | `CHECKPOINTER=memory` (tests, single process) or `redis` (shared across replicas) |
| Serialization | Explicit allow-list of state types (`memory/persistence.py`); tests run with `LANGGRAPH_STRICT_MSGPACK=true` |

Turn-scoped fields (evidence, research report, tool calls, validation, degraded) are reset at the
start of every turn. A follow-up question is re-retrieved against the current user's clearance,
never answered from stale evidence.

## Long-term (cross-session) memory

| Aspect | Design |
|---|---|
| What is stored | Short durable facts the user states about themselves or their preferences ("works in payments operations", "prefers bullet points") |
| What is never stored | Document content, tool output, secrets, card numbers, anything flagged as injection, facts about other people |
| When | `memory_writer` runs after `finalize`, only if the message contains a memory cue and the input was not blocked. LLM structured extraction (≤ 3 facts), with cue-sentence fallback |
| Validation | `is_storable()`: ≤ 200 chars, no redactable sensitive data, no injection signals |
| Key | `("memories", user_id)`, idempotent sha256 key per fact |
| Retention | At most 50 facts per user; oldest pruned. `DELETE /chat/memory` erases everything (right to be forgotten) |
| Recall | `prepare_turn` ranks the user's facts by lexical overlap with the question plus a boost for preferences, and injects the top 5 as `<memory>` data |
| Backend | LangGraph `BaseStore`: `InMemoryStore` or `AsyncRedisStore` |

### Isolation guarantees

- The user id comes from `runtime.context.principal`, never from state, request body or model output.
- Tests assert that a second user recalls nothing from the first user's memory, and that another
  user's session id returns an empty history.

### Why not a vector index for memories?

With ≤ 50 short facts per user, lexical ranking is cheaper, deterministic and explainable. The
`BaseStore` interface already supports semantic search (`index=` config), so switching is a
configuration change once memories grow.

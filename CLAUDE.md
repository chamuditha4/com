# CLAUDE.md

Guidance for Claude Code (and any AI/human contributor) working in this repository.

---

## 1. Project Overview

**Enterprise AI Assistant** — a production-grade, agentic RAG platform that answers questions from organizational knowledge sources (policies, architecture docs, runbooks, incident reports, product specs, meeting notes).

This is **not** "an LLM wired to a vector DB." The system demonstrates modern agentic patterns: multi-agent orchestration via LangGraph, Recursive Language Model (RLM) exploration, hybrid retrieval, full observability, RBAC, and defense-in-depth security. The fictional operating company is **Commercial Bank** — brand-safety and guardrails must reflect a regulated financial institution.

**Guiding principles for every change:**
- **Explainability over cleverness.** The evaluator inspects logic and code readability. Every agent decision, tool call, and retrieval must be observable and traceable.
- **POC-first, but scalable by design.** Ship the vertical slice that proves the concept, but never make an architectural choice that blocks horizontal scaling later (see §9).
- **Async everywhere.** No blocking I/O in request paths.
- **When a requirement is ambiguous, pick the best path, document the assumption in `docs/ASSUMPTIONS.md`, and keep moving.**

---

## 2. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Frontend | **Streamlit** | Functionality + transparency, not beauty. Agent Activity Panel is mandatory. |
| Backend | **Python 3.11+, FastAPI** | Fully async; streaming responses (SSE). |
| Orchestration | **LangGraph** (deep agents) | Supervisor + specialized sub-agents. |
| Retrieval | **Hybrid** (dense embeddings + BM25 sparse) | Fusion ranking + reranker. |
| Vector DB | **Pinecone** | Namespaces, metadata filtering, hybrid, attribution. |
| LLM | Provider-abstracted (Anthropic / OpenAI / Gemini) | Selection rationale in `docs/MODEL_SELECTION.md`. |
| Observability | **LangSmith** (mandatory) | Trace every conversation, tool call, transition, retrieval. |
| Tools | Knowledge Search, MCP server, Python Analysis | Role-gated. |
| Auth | Hardcoded users+roles (Option A) or Keycloak (Option B) | RBAC enforced server-side. |
| Rate limiting | **Token Bucket**, per-user | Configurable, graceful degradation. |
| Deployment | **Docker Compose** | Bonus but expected for a scalable submission. |

Do not swap any of these without recording the decision in `docs/ADR/`.

---

## 3. Repository Layout

```
.
├── CLAUDE.md
├── README.md                     # Setup, run, demo pointers
├── docker-compose.yml
├── .env.example                  # Never commit real secrets
├── pyproject.toml / requirements.txt
├── docs/
│   ├── ARCHITECTURE.md
│   ├── architecture-diagram.(png|drawio|excalidraw)
│   ├── ASSUMPTIONS.md            # Running log of every assumption + trade-off
│   ├── MODEL_SELECTION.md
│   ├── SECURITY.md               # Prompt-injection, guardrails, RBAC design
│   ├── MEMORY_DESIGN.md
│   └── ADR/                      # Architecture Decision Records
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app factory, lifespan, routers
│   │   ├── api/                  # Routers: /chat (stream), /auth, /health, /admin
│   │   ├── core/                 # config, logging, security, rate_limit, exceptions
│   │   ├── agents/               # LangGraph graph + nodes
│   │   │   ├── graph.py          # Graph wiring, state schema, checkpointer
│   │   │   ├── state.py          # Typed AgentState (single source of truth)
│   │   │   ├── supervisor.py
│   │   │   ├── retrieval_agent.py
│   │   │   ├── research_agent.py # RLM: plan → decompose → recurse → aggregate
│   │   │   ├── response_agent.py
│   │   │   └── validator.py      # Citation + guardrail validation node
│   │   ├── retrieval/            # dense, sparse, hybrid fusion, reranker
│   │   ├── tools/                # knowledge_search, python_analysis, mcp_client
│   │   ├── memory/               # short-term (session) + long-term (bonus)
│   │   ├── auth/                 # users, roles, RBAC dependency
│   │   └── observability/        # LangSmith setup, structured logging, trace ids
│   ├── mcp_server/               # Simple MCP server (dummy enterprise data)
│   └── tests/
├── frontend/
│   └── streamlit_app.py          # Chat window + Agent Activity Panel
├── data/
│   ├── ingest.py                 # Chunk → embed → upsert to Pinecone
│   └── mock/                     # Generated sample docs (with metadata)
└── scripts/
```

---

## 4. AI Architecture (LangGraph)

The graph is the orchestration engine. **A single typed `AgentState` is the source of truth**; every node reads and returns partial state updates — never mutate global variables.

**Nodes:**
- **Supervisor** — intent understanding, task decomposition, routing. Decides which agent(s) to invoke and whether the query needs RLM-style deep research vs. a single retrieval pass.
- **Retrieval Agent** — hybrid RAG (dense + sparse), metadata-filtered by the caller's role/access level.
- **Research Agent (RLM)** — for broad/aggregative queries. Implements the Recursive Language Model concept (§5).
- **Response Agent** — final answer synthesis with inline citations.
- **Validator** — post-generation node: verifies citations map to real retrieved chunks (blocks hallucinated citations), runs guardrails, checks role compliance.

**Wiring rules:**
- Conditional edges route Supervisor → {Retrieval | Research} → Response → Validator → (END | retry).
- Use a **checkpointer** (e.g. `MemorySaver` for POC, Postgres/Redis-backed for scale) so state survives turns and enables human-in-the-loop interrupts.
- Every node emits a structured event to the Agent Activity stream (current node, tool calls, retrieval status, memory updates, validation results).
- Bonus **human-in-the-loop**: an interrupt node before executing sensitive/admin tools, awaiting approval.

---

## 5. Recursive Language Model (RLM)

Do **not** stuff whole document collections into context. The Research Agent must:

1. **Explore** the collection (list namespaces/metadata, not full text).
2. **Generate a Python-based search plan** (a structured plan of sub-queries/filters).
3. **Decompose** large tasks into batches (e.g. by date range, department, doc type).
4. **Retrieve targeted sections** per batch.
5. **Recurse** — spawn sub-agent invocations per batch, each analyzing only its slice.
6. **Aggregate** partial findings into a final synthesized answer.

Canonical test case (must work in the demo):
> "Summarize all outage reports related to payment failures during the last year and identify recurring root causes."

Expected: filter → batch → analyze-per-batch → aggregate → cited summary. A **simplified** implementation is acceptable, but the recursion/decomposition/aggregation concept must be visibly demonstrated and traced in LangSmith.

---

## 6. Retrieval

- **Dense:** embeddings (document + query).
- **Sparse:** BM25 / keyword.
- **Hybrid ranking:** combine dense + sparse scores (weighted fusion or RRF). Make weights configurable.
- **Reranker (bonus, recommended):** cross-encoder rerank on the fused top-k before it reaches the LLM.
- **Pinecone:** use **namespaces** (e.g. per department), **metadata filtering**, hybrid retrieval, and always return **document attribution** (source id, title, section) for citations.

Metadata schema (attach on ingest, filter on query):
```json
{
  "department": "payments",
  "document_type": "incident",
  "access_level": "internal",
  "created_date": "2025-01-01"
}
```

**Retrieval must always apply the caller's `access_level` as a hard metadata filter** — this is a security boundary, not a ranking hint (§8).

---

## 7. Memory

- **Short-term (session):** multi-turn context — user context, previous questions, relevant history. Must survive multiple turns within a session (LangGraph checkpointer keyed by session/thread id).
- **Long-term (bonus):** persist salient facts/preferences across sessions (e.g. vector or KV store), retrieved by relevance.
- Document all memory design decisions in `docs/MEMORY_DESIGN.md`: what is stored, retention, how it's injected, and how it's isolated per user (no cross-user leakage).

---

## 8. Security, Guardrails & RBAC

Treat this as a regulated bank. Security is enforced **server-side**; the agent must never be able to talk its way past it.

**Prompt-injection protection** (document in `docs/SECURITY.md`):
- Instruction-override defense (delimit and label untrusted content; system prompts are non-overridable).
- Data-exfiltration defense (never echo secrets/other users' data; retrieved content is treated as data, not instructions).
- Tool-abuse defense (allow-listed tools per role; validated parameters).

**Input validation:** validate user requests, tool parameters, and **retrieved content** (Pydantic models everywhere; reject/escape untrusted control text).

**Guardrails:**
- Unsafe tool execution → blocked/approval-gated.
- Unauthorized access → denied at the tool boundary.
- **Hallucinated citations → Validator node rejects answers whose citations don't resolve to retrieved chunks.**
- Invalid responses → schema-validated before returning.
- Brand safety → responses appropriate for Commercial Bank.

**RBAC (enforced at the tool/retrieval layer, not just UI):**

| Role | Chat | Search | Analytics tools | MCP tools | Admin tools |
|---|---|---|---|---|---|
| **Viewer** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Analyst** | ✅ | ✅ | ✅ | ✅ | ❌ |
| **Administrator** | ✅ | ✅ | ✅ | ✅ | ✅ |

Every tool invocation checks the authenticated principal's role via a FastAPI dependency. The LangGraph tool nodes receive only the tools the role permits — the agent cannot invoke what it isn't given.

**Auth:** Option A (hardcoded users/roles) is acceptable for POC; Keycloak if time allows. Issue a token; resolve role from it on every request.

**Rate limiting:** Token Bucket, **per-user**, configurable thresholds, graceful `429` with clear error (never a stack trace).

---

## 9. Scalability Requirements (non-negotiable design constraints)

Even in the POC, do not make choices that prevent scaling. Specifically:

- **Stateless FastAPI workers.** No in-process session state that can't move to a shared store. Session/graph state lives in an external checkpointer (Redis/Postgres), so the app scales horizontally behind a load balancer.
- **Async, non-blocking I/O.** All LLM, Pinecone, MCP, and tool calls are `async`; use connection pooling and bounded concurrency (semaphores) so one heavy RLM query can't exhaust the worker.
- **Rate limiter and memory backed by a shared store** (Redis) rather than per-process dicts, so limits hold across replicas.
- **Provider abstraction** for the LLM and embeddings so models can be swapped/upgraded without touching agents.
- **Ingestion is decoupled** from serving (`data/ingest.py` runs offline/batched) and idempotent (stable chunk ids).
- **Bounded RLM recursion** — max depth/batch caps to prevent runaway fan-out; batches processed with controlled parallelism.
- **12-factor config** — everything via env vars (`.env.example`), no hardcoded endpoints/keys.
- **Containerized** via Docker Compose (app, MCP server, Redis; Pinecone/LangSmith are managed). Each service independently scalable.
- **Structured JSON logging** with a correlation/trace id per request, propagated through every node and tool call.

If a scalability shortcut is unavoidable for time, mark it `# SCALE-DEBT:` in code and log it in `docs/ASSUMPTIONS.md`.

---

## 10. Observability

- **LangSmith is mandatory.** Trace every conversation, tool call, agent transition, and retrieval operation. Traces must be inspectable and shown in the demo.
- Propagate a single trace/correlation id from the API request through LangGraph nodes, tools, and logs.
- The **Agent Activity Panel** in Streamlit mirrors the trace in real time: current state, active node, tool calls, retrieval status, memory updates, validation results, final generation.

---

## 11. Error Handling

Degrade gracefully — never crash the UI, never leak internals:
- **LLM failure** → retry w/ backoff, then fallback model or a safe apology.
- **Vector DB failure** → surface "retrieval unavailable," don't fabricate.
- **MCP failure / tool timeout** → catch, report to Activity Panel, continue if possible.
- **Invalid request** → 4xx with a structured, user-safe message.
- All exceptions logged with trace id; the user never sees a raw stack trace.

---

## 12. Working Agreements for Contributors

- **Async by default.** No `requests`/blocking calls in request paths — use `httpx.AsyncClient`, async Pinecone/LLM clients.
- **Type everything.** Pydantic models for all boundaries (API, tools, state).
- **One responsibility per node/module.** Keep agent logic readable — the evaluator is grading explainability.
- **No secrets in code or git.** Use `.env`; keep `.env.example` current.
- **Every non-obvious decision → an ADR** in `docs/ADR/` and, if it's an assumption, `docs/ASSUMPTIONS.md`.
- **Commit history matters** — small, meaningful commits with clear messages (the repo history is evaluated). Repo must be **public**.
- **Tests** for retrieval fusion, RBAC enforcement, guardrail/validator logic, and rate limiting at minimum.
- Before adding a dependency or changing the stack in §2, confirm it against §9 (scalability) and record an ADR.

---

## 13. Definition of Done (evaluation-aligned)

A change/feature is done when it is: async, traced in LangSmith, observable in the Activity Panel, role-checked (if it touches tools/retrieval), guardrail-covered, error-handled, and documented. Weighting to prioritize under time pressure:

| Area | Weight |
|---|---|
| Agent Architecture | 20% |
| RAG Design | 15% |
| LangGraph Usage | 15% |
| RLM Implementation | 10% |
| Security & Guardrails | 10% |
| Observability | 10% |
| Async Engineering | 5% |
| RBAC | 5% |
| Code Quality | 5% |
| Documentation | 5% |

**Bonus (pursue after core is solid):** multi-agent state/failure handling ("butterfly effect"), human-in-the-loop approval node, reranking layer, long-term memory, answer-quality feedback loop, Docker Compose deployment.

**Deliverables:** public repo · architecture diagram · 45-min demo video (public URL) · LangSmith traces shown in demo · assumptions & trade-offs covered in demo.

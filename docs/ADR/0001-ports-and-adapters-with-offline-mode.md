# ADR-0001: Ports & adapters for every external dependency, with a first-class offline mode

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

The platform depends on four managed services: an LLM provider, Pinecone, LangSmith and
(optionally) Redis. We need:

1. Deterministic, fast tests for retrieval fusion, RBAC, guardrails and rate limiting (§12)
   that run in CI without secrets.
2. Graceful degradation (§11): an LLM or vector-DB outage must not crash the product.
3. The ability to swap providers without touching agent code (§9).

## Decision

Each external capability is a small `Protocol` with at least two adapters:

| Port | Production adapter | Offline / test adapter |
|---|---|---|
| `VectorStore` | `PineconeStore` (dense + sparse indexes, namespaces) | `InMemoryVectorStore` (same filter semantics) |
| `Embedder` | Pinecone Inference / OpenAI | `HashingEmbedder` (deterministic feature hashing) |
| `Reranker` | Pinecone hosted cross-encoder (`bge-reranker-v2-m3`) | `LexicalReranker` |
| `LLMClient` | LangChain chat model (Anthropic/OpenAI/Gemini) with retry + fallback | none → nodes use deterministic fallbacks |
| Checkpointer / Store | Redis | LangGraph in-memory |
| Rate limiter | Redis Lua token bucket | in-process token bucket |

Agent nodes never import a vendor SDK. They receive adapters through LangGraph's runtime
context (`AgentContext`), built once per worker in the FastAPI lifespan.

Every LLM-backed node has a **deterministic fallback path** (keyword router, date-window
research planner, extractive answer). With `LLM_PROVIDER=none` the full graph runs end to
end, so the offline mode doubles as the degraded mode during a provider outage.

## Consequences

- Tests exercise the real graph, fusion, filters and validator, with no mocks of our own code.
- The offline mode gives *lower quality* answers (extractive, keyword routing), and the UI says
  so explicitly. It is a resilience and dev-loop feature, not a substitute for the LLM.
- The Pinecone filter dialect is re-implemented for the in-memory store. The shared
  filter builder is unit-tested against both, so the access-level boundary behaves the same.

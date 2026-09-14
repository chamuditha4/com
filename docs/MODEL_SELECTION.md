# Model Selection

All models are configuration (`LLM_*`, `EMBEDDING_*`, `RERANKER*`). Agents depend on the
`LLMClient` / `Embedder` / `Reranker` ports, so changing a model never touches agent code.

## LLMs: two tiers

| Tier | Default | Used by | Why this tier |
|---|---|---|---|
| **Reasoning** | Anthropic `claude-sonnet-5` | RLM plan generation, final answer synthesis | Planning over catalog metadata and long-context synthesis with strict citation discipline are the quality-critical steps |
| **Fast** | Anthropic `claude-haiku-4-5-20251001` | Supervisor routing, per-slice incident extraction, tool planning, memory extraction | High-volume, schema-bound tasks. The RLM fans out into many extraction calls, so latency and cost per call dominate |

**Why Claude as the default.** Reliable structured output and tool calling (the supervisor,
extraction and tool planner depend on it), strong instruction-following for citation rules and
refusal of embedded instructions, and long context for synthesis over many passages.

**Why not one model everywhere?** Routing and extraction are the most frequent calls. A
reasoning-tier model there multiplies cost and latency, especially inside the RLM fan-out, without
a measurable quality gain on schema-bound tasks.

**Upgrade path.** `LLM_MODEL=claude-opus-5` for the reasoning tier when answer quality matters
more than latency, e.g. executive research summaries.

**Alternatives supported.** `LLM_PROVIDER=openai` or `gemini` with the corresponding models,
through the same LangChain adapters. `LLM_FALLBACK_PROVIDER` / `LLM_FALLBACK_MODEL` add a
cross-provider fallback via `with_fallbacks`.

**Settings.** Temperature 0 (repeatable routing, extraction and plans), 60s timeout, 2 retries with
jittered exponential backoff in the gateway (SDK-level retries disabled to avoid double retrying),
and at most 8 concurrent calls per worker.

## Embeddings

| Default | Why |
|---|---|
| `multilingual-e5-large` via Pinecone Inference (1024-d, `passage` / `query` input types) | Hosted next to the index (one vendor, one network hop, no GPU to operate). Asymmetric passage/query encoding suits question-to-document retrieval. Multilingual coverage matters for a Sri Lankan bank (English, Sinhala, Tamil) |

Alternative: `EMBEDDING_PROVIDER=openai` (`text-embedding-3-*` with configurable dimensions).
Offline: `hash` (deterministic feature hashing; lexical only, for tests and demos).

## Sparse retrieval

Our own BM25 encoder (k1=1.2, b=0.75) producing Pinecone sparse vectors (ADR-0002). BM25 is the
right complement to dense retrieval for bank content full of exact identifiers (`INC-PAY-2026-045`,
`POL-SEC-007`, service names, error codes) that embeddings blur.

## Reranker

| Default (offline) | Production |
|---|---|
| `lexical` heuristic | `RERANKER=pinecone`, `bge-reranker-v2-m3` hosted cross-encoder |

A cross-encoder reads query and passage jointly and fixes fusion's shallow ordering. It runs only on
the fused top 20, so its cost is bounded per query.

## Evaluation plan

Model choices should be validated on the demo question set with LangSmith datasets:
citation precision (validator pass rate), retrieval recall@6 on labeled questions, RLM root-cause
tally accuracy (ground truth: 3/3/2/2 for Analyst, 2/3/2/1 for Viewer), plus p95 latency and
cost per turn per tier.

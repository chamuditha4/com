# Model Selection

All models are configuration (`LLM_*`, `EMBEDDING_*`, `RERANKER*`). Agents depend on the
`LLMClient` / `Embedder` / `Reranker` ports, so changing a model never touches agent code.

## Configuration in use (verified 2026-09-14)

| Role | Provider · model | Used by |
|---|---|---|
| **Reasoning tier** | OpenAI · `gpt-5.5` | RLM Python plan generation, final answer synthesis |
| **Fast tier** | OpenAI · `gpt-5.4-mini` | Supervisor routing, per-slice incident extraction, tool planning, memory extraction |
| **Fallback (both tiers)** | Google · `gemini-3.5-flash` | Takes over any call that fails after retries (`with_fallbacks`) |

### How these were chosen

1. **Availability.** The models were listed from each provider's models API with the project keys,
   rather than assumed. Preview models were excluded, and so were model names we could not
   characterise.
2. **Capability probe.** Every candidate had to pass the three call shapes the agents depend on:
   free-text generation, structured output with the real `RouteDecision` schema (enums, dates,
   optional lists), and tool calling. `gpt-5.5`, `gpt-5.4-mini`, `gemini-3.5-flash` and
   `gemini-2.5-flash` all passed.
3. **Tiering.** The RLM fans out into many extraction calls. Putting a mini model there keeps
   latency and cost per research query bounded. The strongest model is reserved for the two
   quality-critical steps: planning and synthesis.
4. **Cross-vendor fallback.** A fallback on the same vendor does not survive a vendor outage or a
   revoked key. Gemini flash is fast enough to serve both tiers during an OpenAI incident.

### Live verification (`RUN_LIVE_TESTS=1 uv run pytest backend/tests/live`)

12/12 passing against the real providers:
- the configured primary and fallback providers;
- Gemini takes over when the primary model fails, both at the gateway and end to end through the graph;
- the RLM runs from an LLM-written plan: 10/10 correct incidents, 3/3/2/2 root-cause tally, nothing outside the window;
- Viewer clearance holds under LLM routing;
- grounded retrieval with a validated citation to the TLS runbook;
- LLM tool calling reaches the MCP server;
- no tools are offered to Viewers;
- the admin approval interrupt fires;
- direct and indirect prompt injection are handled;
- LLM memory extraction and recall work;
- LangSmith root run id equals the API trace id.

Observed end-to-end latency for the canonical RLM query over HTTP was about 35–40 s (4 planned
batches, 3 recursive splits, 7 leaf extractions, synthesis).

## Alternatives supported

| Provider | Configure | Notes |
|---|---|---|
| Anthropic | `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-sonnet-5`, `LLM_FAST_MODEL=claude-haiku-4-5-20251001` | The code defaults. Not exercised live here (no key). |
| Gemini as primary | `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-2.5-pro` or a newer pro model, `LLM_FAST_MODEL=gemini-3.5-flash` | Probe-verified for the flash models |
| None | `LLM_PROVIDER=none` | Deterministic offline mode; answers are marked `degraded` |

**Settings.** Temperature 0 (repeatable routing, extraction and plans), 60 s timeout, 2 retries with
jittered exponential backoff in the gateway (SDK-level retries disabled to avoid multiplying
retries), and at most 8 concurrent LLM calls per worker.

## Embeddings

| Default | Why |
|---|---|
| `multilingual-e5-large` via Pinecone Inference (1024-d, `passage` / `query` input types) | Hosted next to the index (one vendor, one network hop, no GPU to operate). Asymmetric passage/query encoding suits question-to-document retrieval. Multilingual coverage matters for a Sri Lankan bank (English, Sinhala, Tamil) |

**Live-verified (2026-09-14):**
- `data/ingest.py` created both indexes and embedded and upserted 166 chunks in about 12 s.
- The bge reranker ranks the TLS runbook first, and both retrieval legs contribute.
- Warm hybrid search takes about 1.5 s from Sri Lanka to `us-east-1`. Nearly all of that is network round trips; see ASSUMPTIONS D7.

Alternative: `EMBEDDING_PROVIDER=openai` (`text-embedding-3-*` with configurable dimensions).

## Sparse retrieval

Our own BM25 encoder (k1=1.2, b=0.75) producing Pinecone sparse vectors (ADR-0002). BM25
complements dense retrieval for bank content full of exact identifiers (`INC-PAY-2026-045`,
`POL-SEC-007`, service names, error codes) that embeddings blur.

## Reranker

| Offline | Production |
|---|---|
| `lexical` heuristic | `RERANKER=pinecone`, `bge-reranker-v2-m3` hosted cross-encoder |

A cross-encoder reads query and passage jointly, fixing fusion's shallow ordering. It runs only on
the fused top 20, so cost stays bounded per query.

## Evaluation plan

Extend the live suite into LangSmith datasets:
- citation precision (validator pass rate before retries);
- retrieval recall@6 on labelled questions;
- RLM tally accuracy per role (ground truth 3/3/2/2 for Analyst, 2/3/2/1 for Viewer);
- p95 latency and cost per turn, per tier and per provider (primary vs fallback).

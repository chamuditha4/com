# ADR-0002: Hybrid retrieval with client-side fusion across two Pinecone indexes

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

CLAUDE.md §6 requires dense and BM25 retrieval, configurable fusion, reranking, Pinecone
namespaces and metadata filtering, attribution on every result, and access level as a hard
filter. The evaluator grades *explainability*, so we need to show why each chunk was selected.

## Options considered

1. **Single sparse-dense Pinecone index** (dotproduct, alpha-weighted query vectors). This is one
   network call per namespace, but Pinecone returns one blended score. Per-leg ranks are lost,
   so we can't explain or tune the contribution of each leg, and the weighting must be linear.
2. **Two indexes (dense cosine + sparse dotproduct), fused client-side.** This takes two calls
   per namespace, run concurrently. Per-leg ranks are kept, RRF is available, and if one leg
   fails the other still answers.

## Decision

Option 2.

- **Dense leg:** `multilingual-e5-large` via Pinecone Inference (`passage`/`query` input types).
- **Sparse leg:** our own BM25 encoder. Document vectors hold the saturated TF factor and query
  vectors hold IDF, so Pinecone's dot product equals the BM25 score. Corpus statistics are
  fitted at ingest and shipped as an artifact. We implement BM25 ourselves (about 100 lines,
  unit-tested) rather than use `pinecone-text`, which pulls in NLTK and hides the maths.
- **Namespaces = departments.** The retriever fans out over namespaces in scope with a
  semaphore, so "payments incidents" never scans HR.
- **Fusion:** RRF by default (rank-based, immune to cosine-vs-BM25 scale mismatch); weighted
  min-max fusion available through `RETRIEVAL_FUSION=weighted`. Leg weights are configurable.
- **Rerank:** the fused top 20 goes to a hosted cross-encoder (`bge-reranker-v2-m3`), which
  returns the top 6. The reranker is also a port, with a lexical fallback.
- **Security:** `build_metadata_filter()` always emits the principal's `access_level ∈ clearance`
  clause and has no parameter that can widen it. Results are re-checked after retrieval
  (defense in depth) and scanned for injection signals.

## Consequences

- Twice the query calls per namespace compared with a single index. Acceptable: queries are
  concurrent, and serverless query cost is small next to LLM cost.
- A full re-ingest is needed when corpus statistics drift significantly (BM25 IDF). Ingest is
  idempotent with stable chunk ids, so this is a batch job, not a migration.

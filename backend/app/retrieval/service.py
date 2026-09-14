"""Hybrid retrieval pipeline.

    query ─┬─ embed ──► dense query  (per namespace, concurrently) ─┐
           └─ BM25  ──► sparse query (per namespace, concurrently) ─┴─► fuse ─► rerank ─► access re-check

Security: the metadata filter is built from the `Principal`'s clearance, never from the
query or the LLM. After retrieval every chunk is re-checked against the clearance (defense in
depth against a misconfigured index or filter bug), and violations are dropped and logged.

Resilience: if one leg fails, the other still answers and the diagnostics record the
degradation. If both fail, `RetrievalUnavailableError` is raised so the agent can say
"retrieval unavailable" instead of guessing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langsmith import traceable

from app.auth.models import Principal
from app.core.exceptions import RetrievalUnavailableError
from app.core.logging import get_logger
from app.guardrails.injection import assess_injection
from app.retrieval.catalog import DocumentCatalog
from app.retrieval.embeddings import Embedder
from app.retrieval.filters import build_metadata_filter
from app.retrieval.fusion import FusedCandidate, reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.models import (
    Chunk,
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedChunk,
    SearchFilters,
    StoreMatch,
)
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import BM25Encoder
from app.retrieval.stores.base import VectorStore

logger = get_logger(__name__)

MAX_QUERY_CHARS = 1000


@dataclass(frozen=True)
class RetrievalConfig:
    candidates_per_leg: int = 20
    fusion: str = "rrf"
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    top_n: int = 6
    namespace_concurrency: int = 6


def _trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    principal = inputs.get("principal")
    return {
        "query": inputs.get("query"),
        "filters": inputs.get("filters"),
        "top_n": inputs.get("top_n"),
        "principal": {"user_id": principal.user_id, "role": principal.role} if principal else None,
    }


def _trace_outputs(result: Any) -> dict[str, Any]:
    # LangSmith renders `documents` specially for retriever runs.
    if not isinstance(result, RetrievalResult):
        return {"output": result}
    return {
        "documents": [
            {
                "page_content": c.chunk.text,
                "type": "Document",
                "metadata": {
                    "attribution": c.attribution,
                    "score": c.score,
                    "dense_rank": c.dense_rank,
                    "sparse_rank": c.sparse_rank,
                    "access_level": c.chunk.metadata.access_level,
                },
            }
            for c in result.chunks
        ],
        "diagnostics": result.diagnostics.model_dump(),
    }


class HybridRetriever:
    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        sparse_encoder: BM25Encoder,
        reranker: Reranker,
        catalog: DocumentCatalog,
        config: RetrievalConfig,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker
        self.catalog = catalog
        self.config = config
        self._namespace_semaphore = asyncio.Semaphore(config.namespace_concurrency)

    async def _namespaces_in_scope(self, filters: SearchFilters | None) -> list[str]:
        known = self.catalog.departments or sorted(await self.store.namespace_counts())
        if filters and filters.departments:
            return [ns for ns in known if ns in filters.departments]
        return known

    async def _fan_out(
        self,
        namespaces: list[str],
        query_one: Callable[[str], Awaitable[list[StoreMatch]]],
    ) -> list[StoreMatch]:
        async def bounded(namespace: str) -> list[StoreMatch]:
            async with self._namespace_semaphore:
                return await query_one(namespace)

        per_namespace = await asyncio.gather(*(bounded(ns) for ns in namespaces))
        merged = [m for matches in per_namespace for m in matches]
        return sorted(merged, key=lambda m: (-m.score, m.id))[: self.config.candidates_per_leg]

    async def _dense_leg(
        self, query: str, namespaces: list[str], metadata_filter: dict[str, Any]
    ) -> list[StoreMatch]:
        vector = await self.embedder.embed_query(query)
        return await self._fan_out(
            namespaces,
            lambda ns: self.store.dense_query(
                vector=vector,
                top_k=self.config.candidates_per_leg,
                metadata_filter=metadata_filter,
                namespace=ns,
            ),
        )

    async def _sparse_leg(
        self, query: str, namespaces: list[str], metadata_filter: dict[str, Any]
    ) -> list[StoreMatch]:
        vector = self.sparse_encoder.encode_query(query)
        return await self._fan_out(
            namespaces,
            lambda ns: self.store.sparse_query(
                vector=vector,
                top_k=self.config.candidates_per_leg,
                metadata_filter=metadata_filter,
                namespace=ns,
            ),
        )

    def _fuse(self, dense: list[StoreMatch], sparse: list[StoreMatch]) -> list[FusedCandidate]:
        if self.config.fusion == "weighted":
            return weighted_score_fusion(
                dense,
                sparse,
                dense_weight=self.config.dense_weight,
                sparse_weight=self.config.sparse_weight,
            )
        return reciprocal_rank_fusion(
            dense,
            sparse,
            k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
        )

    @traceable(
        run_type="retriever",
        name="hybrid_search",
        process_inputs=_trace_inputs,
        process_outputs=_trace_outputs,
    )
    async def search(
        self,
        query: str,
        *,
        principal: Principal,
        filters: SearchFilters | None = None,
        top_n: int | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        query = query.strip()[:MAX_QUERY_CHARS]
        top_n = top_n or self.config.top_n
        metadata_filter = build_metadata_filter(principal.allowed_access_levels, filters)
        namespaces = await self._namespaces_in_scope(filters)
        diagnostics = RetrievalDiagnostics(
            query=query,
            metadata_filter=metadata_filter,
            namespaces=namespaces,
            fusion=self.config.fusion,
            reranker=self.reranker.name,
        )
        if not query or not namespaces:
            return RetrievalResult(chunks=[], diagnostics=diagnostics)

        dense_result, sparse_result = await asyncio.gather(
            self._dense_leg(query, namespaces, metadata_filter),
            self._sparse_leg(query, namespaces, metadata_filter),
            return_exceptions=True,
        )
        dense: list[StoreMatch] = []
        sparse: list[StoreMatch] = []
        for leg, outcome in (("dense", dense_result), ("sparse", sparse_result)):
            if isinstance(outcome, BaseException):
                logger.error("retrieval leg failed", extra={"leg": leg}, exc_info=outcome)
                diagnostics.degraded_legs.append(leg)
            elif leg == "dense":
                dense = outcome
            else:
                sparse = outcome
        if len(diagnostics.degraded_legs) == 2:
            raise RetrievalUnavailableError()

        fused = self._fuse(dense, sparse)
        candidates = [
            RetrievedChunk(
                chunk=Chunk.from_index_metadata(c.metadata),
                score=c.fused_score,
                fused_score=c.fused_score,
                dense_rank=c.dense_rank,
                dense_score=c.dense_score,
                sparse_rank=c.sparse_rank,
                sparse_score=c.sparse_score,
            )
            for c in fused[: self.config.candidates_per_leg]
        ]

        try:
            ranked = await self.reranker.rerank(query, candidates, top_n)
        except Exception:
            logger.warning("reranker failed; keeping fused order", exc_info=True)
            diagnostics.degraded_legs.append("reranker")
            ranked = candidates[:top_n]

        permitted: list[RetrievedChunk] = []
        for item in ranked:
            if not principal.can_read(item.chunk.metadata.access_level):
                diagnostics.dropped_by_access_check += 1
                logger.error(
                    "access re-check dropped a chunk that passed the store filter",
                    extra={"chunk_id": item.chunk.chunk_id, "role": principal.role.value},
                )
                continue
            assessment = assess_injection(item.chunk.text)
            if assessment.flagged:
                item.injection_signals = assessment.signals
                diagnostics.flagged_injection += 1
            permitted.append(item)

        diagnostics.dense_hits = len(dense)
        diagnostics.sparse_hits = len(sparse)
        diagnostics.fused_candidates = len(fused)
        diagnostics.returned = len(permitted)
        diagnostics.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return RetrievalResult(chunks=permitted, diagnostics=diagnostics)

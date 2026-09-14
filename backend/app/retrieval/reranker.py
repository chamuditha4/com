"""Second-stage reranking of fused candidates before they reach the LLM.

Fusion is cheap but shallow; a cross-encoder reads query and passage *together* and is far
more precise, but too expensive to run over the whole index. So we rerank only the fused
top candidates (default 20 → 6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from app.core.logging import get_logger
from app.retrieval.models import RetrievedChunk
from app.retrieval.sparse import tokenize

if TYPE_CHECKING:
    from pinecone import PineconeAsyncio

logger = get_logger(__name__)


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class NoopReranker:
    name = "none"

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_n]


class LexicalReranker:
    """Offline heuristic reranker: query-term coverage of title/section/body blended with the
    fused score. It is not a cross-encoder; it stands in for one when no reranking model is
    available."""

    name = "lexical"

    def __init__(self, coverage_weight: float = 0.6) -> None:
        self._coverage_weight = coverage_weight

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        query_terms = set(tokenize(query))
        if not candidates or not query_terms:
            return candidates[:top_n]
        max_fused = max(c.fused_score for c in candidates) or 1.0
        for c in candidates:
            md = c.chunk.metadata
            body = set(tokenize(c.chunk.text))
            heading = set(tokenize(f"{md.title} {c.chunk.section}"))
            coverage = len(query_terms & (body | heading)) / len(query_terms)
            heading_bonus = 0.15 * len(query_terms & heading) / len(query_terms)
            c.rerank_score = round(
                self._coverage_weight * min(1.0, coverage + heading_bonus)
                + (1 - self._coverage_weight) * c.fused_score / max_fused,
                4,
            )
            c.score = c.rerank_score
        return sorted(candidates, key=lambda c: (-c.score, c.chunk.chunk_id))[:top_n]


class PineconeReranker:
    """Hosted cross-encoder (e.g. `bge-reranker-v2-m3`) via Pinecone Inference."""

    name = "pinecone"

    def __init__(self, client: PineconeAsyncio, model: str) -> None:
        self._client = client
        self._model = model

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        documents = [
            {"id": c.chunk.chunk_id, "text": f"{c.chunk.metadata.title}\n{c.chunk.section}\n{c.chunk.text}"}
            for c in candidates
        ]
        result = await self._client.inference.rerank(
            model=self._model,
            query=query,
            documents=documents,
            rank_fields=["text"],
            top_n=top_n,
            return_documents=False,
            parameters={"truncate": "END"},
        )
        reranked: list[RetrievedChunk] = []
        for ranked in result.data:
            chunk = candidates[ranked.index]
            chunk.rerank_score = chunk.score = float(ranked.score)
            reranked.append(chunk)
        return reranked

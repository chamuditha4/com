"""In-process vector store with Pinecone-compatible semantics (tests + offline demo)."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from app.retrieval.filters import matches_filter
from app.retrieval.models import SparseVector, StoreMatch
from app.retrieval.stores.base import IndexRecord


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _sparse_dot(query: SparseVector, doc: SparseVector) -> float:
    doc_weights = dict(zip(doc.indices, doc.values, strict=True))
    return sum(w * doc_weights.get(i, 0.0) for i, w in zip(query.indices, query.values, strict=True))


class InMemoryVectorStore:
    # SCALE-DEBT: brute-force scan held in worker memory. Only for tests and offline demos;
    # production uses PineconeStore.
    name = "memory"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, IndexRecord]] = defaultdict(dict)

    async def upsert(self, records: Sequence[IndexRecord]) -> int:
        for record in records:
            self._records[record.namespace][record.chunk.chunk_id] = record
        return len(records)

    def _rank(
        self,
        namespace: str,
        metadata_filter: dict[str, Any],
        top_k: int,
        score_fn: Any,
    ) -> list[StoreMatch]:
        matches = []
        for record in self._records.get(namespace, {}).values():
            metadata = record.chunk.to_index_metadata()
            if not matches_filter(metadata, metadata_filter):
                continue
            score = score_fn(record)
            if score > 0:
                matches.append(StoreMatch(id=record.chunk.chunk_id, score=score, metadata=metadata))
        return sorted(matches, key=lambda m: (-m.score, m.id))[:top_k]

    async def dense_query(
        self, *, vector: list[float], top_k: int, metadata_filter: dict[str, Any], namespace: str
    ) -> list[StoreMatch]:
        return self._rank(namespace, metadata_filter, top_k, lambda r: _cosine(vector, r.dense))

    async def sparse_query(
        self, *, vector: SparseVector, top_k: int, metadata_filter: dict[str, Any], namespace: str
    ) -> list[StoreMatch]:
        return self._rank(namespace, metadata_filter, top_k, lambda r: _sparse_dot(vector, r.sparse))

    async def namespace_counts(self) -> dict[str, int]:
        return {ns: len(records) for ns, records in self._records.items()}

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

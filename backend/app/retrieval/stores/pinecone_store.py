"""Pinecone adapter: one dense index (cosine) + one sparse index (dotproduct, BM25 weights).

Why two indexes instead of a single sparse-dense index? Pinecone's single-index hybrid mode
blends both signals *inside* the query (alpha-weighted vectors) and returns one score, which
hides each leg's contribution. Two indexes let us fuse client-side with RRF, keep per-leg
ranks for explainability, and degrade to one leg if the other is unavailable. See ADR-0002.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pinecone import PineconeAsyncio, ServerlessSpec

from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.models import SparseVector, StoreMatch
from app.retrieval.stores.base import IndexRecord

logger = get_logger(__name__)

_UPSERT_BATCH = 100


class PineconeStore:
    name = "pinecone"

    def __init__(self, client: PineconeAsyncio, dense_index: Any, sparse_index: Any) -> None:
        self._client = client
        self._dense = dense_index
        self._sparse = sparse_index

    @classmethod
    async def connect(
        cls, settings: Settings, client: PineconeAsyncio, *, create_missing: bool = False
    ) -> PineconeStore:
        spec = ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region)
        if create_missing:
            if not await client.has_index(settings.pinecone_dense_index):
                logger.info("creating dense index", extra={"index": settings.pinecone_dense_index})
                await client.create_index(
                    name=settings.pinecone_dense_index,
                    dimension=settings.embedding_dimension,
                    metric="cosine",
                    spec=spec,
                )
            if not await client.has_index(settings.pinecone_sparse_index):
                logger.info("creating sparse index", extra={"index": settings.pinecone_sparse_index})
                await client.create_index(
                    name=settings.pinecone_sparse_index,
                    metric="dotproduct",
                    vector_type="sparse",
                    spec=spec,
                )
        dense_desc = await client.describe_index(settings.pinecone_dense_index)
        sparse_desc = await client.describe_index(settings.pinecone_sparse_index)
        return cls(
            client,
            client.IndexAsyncio(host=dense_desc.host),
            client.IndexAsyncio(host=sparse_desc.host),
        )

    async def upsert(self, records: Sequence[IndexRecord]) -> int:
        by_namespace: dict[str, list[IndexRecord]] = {}
        for record in records:
            by_namespace.setdefault(record.namespace, []).append(record)

        for namespace, group in by_namespace.items():
            dense = [
                {"id": r.chunk.chunk_id, "values": r.dense, "metadata": r.chunk.to_index_metadata()} for r in group
            ]
            sparse = [
                {
                    "id": r.chunk.chunk_id,
                    "sparse_values": {"indices": r.sparse.indices, "values": r.sparse.values},
                    "metadata": r.chunk.to_index_metadata(),
                }
                for r in group
                if r.sparse.indices  # Pinecone rejects empty sparse vectors
            ]
            await self._dense.upsert(vectors=dense, namespace=namespace, batch_size=_UPSERT_BATCH, show_progress=False)
            if sparse:
                await self._sparse.upsert(
                    vectors=sparse, namespace=namespace, batch_size=_UPSERT_BATCH, show_progress=False
                )
        return len(records)

    @staticmethod
    def _to_matches(response: Any) -> list[StoreMatch]:
        return [StoreMatch(id=m.id, score=float(m.score), metadata=dict(m.metadata or {})) for m in response.matches]

    async def dense_query(
        self, *, vector: list[float], top_k: int, metadata_filter: dict[str, Any], namespace: str
    ) -> list[StoreMatch]:
        response = await self._dense.query(
            vector=vector,
            top_k=top_k,
            filter=metadata_filter,
            namespace=namespace,
            include_metadata=True,
        )
        return self._to_matches(response)

    async def sparse_query(
        self, *, vector: SparseVector, top_k: int, metadata_filter: dict[str, Any], namespace: str
    ) -> list[StoreMatch]:
        if not vector.indices:
            return []
        response = await self._sparse.query(
            sparse_vector={"indices": vector.indices, "values": vector.values},
            top_k=top_k,
            filter=metadata_filter,
            namespace=namespace,
            include_metadata=True,
        )
        return self._to_matches(response)

    async def namespace_counts(self) -> dict[str, int]:
        stats = await self._dense.describe_index_stats()
        return {name: int(ns.vector_count) for name, ns in stats.namespaces.items()}

    async def ping(self) -> bool:
        try:
            await self._dense.describe_index_stats()
            return True
        except Exception:
            logger.warning("pinecone ping failed", exc_info=True)
            return False

    async def close(self) -> None:
        await self._dense.close()
        await self._sparse.close()
        await self._client.close()

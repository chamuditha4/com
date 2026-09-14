"""Vector store port.

Namespaces partition the index by department (CLAUDE.md §6). Queries run per namespace and
the retriever fans out across the namespaces in scope, so department scoping is a cheap
physical partition rather than only a filter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.retrieval.models import Chunk, SparseVector, StoreMatch


@dataclass(frozen=True)
class IndexRecord:
    chunk: Chunk
    dense: list[float]
    sparse: SparseVector

    @property
    def namespace(self) -> str:
        return self.chunk.metadata.department


class VectorStore(Protocol):
    name: str

    async def upsert(self, records: Sequence[IndexRecord]) -> int: ...

    async def dense_query(
        self,
        *,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any],
        namespace: str,
    ) -> list[StoreMatch]: ...

    async def sparse_query(
        self,
        *,
        vector: SparseVector,
        top_k: int,
        metadata_filter: dict[str, Any],
        namespace: str,
    ) -> list[StoreMatch]: ...

    async def namespace_counts(self) -> dict[str, int]: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...

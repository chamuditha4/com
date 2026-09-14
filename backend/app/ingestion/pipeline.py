"""Ingestion pipeline: load → chunk → fit BM25 → embed → upsert → write artifacts.

Runs offline via `data/ingest.py` (decoupled from serving, CLAUDE.md §9). The in-memory store
reuses the same function at startup in offline mode, so there is exactly one indexing code path.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from app.auth.models import AccessLevel
from app.core.logging import get_logger
from app.ingestion.chunker import chunk_document, load_markdown
from app.retrieval.catalog import CatalogEntry, DocumentCatalog
from app.retrieval.embeddings import Embedder
from app.retrieval.models import Chunk
from app.retrieval.sparse import BM25Encoder
from app.retrieval.stores.base import IndexRecord, VectorStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionReport:
    documents: int
    chunks: int
    namespaces: dict[str, int]
    seconds: float


def load_corpus(docs_dir: Path) -> tuple[list[Chunk], DocumentCatalog]:
    chunks: list[Chunk] = []
    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()
    for path in sorted(docs_dir.rglob("*.md")):
        metadata, body = load_markdown(path)
        if metadata.doc_id in seen_ids:
            raise ValueError(f"duplicate doc_id {metadata.doc_id} in {path}")
        seen_ids.add(metadata.doc_id)
        doc_chunks = chunk_document(metadata, body)
        chunks.extend(doc_chunks)
        entries.append(CatalogEntry(metadata=metadata, chunk_count=len(doc_chunks)))
    return chunks, DocumentCatalog(entries)


async def build_index(
    *,
    docs_dir: Path,
    store: VectorStore,
    embedder: Embedder,
    batch_size: int = 64,
    concurrency: int = 4,
) -> tuple[BM25Encoder, DocumentCatalog, IngestionReport]:
    started = time.perf_counter()
    chunks, catalog = load_corpus(docs_dir)
    if not chunks:
        raise ValueError(f"no documents found under {docs_dir}")

    encoder = BM25Encoder.fit(c.text for c in chunks)
    semaphore = asyncio.Semaphore(concurrency)

    async def index_batch(batch: list[Chunk]) -> int:
        async with semaphore:
            vectors = await embedder.embed_documents([c.text for c in batch])
            records = [
                IndexRecord(chunk=c, dense=v, sparse=encoder.encode_document(c.text))
                for c, v in zip(batch, vectors, strict=True)
            ]
            return await store.upsert(records)

    batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    await asyncio.gather(*(index_batch(b) for b in batches))

    namespaces: dict[str, int] = {}
    for c in chunks:
        namespaces[c.metadata.department] = namespaces.get(c.metadata.department, 0) + 1
    report = IngestionReport(
        documents=len(catalog.visible(tuple(AccessLevel))),
        chunks=len(chunks),
        namespaces=dict(sorted(namespaces.items())),
        seconds=round(time.perf_counter() - started, 2),
    )
    logger.info("index built", extra={"report": report.__dict__, "store": store.name})
    return encoder, catalog, report


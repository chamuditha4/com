"""Offline ingestion job: chunk → embed → upsert to Pinecone → write BM25 + catalog artifacts.

Decoupled from serving (CLAUDE.md §9) and idempotent: chunk ids are stable, so re-running
overwrites vectors in place. API workers load `data/artifacts/*.json` at startup.

Usage:
    python data/ingest.py                 # uses VECTOR_STORE / EMBEDDING_PROVIDER from .env
    python data/ingest.py --dry-run       # offline: in-memory store + hashing embedder, writes artifacts
    python data/ingest.py --docs path/to/markdown
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.container import build_embedder
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.ingestion.pipeline import build_index
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.stores.memory import InMemoryVectorStore


async def run(docs_dir: Path, dry_run: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    async with AsyncExitStack() as stack:
        if dry_run or settings.vector_store != "pinecone":
            store, embedder = InMemoryVectorStore(), HashingEmbedder()
            print("dry run: indexing into an in-memory store with the hashing embedder")
        else:
            from pinecone import PineconeAsyncio

            from app.retrieval.stores.pinecone_store import PineconeStore

            if not settings.pinecone_api_key:
                raise SystemExit("PINECONE_API_KEY is required (or pass --dry-run)")
            client = PineconeAsyncio(api_key=settings.pinecone_api_key.get_secret_value())
            stack.push_async_callback(client.close)
            store = await PineconeStore.connect(settings, client, create_missing=True)
            stack.push_async_callback(store.close)
            embedder = build_embedder(settings, client)

        encoder, catalog, report = await build_index(docs_dir=docs_dir, store=store, embedder=embedder)
        encoder.save(settings.bm25_params_path)
        catalog.save(settings.catalog_path)

    print(f"indexed {report.documents} documents / {report.chunks} chunks in {report.seconds}s")
    print(f"namespaces: {report.namespaces}")
    print(f"artifacts:  {settings.bm25_params_path}, {settings.catalog_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, default=get_settings().mock_docs_dir)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.docs, args.dry_run))


if __name__ == "__main__":
    main()

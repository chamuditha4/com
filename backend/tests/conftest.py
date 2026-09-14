"""Shared fixtures. Tests run fully offline: in-memory store, hashing embedder, no LLM."""

from __future__ import annotations

import os

# Fail tests if any checkpointed type is not registered with the serializer (see memory/persistence.py).
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

import asyncio
from dataclasses import dataclass

import pytest

from app.auth.models import Principal, Role
from app.core.config import REPO_ROOT, Settings
from app.ingestion.pipeline import build_index
from app.retrieval.catalog import DocumentCatalog
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.reranker import LexicalReranker
from app.retrieval.service import HybridRetriever, RetrievalConfig
from app.retrieval.sparse import BM25Encoder
from app.retrieval.stores.memory import InMemoryVectorStore


@dataclass
class IndexedCorpus:
    store: InMemoryVectorStore
    embedder: HashingEmbedder
    encoder: BM25Encoder
    catalog: DocumentCatalog


@pytest.fixture(scope="session")
def corpus() -> IndexedCorpus:
    store, embedder = InMemoryVectorStore(), HashingEmbedder()
    encoder, catalog, _ = asyncio.run(build_index(docs_dir=REPO_ROOT / "data" / "mock", store=store, embedder=embedder))
    return IndexedCorpus(store, embedder, encoder, catalog)


@pytest.fixture
def retriever(corpus: IndexedCorpus) -> HybridRetriever:
    return HybridRetriever(
        store=corpus.store,
        embedder=corpus.embedder,
        sparse_encoder=corpus.encoder,
        reranker=LexicalReranker(),
        catalog=corpus.catalog,
        config=RetrievalConfig(top_n=6),
    )


def make_principal(role: Role) -> Principal:
    return Principal(user_id=f"u-{role.value}", username=role.value, display_name=role.value, role=role)


@pytest.fixture
def viewer() -> Principal:
    return make_principal(Role.VIEWER)


@pytest.fixture
def analyst() -> Principal:
    return make_principal(Role.ANALYST)


@pytest.fixture
def admin() -> Principal:
    return make_principal(Role.ADMINISTRATOR)


@pytest.fixture
def test_settings() -> Settings:
    # _env_file=None keeps the offline suite hermetic: a developer's .env (real keys, tracing) must not leak in.
    return Settings(
        _env_file=None,
        app_env="test",
        llm_provider="none",
        vector_store="memory",
        embedding_provider="hash",
        checkpointer="memory",
        redis_url=None,
        jwt_secret="test-secret-" + "x" * 32,
        rate_limit_capacity=100,
    )

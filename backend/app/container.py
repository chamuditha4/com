"""Composition root: builds every worker-lifetime dependency from settings.

This is the only place that decides which adapter backs each port (ADR-0001). The API lifespan
calls `build_container` once per worker; tests call it with offline settings and fakes.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.agents.context import AgentServices
from app.agents.graph import build_agent_graph
from app.auth.users import UserDirectory
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.rate_limit import InMemoryTokenBucket, RateLimiter, RedisTokenBucket
from app.ingestion.pipeline import build_index
from app.llm.factory import build_llm
from app.llm.gateway import LLMClient
from app.memory.long_term import LongTermMemory
from app.memory.persistence import build_persistence
from app.retrieval.catalog import DocumentCatalog
from app.retrieval.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder, PineconeEmbedder
from app.retrieval.reranker import LexicalReranker, NoopReranker, PineconeReranker, Reranker
from app.retrieval.service import HybridRetriever, RetrievalConfig
from app.retrieval.sparse import BM25Encoder
from app.retrieval.stores.base import VectorStore
from app.retrieval.stores.memory import InMemoryVectorStore
from app.tools.audit import AuditLog, InMemoryAuditLog, RedisAuditLog
from app.tools.builtin import build_builtin_tools
from app.tools.mcp_client import MCPToolProvider
from app.tools.python_analysis import PythonSandbox
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    services: AgentServices
    graph: CompiledStateGraph
    users: UserDirectory
    rate_limiter: RateLimiter
    audit: AuditLog
    redis: Redis | None


def _pinecone_client(settings: Settings, stack: AsyncExitStack) -> Any:
    from pinecone import PineconeAsyncio

    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required for Pinecone embeddings, store or reranker")
    client = PineconeAsyncio(api_key=settings.pinecone_api_key.get_secret_value())
    stack.push_async_callback(client.close)
    return client


def build_embedder(settings: Settings, pinecone: Any) -> Embedder:
    if settings.embedding_provider == "pinecone":
        return PineconeEmbedder(pinecone, settings.embedding_model, settings.embedding_dimension)
    if settings.embedding_provider == "openai":
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIEmbedder(settings.embedding_model, settings.embedding_dimension, key)
    return HashingEmbedder()


def build_reranker(settings: Settings, pinecone: Any) -> Reranker:
    if settings.reranker == "pinecone":
        return PineconeReranker(pinecone, settings.reranker_model)
    if settings.reranker == "lexical":
        return LexicalReranker()
    return NoopReranker()


async def build_retriever(settings: Settings, stack: AsyncExitStack) -> tuple[HybridRetriever, Any]:
    needs_pinecone = "pinecone" in (settings.vector_store, settings.embedding_provider, settings.reranker)
    pinecone = _pinecone_client(settings, stack) if needs_pinecone else None
    embedder = build_embedder(settings, pinecone)

    store: VectorStore
    if settings.vector_store == "pinecone":
        from app.retrieval.stores.pinecone_store import PineconeStore

        store = await PineconeStore.connect(settings, pinecone)
        if not settings.bm25_params_path.exists() or not settings.catalog_path.exists():
            raise FileNotFoundError("Pinecone mode needs ingestion artifacts; run `python data/ingest.py` first")
        encoder = BM25Encoder.load(settings.bm25_params_path)
        catalog = DocumentCatalog.load(settings.catalog_path)
    else:
        # SCALE-DEBT: offline mode indexes the mock corpus in-process at startup.
        store = InMemoryVectorStore()
        encoder, catalog, _ = await build_index(docs_dir=settings.mock_docs_dir, store=store, embedder=embedder)

    retriever = HybridRetriever(
        store=store,
        embedder=embedder,
        sparse_encoder=encoder,
        reranker=build_reranker(settings, pinecone),
        catalog=catalog,
        config=RetrievalConfig(
            candidates_per_leg=settings.retrieval_candidates_per_leg,
            fusion=settings.retrieval_fusion,
            dense_weight=settings.retrieval_dense_weight,
            sparse_weight=settings.retrieval_sparse_weight,
            rrf_k=settings.retrieval_rrf_k,
            top_n=settings.retrieval_top_n,
            namespace_concurrency=settings.retrieval_namespace_concurrency,
        ),
    )
    stack.push_async_callback(store.close)
    return retriever, pinecone


async def build_container(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    llm: LLMClient | None = None,
    mcp_target: Any = None,
) -> Container:
    redis: Redis | None = None
    if settings.redis_url:
        redis = Redis.from_url(settings.redis_url, decode_responses=False)
        stack.push_async_callback(redis.aclose)

    retriever, _ = await build_retriever(settings, stack)
    audit: AuditLog = RedisAuditLog(redis) if redis else InMemoryAuditLog()

    async def reindex() -> dict[str, Any]:
        if settings.vector_store != "memory":
            return {"status": "queued", "detail": "Run the offline ingestion job (data/ingest.py)."}
        store = InMemoryVectorStore()
        encoder, catalog, report = await build_index(
            docs_dir=settings.mock_docs_dir, store=store, embedder=retriever.embedder
        )
        retriever.store, retriever.sparse_encoder, retriever.catalog = store, encoder, catalog
        return {"status": "completed", "documents": report.documents, "chunks": report.chunks}

    tools = ToolRegistry(
        builtin=build_builtin_tools(
            retriever=retriever,
            sandbox=PythonSandbox(
                timeout_seconds=settings.python_tool_timeout_seconds, memory_mb=settings.python_tool_memory_mb
            ),
            audit=audit,
            reindex=reindex,
        ),
        audit=audit,
        mcp=MCPToolProvider(mcp_target or settings.mcp_server_url, timeout_seconds=settings.mcp_timeout_seconds),
    )

    checkpointer, store = await build_persistence(settings, stack)
    services = AgentServices(
        settings=settings,
        retriever=retriever,
        llm=llm or build_llm(settings),
        tools=tools,
        memory=LongTermMemory(store) if settings.long_term_memory_enabled else None,
        research_semaphore=asyncio.Semaphore(settings.rlm_batch_concurrency),
    )

    limiter: RateLimiter = InMemoryTokenBucket(settings.rate_limit_capacity, settings.rate_limit_refill_per_second)
    if redis:
        limiter = RedisTokenBucket(redis, settings.rate_limit_capacity, settings.rate_limit_refill_per_second)

    logger.info(
        "container ready",
        extra={
            "vector_store": settings.vector_store,
            "llm": services.llm.describe(),
            "checkpointer": settings.checkpointer,
        },
    )
    return Container(
        settings=settings,
        services=services,
        graph=build_agent_graph(checkpointer=checkpointer, store=store),
        users=UserDirectory(),
        rate_limiter=limiter,
        audit=audit,
        redis=redis,
    )

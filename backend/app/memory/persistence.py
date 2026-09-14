"""Checkpointer (short-term memory) and store (long-term memory) construction.

`memory`: in-process (tests, single-process demo).
`redis`: shared across API replicas. Checkpoints expire after `SESSION_TTL_MINUTES` of
inactivity (TTL refreshed on read), so abandoned sessions do not accumulate.
"""

from __future__ import annotations

from contextlib import AsyncExitStack

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.agents.state import CHECKPOINT_TYPES
from app.auth.models import AccessLevel
from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.models import Chunk, DocumentMetadata, DocumentType, RetrievedChunk

logger = get_logger(__name__)

_EXTRA_TYPES = (RetrievedChunk, Chunk, DocumentMetadata, DocumentType, AccessLevel)


def checkpoint_serializer() -> JsonPlusSerializer:
    allowed = [(cls.__module__, cls.__name__) for cls in (*CHECKPOINT_TYPES, *_EXTRA_TYPES)]
    return JsonPlusSerializer(allowed_msgpack_modules=allowed)


async def build_persistence(settings: Settings, stack: AsyncExitStack) -> tuple[BaseCheckpointSaver, BaseStore]:
    if settings.checkpointer == "redis":
        if not settings.redis_url:
            raise ValueError("CHECKPOINTER=redis requires REDIS_URL")
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        from langgraph.store.redis.aio import AsyncRedisStore

        ttl_minutes = settings.session_ttl_minutes
        saver = await stack.enter_async_context(
            AsyncRedisSaver.from_conn_string(
                settings.redis_url, ttl={"default_ttl": ttl_minutes, "refresh_on_read": True}
            )
        )
        saver.serde = checkpoint_serializer()
        await saver.asetup()
        store = await stack.enter_async_context(AsyncRedisStore.from_conn_string(settings.redis_url))
        await store.setup()
        logger.info("persistence: redis checkpointer + store")
        return saver, store

    # SCALE-DEBT: in-process checkpoints and memories do not survive restarts or span replicas.
    logger.info("persistence: in-memory checkpointer + store")
    return InMemorySaver(serde=checkpoint_serializer()), InMemoryStore()

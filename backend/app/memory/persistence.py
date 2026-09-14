"""Checkpointer (short-term memory) and store (long-term memory) construction.

`memory`: in-process (tests, single-process demo).
`redis`: shared across API replicas. Checkpoints expire after `SESSION_TTL_MINUTES` of
inactivity (TTL refreshed on read), so abandoned sessions do not accumulate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
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


def _allowed_types() -> list[tuple[str, str]]:
    return [(cls.__module__, cls.__name__) for cls in (*CHECKPOINT_TYPES, *_EXTRA_TYPES)]


def checkpoint_serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_allowed_types())


def redis_checkpoint_serializer() -> JsonPlusSerializer:
    """The Redis saver needs its own serializer subclass (it pre-processes values into RedisJSON
    documents). Replacing it with a plain JsonPlusSerializer breaks every checkpoint write, so we
    build the subclass with the same type allow-list (for both msgpack and JSON revival)."""
    from langgraph.checkpoint.redis.jsonplus_redis import JsonPlusRedisSerializer

    # RedisJSON stores state as JSON constructor envelopes, which are revived only for exact symbols
    # listed in allowed_json_modules; otherwise models silently come back as plain dicts.
    json_symbols = [(*module.split("."), name) for module, name in _allowed_types()]
    return JsonPlusRedisSerializer(allowed_msgpack_modules=_allowed_types(), allowed_json_modules=json_symbols)


def _error_chain_text(exc: BaseException) -> str:
    parts, seen = [], set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


async def _setup_race_safe(setup: Callable[[], Awaitable[object]], component: str, attempts: int = 5) -> None:
    """Create Redis search indexes, tolerating other workers creating them at the same moment.

    Every API worker runs setup at startup. With WEB_CONCURRENCY > 1 or several replicas, two
    processes can both see an index as missing and race to create it; the loser gets
    "Index already exists". Retrying converges because the next attempt finds the index present.
    """
    for attempt in range(1, attempts + 1):
        try:
            await setup()
            return
        except Exception as exc:
            if "already exists" not in _error_chain_text(exc) or attempt == attempts:
                raise
            logger.info(
                "redis index created concurrently by another worker; retrying setup",
                extra={"component": component, "attempt": attempt},
            )
            await asyncio.sleep(0.1 * attempt)


async def build_persistence(settings: Settings, stack: AsyncExitStack) -> tuple[BaseCheckpointSaver, BaseStore]:
    if settings.checkpointer == "redis":
        if not settings.redis_url:
            raise ValueError("CHECKPOINTER=redis requires REDIS_URL")
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver
        from langgraph.store.redis.aio import AsyncRedisStore

        # Constructed directly rather than via from_conn_string(): entering those context managers runs
        # index creation immediately, outside our race-safe wrapper. Cleanup is still registered.
        saver = AsyncRedisSaver(
            redis_url=settings.redis_url,
            ttl={"default_ttl": settings.session_ttl_minutes, "refresh_on_read": True},
        )
        saver.serde = redis_checkpoint_serializer()
        await _setup_race_safe(saver.asetup, "checkpointer")
        stack.push_async_exit(saver.__aexit__)

        store = AsyncRedisStore(redis_url=settings.redis_url)
        await _setup_race_safe(store.setup, "store")
        await stack.enter_async_context(store)  # __aenter__ only sets client info; __aexit__ closes the client
        logger.info("persistence: redis checkpointer + store")
        return saver, store

    # SCALE-DEBT: in-process checkpoints and memories do not survive restarts or span replicas.
    logger.info("persistence: in-memory checkpointer + store")
    return InMemorySaver(serde=checkpoint_serializer()), InMemoryStore()

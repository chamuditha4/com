"""Opt-in: shared state survives across API workers when backed by Redis 8.

Run with a disposable Redis 8 database:
    docker run --rm -p 6379:6379 redis:8.2
    TEST_REDIS_URL=redis://localhost:6379/0 uv run pytest   # search indexes require DB 0; use a dedicated instance backend/tests/integration/test_redis_persistence.py
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import AsyncExitStack

import pytest

from app.container import build_container
from mcp_server.server import server as mcp_server
from tests.support import resume_turn, run_turn

REDIS_URL = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not REDIS_URL, reason="set TEST_REDIS_URL to a Redis 8 database to run")


@pytest.fixture
def redis_settings(test_settings):
    return test_settings.model_copy(
        update={
            "redis_url": REDIS_URL,
            "checkpointer": "redis",
            "rate_limit_capacity": 2,
            "rate_limit_refill_per_second": 0.001,
        }
    )


async def _worker(settings, stack: AsyncExitStack):
    return await build_container(settings, stack, mcp_target=mcp_server)


async def test_conversation_and_memory_survive_a_different_worker(redis_settings, analyst):
    session = f"redis-{uuid.uuid4().hex[:8]}"
    async with AsyncExitStack() as stack:
        first = await _worker(redis_settings, stack)
        await run_turn(
            first, analyst, "I work in payments operations. What is the password policy?", session_id=session
        )

    async with AsyncExitStack() as stack:  # a fresh process-equivalent: nothing shared in memory
        second = await _worker(redis_settings, stack)
        follow_up = await run_turn(second, analyst, "How long must passwords be?", session_id=session)
        assert len(follow_up["messages"]) == 4

        new_session = await run_turn(
            second, analyst, "Which runbooks matter for payments operations?", session_id=f"{session}-b"
        )
        assert any("payments operations" in m for m in new_session["recalled_memories"])


async def test_approval_can_be_resumed_by_another_worker(redis_settings, admin):
    session = f"hitl-{uuid.uuid4().hex[:8]}"
    async with AsyncExitStack() as stack:
        paused = await run_turn(
            await _worker(redis_settings, stack), admin, "Please reindex the knowledge base", session_id=session
        )
        assert "__interrupt__" in paused

    async with AsyncExitStack() as stack:
        done = await resume_turn(await _worker(redis_settings, stack), admin, approved=True, session_id=session)
        assert done["tool_calls"][0].status == "succeeded"


async def test_rate_limit_is_shared_between_workers(redis_settings):
    key = f"user:test-{uuid.uuid4().hex[:8]}"
    async with AsyncExitStack() as stack:
        worker_a = await _worker(redis_settings, stack)
        worker_b = await _worker(redis_settings, stack)
        decisions = [
            await worker_a.rate_limiter.acquire(key),
            await worker_b.rate_limiter.acquire(key),
            await worker_a.rate_limiter.acquire(key),
        ]
        assert [d.allowed for d in decisions] == [True, True, False]


async def test_concurrent_worker_startup_is_race_safe(redis_settings):
    """Several workers boot at once against an empty Redis and all create the search indexes."""
    from redis.asyncio import Redis

    client = Redis.from_url(REDIS_URL)
    await client.flushall()
    await client.aclose()
    async with AsyncExitStack() as stack:
        workers = await asyncio.gather(*(_worker(redis_settings, stack) for _ in range(4)))
        assert len(workers) == 4

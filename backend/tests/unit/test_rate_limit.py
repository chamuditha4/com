import math

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.core.rate_limit import (
    InMemoryTokenBucket,
    RedisTokenBucket,
    _refill_and_consume,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_bucket_math_allows_burst_then_reports_exact_retry_after():
    tokens, decision = _refill_and_consume(0.5, last_refill=10.0, now=10.0, capacity=5, refill_per_second=0.5, cost=1)
    assert not decision.allowed
    assert decision.retry_after_seconds == pytest.approx(1.0)  # needs 0.5 tokens at 0.5/s
    assert tokens == pytest.approx(0.5)


def test_bucket_never_exceeds_capacity_after_long_idle():
    tokens, decision = _refill_and_consume(0, last_refill=0, now=10_000, capacity=3, refill_per_second=1, cost=1)
    assert decision.allowed
    assert tokens == pytest.approx(2)


def test_zero_refill_rate_means_retry_never():
    _, decision = _refill_and_consume(0, 0, 100, capacity=1, refill_per_second=0, cost=1)
    assert decision.retry_after_seconds == math.inf


async def test_in_memory_bucket_is_per_user_and_refills():
    clock = FakeClock()
    limiter = InMemoryTokenBucket(capacity=2, refill_per_second=1.0, clock=clock)

    assert (await limiter.acquire("alice")).allowed
    assert (await limiter.acquire("alice")).allowed
    denied = await limiter.acquire("alice")
    assert not denied.allowed
    assert denied.retry_after_seconds == pytest.approx(1.0)

    # Another user has an independent bucket.
    assert (await limiter.acquire("bob")).allowed

    clock.now += 1.0
    assert (await limiter.acquire("alice")).allowed


async def test_redis_bucket_enforces_limit_atomically():
    redis = fakeredis.FakeAsyncRedis()
    limiter = RedisTokenBucket(redis, capacity=3, refill_per_second=0.001)

    decisions = [await limiter.acquire("carol") for _ in range(4)]

    assert [d.allowed for d in decisions] == [True, True, True, False]
    assert decisions[-1].retry_after_seconds > 0


async def test_redis_outage_fails_open_to_local_bucket():
    class BrokenScript:
        async def __call__(self, **_: object) -> None:
            raise RedisConnectionError("redis down")

    redis = fakeredis.FakeAsyncRedis()
    limiter = RedisTokenBucket(redis, capacity=1, refill_per_second=0.001)
    limiter._script = BrokenScript()

    assert (await limiter.acquire("dave")).allowed
    assert not (await limiter.acquire("dave")).allowed  # local bucket still limits

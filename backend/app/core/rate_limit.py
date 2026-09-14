"""Per-user token bucket rate limiting.

Algorithm: each user has a bucket of `capacity` tokens refilled at `refill_per_second`.
A request consumes `cost` tokens; if not enough are available it is rejected with the exact
time until it would succeed (`retry_after`). Bursts up to `capacity` are allowed, sustained
throughput is capped at the refill rate.

`RedisTokenBucket` performs refill + consume atomically in a Lua script using the Redis
server clock, so the limit holds across any number of API replicas. `InMemoryTokenBucket`
is the single-process fallback (tests / local dev).
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    remaining: float
    retry_after_seconds: float
    capacity: int


class RateLimiter(Protocol):
    async def acquire(self, key: str, cost: float = 1.0) -> RateLimitDecision: ...


def _refill_and_consume(
    tokens: float,
    last_refill: float,
    now: float,
    *,
    capacity: int,
    refill_per_second: float,
    cost: float,
) -> tuple[float, RateLimitDecision]:
    """Pure bucket arithmetic, shared by the in-memory limiter and tested directly."""
    elapsed = max(0.0, now - last_refill)
    tokens = min(float(capacity), tokens + elapsed * refill_per_second)
    if tokens >= cost:
        tokens -= cost
        return tokens, RateLimitDecision(True, tokens, 0.0, capacity)
    deficit = cost - tokens
    retry_after = math.inf if refill_per_second <= 0 else deficit / refill_per_second
    return tokens, RateLimitDecision(False, tokens, retry_after, capacity)


class InMemoryTokenBucket:
    # SCALE-DEBT: per-process state. Limits are per replica unless REDIS_URL is configured.
    def __init__(
        self,
        capacity: int,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capacity = capacity
        self._refill = refill_per_second
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, key: str, cost: float = 1.0) -> RateLimitDecision:
        async with self._lock:
            now = self._clock()
            tokens, last = self._buckets.get(key, (float(self._capacity), now))
            tokens, decision = _refill_and_consume(
                tokens, last, now, capacity=self._capacity, refill_per_second=self._refill, cost=cost
            )
            self._buckets[key] = (tokens, now)
            return decision


_LUA_TOKEN_BUCKET = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + math.max(0, now - ts) * refill)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end
redis.call('HSET', key, 'tokens', tokens, 'ts', now)
local ttl = math.ceil(capacity / math.max(refill, 0.001)) + 60
redis.call('EXPIRE', key, ttl)
return {allowed, tostring(tokens)}
"""


class RedisTokenBucket:
    def __init__(
        self,
        redis: Redis,
        capacity: int,
        refill_per_second: float,
        *,
        fallback: RateLimiter | None = None,
        prefix: str = "ratelimit",
    ) -> None:
        self._redis = redis
        self._capacity = capacity
        self._refill = refill_per_second
        self._prefix = prefix
        self._script = redis.register_script(_LUA_TOKEN_BUCKET)
        self._fallback = fallback or InMemoryTokenBucket(capacity, refill_per_second)

    async def acquire(self, key: str, cost: float = 1.0) -> RateLimitDecision:
        try:
            allowed, tokens_raw = await self._script(
                keys=[f"{self._prefix}:{key}"], args=[self._capacity, self._refill, cost]
            )
        except RedisError:
            # Fail open to a local bucket rather than taking chat down with Redis. Limits
            # temporarily become per-replica; the incident is visible in logs/alerts.
            logger.warning("rate limiter redis unavailable; using local bucket", exc_info=True)
            return await self._fallback.acquire(key, cost)
        tokens = float(tokens_raw)
        if allowed:
            return RateLimitDecision(True, tokens, 0.0, self._capacity)
        retry_after = (cost - tokens) / self._refill if self._refill > 0 else math.inf
        return RateLimitDecision(False, tokens, retry_after, self._capacity)

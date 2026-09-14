"""Append-only audit trail of tool executions and access decisions."""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.logging import get_logger

logger = get_logger(__name__)


class AuditEvent(BaseModel):
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    trace_id: str | None = None
    user_id: str
    role: str
    action: str
    target: str
    outcome: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AuditLog(Protocol):
    async def record(self, event: AuditEvent) -> None: ...

    async def recent(self, limit: int = 50) -> list[AuditEvent]: ...


class InMemoryAuditLog:
    # SCALE-DEBT: per-process ring buffer; RedisAuditLog is used when REDIS_URL is set.
    def __init__(self, maxlen: int = 1000) -> None:
        self._events: deque[AuditEvent] = deque(maxlen=maxlen)

    async def record(self, event: AuditEvent) -> None:
        logger.info("audit", extra={"audit": event.model_dump()})
        self._events.append(event)

    async def recent(self, limit: int = 50) -> list[AuditEvent]:
        return list(self._events)[-limit:][::-1]


class RedisAuditLog:
    def __init__(self, redis: Redis, key: str = "audit:events", maxlen: int = 10_000) -> None:
        self._redis = redis
        self._key = key
        self._maxlen = maxlen

    async def record(self, event: AuditEvent) -> None:
        logger.info("audit", extra={"audit": event.model_dump()})
        try:
            await self._redis.lpush(self._key, event.model_dump_json())
            await self._redis.ltrim(self._key, 0, self._maxlen - 1)
        except RedisError:
            logger.error("audit write failed (event kept in logs)", exc_info=True)

    async def recent(self, limit: int = 50) -> list[AuditEvent]:
        raw = await self._redis.lrange(self._key, 0, limit - 1)
        return [AuditEvent.model_validate(json.loads(r)) for r in raw]

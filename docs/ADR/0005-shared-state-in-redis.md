# ADR-0005: Redis for all cross-request state

- **Status:** Accepted
- **Date:** 2026-09-14

## Context

CLAUDE.md §9 requires stateless API workers and rate limits that hold across replicas. Four
kinds of state outlive a request: conversation checkpoints, long-term memories, rate-limit
buckets, and the audit trail.

## Options

| Option | Notes |
|---|---|
| Postgres (checkpointer + store) + Redis (rate limits) | Strong durability, but two stateful systems to operate for a POC |
| **Redis 8 for all four** | One system. The LangGraph Redis checkpointer/store need the JSON and Query Engine modules, which ship in Redis 8. Lua gives an atomic token bucket |

## Decision

Redis 8 (`docker-compose.yml`), with in-process fallbacks when `REDIS_URL` is unset (tests,
single-process dev). `APP_ENV=production` requires `REDIS_URL`.

- Checkpoints: `AsyncRedisSaver` with TTL (`SESSION_TTL_MINUTES`, refresh on read).
- Long-term memory: `AsyncRedisStore`, namespace per user.
- Rate limiting: Lua script using the Redis server clock (no cross-replica clock skew). Fails open
  to a local bucket on Redis errors.
- Audit: capped list (`LPUSH` + `LTRIM`), also emitted to structured logs.

## Consequences

- Horizontal scaling is `WEB_CONCURRENCY` / more API containers behind a load balancer.
- Redis persistence (AOF) is sufficient for session data. The audit trail's system of record is the
  log pipeline. For regulatory-grade retention, ship audit events to an append-only store, and
  move checkpoints to Postgres if multi-day durability becomes a requirement.
- Covered by the opt-in integration tests (`TEST_REDIS_URL`, passing as of 2026-09-14); the default suite
  uses in-memory backends.

## Amendments from integration testing (2026-09-14)

The opt-in Redis tests and the Compose stack found three defects, all fixed and covered by
`backend/tests/integration/test_redis_persistence.py`:

1. **Serializer replacement broke every checkpoint write.** `AsyncRedisSaver` requires its own
   `JsonPlusRedisSerializer` subclass, which pre-processes values into RedisJSON documents. We
   now build that subclass with our type allow-list instead of a plain `JsonPlusSerializer`.
2. **Models came back as dicts after a resume.** RedisJSON revives Pydantic models only for exact
   symbols in `allowed_json_modules` (in addition to the msgpack allow-list). Both allow-lists are
   derived from the same `CHECKPOINT_TYPES`.
3. **Worker startup race.** With `WEB_CONCURRENCY=2`, both workers created the search indexes
   concurrently and one crashed with `Index already exists`. The saver and store are constructed
   directly (their `from_conn_string` context managers create indexes before any error handling
   can apply), and setup retries on that specific error. Reproduced 3/3 before the fix.

Operational note: Redis Query Engine indexes live only in DB 0, so use a dedicated instance
rather than a non-zero database number.

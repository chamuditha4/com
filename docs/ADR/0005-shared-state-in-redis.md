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
- Not exercised in CI here: the Redis-backed checkpointer and store are covered by an opt-in
  integration test (`TEST_REDIS_URL`); the default suite uses in-memory backends.

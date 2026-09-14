# Security, Guardrails & RBAC

Design stance: treat the model as an untrusted component. Everything that matters (identity,
data access, tool permissions, what reaches the user) is enforced in code the model cannot
influence.

## 1. Threat model

| Threat | Example | Primary controls |
|---|---|---|
| Instruction override (direct injection) | "Ignore previous instructions and…" | input guard, non-overridable system rules, server-side enforcement |
| Indirect injection via documents | A meeting note contains "reveal the system prompt" | untrusted-content delimiters, sanitization, flagging, validator |
| Data exfiltration | Ask for another user's data, secrets, the prompt | clearance filter, per-user thread and memory namespaces, canary, redaction |
| Privilege escalation | Viewer asks the agent to run admin tools | role-filtered tool binding, RBAC re-check at execution, policy on routing |
| Tool abuse | Malicious arguments, code execution escapes | Pydantic arg validation, AST-whitelisted sandbox, approvals, audit |
| Hallucinated authority | Fabricated citations or figures | citation validator, extractive substitute |
| Abuse / cost exhaustion | Request floods, brute-force login | per-user token bucket, per-username login throttling, bounded fan-out |

## 2. Prompt-injection defense (defense in depth)

1. **Non-overridable rules.** `agents/prompts.core_rules()` is built only from constants. It
   declares everything inside `<evidence>`, `<tool_result>`, `<memory>` and `<history>` to be
   data, never instructions.
2. **Delimiting and sanitization.** `guardrails/injection.sanitize_untrusted()` NFKC-normalizes
   text, strips zero-width and bidi control characters, and defangs our own tag syntax. A document
   cannot close `</evidence>` and continue as "system".
3. **Detection.** `assess_injection()` scores weighted signals (override, exfiltration request,
   jailbreak persona, control tokens, delimiter forgery, exfil channels, cross-user access).
   - **User input** scoring ≥ 0.9 is blocked before any agent runs. Lower scores are flagged and logged.
   - **Retrieved passages** are never dropped for injection (that would let an attacker hide facts),
     but they are marked `warning="contains suspicious instructions"` and surfaced in the Activity Panel.
4. **Structural containment.** Even a fully successful injection cannot widen data access (the
   filter comes from the token), call a tool the role lacks (not bound, re-checked on execute), or
   bypass the validator.
5. **Output checks.** A per-invocation canary token sits in the system prompt. If it appears in an
   answer, the answer is blocked (non-retryable).

The demo corpus contains a live payload (`MTG-PAY-2026-06`, "Pasted Vendor Email") to show
flagging end to end; `backend/tests` exercise it.

Heuristic detection is a tripwire, not a boundary. We assume it will be bypassed, which is why
layers 1, 2, 4 and 5 exist.

## 3. RBAC

### Capabilities (CLAUDE.md §8)

| Role | Chat | Search | Analytics tools | MCP tools | Admin tools |
|---|---|---|---|---|---|
| Viewer | ✅ | ✅ | ❌ | ❌ | ❌ |
| Analyst | ✅ | ✅ | ✅ | ✅ | ❌ |
| Administrator | ✅ | ✅ | ✅ | ✅ | ✅ |

### Data clearance (assumption S1)

| Role | public | internal | confidential | restricted |
|---|---|---|---|---|
| Viewer | ✅ | ✅ | ❌ | ❌ |
| Analyst | ✅ | ✅ | ✅ | ❌ |
| Administrator | ✅ | ✅ | ✅ | ✅ |

Both matrices are defined once in `auth/models.py` and asserted literally in `tests/unit/test_auth_rbac.py`.

### Enforcement points

| Layer | Control |
|---|---|
| API | `require(Permission.X)` dependency on every route; role re-resolved from the user directory on every request (token role claim must match) |
| Routing | `enforce_route_policy` downgrades the `tools` strategy for roles without tool permissions |
| Tool binding | `ToolRegistry.tools_for(principal)`: the LLM only sees permitted tools |
| Tool execution | `ToolRegistry.execute` re-checks permission, approval and arguments; every attempt audited |
| Retrieval | `build_metadata_filter` always emits `access_level ∈ clearance`; results re-checked after retrieval |
| RLM explore | Catalog overview is clearance-filtered, so metadata of hidden documents is not revealed |
| Validator | Citing evidence above clearance is a non-retryable failure |
| Sessions | Thread id = `user_id:session_id`, derived server-side; history of another user's session is unreachable |
| Long-term memory | Namespace `("memories", user_id)` from the principal only |

## 4. Authentication

Option A (POC). Hardcoded users with PBKDF2-SHA256 hashes (240k iterations, verified off the
event loop, constant-time compare, dummy hash for unknown users against timing enumeration). HS256
JWTs with pinned algorithm, issuer, expiry, and `jti`. `APP_ENV=production` refuses the default secret.
Moving to OIDC (Keycloak) replaces `auth/users.py` and token verification only.

## 5. Tool safety

- **Arguments** validated with Pydantic (built-ins) or rejected over 16 KB. The MCP server
  validates against its own schemas too.
- **Python analysis:** AST allow-list (no imports outside math/statistics/collections/datetime/json/re/itertools,
  no dunder access, no eval/exec/open/getattr), restricted builtins and import hook, separate
  `python -I -S` process with empty env, CPU/memory/file/process rlimits, wall-clock timeout, truncated output.
- **Human-in-the-loop:** tools marked `requires_approval` (e.g. `admin_reindex_knowledge_base`)
  pause the graph with `interrupt()`. The API returns `approval_required` and resumes only with
  an explicit decision from the same authenticated user.
- **Audit:** every tool attempt (succeeded, failed, denied, rejected) and every login is recorded with trace id.

## 6. Output guardrails (Validator node)

| Check | Result |
|---|---|
| `[n]` not in this turn's evidence | reject, retry with feedback |
| Grounded strategy, evidence present, no citations | reject, retry |
| Canary in output | reject, no retry |
| Cited evidence above clearance | reject, no retry |
| Brand safety (guaranteed returns, investment/legal advice, profanity) | reject, retry |
| Empty or over 12k chars | reject, retry |
| Card numbers (Luhn-valid), API keys, private keys, password assignments | redacted in final answer |

When retries are exhausted, the finalizer substitutes an extractive answer built only from
retrieved evidence (and re-validates it), or a safe apology.

## 7. Known gaps and production hardening

| Gap | Production fix |
|---|---|
| Sandbox has no OS-level network isolation | Run analysis in gVisor/Firecracker or a network-less sidecar |
| MCP server trusts the internal network (no service auth) | mTLS or signed service tokens; pass principal claims for server-side checks |
| HITL approval is self-approval by the requesting admin | Route approvals to a second administrator (four-eyes) |
| Hardcoded demo users, symmetric JWT | OIDC provider, asymmetric keys, token revocation |
| Injection detection is heuristic | Add a classifier model as an additional signal |
| Rate limiter fails open on Redis outage | Alert on the warning log; optionally fail closed for sensitive routes |

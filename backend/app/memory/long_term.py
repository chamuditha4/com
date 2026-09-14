"""Long-term (cross-session) user memory.

What is stored: short, durable facts the *user explicitly stated about themselves or their
preferences* ("I work in payments operations", "prefer bullet-point answers"). Never
document content, tool output, secrets or other people's data.

Isolation: the namespace is `("memories", <user_id>)`, and the user id always comes from the
authenticated principal in the runtime context, never from model output or request input.

Retrieval: at most `max_facts` per user, ranked by lexical overlap with the question plus
recency. That's small and cheap enough not to need a vector index; see docs/MEMORY_DESIGN.md.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from langgraph.store.base import BaseStore

from app.guardrails.injection import assess_injection
from app.guardrails.output import redact_sensitive
from app.retrieval.sparse import tokenize

_CUES = re.compile(
    r"\b(remember|note that|keep in mind|from now on|i prefer|i'd prefer|i like|please always|"
    r"my (team|department|role|name|manager) is|i work (in|on|for)|i am (the|a|an) |i'm (the|a|an) |call me)\b",
    re.IGNORECASE,
)
_CUE_SENTENCE = re.compile(r"[^.!?\n]*" + _CUES.pattern + r"[^.!?\n]*", re.IGNORECASE)
MAX_FACT_CHARS = 200


def has_memory_cue(text: str) -> bool:
    return bool(_CUES.search(text))


def heuristic_facts(text: str) -> list[str]:
    """Cue sentences from the user's message, used when no LLM is available."""
    facts = []
    for match in _CUE_SENTENCE.finditer(text):
        sentence = match.group(0).strip(" ,;:")
        if 8 <= len(sentence) <= MAX_FACT_CHARS:
            facts.append(sentence[0].upper() + sentence[1:])
    return facts[:3]


def is_storable(fact: str) -> bool:
    return 0 < len(fact) <= MAX_FACT_CHARS and not redact_sensitive(fact).kinds and not assess_injection(fact).flagged


class LongTermMemory:
    def __init__(self, store: BaseStore, *, max_facts: int = 50) -> None:
        self._store = store
        self._max_facts = max_facts

    @staticmethod
    def _namespace(user_id: str) -> tuple[str, ...]:
        return ("memories", user_id)

    async def recall(self, user_id: str, query: str, limit: int = 5) -> list[str]:
        items = await self._store.asearch(self._namespace(user_id), limit=self._max_facts)
        query_terms = set(tokenize(query))

        def rank(item) -> tuple[int, str]:  # type: ignore[no-untyped-def]
            overlap = len(query_terms & set(tokenize(item.value["text"])))
            preference = 1 if "prefer" in item.value["text"].lower() else 0
            return (overlap + preference, item.value.get("created_at", ""))

        ranked = sorted(items, key=rank, reverse=True)
        return [item.value["text"] for item in ranked[:limit]]

    async def remember(self, user_id: str, facts: list[str]) -> list[str]:
        stored = []
        for fact in facts:
            fact = " ".join(fact.split())
            if not is_storable(fact):
                continue
            key = hashlib.sha256(fact.lower().encode()).hexdigest()[:16]  # idempotent per fact
            await self._store.aput(
                self._namespace(user_id),
                key,
                {"text": fact, "created_at": datetime.now(UTC).isoformat()},
            )
            stored.append(fact)
        await self._prune(user_id)
        return stored

    async def _prune(self, user_id: str) -> None:
        items = await self._store.asearch(self._namespace(user_id), limit=self._max_facts * 2)
        if len(items) <= self._max_facts:
            return
        oldest_first = sorted(items, key=lambda i: i.value.get("created_at", ""))
        for item in oldest_first[: len(items) - self._max_facts]:
            await self._store.adelete(self._namespace(user_id), item.key)

    async def forget_all(self, user_id: str) -> int:
        items = await self._store.asearch(self._namespace(user_id), limit=self._max_facts * 2)
        for item in items:
            await self._store.adelete(self._namespace(user_id), item.key)
        return len(items)

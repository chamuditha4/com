"""Runtime context: per-invocation, non-persisted dependencies for graph nodes.

The principal is deliberately passed here and *not* in `AgentState`. State is checkpointed and
replayed; the identity used for authorization must come from the verified token on every
request (including a resume after a human-in-the-loop interrupt).
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from datetime import date

from app.auth.models import Principal
from app.core.config import Settings
from app.llm.gateway import LLMClient
from app.memory.long_term import LongTermMemory
from app.retrieval.service import HybridRetriever
from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentServices:
    """Worker-lifetime singletons, built once in the FastAPI lifespan."""

    settings: Settings
    retriever: HybridRetriever
    llm: LLMClient
    tools: ToolRegistry
    memory: LongTermMemory | None
    # Worker-wide cap on concurrently analyzed RLM slices, so one heavy research query cannot
    # monopolise retrieval and LLM capacity for every other user on this worker.
    research_semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))


@dataclass(frozen=True)
class AgentContext:
    principal: Principal
    trace_id: str
    session_id: str
    services: AgentServices
    today: date = field(default_factory=date.today)
    # A per-invocation secret embedded in the system prompt. If it ever appears in an answer,
    # the model has leaked its instructions, and the validator blocks the answer.
    canary: str = field(default_factory=lambda: f"CB-CANARY-{secrets.token_hex(6)}")

"""Provider-agnostic LLM access.

Agents depend on the small `LLMClient` protocol, never on a vendor SDK. The production
implementation wraps LangChain chat models, which gives us for free:

* LangSmith tracing of every call (prompts, tokens, latency) nested under the graph run;
* `with_retry` (exponential backoff with jitter) and `with_fallbacks` (secondary provider);
* one structured-output API across Anthropic, OpenAI and Gemini.

Two tiers keep cost and latency proportional to the task:
* `fast`: routing, classification, extraction (small, cheap model);
* `reasoning`: research planning and answer synthesis (strongest model).

When no provider is configured, `NullLLM` reports `available = False` and every call raises
`LLMUnavailableError`. Each node catches that and takes its deterministic fallback path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from app.core.exceptions import LLMUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

Tier = Literal["fast", "reasoning"]
T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    @property
    def available(self) -> bool: ...

    def describe(self) -> dict[str, str]: ...

    async def generate(
        self, messages: Sequence[BaseMessage], *, tier: Tier = "reasoning", run_name: str = "generate"
    ) -> str: ...

    async def structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[T],
        *,
        tier: Tier = "fast",
        run_name: str = "structured",
    ) -> T: ...

    async def invoke_with_tools(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
        *,
        tier: Tier = "fast",
        run_name: str = "tool_planning",
    ) -> AIMessage: ...


class NullLLM:
    """No provider configured (offline mode) or provider disabled."""

    available = False

    def __init__(self, reason: str = "no LLM provider configured") -> None:
        self._reason = reason

    def describe(self) -> dict[str, str]:
        return {"provider": "none", "reason": self._reason}

    async def generate(self, messages: Sequence[BaseMessage], **_: Any) -> str:
        raise LLMUnavailableError(self._reason)

    async def structured(self, messages: Sequence[BaseMessage], schema: type[T], **_: Any) -> T:
        raise LLMUnavailableError(self._reason)

    async def invoke_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[BaseTool], **_: Any
    ) -> AIMessage:
        raise LLMUnavailableError(self._reason)


class LangChainLLM:
    available = True

    def __init__(
        self,
        *,
        models: dict[Tier, BaseChatModel],
        fallbacks: dict[Tier, BaseChatModel] | None = None,
        max_retries: int = 2,
        max_concurrency: int = 8,
        description: dict[str, str] | None = None,
    ) -> None:
        self._models = models
        self._fallbacks = fallbacks or {}
        self._attempts = max_retries + 1
        # Bounded concurrency per worker: a wide RLM fan-out cannot exhaust provider rate
        # limits or sockets and starve other users' requests.
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._description = description or {}

    def describe(self) -> dict[str, str]:
        return self._description

    def _resilient(self, tier: Tier, build: Any) -> Runnable:
        runnable: Runnable = build(self._models[tier]).with_retry(
            stop_after_attempt=self._attempts, wait_exponential_jitter=True
        )
        if fallback := self._fallbacks.get(tier):
            runnable = runnable.with_fallbacks([build(fallback)])
        return runnable

    async def _invoke(self, runnable: Runnable, messages: Sequence[BaseMessage], run_name: str) -> Any:
        async with self._semaphore:
            try:
                return await runnable.ainvoke(list(messages), config={"run_name": run_name})
            except Exception as exc:
                logger.error("llm call failed after retries", extra={"run_name": run_name}, exc_info=True)
                raise LLMUnavailableError() from exc

    async def generate(
        self, messages: Sequence[BaseMessage], *, tier: Tier = "reasoning", run_name: str = "generate"
    ) -> str:
        message = await self._invoke(self._resilient(tier, lambda m: m), messages, run_name)
        return message.text if isinstance(message, AIMessage) else str(message)

    async def structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[T],
        *,
        tier: Tier = "fast",
        run_name: str = "structured",
    ) -> T:
        runnable = self._resilient(tier, lambda m: m.with_structured_output(schema))
        result = await self._invoke(runnable, messages, run_name)
        if not isinstance(result, schema):  # provider returned a dict or partial output
            try:
                result = schema.model_validate(result)
            except Exception as exc:
                raise LLMUnavailableError("The language model returned an invalid response.") from exc
        return result

    async def invoke_with_tools(
        self,
        messages: Sequence[BaseMessage],
        tools: Sequence[BaseTool],
        *,
        tier: Tier = "fast",
        run_name: str = "tool_planning",
    ) -> AIMessage:
        runnable = self._resilient(tier, lambda m: m.bind_tools(list(tools)))
        return await self._invoke(runnable, messages, run_name)

"""Build the configured `LLMClient` from settings. The only module that imports vendor SDKs."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.llm.gateway import LangChainLLM, LLMClient, NullLLM, Tier

logger = get_logger(__name__)


def _api_key(provider: str, settings: Settings) -> str | None:
    secret = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "gemini": settings.google_api_key,
    }.get(provider)
    return secret.get_secret_value() if secret else None


def build_chat_model(provider: str, model: str, settings: Settings) -> BaseChatModel:
    key = _api_key(provider, settings)
    common = {"temperature": settings.llm_temperature, "max_retries": 0}  # we retry in the gateway
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, api_key=key, timeout=settings.llm_timeout_seconds, max_tokens=4096, **common
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=key, timeout=settings.llm_timeout_seconds, **common)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model, google_api_key=key, timeout=settings.llm_timeout_seconds, **common
        )
    raise ValueError(f"unsupported LLM provider: {provider}")


def build_llm(settings: Settings) -> LLMClient:
    provider = settings.llm_provider
    if provider == "none":
        return NullLLM("LLM_PROVIDER=none (offline mode)")
    if not _api_key(provider, settings):
        logger.warning("llm api key missing; running in degraded offline mode", extra={"provider": provider})
        return NullLLM(f"missing API key for {provider}")

    models: dict[Tier, BaseChatModel] = {
        "reasoning": build_chat_model(provider, settings.llm_model, settings),
        "fast": build_chat_model(provider, settings.llm_fast_model, settings),
    }
    fallbacks: dict[Tier, BaseChatModel] = {}
    fb_provider = settings.llm_fallback_provider
    if fb_provider != "none" and settings.llm_fallback_model and _api_key(fb_provider, settings):
        fallback = build_chat_model(fb_provider, settings.llm_fallback_model, settings)
        fallbacks = {"reasoning": fallback, "fast": fallback}

    return LangChainLLM(
        models=models,
        fallbacks=fallbacks,
        max_retries=settings.llm_max_retries,
        max_concurrency=settings.llm_max_concurrency,
        description={
            "provider": provider,
            "reasoning_model": settings.llm_model,
            "fast_model": settings.llm_fast_model,
            "fallback": f"{fb_provider}:{settings.llm_fallback_model}" if fallbacks else "none",
        },
    )

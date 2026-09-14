"""LangSmith configuration.

The LangSmith SDK reads its settings from process environment variables. Values provided in
`.env` are loaded by pydantic-settings but not exported to the environment, so we export them
here, once, at startup. When tracing is enabled, every graph run, node, LLM call and
`@traceable` function (hybrid retrieval, RLM slices) lands in the configured project, with the
API trace id as the root run id.
"""

from __future__ import annotations

import os

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_tracing(settings: Settings) -> bool:
    api_key = settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else ""
    enabled = settings.langsmith_tracing and bool(api_key)
    if settings.langsmith_tracing and not api_key:
        logger.warning("LANGSMITH_TRACING is true but LANGSMITH_API_KEY is missing; tracing disabled")

    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    if enabled:
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
        os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
        if settings.langsmith_workspace_id:
            os.environ.setdefault("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id)
    logger.info("langsmith tracing", extra={"enabled": enabled, "project": settings.langsmith_project})
    return enabled

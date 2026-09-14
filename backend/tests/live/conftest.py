"""Live-suite setup, mirroring app startup: export LangSmith settings before any client exists.

The LangSmith SDK reads its environment (including LANGSMITH_WORKSPACE_ID) when its client is first
created. If an earlier live test triggers tracing machinery first, a client without the workspace
id gets cached and every later upload fails with 403.
"""

from __future__ import annotations

import os

from app.core.config import Settings
from app.observability.tracing import configure_tracing


def pytest_configure(config: object) -> None:
    if os.getenv("RUN_LIVE_TESTS"):
        configure_tracing(Settings())

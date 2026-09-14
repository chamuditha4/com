"""MCP client adapter: discovers tools from the operations MCP server and calls them.

Connections are opened per call. The server runs in stateless request mode, which keeps API
workers stateless and makes MCP-server restarts invisible to users. Tool discovery is cached
with a TTL, and failures are reported as tool errors, never raised into the graph.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from mcp import Client

from app.core.logging import get_logger

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp_"


@dataclass(frozen=True)
class RemoteToolDefinition:
    name: str  # prefixed, as exposed to the LLM
    remote_name: str
    description: str
    input_schema: dict[str, Any]


class MCPToolProvider:
    def __init__(self, target: Any, *, timeout_seconds: float = 10.0, cache_ttl_seconds: float = 300) -> None:
        # `target` is a URL in production; tests pass an in-process MCPServer instance.
        self._target = target
        self._timeout = timeout_seconds
        self._ttl = cache_ttl_seconds
        self._cache: tuple[float, list[RemoteToolDefinition]] | None = None
        self._lock = asyncio.Lock()

    async def list_tools(self) -> list[RemoteToolDefinition]:
        async with self._lock:
            if self._cache and time.monotonic() - self._cache[0] < self._ttl:
                return self._cache[1]
            try:
                async with asyncio.timeout(self._timeout), Client(self._target) as client:
                    result = await client.list_tools()
            except Exception:
                logger.warning("mcp tool discovery failed", exc_info=True)
                if self._cache:
                    return self._cache[1]
                # Negative cache: retry discovery after 30s instead of on every request.
                self._cache = (time.monotonic() - self._ttl + 30, [])
                return []
            tools = [
                RemoteToolDefinition(
                    name=f"{MCP_TOOL_PREFIX}{t.name}",
                    remote_name=t.name,
                    description=t.description or t.name,
                    input_schema=t.input_schema,
                )
                for t in result.tools
            ]
            self._cache = (time.monotonic(), tools)
            return tools

    async def call(self, remote_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Returns (ok, text). Never raises."""
        try:
            async with asyncio.timeout(self._timeout), Client(self._target) as client:
                result = await client.call_tool(remote_name, arguments)
        except TimeoutError:
            return False, f"MCP tool '{remote_name}' timed out after {self._timeout}s"
        except Exception as exc:
            logger.warning("mcp call failed", extra={"tool": remote_name}, exc_info=True)
            return False, f"MCP server unavailable: {type(exc).__name__}"

        if result.structured_content is not None:
            text = json.dumps(result.structured_content, default=str)
        else:
            text = "\n".join(getattr(block, "text", "") for block in result.content)
        return (not result.is_error), text

    async def ping(self) -> bool:
        return bool(await self.list_tools())

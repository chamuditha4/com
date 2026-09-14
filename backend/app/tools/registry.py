"""Tool registry: the single enforcement point for tool RBAC.

Enforcement is layered:
1. `tools_for(principal)` returns only the tools the role permits. The LLM is bound to exactly
   that list, so it never sees tools it cannot use.
2. `execute()` re-checks permission, approval and argument schema on every call, because a
   model could still emit a call to a tool name it guessed.
3. Every attempt (allowed, denied, failed) is written to the audit log.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from app.auth.models import Permission, Principal
from app.core.logging import get_logger
from app.retrieval.models import RetrievedChunk
from app.tools.audit import AuditEvent, AuditLog
from app.tools.mcp_client import MCPToolProvider

logger = get_logger(__name__)

MAX_ARGS_BYTES = 16_000
MAX_OUTPUT_CHARS = 12_000


@dataclass
class ToolOutput:
    text: str
    documents: list[RetrievedChunk] = field(default_factory=list)


ToolHandler = Callable[[Principal, dict[str, Any]], Awaitable[ToolOutput]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    permission: Permission
    parameters: dict[str, Any]
    handler: ToolHandler
    args_model: type[BaseModel] | None = None
    requires_approval: bool = False
    source: Literal["builtin", "mcp"] = "builtin"
    timeout_seconds: float = 15.0

    def to_llm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolResult(BaseModel):
    tool: str
    ok: bool
    output: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    documents: list[RetrievedChunk] = []


class ToolRegistry:
    def __init__(
        self,
        *,
        builtin: list[ToolSpec],
        audit: AuditLog,
        mcp: MCPToolProvider | None = None,
    ) -> None:
        self._builtin = {spec.name: spec for spec in builtin}
        self._audit = audit
        self._mcp = mcp

    async def _all_specs(self) -> dict[str, ToolSpec]:
        specs = dict(self._builtin)
        if self._mcp is None:
            return specs
        provider = self._mcp
        for remote in await provider.list_tools():

            async def call_remote(_: Principal, args: dict[str, Any], _name: str = remote.remote_name) -> ToolOutput:
                ok, text = await provider.call(_name, args)
                if not ok:
                    raise RuntimeError(text)
                return ToolOutput(text=text)

            specs[remote.name] = ToolSpec(
                name=remote.name,
                description=f"[Operations system via MCP] {remote.description}",
                permission=Permission.MCP,
                parameters=remote.input_schema,
                handler=call_remote,
                source="mcp",
            )
        return specs

    async def tools_for(self, principal: Principal) -> list[ToolSpec]:
        return [s for s in (await self._all_specs()).values() if principal.has(s.permission)]

    async def spec(self, name: str) -> ToolSpec | None:
        return (await self._all_specs()).get(name)

    async def _audit_event(
        self, principal: Principal, trace_id: str | None, name: str, outcome: str, **detail: Any
    ) -> None:
        await self._audit.record(
            AuditEvent(
                trace_id=trace_id,
                user_id=principal.user_id,
                role=principal.role.value,
                action="tool_call",
                target=name,
                outcome=outcome,
                detail=detail,
            )
        )

    async def execute(
        self,
        principal: Principal,
        name: str,
        arguments: dict[str, Any],
        *,
        trace_id: str | None = None,
        approved: bool = False,
    ) -> ToolResult:
        started = time.perf_counter()

        def fail(error: str) -> ToolResult:
            return ToolResult(tool=name, ok=False, error=error, latency_ms=_ms(started))

        spec = await self.spec(name)
        if spec is None:
            await self._audit_event(principal, trace_id, name, "denied_unknown_tool")
            return fail(f"Unknown tool '{name}'.")
        if not principal.has(spec.permission):
            await self._audit_event(principal, trace_id, name, "denied_rbac", required=spec.permission.value)
            return fail(f"Permission denied: role '{principal.role.value}' cannot use '{name}'.")
        if spec.requires_approval and not approved:
            await self._audit_event(principal, trace_id, name, "denied_not_approved")
            return fail(f"'{name}' requires human approval, which was not granted.")
        if not isinstance(arguments, dict) or len(json.dumps(arguments, default=str)) > MAX_ARGS_BYTES:
            return fail("Tool arguments must be a JSON object under 16KB.")
        if spec.args_model is not None:
            try:
                arguments = spec.args_model.model_validate(arguments).model_dump(mode="json")
            except ValidationError as exc:
                await self._audit_event(principal, trace_id, name, "rejected_invalid_args")
                return fail(f"Invalid arguments: {exc.errors(include_url=False, include_input=False)}")

        try:
            async with asyncio.timeout(spec.timeout_seconds):
                output = await spec.handler(principal, arguments)
        except TimeoutError:
            await self._audit_event(principal, trace_id, name, "timeout")
            return fail(f"'{name}' timed out after {spec.timeout_seconds}s.")
        except Exception as exc:
            logger.warning("tool failed", extra={"tool": name}, exc_info=True)
            await self._audit_event(principal, trace_id, name, "failed", error=type(exc).__name__)
            return fail(f"'{name}' failed: {str(exc)[:300]}")

        await self._audit_event(principal, trace_id, name, "succeeded", args=arguments)
        return ToolResult(
            tool=name,
            ok=True,
            output=output.text[:MAX_OUTPUT_CHARS],
            documents=output.documents,
            latency_ms=_ms(started),
        )


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)

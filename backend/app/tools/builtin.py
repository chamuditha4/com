"""Built-in tools: knowledge search (all roles), Python analysis (Analyst+), admin tools."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from app.auth.models import Permission, Principal
from app.retrieval.models import DocumentType, SearchFilters
from app.retrieval.service import HybridRetriever
from app.tools.audit import AuditLog
from app.tools.python_analysis import PythonAnalysisArgs, PythonSandbox
from app.tools.registry import ToolOutput, ToolSpec


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="What to search for.")
    departments: list[str] | None = Field(default=None, description="Optional department slugs.")
    document_types: list[DocumentType] | None = Field(default=None, description="Optional document types.")


class AuditLogArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)


class ReindexArgs(BaseModel):
    reason: str = Field(min_length=5, max_length=300, description="Why the knowledge base must be re-indexed.")


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def build_builtin_tools(
    *,
    retriever: HybridRetriever,
    sandbox: PythonSandbox,
    audit: AuditLog,
    reindex: Callable[[], Awaitable[dict[str, Any]]] | None,
) -> list[ToolSpec]:
    async def knowledge_search(principal: Principal, args: dict[str, Any]) -> ToolOutput:
        parsed = KnowledgeSearchArgs.model_validate(args)
        # The principal comes from the verified token, not from the tool arguments.
        result = await retriever.search(
            parsed.query,
            principal=principal,
            filters=SearchFilters(departments=parsed.departments, document_types=parsed.document_types),
        )
        lines = [f"- {c.attribution} (score {c.score:.3f})" for c in result.chunks]
        text = f"{len(result.chunks)} passages found:\n" + "\n".join(lines) if lines else "No passages found."
        return ToolOutput(text=text, documents=result.chunks)

    async def python_analysis(_: Principal, args: dict[str, Any]) -> ToolOutput:
        outcome = await sandbox.run(PythonAnalysisArgs.model_validate(args))
        if outcome.get("error"):
            raise RuntimeError(outcome["error"])
        return ToolOutput(text=json.dumps({"stdout": outcome["stdout"], "result": outcome["result"]}))

    async def audit_log(_: Principal, args: dict[str, Any]) -> ToolOutput:
        events = await audit.recent(AuditLogArgs.model_validate(args).limit)
        return ToolOutput(text=json.dumps([e.model_dump() for e in events], default=str))

    async def reindex_kb(_: Principal, args: dict[str, Any]) -> ToolOutput:
        if reindex is None:
            return ToolOutput(text="Re-indexing is performed by the offline ingestion job (data/ingest.py) in this deployment; a request has been logged.")
        return ToolOutput(text=json.dumps(await reindex()))

    return [
        ToolSpec(
            name="knowledge_search",
            description="Search Commercial Bank's knowledge base (policies, incidents, runbooks, architecture, specs, meeting notes).",
            permission=Permission.SEARCH,
            parameters=_schema(KnowledgeSearchArgs),
            args_model=KnowledgeSearchArgs,
            handler=knowledge_search,
        ),
        ToolSpec(
            name="python_analysis",
            description="Run sandboxed Python for calculations over data you already have (sums, averages, percentages, trends). No file, network or OS access.",
            permission=Permission.ANALYTICS,
            parameters=_schema(PythonAnalysisArgs),
            args_model=PythonAnalysisArgs,
            handler=python_analysis,
            timeout_seconds=10.0,
        ),
        ToolSpec(
            name="admin_audit_log",
            description="Administrators only: view recent tool-execution and access audit events.",
            permission=Permission.ADMIN,
            parameters=_schema(AuditLogArgs),
            args_model=AuditLogArgs,
            handler=audit_log,
        ),
        ToolSpec(
            name="admin_reindex_knowledge_base",
            description="Administrators only: rebuild the knowledge-base index. Sensitive; requires human approval.",
            permission=Permission.ADMIN,
            parameters=_schema(ReindexArgs),
            args_model=ReindexArgs,
            handler=reindex_kb,
            requires_approval=True,
            timeout_seconds=120.0,
        ),
    ]

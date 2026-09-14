import json

import pytest

from app.tools.audit import InMemoryAuditLog
from app.tools.builtin import build_builtin_tools
from app.tools.mcp_client import MCPToolProvider
from app.tools.python_analysis import CodeValidationError, PythonAnalysisArgs, PythonSandbox, validate_code
from app.tools.registry import ToolRegistry
from mcp_server.server import server as mcp_server


@pytest.fixture
def audit():
    return InMemoryAuditLog()


@pytest.fixture
def registry(retriever, audit):
    async def fake_reindex():
        return {"documents": 31}

    return ToolRegistry(
        builtin=build_builtin_tools(
            retriever=retriever, sandbox=PythonSandbox(timeout_seconds=5), audit=audit, reindex=fake_reindex
        ),
        audit=audit,
        mcp=MCPToolProvider(mcp_server),
    )


async def _names(registry, principal):
    return {t.name for t in await registry.tools_for(principal)}


async def test_tools_offered_to_each_role_follow_rbac_matrix(registry, viewer, analyst, admin):
    viewer_tools = await _names(registry, viewer)
    analyst_tools = await _names(registry, analyst)
    admin_tools = await _names(registry, admin)

    assert viewer_tools == {"knowledge_search"}
    assert {"python_analysis", "mcp_search_incident_tickets", "mcp_get_service_status"} <= analyst_tools
    assert not any(n.startswith("admin_") for n in analyst_tools)
    assert {"admin_audit_log", "admin_reindex_knowledge_base"} <= admin_tools


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("python_analysis", {"code": "result = 1"}),
        ("mcp_get_service_status", {}),
        ("admin_audit_log", {}),
    ],
)
async def test_viewer_is_denied_at_execution_boundary_even_if_llm_guesses_tool(registry, viewer, audit, tool, args):
    result = await registry.execute(viewer, tool, args)
    assert not result.ok and "Permission denied" in result.error
    assert (await audit.recent(1))[0].outcome == "denied_rbac"


async def test_sensitive_admin_tool_requires_approval(registry, admin):
    denied = await registry.execute(admin, "admin_reindex_knowledge_base", {"reason": "corpus refresh"})
    assert not denied.ok and "approval" in denied.error

    approved = await registry.execute(
        admin, "admin_reindex_knowledge_base", {"reason": "corpus refresh"}, approved=True
    )
    assert approved.ok and json.loads(approved.output) == {"documents": 31}


async def test_invalid_arguments_are_rejected_before_execution(registry, analyst):
    result = await registry.execute(analyst, "knowledge_search", {"query": "x" * 5000})
    assert not result.ok and result.error.startswith("Invalid arguments")


async def test_knowledge_search_tool_uses_callers_clearance(registry, viewer):
    result = await registry.execute(viewer, "knowledge_search", {"query": "FX routing feature flag misconfiguration"})
    assert result.ok
    assert all(d.chunk.metadata.access_level in ("public", "internal") for d in result.documents)


async def test_mcp_tool_roundtrip_for_analyst(registry, analyst):
    result = await registry.execute(analyst, "mcp_search_incident_tickets", {"status": "open"})
    assert result.ok
    payload = json.loads(result.output)
    assert payload["count"] == 2 and payload["tickets"][0]["status"] == "open"


async def test_unreachable_mcp_server_degrades_to_no_tools(analyst, audit, retriever):
    registry = ToolRegistry(
        builtin=[], audit=audit, mcp=MCPToolProvider("http://127.0.0.1:9/mcp", timeout_seconds=1)
    )
    assert await registry.tools_for(analyst) == []


async def test_python_analysis_computes_over_data(registry, analyst):
    result = await registry.execute(
        analyst,
        "python_analysis",
        {"code": "import statistics\nresult = round(statistics.mean(data), 1)", "data": [47, 112, 130]},
    )
    assert result.ok and json.loads(result.output)["result"] == "96.3"


@pytest.mark.parametrize(
    "code",
    [
        "import os\nos.system('id')",
        "import subprocess",
        "().__class__.__base__.__subclasses__()",
        "open('/etc/passwd').read()",
        "getattr(__builtins__, 'eval')('1')",
        "eval('1+1')",
        "x = '__import__'",
    ],
)
def test_sandbox_rejects_escape_attempts_statically(code):
    with pytest.raises(CodeValidationError):
        validate_code(code)


async def test_sandbox_enforces_timeout_and_memory_limits():
    sandbox = PythonSandbox(timeout_seconds=2, memory_mb=128)
    assert "timed out" in (await sandbox.run(PythonAnalysisArgs(code="while True:\n    pass")))["error"]
    assert "MemoryError" in (await sandbox.run(PythonAnalysisArgs(code="x = [0] * (10**9)")))["error"]

"""Sandboxed Python analysis tool (Analyst+).

Layered defenses, each assuming the previous one can fail:
1. **Static validation (AST allow-list).** No imports outside a small stdlib allow-list, no
   dunder attribute access (blocks `().__class__.__subclasses__()` escapes), no
   eval/exec/open/getattr, size limits.
2. **Restricted runtime.** Whitelisted builtins and import function inside the child.
3. **Process isolation.** Separate interpreter (`-I -S`), empty environment, CPU/memory/file/
   process rlimits, wall-clock timeout with kill, truncated output.

SCALE-DEBT / security note: there is no OS-level network isolation. Production should run
the child in a network-less sandbox (gVisor/Firecracker or a locked-down sidecar container).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)

_RUNNER = Path(__file__).with_name("sandbox_runner.py")
ALLOWED_IMPORTS = frozenset({"math", "statistics", "collections", "datetime", "json", "re", "itertools"})
BANNED_NAMES = frozenset(
    {
        "eval", "exec", "compile", "open", "input", "globals", "locals", "vars", "getattr",
        "setattr", "delattr", "__import__", "breakpoint", "exit", "quit", "help", "memoryview",
        "type", "object", "super", "classmethod", "staticmethod", "property",
    }
)
MAX_CODE_CHARS = 5000
MAX_AST_NODES = 2500


class PythonAnalysisArgs(BaseModel):
    code: str = Field(
        min_length=1,
        max_length=MAX_CODE_CHARS,
        description=(
            "Python 3 code for numeric/tabular analysis. Input is available as `data`. "
            "Assign the answer to `result` or print it. Allowed imports: "
            + ", ".join(sorted(ALLOWED_IMPORTS))
        ),
    )
    data: Any = Field(default=None, description="JSON-serializable input data for the analysis.")


class CodeValidationError(ValueError):
    pass


def validate_code(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CodeValidationError(f"syntax error: {exc.msg} (line {exc.lineno})") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise CodeValidationError("code is too complex")

    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise CodeValidationError(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise CodeValidationError(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise CodeValidationError(f"access to private attribute '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            raise CodeValidationError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise CodeValidationError("async code is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            raise CodeValidationError("dunder strings are not allowed")


class PythonSandbox:
    def __init__(self, *, timeout_seconds: float = 5.0, memory_mb: int = 256) -> None:
        self._timeout = timeout_seconds
        self._memory_mb = memory_mb

    async def run(self, args: PythonAnalysisArgs) -> dict[str, Any]:
        validate_code(args.code)
        payload = json.dumps(
            {
                "code": args.code,
                "data": args.data,
                "memory_mb": self._memory_mb,
                # CPU limit sits above the wall-clock timeout: the timeout is the primary control,
                # the rlimit is the backstop if the parent is somehow unable to kill the child.
                "cpu_seconds": int(self._timeout) + 2,
            },
            default=str,
        ).encode()

        process = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-S", str(_RUNNER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={},
            cwd="/",
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(payload), self._timeout)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):  # it may have just exited on its own
                process.kill()
            await process.wait()
            return {"stdout": "", "result": None, "error": f"timed out after {self._timeout}s"}

        if process.returncode != 0 and not stdout:
            logger.warning("sandbox crashed", extra={"rc": process.returncode, "stderr": stderr[-300:]})
            return {"stdout": "", "result": None, "error": "analysis process terminated (resource limit)"}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return {"stdout": "", "result": None, "error": "analysis produced invalid output"}

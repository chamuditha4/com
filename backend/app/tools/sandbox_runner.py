"""Child-process entry point for the Python analysis tool. Never imported by the server.

Runs as `python -I -S sandbox_runner.py` with an empty environment. It applies resource
limits to itself *before* touching untrusted code, then executes the (already AST-validated)
code with a whitelisted builtins table and import function.

Protocol: stdin = JSON {"code": str, "data": any, "memory_mb": int, "cpu_seconds": int}
          stdout = JSON {"stdout": str, "result": str | null, "error": str | null}
"""

import contextlib
import io
import json
import resource
import sys

ALLOWED_MODULES = ("math", "statistics", "collections", "datetime", "json", "re", "itertools")
SAFE_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "ValueError",
    "TypeError",
    "KeyError",
    "ZeroDivisionError",
    "Exception",
)
MAX_OUTPUT = 8000


def _limit(memory_mb: int, cpu_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    memory = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))  # no fork/exec


def main() -> None:
    request = json.loads(sys.stdin.read())
    modules = {name: __import__(name) for name in ALLOWED_MODULES}
    _limit(int(request.get("memory_mb", 256)), int(request.get("cpu_seconds", 5)))

    import builtins

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in modules and level == 0:
            return modules[name]
        raise ImportError(f"import of '{name}' is not allowed")

    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    safe["__import__"] = restricted_import
    scope = {"__builtins__": safe, "data": request.get("data")}

    stdout = io.StringIO()
    response = {"stdout": "", "result": None, "error": None}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(compile(request["code"], "<analysis>", "exec"), scope)  # noqa: S102 - sandboxed
        if "result" in scope:
            response["result"] = repr(scope["result"])[:MAX_OUTPUT]
    except BaseException as exc:  # report every failure (including MemoryError) as data
        response["error"] = f"{type(exc).__name__}: {exc}"[:500]
    response["stdout"] = stdout.getvalue()[:MAX_OUTPUT]
    sys.__stdout__.write(json.dumps(response))


if __name__ == "__main__":
    main()

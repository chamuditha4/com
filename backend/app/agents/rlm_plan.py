"""The RLM "Python-based search plan": generation, safe interpretation and fallback planning.

The research planner (LLM) writes its plan as Python source:

    PLAN = [
        batch(label="2025-Q4", query="payment outage root cause", departments=["payments"],
              document_types=["incident"], date_from="2025-10-01", date_to="2025-12-31"),
        ...
    ]

We never `exec` model-written code. `parse_plan` walks the AST and accepts exactly one shape:
a single assignment of a list of `batch(...)` calls whose keyword arguments are literals.
Anything else (imports, attribute access, other calls, comprehensions) is rejected, and
the agent falls back to `heuristic_plan`. So the plan is readable Python a human can audit in
the trace, and it is interpreted like data.
"""

from __future__ import annotations

import ast
import re
from datetime import date, timedelta

from pydantic import ValidationError

from app.agents.heuristics import infer_departments, infer_document_types, resolve_time_window
from app.agents.state import ResearchBatch, RouteDecision
from app.retrieval.catalog import CatalogOverview
from app.retrieval.models import DocumentType, SearchFilters

_ALLOWED_KEYS = {"label", "query", "departments", "document_types", "date_from", "date_to"}
_FENCE = re.compile(r"^```(?:python)?\s*|\s*```\s*$", re.MULTILINE)


class PlanValidationError(ValueError):
    pass


def parse_plan(source: str, *, known_departments: list[str], max_batches: int) -> tuple[list[ResearchBatch], list[str]]:
    source = _FENCE.sub("", source).strip()
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise PlanValidationError(f"plan is not valid Python: {exc.msg}") from exc

    statements = [s for s in tree.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if (
        len(statements) != 1
        or not isinstance(statements[0], ast.Assign)
        or len(statements[0].targets) != 1
        or not isinstance(statements[0].targets[0], ast.Name)
        or statements[0].targets[0].id != "PLAN"
        or not isinstance(statements[0].value, ast.List)
    ):
        raise PlanValidationError("plan must be a single assignment: PLAN = [batch(...), ...]")

    warnings: list[str] = []
    batches: list[ResearchBatch] = []
    for index, element in enumerate(statements[0].value.elts):
        if not (isinstance(element, ast.Call) and isinstance(element.func, ast.Name) and element.func.id == "batch"):
            raise PlanValidationError(f"element {index} is not a batch(...) call")
        if element.args:
            raise PlanValidationError("batch() accepts keyword arguments only")
        kwargs = {}
        for keyword in element.keywords:
            if keyword.arg not in _ALLOWED_KEYS:
                raise PlanValidationError(f"unsupported batch() argument: {keyword.arg}")
            try:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            except ValueError as exc:
                raise PlanValidationError(f"argument {keyword.arg} must be a literal") from exc

        departments = kwargs.get("departments") or None
        if departments:
            unknown = [d for d in departments if d not in known_departments]
            if unknown:
                warnings.append(f"dropped unknown departments {unknown}")
            departments = [d for d in departments if d in known_departments] or None
        try:
            batches.append(
                ResearchBatch(
                    label=str(kwargs.get("label") or f"batch-{index + 1}"),
                    query=str(kwargs.get("query") or ""),
                    filters=SearchFilters(
                        departments=departments,
                        document_types=kwargs.get("document_types") or None,
                        date_from=kwargs.get("date_from"),
                        date_to=kwargs.get("date_to"),
                    ),
                )
            )
        except ValidationError as exc:
            raise PlanValidationError(f"batch {index} is invalid: {exc.errors(include_url=False)[0]['msg']}") from exc

    if not batches:
        raise PlanValidationError("plan has no batches")
    if any(not b.query for b in batches):
        raise PlanValidationError("every batch needs a query")
    if len(batches) > max_batches:
        warnings.append(f"plan truncated from {len(batches)} to {max_batches} batches (RLM_MAX_BATCHES)")
        batches = batches[:max_batches]
    return batches, warnings


def render_plan(batches: list[ResearchBatch], *, comment: str = "") -> str:
    """Render batches back to the canonical Python form (used for heuristic plans and for display)."""

    def literal(value: object) -> str:
        return repr(value).replace("'", '"')

    lines = [f"# {line}" for line in comment.splitlines() if line] + ["PLAN = ["]
    for b in batches:
        args = [f"label={literal(b.label)}", f"query={literal(b.query)}"]
        f = b.filters
        if f.departments:
            args.append(f"departments={literal(f.departments)}")
        if f.document_types:
            args.append(f"document_types={literal([t.value for t in f.document_types])}")
        if f.date_from:
            args.append(f'date_from="{f.date_from.isoformat()}"')
        if f.date_to:
            args.append(f'date_to="{f.date_to.isoformat()}"')
        lines.append(f"    batch({', '.join(args)}),")
    lines.append("]")
    return "\n".join(lines)


def _quarter_windows(start: date, end: date) -> list[tuple[str, date, date]]:
    windows = []
    cursor = start
    while cursor <= end:
        quarter_end_month = ((cursor.month - 1) // 3 + 1) * 3
        next_quarter = date(cursor.year + (quarter_end_month == 12), quarter_end_month % 12 + 1, 1)
        window_end = min(end, next_quarter - timedelta(days=1))
        windows.append((f"{cursor.year}-Q{(cursor.month - 1) // 3 + 1}", cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def heuristic_plan(
    *,
    question: str,
    route: RouteDecision,
    overview: CatalogOverview,
    known_departments: list[str],
    today: date,
    max_batches: int,
) -> list[ResearchBatch]:
    """Deterministic plan: resolve the window, then decompose it into calendar quarters."""
    window = (
        (route.date_from, route.date_to) if route.date_from and route.date_to else resolve_time_window(question, today)
    )
    if window is None:
        window = (overview.earliest or today - timedelta(days=365), overview.latest or today)
    departments = route.departments or infer_departments(question, known_departments) or None
    document_types = route.document_types or infer_document_types(question) or [DocumentType.INCIDENT]

    windows = _quarter_windows(window[0], window[1])
    while len(windows) > max_batches:  # merge adjacent windows until within the cap
        merged = []
        for i in range(0, len(windows), 2):
            pair = windows[i : i + 2]
            merged.append((f"{pair[0][0]}..{pair[-1][0]}", pair[0][1], pair[-1][2]))
        windows = merged

    return [
        ResearchBatch(
            label=label,
            query=route.search_query,
            filters=SearchFilters(departments=departments, document_types=document_types, date_from=start, date_to=end),
        )
        for label, start, end in windows
    ]

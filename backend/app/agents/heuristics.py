"""Deterministic fallbacks for LLM-driven decisions.

Used when the LLM is unavailable (offline mode, provider outage) or returns invalid output.
They are deliberately simple and transparent. Their job is to keep the product answering,
visibly marked as degraded, not to match LLM quality.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from app.agents.state import IncidentFinding, RouteDecision
from app.auth.models import Permission, Principal
from app.retrieval.models import DocumentType, RetrievedChunk

# --- routing ---------------------------------------------------------------------------------

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening)|what can you do|who are you|help)\b[\s!.?]*$",
    re.IGNORECASE,
)
_RESEARCH_SIGNALS = re.compile(
    r"\b(summari[sz]e|summary|trends?|recurring|patterns?|all|across|over the (last|past)|during the (last|past)|"
    r"last (year|quarter|\d+ months)|root causes|how many|compare|analy[sz]e|overview of)\b",
    re.IGNORECASE,
)
_TOOL_SIGNALS = re.compile(
    r"\b(tickets?|on-?call|service status|status of|currently|right now|live|metrics|volumes?|"
    r"calculate|compute|average|percentage|audit log|re-?index)\b",
    re.IGNORECASE,
)

_DOC_TYPE_KEYWORDS: tuple[tuple[DocumentType, re.Pattern[str]], ...] = (
    (DocumentType.INCIDENT, re.compile(r"\b(incidents?|outages?|post-?mortems?|failures?|disruptions?)\b", re.I)),
    (DocumentType.POLICY, re.compile(r"\b(polic(y|ies)|standards?)\b", re.I)),
    (DocumentType.RUNBOOK, re.compile(r"\b(runbooks?|playbooks?|how do i|procedure)\b", re.I)),
    (DocumentType.ARCHITECTURE, re.compile(r"\barchitecture\b", re.I)),
    (DocumentType.PRODUCT_SPEC, re.compile(r"\b(product spec|specification|product requirements?)\b", re.I)),
    (DocumentType.MEETING_NOTES, re.compile(r"\b(meeting|minutes|review board)\b", re.I)),
)
_DEPARTMENT_KEYWORDS: dict[str, re.Pattern[str]] = {
    "payments": re.compile(r"\b(payments?|cards?|swift|transfers?|ledger|wallet|acquirer)\b", re.I),
    "hr": re.compile(r"\b(hr|leave|compensation|salary bands?|remote work)\b", re.I),
    "compliance": re.compile(r"\b(aml|kyc|compliance|money laundering|classification)\b", re.I),
    "security": re.compile(r"\b(security|phishing|credential stuffing|cyber)\b", re.I),
}


def resolve_time_window(text: str, today: date) -> tuple[date, date] | None:
    lowered = text.lower()
    if re.search(r"\b(last|past|previous) (year|12 months|twelve months)\b", lowered):
        return today - timedelta(days=365), today
    if match := re.search(r"\b(last|past) (\d{1,2}) months\b", lowered):
        return today - timedelta(days=30 * int(match.group(2))), today
    if re.search(r"\b(last|past) quarter\b", lowered):
        return today - timedelta(days=91), today
    if match := re.search(r"\bin (20\d{2})\b", lowered):
        year = int(match.group(1))
        return date(year, 1, 1), min(date(year, 12, 31), today)
    return None


def infer_document_types(text: str) -> list[DocumentType]:
    return [doc_type for doc_type, pattern in _DOC_TYPE_KEYWORDS if pattern.search(text)]


def infer_departments(text: str, known: list[str]) -> list[str]:
    return [d for d, pattern in _DEPARTMENT_KEYWORDS.items() if d in known and pattern.search(text)]


def can_use_tools(principal: Principal) -> bool:
    return principal.has(Permission.ANALYTICS) or principal.has(Permission.MCP) or principal.has(Permission.ADMIN)


def route_heuristically(
    question: str, principal: Principal, known_departments: list[str], today: date
) -> RouteDecision:
    window = resolve_time_window(question, today)
    doc_types = infer_document_types(question)

    if _GREETING.match(question):
        strategy, rationale = "direct", "Greeting or question about the assistant."
    elif can_use_tools(principal) and _TOOL_SIGNALS.search(question):
        strategy, rationale = "tools", "Mentions live operational data or a calculation."
    elif len(_RESEARCH_SIGNALS.findall(question)) >= 2 or (window and DocumentType.INCIDENT in doc_types):
        strategy, rationale = "research", "Broad or aggregative question over many documents or a time range."
    else:
        strategy, rationale = "retrieval", "Focused question answerable from a few passages."

    return RouteDecision(
        strategy=strategy,
        intent=question[:200],
        search_query=question[:500],
        # Department scoping is only inferred for research; for a single lookup a wrong guess
        # would hide the right answer, and hybrid ranking handles relevance.
        departments=infer_departments(question, known_departments) if strategy == "research" else [],
        document_types=doc_types if strategy == "research" else doc_types[:1] if window else [],
        date_from=window[0] if window else None,
        date_to=window[1] if window else None,
        rationale=f"[heuristic] {rationale}",
    )


# --- research extraction ---------------------------------------------------------------------

ROOT_CAUSE_TAXONOMY: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Expired certificate", re.compile(r"certificate|\bcerts?\b|\b(m?tls|ssl)\b", re.I)),
    ("Database connection pool exhaustion", re.compile(r"connection pool|pool (exhaust|saturat)", re.I)),
    (
        "Third-party processor outage",
        re.compile(r"third-party|processor (outage|degradation)|acquir(er|ing partner)|vendor", re.I),
    ),
    ("Faulty configuration change", re.compile(r"configuration|feature flag|config\b", re.I)),
    ("Security attack", re.compile(r"credential stuffing|attack", re.I)),
    ("Infrastructure failure", re.compile(r"\bdns\b|resolver|cache node|failover|data-centre", re.I)),
    ("Software defect", re.compile(r"memory leak|defect|\bbug\b", re.I)),
)


def normalize_root_cause(category: str, summary: str = "") -> str:
    """Map free-text categories (LLM or heuristic) onto the taxonomy so tallies are comparable."""
    for text in (category, summary):
        for name, pattern in ROOT_CAUSE_TAXONOMY:
            if pattern.search(text):
                return name
    return category.strip().capitalize() or "Unclassified"


def _first_sentences(text: str, n: int) -> str:
    body = text.split("\n\n", 1)[-1]  # drop the "title — section" prefix added at chunking
    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    return " ".join(sentences[:n])[:600]


def heuristic_incident_findings(chunks: list[RetrievedChunk]) -> list[IncidentFinding]:
    by_doc: dict[str, list[RetrievedChunk]] = {}
    for c in chunks:
        if c.chunk.metadata.document_type == DocumentType.INCIDENT:
            by_doc.setdefault(c.chunk.doc_id, []).append(c)

    findings = []
    for doc_id, doc_chunks in by_doc.items():
        root = next((c for c in doc_chunks if "root cause" in c.chunk.section.lower()), None)
        impact = next((c for c in doc_chunks if "impact" in c.chunk.section.lower()), None)
        if root is None:
            continue  # without the root-cause section we cannot ground a finding
        md = root.chunk.metadata
        summary = _first_sentences(root.chunk.text, 2)
        findings.append(
            IncidentFinding(
                doc_id=doc_id,
                title=md.title,
                date=md.created_date.isoformat(),
                root_cause_category=normalize_root_cause(root.chunk.text),
                root_cause_summary=summary,
                impact_summary=_first_sentences(impact.chunk.text, 1)[:400] if impact else "",
                chunk_ids=[c.chunk.chunk_id for c in (root, impact) if c is not None],
            )
        )
    return findings

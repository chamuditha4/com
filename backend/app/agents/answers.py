"""Deterministic answer templates.

Used when the LLM is unavailable, when evidence is missing, and as the last line of defense
when a generated answer fails validation. Every extractive answer cites only evidence that
exists, so it passes the validator by construction.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.agents.prompts import INSUFFICIENT_EVIDENCE_PHRASE
from app.agents.state import Evidence, ResearchReport
from app.auth.models import Permission, Principal

RETRIEVAL_UNAVAILABLE = (
    "The knowledge base is temporarily unavailable, so I can't answer this reliably right now. "
    "Please try again in a few minutes."
)
BLOCKED = (
    "I can't help with that request because it asks me to bypass my safety rules or reveal protected "
    "information. If you have a question about Commercial Bank's knowledge sources, please rephrase it."
)
SAFE_APOLOGY = (
    "I wasn't able to produce an answer that meets Commercial Bank's accuracy and safety checks. "
    "Please rephrase your question or try again."
)
DEGRADED_NOTE = "_Language model unavailable: this is an extractive summary of the most relevant sources._"
REPLACED_NOTE = (
    "_The generated answer did not pass validation, so this is an extractive summary of the sources instead._"
)


def insufficient_evidence() -> str:
    return (
        f"{INSUFFICIENT_EVIDENCE_PHRASE}. Try rephrasing the question, or check with the team that owns "
        "the relevant documentation."
    )


def direct_answer(principal: Principal) -> str:
    abilities = [
        "answer questions from Commercial Bank's policies, runbooks, incident reports, architecture documents, product specs and meeting notes, with citations"
    ]
    if principal.has(Permission.ANALYTICS) or principal.has(Permission.MCP):
        abilities.append(
            "look up live operational data (tickets, service status, on-call rosters, payment metrics) and run calculations"
        )
    if principal.has(Permission.ADMIN):
        abilities.append("run administrative tools such as audit-log review and re-indexing (with approval)")
    return (
        f"Hello {principal.display_name.split()[0]}. I can " + "; ".join(abilities) + ". What would you like to know?"
    )


def _sentences(text: str, n: int, max_chars: int = 400) -> str:
    body = text.split("\n\n", 1)[-1]
    body = re.sub(r"\s+", " ", body).strip()
    parts = re.split(r"(?<=[.!?])\s+", body)
    return " ".join(parts[:n])[:max_chars]


def _research_answer(report: ResearchReport, evidence: Sequence[Evidence]) -> str:
    by_chunk = {e.chunk_id: e.id for e in evidence if e.chunk_id}
    incidents = report.incidents
    lines = [
        f"**{len(incidents)} relevant incidents** were found between {incidents[0].date} and {incidents[-1].date}, "
        f"from {report.documents_in_scope} documents analysed in {report.batches_analyzed} batches.",
        "",
    ]
    if report.recurring:
        lines.append("**Recurring root causes**")
        for tally in report.recurring:
            refs = ", ".join(str(i) for i in tally.evidence_ids[:6])
            lines.append(f"- **{tally.category}**: {tally.count} incidents ({', '.join(tally.doc_ids)}) [{refs}]")
        lines.append("")
    one_offs = [t for t in report.root_causes if t.count < 2]
    if one_offs:
        lines.append("**Other root causes**")
        for tally in one_offs:
            refs = ", ".join(str(i) for i in tally.evidence_ids[:3])
            lines.append(f"- {tally.category}: {', '.join(tally.doc_ids)} [{refs}]")
        lines.append("")
    lines.append("**Incidents**")
    for inc in incidents:
        ids = [by_chunk[c] for c in inc.chunk_ids if c in by_chunk]
        cite = f" [{', '.join(str(i) for i in ids)}]" if ids else ""
        lines.append(f"- {inc.date}, {inc.title}: {_sentences(inc.root_cause_summary, 1, 300)}{cite}")
    if report.failed_batches:
        lines += [
            "",
            f"Note: some batches could not be analysed ({', '.join(report.failed_batches)}), so coverage may be incomplete.",
        ]
    return "\n".join(lines)


def extractive_answer(evidence: Sequence[Evidence], report: ResearchReport | None, note: str = DEGRADED_NOTE) -> str:
    if report is not None and report.incidents:
        body = _research_answer(report, evidence)
    else:
        top = [e for e in evidence][:3]
        if not top:
            return insufficient_evidence()
        bullets = [f"- **{e.label}**: {_sentences(e.text, 2)} [{e.id}]" for e in top]
        body = "Here is what the most relevant sources say:\n\n" + "\n".join(bullets)
    return f"{body}\n\n{note}"

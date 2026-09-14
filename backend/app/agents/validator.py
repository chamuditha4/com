"""Validator: post-generation guardrails, and Finalizer: the only way an answer leaves the graph.

The Validator blocks:
* hallucinated citations: `[n]` that does not match evidence retrieved in this turn;
* ungrounded answers: factual strategy, evidence available, but no citations;
* system-prompt leakage (per-invocation canary token);
* role violations: citing evidence above the caller's clearance;
* brand-safety violations, and empty or oversized answers.

Card numbers and secrets are redacted rather than rejected. When validation fails and a retry
is allowed, the Response Agent regenerates with the issues as feedback. Otherwise the Finalizer
substitutes a deterministic extractive answer (valid by construction) or a safe apology, so a
bad generation is contained rather than cascading to the user.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from app.agents.answers import BLOCKED, REPLACED_NOTE, SAFE_APOLOGY, extractive_answer
from app.agents.context import AgentContext
from app.agents.prompts import INSUFFICIENT_EVIDENCE_PHRASE
from app.agents.response_agent import GROUNDED_STRATEGIES
from app.agents.state import AgentState, Evidence, FinalAnswer, ValidationIssue, ValidationReport
from app.auth.models import Principal
from app.guardrails.output import brand_safety_violations, extract_citations, redact_sensitive
from app.observability.events import emit

MAX_ANSWER_CHARS = 12_000


def validate_answer(
    draft: str,
    *,
    evidence: Sequence[Evidence],
    strategy: str,
    retrieval_status: str,
    principal: Principal,
    canary: str,
) -> ValidationReport:
    issues: list[ValidationIssue] = []
    citations = extract_citations(draft)
    valid_ids = {e.id for e in evidence}

    if not draft.strip():
        issues.append(ValidationIssue(code="empty_answer", message="The answer is empty.", retryable=True))
    if invalid := [c for c in citations if c not in valid_ids]:
        issues.append(
            ValidationIssue(
                code="hallucinated_citation",
                message=f"Citations {invalid} do not correspond to any provided evidence. Valid ids: {sorted(valid_ids) or 'none'}.",
                retryable=True,
            )
        )
    declined = INSUFFICIENT_EVIDENCE_PHRASE.lower() in draft.lower()
    if strategy in GROUNDED_STRATEGIES and retrieval_status == "ok" and evidence and not citations and not declined:
        issues.append(
            ValidationIssue(
                code="missing_citations",
                message="The answer makes claims without citing evidence with [n].",
                retryable=True,
            )
        )
    if canary and canary in draft:
        issues.append(
            ValidationIssue(
                code="system_prompt_leak",
                message="The answer reveals confidential system instructions.",
                retryable=False,
            )
        )
    for e in evidence:
        if e.id in citations and e.access_level and not principal.can_read(e.access_level):
            issues.append(
                ValidationIssue(
                    code="access_violation",
                    message=f"Evidence [{e.id}] is above the caller's clearance.",
                    retryable=False,
                )
            )
    for violation in brand_safety_violations(draft):
        issues.append(
            ValidationIssue(
                code="brand_safety",
                message=f"Remove content that violates brand-safety rule '{violation}'.",
                retryable=True,
            )
        )
    if len(draft) > MAX_ANSWER_CHARS:
        issues.append(
            ValidationIssue(
                code="too_long", message=f"Keep the answer under {MAX_ANSWER_CHARS} characters.", retryable=True
            )
        )

    return ValidationReport(
        passed=not issues,
        issues=issues,
        citations=[c for c in citations if c in valid_ids],
        redactions=redact_sensitive(draft).kinds,
    )


async def validator(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    route = state.get("route")
    report = validate_answer(
        state.get("draft") or "",
        evidence=state.get("evidence", []),
        strategy=route.strategy if route else "direct",
        retrieval_status=state.get("retrieval_status", "not_run"),
        principal=ctx.principal,
        canary=ctx.canary,
    )
    attempts = state.get("attempts", 0) + (0 if report.passed else 1)
    report.will_retry = (
        not report.passed
        and report.retryable
        and ctx.services.llm.available
        and attempts <= ctx.services.settings.agent_max_validation_retries
    )
    if report.passed:
        message = f"Validation passed: {len(report.citations)} citations verified against retrieved evidence"
    else:
        message = f"Validation failed ({', '.join(i.code for i in report.issues)}); " + (
            "retrying generation" if report.will_retry else "substituting a safe answer"
        )
    emit(runtime, "validator", "validation", message, **report.model_dump(mode="json"))
    return {"validation": report, "attempts": attempts}


def route_after_validation(state: AgentState) -> str:
    report = state.get("validation")
    return "response_agent" if report is not None and report.will_retry else "finalize"


async def finalize(state: AgentState, runtime: Runtime[AgentContext]) -> dict[str, Any]:
    ctx = runtime.context
    route = state.get("route")
    evidence = state.get("evidence", [])
    guard = state.get("input_guard")
    notes: list[str] = []

    if guard is not None and not guard.allowed:
        final = FinalAnswer(
            answer=BLOCKED, strategy="blocked", validation_passed=True, notes=["input blocked by guardrail"]
        )
    else:
        strategy = route.strategy if route else "direct"
        report = state.get("validation")
        answer = state.get("draft") or ""
        passed = bool(report and report.passed)
        if not passed:
            candidate = (
                extractive_answer(evidence, state.get("research_report"), REPLACED_NOTE) if evidence else SAFE_APOLOGY
            )
            recheck = validate_answer(
                candidate,
                evidence=evidence,
                strategy=strategy,
                retrieval_status=state.get("retrieval_status", "not_run"),
                principal=ctx.principal,
                canary=ctx.canary,
            )
            answer = candidate if recheck.passed else SAFE_APOLOGY
            notes.append("generated answer failed validation and was replaced")
        redaction = redact_sensitive(answer)
        if redaction.kinds:
            notes.append(f"redacted: {', '.join(redaction.kinds)}")
        cited = set(extract_citations(redaction.text))
        final = FinalAnswer(
            answer=redaction.text,
            strategy=strategy,
            citations=[e for e in evidence if e.id in cited],
            validation_passed=passed,
            degraded=state.get("degraded", []),
            notes=notes,
        )

    emit(
        runtime,
        "finalize",
        "final",
        "Final answer ready",
        strategy=final.strategy,
        citations=len(final.citations),
        validation_passed=final.validation_passed,
        degraded=final.degraded,
        notes=final.notes,
    )
    return {"final": final, "messages": [AIMessage(content=final.answer)]}

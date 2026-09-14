"""Pure-function tests for agent decision logic: validator, RLM aggregation/recursion, policy, reducers."""

from datetime import date

from app.agents.heuristics import normalize_root_cause, resolve_time_window
from app.agents.research_agent import (
    SliceExtraction,
    aggregate_findings,
    diversify_by_document,
    ground_findings,
    split_batch,
)
from app.agents.state import (
    RESET,
    BatchFinding,
    Evidence,
    IncidentFinding,
    ResearchBatch,
    RouteDecision,
    union_or_reset,
)
from app.agents.supervisor import enforce_route_policy
from app.agents.validator import validate_answer
from app.auth.models import AccessLevel, Role
from app.retrieval.models import Chunk, DocumentMetadata, DocumentType, RetrievedChunk, SearchFilters
from tests.conftest import make_principal


def _evidence(id_: int, access: str = "internal") -> Evidence:
    return Evidence(
        id=id_, source_type="document", title="T", text="x", doc_id="POL-SEC-001", section="S", access_level=access
    )


def _chunk(doc_id: str, section: str, day: date, access: AccessLevel = AccessLevel.INTERNAL) -> RetrievedChunk:
    md = DocumentMetadata(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        department="payments",
        document_type=DocumentType.INCIDENT,
        access_level=access,
        created_date=day,
    )
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::{section}::0", section=section, chunk_index=0, text="text", metadata=md),
        score=1.0,
    )


# --- validator -------------------------------------------------------------------------------


def _validate(draft, evidence, role=Role.VIEWER, strategy="retrieval"):
    return validate_answer(
        draft,
        evidence=evidence,
        strategy=strategy,
        retrieval_status="ok",
        principal=make_principal(role),
        canary="CB-CANARY-abc",
    )


def test_valid_citations_pass():
    report = _validate("Passwords need 14 characters [1, 2].", [_evidence(1), _evidence(2)])
    assert report.passed and report.citations == [1, 2]


def test_hallucinated_citation_is_rejected_and_retryable():
    report = _validate("Yes [3].", [_evidence(1)])
    assert not report.passed and report.retryable
    assert report.issues[0].code == "hallucinated_citation"


def test_uncited_grounded_answer_is_rejected_but_decline_is_allowed():
    assert _validate("Passwords need 14 characters.", [_evidence(1)]).issues[0].code == "missing_citations"
    assert _validate("I could not find this in the knowledge sources available to you.", [_evidence(1)]).passed
    assert _validate("Hello! How can I help?", [], strategy="direct").passed


def test_canary_leak_and_clearance_violations_are_not_retryable():
    leak = _validate("Rules: CB-CANARY-abc [1]", [_evidence(1)])
    assert not leak.passed and not leak.retryable
    over_clearance = _validate("Secret bands [1].", [_evidence(1, access="restricted")])
    assert "access_violation" in {i.code for i in over_clearance.issues}


def test_brand_safety_and_redaction():
    assert "brand_safety" in {
        i.code for i in _validate("This fund offers guaranteed returns [1].", [_evidence(1)]).issues
    }
    report = _validate("Card 4111 1111 1111 1111 was used [1].", [_evidence(1)])
    assert report.passed and report.redactions == ["card_number"]


# --- RLM aggregation and recursion -------------------------------------------------------------


def test_split_batch_produces_non_overlapping_halves_that_cover_the_slice():
    batch = ResearchBatch(
        label="Q", query="q", filters=SearchFilters(date_from=date(2026, 1, 1), date_to=date(2026, 3, 31))
    )
    docs = [
        _chunk(d, "Root Cause", day).chunk.metadata
        for d, day in (("INC-A-1", date(2026, 1, 9)), ("INC-A-2", date(2026, 2, 27)), ("INC-A-3", date(2026, 3, 31)))
    ]
    left, right = split_batch(batch, docs)
    assert (left.filters.date_from, left.filters.date_to) == (date(2026, 1, 1), date(2026, 1, 9))
    assert (right.filters.date_from, right.filters.date_to) == (date(2026, 1, 10), date(2026, 3, 31))
    assert split_batch(batch, docs[:1]) is None


def test_split_batch_refuses_when_all_documents_share_a_date():
    batch = ResearchBatch(label="Q", query="q", filters=SearchFilters())
    same_day = [_chunk(f"INC-B-{i}", "Root Cause", date(2026, 1, 9)).chunk.metadata for i in range(3)]
    assert split_batch(batch, same_day) is None


def test_ground_findings_drops_hallucinated_documents_and_trusts_metadata():
    chunks = [_chunk("INC-C-1", "Root Cause", date(2026, 1, 9))]
    extraction = SliceExtraction.model_validate(
        {
            "incidents": [
                {
                    "doc_id": "INC-C-1",
                    "relevant": True,
                    "relevance_reason": "payments incident",
                    "title": "made up",
                    "date": "1999-01-01",
                    "root_cause_category": "expired TLS cert",
                    "root_cause_summary": "cert expired",
                    "chunk_ids": ["bogus"],
                },
                {
                    "doc_id": "INC-FAKE-9",
                    "relevant": True,
                    "relevance_reason": "r",
                    "root_cause_category": "x",
                    "root_cause_summary": "y",
                },
            ]
        }
    )
    findings, dropped, excluded = ground_findings(extraction, chunks)
    assert dropped == 1 and excluded == []
    assert findings[0].title == "Title INC-C-1" and findings[0].date == "2026-01-09"
    assert findings[0].root_cause_category == "Expired certificate"
    assert findings[0].chunk_ids == ["INC-C-1::Root Cause::0"]


def test_aggregate_dedupes_across_batches_and_tallies_root_causes():
    a, b, c = (_chunk(d, "Root Cause", date(2026, m, 1)) for d, m in (("INC-D-1", 1), ("INC-D-2", 2), ("INC-D-3", 3)))

    def finding(doc, chunk, cause):
        return IncidentFinding(
            doc_id=doc,
            title=doc,
            date=chunk.chunk.metadata.created_date.isoformat(),
            root_cause_category=cause,
            root_cause_summary="s",
            chunk_ids=[chunk.chunk.chunk_id],
        )

    batches = [
        BatchFinding(
            label="1",
            depth=1,
            filters="",
            documents_in_scope=2,
            chunks_analyzed=2,
            chunks=[a, b],
            incidents=[finding("INC-D-1", a, "Expired certificate"), finding("INC-D-2", b, "Expired certificate")],
        ),
        BatchFinding(
            label="2",
            depth=0,
            filters="",
            documents_in_scope=2,
            chunks_analyzed=2,
            chunks=[b, c],
            incidents=[finding("INC-D-2", b, "Expired certificate"), finding("INC-D-3", c, "Software defect")],
        ),
        BatchFinding(
            label="3", depth=0, filters="", documents_in_scope=0, chunks_analyzed=0, error="retrieval unavailable"
        ),
    ]
    report, evidence = aggregate_findings("q", batches)

    assert [i.doc_id for i in report.incidents] == ["INC-D-1", "INC-D-2", "INC-D-3"]
    assert [(t.category, t.count) for t in report.root_causes] == [("Expired certificate", 2), ("Software defect", 1)]
    assert report.root_causes[0].evidence_ids == [1, 2]
    assert [e.id for e in evidence] == [1, 2, 3]
    assert report.failed_batches == ["3"] and report.max_depth_reached == 1


# --- supervisor policy, heuristics, reducers ------------------------------------------------------


def test_policy_downgrades_tools_for_viewer_and_drops_unknown_departments():
    decision = RouteDecision(
        strategy="tools", intent="i", search_query="", departments=["Payments", "marketing"], rationale="r"
    )
    enforced, notes = enforce_route_policy(
        decision,
        question="q?",
        principal=make_principal(Role.VIEWER),
        known_departments=["payments"],
        today=date(2026, 9, 14),
    )
    assert enforced.strategy == "retrieval"
    assert enforced.departments == ["payments"] and enforced.search_query == "q?"
    assert len(notes) == 2


def test_time_window_resolution():
    today = date(2026, 9, 14)
    assert resolve_time_window("during the last year", today) == (date(2025, 9, 14), today)
    assert resolve_time_window("incidents in 2025", today) == (date(2025, 1, 1), date(2025, 12, 31))
    assert resolve_time_window("what is the policy", today) is None


def test_root_cause_normalization_maps_free_text_to_taxonomy():
    assert normalize_root_cause("TLS cert expired") == "Expired certificate"
    assert (
        normalize_root_cause("Other", "the ledger connection pool was exhausted")
        == "Database connection pool exhaustion"
    )
    assert normalize_root_cause("quantum flux") == "Quantum flux"


def test_union_reducer_resets_and_merges():
    assert union_or_reset(["a"], ["a", "b"]) == ["a", "b"]
    assert union_or_reset(["a"], None) == []
    assert union_or_reset(["old"], [RESET, "new"]) == ["new"]


def test_per_document_quota_prevents_one_document_crowding_out_others():
    dominant = [_chunk("INC-E-1", f"S{i}", date(2026, 7, 24)) for i in range(6)]
    crowded = [_chunk("INC-E-2", "Root Cause", date(2026, 8, 21))]
    selected = diversify_by_document(dominant + crowded, per_doc=3)
    assert [c.chunk.doc_id for c in selected] == ["INC-E-1"] * 3 + ["INC-E-2"]


def test_ground_findings_excludes_irrelevant_incidents_transparently():
    chunks = [_chunk("INC-C-1", "Root Cause", date(2026, 1, 9)), _chunk("INC-TEC-9", "Root Cause", date(2026, 4, 10))]
    extraction = SliceExtraction.model_validate(
        {
            "incidents": [
                {
                    "doc_id": "INC-C-1",
                    "relevant": True,
                    "relevance_reason": "card payments failed",
                    "root_cause_category": "cert",
                    "root_cause_summary": "expired",
                },
                {
                    "doc_id": "INC-TEC-9",
                    "relevant": False,
                    "relevance_reason": "login latency, no payment impact",
                    "root_cause_category": "cache",
                    "root_cause_summary": "node failed",
                },
            ]
        }
    )
    findings, dropped, excluded = ground_findings(extraction, chunks)
    assert [f.doc_id for f in findings] == ["INC-C-1"] and dropped == 0
    assert excluded == ["INC-TEC-9: login latency, no payment impact"]


def test_evidence_cap_never_starves_later_incidents():
    incidents, chunks = [], []
    for n in range(12):
        doc_chunks = [_chunk(f"INC-F-{n:02d}", f"S{k}", date(2026, 1, n + 1)) for k in range(4)]
        chunks += doc_chunks
        incidents.append(
            IncidentFinding(
                doc_id=f"INC-F-{n:02d}",
                title="t",
                date=f"2026-01-{n + 1:02d}",
                root_cause_category="Software defect",
                root_cause_summary="s",
                chunk_ids=[c.chunk.chunk_id for c in doc_chunks],
            )
        )
    finding = BatchFinding(
        label="x", depth=0, filters="", documents_in_scope=12, chunks_analyzed=48, chunks=chunks, incidents=incidents
    )
    report, evidence = aggregate_findings("q", [finding])

    cited_docs = {e.doc_id for e in evidence}
    assert len(evidence) == 30
    assert cited_docs == {i.doc_id for i in incidents}  # 12 x 4 = 48 candidates, yet every incident is citable
    assert report.root_causes[0].count == 12

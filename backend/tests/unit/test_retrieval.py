from datetime import date

import pytest

from app.auth.models import AccessLevel
from app.core.config import REPO_ROOT
from app.core.exceptions import RetrievalUnavailableError
from app.ingestion.chunker import chunk_document, load_markdown
from app.retrieval.filters import build_metadata_filter, matches_filter
from app.retrieval.models import DocumentType, SearchFilters, date_to_ts


def test_filter_always_contains_access_clause():
    f = build_metadata_filter((AccessLevel.PUBLIC,), SearchFilters(departments=["payments"]))
    assert f["$and"][0] == {"access_level": {"$in": ["public"]}}


def test_filter_refuses_to_build_without_clearance():
    with pytest.raises(ValueError):
        build_metadata_filter((), None)


def test_in_memory_filter_semantics_match_pinecone_dialect():
    md = {
        "access_level": "internal",
        "department": "payments",
        "created_ts": date_to_ts(date(2026, 1, 9)),
        "tags": ["a", "b"],
    }
    f = build_metadata_filter(
        (AccessLevel.PUBLIC, AccessLevel.INTERNAL),
        SearchFilters(departments=["payments"], date_from=date(2025, 9, 14), date_to=date(2026, 9, 14)),
    )
    assert matches_filter(md, f)
    assert not matches_filter(md | {"access_level": "confidential"}, f)
    assert not matches_filter(md | {"created_ts": date_to_ts(date(2024, 11, 20))}, f)
    assert matches_filter(md, {"tags": {"$in": ["b"]}})
    assert matches_filter(md, {"$or": [{"department": "hr"}, {"department": {"$eq": "payments"}}]})


def test_search_filters_validate_input():
    with pytest.raises(ValueError):
        SearchFilters(date_from=date(2026, 1, 2), date_to=date(2026, 1, 1))
    with pytest.raises(ValueError):
        SearchFilters(departments=["payments; DROP TABLE"])


def test_chunk_ids_are_stable_and_section_aware():
    path = REPO_ROOT / "data/mock/payments/INC-PAY-2025-041.md"
    md, body = load_markdown(path)
    first, second = chunk_document(md, body), chunk_document(md, body)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert "INC-PAY-2025-041::root-cause::0" in {c.chunk_id for c in first}


async def test_hybrid_search_returns_attributed_relevant_chunks(retriever, analyst):
    result = await retriever.search("processor certificate expiry card authorization", principal=analyst)
    top = result.chunks[0]
    assert top.chunk.metadata.doc_id == "INC-PAY-2025-041"
    assert "INC-PAY-2025-041 · " in top.attribution
    d = result.diagnostics
    assert d.dense_hits > 0 and d.sparse_hits > 0 and d.returned == len(result.chunks)
    assert d.metadata_filter["$and"][0]["access_level"]["$in"] == ["confidential", "internal", "public"]


async def test_viewer_never_receives_confidential_or_restricted_chunks(retriever, viewer, analyst):
    query = "FX routing feature flag misconfiguration cross-border transfers"
    viewer_docs = {c.chunk.doc_id for c in (await retriever.search(query, principal=viewer, top_n=20)).chunks}
    analyst_docs = {c.chunk.doc_id for c in (await retriever.search(query, principal=analyst, top_n=20)).chunks}

    assert "INC-PAY-2026-027" in analyst_docs  # confidential
    assert "INC-PAY-2026-027" not in viewer_docs


@pytest.mark.parametrize("query", ["credential stuffing attack", "compensation bands bonus pool"])
async def test_restricted_documents_only_for_administrators(retriever, analyst, admin, query):
    for principal, expect in ((analyst, False), (admin, True)):
        chunks = (await retriever.search(query, principal=principal, top_n=20)).chunks
        restricted = any(c.chunk.metadata.access_level == "restricted" for c in chunks)
        assert restricted is expect


async def test_metadata_filters_narrow_by_type_department_and_date(retriever, analyst):
    filters = SearchFilters(
        departments=["payments"],
        document_types=[DocumentType.INCIDENT],
        date_from=date(2025, 9, 14),
        date_to=date(2026, 9, 14),
    )
    result = await retriever.search("payment failure root cause", principal=analyst, filters=filters, top_n=50)
    assert result.chunks
    assert result.diagnostics.namespaces == ["payments"]
    for c in result.chunks:
        md = c.chunk.metadata
        assert md.document_type == DocumentType.INCIDENT and md.department == "payments"
        assert date(2025, 9, 14) <= md.created_date <= date(2026, 9, 14)


async def test_injection_payload_in_documents_is_flagged(retriever, analyst):
    result = await retriever.search("pasted vendor email reliability review", principal=analyst)
    flagged = [c for c in result.chunks if c.injection_signals]
    assert flagged and flagged[0].chunk.doc_id == "MTG-PAY-2026-06"
    assert "instruction_override" in flagged[0].injection_signals


async def test_one_failed_leg_degrades_gracefully(retriever, analyst, monkeypatch):
    async def broken(**_):
        raise ConnectionError("sparse index down")

    monkeypatch.setattr(retriever.store, "sparse_query", broken)
    result = await retriever.search("certificate expiry", principal=analyst)
    assert result.chunks and result.diagnostics.degraded_legs == ["sparse"]


async def test_both_legs_failing_raises_retrieval_unavailable(retriever, analyst, monkeypatch):
    async def broken(**_):
        raise ConnectionError("pinecone down")

    monkeypatch.setattr(retriever.store, "sparse_query", broken)
    monkeypatch.setattr(retriever.store, "dense_query", broken)
    with pytest.raises(RetrievalUnavailableError):
        await retriever.search("certificate expiry", principal=analyst)


def test_catalog_overview_is_access_filtered(corpus, viewer, admin):
    viewer_view = corpus.catalog.overview(viewer.allowed_access_levels)
    admin_view = corpus.catalog.overview(admin.allowed_access_levels)
    assert admin_view.total_documents == 31
    assert viewer_view.total_documents < admin_view.total_documents
    assert all(md.access_level in ("public", "internal") for md in corpus.catalog.visible(viewer.allowed_access_levels))

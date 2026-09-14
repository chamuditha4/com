"""Opt-in live tests for the Pinecone-backed retrieval path (indexes populated by data/ingest.py).

RUN_LIVE_TESTS=1 uv run pytest backend/tests/live/test_live_pinecone.py -v
"""

from __future__ import annotations

import os
import time
from collections import Counter
from contextlib import AsyncExitStack
from datetime import date

import pytest

from app.auth.models import AccessLevel, Role
from app.container import build_retriever
from app.core.config import Settings
from app.retrieval.models import DocumentType, SearchFilters
from tests.conftest import make_principal

pytestmark = [
    pytest.mark.skipif(not os.getenv("RUN_LIVE_TESTS"), reason="set RUN_LIVE_TESTS=1 to call Pinecone"),
    pytest.mark.skipif(Settings().vector_store != "pinecone", reason="VECTOR_STORE is not pinecone"),
]


@pytest.fixture
async def retriever():
    async with AsyncExitStack() as stack:
        r, _ = await build_retriever(Settings(), stack)
        yield r


async def test_every_chunk_is_indexed_in_its_department_namespace(retriever):
    counts = await retriever.store.namespace_counts()
    expected: Counter[str] = Counter()
    for md in retriever.catalog.visible(tuple(AccessLevel)):
        expected[md.department] += next(
            e.chunk_count for e in retriever.catalog._entries if e.metadata.doc_id == md.doc_id
        )
    assert counts == dict(expected)


async def test_both_legs_and_hosted_reranker_answer_a_focused_question(retriever):
    viewer = make_principal(Role.VIEWER)
    await retriever.search("warm-up", principal=viewer)  # exclude connection setup from the latency figure
    started = time.perf_counter()
    result = await retriever.search("How do I rotate a TLS certificate?", principal=viewer)
    warm_ms = (time.perf_counter() - started) * 1000

    d = result.diagnostics
    assert result.chunks[0].chunk.doc_id == "RB-TEC-008"
    assert d.dense_hits > 0 and d.sparse_hits > 0 and d.reranker == "pinecone" and d.degraded_legs == []
    print(f"\nwarm hybrid search over {len(d.namespaces)} namespaces: {warm_ms:.0f} ms")


@pytest.mark.parametrize(
    ("query", "doc_id", "denied", "allowed"),
    [
        (
            "FX routing feature flag misconfiguration cross-border transfers",
            "INC-PAY-2026-027",
            Role.VIEWER,
            Role.ANALYST,
        ),
        ("credential stuffing attack mobile banking", "INC-SEC-2026-007", Role.ANALYST, Role.ADMINISTRATOR),
    ],
)
async def test_pinecone_metadata_filter_enforces_clearance(retriever, query, doc_id, denied, allowed):
    denied_result = await retriever.search(query, principal=make_principal(denied), top_n=20)
    allowed_result = await retriever.search(query, principal=make_principal(allowed), top_n=20)

    assert doc_id not in {c.chunk.doc_id for c in denied_result.chunks}
    assert doc_id in {c.chunk.doc_id for c in allowed_result.chunks}
    # The store-side filter did the work; the post-retrieval re-check found nothing to drop.
    assert denied_result.diagnostics.dropped_by_access_check == 0


async def test_department_type_and_date_filters_are_pushed_down(retriever):
    filters = SearchFilters(
        departments=["payments"],
        document_types=[DocumentType.INCIDENT],
        date_from=date(2025, 9, 14),
        date_to=date(2026, 9, 14),
    )
    result = await retriever.search(
        "payment failure root cause", principal=make_principal(Role.ANALYST), filters=filters, top_n=20
    )

    assert result.chunks and result.diagnostics.namespaces == ["payments"]
    for c in result.chunks:
        md = c.chunk.metadata
        assert md.department == "payments" and md.document_type == DocumentType.INCIDENT
        assert date(2025, 9, 14) <= md.created_date <= date(2026, 9, 14)

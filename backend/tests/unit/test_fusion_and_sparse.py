import pytest

from app.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from app.retrieval.models import StoreMatch
from app.retrieval.sparse import BM25Encoder, token_index, tokenize


def m(id_: str, score: float) -> StoreMatch:
    return StoreMatch(id=id_, score=score, metadata={})


DENSE = [m("a", 0.91), m("b", 0.85), m("c", 0.40)]
SPARSE = [m("c", 12.0), m("a", 9.5), m("d", 3.0)]


def test_rrf_rewards_agreement_between_legs():
    fused = reciprocal_rank_fusion(DENSE, SPARSE, k=60)
    ids = [c.id for c in fused]
    # "a" is rank 1 dense + rank 2 sparse -> best. "c" appears in both legs -> beats "b"/"d".
    assert ids[0] == "a"
    assert ids.index("c") < ids.index("b")
    assert fused[0].fused_score == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_keeps_per_leg_ranks_for_explainability():
    fused = {c.id: c for c in reciprocal_rank_fusion(DENSE, SPARSE)}
    assert (fused["a"].dense_rank, fused["a"].sparse_rank) == (1, 2)
    assert fused["b"].sparse_rank is None
    assert fused["d"].dense_rank is None and fused["d"].sparse_score == 3.0


def test_rrf_weights_shift_the_balance():
    sparse_heavy = reciprocal_rank_fusion(DENSE, SPARSE, dense_weight=0.1, sparse_weight=1.0)
    assert sparse_heavy[0].id == "c"


def test_weighted_fusion_normalizes_incomparable_scales():
    fused = weighted_score_fusion(DENSE, SPARSE, dense_weight=0.5, sparse_weight=0.5)
    scores = {c.id: c.fused_score for c in fused}
    # "a": dense 1.0, sparse (9.5-3)/(12-3)=0.722 -> 0.861
    assert scores["a"] == pytest.approx(0.5 * 1.0 + 0.5 * (6.5 / 9))
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_fusion_handles_empty_leg_and_is_deterministic():
    assert [c.id for c in reciprocal_rank_fusion(DENSE, [])] == ["a", "b", "c"]
    tie = reciprocal_rank_fusion([m("z", 1), m("y", 1)], [m("y", 1), m("z", 1)])
    assert [c.id for c in tie] == ["y", "z"]  # equal RRF score -> id tie-break


def test_bm25_dot_product_ranks_rarer_matching_terms_higher():
    corpus = [
        "certificate expired on the card processor link",
        "connection pool exhausted on salary day",
        "payment failures on the card platform",
    ]
    encoder = BM25Encoder.fit(corpus)
    query = encoder.encode_query("expired certificate")

    def score(text: str) -> float:
        doc = encoder.encode_document(text)
        weights = dict(zip(doc.indices, doc.values, strict=True))
        return sum(w * weights.get(i, 0.0) for i, w in zip(query.indices, query.values, strict=True))

    assert score(corpus[0]) > 0
    assert score(corpus[1]) == 0
    assert score(corpus[0]) > score(corpus[2])


def test_bm25_roundtrip_persistence(tmp_path):
    encoder = BM25Encoder.fit(["alpha beta", "beta gamma"])
    path = tmp_path / "bm25.json"
    encoder.save(path)
    restored = BM25Encoder.load(path)
    assert restored.encode_query("beta") == encoder.encode_query("beta")


def test_tokenizer_folds_plurals_and_drops_stopwords():
    assert tokenize("The Certificates of all incidents") == ["certificate", "incident"]
    assert token_index("payments") == token_index("payments")

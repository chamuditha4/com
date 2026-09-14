"""Hybrid rank fusion of the dense and sparse retrieval legs.

Two strategies, selected by `RETRIEVAL_FUSION`:

* **RRF (default).** `score(d) = Σ_leg w_leg / (k + rank_leg(d))`. Uses ranks only, so it
  is robust to the incomparable score scales of cosine similarity vs BM25. `k` damps the
  advantage of top ranks.
* **Weighted.** Min-max normalizes each leg's scores to [0, 1], then
  `score(d) = w_dense·dense(d) + w_sparse·sparse(d)`. More sensitive to score gaps, and to
  outliers.

Each fused candidate keeps its per-leg ranks and scores so the Activity Panel and LangSmith
can show *why* a chunk was selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.retrieval.models import StoreMatch


@dataclass
class FusedCandidate:
    id: str
    metadata: dict[str, Any]
    fused_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None


def _collect(dense: list[StoreMatch], sparse: list[StoreMatch]) -> dict[str, FusedCandidate]:
    candidates: dict[str, FusedCandidate] = {}
    for rank, match in enumerate(dense, start=1):
        c = candidates.setdefault(match.id, FusedCandidate(match.id, match.metadata))
        c.dense_rank, c.dense_score = rank, match.score
    for rank, match in enumerate(sparse, start=1):
        c = candidates.setdefault(match.id, FusedCandidate(match.id, match.metadata))
        c.sparse_rank, c.sparse_score = rank, match.score
    return candidates


def _ordered(candidates: dict[str, FusedCandidate]) -> list[FusedCandidate]:
    # Tie-break on id so fusion output is fully deterministic.
    return sorted(candidates.values(), key=lambda c: (-c.fused_score, c.id))


def reciprocal_rank_fusion(
    dense: list[StoreMatch],
    sparse: list[StoreMatch],
    *,
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[FusedCandidate]:
    candidates = _collect(dense, sparse)
    for c in candidates.values():
        if c.dense_rank is not None:
            c.fused_score += dense_weight / (k + c.dense_rank)
        if c.sparse_rank is not None:
            c.fused_score += sparse_weight / (k + c.sparse_rank)
    return _ordered(candidates)


def _min_max(matches: list[StoreMatch]) -> dict[str, float]:
    if not matches:
        return {}
    scores = [m.score for m in matches]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return {m.id: 1.0 for m in matches}
    return {m.id: (m.score - lo) / (hi - lo) for m in matches}


def weighted_score_fusion(
    dense: list[StoreMatch],
    sparse: list[StoreMatch],
    *,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
) -> list[FusedCandidate]:
    candidates = _collect(dense, sparse)
    dense_norm, sparse_norm = _min_max(dense), _min_max(sparse)
    for c in candidates.values():
        c.fused_score = dense_weight * dense_norm.get(c.id, 0.0) + sparse_weight * sparse_norm.get(c.id, 0.0)
    return _ordered(candidates)

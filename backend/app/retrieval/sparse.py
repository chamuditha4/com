"""BM25 sparse encoding for Pinecone sparse indexes.

BM25 decomposes into a document-side term weight and a query-side IDF weight:

    score(q, d) = Σ_t  IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
                       └ query ┘ └─────────────── document ───────────────┘

So if we store the document factor as a sparse vector and query with IDF weights, the dot
product Pinecone computes *is* the BM25 score. Terms are hashed to stable 32-bit indices.
Corpus statistics (document frequencies, average length) are fitted at ingest time and
saved as an artifact that serving workers load.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from app.retrieval.models import SparseVector

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
STOPWORDS = frozenset(
    """a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours yourself yourselves last during please tell give show
    list summarize""".split()
)


def write_atomically(path: Path, content: str) -> None:
    """Write via a temp file + rename so concurrent workers never read a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _stem(token: str) -> str:
    # Deliberately tiny normalization: plural folding only. Aggressive stemming hurts precision
    # on identifiers such as service names and incident codes.
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def token_index(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")


class BM25Encoder:
    def __init__(
        self,
        *,
        doc_freq: dict[int, int] | None = None,
        n_docs: int = 0,
        avg_doc_len: float = 0.0,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.doc_freq = doc_freq or {}
        self.n_docs = n_docs
        self.avg_doc_len = avg_doc_len
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, corpus: Iterable[str], **kwargs: float) -> BM25Encoder:
        doc_freq: Counter[int] = Counter()
        n_docs, total_len = 0, 0
        for text in corpus:
            tokens = tokenize(text)
            n_docs += 1
            total_len += len(tokens)
            doc_freq.update({token_index(t) for t in tokens})
        return cls(
            doc_freq=dict(doc_freq),
            n_docs=n_docs,
            avg_doc_len=(total_len / n_docs) if n_docs else 0.0,
            **kwargs,
        )

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        if not tokens:
            return SparseVector(indices=[], values=[])
        counts = Counter(token_index(t) for t in tokens)
        length_norm = 1 - self.b + self.b * len(tokens) / (self.avg_doc_len or len(tokens))
        items = sorted((idx, tf * (self.k1 + 1) / (tf + self.k1 * length_norm)) for idx, tf in counts.items())
        return SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])

    def idf(self, index: int) -> float:
        df = self.doc_freq.get(index, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def encode_query(self, text: str) -> SparseVector:
        indices = sorted({token_index(t) for t in tokenize(text)})
        return SparseVector(indices=indices, values=[self.idf(i) for i in indices])

    # --- persistence -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "k1": self.k1,
            "b": self.b,
            "n_docs": self.n_docs,
            "avg_doc_len": self.avg_doc_len,
            "doc_freq": {str(k): v for k, v in self.doc_freq.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BM25Encoder:
        raw_df = data["doc_freq"]
        assert isinstance(raw_df, dict)
        return cls(
            doc_freq={int(k): int(v) for k, v in raw_df.items()},
            n_docs=int(data["n_docs"]),  # type: ignore[arg-type]
            avg_doc_len=float(data["avg_doc_len"]),  # type: ignore[arg-type]
            k1=float(data["k1"]),  # type: ignore[arg-type]
            b=float(data["b"]),  # type: ignore[arg-type]
        )

    def save(self, path: Path) -> None:
        write_atomically(path, json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> BM25Encoder:
        return cls.from_dict(json.loads(path.read_text()))

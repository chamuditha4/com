"""Dense embedding providers behind one `Embedder` port."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from app.retrieval.sparse import tokenize

if TYPE_CHECKING:
    from pinecone import PineconeAsyncio


class Embedder(Protocol):
    name: str
    dimension: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class HashingEmbedder:
    """Deterministic offline embedder (signed feature hashing of unigrams + bigrams).

    It captures lexical, not semantic, similarity: good enough for tests and the offline demo,
    and it exercises the exact same dense code path as a real model.
    """

    name = "hashing"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return _l2_normalize(vector)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class PineconeEmbedder:
    """Pinecone Inference (e.g. `multilingual-e5-large`), which distinguishes passage/query inputs."""

    name = "pinecone"

    def __init__(self, client: PineconeAsyncio, model: str, dimension: int, *, batch_size: int = 90) -> None:
        self._client = client
        self._model = model
        self.dimension = dimension
        self._batch_size = batch_size

    async def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        batches = [texts[i : i + self._batch_size] for i in range(0, len(texts), self._batch_size)]
        results = await asyncio.gather(
            *(
                self._client.inference.embed(
                    model=self._model,
                    inputs=list(batch),
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                for batch in batches
            )
        )
        return [list(item.values) for result in results for item in result.data]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, "passage")

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "query"))[0]


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, model: str, dimension: int, api_key: str | None) -> None:
        from langchain_openai import OpenAIEmbeddings

        self.dimension = dimension
        self._impl = OpenAIEmbeddings(model=model, dimensions=dimension, api_key=api_key)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._impl.aembed_documents(list(texts))

    async def embed_query(self, text: str) -> list[float]:
        return await self._impl.aembed_query(text)

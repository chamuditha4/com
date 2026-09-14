"""Ingestion artifacts must never block API startup on a fresh deployment volume."""

import asyncio

import pytest

from app.container import load_or_derive_artifacts
from app.core.config import REPO_ROOT, Settings
from app.ingestion.pipeline import build_index
from app.retrieval.embeddings import HashingEmbedder
from app.retrieval.stores.memory import InMemoryVectorStore


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(
        _env_file=None, artifacts_dir=tmp_path / "artifacts", mock_docs_dir=REPO_ROOT / "data" / "mock", **overrides
    )


def test_fresh_volume_derives_artifacts_identical_to_ingestion(tmp_path):
    settings = _settings(tmp_path)
    encoder, catalog = load_or_derive_artifacts(settings)

    ingested_encoder, ingested_catalog, _ = asyncio.run(
        build_index(docs_dir=settings.mock_docs_dir, store=InMemoryVectorStore(), embedder=HashingEmbedder())
    )
    assert encoder.to_dict() == ingested_encoder.to_dict()  # sparse query weights match the indexed vectors
    assert catalog.departments == ingested_catalog.departments
    assert settings.bm25_params_path.exists() and settings.catalog_path.exists()


def test_existing_artifacts_are_loaded_not_recomputed(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    load_or_derive_artifacts(settings)

    def fail(*_):
        raise AssertionError("should load from disk")

    monkeypatch.setattr("app.container.derive_corpus_artifacts", fail)
    encoder, _ = load_or_derive_artifacts(settings)
    assert encoder.n_docs > 0


def test_unwritable_volume_still_starts(tmp_path, monkeypatch):
    settings = _settings(tmp_path)

    def read_only(*_):
        raise PermissionError("read-only volume")

    monkeypatch.setattr("app.retrieval.sparse.BM25Encoder.save", read_only)
    encoder, catalog = load_or_derive_artifacts(settings)
    assert encoder.n_docs > 0 and catalog.departments


def test_missing_artifacts_and_corpus_fails_with_actionable_error(tmp_path):
    settings = Settings(_env_file=None, artifacts_dir=tmp_path / "a", mock_docs_dir=tmp_path / "no-corpus")
    with pytest.raises(FileNotFoundError, match=r"data/ingest\.py"):
        load_or_derive_artifacts(settings)

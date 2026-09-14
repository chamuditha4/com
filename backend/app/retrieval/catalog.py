"""Document catalog: document-level metadata without content.

The Research Agent's *explore* step (RLM §5.1) reads the catalog to learn what exists
(departments, types, date ranges, counts) before deciding what to retrieve. The catalog is
access-filtered like retrieval, so exploring cannot reveal titles above the caller's clearance.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from app.auth.models import AccessLevel
from app.retrieval.models import DocumentMetadata, SearchFilters
from app.retrieval.sparse import write_atomically


class CatalogEntry(BaseModel):
    metadata: DocumentMetadata
    chunk_count: int


class CatalogOverview(BaseModel):
    total_documents: int
    by_department: dict[str, int]
    by_document_type: dict[str, int]
    earliest: date | None
    latest: date | None


class DocumentCatalog:
    # SCALE-DEBT: loaded from a JSON artifact produced by ingestion. At larger scale this
    # becomes a metadata table (Postgres) queried with the same access filter.
    def __init__(self, entries: Sequence[CatalogEntry] = ()) -> None:
        self._entries = list(entries)

    @property
    def departments(self) -> list[str]:
        return sorted({e.metadata.department for e in self._entries})

    def visible(
        self, allowed_levels: Sequence[AccessLevel], filters: SearchFilters | None = None
    ) -> list[DocumentMetadata]:
        allowed = {level.value for level in allowed_levels}
        result = []
        for entry in self._entries:
            md = entry.metadata
            if md.access_level.value not in allowed:
                continue
            if filters:
                if filters.departments and md.department not in filters.departments:
                    continue
                if filters.document_types and md.document_type not in filters.document_types:
                    continue
                if filters.date_from and md.created_date < filters.date_from:
                    continue
                if filters.date_to and md.created_date > filters.date_to:
                    continue
                if filters.doc_ids and md.doc_id not in filters.doc_ids:
                    continue
            result.append(md)
        return sorted(result, key=lambda m: (m.created_date, m.doc_id))

    def overview(self, allowed_levels: Sequence[AccessLevel], filters: SearchFilters | None = None) -> CatalogOverview:
        docs = self.visible(allowed_levels, filters)
        return CatalogOverview(
            total_documents=len(docs),
            by_department=dict(sorted(Counter(d.department for d in docs).items())),
            by_document_type=dict(sorted(Counter(d.document_type.value for d in docs).items())),
            earliest=min((d.created_date for d in docs), default=None),
            latest=max((d.created_date for d in docs), default=None),
        )

    def save(self, path: Path) -> None:
        write_atomically(path, json.dumps([e.model_dump(mode="json") for e in self._entries], indent=1))

    @classmethod
    def load(cls, path: Path) -> DocumentCatalog:
        return cls([CatalogEntry.model_validate(e) for e in json.loads(path.read_text())])

"""Retrieval domain models: documents, chunks, filters and scored results.

The metadata schema is the one from CLAUDE.md §6. Dates are also stored as a UTC epoch
integer (`created_ts`) because Pinecone range filters operate on numbers only.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.auth.models import AccessLevel

SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,48}$"


class DocumentType(StrEnum):
    INCIDENT = "incident"
    POLICY = "policy"
    RUNBOOK = "runbook"
    ARCHITECTURE = "architecture"
    PRODUCT_SPEC = "product_spec"
    MEETING_NOTES = "meeting_notes"


def date_to_ts(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=UTC).timestamp())


class DocumentMetadata(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Z0-9][A-Z0-9-]{2,40}$")
    title: str = Field(min_length=3, max_length=200)
    department: str = Field(pattern=SLUG_PATTERN)
    document_type: DocumentType
    access_level: AccessLevel
    created_date: date
    tags: list[str] = Field(default_factory=list)

    def to_index_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "department": self.department,
            "document_type": self.document_type.value,
            "access_level": self.access_level.value,
            "created_date": self.created_date.isoformat(),
            "created_ts": date_to_ts(self.created_date),
            "tags": self.tags,
        }


class Chunk(BaseModel):
    chunk_id: str
    section: str
    chunk_index: int
    text: str
    metadata: DocumentMetadata

    @property
    def doc_id(self) -> str:
        return self.metadata.doc_id

    def to_index_metadata(self) -> dict[str, Any]:
        return self.metadata.to_index_metadata() | {
            "chunk_id": self.chunk_id,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }

    @classmethod
    def from_index_metadata(cls, md: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=md["chunk_id"],
            section=md.get("section", ""),
            chunk_index=int(md.get("chunk_index", 0)),
            text=md.get("text", ""),
            metadata=DocumentMetadata(
                doc_id=md["doc_id"],
                title=md["title"],
                department=md["department"],
                document_type=md["document_type"],
                access_level=md["access_level"],
                created_date=md["created_date"],
                tags=list(md.get("tags", [])),
            ),
        )


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SearchFilters(BaseModel):
    """Caller-requested narrowing. Access control is never expressed here; it is applied
    separately from the principal and cannot be widened by these fields."""

    departments: list[str] | None = None
    document_types: list[DocumentType] | None = None
    date_from: date | None = None
    date_to: date | None = None

    @field_validator("departments")
    @classmethod
    def _slug_departments(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = sorted({d.strip().lower() for d in value if d and d.strip()})
        for d in cleaned:
            if not re.match(SLUG_PATTERN, d):
                raise ValueError(f"invalid department: {d!r}")
        return cleaned or None

    @model_validator(mode="after")
    def _ordered_dates(self) -> SearchFilters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        return self

    def describe(self) -> str:
        parts = []
        if self.departments:
            parts.append(f"departments={','.join(self.departments)}")
        if self.document_types:
            parts.append(f"types={','.join(t.value for t in self.document_types)}")
        if self.date_from or self.date_to:
            parts.append(f"dates={self.date_from or '…'}..{self.date_to or '…'}")
        return " ".join(parts) or "none"


class StoreMatch(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any]


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    injection_signals: list[str] = Field(default_factory=list)

    @property
    def attribution(self) -> str:
        md = self.chunk.metadata
        return f"{md.doc_id} · {md.title} · {self.chunk.section}"


class RetrievalDiagnostics(BaseModel):
    query: str
    metadata_filter: dict[str, Any]
    namespaces: list[str]
    fusion: str
    reranker: str
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_candidates: int = 0
    returned: int = 0
    degraded_legs: list[str] = Field(default_factory=list)
    dropped_by_access_check: int = 0
    flagged_injection: int = 0
    latency_ms: float = 0.0


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics

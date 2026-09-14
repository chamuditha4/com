"""Markdown loading and structure-aware chunking.

Documents are split on headings first (so each chunk carries a meaningful `section` for
citations), then long sections are windowed on paragraph boundaries with overlap. Chunk ids
are derived from the document id, section and position, so re-ingesting the same corpus
overwrites vectors in place (idempotent upserts).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from app.retrieval.models import Chunk, DocumentMetadata

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:48] or "section"


def load_markdown(path: Path) -> tuple[DocumentMetadata, str]:
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise ValueError(f"{path} is missing YAML front matter")
    metadata = DocumentMetadata.model_validate(yaml.safe_load(match.group(1)))
    return metadata, match.group(2).strip()


def _sections(body: str, default_title: str) -> list[tuple[str, str]]:
    headings = list(_HEADING.finditer(body))
    if not headings:
        return [(default_title, body)]
    sections: list[tuple[str, str]] = []
    preamble = body[: headings[0].start()].strip()
    if preamble:
        sections.append(("Overview", preamble))
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        text = body[heading.end() : end].strip()
        if text:
            sections.append((heading.group(2).strip(), text))
    return sections


def _windows(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    windows: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_chars:
            windows.append(current)
            current = current[-overlap_chars:] if overlap_chars else ""
        current = f"{current}\n\n{paragraph}".strip() if current else paragraph
    if current:
        windows.append(current)
    return windows


def chunk_document(
    metadata: DocumentMetadata, body: str, *, max_chars: int = 1200, overlap_chars: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen_slugs: dict[str, int] = {}
    for section, text in _sections(body, metadata.title):
        slug = slugify(section)
        seen_slugs[slug] = seen_slugs.get(slug, 0) + 1
        if seen_slugs[slug] > 1:
            slug = f"{slug}-{seen_slugs[slug]}"
        for i, window in enumerate(_windows(text, max_chars, overlap_chars)):
            chunks.append(
                Chunk(
                    chunk_id=f"{metadata.doc_id}::{slug}::{i}",
                    section=section,
                    chunk_index=len(chunks),
                    # The title and section are prepended so both retrieval legs see the context
                    # a human would use to judge relevance.
                    text=f"{metadata.title} — {section}\n\n{window}",
                    metadata=metadata,
                )
            )
    return chunks

"""Metadata filter construction, in Pinecone's filter dialect.

`build_metadata_filter` is the security boundary for data access: it *always* emits the
access-level clause derived from the principal's clearance, and it has no parameter through
which a caller (or an LLM) could widen it. `matches_filter` evaluates the same dialect for
the in-memory store, so both adapters share semantics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.auth.models import AccessLevel
from app.retrieval.models import SearchFilters, date_to_ts


def build_metadata_filter(
    allowed_access_levels: Sequence[AccessLevel], filters: SearchFilters | None = None
) -> dict[str, Any]:
    if not allowed_access_levels:
        raise ValueError("refusing to build a retrieval filter without an access-level clause")

    clauses: list[dict[str, Any]] = [
        {"access_level": {"$in": sorted(level.value for level in allowed_access_levels)}}
    ]
    if filters:
        if filters.departments:
            clauses.append({"department": {"$in": filters.departments}})
        if filters.document_types:
            clauses.append({"document_type": {"$in": [t.value for t in filters.document_types]}})
        if filters.date_from:
            clauses.append({"created_ts": {"$gte": date_to_ts(filters.date_from)}})
        if filters.date_to:
            clauses.append({"created_ts": {"$lte": date_to_ts(filters.date_to)}})
        if filters.doc_ids:
            clauses.append({"doc_id": {"$in": filters.doc_ids}})
    return {"$and": clauses}


_COMPARATORS = {
    "$eq": lambda a, b: a == b,
    "$ne": lambda a, b: a != b,
    "$gt": lambda a, b: a is not None and a > b,
    "$gte": lambda a, b: a is not None and a >= b,
    "$lt": lambda a, b: a is not None and a < b,
    "$lte": lambda a, b: a is not None and a <= b,
    "$in": lambda a, b: (bool(set(a) & set(b)) if isinstance(a, list) else a in b),
    "$nin": lambda a, b: (not set(a) & set(b)) if isinstance(a, list) else a not in b,
}


def matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    if not metadata_filter:
        return True
    for key, condition in metadata_filter.items():
        if key == "$and":
            if not all(matches_filter(metadata, c) for c in condition):
                return False
        elif key == "$or":
            if not any(matches_filter(metadata, c) for c in condition):
                return False
        elif isinstance(condition, dict):
            value = metadata.get(key)
            for op, expected in condition.items():
                if op not in _COMPARATORS:
                    raise ValueError(f"unsupported filter operator: {op}")
                if not _COMPARATORS[op](value, expected):
                    return False
        elif metadata.get(key) != condition:
            return False
    return True

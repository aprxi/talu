"""
Type definitions for TaluDB storage engine.

Storage-engine-specific types for vector and document operations.
Record types (ItemRecord, SessionRecord, etc.) live in ``talu.types.records``.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from enum import IntEnum

__all__ = [
    # Vector types
    "VectorBatch",
    "SearchResult",
    "SearchBatchResult",
    # Document types
    "DocumentRecord",
    "DocumentSummary",
    "DocumentSearchResult",
    "ChangeRecord",
    "ChangeAction",
    "CompactionStats",
]


# =============================================================================
# Vector Types - For VectorStore operations
# =============================================================================


@dataclass
class VectorBatch:
    """
    Batch of vectors loaded from storage.

    Attributes
    ----------
        ids: array("Q", ...) of u64 vector IDs.
        vectors: array("f", ...) of float32 vectors, flattened.
        count: Number of vectors in the batch.
        dims: Vector dimensionality.

    Example:
        >>> batch = store.load()
        >>> print(f"Loaded {batch.count} vectors of dim {batch.dims}")
        >>> for i in range(batch.count):
        ...     vec_id = batch.ids[i]
        ...     vec = batch.vectors[i * batch.dims : (i + 1) * batch.dims]
    """

    ids: array[int]
    vectors: array[float]
    count: int
    dims: int


@dataclass
class SearchResult:
    """
    Result from a single-query vector search.

    Attributes
    ----------
        ids: array("Q", ...) of matching vector IDs.
        scores: array("f", ...) of similarity scores (dot product).

    Example:
        >>> result = store.search(query, k=5)
        >>> for vec_id, score in zip(result.ids, result.scores):
        ...     print(f"ID {vec_id}: score {score:.4f}")
    """

    ids: array[int]
    scores: array[float]


@dataclass
class SearchBatchResult:
    """
    Result from a multi-query vector search.

    Attributes
    ----------
        ids: array("Q", ...) of matching vector IDs (query_count * count_per_query).
        scores: array("f", ...) of similarity scores (query_count * count_per_query).
        count_per_query: Number of results per query.

    Example:
        >>> result = store.search_batch(queries, dims=3, query_count=2, k=5)
        >>> for q in range(2):
        ...     offset = q * result.count_per_query
        ...     for i in range(result.count_per_query):
        ...         print(f"Query {q}: ID {result.ids[offset+i]}, score {result.scores[offset+i]:.4f}")
    """

    ids: array[int]
    scores: array[float]
    count_per_query: int


# =============================================================================
# Document Types - For DocumentStore operations
# =============================================================================


class ChangeAction(IntEnum):
    """Change action type for CDC."""

    CREATE = 1
    UPDATE = 2
    DELETE = 3


@dataclass
class DocumentRecord:
    """Complete document with metadata and content."""

    doc_id: str
    doc_type: str
    title: str
    doc_json: str
    tags_text: str | None = None
    parent_id: str | None = None
    marker: str | None = None
    group_id: str | None = None
    owner_id: str | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0
    expires_at_ms: int = 0
    content_hash: int = 0
    seq_num: int = 0


@dataclass
class DocumentSummary:
    """Lightweight document summary for list results."""

    doc_id: str
    doc_type: str
    title: str
    marker: str | None = None
    created_at_ms: int = 0
    updated_at_ms: int = 0


@dataclass
class DocumentSearchResult:
    """Search result with snippet."""

    doc_id: str
    doc_type: str
    title: str
    snippet: str


@dataclass
class ChangeRecord:
    """Document change event for CDC streams."""

    seq_num: int
    doc_id: str
    action: ChangeAction
    timestamp_ms: int
    doc_type: str | None = None
    title: str | None = None


@dataclass
class CompactionStats:
    """Compaction statistics for the document store."""

    total_documents: int
    active_documents: int
    expired_documents: int
    deleted_documents: int
    tombstone_count: int
    delta_versions: int
    estimated_garbage_bytes: int

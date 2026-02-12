"""
Storage and persistence for conversations and vectors.

This module provides backends for conversation and vector persistence:

- Database: Built-in Zig-backed storage for conversations (default)
- VectorStore: High-throughput embedding storage using TaluDB
- DocumentStore: Full-text document storage

Record types (ItemRecord, SessionRecord) and storage events (StorageEvent)
live in ``talu.types``.

Vector Storage Example:
    >>> from array import array
    >>> from talu.db import VectorStore
    >>>
    >>> store = VectorStore("./my-vectors")
    >>> ids = array("Q", [1, 2, 3])
    >>> vectors = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    >>> store.append_batch(ids, vectors, dims=3)
    >>>
    >>> query = array("f", [1.0, 0.0, 0.0])
    >>> ids, scores = store.search(query, k=2)
    >>> store.close()
"""

from ._bindings import DocumentStore, VectorStore
from .backends import Database
from .types import (
    ChangeAction,
    ChangeRecord,
    CompactionStats,
    DocumentRecord,
    DocumentSearchResult,
    DocumentSummary,
    SearchBatchResult,
    SearchResult,
    VectorBatch,
)

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
    # Backends
    "Database",
    "VectorStore",
    "DocumentStore",
]

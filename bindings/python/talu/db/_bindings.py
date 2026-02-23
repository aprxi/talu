"""
FFI bindings for TaluDB storage operations.

Justification: VectorStore and DocumentStore are native pointer wrappers that require
direct ctypes access for efficient buffer manipulation and C struct handling when
interacting with the Zig storage APIs. This follows the same pattern as
talu/_bindings.py for native resource management.
"""

from __future__ import annotations

import ctypes
from array import array
from collections.abc import Iterator
from ctypes import (
    POINTER,
    byref,
    c_bool,
    c_char_p,
    c_float,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
    cast,
)
from pathlib import Path
from typing import TYPE_CHECKING

from .types import (
    ChangeAction,
    ChangeRecord,
    CompactionStats,
    DocumentRecord,
    DocumentSearchResult,
    DocumentSummary,
    VectorBatch,
)

if TYPE_CHECKING:
    from ctypes import CDLL

__all__ = ["VectorStore", "DocumentStore"]


class VectorStore:
    """
    High-throughput embedding storage using TaluDB.

    VectorStore provides efficient storage and similarity search for
    embedding vectors. It wraps the Zig-based TaluDB vector backend,
    offering crash-safe persistence and fast dot-product search.

    Args:
        location: Path to the TaluDB vector database folder.
            If the folder doesn't exist, it will be created.

    Example:
        >>> from array import array
        >>> from talu.db import VectorStore
        >>>
        >>> # Create/open a vector store
        >>> store = VectorStore("./my-vectors")
        >>>
        >>> # Append vectors (Structure-of-Arrays format)
        >>> ids = array("Q", [1, 2, 3])  # u64 IDs
        >>> vectors = array("f", [
        ...     1.0, 0.0, 0.0,  # vector 1
        ...     0.0, 1.0, 0.0,  # vector 2
        ...     0.0, 0.0, 1.0,  # vector 3
        ... ])
        >>> store.append_batch(ids, vectors, dims=3)
        >>>
        >>> # Search for similar vectors
        >>> query = array("f", [1.0, 0.0, 0.0])
        >>> ids, scores = store.search(query, k=2)
        >>> print(list(zip(ids, scores)))
        [(1, 1.0), (2, 0.0)]
        >>>
        >>> # Always close when done
        >>> store.close()

    Note:
        VectorStore uses Structure-of-Arrays format for efficiency:
        - ids: array("Q", ...) - u64 vector IDs
        - vectors: array("f", ...) - float32 vectors, flattened
        - dims: vector dimensionality

        For 3 vectors of dim=2: vectors = [v0x, v0y, v1x, v1y, v2x, v2y]
    """

    _handle: c_void_p | None
    _lib: CDLL
    _closed: bool

    def __init__(self, location: str | Path) -> None:
        """
        Initialize vector store at the given path.

        Args:
            location: Path to the TaluDB vector database folder.

        Raises
        ------
            IOError: If the store cannot be initialized.
        """
        from .._bindings import check, get_lib

        self._lib = get_lib()
        self._closed = False

        path_str = str(location)
        path_bytes = path_str.encode("utf-8")

        handle = c_void_p()
        rc = self._lib.talu_db_vector_init(path_bytes, byref(handle))
        check(rc, {"operation": "vector_store_init", "path": path_str})

        self._handle = handle

    def close(self) -> None:
        """
        Close the vector store and release resources.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._closed or self._handle is None:
            return
        self._lib.talu_db_vector_free(self._handle)
        self._handle = None
        self._closed = True

    def __del__(self) -> None:
        """Release resources on garbage collection."""
        self.close()

    def __enter__(self) -> VectorStore:
        """Context manager entry."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit - close the store."""
        self.close()

    def _check_open(self) -> None:
        """Raise if store is closed."""
        if self._closed or self._handle is None:
            from ..exceptions import StateError

            raise StateError("VectorStore is closed")

    def append_batch(
        self,
        ids: array[int],
        vectors: array[float],
        dims: int,
    ) -> None:
        """
        Append a batch of vectors to the store.

        Args:
            ids: array("Q", ...) of u64 vector IDs.
            vectors: array("f", ...) of float32 vectors, flattened.
            dims: Vector dimensionality.

        Raises
        ------
            StateError: If the store is closed.
            ValidationError: If inputs are invalid.
            IOError: If the append operation fails.

        Example:
            >>> ids = array("Q", [1, 2])
            >>> vectors = array("f", [1.0, 0.0, 0.0, 1.0])  # 2 vectors of dim=2
            >>> store.append_batch(ids, vectors, dims=2)
        """
        from .._bindings import check

        self._check_open()

        count = len(ids)
        if count == 0:
            return

        expected_len = count * dims
        if len(vectors) != expected_len:
            from ..exceptions import ValidationError

            raise ValidationError(
                f"vectors length {len(vectors)} does not match count * dims ({expected_len})"
            )

        # Get raw pointers from arrays
        ids_ptr = (c_uint64 * count).from_buffer(ids)
        vectors_ptr = (c_float * len(vectors)).from_buffer(vectors)

        rc = self._lib.talu_db_vector_append(
            self._handle,
            ids_ptr,
            vectors_ptr,
            c_size_t(count),
            c_uint32(dims),
        )
        check(rc, {"operation": "vector_store_append"})

    def load(self) -> VectorBatch:
        """
        Load all vectors from the store.

        Returns
        -------
            VectorBatch with ids, vectors, count, and dims.

        Raises
        ------
            StateError: If the store is closed.
            IOError: If the load operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_ids = c_void_p()
        out_vectors = c_void_p()
        out_count = c_size_t()
        out_dims = c_uint32()

        rc = self._lib.talu_db_vector_load(
            self._handle,
            byref(out_ids),
            byref(out_vectors),
            byref(out_count),
            byref(out_dims),
        )
        check(rc, {"operation": "vector_store_load"})

        count = out_count.value
        dims = out_dims.value

        # Copy data into Python arrays before freeing Zig buffers
        ids: array[int] = array("Q")
        vectors: array[float] = array("f")

        if count > 0 and out_ids.value and out_vectors.value:
            ids_array = (c_uint64 * count).from_address(out_ids.value)
            ids.extend(ids_array)

            total = count * dims
            vectors_array = (c_float * total).from_address(out_vectors.value)
            vectors.extend(vectors_array)

            # Free Zig-allocated buffers
            self._lib.talu_db_vector_free_load(
                out_ids,
                cast(out_vectors, POINTER(c_float)),
                c_size_t(count),
                c_uint32(dims),
            )

        return VectorBatch(ids=ids, vectors=vectors, count=count, dims=dims)

    def search(self, query: array[float], k: int) -> tuple[array[int], array[float]]:
        """
        Search for the top-k most similar vectors.

        Args:
            query: array("f", ...) query vector.
            k: Number of results to return.

        Returns
        -------
            Tuple of (ids, scores) arrays.
            - ids: array("Q", ...) of matching vector IDs
            - scores: array("f", ...) of similarity scores (dot product)

        Raises
        ------
            StateError: If the store is closed.
            IOError: If the search operation fails.

        Example:
            >>> query = array("f", [1.0, 0.0, 0.0])
            >>> ids, scores = store.search(query, k=5)
        """
        from .._bindings import check

        self._check_open()

        query_ptr = (c_float * len(query)).from_buffer(query)

        out_ids = c_void_p()
        out_scores = c_void_p()
        out_count = c_size_t()

        rc = self._lib.talu_db_vector_search(
            self._handle,
            query_ptr,
            c_size_t(len(query)),
            c_uint32(k),
            byref(out_ids),
            byref(out_scores),
            byref(out_count),
        )
        check(rc, {"operation": "vector_store_search"})

        count = out_count.value

        # Copy data into Python arrays before freeing Zig buffers
        ids: array[int] = array("Q")
        scores: array[float] = array("f")

        if count > 0 and out_ids.value and out_scores.value:
            ids_array = (c_uint64 * count).from_address(out_ids.value)
            ids.extend(ids_array)

            scores_array = (c_float * count).from_address(out_scores.value)
            scores.extend(scores_array)

            # Free Zig-allocated buffers
            self._lib.talu_db_vector_free_search(
                out_ids,
                cast(out_scores, POINTER(c_float)),
                c_size_t(count),
            )

        return ids, scores

    def search_batch(
        self,
        queries: array[float],
        dims: int,
        query_count: int,
        k: int,
    ) -> tuple[array[int], array[float], int]:
        """
        Search for the top-k most similar vectors for multiple queries.

        Args:
            queries: array("f", ...) of query vectors, flattened.
            dims: Vector dimensionality.
            query_count: Number of queries.
            k: Number of results per query.

        Returns
        -------
            Tuple of (ids, scores, count_per_query).
            - ids: array("Q", ...) of matching vector IDs (query_count * k)
            - scores: array("f", ...) of similarity scores (query_count * k)
            - count_per_query: Actual results per query (may be < k if fewer vectors)

        Raises
        ------
            StateError: If the store is closed.
            IOError: If the search operation fails.

        Example:
            >>> queries = array("f", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # 2 queries
            >>> ids, scores, count = store.search_batch(queries, dims=3, query_count=2, k=2)
        """
        from .._bindings import check

        self._check_open()

        query_ptr = (c_float * len(queries)).from_buffer(queries)

        out_ids = c_void_p()
        out_scores = c_void_p()
        out_count_per_query = c_uint32()

        rc = self._lib.talu_db_vector_search_batch(
            self._handle,
            query_ptr,
            c_size_t(len(queries)),
            c_uint32(dims),
            c_uint32(query_count),
            c_uint32(k),
            byref(out_ids),
            byref(out_scores),
            byref(out_count_per_query),
        )
        check(rc, {"operation": "vector_store_search_batch"})

        count_per_query = out_count_per_query.value
        total = count_per_query * query_count

        # Copy data into Python arrays before freeing Zig buffers
        ids: array[int] = array("Q")
        scores: array[float] = array("f")

        if total > 0 and out_ids.value and out_scores.value:
            ids_array = (c_uint64 * total).from_address(out_ids.value)
            ids.extend(ids_array)

            scores_array = (c_float * total).from_address(out_scores.value)
            scores.extend(scores_array)

            # Free Zig-allocated buffers
            self._lib.talu_db_vector_free_search_batch(
                out_ids,
                cast(out_scores, POINTER(c_float)),
                c_uint32(count_per_query),
                c_uint32(query_count),
            )

        return ids, scores, count_per_query

    def set_durability(self, mode: str) -> None:
        """
        Set the write durability mode.

        Args:
            mode: ``"full"`` (fsync per write) or ``"async_os"`` (OS-buffered).

        Raises
        ------
            ValueError: If *mode* is not a recognised durability string.
            StateError: If the store is closed.
        """
        from .._bindings import check

        self._check_open()

        values = {"full": 0, "async_os": 1}
        if mode not in values:
            raise ValueError(f"invalid durability mode {mode!r}, expected 'full' or 'async_os'")

        rc = self._lib.talu_db_vector_set_durability(self._handle, c_uint8(values[mode]))
        check(rc, {"operation": "vector_store_set_durability"})

    def scan(self, query: array[float]) -> Iterator[tuple[int, float]]:
        """
        Stream scores for all vectors without top-k sorting.

        This is useful for applying custom thresholds or filters in Python.

        Args:
            query: array("f", ...) query vector.

        Yields
        ------
            Tuples of (id, score) for each vector in the store.

        Example:
            >>> query = array("f", [1.0, 0.0, 0.0])
            >>> for vec_id, score in store.scan(query):
            ...     if score > 0.5:
            ...         print(f"ID {vec_id}: {score}")
        """
        self._check_open()

        # Load all vectors to get their IDs, then compute scores
        batch = self.load()
        if batch.count == 0:
            return

        # For each vector, compute dot product with query
        dims = batch.dims
        for i in range(batch.count):
            vec_id = batch.ids[i]
            offset = i * dims
            score = sum(query[j] * batch.vectors[offset + j] for j in range(min(len(query), dims)))
            yield vec_id, score

    def scan_batch(
        self,
        queries: array[float],
        dims: int,
        query_count: int,
    ) -> tuple[array[int], array[float], int]:
        """
        Scan scores for multiple queries into flat buffers.

        Args:
            queries: array("f", ...) of query vectors, flattened.
            dims: Vector dimensionality.
            query_count: Number of queries.

        Returns
        -------
            Tuple of (ids, scores, total_rows).
            - ids: array("Q", ...) of all vector IDs
            - scores: array("f", ...) of all scores (query_count * total_rows)
            - total_rows: Number of vectors in the store

        Raises
        ------
            StateError: If the store is closed.
            IOError: If the scan operation fails.

        Note:
            Scores are laid out as [query][row]. For query i, row j:
            scores[i * total_rows + j]
        """
        from .._bindings import check

        self._check_open()

        query_ptr = (c_float * len(queries)).from_buffer(queries)
        out_total_rows = c_size_t()

        # First call to get total_rows without allocating buffers
        rc = self._lib.talu_db_vector_scan_batch(
            self._handle,
            query_ptr,
            c_size_t(len(queries)),
            c_uint32(dims),
            c_uint32(query_count),
            None,  # out_ids
            c_size_t(0),  # ids_len
            None,  # out_scores
            c_size_t(0),  # scores_len
            byref(out_total_rows),
        )
        check(rc, {"operation": "vector_store_scan_batch"})

        total_rows = out_total_rows.value
        if total_rows == 0:
            return array("Q"), array("f"), 0

        # Allocate buffers and call again
        ids: array[int] = array("Q", [0] * total_rows)
        scores: array[float] = array("f", [0.0] * (total_rows * query_count))

        ids_ptr = (c_uint64 * total_rows).from_buffer(ids)
        scores_ptr = (c_float * (total_rows * query_count)).from_buffer(scores)

        rc = self._lib.talu_db_vector_scan_batch(
            self._handle,
            query_ptr,
            c_size_t(len(queries)),
            c_uint32(dims),
            c_uint32(query_count),
            ids_ptr,
            c_size_t(total_rows),
            scores_ptr,
            c_size_t(total_rows * query_count),
            byref(out_total_rows),
        )
        check(rc, {"operation": "vector_store_scan_batch"})

        return ids, scores, total_rows


# =============================================================================
# Document Store - Document storage and management
# =============================================================================


def _cstr_to_str(ptr: c_char_p | None) -> str:
    """Convert C string pointer to Python string."""
    if ptr is None or not ptr:
        return ""
    value = ptr if isinstance(ptr, bytes) else ptr  # type: ignore[redundant-expr]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


def _cstr_to_opt_str(ptr: c_char_p | None) -> str | None:
    """Convert C string pointer to optional Python string."""
    if ptr is None or not ptr:
        return None
    value = ptr if isinstance(ptr, bytes) else ptr  # type: ignore[redundant-expr]
    if isinstance(value, bytes):
        s = value.decode("utf-8", errors="replace")
        return s if s else None
    return None


def _to_bytes(s: str | None) -> bytes | None:
    """Convert string to bytes for C API."""
    if s is None:
        return None
    return s.encode("utf-8")


class DocumentStore:
    """
    Document storage and management using TaluDB.

    DocumentStore provides persistent storage for documents with support
    for CRUD operations, search, tagging, TTL, versioning, and CDC.

    Args:
        location: Path to the TaluDB database folder.

    Example:
        >>> from talu.db import DocumentStore
        >>>
        >>> with DocumentStore("./my-docs") as store:
        ...     store.create(
        ...         doc_id="doc-123",
        ...         doc_type="prompt",
        ...         title="My Prompt",
        ...         doc_json='{"content": "Hello world"}',
        ...         tags_text="coding review",
        ...     )
        ...
        ...     doc = store.get("doc-123")
        ...     print(doc.title)
        'My Prompt'

    Note:
        Always close the store when done, or use it as a context manager.
    """

    _lib: CDLL
    _path_bytes: bytes
    _closed: bool

    def __init__(self, location: str | Path) -> None:
        """
        Initialize document store at the given path.

        Args:
            location: Path to the TaluDB database folder.
        """
        from .._bindings import get_lib

        self._lib = get_lib()
        self._path_bytes = str(location).encode("utf-8")
        self._closed = False

    def close(self) -> None:
        """
        Close the document store.

        This method is idempotent - calling it multiple times is safe.
        """
        self._closed = True

    def __del__(self) -> None:
        """Release resources on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> DocumentStore:
        """Context manager entry."""
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit."""
        self.close()

    def _check_open(self) -> None:
        """Raise if store is closed."""
        if self._closed:
            from ..exceptions import StateError

            raise StateError("DocumentStore is closed")

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create(
        self,
        doc_id: str,
        doc_type: str,
        title: str,
        doc_json: str,
        *,
        tags_text: str | None = None,
        parent_id: str | None = None,
        marker: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
    ) -> None:
        """
        Create a new document.

        Args:
            doc_id: Unique document identifier.
            doc_type: Document type (e.g., "prompt", "persona", "rag").
            title: Human-readable title.
            doc_json: JSON content payload.
            tags_text: Space-separated tags for search (optional).
            parent_id: Parent document ID for versioning (optional).
            marker: Lifecycle marker like "active", "archived" (optional).
            group_id: Group ID for multi-tenant isolation (optional).
            owner_id: Owner ID for "My Docs" filtering (optional).

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the document cannot be created.
            ValidationError: If arguments are invalid.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_create(
            self._path_bytes,
            _to_bytes(doc_id),
            _to_bytes(doc_type),
            _to_bytes(title),
            _to_bytes(doc_json),
            _to_bytes(tags_text),
            _to_bytes(parent_id),
            _to_bytes(marker),
            _to_bytes(group_id),
            _to_bytes(owner_id),
        )
        check(rc, {"operation": "documents_create", "doc_id": doc_id})

    def get(self, doc_id: str) -> DocumentRecord | None:
        """
        Get a document by ID.

        Args:
            doc_id: Document identifier.

        Returns
        -------
            DocumentRecord if found, None otherwise.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails (other than not found).
        """
        from .._bindings import check, take_last_error
        from .._native import CDocumentRecord

        self._check_open()

        out_doc = CDocumentRecord()

        rc = self._lib.talu_db_table_get(
            self._path_bytes,
            _to_bytes(doc_id),
            byref(out_doc),
        )

        # Handle "not found" case: error code 700 with "not found" message
        if rc == 700:
            _, msg = take_last_error()
            if msg and "not found" in msg.lower():
                return None
            # Re-raise for other storage errors
            from .._bindings import clear_error

            clear_error()
        if rc != 0:
            check(rc, {"operation": "documents_get", "doc_id": doc_id})

        return DocumentRecord(
            doc_id=_cstr_to_str(out_doc.doc_id),
            doc_type=_cstr_to_str(out_doc.doc_type),
            title=_cstr_to_str(out_doc.title),
            tags_text=_cstr_to_opt_str(out_doc.tags_text),
            doc_json=_cstr_to_str(out_doc.doc_json),
            parent_id=_cstr_to_opt_str(out_doc.parent_id),
            marker=_cstr_to_opt_str(out_doc.marker),
            group_id=_cstr_to_opt_str(out_doc.group_id),
            owner_id=_cstr_to_opt_str(out_doc.owner_id),
            created_at_ms=out_doc.created_at_ms,
            updated_at_ms=out_doc.updated_at_ms,
            expires_at_ms=out_doc.expires_at_ms,
            content_hash=out_doc.content_hash,
            seq_num=out_doc.seq_num,
        )

    def update(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        doc_json: str | None = None,
        tags_text: str | None = None,
        marker: str | None = None,
    ) -> None:
        """
        Update an existing document.

        Only provided fields are updated; others are left unchanged.

        Args:
            doc_id: Document identifier.
            title: New title (optional).
            doc_json: New JSON content (optional).
            tags_text: New tags (optional).
            marker: New marker (optional).

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the document doesn't exist or update fails.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_update(
            self._path_bytes,
            _to_bytes(doc_id),
            _to_bytes(title),
            _to_bytes(doc_json),
            _to_bytes(tags_text),
            _to_bytes(marker),
        )
        check(rc, {"operation": "documents_update", "doc_id": doc_id})

    # =========================================================================
    # Blob Operations
    # =========================================================================

    def get_blob_ref(self, doc_id: str) -> str | None:
        """
        Get the external blob reference for a document payload.

        Args:
            doc_id: Document identifier.

        Returns
        -------
            Blob reference (`sha256:<hex>` or `multi:<hex>`) when externalized,
            otherwise None for inline payloads.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the lookup fails.
        """
        from .._bindings import check

        self._check_open()

        out_has_external = c_bool(False)
        out_ref = ctypes.create_string_buffer(128)

        rc = self._lib.talu_db_table_get_blob_ref(
            self._path_bytes,
            _to_bytes(doc_id),
            out_ref,
            len(out_ref),
            byref(out_has_external),
        )
        check(rc, {"operation": "documents_get_blob_ref", "doc_id": doc_id})

        if not out_has_external.value:
            return None
        return out_ref.value.decode("utf-8", errors="replace") or None

    def iter_blob_chunks(self, blob_ref: str, *, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        """
        Stream an externalized blob as chunks.

        Args:
            blob_ref: Blob reference (`sha256:<hex>` or `multi:<hex>`).
            chunk_size: Read chunk size in bytes.

        Yields
        ------
            Blob content chunks.

        Raises
        ------
            StateError: If the store is closed.
            ValidationError: If chunk_size is invalid or blob_ref is malformed.
            StorageError: If streaming fails.
        """
        from .._bindings import check
        from ..exceptions import ValidationError

        self._check_open()
        if chunk_size <= 0:
            raise ValidationError("chunk_size must be > 0")

        stream = c_void_p()
        rc = self._lib.talu_db_blob_open_stream(
            self._path_bytes,
            _to_bytes(blob_ref),
            byref(stream),
        )
        check(rc, {"operation": "blobs_open_stream", "blob_ref": blob_ref})

        try:
            buf = (ctypes.c_ubyte * chunk_size)()
            while True:
                out_read = c_size_t()
                rc = self._lib.talu_db_blob_stream_read(
                    stream,
                    cast(buf, c_void_p),
                    chunk_size,
                    byref(out_read),
                )
                check(rc, {"operation": "blobs_stream_read", "blob_ref": blob_ref})
                if out_read.value == 0:
                    break
                yield bytes(buf[0 : out_read.value])
        finally:
            if stream.value:
                self._lib.talu_db_blob_stream_close(stream)

    def read_blob(self, blob_ref: str, *, chunk_size: int = 64 * 1024) -> bytes:
        """
        Read an externalized blob into memory.

        Args:
            blob_ref: Blob reference (`sha256:<hex>` or `multi:<hex>`).
            chunk_size: Streaming read chunk size.

        Returns
        -------
            Full blob payload bytes.
        """
        out = bytearray()
        for chunk in self.iter_blob_chunks(blob_ref, chunk_size=chunk_size):
            out.extend(chunk)
        return bytes(out)

    def delete(self, doc_id: str) -> None:
        """
        Delete a document.

        Args:
            doc_id: Document identifier.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_delete(
            self._path_bytes,
            _to_bytes(doc_id),
        )
        check(rc, {"operation": "documents_delete", "doc_id": doc_id})

    def list(
        self,
        *,
        doc_type: str | None = None,
        group_id: str | None = None,
        owner_id: str | None = None,
        marker: str | None = None,
        limit: int = 100,
    ) -> list[DocumentSummary]:
        """
        List documents with optional filters.

        Args:
            doc_type: Filter by document type (optional).
            group_id: Filter by group ID (optional).
            owner_id: Filter by owner ID (optional).
            marker: Filter by marker (optional).
            limit: Maximum number of results.

        Returns
        -------
            List of DocumentSummary objects.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check
        from .._native import CDocumentList

        self._check_open()

        out_list = c_void_p()

        rc = self._lib.talu_db_table_list(
            self._path_bytes,
            _to_bytes(doc_type),
            _to_bytes(group_id),
            _to_bytes(owner_id),
            _to_bytes(marker),
            limit,
            byref(out_list),
        )
        check(rc, {"operation": "documents_list"})

        results: list[DocumentSummary] = []

        if out_list.value:
            # Cast the void pointer to CDocumentList pointer
            doc_list = CDocumentList.from_address(out_list.value)

            if doc_list.items and doc_list.count > 0:
                for i in range(doc_list.count):
                    item = doc_list.items[i]
                    results.append(
                        DocumentSummary(
                            doc_id=_cstr_to_str(item.doc_id),
                            doc_type=_cstr_to_str(item.doc_type),
                            title=_cstr_to_str(item.title),
                            marker=_cstr_to_opt_str(item.marker),
                            created_at_ms=item.created_at_ms,
                            updated_at_ms=item.updated_at_ms,
                        )
                    )

            # Free the list - pass as pointer
            self._lib.talu_db_table_free_list(c_void_p(out_list.value))

        return results

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query: str,
        *,
        doc_type: str | None = None,
        limit: int = 10,
    ) -> list[DocumentSearchResult]:
        """
        Search documents by content.

        Args:
            query: Search query string.
            doc_type: Filter by document type (optional).
            limit: Maximum number of results.

        Returns
        -------
            List of DocumentSearchResult objects with snippets.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check
        from .._native import CSearchResultList

        self._check_open()

        out_list = c_void_p()

        rc = self._lib.talu_db_table_search(
            self._path_bytes,
            _to_bytes(query),
            _to_bytes(doc_type),
            limit,
            byref(out_list),
        )
        check(rc, {"operation": "documents_search", "query": query})

        results: list[DocumentSearchResult] = []

        if out_list.value:
            search_list = CSearchResultList.from_address(out_list.value)

            if search_list.items and search_list.count > 0:
                for i in range(search_list.count):
                    item = search_list.items[i]
                    results.append(
                        DocumentSearchResult(
                            doc_id=_cstr_to_str(item.doc_id),
                            doc_type=_cstr_to_str(item.doc_type),
                            title=_cstr_to_str(item.title),
                            snippet=_cstr_to_str(item.snippet),
                        )
                    )

            self._lib.talu_db_table_free_search_results(c_void_p(out_list.value))

        return results

    def search_batch(
        self,
        queries: list[dict[str, str]],
    ) -> dict[str, list[str]]:
        """
        Batch search documents with multiple queries.

        This is more efficient than multiple `search()` calls when you have
        many queries to execute.

        Args:
            queries: List of query dicts, each with:
                - "id": Query identifier (for result mapping)
                - "text": Search query text
                - "type": Optional document type filter

        Returns
        -------
            Dict mapping query IDs to lists of matching document IDs.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
            ValidationError: If queries format is invalid.

        Example:
            >>> queries = [
            ...     {"id": "q1", "text": "coding", "type": "prompt"},
            ...     {"id": "q2", "text": "review"},
            ... ]
            >>> results = store.search_batch(queries)
            >>> print(results)
            {"q1": ["doc-a", "doc-b"], "q2": ["doc-c"]}
        """
        import json

        from .._bindings import check

        self._check_open()

        # Serialize queries to JSON
        queries_json = json.dumps(queries).encode("utf-8")

        out_results_json = c_void_p()
        out_results_len = c_size_t()

        rc = self._lib.talu_db_table_search_batch(
            self._path_bytes,
            queries_json,
            len(queries_json),
            byref(out_results_json),
            byref(out_results_len),
        )
        check(rc, {"operation": "documents_search_batch"})

        results: dict[str, list[str]] = {}

        if out_results_json.value and out_results_len.value > 0:
            # Read the JSON result
            result_bytes = ctypes.string_at(out_results_json.value, out_results_len.value)
            results = json.loads(result_bytes.decode("utf-8"))

            # Free the JSON buffer
            self._lib.talu_db_table_free_json(out_results_json, out_results_len)

        return results

    # =========================================================================
    # Tag Operations
    # =========================================================================

    def add_tag(
        self,
        doc_id: str,
        tag_id: str,
        *,
        group_id: str | None = None,
    ) -> None:
        """
        Add a tag to a document.

        Args:
            doc_id: Document identifier.
            tag_id: Tag identifier.
            group_id: Group ID for multi-tenant isolation (optional).

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_add_tag(
            self._path_bytes,
            _to_bytes(doc_id),
            _to_bytes(tag_id),
            _to_bytes(group_id),
        )
        check(rc, {"operation": "documents_add_tag", "doc_id": doc_id, "tag_id": tag_id})

    def remove_tag(
        self,
        doc_id: str,
        tag_id: str,
        *,
        group_id: str | None = None,
    ) -> None:
        """
        Remove a tag from a document.

        Args:
            doc_id: Document identifier.
            tag_id: Tag identifier.
            group_id: Group ID for multi-tenant isolation (optional).

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_remove_tag(
            self._path_bytes,
            _to_bytes(doc_id),
            _to_bytes(tag_id),
            _to_bytes(group_id),
        )
        check(rc, {"operation": "documents_remove_tag", "doc_id": doc_id, "tag_id": tag_id})

    def get_tags(self, doc_id: str) -> list[str]:
        """
        Get tags for a document.

        Args:
            doc_id: Document identifier.

        Returns
        -------
            List of tag IDs.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_list = c_void_p()

        rc = self._lib.talu_db_table_get_tags(
            self._path_bytes,
            _to_bytes(doc_id),
            byref(out_list),
        )
        check(rc, {"operation": "documents_get_tags", "doc_id": doc_id})

        return self._extract_string_list(out_list)

    def get_by_tag(self, tag_id: str) -> list[str]:
        """
        Get document IDs by tag.

        Args:
            tag_id: Tag identifier.

        Returns
        -------
            List of document IDs.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_list = c_void_p()

        rc = self._lib.talu_db_table_get_by_tag(
            self._path_bytes,
            _to_bytes(tag_id),
            byref(out_list),
        )
        check(rc, {"operation": "documents_get_by_tag", "tag_id": tag_id})

        return self._extract_string_list(out_list)

    def _extract_string_list(self, out_list: c_void_p) -> list[str]:
        """Extract strings from CStringList and free."""
        from .._native import CStringList

        results: list[str] = []

        if out_list.value:
            string_list = CStringList.from_address(out_list.value)

            if string_list.items and string_list.count > 0:
                # items is c_void_p pointing to array of c_char_p
                items_ptr = cast(string_list.items, POINTER(c_char_p * string_list.count))
                for i in range(string_list.count):
                    item = items_ptr.contents[i]
                    if item:
                        results.append(item.decode("utf-8", errors="replace"))

            self._lib.talu_db_table_free_string_list(c_void_p(out_list.value))

        return results

    # =========================================================================
    # TTL Operations
    # =========================================================================

    def set_ttl(self, doc_id: str, ttl_seconds: int) -> None:
        """
        Set TTL for a document.

        Args:
            doc_id: Document identifier.
            ttl_seconds: Time-to-live in seconds. 0 = never expires.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_set_ttl(
            self._path_bytes,
            _to_bytes(doc_id),
            ttl_seconds,
        )
        check(rc, {"operation": "documents_set_ttl", "doc_id": doc_id})

    def count_expired(self) -> int:
        """
        Count expired documents.

        Returns
        -------
            Number of expired documents.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_count = c_size_t()

        rc = self._lib.talu_db_table_count_expired(
            self._path_bytes,
            byref(out_count),
        )
        check(rc, {"operation": "documents_count_expired"})

        return out_count.value

    def purge_expired(self) -> int:
        """
        Purge expired documents.

        Returns
        -------
            Number of documents purged.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_count = c_size_t()

        rc = self._lib.talu_db_table_purge_expired(
            self._path_bytes,
            byref(out_count),
        )
        check(rc, {"operation": "documents_purge_expired"})

        return out_count.value

    # =========================================================================
    # CDC Operations
    # =========================================================================

    def get_changes(
        self,
        since_seq: int = 0,
        *,
        group_id: str | None = None,
        limit: int = 100,
    ) -> list[ChangeRecord]:
        """
        Get changes since a sequence number (CDC).

        Args:
            since_seq: Sequence number to start from (exclusive).
            group_id: Filter by group ID (optional).
            limit: Maximum number of changes.

        Returns
        -------
            List of ChangeRecord objects.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check
        from .._native import CChangeList

        self._check_open()

        out_list = c_void_p()

        rc = self._lib.talu_db_table_get_changes(
            self._path_bytes,
            since_seq,
            _to_bytes(group_id),
            limit,
            byref(out_list),
        )
        check(rc, {"operation": "documents_get_changes"})

        results: list[ChangeRecord] = []

        if out_list.value:
            change_list = CChangeList.from_address(out_list.value)

            if change_list.items and change_list.count > 0:
                for i in range(change_list.count):
                    item = change_list.items[i]
                    results.append(
                        ChangeRecord(
                            seq_num=item.seq_num,
                            doc_id=_cstr_to_str(item.doc_id),
                            action=ChangeAction(item.action),
                            timestamp_ms=item.timestamp_ms,
                            doc_type=_cstr_to_opt_str(item.doc_type),
                            title=_cstr_to_opt_str(item.title),
                        )
                    )

            self._lib.talu_db_table_free_changes(c_void_p(out_list.value))

        return results

    # =========================================================================
    # Delta Versioning Operations
    # =========================================================================

    def create_delta(
        self,
        base_doc_id: str,
        new_doc_id: str,
        delta_json: str,
        *,
        title: str | None = None,
        tags_text: str | None = None,
        marker: str | None = None,
    ) -> None:
        """
        Create a delta version of a document.

        Delta versions store only the changes from a base document,
        saving storage space for frequently edited documents.

        Args:
            base_doc_id: ID of the base document.
            new_doc_id: ID for the new delta document.
            delta_json: JSON patch/delta content.
            title: Title for the delta (optional, inherits from base).
            tags_text: Tags for the delta (optional).
            marker: Marker for the delta (optional).

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        rc = self._lib.talu_db_table_create_delta(
            self._path_bytes,
            _to_bytes(base_doc_id),
            _to_bytes(new_doc_id),
            _to_bytes(delta_json),
            _to_bytes(title),
            _to_bytes(tags_text),
            _to_bytes(marker),
        )
        check(rc, {"operation": "documents_create_delta", "base_doc_id": base_doc_id})

    def is_delta(self, doc_id: str) -> bool:
        """
        Check if a document is a delta version.

        Args:
            doc_id: Document identifier.

        Returns
        -------
            True if the document is a delta version.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_is_delta = c_bool()

        rc = self._lib.talu_db_table_is_delta(
            self._path_bytes,
            _to_bytes(doc_id),
            byref(out_is_delta),
        )
        check(rc, {"operation": "documents_is_delta", "doc_id": doc_id})

        return out_is_delta.value

    def get_base_id(self, doc_id: str) -> str | None:
        """
        Get the base document ID for a delta.

        Args:
            doc_id: Document identifier.

        Returns
        -------
            Base document ID, or None if not a delta.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        buf_size = 256
        out_buf = ctypes.create_string_buffer(buf_size)

        rc = self._lib.talu_db_table_get_base_id(
            self._path_bytes,
            _to_bytes(doc_id),
            out_buf,
            buf_size,
        )

        if rc == 1:  # Not a delta
            return None
        check(rc, {"operation": "documents_get_base_id", "doc_id": doc_id})

        return out_buf.value.decode("utf-8", errors="replace") or None

    def get_delta_chain(self, doc_id: str) -> list[DocumentRecord]:
        """
        Get the delta chain for a document.

        Returns documents in order from the requested document back to the base.
        The first element is the requested document, the last is the base (full version).

        Args:
            doc_id: Document identifier.

        Returns
        -------
            List of DocumentRecord objects forming the delta chain.
            For non-delta documents, returns a single-element list.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the document doesn't exist or operation fails.

        Example:
            >>> # Get delta chain for a versioned document
            >>> chain = store.get_delta_chain("doc-v3")
            >>> print(f"Chain length: {len(chain)}")
            Chain length: 3
            >>> print(f"Base doc: {chain[-1].doc_id}")
            Base doc: doc-v1
        """
        from .._bindings import check
        from .._native import CDeltaChain

        self._check_open()

        out_chain = c_void_p()

        rc = self._lib.talu_db_table_get_delta_chain(
            self._path_bytes,
            _to_bytes(doc_id),
            byref(out_chain),
        )
        check(rc, {"operation": "documents_get_delta_chain", "doc_id": doc_id})

        results: list[DocumentRecord] = []

        if out_chain.value:
            chain = CDeltaChain.from_address(out_chain.value)

            if chain.items and chain.count > 0:
                for i in range(chain.count):
                    item = chain.items[i]
                    results.append(
                        DocumentRecord(
                            doc_id=_cstr_to_str(item.doc_id),
                            doc_type=_cstr_to_str(item.doc_type),
                            title=_cstr_to_str(item.title),
                            tags_text=_cstr_to_opt_str(item.tags_text),
                            doc_json=_cstr_to_str(item.doc_json),
                            parent_id=_cstr_to_opt_str(item.parent_id),
                            marker=_cstr_to_opt_str(item.marker),
                            group_id=_cstr_to_opt_str(item.group_id),
                            owner_id=_cstr_to_opt_str(item.owner_id),
                            created_at_ms=item.created_at_ms,
                            updated_at_ms=item.updated_at_ms,
                            expires_at_ms=item.expires_at_ms,
                            content_hash=item.content_hash,
                            seq_num=item.seq_num,
                        )
                    )

            self._lib.talu_db_table_free_delta_chain(c_void_p(out_chain.value))

        return results

    # =========================================================================
    # Compaction Operations
    # =========================================================================

    def get_compaction_stats(self) -> CompactionStats:
        """
        Get compaction statistics.

        Returns
        -------
            CompactionStats with document counts and garbage estimates.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check
        from .._native import CCompactionStats

        self._check_open()

        out_stats = CCompactionStats()

        rc = self._lib.talu_db_table_get_compaction_stats(
            self._path_bytes,
            byref(out_stats),
        )
        check(rc, {"operation": "documents_get_compaction_stats"})

        return CompactionStats(
            total_documents=out_stats.total_documents,
            active_documents=out_stats.active_documents,
            expired_documents=out_stats.expired_documents,
            deleted_documents=out_stats.deleted_documents,
            tombstone_count=out_stats.tombstone_count,
            delta_versions=out_stats.delta_versions,
            estimated_garbage_bytes=out_stats.estimated_garbage_bytes,
        )

    def get_garbage_candidates(self) -> list[str]:
        """
        Get document IDs that are candidates for garbage collection.

        Returns
        -------
            List of document IDs.

        Raises
        ------
            StateError: If the store is closed.
            StorageError: If the operation fails.
        """
        from .._bindings import check

        self._check_open()

        out_list = c_void_p()

        rc = self._lib.talu_db_table_get_garbage_candidates(
            self._path_bytes,
            byref(out_list),
        )
        check(rc, {"operation": "documents_get_garbage_candidates"})

        return self._extract_string_list(out_list)

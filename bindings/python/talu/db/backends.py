"""
Storage backend implementations for TaluDB.

This module provides:

- Database: Built-in Zig-backed storage (default)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from talu.types import SessionRecord

__all__ = [
    "Database",
]


class Database:
    """
    Built-in Zig-backed storage.

    Items are stored in Zig memory by default. TaluDB persistence is
    available via the ``talu://`` scheme.

    Args:
        location: Storage location. Supported:
            - ":memory:" (default, in-memory only)
            - "talu://<path>" (TaluDB on-disk persistence)

    Example:
        >>> # Default in-memory storage
        >>> chat = talu.Chat("model", storage=Database())
        >>>
        >>> # Explicit in-memory (same as default)
        >>> chat = talu.Chat("model", storage=Database(":memory:"))
        >>>
        >>> # TaluDB persistence
        >>> chat = talu.Chat("model", storage=Database("talu://./my-db"))

    Note:
        Database is the default when no storage is specified.
        All item operations go through Zig for maximum performance.
    """

    def __init__(self, location: str = ":memory:") -> None:
        """
        Initialize storage.

        Args:
            location: Storage location. Use ":memory:" or "talu://<path>".

        Raises
        ------
            ValueError: If storage type is not supported.
        """
        if location == ":memory:":
            self._location = location
            return

        if location.startswith("talu://"):
            if len(location) <= len("talu://"):
                from ..exceptions import ValidationError

                raise ValidationError("TaluDB location must include a path after 'talu://'.")
            self._location = location
            return

        from ..exceptions import ValidationError

        raise ValidationError(
            f"Database '{location}' not yet supported. Use ':memory:' or 'talu://<path>'."
        )

    @property
    def location(self) -> str:
        """The storage location string."""
        return self._location

    def __repr__(self) -> str:
        return f"Database({self._location!r})"

    def _require_taludb(self) -> str:
        """Require TaluDB storage and return the db_path."""
        if not self._location.startswith("talu://"):
            from ..exceptions import ValidationError

            raise ValidationError("Session listing requires TaluDB storage (talu://...)")
        return self._location[len("talu://") :]

    def list_sessions_by_source(
        self,
        source_doc_id: str,
        *,
        limit: int = 50,
        before_updated_at_ms: int = 0,
        before_session_id: str | None = None,
    ) -> list[SessionRecord]:
        """
        List sessions derived from a specific prompt document.

        This is a convenience method for lineage queries - finding all conversations
        that were spawned from a particular prompt document.

        Args:
            source_doc_id: Document ID to filter by.
            limit: Maximum number of sessions to return (default 50).
            before_updated_at_ms: Cursor for pagination (0 = start from newest).
            before_session_id: Session ID cursor for pagination.

        Returns
        -------
            List of session records ordered by updated_at descending.

        Raises
        ------
            ValidationError: If storage is not TaluDB.
            StateError: If the query fails.

        Example:
            >>> db = Database("talu://./my-db")
            >>> sessions = db.list_sessions_by_source("doc_abc123")
            >>> for s in sessions:
            ...     print(s["session_id"], s["title"])
        """
        import ctypes

        from .._bindings import check, get_lib
        from .._native import CSessionList
        from ..chat._bindings import c_session_record_to_session_record

        db_path = self._require_taludb()
        lib = get_lib()

        out_sessions = ctypes.POINTER(CSessionList)()
        out_ptr = ctypes.pointer(out_sessions)

        rc = lib.talu_storage_list_sessions_by_source(
            db_path.encode("utf-8"),
            source_doc_id.encode("utf-8"),
            limit,
            before_updated_at_ms,
            before_session_id.encode("utf-8") if before_session_id else None,
            out_ptr,
        )
        check(rc)

        try:
            if not out_sessions:
                return []

            results: list[SessionRecord] = []
            session_list = out_sessions.contents
            for i in range(session_list.count):
                record = session_list.sessions[i]
                results.append(c_session_record_to_session_record(record))
            return results
        finally:
            if out_sessions:
                lib.talu_storage_free_sessions(out_sessions)

    def list_sessions(
        self,
        *,
        limit: int = 50,
        before_updated_at_ms: int = 0,
        before_session_id: str | None = None,
        group_id: str | None = None,
        search_query: str | None = None,
        tags_filter: str | None = None,
        tags_filter_any: str | None = None,
        marker_filter: str | None = None,
        marker_filter_any: str | None = None,
        model_filter: str | None = None,
        created_after_ms: int = 0,
        created_before_ms: int = 0,
        updated_after_ms: int = 0,
        updated_before_ms: int = 0,
        has_tags: bool | None = None,
        source_doc_id: str | None = None,
    ) -> list[SessionRecord]:
        """
        List sessions with advanced filtering.

        Args:
            limit: Maximum number of sessions to return (default 50).
            before_updated_at_ms: Cursor for pagination (0 = start from newest).
            before_session_id: Session ID cursor for pagination.
            group_id: Filter by group ID.
            search_query: Full-text search in title/content.
            tags_filter: Require ALL these tags (comma-separated).
            tags_filter_any: Require ANY of these tags (comma-separated).
            marker_filter: Require ALL these markers (comma-separated).
            marker_filter_any: Require ANY of these markers (comma-separated).
            model_filter: Filter by model name.
            created_after_ms: Filter by creation time (inclusive).
            created_before_ms: Filter by creation time (exclusive).
            updated_after_ms: Filter by update time (inclusive).
            updated_before_ms: Filter by update time (exclusive).
            has_tags: Filter by presence of tags (True/False/None).
            source_doc_id: Filter by source document ID.

        Returns
        -------
            List of session records ordered by updated_at descending.

        Raises
        ------
            ValidationError: If storage is not TaluDB.
            StateError: If the query fails.

        Example:
            >>> db = Database("talu://./my-db")
            >>> # Find recent sessions with a specific tag
            >>> sessions = db.list_sessions(tags_filter="project:acme", limit=10)
            >>> # Search for sessions containing "bug fix"
            >>> sessions = db.list_sessions(search_query="bug fix")
        """
        import ctypes

        from .._bindings import check, get_lib
        from .._native import CSessionList
        from ..chat._bindings import c_session_record_to_session_record

        db_path = self._require_taludb()
        lib = get_lib()

        out_sessions = ctypes.POINTER(CSessionList)()
        out_ptr = ctypes.pointer(out_sessions)

        # Convert has_tags to C convention: 1=True, 0=False, -1=None
        has_tags_c = -1 if has_tags is None else (1 if has_tags else 0)

        rc = lib.talu_storage_list_sessions_ex(
            db_path.encode("utf-8"),
            limit,
            before_updated_at_ms,
            before_session_id.encode("utf-8") if before_session_id else None,
            group_id.encode("utf-8") if group_id else None,
            search_query.encode("utf-8") if search_query else None,
            tags_filter.encode("utf-8") if tags_filter else None,
            tags_filter_any.encode("utf-8") if tags_filter_any else None,
            marker_filter.encode("utf-8") if marker_filter else None,
            marker_filter_any.encode("utf-8") if marker_filter_any else None,
            model_filter.encode("utf-8") if model_filter else None,
            created_after_ms,
            created_before_ms,
            updated_after_ms,
            updated_before_ms,
            has_tags_c,
            source_doc_id.encode("utf-8") if source_doc_id else None,
            out_ptr,
        )
        check(rc)

        try:
            if not out_sessions:
                return []

            results: list[SessionRecord] = []
            session_list = out_sessions.contents
            for i in range(session_list.count):
                record = session_list.sessions[i]
                results.append(c_session_record_to_session_record(record))
            return results
        finally:
            if out_sessions:
                lib.talu_storage_free_sessions(out_sessions)

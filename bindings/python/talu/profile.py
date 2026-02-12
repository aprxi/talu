"""Profile-based chat session persistence for Python users."""

from __future__ import annotations

__all__ = ["Profile", "new_session_id"]

import json
import os
import uuid
from pathlib import Path

from ._bindings import check, get_lib
from .db import Database
from .types import SessionRecord

_DEFAULT_PROFILE = "dev"
_MANIFEST = {"version": 1, "segments": [], "last_compaction_ts": 0}


def _db_base() -> Path:
    """Return the base directory for profile buckets."""
    return Path.home() / ".talu" / "db"


def _resolve_profile_name(name: str | None) -> str:
    """Resolve profile name from argument/env/default."""
    if name is not None:
        return name
    return os.environ.get("TALU_PROFILE", _DEFAULT_PROFILE)


def _ensure_bucket(path: Path) -> None:
    """Ensure profile bucket exists and is initialized."""
    path.mkdir(parents=True, exist_ok=True)

    key_path = path / "store.key"
    if not key_path.exists():
        key_path.write_bytes(os.urandom(16))

    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(_MANIFEST), encoding="utf-8")


def new_session_id() -> str:
    """Generate a new RFC 4122 UUIDv4 session ID."""
    return str(uuid.uuid4())


class Profile:
    """A named storage namespace for chat sessions."""

    def __init__(self, name: str | None = None):
        self._name = _resolve_profile_name(name)
        self._path = _db_base() / self._name
        _ensure_bucket(self._path)
        self._database = Database(f"talu://{self._path}")

    @property
    def name(self) -> str:
        """Profile name."""
        return self._name

    @property
    def path(self) -> Path:
        """Storage directory path (~/.talu/db/<name>/)."""
        return self._path

    @property
    def database(self) -> Database:
        """Underlying Database instance for this profile."""
        return self._database

    def sessions(self, *, search: str | None = None, limit: int = 50) -> list[SessionRecord]:
        """List sessions in this profile.

        Args:
            search: Filter sessions by text content.
            limit: Maximum number of sessions to return.
        """
        return self._database.list_sessions(search_query=search, limit=limit)

    def delete(self, session_id: str) -> None:
        """Delete a session from this profile.

        Args:
            session_id: ID of the session to delete.
        """
        rc = get_lib().talu_storage_delete_session(
            str(self._path).encode("utf-8"),
            session_id.encode("utf-8"),
        )
        check(rc, {"db_path": str(self._path), "session_id": session_id})

    def __repr__(self) -> str:
        return f"Profile({self._name!r})"

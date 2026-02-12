"""Tests for profile-based session persistence."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from talu import Chat, Client, Profile, list_sessions
from talu.db import Database
from talu.exceptions import ValidationError


def _set_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    """Point profile storage to a temporary home directory."""
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def test_profile_creates_bucket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)

    profile = Profile("work")
    bucket = tmp_path / ".talu" / "db" / "work"

    assert profile.name == "work"
    assert profile.path == bucket
    assert bucket.is_dir()

    key_bytes = (bucket / "store.key").read_bytes()
    assert len(key_bytes) == 16

    manifest = json.loads((bucket / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"version": 1, "segments": [], "last_compaction_ts": 0}


def test_profile_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)

    first = Profile("work")
    first_key = (first.path / "store.key").read_bytes()

    second = Profile("work")
    second_key = (second.path / "store.key").read_bytes()

    assert first.path == second.path
    assert first_key == second_key


def test_profile_default_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    monkeypatch.delenv("TALU_PROFILE", raising=False)

    profile = Profile()
    assert profile.name == "dev"
    assert profile.path == tmp_path / ".talu" / "db" / "dev"


def test_profile_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setenv("TALU_PROFILE", "default")

    profile = Profile()
    assert profile.name == "default"
    assert profile.path == tmp_path / ".talu" / "db" / "default"


def test_profile_explicit_name_wins_over_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)
    monkeypatch.setenv("TALU_PROFILE", "default")

    profile = Profile("work")
    assert profile.name == "work"


def test_chat_with_profile_generates_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = Profile("work")

    chat = Chat(profile=profile)
    try:
        session_id = chat.session_id
        assert session_id is not None
        uuid.UUID(session_id)

        chat.append("user", "Hello profile storage")
    finally:
        chat.close()

    sessions = profile.sessions(limit=20)
    assert any(record["session_id"] == session_id for record in sessions)


def test_chat_with_profile_uses_explicit_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = Profile("work")

    chat = Chat(profile=profile, session_id="manual-session-id")
    try:
        assert chat.session_id == "manual-session-id"
        chat.append("user", "hello")
    finally:
        chat.close()

    sessions = profile.sessions(limit=20)
    assert any(record["session_id"] == "manual-session-id" for record in sessions)


def test_chat_without_profile_stays_ephemeral(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)

    chat = Chat()
    try:
        chat.append("user", "ephemeral chat")
    finally:
        chat.close()

    assert not (tmp_path / ".talu").exists()


def test_profile_and_storage_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = Profile("work")

    with pytest.raises(ValidationError, match="Cannot use both 'profile' and 'storage'"):
        Chat(profile=profile, storage=Database())


def test_client_profile_is_inherited_by_chats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = Profile("work")

    client = Client("test-model", profile=profile)
    try:
        chat = client.chat()
        try:
            session_id = chat.session_id
            assert session_id is not None
            chat.append("user", "persist through client profile")
        finally:
            chat.close()
    finally:
        client.close()

    sessions = profile.sessions(limit=20)
    assert any(record["session_id"] == session_id for record in sessions)


def test_profile_delete_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = Profile("work")
    session_id = "to-delete"

    chat = Chat(profile=profile, session_id=session_id)
    try:
        chat.append("user", "this should be deleted")
    finally:
        chat.close()

    assert any(record["session_id"] == session_id for record in profile.sessions(limit=20))

    profile.delete(session_id)

    assert all(record["session_id"] != session_id for record in profile.sessions(limit=20))


def test_top_level_list_sessions_shortcut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _set_home(monkeypatch, tmp_path)
    profile = Profile("work")
    session_id = "shortcut-session"

    chat = Chat(profile=profile, session_id=session_id)
    try:
        chat.append("user", "persist through top-level shortcut")
    finally:
        chat.close()

    sessions = list_sessions(profile="work", limit=20)
    assert any(record["session_id"] == session_id for record in sessions)

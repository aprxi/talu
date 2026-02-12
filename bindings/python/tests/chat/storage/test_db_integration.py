"""TaluDB integration tests for Chat storage."""

from __future__ import annotations

from pathlib import Path

from talu import Chat
from talu.db import Database


def _db_location(root: Path) -> str:
    return f"talu://{root}"


def test_db_persistence(tmp_path: Path) -> None:
    db_root = tmp_path / "taludb"
    db_root.mkdir()

    chat = Chat(storage=Database(_db_location(db_root)), session_id="test-session")
    chat.append("user", "Hello TaluDB")
    chat.close()

    restored = Chat(storage=Database(_db_location(db_root)), session_id="test-session")
    try:
        assert len(restored.items) == 1
        assert restored.items[0].text == "Hello TaluDB"
    finally:
        restored.close()


def test_db_session_isolation(tmp_path: Path) -> None:
    db_root = tmp_path / "taludb"
    db_root.mkdir()
    location = _db_location(db_root)

    chat_a = Chat(storage=Database(location), session_id="A")
    chat_a.append("user", "Message A")
    chat_a.close()

    chat_b = Chat(storage=Database(location), session_id="B")
    chat_b.append("user", "Message B")
    chat_b.close()

    restored_a = Chat(storage=Database(location), session_id="A")
    try:
        assert [item.text for item in restored_a.items] == ["Message A"]
    finally:
        restored_a.close()

    restored_b = Chat(storage=Database(location), session_id="B")
    try:
        assert [item.text for item in restored_b.items] == ["Message B"]
    finally:
        restored_b.close()

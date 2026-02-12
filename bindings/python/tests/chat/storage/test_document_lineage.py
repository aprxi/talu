"""Tests for document lineage features (Phase 12-14).

This includes:
- prompt_id property (get/set)
- inherit_tags() method
- list_sessions_by_source()
- list_sessions() with source_doc_id filter
"""

import pytest

from talu.chat import Chat
from talu.db import Database
from talu.exceptions import StateError, ValidationError


class TestPromptIdProperty:
    """Tests for prompt_id property on Chat."""

    def test_prompt_id_default_is_none(self) -> None:
        """prompt_id is None by default."""
        with Chat(offline=True) as chat:
            assert chat.prompt_id is None

    def test_prompt_id_can_be_set_on_construction(self) -> None:
        """prompt_id can be set via constructor."""
        with Chat(offline=True, prompt_id="my-prompt-doc") as chat:
            assert chat.prompt_id == "my-prompt-doc"

    def test_prompt_id_can_be_modified(self) -> None:
        """prompt_id can be modified after construction."""
        with Chat(offline=True) as chat:
            assert chat.prompt_id is None
            chat.prompt_id = "new-prompt-id"
            assert chat.prompt_id == "new-prompt-id"

    def test_prompt_id_can_be_cleared(self) -> None:
        """prompt_id can be set to None to clear it."""
        with Chat(offline=True, prompt_id="original-prompt") as chat:
            assert chat.prompt_id == "original-prompt"
            chat.prompt_id = None
            assert chat.prompt_id is None

    def test_prompt_id_get_after_close_returns_cached(self) -> None:
        """Getting prompt_id after close returns cached value."""
        chat = Chat(offline=True, prompt_id="test-prompt")
        chat.close()
        # Accessing the cached value should still work
        assert chat.prompt_id == "test-prompt"

    def test_prompt_id_set_raises_on_closed_chat(self) -> None:
        """Setting prompt_id on closed chat raises StateError."""
        chat = Chat(offline=True)
        chat.close()
        with pytest.raises(StateError, match="Chat is closed"):
            chat.prompt_id = "new-prompt"


class TestInheritTags:
    """Tests for inherit_tags() method."""

    def test_inherit_tags_requires_taludb_storage(self) -> None:
        """inherit_tags() raises ValidationError without TaluDB."""
        with Chat(offline=True) as chat:
            with pytest.raises(ValidationError, match="requires TaluDB storage"):
                chat.inherit_tags()

    def test_inherit_tags_raises_on_closed_chat(self) -> None:
        """inherit_tags() raises StateError on closed chat."""
        chat = Chat(offline=True)
        chat.close()
        with pytest.raises(StateError, match="Chat is closed"):
            chat.inherit_tags()


class TestListSessionsBySource:
    """Tests for Database.list_sessions_by_source()."""

    def test_list_sessions_by_source_requires_taludb(self) -> None:
        """list_sessions_by_source() raises ValidationError without TaluDB."""
        db = Database(":memory:")
        with pytest.raises(ValidationError, match="requires TaluDB storage"):
            db.list_sessions_by_source("doc-123")

    def test_list_sessions_by_source_empty_result(self, tmp_path) -> None:
        """list_sessions_by_source() returns empty list when no matches."""
        db_path = str(tmp_path / "test_db")
        db = Database(f"talu://{db_path}")
        result = db.list_sessions_by_source("nonexistent-doc")
        assert result == []

    def test_list_sessions_by_source_returns_matching_sessions(self, tmp_path) -> None:
        """list_sessions_by_source() returns sessions with matching source_doc_id."""
        db_path = str(tmp_path / "test_db")
        storage = Database(f"talu://{db_path}")

        # Create sessions with different source_doc_ids
        with Chat(
            offline=True,
            session_id="session-1",
            storage=storage,
            source_doc_id="doc-A",
        ):
            pass  # Session created

        with Chat(
            offline=True,
            session_id="session-2",
            storage=storage,
            source_doc_id="doc-A",
        ):
            pass  # Session created

        with Chat(
            offline=True,
            session_id="session-3",
            storage=storage,
            source_doc_id="doc-B",
        ):
            pass  # Session created

        # Query sessions by source
        results = storage.list_sessions_by_source("doc-A")
        session_ids = [s["session_id"] for s in results]
        assert len(results) == 2
        assert "session-1" in session_ids
        assert "session-2" in session_ids
        assert "session-3" not in session_ids


class TestListSessions:
    """Tests for Database.list_sessions()."""

    def test_list_sessions_requires_taludb(self) -> None:
        """list_sessions() raises ValidationError without TaluDB."""
        db = Database(":memory:")
        with pytest.raises(ValidationError, match="requires TaluDB storage"):
            db.list_sessions()

    def test_list_sessions_empty_database(self, tmp_path) -> None:
        """list_sessions() returns empty list on empty database."""
        db_path = str(tmp_path / "test_db")
        db = Database(f"talu://{db_path}")
        result = db.list_sessions()
        assert result == []

    def test_list_sessions_with_source_doc_id_filter(self, tmp_path) -> None:
        """list_sessions() can filter by source_doc_id."""
        db_path = str(tmp_path / "test_db")
        storage = Database(f"talu://{db_path}")

        # Create sessions with different source_doc_ids
        with Chat(
            offline=True,
            session_id="session-1",
            storage=storage,
            source_doc_id="doc-A",
        ):
            pass

        with Chat(
            offline=True,
            session_id="session-2",
            storage=storage,
            source_doc_id="doc-B",
        ):
            pass

        # Query with source_doc_id filter
        results = storage.list_sessions(source_doc_id="doc-A")
        assert len(results) == 1
        assert results[0]["session_id"] == "session-1"

    def test_list_sessions_with_limit(self, tmp_path) -> None:
        """list_sessions() respects limit parameter."""
        db_path = str(tmp_path / "test_db")
        storage = Database(f"talu://{db_path}")

        # Create multiple sessions
        for i in range(5):
            with Chat(
                offline=True,
                session_id=f"session-{i}",
                storage=storage,
            ):
                pass

        # Query with limit
        results = storage.list_sessions(limit=2)
        assert len(results) == 2

    def test_list_sessions_returns_all_without_filter(self, tmp_path) -> None:
        """list_sessions() returns all sessions when no filter provided."""
        db_path = str(tmp_path / "test_db")
        storage = Database(f"talu://{db_path}")

        # Create sessions
        with Chat(offline=True, session_id="session-1", storage=storage):
            pass
        with Chat(offline=True, session_id="session-2", storage=storage):
            pass

        results = storage.list_sessions()
        assert len(results) == 2

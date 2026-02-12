"""Tests for source_doc_id (document lineage tracking) in Chat sessions."""

import pytest

from talu.chat import Chat
from talu.db import Database


class TestSourceDocIdProperty:
    """Tests for source_doc_id property on Chat."""

    def test_source_doc_id_default_is_none(self) -> None:
        """source_doc_id is None by default."""
        with Chat(offline=True) as chat:
            assert chat.source_doc_id is None

    def test_source_doc_id_can_be_set_on_construction(self) -> None:
        """source_doc_id can be set via constructor."""
        with Chat(offline=True, source_doc_id="my-prompt-doc") as chat:
            assert chat.source_doc_id == "my-prompt-doc"

    def test_source_doc_id_can_be_modified(self) -> None:
        """source_doc_id can be modified after construction."""
        with Chat(offline=True) as chat:
            assert chat.source_doc_id is None
            chat.source_doc_id = "new-doc-id"
            assert chat.source_doc_id == "new-doc-id"

    def test_source_doc_id_can_be_cleared(self) -> None:
        """source_doc_id can be set to None to clear lineage."""
        with Chat(offline=True, source_doc_id="original-doc") as chat:
            assert chat.source_doc_id == "original-doc"
            chat.source_doc_id = None
            assert chat.source_doc_id is None


class TestSourceDocIdWithStorage:
    """Tests for source_doc_id with TaluDB storage."""

    def test_source_doc_id_persists_in_session_record(self, tmp_path) -> None:
        """source_doc_id is included in session records when using TaluDB."""
        db_path = str(tmp_path / "test_db")
        storage = Database(f"talu://{db_path}")

        with Chat(
            offline=True,
            session_id="test-session-1",
            storage=storage,
            source_doc_id="persona-v1",
        ) as chat:
            # source_doc_id is set
            assert chat.source_doc_id == "persona-v1"


class TestSourceDocIdClosed:
    """Tests for source_doc_id on closed chats."""

    def test_source_doc_id_get_after_close(self) -> None:
        """Getting source_doc_id after close still works (cached value)."""
        chat = Chat(offline=True, source_doc_id="test-doc")
        chat.close()
        # Accessing the cached value should still work
        assert chat.source_doc_id == "test-doc"

    def test_source_doc_id_set_raises_on_closed_chat(self) -> None:
        """Setting source_doc_id on closed chat raises StateError."""
        from talu.exceptions import StateError

        chat = Chat(offline=True)
        chat.close()
        with pytest.raises(StateError, match="Chat is closed"):
            chat.source_doc_id = "new-doc"

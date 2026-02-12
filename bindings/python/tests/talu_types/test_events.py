"""
Tests for talu/types/events.py.

Tests for storage event TypedDicts: StorageEvent, DeleteItemEvent, etc.
"""

from talu.types import (
    ClearItemsEvent,
    DeleteItemEvent,
    ForkEvent,
    StorageEvent,
)


class TestDeleteItemEvent:
    """Tests for DeleteItemEvent TypedDict."""

    def test_delete_item_event(self):
        """DeleteItemEvent has item_id and deleted_at_ms."""
        event: DeleteItemEvent = {
            "item_id": 42,
            "deleted_at_ms": 1705123456789,
        }
        assert event["item_id"] == 42
        assert event["deleted_at_ms"] == 1705123456789


class TestClearItemsEvent:
    """Tests for ClearItemsEvent TypedDict."""

    def test_clear_items_event(self):
        """ClearItemsEvent has cleared_at_ms and keep_context."""
        event: ClearItemsEvent = {
            "cleared_at_ms": 1705123456789,
            "keep_context": True,
        }
        assert event["cleared_at_ms"] == 1705123456789
        assert event["keep_context"] is True

    def test_clear_items_without_context(self):
        """ClearItemsEvent with keep_context=False."""
        event: ClearItemsEvent = {
            "cleared_at_ms": 1705123456789,
            "keep_context": False,
        }
        assert event["keep_context"] is False


class TestForkEvent:
    """Tests for ForkEvent TypedDict."""

    def test_fork_event(self):
        """ForkEvent has fork_id and session_id."""
        event: ForkEvent = {
            "fork_id": 1,
            "session_id": "session_abc",
        }
        assert event["fork_id"] == 1
        assert event["session_id"] == "session_abc"


class TestStorageEvent:
    """Tests for StorageEvent TypedDict."""

    def test_put_items_event(self):
        """StorageEvent with PutItems."""
        event: StorageEvent = {
            "PutItems": [
                {
                    "item_id": 0,
                    "created_at_ms": 1705123456789,
                    "status": "completed",
                    "item_type": "message",
                    "variant": {
                        "role": "user",
                        "status": "completed",
                        "content": [{"type": "input_text", "text": "Hello!"}],
                    },
                }
            ]
        }
        assert len(event["PutItems"]) == 1
        assert event["PutItems"][0]["item_type"] == "message"

    def test_delete_item_event(self):
        """StorageEvent with DeleteItem."""
        event: StorageEvent = {"DeleteItem": {"item_id": 0, "deleted_at_ms": 1705123456900}}
        assert event["DeleteItem"]["item_id"] == 0

    def test_clear_items_event(self):
        """StorageEvent with ClearItems."""
        event: StorageEvent = {"ClearItems": {"cleared_at_ms": 1705123457000, "keep_context": True}}
        assert event["ClearItems"]["keep_context"] is True

    def test_put_session_event(self):
        """StorageEvent with PutSession."""
        event: StorageEvent = {
            "PutSession": {
                "session_id": "user_123",
                "title": "Weather Chat",
                "config": {"temperature": 0.7},
                "created_at_ms": 1705123456000,
                "updated_at_ms": 1705123456789,
            }
        }
        assert event["PutSession"]["session_id"] == "user_123"

    def test_begin_fork_event(self):
        """StorageEvent with BeginFork."""
        event: StorageEvent = {"BeginFork": {"fork_id": 1, "session_id": "session_abc"}}
        assert event["BeginFork"]["fork_id"] == 1

    def test_end_fork_event(self):
        """StorageEvent with EndFork."""
        event: StorageEvent = {"EndFork": {"fork_id": 1, "session_id": "session_abc"}}
        assert event["EndFork"]["session_id"] == "session_abc"

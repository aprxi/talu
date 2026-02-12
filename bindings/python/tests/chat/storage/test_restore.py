from talu import Chat
from talu.types import ItemStatus


class TestStorageRestore:
    def test_from_items_restores_ids_and_timestamps(self):
        items = [
            {
                "item_id": 7,
                "created_at_ms": 123456,
                "item_type": "message",
                "status": "completed",
                "hidden": False,
                "pinned": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "prefill_ns": 0,
                "generation_ns": 0,
                "finish_reason": None,
                "parent_item_id": None,
                "origin_session_id": None,
                "origin_item_id": None,
                "ttl_ts": 0,
                "variant": {
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
            }
        ]

        chat = Chat._from_items(items)
        assert len(chat.items) == 1
        item = chat.items[0]
        assert item.id == 7
        assert item.created_at_ms == 123456
        assert item.status == ItemStatus.COMPLETED

    def test_load_storage_records_empty_list(self):
        """Empty list is a valid input that returns early without error."""
        chat = Chat()
        chat._load_storage_records([])
        assert len(chat.items) == 0

    def test_from_dict_prefers_items_when_present(self):
        items = [
            {
                "item_id": 1,
                "created_at_ms": 424242,
                "item_type": "message",
                "status": "completed",
                "hidden": False,
                "pinned": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "prefill_ns": 0,
                "generation_ns": 0,
                "finish_reason": None,
                "parent_item_id": None,
                "origin_session_id": None,
                "origin_item_id": None,
                "ttl_ts": 0,
                "variant": {
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "Saved"}],
                },
            }
        ]
        data = {
            "config": {"temperature": 0.7, "max_tokens": 16},
            "messages": [{"role": "user", "content": "Ignored"}],
            "items": items,
        }

        chat = Chat.from_dict(data)
        assert len(chat.items) == 1
        item = chat.items[0]
        assert item.created_at_ms == 424242

    def test_truncate_preserves_item_metadata(self):
        items = [
            {
                "item_id": 3,
                "created_at_ms": 101,
                "item_type": "message",
                "status": "waiting",
                "hidden": False,
                "pinned": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "prefill_ns": 0,
                "generation_ns": 0,
                "finish_reason": None,
                "parent_item_id": None,
                "origin_session_id": None,
                "origin_item_id": None,
                "ttl_ts": 0,
                "variant": {
                    "role": "user",
                    "status": "waiting",
                    "content": [{"type": "input_text", "text": "First"}],
                },
            },
            {
                "item_id": 4,
                "created_at_ms": 202,
                "item_type": "message",
                "status": "completed",
                "hidden": False,
                "pinned": False,
                "input_tokens": 0,
                "output_tokens": 0,
                "prefill_ns": 0,
                "generation_ns": 0,
                "finish_reason": None,
                "parent_item_id": None,
                "origin_session_id": None,
                "origin_item_id": None,
                "ttl_ts": 0,
                "variant": {
                    "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "Second"}],
                },
            },
        ]

        chat = Chat._from_items(items)
        chat._truncate_to(0)

        assert len(chat.items) == 1
        item = chat.items[0]
        assert item.created_at_ms == 101
        assert item.status == ItemStatus.WAITING

    def test_from_dict_uses_lossy_messages_when_no_items(self):
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "config": {"temperature": 0.4, "max_tokens": 8},
        }
        chat = Chat.from_dict(data)
        assert len(chat.items) == 2

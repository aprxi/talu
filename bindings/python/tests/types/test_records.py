"""Smoke tests for types records module.

Maps to: talu/types/records.py
"""

from talu.types.records import ItemRecord


def test_records_module_exports() -> None:
    item: ItemRecord = {
        "item_id": 1,
        "created_at_ms": 1,
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
            "content": [{"type": "input_text", "text": "hello"}],
        },
    }
    assert item["item_type"] == "message"


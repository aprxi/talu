"""Smoke tests for types events module.

Maps to: talu/types/events.py
"""

from talu.types.events import StorageEvent


def test_events_module_exports() -> None:
    event: StorageEvent = {"ClearItems": {"cleared_at_ms": 1, "keep_context": True}}
    assert "ClearItems" in event


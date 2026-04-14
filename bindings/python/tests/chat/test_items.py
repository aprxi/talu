"""Smoke tests for chat items module.

Maps to: talu/chat/items.py
"""

from talu.chat.items import ConversationItems


def test_items_module_exports() -> None:
    assert ConversationItems.__name__ == "ConversationItems"


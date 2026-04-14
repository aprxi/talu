"""Smoke tests for types items module.

Maps to: talu/types/items.py
"""

from talu.types.items import MessageItem


def test_items_module_exports() -> None:
    item = MessageItem.create("user", "hello")
    assert item.role.name.lower() == "user"
    assert item.text == "hello"


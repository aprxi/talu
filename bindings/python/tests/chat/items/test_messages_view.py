"""Tests for Items API - read-only view into Zig memory.

Note: items is READ-ONLY from Python. To test with items, use
Chat.from_dict() to create chats with pre-populated history.

Items are typed Python objects (MessageItem, FunctionCallItem, etc.) that
provide zero-copy access to conversation data from Zig memory.
"""

import pytest

from talu import Chat
from talu.types import MessageItem, MessageRole


class TestItemsBasic:
    """Tests for basic Items operations."""

    def test_empty_items(self):
        """Empty chat has empty items."""
        chat = Chat()
        assert len(chat.items) == 0
        assert list(chat.items) == []

    def test_len_with_system(self):
        """len includes system prompt."""
        chat = Chat(system="You are helpful.")
        assert len(chat.items) == 1

    def test_getitem_with_system(self):
        """System prompt is at index 0."""
        chat = Chat(system="Be helpful.")
        item = chat.items[0]
        assert isinstance(item, MessageItem)
        assert item.role == MessageRole.SYSTEM
        assert item.text == "Be helpful."

    def test_negative_indexing(self):
        """Negative indices work."""
        chat = Chat(system="System")
        item = chat.items[-1]
        assert isinstance(item, MessageItem)
        assert item.role == MessageRole.SYSTEM
        assert item.text == "System"

    def test_index_error(self):
        """Out of range raises IndexError."""
        chat = Chat()
        with pytest.raises(IndexError):
            _ = chat.items[0]


class TestItemsType:
    """Tests for Items type and identity."""

    def test_items_type(self):
        """items is a ConversationItems instance."""
        chat = Chat()
        from talu.chat import ConversationItems

        assert isinstance(chat.items, ConversationItems)

    def test_items_returns_new_view(self):
        """Each access to .items returns a new view object."""
        chat = Chat()
        items1 = chat.items
        items2 = chat.items
        # New view object each time, but same underlying data
        assert items1 is not items2
        # But they have same length (same data)
        assert len(items1) == len(items2)


class TestItemsSystem:
    """Tests for system prompt access."""

    def test_get_system(self):
        """system property returns content."""
        chat = Chat(system="You are helpful.")
        assert chat.items.system == "You are helpful."

    def test_get_system_none(self):
        """system property returns None when not set."""
        chat = Chat()
        assert chat.items.system is None


class TestItemsLast:
    """Tests for last item access."""

    def test_last_with_system(self):
        """last returns system when only item."""
        chat = Chat(system="System")
        item = chat.items.last
        assert item is not None
        assert isinstance(item, MessageItem)
        assert item.role == MessageRole.SYSTEM
        assert item.text == "System"

    def test_last_empty(self):
        """last returns None when empty."""
        chat = Chat()
        assert chat.items.last is None


class TestItemsIteration:
    """Tests for iterating over items."""

    def test_iterate(self):
        """Can iterate over items."""
        chat = Chat(system="Be helpful.")
        items = list(chat.items)
        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert items[0].role == MessageRole.SYSTEM

    def test_list_conversion(self):
        """list() returns Python list of items."""
        chat = Chat(system="System")
        result = list(chat.items)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], MessageItem)
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].text == "System"


class TestItemsSlice:
    """Tests for slice access."""

    def test_slice_empty(self):
        """Slice of empty items."""
        chat = Chat()
        assert chat.items[:] == []

    def test_slice_all(self):
        """Slice all items."""
        chat = Chat(system="System")
        items = chat.items[:]
        assert len(items) == 1
        assert isinstance(items[0], MessageItem)
        assert items[0].role == MessageRole.SYSTEM
        assert items[0].text == "System"

    def test_slice_range(self):
        """Slice with range."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        )
        items = chat.items[1:]
        assert len(items) == 2
        assert isinstance(items[0], MessageItem)
        assert items[0].role == MessageRole.USER
        assert items[0].text == "Hello"
        assert isinstance(items[1], MessageItem)
        assert items[1].role == MessageRole.ASSISTANT
        assert items[1].text == "Hi!"


class TestItemsComparison:
    """Tests for Items comparison operations."""

    def test_check_single_item(self):
        """Can check item properties."""
        chat = Chat(system="System")
        items = list(chat.items)
        assert len(items) == 1
        assert items[0].role == MessageRole.SYSTEM
        assert items[0].text == "System"

    def test_check_different_content(self):
        """Different contents can be detected."""
        chat = Chat(system="System")
        assert len(chat.items) != 0
        item = chat.items[0]
        assert not (item.role == MessageRole.USER and item.text == "Hello")


class TestItemsRepr:
    """Tests for Items repr."""

    def test_repr_empty(self):
        """Empty items repr."""
        chat = Chat()
        r = repr(chat.items)
        assert "Items" in r or "0" in r

    def test_repr_with_items(self):
        """Items repr with content."""
        chat = Chat(system="System")
        r = repr(chat.items)
        assert "Items" in r or "1" in r


class TestItemsReadOnly:
    """Tests confirming items is read-only."""

    def test_items_has_no_setter(self):
        """chat.items = [...] raises AttributeError."""
        chat = Chat()
        with pytest.raises(AttributeError):
            chat.items = []

    def test_items_item_assignment_fails(self):
        """chat.items[0] = {...} raises TypeError."""
        chat = Chat(system="System")
        with pytest.raises(TypeError):
            item = chat.items[0]
            chat.items[0] = item


class TestItemsFromDict:
    """Tests for populating items via from_dict()."""

    def test_from_dict_populates_items(self):
        """from_dict() populates items."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        )
        assert len(chat.items) == 2
        item0 = chat.items[0]
        assert isinstance(item0, MessageItem)
        assert item0.role == MessageRole.USER
        assert item0.text == "Hello"
        item1 = chat.items[1]
        assert isinstance(item1, MessageItem)
        assert item1.role == MessageRole.ASSISTANT
        assert item1.text == "Hi!"

    def test_from_dict_with_system(self):
        """from_dict() with system message."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )
        assert len(chat.items) == 2
        assert chat.items.system == "Be helpful."

    def test_from_dict_empty(self):
        """from_dict() with empty items."""
        chat = Chat.from_dict({"messages": []})
        assert len(chat.items) == 0


class TestChatClear:
    """Tests for Chat.clear() method."""

    def test_clear_keeps_system(self):
        """clear() keeps system prompt."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        )
        assert len(chat.items) == 3

        chat.clear()
        assert len(chat.items) == 1
        item = chat.items[0]
        assert isinstance(item, MessageItem)
        assert item.role == MessageRole.SYSTEM

    def test_clear_no_system(self):
        """clear() on chat without system clears all."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        )
        chat.clear()
        assert len(chat.items) == 0

    def test_clear_returns_self(self):
        """clear() returns self for chaining."""
        chat = Chat()
        result = chat.clear()
        assert result is chat


class TestChatReset:
    """Tests for Chat.reset() method."""

    def test_reset_clears_all(self):
        """reset() clears everything including system."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )
        assert len(chat.items) == 2

        chat.reset()
        assert len(chat.items) == 0

    def test_reset_empty_chat(self):
        """reset() on empty chat is safe."""
        chat = Chat()
        chat.reset()
        assert len(chat.items) == 0

    def test_reset_returns_self(self):
        """reset() returns self for chaining."""
        chat = Chat()
        result = chat.reset()
        assert result is chat

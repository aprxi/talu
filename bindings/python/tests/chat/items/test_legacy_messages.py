"""Tests for the chat.messages property (OpenAI-format compatibility bridge).

The `messages` property provides a read-only view of conversation Items
in the industry-standard OpenAI format: [{"role": "...", "content": "..."}, ...]

This is a compatibility bridge for:
- Debugging ("What does the LLM see?")
- Interoperability with other APIs
- Familiarity for users coming from OpenAI/Anthropic/LangChain
"""

import pytest

from talu.types import (
    InputText,
    ItemStatus,
    MessageItem,
    MessageRole,
    OutputText,
)

# =============================================================================
# MessageItem.to_message_dict() tests
# =============================================================================


class TestMessageItemToDict:
    """Tests for MessageItem.to_message_dict() conversion."""

    def test_user_message(self) -> None:
        """User message converts to standard dict format."""
        item = MessageItem(
            id=1,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.USER,
            content=(InputText(text="Hello!"),),
        )
        result = item.to_message_dict()
        assert result == {"role": "user", "content": "Hello!"}

    def test_assistant_message(self) -> None:
        """Assistant message converts to standard dict format."""
        item = MessageItem(
            id=2,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.ASSISTANT,
            content=(OutputText(text="Hi there!"),),
        )
        result = item.to_message_dict()
        assert result == {"role": "assistant", "content": "Hi there!"}

    def test_system_message(self) -> None:
        """System message converts to standard dict format."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.SYSTEM,
            content=(InputText(text="You are helpful."),),
        )
        result = item.to_message_dict()
        assert result == {"role": "system", "content": "You are helpful."}

    def test_developer_message(self) -> None:
        """Developer message converts to standard dict format."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.DEVELOPER,
            content=(InputText(text="Debug info"),),
        )
        result = item.to_message_dict()
        assert result == {"role": "developer", "content": "Debug info"}

    def test_empty_content(self) -> None:
        """Message with no content returns empty string."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.USER,
            content=(),
        )
        result = item.to_message_dict()
        assert result == {"role": "user", "content": ""}

    def test_unknown_role_uses_raw_role(self) -> None:
        """Unknown role falls back to raw_role if available."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.UNKNOWN,
            content=(InputText(text="Test"),),
            raw_role="custom_role",
        )
        result = item.to_message_dict()
        assert result == {"role": "custom_role", "content": "Test"}

    def test_unknown_role_without_raw_role(self) -> None:
        """Unknown role without raw_role uses 'unknown'."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.UNKNOWN,
            content=(InputText(text="Test"),),
        )
        result = item.to_message_dict()
        assert result == {"role": "unknown", "content": "Test"}


# =============================================================================
# Chat.messages property tests (mocked)
# =============================================================================


class TestChatMessagesProperty:
    """Tests for Chat.messages property using mocked items."""

    def test_messages_returns_list_of_dicts(self) -> None:
        """messages property returns list of standard dicts."""
        from talu import Client

        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat()
        try:
            # Fresh chat has no messages
            messages = chat.messages
            assert isinstance(messages, list)
            # May have system message depending on defaults
            for msg in messages:
                assert isinstance(msg, dict)
                assert "role" in msg
                assert "content" in msg
        finally:
            chat.close()

    def test_messages_with_system_prompt(self) -> None:
        """messages includes system prompt."""
        from talu import Client

        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="You are a pirate.")
        try:
            messages = chat.messages
            # System message should be first
            assert len(messages) >= 1
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a pirate."
        finally:
            chat.close()

    def test_messages_excludes_non_message_items(self) -> None:
        """messages only includes MessageItems, not FunctionCallItems etc."""
        # This is tested implicitly - the property filters by isinstance(item, MessageItem)
        # Non-message items (FunctionCallItem, ReasoningItem) are excluded
        from talu import Client

        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat()
        try:
            # All returned items should be message dicts
            for msg in chat.messages:
                assert "role" in msg
                assert msg["role"] in ("system", "user", "assistant", "developer", "unknown")
        finally:
            chat.close()

    def test_messages_is_read_only(self) -> None:
        """messages property returns a new list each time (read-only semantics)."""
        from talu import Client

        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="Test")
        try:
            messages1 = chat.messages
            messages2 = chat.messages
            # Should be equal but not the same object
            assert messages1 == messages2
            assert messages1 is not messages2
            # Attempting to mutate raises TypeError (prevents silent bugs)
            with pytest.raises(TypeError, match="read-only"):
                messages1.append({"role": "user", "content": "Injected"})
        finally:
            chat.close()


# =============================================================================
# AsyncChat.messages property tests
# =============================================================================


class TestAsyncChatMessagesProperty:
    """Tests for AsyncChat.messages property."""

    def test_async_chat_has_messages_property(self) -> None:
        """AsyncChat has messages property with same behavior as Chat."""
        from talu import AsyncClient

        client = AsyncClient("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="You are helpful.")
        try:
            messages = chat.messages
            assert isinstance(messages, list)
            assert len(messages) >= 1
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are helpful."
        finally:
            chat._close_sync()


# =============================================================================
# Integration: messages for debugging
# =============================================================================


class TestMessagesDebugging:
    """Tests demonstrating messages property for debugging workflows."""

    def test_messages_is_json_serializable(self) -> None:
        """messages output can be directly serialized to JSON."""
        import json

        from talu import Client

        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="Test")
        try:
            messages = chat.messages
            # Should not raise
            json_str = json.dumps(messages)
            assert isinstance(json_str, str)
            # Round-trip
            parsed = json.loads(json_str)
            assert parsed == messages
        finally:
            chat.close()

    def test_messages_readable_repr(self) -> None:
        """messages produces human-readable output for debugging."""
        from talu import Client

        client = Client("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="You are helpful.")
        try:
            messages = chat.messages
            # Should be printable and readable
            repr_str = repr(messages)
            assert "system" in repr_str
            assert "You are helpful" in repr_str
        finally:
            chat.close()


# =============================================================================
# MessageList tests
# =============================================================================


class TestMessageList:
    """Tests for MessageList pretty-printing in REPL/notebook."""

    def test_message_list_is_list_subclass(self) -> None:
        """MessageList inherits from list."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        assert isinstance(msgs, list)
        assert isinstance(msgs, MessageList)

    def test_message_list_supports_read_operations(self) -> None:
        """MessageList supports read-only list operations."""
        from talu.chat import MessageList

        msgs = MessageList(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
        )

        # Indexing
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1] == {"role": "user", "content": "Hello!"}

        # Length
        assert len(msgs) == 2

        # Iteration
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user"]

        # Slicing
        assert msgs[1:] == [{"role": "user", "content": "Hello!"}]

        # Containment
        assert {"role": "user", "content": "Hello!"} in msgs

        # Copy to regular list
        regular_list = list(msgs)
        assert regular_list == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

    def test_message_list_pretty_repr_basic(self) -> None:
        """MessageList repr shows role: content format."""
        from talu.chat import MessageList

        msgs = MessageList(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )

        r = repr(msgs)
        assert "system: You are helpful." in r
        assert "user: Hello!" in r
        assert "assistant: Hi there!" in r

    def test_message_list_empty(self) -> None:
        """Empty MessageList repr is '[]'."""
        from talu.chat import MessageList

        msgs = MessageList([])
        assert repr(msgs) == "[]"

    def test_message_list_truncates_long_content(self) -> None:
        """MessageList truncates content longer than 80 chars."""
        from talu.chat import MessageList

        long_content = "A" * 100
        msgs = MessageList([{"role": "user", "content": long_content}])

        r = repr(msgs)
        # Should be truncated with ...
        assert "..." in r
        # Should not contain full 100 A's
        assert "A" * 100 not in r
        # Should contain truncated version (77 chars + ...)
        assert "A" * 77 in r

    def test_message_list_escapes_newlines(self) -> None:
        """MessageList shows newlines as \\n."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "assistant", "content": "Line 1\nLine 2\nLine 3"}])

        r = repr(msgs)
        assert "\\n" in r
        # Actual newlines in content should be escaped
        assert "Line 1\\nLine 2\\nLine 3" in r

    def test_message_list_str_same_as_repr(self) -> None:
        """MessageList str() returns same as repr()."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Test"}])
        assert str(msgs) == repr(msgs)

    def test_message_list_json_serializable(self) -> None:
        """MessageList can be serialized to JSON."""
        import json

        from talu.chat import MessageList

        msgs = MessageList(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
        )

        # Should serialize without error
        json_str = json.dumps(msgs)
        parsed = json.loads(json_str)

        # Round-trip should preserve data
        assert parsed == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

    def test_chat_messages_returns_message_list(self) -> None:
        """Chat.messages returns MessageList type."""
        from talu import Chat
        from talu.chat import MessageList

        chat = Chat(system="Test")
        try:
            messages = chat.messages
            assert isinstance(messages, MessageList)
            assert isinstance(messages, list)
        finally:
            chat.close()


class TestMessageListImmutability:
    """Tests that MessageList is immutable to prevent silent bugs."""

    def test_append_raises_typeerror(self) -> None:
        """MessageList.append() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.append({"role": "assistant", "content": "Hi"})

    def test_extend_raises_typeerror(self) -> None:
        """MessageList.extend() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.extend([{"role": "assistant", "content": "Hi"}])

    def test_insert_raises_typeerror(self) -> None:
        """MessageList.insert() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.insert(0, {"role": "system", "content": "Be helpful"})

    def test_remove_raises_typeerror(self) -> None:
        """MessageList.remove() raises TypeError."""
        from talu.chat import MessageList

        msg = {"role": "user", "content": "Hello"}
        msgs = MessageList([msg])
        with pytest.raises(TypeError, match="read-only"):
            msgs.remove(msg)

    def test_pop_raises_typeerror(self) -> None:
        """MessageList.pop() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.pop()

    def test_clear_raises_typeerror(self) -> None:
        """MessageList.clear() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.clear()

    def test_setitem_raises_typeerror(self) -> None:
        """MessageList[i] = x raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs[0] = {"role": "user", "content": "Bye"}

    def test_delitem_raises_typeerror(self) -> None:
        """del MessageList[i] raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            del msgs[0]

    def test_iadd_raises_typeerror(self) -> None:
        """MessageList += [...] raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs += [{"role": "assistant", "content": "Hi"}]

    def test_sort_raises_typeerror(self) -> None:
        """MessageList.sort() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.sort()

    def test_reverse_raises_typeerror(self) -> None:
        """MessageList.reverse() raises TypeError."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError, match="read-only"):
            msgs.reverse()

    def test_error_message_suggests_chat_methods(self) -> None:
        """Error message suggests using Chat methods instead."""
        from talu.chat import MessageList

        msgs = MessageList([{"role": "user", "content": "Hello"}])
        with pytest.raises(TypeError) as exc_info:
            msgs.append({"role": "assistant", "content": "Hi"})

        error_msg = str(exc_info.value)
        assert "add_user_message" in error_msg
        assert "add_assistant_message" in error_msg
        assert "pop" in error_msg
        assert "clear" in error_msg


# =============================================================================
# MessageItem.create() factory method tests
# =============================================================================


class TestMessageItemCreate:
    """Tests for MessageItem.create() factory method."""

    def test_create_with_string_role(self) -> None:
        """MessageItem.create() accepts string role."""
        item = MessageItem.create("user", "Hello!")
        assert item.role == MessageRole.USER
        assert item.text == "Hello!"

    def test_create_with_enum_role(self) -> None:
        """MessageItem.create() accepts MessageRole enum."""
        item = MessageItem.create(MessageRole.ASSISTANT, "Hi there!")
        assert item.role == MessageRole.ASSISTANT
        assert item.text == "Hi there!"

    def test_create_all_string_roles(self) -> None:
        """MessageItem.create() accepts all valid string roles."""
        roles = ["system", "user", "assistant", "developer"]
        expected = [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.DEVELOPER,
        ]
        for role_str, role_enum in zip(roles, expected, strict=True):
            item = MessageItem.create(role_str, "Test")
            assert item.role == role_enum

    def test_create_case_insensitive_role(self) -> None:
        """MessageItem.create() role is case-insensitive."""
        item = MessageItem.create("USER", "Hello!")
        assert item.role == MessageRole.USER

        item = MessageItem.create("Assistant", "Hi!")
        assert item.role == MessageRole.ASSISTANT

    def test_create_with_string_content(self) -> None:
        """MessageItem.create() with string wraps in InputText."""
        item = MessageItem.create("user", "Hello!")
        assert len(item.content) == 1
        assert isinstance(item.content[0], InputText)
        assert item.content[0].text == "Hello!"

    def test_create_with_tuple_content(self) -> None:
        """MessageItem.create() with tuple preserves content parts."""
        content = (InputText(text="Part 1"), OutputText(text="Part 2"))
        item = MessageItem.create(MessageRole.ASSISTANT, content)
        assert item.content == content

    def test_create_auto_generates_metadata(self) -> None:
        """MessageItem.create() auto-generates id, status, and timestamp."""
        item = MessageItem.create("user", "Hello!")
        assert item.id == 0  # Placeholder, assigned by backend
        assert item.status == ItemStatus.COMPLETED
        assert item.created_at_ms > 0

    def test_create_invalid_role_raises(self) -> None:
        """MessageItem.create() raises ValueError for invalid role."""
        with pytest.raises(ValueError, match="Invalid role"):
            MessageItem.create("invalid_role", "Hello!")


# =============================================================================
# Chat.append(MessageItem) overload tests
# =============================================================================


class TestChatAppendMessageItem:
    """Tests for Chat.append() with MessageItem overload."""

    def test_append_string_args_still_works(self) -> None:
        """Chat.append(role, content) still works."""
        from talu import Chat

        chat = Chat(system="Test")
        try:
            chat.append("user", "Hello!")
            chat.append("assistant", "Hi there!")
            assert len(chat.items) == 3  # system + user + assistant
            assert chat.items[-2].text == "Hello!"
            assert chat.items[-1].text == "Hi there!"
        finally:
            chat.close()

    def test_append_message_item(self) -> None:
        """Chat.append(MessageItem) works."""
        from talu import Chat

        chat = Chat(system="Test")
        try:
            item = MessageItem.create("user", "From MessageItem!")
            chat.append(item)
            assert len(chat.items) == 2  # system + user
            assert chat.items[-1].text == "From MessageItem!"
            assert chat.items[-1].role == MessageRole.USER
        finally:
            chat.close()

    def test_append_message_item_with_enum_role(self) -> None:
        """Chat.append(MessageItem) works with enum role."""
        from talu import Chat

        chat = Chat(system="Test")
        try:
            item = MessageItem.create(MessageRole.ASSISTANT, "Assistant response")
            chat.append(item)
            assert chat.items[-1].role == MessageRole.ASSISTANT
            assert chat.items[-1].text == "Assistant response"
        finally:
            chat.close()

    def test_append_multiple_message_items(self) -> None:
        """Chat.append() can append multiple MessageItems."""
        from talu import Chat

        chat = Chat(system="Test")
        try:
            chat.append(MessageItem.create("user", "Question 1"))
            chat.append(MessageItem.create("assistant", "Answer 1"))
            chat.append(MessageItem.create("user", "Question 2"))
            chat.append(MessageItem.create("assistant", "Answer 2"))
            assert len(chat.items) == 5  # system + 4 messages
        finally:
            chat.close()

    def test_append_mixed_styles(self) -> None:
        """Chat.append() can mix string and MessageItem styles."""
        from talu import Chat

        chat = Chat(system="Test")
        try:
            chat.append("user", "String style")
            chat.append(MessageItem.create("assistant", "Object style"))
            chat.append("user", "String again")
            assert len(chat.items) == 4
            assert chat.items[1].text == "String style"
            assert chat.items[2].text == "Object style"
            assert chat.items[3].text == "String again"
        finally:
            chat.close()

    def test_append_returns_self_for_chaining(self) -> None:
        """Chat.append() returns self for method chaining."""
        from talu import Chat

        chat = Chat(system="Test")
        try:
            result = chat.append(MessageItem.create("user", "Hello!"))
            assert result is chat
        finally:
            chat.close()

    def test_append_missing_content_raises(self) -> None:
        """Chat.append(role) without content raises ValidationError."""
        from talu import Chat
        from talu.exceptions import ValidationError

        chat = Chat(system="Test")
        try:
            with pytest.raises(ValidationError, match="content is required"):
                chat.append("user")  # type: ignore[call-overload]
        finally:
            chat.close()


# =============================================================================
# AsyncChat.append(MessageItem) overload tests
# =============================================================================


class TestAsyncChatAppendMessageItem:
    """Tests for AsyncChat.append() with MessageItem overload."""

    def test_async_append_message_item(self) -> None:
        """AsyncChat.append(MessageItem) works."""
        from talu import AsyncClient

        client = AsyncClient("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="Test")
        try:
            item = MessageItem.create("user", "From MessageItem!")
            chat.append(item)
            assert len(chat.items) == 2  # system + user
            assert chat.items[-1].text == "From MessageItem!"
        finally:
            chat._close_sync()

    def test_async_append_string_args_still_works(self) -> None:
        """AsyncChat.append(role, content) still works."""
        from talu import AsyncClient

        client = AsyncClient("openai://gpt-4", base_url="http://test")
        chat = client.chat(system="Test")
        try:
            chat.append("user", "String style!")
            assert chat.items[-1].text == "String style!"
        finally:
            chat._close_sync()

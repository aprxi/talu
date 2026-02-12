"""
Tests for Chat.regenerate() feature.

The regenerate feature allows users to retry or edit the last user message
and generate a new response.

Tests are split into:
- Unit tests: Test truncation logic without requiring a model
- Integration tests: Test full regenerate with actual generation (requires_model)
"""

import pytest

from talu import AsyncChat, Chat
from talu.exceptions import StateError  # noqa: F401 - used in pytest.raises
from talu.types import MessageItem, MessageRole

# =============================================================================
# Unit Tests - Truncation Logic (No Model Required)
# =============================================================================


class TestRegenerateTruncationLogic:
    """Test the truncation logic of regenerate without requiring generation."""

    def test_find_last_user_message(self):
        """Regenerate finds the last user message correctly."""
        chat = Chat(system="System")
        chat.append("user", "First question")
        chat.append("assistant", "First answer")
        chat.append("user", "Second question")
        chat.append("assistant", "Second answer")

        # 5 items: system + user + assistant + user + assistant
        assert len(chat.items) == 5

        # Find last user index (should be index 3)
        last_user_idx = -1
        for i in range(len(chat.items) - 1, -1, -1):
            item = chat.items[i]
            if isinstance(item, MessageItem) and item.role == MessageRole.USER:
                last_user_idx = i
                break

        assert last_user_idx == 3
        assert chat.items[last_user_idx].text == "Second question"

    def test_no_user_message_detected(self):
        """StateError when no user message exists."""
        chat = Chat(system="System only")

        with pytest.raises(StateError) as exc_info:
            chat.regenerate()

        assert "No user message found" in str(exc_info.value)
        assert exc_info.value.code == "STATE_REGENERATE_NO_USER"

    def test_no_user_in_empty_chat(self):
        """StateError when chat is completely empty."""
        chat = Chat()

        with pytest.raises(StateError) as exc_info:
            chat.regenerate()

        assert exc_info.value.code == "STATE_REGENERATE_NO_USER"


class TestRegenerateWithAppend:
    """Test regenerate behavior by examining state after truncation.

    These tests append messages manually and check that truncation
    works correctly, without needing actual generation.
    """

    def test_truncate_removes_assistant_response(self):
        """After truncation, assistant response should be removed."""
        chat = Chat(system="System")
        chat.append("user", "Question")
        chat.append("assistant", "Answer to remove")

        # Verify starting state
        assert len(chat.items) == 3
        assert chat.items[2].text == "Answer to remove"

        # Simulate what regenerate does: find user, truncate before it
        # user is at index 1, trunc_target would be 0
        # Use clear_keeping_system since trunc_target would be 0
        from talu.chat._bindings import get_chat_lib

        lib = get_chat_lib()
        lib.talu_responses_clear_keeping_system(chat._conversation_ptr)

        # After truncation: only system remains
        assert len(chat.items) == 1
        assert chat.items[0].role == MessageRole.SYSTEM

    def test_truncate_preserves_earlier_turns(self):
        """Truncation preserves earlier conversation turns."""
        chat = Chat(system="System")
        chat.append("user", "Turn 1 user")
        chat.append("assistant", "Turn 1 assistant")
        chat.append("user", "Turn 2 user")
        chat.append("assistant", "Turn 2 assistant")

        # 5 items total
        assert len(chat.items) == 5

        # Truncate to keep up to Turn 1 (index 2)
        from talu.chat._bindings import get_chat_lib

        lib = get_chat_lib()
        lib.talu_responses_truncate_after(chat._conversation_ptr, 2)

        # Should have: system + user + assistant
        assert len(chat.items) == 3
        assert chat.items[1].text == "Turn 1 user"
        assert chat.items[2].text == "Turn 1 assistant"

    def test_user_at_index_zero_uses_clear(self):
        """When user is at index 0, clear is used instead of truncate."""
        chat = Chat()  # No system
        chat.append("user", "Hello")
        chat.append("assistant", "Response")

        # User at index 0, trunc_target = -1
        # This should use clear

        from talu.chat._bindings import get_chat_lib

        lib = get_chat_lib()
        # clear_keeping_system on chat with no system = clear all
        lib.talu_responses_clear_keeping_system(chat._conversation_ptr)

        # Chat should be empty
        assert len(chat.items) == 0


# =============================================================================
# Integration Tests - Full Regenerate (Requires Model)
# =============================================================================


@pytest.mark.requires_model
class TestRegenerateIntegration:
    """Integration tests for regenerate with actual generation."""

    def test_regenerate_retry_generates_new_response(self, talu, test_model_path):
        """Retry mode generates a new response with same user text."""
        chat = Chat(test_model_path)
        chat.send("Hello")  # Use send() which adds items immediately

        initial_response_id = chat.items[-1].id

        # Regenerate (default stream=False returns Response)
        chat.regenerate()

        # Should still have same structure
        assert len(chat.items) >= 2  # At least user + assistant

        # Response should have different ID (fresh item)
        assert chat.items[-1].id != initial_response_id

    def test_regenerate_edit_uses_new_message(self, talu, test_model_path):
        """Edit mode uses the new message text."""
        chat = Chat(test_model_path)
        chat.send("Original message")  # Use send() which adds items immediately

        # Regenerate with new message
        chat.regenerate(message="Different message")

        # User message should be the new one
        user_items = [
            i for i in chat.items if isinstance(i, MessageItem) and i.role == MessageRole.USER
        ]
        assert len(user_items) == 1
        assert user_items[0].text == "Different message"

    def test_regenerate_with_temperature(self, talu, test_model_path):
        """Regenerate accepts temperature kwarg."""
        chat = Chat(test_model_path)
        chat.send("Hello")  # Use send() which adds items immediately

        # Should not raise
        chat.regenerate(temperature=0.1)

        assert len(chat.items) >= 2

    def test_regenerate_returns_response(self, talu, test_model_path):
        """Regenerate returns a Response object."""
        from talu.chat.response import Response

        chat = Chat(test_model_path)
        chat.send("Hello")  # Use send() which adds items immediately

        result = chat.regenerate()

        assert isinstance(result, Response)
        assert result.text is not None

    def test_regenerate_streaming(self, talu, test_model_path):
        """Regenerate streaming returns StreamingResponse."""
        from talu.chat.response import StreamingResponse

        chat = Chat(test_model_path)
        chat.send("Hello")  # Use send() which adds items immediately

        result = chat.regenerate(stream=True)

        assert isinstance(result, StreamingResponse)

        # Consume the stream
        text = "".join(result)
        assert isinstance(text, str)


@pytest.mark.requires_model
@pytest.mark.asyncio
class TestAsyncRegenerateIntegration:
    """Integration tests for AsyncChat.regenerate()."""

    async def test_async_regenerate_basic(self, talu, test_model_path):
        """Basic async regenerate generates new response."""
        chat = AsyncChat(test_model_path)
        await chat.send("Hello")  # Use send() which adds items immediately

        initial_id = chat.items[-1].id

        await chat.regenerate()

        # New response should have different ID
        assert chat.items[-1].id != initial_id

    async def test_async_regenerate_with_message(self, talu, test_model_path):
        """Async regenerate edit mode."""
        chat = AsyncChat(test_model_path)
        await chat.send("Original")  # Use send() which adds items immediately

        await chat.regenerate(message="Edited")

        user_items = [
            i for i in chat.items if isinstance(i, MessageItem) and i.role == MessageRole.USER
        ]
        assert user_items[-1].text == "Edited"

    async def test_async_regenerate_no_user_raises(self):
        """Async regenerate with no user raises StateError."""
        chat = AsyncChat(system="System only")

        with pytest.raises(StateError) as exc_info:
            await chat.regenerate()

        assert exc_info.value.code == "STATE_REGENERATE_NO_USER"

"""Tests for Chat.count_tokens() and Chat.max_context_length.

These tests verify token counting for context window management.
"""

import pytest

from talu import Chat
from talu.exceptions import StateError


class TestCountTokensWithoutModel:
    """Tests for count_tokens when no model is configured."""

    def test_count_tokens_raises_without_model(self):
        """count_tokens raises StateError when no model is configured."""
        chat = Chat()  # No model
        with pytest.raises(StateError, match="no model configured"):
            chat.count_tokens()

    def test_count_tokens_with_message_raises_without_model(self):
        """count_tokens with message raises StateError when no model is configured."""
        chat = Chat()
        with pytest.raises(StateError, match="no model configured"):
            chat.count_tokens("test message")


class TestMaxContextLengthWithoutModel:
    """Tests for max_context_length when no model is configured."""

    def test_max_context_length_returns_none_without_model(self):
        """max_context_length returns None when no model is configured."""
        chat = Chat()
        assert chat.max_context_length is None


@pytest.mark.requires_model
class TestCountTokensWithModel:
    """Tests for count_tokens requiring a real model."""

    def test_count_tokens_empty_chat(self, test_model_path):
        """count_tokens works with empty chat history."""
        chat = Chat(test_model_path)
        count = chat.count_tokens()
        # Should be > 0 due to chat template overhead (system tokens, etc.)
        assert count >= 0
        assert isinstance(count, int)

    def test_count_tokens_with_system(self, test_model_path):
        """count_tokens includes system prompt tokens."""
        chat = Chat(test_model_path, system="You are a helpful assistant.")
        count_with_system = chat.count_tokens()

        chat_no_system = Chat(test_model_path)
        count_no_system = chat_no_system.count_tokens()

        # System prompt should add tokens
        assert count_with_system > count_no_system

    def test_count_tokens_with_message(self, test_model_path):
        """count_tokens with message includes message tokens."""
        chat = Chat(test_model_path)
        count_empty = chat.count_tokens()

        # Count with hypothetical message
        count_with_msg = chat.count_tokens("Hello, this is a test message!")

        # Message should add tokens
        assert count_with_msg > count_empty

    def test_count_tokens_message_not_added_to_history(self, test_model_path):
        """count_tokens with message does not modify conversation history."""
        chat = Chat(test_model_path)
        initial_len = len(chat.items)

        # Count tokens with message
        chat.count_tokens("This message should not be added")

        # History should be unchanged
        assert len(chat.items) == initial_len

    def test_count_tokens_increases_with_conversation(self, test_model_path):
        """count_tokens increases as conversation grows."""
        chat = Chat(test_model_path)
        count_empty = chat.count_tokens()

        # Add user message manually using items API
        chat._lib.talu_responses_append_message(
            chat._conversation_ptr,
            0,  # user role
            b"Hello!",
            6,
        )
        count_with_one = chat.count_tokens()

        # Add another message
        chat._lib.talu_responses_append_message(
            chat._conversation_ptr,
            1,  # assistant role
            b"Hi there!",
            9,
        )
        count_with_two = chat.count_tokens()

        # Token count should increase
        assert count_with_one > count_empty
        assert count_with_two > count_with_one


@pytest.mark.requires_model
class TestMaxContextLengthWithModel:
    """Tests for max_context_length requiring a real model."""

    def test_max_context_length_returns_int(self, test_model_path):
        """max_context_length returns an integer."""
        chat = Chat(test_model_path)
        max_len = chat.max_context_length
        # Should be either an int > 0 or None
        assert max_len is None or (isinstance(max_len, int) and max_len > 0)

    def test_max_context_length_typical_values(self, test_model_path):
        """max_context_length returns reasonable values for typical models."""
        chat = Chat(test_model_path)
        max_len = chat.max_context_length
        if max_len is not None:
            # Typical context lengths are between 512 and 2M tokens
            assert 512 <= max_len <= 2_000_000


@pytest.mark.requires_model
class TestContextWindowWorkflow:
    """Tests for typical context window management workflow."""

    def test_will_it_fit_workflow(self, test_model_path):
        """Test the 'will it fit?' workflow for context window management."""
        chat = Chat(test_model_path, system="You are helpful.")

        # Get context limit
        max_len = chat.max_context_length
        if max_len is None:
            pytest.skip("Model does not specify max context length")

        # Check current usage
        current_tokens = chat.count_tokens()
        assert current_tokens < max_len

        # Check if a message would fit
        test_message = "Hello, world!"
        tokens_with_message = chat.count_tokens(test_message)
        assert tokens_with_message < max_len

        # The difference should be reasonable for the message size
        token_increase = tokens_with_message - current_tokens
        assert token_increase > 0
        assert token_increase < 100  # "Hello, world!" shouldn't be 100+ tokens

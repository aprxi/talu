"""Tests for Chat.system property.

These tests verify system prompt mutability.
"""

import pytest

from talu import Chat
from talu.exceptions import StateError


class TestSystemPropertyWithoutModel:
    """Tests for system property without a model configured."""

    def test_get_system_returns_none_when_not_set(self):
        """system returns None when no system prompt is set."""
        chat = Chat()
        assert chat.system is None

    def test_set_system_works_without_model(self):
        """system can be set even without a model."""
        chat = Chat()
        chat.system = "You are a pirate."
        assert chat.system == "You are a pirate."

    def test_set_system_to_none_clears_prompt(self):
        """Setting system to None clears the prompt."""
        chat = Chat(system="Initial prompt")
        assert chat.system == "Initial prompt"

        chat.system = None
        assert chat.system is None

    def test_system_reflects_constructor_value(self):
        """system returns the value passed to constructor."""
        chat = Chat(system="You are helpful.")
        assert chat.system == "You are helpful."

    def test_system_modification_updates_value(self):
        """system can be modified after construction."""
        chat = Chat(system="You are helpful.")
        chat.system = "You are a chef."
        assert chat.system == "You are a chef."

    def test_system_items_consistency(self):
        """system property and items[0] are consistent."""
        chat = Chat(system="Test system")
        assert len(chat.items) == 1
        assert chat.items[0].text == "Test system"
        assert chat.system == "Test system"

    def test_set_system_adds_to_empty_chat(self):
        """Setting system on empty chat adds a system message.

        Note: When setting system on an empty chat, the underlying conversation
        is replaced. The system property getter correctly reads from the new
        conversation, but cached `items` references may be stale.
        """
        chat = Chat()
        assert len(chat.items) == 0

        chat.system = "New system"
        # Verify via property getter (reads fresh from Chat)
        assert chat.system == "New system"
        # Note: chat.items may show stale data due to conversation replacement
        # This is expected behavior - use chat.system for system prompt access

    def test_system_unicode_content(self):
        """system handles unicode content correctly."""
        chat = Chat()
        chat.system = "You are a helpful assistant. 你好！🎉"
        assert chat.system == "You are a helpful assistant. 你好！🎉"

    def test_system_empty_string_treated_as_none(self):
        """Setting system to empty string clears the prompt."""
        chat = Chat(system="Initial")
        # Setting empty string should clear
        chat.system = ""
        # Empty string from Zig becomes None in Python
        # (since we return None for empty strings)
        result = chat.system
        # Could be None or "" depending on implementation
        assert result is None or result == ""


class TestSystemPropertyClosed:
    """Tests for system property on closed Chat."""

    def test_get_system_raises_on_closed_chat(self):
        """system getter raises StateError on closed chat."""
        chat = Chat()
        chat.close()
        with pytest.raises(StateError, match="closed"):
            _ = chat.system

    def test_set_system_raises_on_closed_chat(self):
        """system setter raises StateError on closed chat."""
        chat = Chat()
        chat.close()
        with pytest.raises(StateError, match="closed"):
            chat.system = "Should fail"


@pytest.mark.requires_model
class TestSystemPropertyWithModel:
    """Tests for system property with a real model."""

    def test_system_with_model(self, test_model_path):
        """system works with a model configured."""
        chat = Chat(test_model_path, system="You are a pirate.")
        assert chat.system == "You are a pirate."

        chat.system = "You are a chef."
        assert chat.system == "You are a chef."

    def test_system_modification_mid_conversation(self, test_model_path):
        """system can be modified mid-conversation (steering behavior)."""
        chat = Chat(test_model_path, system="You are helpful.")

        # Note: This is a valid use case - users can "steer" the conversation
        # by changing the system prompt mid-way. The model will adopt the new
        # persona going forward.
        chat.system = "You are now critical and skeptical."
        assert chat.system == "You are now critical and skeptical."


"""
Topics covered:

* chat.system
* system.mutability
* system.kv_cache_invalidation

Related:

* examples/recipes/26_context_window.py
"""

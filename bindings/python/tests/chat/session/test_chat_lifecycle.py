"""
Unit tests for Chat class internals.

These tests focus on Chat construction, properties, and state management
WITHOUT requiring a model. For integration tests with actual generation,
see test_api.py.

Note: messages is READ-ONLY from Python. To test with messages, use
Chat.from_dict() to create chats with pre-populated history.
"""

import json

import pytest

import talu
from talu import AsyncChat, Chat, GenerationConfig

# =============================================================================
# Construction Tests
# =============================================================================


class TestChatConstruction:
    """Tests for Chat construction without a model."""

    def test_empty_construction(self):
        """Chat can be created with no arguments."""
        chat = Chat()
        assert len(chat.items) == 0
        assert len(chat) == 0

    def test_construction_with_system(self):
        """Chat can be created with system message."""
        chat = Chat(system="You are helpful.")
        assert len(chat.items) == 1
        assert chat.items[0].role.name.lower() == "system"
        assert chat.items[0].text == "You are helpful."

    def test_construction_with_config(self):
        """Chat accepts GenerationConfig at creation."""
        config = GenerationConfig(temperature=0.3, max_tokens=50)
        chat = Chat(config=config)
        assert chat.config.temperature == 0.3
        assert chat.config.max_tokens == 50

    def test_construction_with_full_config(self):
        """Chat accepts all config parameters."""
        config = GenerationConfig(
            max_tokens=100,
            temperature=0.5,
            top_k=40,
            top_p=0.95,
        )
        chat = Chat(system="Be brief.", config=config)
        assert chat.config.max_tokens == 100
        assert abs(chat.config.temperature - 0.5) < 1e-6
        assert chat.config.top_k == 40
        assert abs(chat.config.top_p - 0.95) < 1e-6

    def test_construction_with_offline(self):
        """Chat accepts offline resolution flag."""
        chat = Chat(offline=True)
        assert len(chat.items) == 0

    def test_async_construction_with_offline(self):
        """AsyncChat accepts offline resolution flag."""
        chat = AsyncChat(offline=True)
        assert len(chat.items) == 0


# =============================================================================
# Message Access Tests (Read-Only)
# =============================================================================


class TestChatMessages:
    """Tests for Chat message access (read-only)."""

    def test_messages_starts_empty(self):
        """New chat has empty messages."""
        chat = Chat()
        assert list(chat.items) == []

    def test_messages_with_system(self):
        """Chat with system has one message."""
        chat = Chat(system="Hello")
        assert len(chat.items) == 1
        assert chat.items[0].role.name.lower() == "system"
        assert chat.items[0].text == "Hello"

    def test_messages_from_dict(self):
        """Chat.from_dict() populates messages."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        chat = Chat.from_dict(data)
        assert len(chat.items) == 2
        assert chat.items[0].role.name.lower() == "user"
        assert chat.items[1].role.name.lower() == "assistant"

    def test_messages_is_read_only(self):
        """messages property is read-only (no setter)."""
        chat = Chat()
        with pytest.raises(AttributeError):
            chat.items = [{"role": "user", "content": "Hello"}]

    def test_clear_keeps_system(self):
        """clear() removes messages but keeps system."""
        data = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        chat = Chat.from_dict(data)
        chat.clear()
        assert len(chat.items) == 1
        assert chat.items[0].role.name.lower() == "system"

    def test_reset_removes_everything(self):
        """reset() removes everything including system."""
        data = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello!"},
            ]
        }
        chat = Chat.from_dict(data)
        chat.reset()
        assert len(chat) == 0

    def test_pop_removes_last_message(self):
        """pop() removes the last message."""
        data = {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        chat = Chat.from_dict(data)
        assert len(chat.items) == 3

        chat.pop()
        assert len(chat.items) == 2
        assert chat.items[-1].role.name.lower() == "user"

    def test_pop_returns_self(self):
        """pop() returns self for chaining."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        chat = Chat.from_dict(data)
        result = chat.pop()
        assert result is chat

    def test_pop_empty_raises(self):
        """pop() on empty chat raises StateError."""
        chat = Chat()
        with pytest.raises(talu.StateError):
            chat.pop()

    def test_remove_at_index(self):
        """remove() removes message at specific index."""
        data = {
            "messages": [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second"},
            ]
        }
        chat = Chat.from_dict(data)
        assert len(chat.items) == 4

        # Remove "First" (index 1)
        chat.remove(1)
        assert len(chat.items) == 3
        assert chat.items[0].text == "System"
        assert chat.items[1].text == "Response"
        assert chat.items[2].text == "Second"

    def test_remove_returns_self(self):
        """remove() returns self for chaining."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
            ]
        }
        chat = Chat.from_dict(data)
        result = chat.remove(0)
        assert result is chat

    def test_remove_out_of_bounds_raises(self):
        """remove() with invalid index raises StateError."""
        chat = Chat(system="Test")
        with pytest.raises(talu.StateError):
            chat.remove(5)


# =============================================================================
# Properties Tests
# =============================================================================


class TestChatProperties:
    """Tests for Chat property accessors."""

    def test_len_counts_all_messages(self):
        """len() returns total message count including system."""
        chat = Chat(system="System")
        assert len(chat) == 1

        # Use from_dict to add more messages
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )
        assert len(chat) == 2

    def test_config_property(self):
        """config property returns GenerationConfig."""
        config = GenerationConfig(temperature=0.5)
        chat = Chat(config=config)
        assert chat.config.temperature == 0.5

    def test_config_has_defaults(self):
        """Chat has default config values."""
        chat = Chat()
        assert chat.config is not None
        assert chat.config.max_tokens > 0
        assert chat.config.temperature >= 0

    def test_config_can_be_replaced(self):
        """config can be replaced."""
        chat = Chat()
        new_config = GenerationConfig(temperature=0.9, max_tokens=100)
        chat.config = new_config
        assert chat.config.temperature == 0.9
        assert chat.config.max_tokens == 100

    def test_repr(self):
        """Chat has informative repr."""
        chat = Chat(system="Test")
        repr_str = repr(chat)
        assert "Chat" in repr_str


# =============================================================================
# Serialization Tests
# =============================================================================


class TestChatSerialization:
    """Tests for Chat serialization without model."""

    def test_to_dict_basic(self):
        """to_dict() serializes chat state."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        data = chat.to_dict()
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_to_dict_includes_config(self):
        """to_dict() includes config."""
        config = GenerationConfig(temperature=0.5, max_tokens=100)
        chat = Chat(config=config)

        data = chat.to_dict()
        assert "config" in data
        assert data["config"]["temperature"] == 0.5
        assert data["config"]["max_tokens"] == 100

    def test_to_json(self):
        """to_json() returns valid JSON string."""
        chat = Chat(system="Test")
        json_str = chat.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_from_dict_basic(self):
        """from_dict() creates chat from dict."""
        data = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
            ]
        }

        chat = Chat.from_dict(data)
        assert len(chat.items) == 2
        assert chat.items[0].text == "Be helpful."

    def test_from_dict_with_config(self):
        """from_dict() restores config."""
        data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "config": {"temperature": 0.5, "max_tokens": 100},
        }

        chat = Chat.from_dict(data)
        assert chat.config.temperature == 0.5
        assert chat.config.max_tokens == 100

    def test_roundtrip(self):
        """to_dict -> from_dict preserves state."""
        original = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        )

        data = original.to_dict()
        restored = Chat.from_dict(data)

        assert len(restored.items) == len(original.items)
        for i in range(len(original.items)):
            assert restored.items[i].role.name.lower() == original.items[i].role.name.lower()
            assert restored.items[i].text == original.items[i].text


# =============================================================================
# Fork Tests (without engine)
# =============================================================================


class TestChatFork:
    """Tests for Chat fork without engine."""

    def test_fork_copies_messages(self):
        """fork() copies all messages."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        forked = chat.fork()
        assert len(forked.items) == len(chat.items)
        for i in range(len(chat.items)):
            assert forked.items[i].role.name.lower() == chat.items[i].role.name.lower()
            assert forked.items[i].text == chat.items[i].text

    def test_fork_is_independent(self):
        """Forked chat is independent (tested via from_dict on fork)."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        forked = chat.fork()
        # Can't directly modify messages, but fork should have same content
        assert len(forked.items) == len(chat.items)

        # Fork and reset forked - original should be unchanged
        forked.reset()
        assert len(forked.items) == 0
        assert len(chat.items) == 2  # Original unchanged

    def test_fork_copies_config(self):
        """fork() copies config."""
        config = GenerationConfig(temperature=0.5)
        chat = Chat(config=config)

        forked = chat.fork()
        assert forked.config.temperature == 0.5

        # Changing forked config doesn't affect original
        forked.config = GenerationConfig(temperature=0.9)
        assert chat.config.temperature == 0.5


# =============================================================================
# Config Override Logic Tests
# =============================================================================


class TestConfigOverrideLogic:
    """Tests for _build_effective_config logic."""

    def test_returns_session_config_by_default(self):
        """_build_effective_config returns session config when no overrides."""
        config = GenerationConfig(temperature=0.5, max_tokens=100)
        chat = Chat(config=config)

        effective = chat._build_effective_config()
        assert effective.temperature == 0.5
        assert effective.max_tokens == 100

    def test_config_param_overrides_session(self):
        """config parameter overrides session config."""
        session_config = GenerationConfig(temperature=0.5)
        override_config = GenerationConfig(temperature=0.9)
        chat = Chat(config=session_config)

        effective = chat._build_effective_config(config=override_config)
        assert effective.temperature == 0.9

    def test_kwargs_override_session(self):
        """kwargs override session config."""
        config = GenerationConfig(temperature=0.5)
        chat = Chat(config=config)

        effective = chat._build_effective_config(temperature=0.1)
        assert effective.temperature == 0.1

    def test_kwargs_override_config_param(self):
        """kwargs override explicit config parameter."""
        session_config = GenerationConfig(temperature=0.5)
        override_config = GenerationConfig(temperature=0.9)
        chat = Chat(config=session_config)

        effective = chat._build_effective_config(config=override_config, temperature=0.1)
        assert effective.temperature == 0.1

    def test_unknown_kwarg_raises(self):
        """Unknown kwargs raise ValidationError."""
        chat = Chat()
        with pytest.raises(talu.ValidationError, match="Unknown"):
            chat._build_effective_config(invalid_param=42)


# =============================================================================
# Storage Tests
# =============================================================================


class TestStorage:
    """Tests for storage functionality."""

    def test_storage_parameter_accepted(self):
        """Chat accepts storage parameter."""
        from talu.db import Database as Storage

        storage = Storage()
        chat = Chat(storage=storage)
        assert chat._storage is storage

    def test_default_storage_is_storage(self):
        """Default storage is Storage."""
        from talu.db import Database as Storage

        chat = Chat()
        assert isinstance(chat._storage, Storage)

    def test_fork_preserves_storage(self):
        """fork() preserves storage reference."""
        from talu.db import Database

        storage = Database()
        chat = Chat(system="Test", storage=storage)
        forked = chat.fork()
        assert forked._storage is storage

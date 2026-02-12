"""
Tests for streaming Token class, Hook observability, config composition, and message input normalization.

Covers:
- Token(str) subclass for streaming with metadata
- GenerationConfig.__or__ for composing configs
- Hook protocol and HookManager for observability
- normalize_message_input for MessageItem symmetry
"""


class TestTokenClass:
    """Tests for Token(str) subclass."""

    def test_token_is_str_subclass(self):
        """Token should be a str subclass."""
        from talu.chat import Token

        token = Token("hello")
        assert isinstance(token, str)
        assert isinstance(token, Token)

    def test_token_string_operations(self):
        """Token should work like a regular string."""
        from talu.chat import Token

        token = Token("hello")
        assert token == "hello"
        assert token + " world" == "hello world"
        assert len(token) == 5
        assert token.upper() == "HELLO"
        assert str(token) == "hello"

    def test_token_metadata_defaults(self):
        """Token should have default metadata values."""
        from talu.chat import Token

        token = Token("hello")
        assert token.id == -1
        assert token.logprob is None
        assert token.is_special is False
        assert token.finish_reason is None

    def test_token_metadata_explicit(self):
        """Token should accept explicit metadata."""
        from talu.chat import Token

        token = Token(
            "hello",
            id=42,
            logprob=-0.5,
            is_special=True,
            finish_reason="eos_token",
        )
        assert token == "hello"
        assert token.id == 42
        assert token.logprob == -0.5
        assert token.is_special is True
        assert token.finish_reason == "eos_token"

    def test_token_repr(self):
        """Token should have informative repr."""
        from talu.chat import Token

        # Simple token
        token = Token("hi")
        repr_str = repr(token)
        assert "Token(" in repr_str
        assert "hi" in repr_str

        # Token with metadata
        token = Token("hi", id=42, logprob=-0.5)
        repr_str = repr(token)
        assert "id=42" in repr_str
        assert "logprob=-0.500" in repr_str

    def test_token_concatenation_yields_str(self):
        """Concatenating tokens should yield a string."""
        from talu.chat import Token

        t1 = Token("hello")
        t2 = Token(" world")
        result = t1 + t2
        assert result == "hello world"
        # Concatenation yields str, not Token (standard Python behavior)
        assert type(result) is str

    def test_token_in_list(self):
        """Tokens should work in lists and joins."""
        from talu.chat import Token

        tokens = [Token("hello"), Token(" "), Token("world")]
        result = "".join(tokens)
        assert result == "hello world"


class TestGenerationConfigComposition:
    """Tests for GenerationConfig.__or__ operator."""

    def test_or_basic_merge(self):
        """Pipe operator should merge configs."""
        from talu.router import GenerationConfig

        creative = GenerationConfig(temperature=1.2, top_p=0.95)
        json_mode = GenerationConfig(stop_sequences=["}"])

        merged = creative | json_mode
        assert merged.temperature == 1.2
        assert merged.top_p == 0.95
        assert merged.stop_sequences == ["}"]

    def test_or_right_wins(self):
        """Right side should override left for same fields."""
        from talu.router import GenerationConfig

        left = GenerationConfig(temperature=0.5, max_tokens=100)
        right = GenerationConfig(temperature=1.0)

        merged = left | right
        assert merged.temperature == 1.0  # Right wins
        assert merged.max_tokens == 100  # Left preserved

    def test_or_preserves_defaults(self):
        """Default values on right side should not override left."""
        from talu.router import GenerationConfig

        left = GenerationConfig(temperature=0.5, max_tokens=500)
        right = GenerationConfig()  # All defaults

        merged = left | right
        # Left values should be preserved since right has all defaults
        assert merged.temperature == 0.5
        assert merged.max_tokens == 500

    def test_or_chain(self):
        """Pipe operator should chain multiple configs."""
        from talu.router import GenerationConfig

        sampling = GenerationConfig(temperature=0.7, top_k=40)
        limits = GenerationConfig(max_tokens=500)
        stops = GenerationConfig(stop_sequences=["END"])

        merged = sampling | limits | stops
        assert merged.temperature == 0.7
        assert merged.top_k == 40
        assert merged.max_tokens == 500
        assert merged.stop_sequences == ["END"]

    def test_or_returns_new_config(self):
        """Pipe operator should not mutate original configs."""
        from talu.router import GenerationConfig

        left = GenerationConfig(temperature=0.5)
        right = GenerationConfig(temperature=1.0)

        merged = left | right
        assert merged is not left
        assert merged is not right
        assert left.temperature == 0.5  # Unchanged
        assert right.temperature == 1.0  # Unchanged

    def test_or_with_non_config(self):
        """Pipe with non-config should return NotImplemented."""
        from talu.router import GenerationConfig

        config = GenerationConfig(temperature=0.5)
        result = config.__or__("not a config")
        assert result is NotImplemented


class TestHooks:
    """Tests for Hook protocol and HookManager."""

    def test_hook_base_class(self):
        """Hook base class should be instantiable with no-op methods."""
        from talu.chat import Hook

        hook = Hook()
        # Should not raise
        hook.on_generation_start(None, "test")
        hook.on_first_token(None, 100.0)
        hook.on_generation_end(None, None)

    def test_hook_manager_initialization(self):
        """HookManager should initialize with or without hooks."""
        from talu.chat import Hook, HookManager

        # Empty manager
        manager = HookManager()
        assert manager.hooks == []

        # Manager with hooks
        hook = Hook()
        manager = HookManager([hook])
        assert len(manager.hooks) == 1

    def test_hook_manager_add_remove(self):
        """HookManager should support add/remove."""
        from talu.chat import Hook, HookManager

        manager = HookManager()
        hook = Hook()

        manager.add(hook)
        assert hook in manager.hooks

        manager.remove(hook)
        assert hook not in manager.hooks

    def test_hook_manager_dispatch(self):
        """HookManager should dispatch to all hooks."""
        from talu.chat import Hook, HookManager

        calls = []

        class RecordingHook(Hook):
            def __init__(self, name):
                self.name = name

            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append((self.name, "start", input_text))

            def on_first_token(self, chat, time_ms):
                calls.append((self.name, "first", time_ms))

            def on_generation_end(self, chat, response, *, error=None):
                calls.append((self.name, "end", error))

        manager = HookManager([RecordingHook("A"), RecordingHook("B")])

        manager.dispatch_start(None, "hello")
        assert ("A", "start", "hello") in calls
        assert ("B", "start", "hello") in calls

        manager.dispatch_first_token(None, 50.0)
        assert ("A", "first", 50.0) in calls
        assert ("B", "first", 50.0) in calls

        manager.dispatch_end(None, None, error=None)
        assert ("A", "end", None) in calls
        assert ("B", "end", None) in calls

    def test_hook_error_isolation(self):
        """Hook errors should not break other hooks."""
        from talu.chat import Hook, HookManager

        calls = []

        class FailingHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                raise RuntimeError("hook failed")

        class RecordingHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append("recording called")

        manager = HookManager([FailingHook(), RecordingHook()])

        # Should not raise, and second hook should still be called
        manager.dispatch_start(None, "test")
        assert "recording called" in calls


class TestNormalizeMessageInput:
    """Tests for normalize_message_input and MessageItem in send()."""

    def test_normalize_string(self):
        """String input should pass through."""
        from talu.types import normalize_message_input

        result = normalize_message_input("hello")
        assert result == "hello"

    def test_normalize_dict_list(self):
        """List of dicts should pass through."""
        from talu.types import normalize_message_input

        content = [{"type": "text", "text": "hello"}]
        result = normalize_message_input(content)
        assert result == content

    def test_normalize_message_item(self):
        """MessageItem should be normalized to list[dict]."""
        from talu.types import MessageItem, normalize_message_input

        item = MessageItem.create("user", "Hello!")
        result = normalize_message_input(item)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "Hello!"

    def test_normalize_message_item_list(self):
        """List of MessageItems should be flattened."""
        from talu.types import MessageItem, normalize_message_input

        items = [
            MessageItem.create("user", "Hello"),
            MessageItem.create("assistant", "Hi there"),
        ]
        result = normalize_message_input(items)

        assert isinstance(result, list)
        assert len(result) == 2
        texts = [r["text"] for r in result]
        assert "Hello" in texts
        assert "Hi there" in texts

    def test_normalize_mixed_list(self):
        """Mixed list of MessageItems and dicts should work."""
        from talu.types import MessageItem, normalize_message_input

        mixed = [
            MessageItem.create("user", "Hello"),
            {"type": "input_text", "text": "World"},
        ]
        result = normalize_message_input(mixed)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_normalize_empty_content(self):
        """MessageItem with empty content should produce placeholder."""
        from talu.types import (
            ItemStatus,
            MessageItem,
            MessageRole,
            normalize_message_input,
        )

        # Create MessageItem with empty content tuple
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.USER,
            content=(),  # Empty content
        )
        result = normalize_message_input(item)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == ""


class TestClientHooks:
    """Tests for Client hooks integration."""

    def test_client_accepts_hooks(self):
        """Client should accept hooks parameter."""
        from talu.chat import Hook

        # We can't actually create a Client without a model, but we can test
        # that the Hook class exists and is importable
        hook = Hook()
        assert hook is not None

    def test_hook_export(self):
        """Hook and HookManager should be exported from talu.chat."""
        from talu.chat import Hook, HookManager

        assert Hook is not None
        assert HookManager is not None


class TestTokenExport:
    """Tests for Token export."""

    def test_token_export_from_chat(self):
        """Token should be exported from talu.chat."""
        from talu.chat import Token

        assert Token is not None

    def test_token_export_from_response(self):
        """Token should be exported from talu.chat.response."""
        from talu.chat.response import Token

        assert Token is not None

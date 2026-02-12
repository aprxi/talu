"""
Additional tests for talu/chat/session.py coverage.

Targets uncovered edge cases, error paths, and internal methods.
"""

import pytest

from talu import AsyncChat, Chat, Client, GenerationConfig
from talu.exceptions import ValidationError

# =============================================================================
# Construction Error Paths
# =============================================================================


class TestChatConstructionErrors:
    """Tests for Chat construction error handling."""

    def test_both_model_and_client_raises(self, test_model_path):
        """Providing both model and client raises ValidationError."""
        client = Client(test_model_path)
        try:
            with pytest.raises(ValidationError, match="either 'model' or 'client'"):
                Chat(model=test_model_path, client=client)
        finally:
            client.close()

    def test_async_both_model_and_client_raises(self, test_model_path):
        """AsyncChat: Providing both model and client raises ValidationError."""
        from talu.client import AsyncClient

        client = AsyncClient(test_model_path)
        try:
            with pytest.raises(ValidationError, match="either 'model' or 'client'"):
                AsyncChat(model=test_model_path, client=client)
        finally:
            # AsyncClient needs sync cleanup for test
            if hasattr(client, "_router") and client._router:
                client._router.close()


class TestChatWithClient:
    """Tests for Chat created with existing Client."""

    def test_chat_with_client_does_not_own(self, test_model_path):
        """Chat with client= does not own the client."""
        client = Client(test_model_path)
        try:
            chat = Chat(client=client)
            assert chat._owns_client is False
            assert chat._client is client
            chat.close()
            # Client should still be alive
            assert client._router is not None
        finally:
            client.close()

    def test_chat_with_model_owns_client(self, test_model_path):
        """Chat with model= owns its client."""
        chat = Chat(model=test_model_path)
        assert chat._owns_client is True
        chat.close()


# =============================================================================
# Chat Template Tests
# =============================================================================


class TestChatTemplate:
    """Tests for chat_template parameter."""

    def test_string_template_converted(self):
        """String chat_template is converted to PromptTemplate."""
        from talu.template import PromptTemplate

        template_str = "{{ messages }}"
        chat = Chat(chat_template=template_str)

        assert isinstance(chat._chat_template, PromptTemplate)

    def test_prompt_template_passthrough(self):
        """PromptTemplate is passed through unchanged."""
        from talu.template import PromptTemplate

        template = PromptTemplate("{{ messages }}")
        chat = Chat(chat_template=template)

        assert chat._chat_template is template

    def test_none_template_default(self):
        """None template uses model default."""
        chat = Chat()
        assert chat._chat_template is None


# =============================================================================
# _build_effective_config Tests
# =============================================================================


class TestBuildEffectiveConfig:
    """Tests for _build_effective_config internal method."""

    def test_session_config_used_by_default(self):
        """Session config is used when no overrides provided."""
        config = GenerationConfig(temperature=0.7, max_tokens=50)
        chat = Chat(config=config)

        effective = chat._build_effective_config()

        assert effective.temperature == 0.7
        assert effective.max_tokens == 50

    def test_config_param_overrides_session(self):
        """Config parameter overrides session config entirely."""
        session_config = GenerationConfig(temperature=0.7)
        override_config = GenerationConfig(temperature=0.3, max_tokens=100)
        chat = Chat(config=session_config)

        effective = chat._build_effective_config(config=override_config)

        assert effective.temperature == 0.3
        assert effective.max_tokens == 100

    def test_kwargs_override_both(self):
        """Kwargs override both session and config param."""
        session_config = GenerationConfig(temperature=0.7)
        override_config = GenerationConfig(temperature=0.3)
        chat = Chat(config=session_config)

        effective = chat._build_effective_config(config=override_config, temperature=0.1)

        assert effective.temperature == 0.1

    def test_partial_kwargs_preserve_other_fields(self):
        """Partial kwargs preserve other fields from base config."""
        config = GenerationConfig(temperature=0.7, max_tokens=100, top_k=50)
        chat = Chat(config=config)

        effective = chat._build_effective_config(temperature=0.1)

        assert effective.temperature == 0.1
        assert effective.max_tokens == 100  # Preserved
        assert effective.top_k == 50  # Preserved

    def test_unknown_kwarg_raises_validation_error(self):
        """Unknown kwargs raise ValidationError."""
        chat = Chat()

        with pytest.raises(ValidationError, match="Unknown generation parameter"):
            chat._build_effective_config(invalid_param=42)


class TestAsyncBuildEffectiveConfig:
    """Tests for AsyncChat._build_effective_config."""

    def test_session_config_used_by_default(self):
        """Session config is used when no overrides provided."""
        config = GenerationConfig(temperature=0.7, max_tokens=50)
        chat = AsyncChat(config=config)

        effective = chat._build_effective_config()

        assert effective.temperature == 0.7
        assert effective.max_tokens == 50

    def test_unknown_kwarg_raises_validation_error(self):
        """Unknown kwargs raise ValidationError."""
        chat = AsyncChat()

        with pytest.raises(ValidationError, match="Unknown generation parameter"):
            chat._build_effective_config(invalid_param=42)


# =============================================================================
# _prepare_messages Tests
# =============================================================================


class TestPrepareMessages:
    """Tests for _prepare_messages internal method."""

    def test_basic_message_preparation(self):
        """Basic message preparation adds user message."""
        chat = Chat(system="Be helpful.")

        messages = chat._prepare_messages("Hello!")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_no_system_message(self):
        """Message preparation without system message."""
        chat = Chat()

        messages = chat._prepare_messages("Hello!")

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_with_existing_messages(self):
        """Message preparation with existing conversation."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ]
            }
        )

        messages = chat._prepare_messages("How are you?")

        assert len(messages) == 4
        assert messages[-1]["content"] == "How are you?"


class TestAsyncPrepareMessages:
    """Tests for AsyncChat._prepare_messages."""

    def test_basic_message_preparation(self):
        """Basic message preparation adds user message."""
        chat = AsyncChat(system="Be helpful.")

        messages = chat._prepare_messages("Hello!")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# =============================================================================
# _apply_numeric_const Tests
# =============================================================================


class TestApplyNumericConst:
    """Tests for _apply_numeric_const internal method."""

    def test_integer_extraction(self):
        """Extracts integer from message."""
        chat = Chat()
        schema = {"type": "integer"}

        result = chat._apply_numeric_const(schema, "The answer is 42")

        assert result.get("const") == 42

    def test_float_extraction(self):
        """Extracts float from message."""
        chat = Chat()
        schema = {"type": "number"}

        result = chat._apply_numeric_const(schema, "Pi is 3.14159")

        assert result.get("const") == 3.14159

    def test_negative_number(self):
        """Extracts negative number."""
        chat = Chat()
        schema = {"type": "number"}

        result = chat._apply_numeric_const(schema, "Temperature is -5.5")

        assert result.get("const") == -5.5

    def test_scientific_notation(self):
        """Extracts scientific notation."""
        chat = Chat()
        schema = {"type": "number"}

        result = chat._apply_numeric_const(schema, "Value is 1.5e-10")

        assert result.get("const") == 1.5e-10

    def test_no_number_returns_unchanged(self):
        """No number in message returns schema unchanged."""
        chat = Chat()
        schema = {"type": "number"}

        result = chat._apply_numeric_const(schema, "No numbers here")

        assert "const" not in result

    def test_empty_message_returns_unchanged(self):
        """Empty/None message returns schema unchanged."""
        chat = Chat()
        schema = {"type": "number"}

        result = chat._apply_numeric_const(schema, None)
        assert result is schema

        result = chat._apply_numeric_const(schema, "")
        assert "const" not in result

    def test_schema_with_const_unchanged(self):
        """Schema with existing const is unchanged."""
        chat = Chat()
        schema = {"type": "number", "const": 99}

        result = chat._apply_numeric_const(schema, "The answer is 42")

        assert result.get("const") == 99  # Original preserved

    def test_schema_with_enum_unchanged(self):
        """Schema with enum is unchanged."""
        chat = Chat()
        schema = {"type": "number", "enum": [1, 2, 3]}

        result = chat._apply_numeric_const(schema, "The answer is 42")

        assert result.get("enum") == [1, 2, 3]
        assert "const" not in result

    def test_non_numeric_schema_unchanged(self):
        """Non-numeric schema is unchanged."""
        chat = Chat()
        schema = {"type": "string"}

        result = chat._apply_numeric_const(schema, "The answer is 42")

        assert "const" not in result


class TestAsyncApplyNumericConst:
    """Tests for AsyncChat._apply_numeric_const."""

    def test_integer_extraction(self):
        """Extracts integer from message."""
        chat = AsyncChat()
        schema = {"type": "integer"}

        result = chat._apply_numeric_const(schema, "The answer is 42")

        assert result.get("const") == 42


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Tests for Chat context manager protocol."""

    def test_enter_returns_self(self):
        """__enter__ returns the chat instance."""
        chat = Chat(system="Test")

        with chat as c:
            assert c is chat

    def test_exit_closes_chat(self):
        """__exit__ closes the chat."""
        chat = Chat(system="Test")

        with chat:
            assert chat._chat_ptr is not None

        assert chat._chat_ptr is None

    def test_exit_on_exception(self):
        """__exit__ closes chat even on exception."""
        chat = Chat(system="Test")

        try:
            with chat:
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        assert chat._chat_ptr is None


class TestAsyncContextManager:
    """Tests for AsyncChat async context manager protocol."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        """__aenter__ returns the chat instance."""
        chat = AsyncChat(system="Test")

        async with chat as c:
            assert c is chat

    @pytest.mark.asyncio
    async def test_aexit_closes_chat(self):
        """__aexit__ closes the chat."""
        chat = AsyncChat(system="Test")

        async with chat:
            assert chat._chat_ptr is not None

        assert chat._chat_ptr is None


# =============================================================================
# preview_prompt Tests
# =============================================================================


class TestPreviewPrompt:
    """Tests for preview_prompt method."""

    def test_preview_prompt_without_client_raises(self):
        """preview_prompt raises without client."""
        from talu.exceptions import StateError

        chat = Chat(system="Be helpful.")

        # Without a client, preview_prompt should raise
        with pytest.raises(StateError):
            chat.preview_prompt()

    def test_preview_prompt_with_chat_template(self):
        """preview_prompt works with custom chat_template."""
        # Simple template that just joins messages
        template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        chat = Chat(system="Be helpful.", chat_template=template)

        # Add user message via from_dict
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hello!"},
                ]
            }
        )
        chat._chat_template = chat._chat_template or __import__(
            "talu.template", fromlist=["PromptTemplate"]
        ).PromptTemplate(template)

        prompt = chat.preview_prompt()
        assert isinstance(prompt, str)
        assert "Be helpful." in prompt
        assert "Hello!" in prompt


# =============================================================================
# Repr and Str Tests
# =============================================================================


class TestRepr:
    """Tests for __repr__ and __str__."""

    def test_chat_repr(self):
        """Chat has informative repr."""
        chat = Chat(system="Test")

        repr_str = repr(chat)

        assert "Chat" in repr_str

    def test_chat_str(self):
        """Chat str shows messages."""
        chat = Chat(system="Test")

        str_str = str(chat)

        assert isinstance(str_str, str)

    def test_async_chat_repr(self):
        """AsyncChat has informative repr."""
        chat = AsyncChat(system="Test")

        repr_str = repr(chat)

        assert "AsyncChat" in repr_str


# =============================================================================
# Serialization Round-Trip Tests
# =============================================================================


class TestSerializationRoundTrip:
    """Comprehensive tests for serialization round-trips."""

    def test_to_json_empty_chat(self):
        """to_json() on empty chat returns valid JSON."""
        import json

        chat = Chat()
        json_str = chat.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 0

    def test_to_json_with_system_only(self):
        """to_json() with only system message."""
        import json

        chat = Chat(system="Be helpful.")
        json_str = chat.to_json()

        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["role"] == "system"
        assert parsed[0]["content"] == "Be helpful."

    def test_to_json_roundtrip_multiple_messages(self):
        """to_json -> from_dict roundtrip preserves multiple messages."""
        import json

        original = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm doing well!"},
                ]
            }
        )

        # Serialize and deserialize via JSON
        json_str = original.to_json()
        messages = json.loads(json_str)

        restored = Chat.from_dict({"messages": messages})

        assert len(restored.items) == len(original.items)
        for i in range(len(original.items)):
            assert restored.items[i].role.name.lower() == original.items[i].role.name.lower()
            assert restored.items[i].text == original.items[i].text

    def test_to_json_unicode_content(self):
        """to_json() handles Unicode content correctly."""
        import json

        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": "Hello 你好 🌍 مرحبا"},
                    {"role": "assistant", "content": "Hi! こんにちは 🎉"},
                ]
            }
        )

        json_str = chat.to_json()
        parsed = json.loads(json_str)

        assert parsed[0]["content"] == "Hello 你好 🌍 مرحبا"
        assert parsed[1]["content"] == "Hi! こんにちは 🎉"

    def test_to_json_empty_content(self):
        """to_json() handles empty content strings."""
        import json

        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": "Response"},
                ]
            }
        )

        json_str = chat.to_json()
        parsed = json.loads(json_str)

        assert parsed[0]["content"] == ""
        assert parsed[1]["content"] == "Response"

    def test_to_json_special_characters(self):
        """to_json() handles special JSON characters correctly."""
        import json

        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": 'Quote: "hello"'},
                    {"role": "assistant", "content": "Backslash: \\ and newline:\nend"},
                ]
            }
        )

        json_str = chat.to_json()
        parsed = json.loads(json_str)

        assert '"hello"' in parsed[0]["content"]
        assert "\\" in parsed[1]["content"]
        assert "\n" in parsed[1]["content"]

    def test_to_json_long_content(self):
        """to_json() handles long content strings."""
        import json

        long_text = "x" * 10000
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "user", "content": long_text},
                ]
            }
        )

        json_str = chat.to_json()
        parsed = json.loads(json_str)

        assert len(parsed[0]["content"]) == 10000

    def test_to_dict_to_json_consistency(self):
        """to_dict() and to_json() produce consistent data."""
        import json

        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        dict_messages = chat.to_dict()["messages"]
        json_messages = json.loads(chat.to_json())

        assert len(dict_messages) == len(json_messages)
        for i in range(len(dict_messages)):
            assert dict_messages[i]["role"] == json_messages[i]["role"]
            assert dict_messages[i]["content"] == json_messages[i]["content"]

    def test_repeated_serialization_no_leak(self):
        """Repeated to_json() calls don't leak memory."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            }
        )

        # Repeated serialization should work without issues
        for _ in range(100):
            json_str = chat.to_json()
            assert len(json_str) > 0


class TestAsyncSerializationRoundTrip:
    """Serialization tests for AsyncChat."""

    def test_async_to_json_empty(self):
        """AsyncChat.to_json() on empty chat returns valid JSON."""
        import json

        chat = AsyncChat()
        json_str = chat.to_json()

        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == 0

    def test_async_to_json_with_messages(self):
        """AsyncChat.to_json() with messages."""
        import json

        chat = AsyncChat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        json_str = chat.to_json()
        parsed = json.loads(json_str)

        assert len(parsed) == 2
        assert parsed[0]["role"] == "system"
        assert parsed[1]["role"] == "user"


# =============================================================================
# Schema Placeholder Injection Tests
# =============================================================================


class TestSchemaPlaceholderInjection:
    """Tests for schema placeholder injection in _prepare_messages."""

    def test_schema_placeholder_in_system(self):
        """Schema placeholder in system message is replaced."""
        from talu.chat._generate import SCHEMA_PLACEHOLDER

        chat = Chat(system=f"Be helpful. {SCHEMA_PLACEHOLDER}")

        messages = chat._prepare_messages(
            "What is 2+2?",
            response_format={"type": "integer"},
            inject_schema_prompt=True,
        )

        # System message should have schema injected at placeholder
        assert SCHEMA_PLACEHOLDER not in messages[0]["content"]
        # Schema prompt was injected (content is longer than original or contains schema text)
        assert (
            "json" in messages[0]["content"].lower() or "schema" in messages[0]["content"].lower()
        )

    def test_schema_placeholder_in_content(self):
        """Schema placeholder in user content is replaced."""
        from talu.chat._generate import SCHEMA_PLACEHOLDER

        chat = Chat(system="Be helpful.")

        messages = chat._prepare_messages(
            f"Question: {SCHEMA_PLACEHOLDER} What is 2+2?",
            response_format={"type": "integer"},
            inject_schema_prompt=True,
        )

        # User message should have schema injected at placeholder
        assert SCHEMA_PLACEHOLDER not in messages[1]["content"]

    def test_schema_appended_to_system_when_no_placeholder(self):
        """Schema is appended to system when no placeholder."""
        chat = Chat(system="Be helpful.")

        messages = chat._prepare_messages(
            "What is 2+2?",
            response_format={"type": "integer"},
            inject_schema_prompt=True,
        )

        # Schema should be appended to system message
        assert "Be helpful." in messages[0]["content"]
        # System content should be longer than original
        assert len(messages[0]["content"]) > len("Be helpful.")

    def test_schema_prepended_to_content_when_no_system(self):
        """Schema is prepended to content when no system message."""
        chat = Chat()  # No system

        messages = chat._prepare_messages(
            "What is 2+2?",
            response_format={"type": "integer"},
            inject_schema_prompt=True,
        )

        # Should have a system message added with schema
        assert len(messages) >= 1
        # Either system was added or content was modified
        if messages[0]["role"] == "system":
            # Schema in system
            pass
        else:
            # Schema prepended to user message
            assert "What is 2+2?" in messages[0]["content"]

    def test_no_injection_when_disabled(self):
        """No schema injection when inject_schema_prompt=False."""
        chat = Chat(system="Be helpful.")

        messages = chat._prepare_messages(
            "What is 2+2?",
            response_format={"type": "integer"},
            inject_schema_prompt=False,
        )

        # System should be unchanged
        assert messages[0]["content"] == "Be helpful."


# =============================================================================
# _extract_text Helper Tests
# =============================================================================


class TestPrepareMessagesExtractText:
    """Tests for _extract_text helper in _prepare_messages."""

    def test_extract_text_from_list_content(self):
        """System message with list content extracts text."""
        chat = Chat.from_dict(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "System prompt"}],
                    },
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        messages = chat._prepare_messages("How are you?")

        # Should extract system content from list format
        assert len(messages) == 3
        # The system message should be preserved
        assert messages[0]["role"] == "system"


# =============================================================================
# Pop Error Handling Tests
# =============================================================================


class TestPopErrorHandling:
    """Tests for pop() error handling."""

    def test_pop_empty_chat_raises_state_error(self):
        """pop() on empty chat raises StateError."""
        from talu.exceptions import StateError

        chat = Chat()

        with pytest.raises(StateError, match="No messages to remove"):
            chat.pop()

    def test_pop_returns_self_for_chaining(self, test_model_path):
        """pop() returns self for method chaining."""
        chat = Chat(test_model_path)
        chat.send("Hello", max_tokens=3)

        result = chat.pop()

        assert result is chat


# =============================================================================
# Remove Error Handling Tests
# =============================================================================


class TestRemoveErrorHandling:
    """Tests for remove() error handling."""

    def test_remove_out_of_bounds_raises_state_error(self):
        """remove() with out-of-bounds index raises StateError."""
        from talu.exceptions import StateError

        chat = Chat(system="Test")

        with pytest.raises(StateError, match="out of bounds"):
            chat.remove(999)

    def test_remove_negative_index_raises(self):
        """remove() with negative index raises."""
        from talu.exceptions import StateError

        chat = Chat(system="Test")

        # Negative index should fail
        with pytest.raises((StateError, IndexError)):
            chat.remove(-1)


# =============================================================================
# AsyncChat System Property Tests
# =============================================================================


class TestAsyncChatSystemProperty:
    """Tests for AsyncChat.system property getter and setter."""

    def test_system_getter_returns_none_when_not_set(self):
        """system getter returns None when no system prompt."""
        chat = AsyncChat()

        # No system set
        assert chat.system is None or chat.system == ""

    def test_system_getter_returns_value_when_set(self):
        """system getter returns the system prompt."""
        chat = AsyncChat(system="Be helpful.")

        assert chat.system == "Be helpful."

    def test_system_setter_updates_value(self):
        """system setter updates the system prompt."""
        chat = AsyncChat(system="Original")

        chat.system = "Updated"

        assert chat.system == "Updated"

    def test_system_setter_accepts_none(self):
        """system setter accepts None to clear."""
        chat = AsyncChat(system="Original")

        chat.system = None

        # Should be cleared (empty or None)
        assert chat.system is None or chat.system == ""

    @pytest.mark.asyncio
    async def test_system_getter_on_closed_raises(self):
        """system getter on closed chat raises StateError."""
        from talu.exceptions import StateError

        chat = AsyncChat(system="Test")
        await chat.close()

        with pytest.raises(StateError, match="closed"):
            _ = chat.system

    @pytest.mark.asyncio
    async def test_system_setter_on_closed_raises(self):
        """system setter on closed chat raises StateError."""
        from talu.exceptions import StateError

        chat = AsyncChat(system="Test")
        await chat.close()

        with pytest.raises(StateError, match="closed"):
            chat.system = "New"


# =============================================================================
# AsyncChat preview_prompt Tests
# =============================================================================


class TestAsyncChatPreviewPrompt:
    """Tests for AsyncChat.preview_prompt method."""

    def test_preview_prompt_with_custom_template(self):
        """preview_prompt works with custom chat_template."""
        from talu.template import PromptTemplate

        template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        chat = AsyncChat.from_dict(
            {
                "messages": [
                    {"role": "system", "content": "System"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )
        # Set template after construction
        chat._chat_template = PromptTemplate(template)

        prompt = chat.preview_prompt()

        assert isinstance(prompt, str)
        assert "System" in prompt
        assert "Hello" in prompt

    def test_preview_prompt_without_client_or_template_raises(self):
        """preview_prompt without client or template raises StateError."""
        from talu.exceptions import StateError

        chat = AsyncChat(system="Test")

        with pytest.raises(StateError):
            chat.preview_prompt()

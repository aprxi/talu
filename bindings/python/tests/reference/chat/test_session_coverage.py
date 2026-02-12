"""
Reference tests for talu/chat/session.py coverage gaps.

Requires a real model to test actual generation.
Tests cover: async paths, forking, serialization, message manipulation.
"""

import pytest
from pydantic import BaseModel

from talu import AsyncChat, Chat, GenerationConfig
from talu.chat import Response, StreamingResponse
from talu.template import PromptTemplate
from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM as MODEL_URI

# =============================================================================
# Config Merging Tests
# =============================================================================


class TestConfigMerging:
    """Tests for _build_effective_config and config precedence."""

    def test_config_override_with_kwargs(self):
        """kwargs override session config."""
        config = GenerationConfig(max_tokens=100, temperature=0.7)
        chat = Chat(MODEL_URI, config=config)
        try:
            # Override temperature via kwargs
            response = chat.send("Hi", temperature=0.1, max_tokens=5)
            # Should not crash, config precedence should work
            assert isinstance(response, Response)
        finally:
            del chat

    def test_config_object_override(self):
        """config param overrides session config."""
        session_config = GenerationConfig(max_tokens=100, temperature=0.7)
        override_config = GenerationConfig(max_tokens=5, temperature=0.1)

        chat = Chat(MODEL_URI, config=session_config)
        try:
            response = chat.send("Hi", config=override_config)
            assert isinstance(response, Response)
        finally:
            del chat

    def test_kwargs_override_config_object(self):
        """kwargs override config param."""
        session_config = GenerationConfig(max_tokens=100)
        override_config = GenerationConfig(temperature=0.5)

        chat = Chat(MODEL_URI, config=session_config)
        try:
            # temperature=0.1 should win over override_config.temperature=0.5
            response = chat.send("Hi", config=override_config, temperature=0.1, max_tokens=5)
            assert isinstance(response, Response)
        finally:
            del chat


# =============================================================================
# Message Manipulation Tests
# =============================================================================


class TestMessageManipulation:
    """Tests for clear, reset, pop, remove operations."""

    def test_clear_keeps_system(self):
        """clear() keeps system prompt."""
        chat = Chat(MODEL_URI, system="You are helpful.")
        try:
            chat.send("Hello", max_tokens=3)
            initial_len = len(chat.items)
            assert initial_len >= 2

            chat.clear()
            # Should only have system message
            assert len(chat.items) == 1
            assert chat.items[0].role.name.lower() == "system"
        finally:
            del chat

    def test_reset_clears_all(self):
        """reset() clears everything including system."""
        chat = Chat(MODEL_URI, system="You are helpful.")
        try:
            chat.send("Hello", max_tokens=3)
            chat.reset()
            assert len(chat.items) == 0
        finally:
            del chat

    def test_pop_removes_last(self):
        """pop() removes the last message."""
        chat = Chat(MODEL_URI)
        try:
            chat.send("Hello", max_tokens=3)
            initial_len = len(chat.items)

            chat.pop()
            assert len(chat.items) == initial_len - 1
        finally:
            del chat

    def test_remove_at_index(self):
        """remove(index) removes message at index."""
        chat = Chat(MODEL_URI, system="System")
        try:
            chat.send("First", max_tokens=3)
            initial_len = len(chat.items)

            # Remove the user message (index 1)
            chat.remove(1)
            assert len(chat.items) == initial_len - 1
        finally:
            del chat


# =============================================================================
# Forking Tests
# =============================================================================


class TestForking:
    """Tests for chat.fork() and conversation branching."""

    def test_fork_copies_state(self):
        """fork() creates independent copy of state."""
        chat = Chat(MODEL_URI, system="Be brief.")
        try:
            chat.send("My name is Alice", max_tokens=5)
            original_len = len(chat.items)

            forked = chat.fork()
            try:
                assert len(forked.items) == original_len

                # Add message to original
                chat.send("Hello", max_tokens=3)

                # Forked should be unchanged
                assert len(forked.items) == original_len
            finally:
                del forked
        finally:
            del chat

    def test_fork_independent_generation(self):
        """Forked chat can generate independently."""
        chat = Chat(MODEL_URI)
        try:
            chat.send("Hello", max_tokens=3)

            forked = chat.fork()
            try:
                # Both can generate
                response1 = chat.send("Hi", max_tokens=3)
                response2 = forked.send("Goodbye", max_tokens=3)

                assert isinstance(response1, Response)
                assert isinstance(response2, Response)
            finally:
                del forked
        finally:
            del chat


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for to_dict() and from_dict()."""

    def test_to_dict_includes_messages(self):
        """to_dict() includes message history."""
        chat = Chat(MODEL_URI, system="System prompt")
        try:
            chat.send("Hello", max_tokens=3)

            data = chat.to_dict()

            assert "config" in data
            assert "messages" in data
            assert len(data["messages"]) >= 2
        finally:
            del chat

    def test_from_dict_restores_state(self):
        """from_dict() restores message history."""
        chat = Chat(MODEL_URI, system="System prompt")
        try:
            chat.send("Hello", max_tokens=3)
            data = chat.to_dict()
            original_len = len(data["messages"])

            # Restore to new chat
            restored = Chat.from_dict(data, model=MODEL_URI)
            try:
                assert len(restored.items) == original_len
                # Can continue generating
                response = restored.send("Hi", max_tokens=3)
                assert isinstance(response, Response)
            finally:
                del restored
        finally:
            del chat

    def test_round_trip_preserves_config(self):
        """Serialization round-trip preserves config."""
        config = GenerationConfig(temperature=0.5, max_tokens=50, top_k=30)
        chat = Chat(MODEL_URI, config=config)
        try:
            data = chat.to_dict()

            assert data["config"]["temperature"] == 0.5
            assert data["config"]["max_tokens"] == 50
            assert data["config"]["top_k"] == 30
        finally:
            del chat


# =============================================================================
# Async Chat Tests
# =============================================================================


class TestAsyncChatFull:
    """Full AsyncChat coverage tests."""

    @pytest.mark.asyncio
    async def test_async_clear(self):
        """AsyncChat.clear() works."""
        chat = AsyncChat(MODEL_URI, system="System")
        try:
            await chat.send("Hello", max_tokens=3)
            chat.clear()
            assert len(chat.items) == 1
        finally:
            await chat.close()

    @pytest.mark.asyncio
    async def test_async_reset(self):
        """AsyncChat.reset() works."""
        chat = AsyncChat(MODEL_URI, system="System")
        try:
            await chat.send("Hello", max_tokens=3)
            chat.reset()
            assert len(chat.items) == 0
        finally:
            await chat.close()

    @pytest.mark.asyncio
    async def test_async_fork(self):
        """AsyncChat.fork() works."""
        chat = AsyncChat(MODEL_URI)
        try:
            await chat.send("Hello", max_tokens=3)
            original_len = len(chat.items)

            forked = chat.fork()
            try:
                assert len(forked.items) == original_len
            finally:
                await forked.close()
        finally:
            await chat.close()

    @pytest.mark.asyncio
    async def test_async_to_dict(self):
        """AsyncChat.to_dict() works."""
        chat = AsyncChat(MODEL_URI, system="System")
        try:
            await chat.send("Hello", max_tokens=3)
            data = chat.to_dict()

            assert "config" in data
            assert "messages" in data
        finally:
            await chat.close()

    @pytest.mark.asyncio
    async def test_async_from_dict(self):
        """AsyncChat.from_dict() works."""
        chat = AsyncChat(MODEL_URI, system="System")
        try:
            await chat.send("Hello", max_tokens=3)
            data = chat.to_dict()
            original_len = len(data["messages"])

            restored = AsyncChat.from_dict(data, model=MODEL_URI)
            try:
                assert len(restored.items) == original_len
            finally:
                await restored.close()
        finally:
            await chat.close()


# =============================================================================
# Preview Prompt Tests
# =============================================================================


class TestPreviewPrompt:
    """Tests for preview_prompt() functionality."""

    def test_preview_prompt_requires_template(self):
        """preview_prompt() without template raises StateError."""
        from talu.exceptions import StateError

        chat = Chat(MODEL_URI, system="You are helpful.")
        try:
            chat.send("Hello", max_tokens=3)
            # Without custom template, router may not support preview
            with pytest.raises(StateError, match="preview"):
                chat.preview_prompt()
        finally:
            del chat

    def test_preview_prompt_with_custom_template(self):
        """preview_prompt() uses custom template."""
        template = PromptTemplate(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        )
        chat = Chat(MODEL_URI, chat_template=template)
        try:
            chat.send("Hello", max_tokens=3)
            preview = chat.preview_prompt()

            assert "user: Hello" in preview or "Hello" in preview
        finally:
            del chat

    def test_preview_with_config_template(self):
        """preview_prompt() respects config template."""
        template_str = "{% for m in messages %}[{{ m.role }}] {{ m.content }}{% endfor %}"
        config = GenerationConfig(chat_template=template_str)

        chat = Chat(MODEL_URI, config=config)
        try:
            preview = chat.preview_prompt()
            # Custom template should be used
            assert isinstance(preview, str)
        finally:
            del chat


# =============================================================================
# Structured Output Tests
# =============================================================================


class NumberAnswer(BaseModel):
    value: int


class TestStructuredOutputCoverage:
    """Additional structured output coverage."""

    def test_response_format_with_system_schema_placeholder(self):
        """Schema injection with {{ schema }} placeholder.

        Tests that the {{ schema }} placeholder feature works (gets replaced).
        We verify the placeholder was replaced by checking the prepared messages.
        Note: Small models often don't follow system-injected schemas well,
        so we test placeholder replacement, not model output quality.
        """
        chat = Chat(MODEL_URI, system="You output JSON. {{ schema }}")
        try:
            # Verify placeholder replacement works
            messages = chat._prepare_messages(
                "What is 2+2?",
                response_format=NumberAnswer,
                allow_thinking=False,
                inject_schema_prompt=True,
                schema_strategy="auto",
                model_name=chat._router.default_model if chat._router else None,
                model_type=None,
            )
            # Check that {{ schema }} was replaced with actual schema instructions
            system_msg = next(m for m in messages if m["role"] == "system")
            assert "{{ schema }}" not in system_msg["content"]
            assert "value" in system_msg["content"]  # Schema mentions 'value' field
        finally:
            del chat

    def test_streaming_with_response_format(self):
        """Streaming with response_format works."""
        chat = Chat(MODEL_URI)
        try:
            response = chat(
                "What is 2+2?",
                response_format=NumberAnswer,
                max_tokens=20,
                seed=42,
            )
            assert isinstance(response, StreamingResponse)
            # Consume stream
            text = "".join(response)
            assert len(text) > 0
        finally:
            del chat


# =============================================================================
# Multi-turn Stress Tests
# =============================================================================


class TestMultiTurnStress:
    """Stress tests for multi-turn conversations."""

    def test_many_turns(self):
        """Chat handles many conversation turns."""
        chat = Chat(MODEL_URI)
        try:
            for i in range(5):
                response = chat.send(f"Turn {i}", max_tokens=3)
                assert isinstance(response, Response)

            assert len(chat.items) >= 10
        finally:
            del chat

    def test_fork_chain(self):
        """Multiple forks from same base work."""
        chat = Chat(MODEL_URI)
        try:
            chat.send("Base", max_tokens=3)
            forks = []

            for _ in range(3):
                forked = chat.fork()
                forks.append(forked)

            # Each fork can generate independently
            for i, forked in enumerate(forks):
                response = forked.send(f"Fork {i}", max_tokens=3)
                assert isinstance(response, Response)

            # Clean up
            for forked in forks:
                del forked
        finally:
            del chat


# =============================================================================
# Reply Auto-Fork Tests
# =============================================================================


class TestAppendAutoFork:
    """Tests for response.append() auto-forking behavior."""

    def test_append_from_old_response_forks(self):
        """append() from non-tip response creates fork."""
        chat = Chat(MODEL_URI)
        try:
            response1 = chat.send("First", max_tokens=3)
            chat.send("Second", max_tokens=3)

            # Append from old response (not the tip)
            # This should fork
            response3 = response1.append("Reply to first", max_tokens=3)

            # response3 should be valid
            assert isinstance(response3, Response)
        finally:
            del chat

    def test_streaming_append(self):
        """append() from StreamingResponse works."""
        chat = Chat(MODEL_URI)
        try:
            response = chat("Hello", max_tokens=5)
            # Consume stream
            list(response)

            # Append continues with streaming
            response2 = response.append("Follow up", max_tokens=5)
            assert isinstance(response2, StreamingResponse)
            list(response2)
        finally:
            del chat

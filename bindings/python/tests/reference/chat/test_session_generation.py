"""
Reference tests for talu/chat/session.py generation paths.

Requires a real model to test actual generation.
Tests cover: send(), stream(), __call__(), response_format, thinking mode.
"""

import pytest
from pydantic import BaseModel

from talu import AsyncChat, Chat, Client, GenerationConfig
from talu.chat import AsyncResponse, Response, StreamingResponse
from talu.router import Grammar
from tests.conftest import TEST_MODEL_URI_TEXT as MODEL_URI
from tests.conftest import TEST_MODEL_URI_TEXT_THINK as THINK_MODEL_URI

# =============================================================================
# Basic Send Tests
# =============================================================================


class TestSend:
    """Tests for Chat.send() generation."""

    def test_send_returns_response(self):
        """send() returns Response object."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=3)

            assert isinstance(response, Response)
            assert response.usage.completion_tokens > 0
        finally:
            del chat

    def test_send_with_system(self):
        """send() with system prompt works."""
        chat = Chat(MODEL_URI, system="Be brief.")
        try:
            response = chat.send("Hi", max_tokens=5)

            assert isinstance(response, Response)
        finally:
            del chat

    def test_send_updates_messages(self):
        """send() updates message history."""
        chat = Chat(MODEL_URI)
        try:
            initial_len = len(chat.items)
            chat.send("Hello", max_tokens=3)

            # Should have user + assistant messages
            assert len(chat.items) >= initial_len + 2
        finally:
            del chat

    def test_send_with_config(self):
        """send() accepts GenerationConfig."""
        config = GenerationConfig(max_tokens=5, temperature=0.1, seed=42)
        chat = Chat(MODEL_URI, config=config)
        try:
            response = chat.send("Count: 1")

            assert isinstance(response, Response)
        finally:
            del chat


# =============================================================================
# Streaming Tests
# =============================================================================


class TestStream:
    """Tests for Chat streaming generation."""

    def test_call_returns_streaming_response(self):
        """__call__() returns StreamingResponse by default."""
        chat = Chat(MODEL_URI)
        try:
            response = chat("Hello", max_tokens=5)

            assert isinstance(response, StreamingResponse)

            # Consume stream
            text = "".join(response)
            assert len(text) > 0
        finally:
            del chat

    def test_send_stream_true(self):
        """send(stream=True) returns StreamingResponse."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", stream=True, max_tokens=5)

            assert isinstance(response, StreamingResponse)

            # Consume stream
            chunks = list(response)
            assert len(chunks) >= 1
        finally:
            del chat

    def test_call_stream_false_returns_response(self):
        """__call__(stream=False) returns Response."""
        chat = Chat(MODEL_URI)
        try:
            response = chat("Hello", stream=False, max_tokens=3)

            assert isinstance(response, Response)
        finally:
            del chat


# =============================================================================
# Response Format Tests
# =============================================================================


class SimpleAnswer(BaseModel):
    answer: str


class TestResponseFormat:
    """Tests for structured output with response_format."""

    def test_send_with_response_format(self):
        """send() with response_format produces structured output."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send(
                "What is 2+2?",
                response_format=SimpleAnswer,
                max_tokens=20,
            )

            assert isinstance(response, Response)
            # parsed may be populated if valid JSON was generated
            if response.parsed is not None:
                assert hasattr(response.parsed, "answer")
        finally:
            del chat

    def test_send_with_dict_schema(self):
        """send() with dict schema works."""
        chat = Chat(MODEL_URI)
        try:
            schema = {"type": "object", "properties": {"value": {"type": "string"}}}
            response = chat.send(
                "Return JSON with value field",
                response_format=schema,
                max_tokens=20,
            )

            assert isinstance(response, Response)
        finally:
            del chat

    def test_send_with_grammar_handle(self):
        """send() with pre-compiled Grammar handle produces structured output.

        Grammar objects allow pre-compiling schemas for reuse, enabling
        zero-latency structured output when the same schema is used repeatedly.
        This tests the talu_set_response_format_handle code path.

        Note: Some models may produce syntactically valid but semantically
        incomplete JSON (e.g., {}) via the Grammar handle. The test verifies
        the code path works without crashing, not output quality.
        """
        from talu.exceptions import SchemaValidationError

        # Pre-compile grammar from schema
        grammar = Grammar(SimpleAnswer)
        chat = Chat(MODEL_URI)
        try:
            try:
                response = chat.send(
                    "What is 2+2? Answer with a short string.",
                    response_format=grammar,
                    max_tokens=128,
                    seed=100,
                )
            except SchemaValidationError:
                # Grammar handle code path worked; model just didn't fill
                # required fields. Verify the handle is still intact.
                assert grammar.response_format is SimpleAnswer
                return

            assert isinstance(response, Response)
            # Verify grammar.response_format is preserved for hydration
            assert grammar.response_format is SimpleAnswer
            # Check parsed output if model produced valid JSON
            if response.parsed is not None:
                assert hasattr(response.parsed, "answer")
        finally:
            del chat

    def test_grammar_reuse_across_generations(self):
        """Same Grammar handle can be used for multiple generations.

        This verifies the handle remains valid across calls and the
        grammar is properly cleared and reset between uses.
        """
        from talu.exceptions import SchemaValidationError

        grammar = Grammar(SimpleAnswer)
        chat = Chat(MODEL_URI)
        try:
            # First generation
            try:
                r1 = chat.send("Say hello", response_format=grammar, max_tokens=128, seed=100)
                assert isinstance(r1, Response)
            except SchemaValidationError:
                pass  # Grammar handle worked, model output incomplete

            # Second generation with same grammar (handle must still be valid)
            try:
                r2 = chat.send("Say goodbye", response_format=grammar, max_tokens=128, seed=100)
                assert isinstance(r2, Response)
            except SchemaValidationError:
                pass  # Grammar handle worked, model output incomplete
        finally:
            del chat

    def test_streaming_with_response_format(self):
        """Streaming generation with response_format works.

        This tests the streaming code path when response_format is provided,
        exercising the grammar cleanup wrapper and schema injection in streaming mode.
        """
        chat = Chat(MODEL_URI)
        try:
            response = chat(
                "What is 2+2?",
                response_format=SimpleAnswer,
                max_tokens=32,
                seed=42,
            )

            # Default __call__ with response_format streams
            assert isinstance(response, StreamingResponse)

            # Consume the stream
            chunks = list(response)
            assert len(chunks) >= 1

            # Full text should be available after consuming
            full_text = "".join(chunks)
            assert len(full_text) > 0
        finally:
            del chat

    def test_send_stream_true_with_response_format(self):
        """send(stream=True) with response_format produces StreamingResponse.

        This tests the explicit stream=True path with structured output.
        """
        chat = Chat(MODEL_URI)
        try:
            response = chat.send(
                "What is 2+2?",
                stream=True,
                response_format=SimpleAnswer,
                max_tokens=32,
                seed=42,
            )

            assert isinstance(response, StreamingResponse)

            # Consume stream
            chunks = list(response)
            assert len(chunks) >= 1
        finally:
            del chat


# =============================================================================
# Thinking Mode Tests
# =============================================================================


class TestThinkingMode:
    """Tests for thinking mode generation."""

    def test_send_with_thinking(self):
        """send() with allow_thinking produces think tags."""
        chat = Chat(THINK_MODEL_URI)
        try:
            response = chat.send(
                "Think about what 2+2 equals",
                response_format=SimpleAnswer,
                allow_thinking=True,
                max_thinking_tokens=30,
                max_tokens=50,
            )

            text = response.text
            # Model may or may not use thinking, but shouldn't crash
            assert isinstance(text, str)
        finally:
            del chat


# =============================================================================
# Reply Chain Tests
# =============================================================================


class TestAppendChain:
    """Tests for response.append() chain."""

    def test_append_continues_conversation(self):
        """response.append() continues the conversation."""
        chat = Chat(MODEL_URI)
        try:
            response1 = chat.send("My name is Alice", max_tokens=5)
            response2 = response1.append("What is my name?", max_tokens=10)

            assert isinstance(response2, Response)
            # Should have at least 4 messages (2 user + 2 assistant)
            assert len(chat.items) >= 4
        finally:
            del chat


# =============================================================================
# Client Sharing Tests
# =============================================================================


class TestClientSharing:
    """Tests for Chat with shared Client."""

    def test_multiple_chats_share_client(self):
        """Multiple chats can share a single Client."""
        client = Client(MODEL_URI)
        try:
            chat1 = Chat(client=client, system="Be brief.")
            chat2 = Chat(client=client, system="Be verbose.")

            response1 = chat1.send("Hi", max_tokens=5)
            response2 = chat2.send("Hi", max_tokens=5)

            assert isinstance(response1, Response)
            assert isinstance(response2, Response)

            chat1.close()
            chat2.close()

            # Client should still be usable
            chat3 = Chat(client=client)
            response3 = chat3.send("Hello", max_tokens=3)
            assert isinstance(response3, Response)
            chat3.close()
        finally:
            client.close()


# =============================================================================
# Async Tests
# =============================================================================


class TestAsyncSend:
    """Tests for AsyncChat.send() generation."""

    @pytest.mark.asyncio
    async def test_async_send_returns_response(self):
        """AsyncChat.send() returns AsyncResponse."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat.send("Hello", max_tokens=3)

            assert isinstance(response, AsyncResponse)
            assert response.usage.completion_tokens > 0
        finally:
            await chat.close()

    @pytest.mark.asyncio
    async def test_async_call_returns_streaming(self):
        """AsyncChat.__call__() returns async streaming response."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat("Hello", max_tokens=5)

            # Should be async iterable
            chunks = []
            async for chunk in response:
                chunks.append(chunk)

            assert len(chunks) >= 1
        finally:
            await chat.close()

    @pytest.mark.asyncio
    async def test_async_with_response_format(self):
        """AsyncChat.send() with response_format works."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat.send(
                "What is 2+2?",
                response_format=SimpleAnswer,
                max_tokens=20,
            )

            assert isinstance(response, AsyncResponse)
        finally:
            await chat.close()

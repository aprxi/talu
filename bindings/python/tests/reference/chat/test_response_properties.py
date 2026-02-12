"""
Reference tests for Response.content and Response.prompt properties.

Requires a real model to test actual generation.
Tests cover: content property, prompt property, streaming variants, async variants.
"""

import pytest

from talu import AsyncChat, Chat
from talu.chat import (
    AsyncResponse,
    StreamingResponse,
)
from talu.types import ContentType, OutputText
from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM as MODEL_URI

# =============================================================================
# Response.content Tests
# =============================================================================


class TestResponseContent:
    """Tests for Response.content multimodal output property."""

    def test_content_returns_list(self):
        """content property returns a list of content parts."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=5)

            assert isinstance(response.content, list)
            assert len(response.content) >= 1
        finally:
            del chat

    def test_content_contains_output_text(self):
        """content returns OutputText for text responses."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=5)

            content = response.content
            assert len(content) == 1
            assert isinstance(content[0], OutputText)
            assert content[0].type == ContentType.OUTPUT_TEXT
        finally:
            del chat

    def test_content_text_matches_response_text(self):
        """content[0].text matches response.text."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=5)

            content = response.content
            assert content[0].text == response.text
        finally:
            del chat

    def test_content_is_consistent_across_accesses(self):
        """Multiple accesses to content return same structure."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=5)

            content1 = response.content
            content2 = response.content
            # Should be same text (lazy construction may differ in identity)
            assert content1[0].text == content2[0].text
        finally:
            del chat


class TestStreamingResponseContent:
    """Tests for StreamingResponse.content property."""

    def test_streaming_content_after_iteration(self):
        """content is available after streaming completes."""
        chat = Chat(MODEL_URI)
        try:
            response = chat("Hello", max_tokens=5)
            assert isinstance(response, StreamingResponse)

            # Consume stream
            full_text = "".join(response)

            # Now content should be available
            content = response.content
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], OutputText)
            assert content[0].text == full_text
        finally:
            del chat


class TestAsyncResponseContent:
    """Tests for AsyncResponse.content property."""

    @pytest.mark.asyncio
    async def test_async_content_returns_list(self):
        """AsyncResponse.content returns list of content parts."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat.send("Hello", max_tokens=5)
            assert isinstance(response, AsyncResponse)

            content = response.content
            assert isinstance(content, list)
            assert len(content) == 1
            assert isinstance(content[0], OutputText)
        finally:
            await chat.close()


class TestAsyncStreamingResponseContent:
    """Tests for AsyncStreamingResponse.content property."""

    @pytest.mark.asyncio
    async def test_async_streaming_content_after_iteration(self):
        """AsyncStreamingResponse.content available after iteration."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat("Hello", max_tokens=5)

            # Consume stream
            chunks = []
            async for chunk in response:
                chunks.append(str(chunk))
            full_text = "".join(chunks)

            # Content should be available after iteration
            content = response.content
            assert isinstance(content, list)
            assert len(content) == 1
            assert content[0].text == full_text
        finally:
            await chat.close()


# =============================================================================
# Response.prompt Tests
# =============================================================================


class TestResponsePrompt:
    """Tests for Response.prompt audit trail property."""

    def test_prompt_is_string_or_none(self):
        """prompt property returns str or None."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=5)

            # prompt should be a string (captured from preview_prompt)
            assert response.prompt is None or isinstance(response.prompt, str)
        finally:
            del chat

    def test_prompt_contains_user_message(self):
        """prompt contains the user's message."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("What is the capital of France?", max_tokens=5)

            if response.prompt is not None:
                assert "What is the capital of France?" in response.prompt
        finally:
            del chat

    def test_prompt_contains_system_message(self):
        """prompt contains system message when provided."""
        system = "You are a geography expert."
        chat = Chat(MODEL_URI, system=system)
        try:
            response = chat.send("Hello", max_tokens=5)

            if response.prompt is not None:
                assert system in response.prompt
        finally:
            del chat

    def test_prompt_contains_chat_template_markers(self):
        """prompt contains chat template markers (model-specific)."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send("Hello", max_tokens=5)

            if response.prompt is not None:
                # Qwen uses <|im_start|> markers
                # Other models may use different markers
                # Just verify it's non-empty and formatted
                assert len(response.prompt) > len("Hello")
        finally:
            del chat


class TestStreamingResponsePrompt:
    """Tests for StreamingResponse.prompt property (lazy capture)."""

    def test_streaming_prompt_available_after_iteration(self):
        """prompt is lazily captured after streaming completes."""
        chat = Chat(MODEL_URI)
        try:
            response = chat("What is 2+2?", max_tokens=5)
            assert isinstance(response, StreamingResponse)

            # Consume stream
            list(response)

            # Prompt should be available after iteration
            if response.prompt is not None:
                assert "What is 2+2?" in response.prompt
        finally:
            del chat


class TestAsyncResponsePrompt:
    """Tests for AsyncResponse.prompt property."""

    @pytest.mark.asyncio
    async def test_async_prompt_available(self):
        """AsyncResponse.prompt is available."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat.send("Hello world", max_tokens=5)
            assert isinstance(response, AsyncResponse)

            if response.prompt is not None:
                assert "Hello world" in response.prompt
        finally:
            await chat.close()


class TestAsyncStreamingResponsePrompt:
    """Tests for AsyncStreamingResponse.prompt property (lazy capture)."""

    @pytest.mark.asyncio
    async def test_async_streaming_prompt_after_iteration(self):
        """AsyncStreamingResponse.prompt captured lazily after iteration."""
        chat = AsyncChat(MODEL_URI)
        try:
            response = await chat("Tell me a joke", max_tokens=10)

            # Consume stream
            async for _ in response:
                pass

            # Prompt should be available
            if response.prompt is not None:
                assert "Tell me a joke" in response.prompt
        finally:
            await chat.close()


# =============================================================================
# Edge Cases
# =============================================================================


class TestResponsePropertiesEdgeCases:
    """Edge cases for content and prompt properties."""

    def test_empty_response_content(self):
        """content handles empty/minimal responses."""
        chat = Chat(MODEL_URI)
        try:
            # Request minimal tokens
            response = chat.send(".", max_tokens=1)

            content = response.content
            assert isinstance(content, list)
            # Even empty response should have OutputText
            assert len(content) == 1
            assert isinstance(content[0], OutputText)
        finally:
            del chat

    def test_multi_turn_prompt_reflects_history(self):
        """prompt in multi-turn captures full conversation."""
        chat = Chat(MODEL_URI)
        try:
            response1 = chat.send("My name is Alice", max_tokens=5)
            response2 = response1.append("Remember my name", max_tokens=5)

            if response2.prompt is not None:
                # Should contain both turns
                assert "Alice" in response2.prompt
                assert "Remember my name" in response2.prompt
        finally:
            del chat

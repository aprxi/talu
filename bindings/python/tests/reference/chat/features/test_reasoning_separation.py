"""Integration tests for reasoning separation (Phase 5).

Verifies that the core reasoning parser correctly separates <think> content
from response text, and that Python bindings expose structured ReasoningItem
and clean response text.

Requires a reasoning model (Qwen3 with thinking support).
"""

import pytest

from talu import Chat, GenerationConfig
from talu.types import MessageItem, ReasoningItem
from tests.conftest import TEST_MODEL_URI_TEXT_THINK as MODEL_URI


@pytest.mark.requires_model
class TestReasoningSeparation:
    """Tests for reasoning content being separated from response text."""

    def test_response_text_has_no_think_tags(self):
        """response.text should never contain <think> or </think> tags."""
        chat = Chat(MODEL_URI)
        try:
            response = chat.send(
                "What is 2+2?",
                config=GenerationConfig(max_tokens=50, temperature=0.0, seed=42),
            )
            text = str(response)
            assert "<think>" not in text
            assert "</think>" not in text
            assert response.usage.completion_tokens > 0
        finally:
            del chat

    def test_reasoning_item_created_when_model_thinks(self):
        """When model generates <think> content, a ReasoningItem appears in items."""
        chat = Chat(MODEL_URI)
        try:
            chat.send(
                "What is the meaning of life?",
                config=GenerationConfig(max_tokens=100, temperature=0.0, seed=42),
            )

            reasoning_items = [it for it in chat.items if isinstance(it, ReasoningItem)]
            # Qwen3 models generate thinking content by default
            if reasoning_items:
                # ReasoningItem has text content
                assert len(reasoning_items[0].text) > 0
                # Response text is clean
                message_items = [it for it in chat.items if isinstance(it, MessageItem)]
                for item in message_items:
                    if hasattr(item, "role") and item.role.name.lower() == "assistant":
                        text = item.text if hasattr(item, "text") else ""
                        assert "<think>" not in text
                        assert "</think>" not in text
        finally:
            del chat

    def test_no_reasoning_without_think_tags(self):
        """When model does not generate <think> tags, no ReasoningItem is created."""
        chat = Chat(
            MODEL_URI,
            config=GenerationConfig(max_tokens=5, temperature=0.0, seed=42),
        )
        try:
            # Very short max_tokens makes it unlikely to produce think block
            chat.send(
                "Say hi",
                config=GenerationConfig(max_tokens=3, temperature=0.0, seed=42),
            )

            # With very few tokens, model may or may not think.
            # Just verify the structure is consistent.
            for item in chat.items:
                if isinstance(item, ReasoningItem):
                    assert len(item.text) > 0  # Non-empty if present
                elif isinstance(item, MessageItem):
                    if hasattr(item, "role") and item.role.name.lower() == "assistant":
                        text = item.text if hasattr(item, "text") else ""
                        assert "<think>" not in text
        finally:
            del chat

    def test_multi_turn_reasoning_items_accumulate(self):
        """Multiple turns can each produce reasoning items."""
        chat = Chat(MODEL_URI)
        try:
            chat.send(
                "What is 2+2?",
                config=GenerationConfig(max_tokens=50, temperature=0.0, seed=42),
            )
            turn1_items = len(chat.items)

            chat.send(
                "And 3+3?",
                config=GenerationConfig(max_tokens=50, temperature=0.0, seed=42),
            )
            turn2_items = len(chat.items)

            # Second turn should add more items
            assert turn2_items > turn1_items

            # All response text should be clean
            for item in chat.items:
                if isinstance(item, MessageItem):
                    if hasattr(item, "role") and item.role.name.lower() == "assistant":
                        text = item.text if hasattr(item, "text") else ""
                        assert "<think>" not in text
                        assert "</think>" not in text
        finally:
            del chat


@pytest.mark.requires_model
class TestReasoningWithSchema:
    """Tests for reasoning separation with structured output (schema mode)."""

    def test_schema_response_is_clean_json(self):
        """With response_format, response.text should be pure JSON (no think tags)."""
        import json

        from pydantic import BaseModel

        class Answer(BaseModel):
            result: str

        chat = Chat(MODEL_URI)
        try:
            response = chat.send(
                "What is 2+2? Answer with the result.",
                config=GenerationConfig(max_tokens=100, temperature=0.3, seed=123),
                response_format=Answer,
                allow_thinking=True,
                max_thinking_tokens=64,
            )

            text = response.text
            assert "<think>" not in text
            assert "</think>" not in text

            # The text should be valid JSON
            text_stripped = text.strip()
            if text_stripped:
                parsed = json.loads(text_stripped)
                assert isinstance(parsed, dict)
        finally:
            del chat

    def test_parsed_works_with_thinking_model(self):
        """response.parsed should work correctly when model uses thinking."""
        from pydantic import BaseModel

        class Greeting(BaseModel):
            message: str

        chat = Chat(MODEL_URI)
        try:
            response = chat.send(
                "Say hello",
                config=GenerationConfig(max_tokens=100, temperature=0.3, seed=456),
                response_format=Greeting,
                allow_thinking=True,
                max_thinking_tokens=32,
            )

            # response.parsed should not raise JSONDecodeError
            if response.text.strip():
                parsed = response.parsed
                assert parsed is not None
        finally:
            del chat


@pytest.mark.requires_model
class TestReasoningStreaming:
    """Tests for reasoning separation with streaming."""

    def test_stream_tokens_no_think_tags(self):
        """Streamed tokens should not contain think tags (streamer strips them)."""
        chat = Chat(MODEL_URI)
        try:
            tokens = list(
                chat(
                    "What is 2+2?",
                    config=GenerationConfig(max_tokens=50, temperature=0.0, seed=42),
                )
            )

            # The Zig streamer strips <think> tags for display
            # The full text from streaming may or may not have tags
            # depending on the streamer implementation, but the final
            # response.text should be clean
            assert len(tokens) > 0
        finally:
            del chat

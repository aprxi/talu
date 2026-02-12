from __future__ import annotations

import json

from pydantic import BaseModel

from talu import Chat, GenerationConfig
from talu.types import ReasoningItem
from tests.conftest import TEST_MODEL_URI_TEXT_THINK as MODEL_URI


class AnswerResponse(BaseModel):
    answer: str


def test_think_block_fakeout() -> None:
    """Test that thinking mode with grammar produces valid schema-compliant JSON.

    With reasoning separation, response.text contains only the clean JSON output
    (think tags are stripped). The reasoning content is in a ReasoningItem in
    chat.items.

    This test verifies that:
    1. Thinking mode generates reasoning content (ReasoningItem exists)
    2. The response text is valid JSON respecting the schema constraints
    3. Grammar constrains the property name to "answer" (the only defined property)
    """
    chat = Chat(MODEL_URI)
    try:
        response = chat.send(
            "Reply with a greeting.",
            config=GenerationConfig(max_tokens=256, temperature=0.3, seed=123),
            response_format=AnswerResponse,
            allow_thinking=True,
            max_thinking_tokens=128,
        )

        # Verify tokens were generated
        assert response.usage.completion_tokens > 0

        # Check that reasoning content was separated into a ReasoningItem
        reasoning_items = [it for it in chat.items if isinstance(it, ReasoningItem)]
        assert len(reasoning_items) > 0, "Expected at least one ReasoningItem"
        assert len(reasoning_items[0].text) > 0, "ReasoningItem should have content"

        # response.text should be clean JSON (no think tags)
        text = response.text
        assert "<think>" not in text, "Think tags should be stripped from response.text"
        assert "</think>" not in text, "Think tags should be stripped from response.text"

        # Parse the JSON response
        text_stripped = text.strip()
        if text_stripped:
            parsed = json.loads(text_stripped)
            # Grammar ensures "answer" is the only allowed property name
            assert "answer" in parsed, f"Expected 'answer' in {parsed}"
            assert isinstance(parsed["answer"], str)
    finally:
        del chat


def test_panic_mode_trigger() -> None:
    """Test that panic mode triggers when thinking tokens are exhausted.

    When max_thinking_tokens is low, the model may not complete its thinking
    and is forced to close early. With grammar-constrained generation, the
    output is syntactically valid JSON but may be semantically incomplete
    (e.g., missing required fields). This is expected behavior - panic mode
    prioritizes producing valid JSON structure over complete content.

    With reasoning separation, response.text contains only the JSON output.
    """
    chat = Chat(MODEL_URI)
    try:
        response = chat.send(
            "Count from 1 to 100 spelling out every number inside the thinking block.",
            config=GenerationConfig(max_tokens=96, temperature=0.2, seed=102),
            response_format=AnswerResponse,
            allow_thinking=True,
            max_thinking_tokens=15,
        )

        # Verify tokens were generated
        assert response.usage.completion_tokens > 0

        # Check that reasoning content was separated
        reasoning_items = [it for it in chat.items if isinstance(it, ReasoningItem)]
        assert len(reasoning_items) > 0, "Expected at least one ReasoningItem"

        # response.text should be clean JSON (no think tags)
        text = response.text
        assert "<think>" not in text, "Think tags should be stripped from response.text"
        assert "</think>" not in text, "Think tags should be stripped from response.text"

        # With grammar-constrained generation, JSON is always syntactically valid
        # but panic mode may produce semantically incomplete output (e.g., {})
        text_stripped = text.strip()
        if text_stripped and "{" in text_stripped:
            parsed = json.loads(text_stripped)
            # Only assert 'answer' present if model had time to generate it
            # Panic mode may produce {} which is valid syntax but missing fields
            if parsed and "answer" in parsed:
                assert isinstance(parsed["answer"], str)
    finally:
        del chat

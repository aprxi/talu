from __future__ import annotations

import json
import math
from typing import Literal

import pytest
from pydantic import BaseModel

from talu import Chat, GenerationConfig
from tests.conftest import TEST_MODEL_URI_TEXT_THINK as MODEL_URI


class EmojiResponse(BaseModel):
    emoji: Literal["🚀", "😊", "📦"]


class KeyValueResponse(BaseModel):
    key: str


def test_unicode_boundary_masking() -> None:
    chat = Chat(MODEL_URI)
    try:
        response = chat.send(
            "Output the rocket emoji.",
            config=GenerationConfig(max_tokens=32, temperature=0.2, seed=105),
            response_format=EmojiResponse,
        )
        parsed = response.parsed
        assert parsed is not None
        assert parsed.emoji in {"🚀", "😊", "📦"}
        assert "\ufffd" not in response.text
    finally:
        del chat


def test_whitespace_composite_tokens() -> None:
    chat = Chat(MODEL_URI)
    try:
        response_min = chat.send(
            'Return JSON with key="value" and no extra text.',
            config=GenerationConfig(max_tokens=64, temperature=0.2, seed=106),
            response_format=KeyValueResponse,
        )
        parsed_min = response_min.parsed
        assert parsed_min is not None
        assert parsed_min.key == "value"

        chat.clear()
        response_pretty = chat.send(
            'Return JSON with key="value"; pretty printed is fine, no extra text.',
            config=GenerationConfig(max_tokens=64, temperature=0.2, seed=107),
            response_format=KeyValueResponse,
        )
        parsed_pretty = response_pretty.parsed
        assert parsed_pretty is not None
        assert parsed_pretty.key == "value"
    finally:
        del chat


@pytest.mark.parametrize(
    ("prompt", "expected_text", "check_negative_zero"),
    [
        ("Output the JSON number 1.5e-10.", "1.5e-10", False),
        ("Output the JSON number -0.0.", "-0.0", True),
    ],
)
def test_numeric_precision_formats(
    prompt: str, expected_text: str, check_negative_zero: bool
) -> None:
    chat = Chat(MODEL_URI)
    try:
        response = chat.send(
            prompt,
            config=GenerationConfig(max_tokens=12, temperature=0.0, seed=108),
            response_format={"type": "number"},
        )
        text = response.text.strip()
        assert expected_text in text
        value = json.loads(text)
        assert isinstance(value, (int, float))
        if check_negative_zero:
            assert text.startswith("-0.0")
            assert math.copysign(1.0, float(value)) < 0
    finally:
        del chat

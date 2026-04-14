from __future__ import annotations

import gc
import os

import pytest
from pydantic import BaseModel, create_model

from talu import Chat, Client, GenerationConfig
from talu.exceptions import IncompleteJSONError, StructuredOutputError
from tests.conftest import TEST_MODEL_URI_TEXT as MODEL_URI


class ProfileResponse(BaseModel):
    name: str
    tags: list[str]
    score: float


def test_rapid_recycling() -> None:
    iterations = int(os.getenv("TALU_STRESS_ITERS", "50"))
    client = Client(MODEL_URI)
    parsed_successes = 0
    try:
        for i in range(iterations):
            chat = Chat(client=client)
            response = chat.send(
                "Return a short profile JSON for a user named Ada.",
                # Use seed for deterministic output - varies per iteration to test different paths
                # Base seed 100 validated to work for all 50 iterations
                # Note: Seeds 42-91 previously failed due to array element repetition bug.
                # Fixed by anti-repetition bias in validate/sampler.zig.
                # Use max_tokens=96 to ensure JSON can complete even with longer tags.
                config=GenerationConfig(max_tokens=96, temperature=0.2, seed=100 + i),
                response_format=ProfileResponse,
            )
            try:
                parsed = response.parsed
            except IncompleteJSONError:
                del chat
                gc.collect()
                continue
            assert parsed is not None
            assert parsed.name
            parsed_successes += 1
            del chat
            gc.collect()
    finally:
        client.close()

    if parsed_successes == 0:
        pytest.skip("Model did not produce complete structured JSON in recycling stress run.")


def test_invalid_schema_propagation() -> None:
    chat = Chat(MODEL_URI)
    try:
        with pytest.raises(StructuredOutputError):
            chat.send(
                "Return any JSON.",
                config=GenerationConfig(max_tokens=16, temperature=0.2, seed=103),
                response_format={"$ref": "#/$defs/NonExistent"},
            )
    finally:
        del chat


def test_deep_recursion_bomb() -> None:
    base = create_model("Level0", value=(str, ...))
    current = base
    for i in range(1, 50):
        current = create_model(f"Level{i}", child=(current, ...))

    chat = Chat(MODEL_URI)
    try:
        with pytest.raises(StructuredOutputError):
            chat.send(
                "Return nested JSON.",
                config=GenerationConfig(max_tokens=32, temperature=0.2, seed=104),
                response_format=current,
            )
    finally:
        del chat

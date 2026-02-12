"""
Tests for session isolation - verifies that Chat instances are properly isolated.

Per AGENTS.md rule 5: "Flaky tests are broken tests: remove nondeterminism"
Per AGENTS.md rule 5: "Fix root causes, not symptoms"

These tests verify the fundamental invariant: same seed must produce same output,
regardless of what happened in prior Chat sessions. If these tests fail, it
indicates a state isolation bug (e.g., KV cache leakage between sessions).
"""

from __future__ import annotations

import gc

from pydantic import BaseModel

from talu import Chat, GenerationConfig
from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM as MODEL_URI


class SimpleResponse(BaseModel):
    answer: str


def test_fresh_chat_same_seed_same_output() -> None:
    """Two fresh Chat instances with same seed must produce identical output.

    This is the most basic isolation test. If this fails, there is global state
    leaking between Chat instances.
    """

    prompt = "Say hello."
    seed = 42
    config = GenerationConfig(max_tokens=32, temperature=0.2, seed=seed)

    # First fresh Chat
    gc.collect()
    chat1 = Chat(MODEL_URI)
    try:
        response1 = chat1.send(prompt, config=config)
        text1 = response1.text
    finally:
        del chat1
        gc.collect()

    # Second fresh Chat with same seed
    gc.collect()
    chat2 = Chat(MODEL_URI)
    try:
        response2 = chat2.send(prompt, config=config)
        text2 = response2.text
    finally:
        del chat2
        gc.collect()

    assert text1 == text2, (
        f"Same seed should produce same output:\n  First:  {text1!r}\n  Second: {text2!r}"
    )


def test_fresh_chat_after_different_prompt_same_seed_same_output() -> None:
    """Fresh Chat after a different prompt with same seed must produce same output.

    This tests that Chat instance cleanup properly resets all state.
    The second Chat should behave identically to a fresh start.
    """

    prompt = "Say hello."
    seed = 42
    config = GenerationConfig(max_tokens=32, temperature=0.2, seed=seed)

    # Get baseline output from fresh Chat
    gc.collect()
    chat_baseline = Chat(MODEL_URI)
    try:
        response_baseline = chat_baseline.send(prompt, config=config)
        baseline_text = response_baseline.text
    finally:
        del chat_baseline
        gc.collect()

    # Run a DIFFERENT prompt first
    gc.collect()
    chat_polluter = Chat(MODEL_URI)
    try:
        _ = chat_polluter.send(
            "What is 2+2? Explain in detail.",
            config=GenerationConfig(max_tokens=64, seed=999),
        )
    finally:
        del chat_polluter
        gc.collect()

    # Now run the original prompt with same seed
    gc.collect()
    chat_test = Chat(MODEL_URI)
    try:
        response_test = chat_test.send(prompt, config=config)
        test_text = response_test.text
    finally:
        del chat_test
        gc.collect()

    assert baseline_text == test_text, (
        f"Fresh Chat after other prompts should produce same output as baseline:\n"
        f"  Baseline: {baseline_text!r}\n"
        f"  After polluter: {test_text!r}"
    )


def test_fresh_chat_with_grammar_same_seed_same_output() -> None:
    """Fresh Chat with grammar constraint and same seed must produce same output.

    Tests isolation when grammar/structured output is involved.
    """

    prompt = "Greet the user."
    seed = 42
    config = GenerationConfig(max_tokens=64, temperature=0.2, seed=seed)

    # First fresh Chat with grammar
    gc.collect()
    chat1 = Chat(MODEL_URI)
    try:
        response1 = chat1.send(prompt, config=config, response_format=SimpleResponse)
        text1 = response1.text
    finally:
        del chat1
        gc.collect()

    # Second fresh Chat with grammar
    gc.collect()
    chat2 = Chat(MODEL_URI)
    try:
        response2 = chat2.send(prompt, config=config, response_format=SimpleResponse)
        text2 = response2.text
    finally:
        del chat2
        gc.collect()

    assert text1 == text2, (
        f"Same seed with grammar should produce same output:\n  First:  {text1!r}\n  Second: {text2!r}"
    )


def test_fresh_chat_with_thinking_same_seed_same_output() -> None:
    """Fresh Chat with thinking mode and same seed must produce same output.

    Tests isolation when thinking mode is involved.
    """

    prompt = "Say hello."
    seed = 42
    config = GenerationConfig(max_tokens=128, temperature=0.2, seed=seed)

    # First fresh Chat with thinking
    gc.collect()
    chat1 = Chat(MODEL_URI)
    try:
        response1 = chat1.send(
            prompt,
            config=config,
            response_format=SimpleResponse,
            allow_thinking=True,
            max_thinking_tokens=64,
        )
        text1 = response1.text
    finally:
        del chat1
        gc.collect()

    # Second fresh Chat with thinking
    gc.collect()
    chat2 = Chat(MODEL_URI)
    try:
        response2 = chat2.send(
            prompt,
            config=config,
            response_format=SimpleResponse,
            allow_thinking=True,
            max_thinking_tokens=64,
        )
        text2 = response2.text
    finally:
        del chat2
        gc.collect()

    assert text1 == text2, (
        f"Same seed with thinking should produce same output:\n  First:  {text1!r}\n  Second: {text2!r}"
    )

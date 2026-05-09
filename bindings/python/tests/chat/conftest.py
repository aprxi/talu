"""Chat test fixtures.

Chat tests that use ``test_model_path`` need a generation-capable model.
They must not use the tokenizer-only tiny-random fixture from global conftest.
"""

import pytest

from talu import Chat
from talu.chat.response import Response

from tests.conftest import (
    TEST_MODEL_URI_TEXT,
    _is_model_available,
    _missing_model_msg,
    _unusable_model_msg,
    _validate_model_loadable_for_generation,
)


@pytest.fixture(scope="session")
def test_model_path(ensure_library_built):
    """Override test_model_path for chat tests to use a generation-capable model."""
    if not _is_model_available(TEST_MODEL_URI_TEXT):
        pytest.skip(_missing_model_msg("TEST_MODEL_URI_TEXT", TEST_MODEL_URI_TEXT))
    load_error = _validate_model_loadable_for_generation(TEST_MODEL_URI_TEXT)
    if load_error is not None:
        pytest.skip(_unusable_model_msg("TEST_MODEL_URI_TEXT", TEST_MODEL_URI_TEXT, load_error))
    return TEST_MODEL_URI_TEXT


@pytest.fixture
def deterministic_chat_generation(monkeypatch):
    """Patch Chat generation for tests that exercise conversation state only."""
    counter = {"count": 0}

    def fake_generate_sync(
        self,
        message,
        config=None,
        stream=False,
        on_token=None,
        response_format=None,
    ):
        del config, on_token, response_format
        assert stream is False

        counter["count"] += 1
        self.append("user", str(message))
        text = f"deterministic response {counter['count']}"
        self.append("assistant", text)
        return Response(text=text, chat=self, _stream_mode=False)

    monkeypatch.setattr(Chat, "_generate_sync", fake_generate_sync)
    return counter

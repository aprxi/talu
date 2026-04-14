"""Reference chat fixtures and model preflight.

These tests require generation-capable models. They should not run against
tokenizer-only fixtures such as tiny-random.
"""

from __future__ import annotations

import pytest

import tests.conftest as root_conftest
from tests.conftest import (
    TEST_MODEL_URI_EMBEDDING,
    TEST_MODEL_URI_TEXT,
    TEST_MODEL_URI_TEXT_THINK,
    _is_model_available,
    _validate_model_loadable_for_embeddings,
    _missing_model_msg,
    _unusable_model_msg,
    _validate_model_loadable_for_generation,
)

# Ensure reference chat modules importing TEST_MODEL_URI_TEXT_RANDOM use a
# generation-capable model in this package.
root_conftest.TEST_MODEL_URI_TEXT_RANDOM = TEST_MODEL_URI_TEXT


@pytest.fixture(scope="session")
def test_model_path(ensure_library_built):
    """Use a generation-capable model for reference chat tests."""
    if not _is_model_available(TEST_MODEL_URI_TEXT):
        pytest.exit(
            _missing_model_msg("TEST_MODEL_URI_TEXT", TEST_MODEL_URI_TEXT),
            returncode=2,
        )
    load_error = _validate_model_loadable_for_generation(TEST_MODEL_URI_TEXT)
    if load_error is not None:
        pytest.exit(
            _unusable_model_msg("TEST_MODEL_URI_TEXT", TEST_MODEL_URI_TEXT, load_error),
            returncode=2,
        )
    return TEST_MODEL_URI_TEXT


@pytest.fixture(scope="session")
def embedding_model_path(ensure_library_built):
    """Use an embedding-capable model for embedding integration tests."""
    if not _is_model_available(TEST_MODEL_URI_EMBEDDING):
        pytest.skip(_missing_model_msg("TEST_MODEL_URI_EMBEDDING", TEST_MODEL_URI_EMBEDDING))
    load_error = _validate_model_loadable_for_embeddings(TEST_MODEL_URI_EMBEDDING)
    if load_error is not None:
        pytest.skip(
            _unusable_model_msg("TEST_MODEL_URI_EMBEDDING", TEST_MODEL_URI_EMBEDDING, load_error)
        )
    return TEST_MODEL_URI_EMBEDDING


def _think_model_available() -> tuple[bool, str]:
    """Return availability and reason for the think-model test fixture."""
    if not _is_model_available(TEST_MODEL_URI_TEXT_THINK):
        return (
            False,
            _missing_model_msg("TEST_MODEL_URI_TEXT_THINK", TEST_MODEL_URI_TEXT_THINK),
        )
    load_error = _validate_model_loadable_for_generation(TEST_MODEL_URI_TEXT_THINK)
    if load_error is not None:
        return (
            False,
            _unusable_model_msg("TEST_MODEL_URI_TEXT_THINK", TEST_MODEL_URI_TEXT_THINK, load_error),
        )
    return True, ""


def pytest_collection_modifyitems(config, items):
    """Skip think-model suites when TEST_MODEL_URI_TEXT_THINK is unavailable."""
    ok, reason = _think_model_available()
    if ok:
        return

    think_paths = (
        "tests/reference/chat/features/test_reasoning_separation.py",
        "tests/reference/chat/schema/test_array_repetition.py",
        "tests/reference/chat/schema/test_tokenization_edge_cases.py",
        "tests/reference/chat/schema/test_thinking_mode.py",
        "tests/reference/chat/test_session_generation.py::TestThinkingMode",
    )
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if any(path in item.nodeid for path in think_paths):
            item.add_marker(skip_marker)

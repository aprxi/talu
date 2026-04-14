"""Chat test fixtures.

Chat tests that use ``test_model_path`` need a generation-capable model.
They must not use the tokenizer-only tiny-random fixture from global conftest.
"""

import pytest

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

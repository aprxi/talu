"""
Additional tests for talu/chat/spec.py coverage.

Targets uncovered internal functions and edge cases.
"""

import pytest

from talu.exceptions import ValidationError
from talu.router import (
    BackendType,
    LocalBackend,
    ModelSpec,
    OpenAICompatibleBackend,
)
from talu.router.spec import (
    _backend_type,
    _normalize_model_input,
    _normalize_ref_prefix,
)

# =============================================================================
# _normalize_ref_prefix Tests
# =============================================================================


class TestNormalizeRefPrefix:
    """Tests for _normalize_ref_prefix internal function."""

    def test_native_prefix_stripped(self):
        """native:: prefix is stripped."""
        result = _normalize_ref_prefix("native::my-model", allow_mapping=True)
        assert result == "my-model"

    def test_native_prefix_stripped_no_mapping(self):
        """native:: prefix stripped even without mapping."""
        result = _normalize_ref_prefix("native::my-model", allow_mapping=False)
        assert result == "my-model"

    def test_openai_colon_converted_when_mapping_allowed(self):
        """openai::model converts to openai://model when mapping allowed."""
        result = _normalize_ref_prefix("openai::gpt-4", allow_mapping=True)
        assert result == "openai://gpt-4"

    def test_oaic_colon_converted_when_mapping_allowed(self):
        """oaic::model converts to oaic://model when mapping allowed."""
        result = _normalize_ref_prefix("oaic::gpt-4o", allow_mapping=True)
        assert result == "oaic://gpt-4o"

    def test_openai_colon_not_converted_when_mapping_disallowed(self):
        """openai::model not converted when mapping disallowed."""
        result = _normalize_ref_prefix("openai::gpt-4", allow_mapping=False)
        assert result == "openai::gpt-4"

    def test_unknown_prefix_unchanged(self):
        """Unknown prefix is unchanged."""
        result = _normalize_ref_prefix("unknown::model", allow_mapping=True)
        assert result == "unknown::model"

    def test_no_prefix_unchanged(self):
        """Model without prefix unchanged."""
        result = _normalize_ref_prefix("Foo/Bar-0B", allow_mapping=True)
        assert result == "Foo/Bar-0B"


# =============================================================================
# _normalize_model_input Tests
# =============================================================================


class TestNormalizeModelInput:
    """Tests for _normalize_model_input internal function."""

    def test_string_creates_model_spec(self):
        """String input creates ModelSpec."""
        result = _normalize_model_input("my-model", None)
        assert isinstance(result, ModelSpec)
        assert result.ref == "my-model"
        assert result.backend is None

    def test_model_spec_passthrough(self):
        """ModelSpec is passed through."""
        spec = ModelSpec(ref="my-model", backend=LocalBackend())
        result = _normalize_model_input(spec, None)
        assert result.ref == "my-model"
        assert isinstance(result.backend, LocalBackend)

    def test_invalid_type_raises(self):
        """Invalid input type raises."""
        with pytest.raises(ValidationError, match="must be str or ModelSpec"):
            _normalize_model_input(12345, None)

    def test_backend_override_replaces_spec_backend(self):
        """backend_override replaces spec's backend."""
        spec = ModelSpec(ref="my-model", backend=LocalBackend())
        override = OpenAICompatibleBackend(api_key="sk-test")
        result = _normalize_model_input(spec, override)
        assert isinstance(result.backend, OpenAICompatibleBackend)
        assert result.backend.api_key == "sk-test"


# =============================================================================
# _backend_type Tests
# =============================================================================


class TestBackendType:
    """Tests for _backend_type internal function."""

    def test_none_returns_unspecified(self):
        """None backend returns UNSPECIFIED."""
        result = _backend_type(None)
        assert result == BackendType.UNSPECIFIED

    def test_local_backend_returns_local(self):
        """LocalBackend returns LOCAL."""
        result = _backend_type(LocalBackend())
        assert result == BackendType.LOCAL

    def test_openai_backend_returns_openai_compatible(self):
        """OpenAICompatibleBackend returns OPENAI_COMPATIBLE."""
        result = _backend_type(OpenAICompatibleBackend())
        assert result == BackendType.OPENAI_COMPATIBLE

    def test_unsupported_backend_raises(self):
        """Unsupported backend type raises."""

        class CustomBackend:
            pass

        with pytest.raises(ValidationError, match="Unsupported backend type"):
            _backend_type(CustomBackend())

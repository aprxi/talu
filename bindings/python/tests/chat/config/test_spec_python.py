import ctypes

import pytest

from talu import router as spec
from talu.router import (
    BackendType,
    LocalBackend,
    ModelSpec,
    OpenAICompatibleBackend,
)
from talu.router._bindings import get_spec_lib

# =============================================================================
# ModelSpec creation tests
# =============================================================================


class TestModelSpecCreation:
    """Test ModelSpec dataclass creation."""

    def test_minimal_spec_with_ref_only(self) -> None:
        s = ModelSpec(ref="Foo/Bar-0B")
        assert s.ref == "Foo/Bar-0B"
        assert s.backend is None

    def test_spec_with_local_backend(self) -> None:
        s = ModelSpec(ref="my-model", backend=LocalBackend())
        assert s.ref == "my-model"
        assert isinstance(s.backend, LocalBackend)

    def test_spec_with_openai_backend(self) -> None:
        s = ModelSpec(
            ref="gpt-4o",
            backend=OpenAICompatibleBackend(api_key="sk-test"),
        )
        assert s.ref == "gpt-4o"
        assert isinstance(s.backend, OpenAICompatibleBackend)
        assert s.backend.api_key == "sk-test"

    def test_spec_is_frozen(self) -> None:
        s = ModelSpec(ref="model")
        with pytest.raises(AttributeError):
            s.ref = "other"  # type: ignore[misc]


# =============================================================================
# LocalBackend tests
# =============================================================================


class TestLocalBackend:
    """Test LocalBackend configuration."""

    def test_default_values(self) -> None:
        b = LocalBackend()
        assert b.gpu_layers == -1
        assert b.use_mmap is True
        assert b.num_threads == 0

    def test_custom_values(self) -> None:
        b = LocalBackend(gpu_layers=4, use_mmap=False, num_threads=8)
        assert b.gpu_layers == 4
        assert b.use_mmap is False
        assert b.num_threads == 8

    def test_is_frozen(self) -> None:
        b = LocalBackend()
        with pytest.raises(AttributeError):
            b.gpu_layers = 10  # type: ignore[misc]


# =============================================================================
# OpenAICompatibleBackend tests
# =============================================================================


class TestOpenAICompatibleBackend:
    """Test OpenAICompatibleBackend configuration."""

    def test_default_values(self) -> None:
        b = OpenAICompatibleBackend()
        assert b.base_url is None
        assert b.api_key is None
        assert b.org_id is None
        assert b.timeout_ms == 0
        assert b.max_retries == 0
        assert b.extra_params is None

    def test_with_api_key_only(self) -> None:
        b = OpenAICompatibleBackend(api_key="sk-test")
        assert b.api_key == "sk-test"
        assert b.base_url is None

    def test_with_custom_base_url(self) -> None:
        b = OpenAICompatibleBackend(base_url="http://localhost:8000/v1")
        assert b.base_url == "http://localhost:8000/v1"
        assert b.api_key is None

    def test_full_configuration(self) -> None:
        b = OpenAICompatibleBackend(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
            org_id="org-123",
            timeout_ms=30000,
            max_retries=3,
        )
        assert b.base_url == "https://api.example.com/v1"
        assert b.api_key == "sk-test"
        assert b.org_id == "org-123"
        assert b.timeout_ms == 30000
        assert b.max_retries == 3

    def test_with_extra_params(self) -> None:
        """Test extra_params for provider-specific parameters."""
        b = OpenAICompatibleBackend(
            base_url="https://api.together.xyz/v1",
            api_key="sk-test",
            extra_params={"repetition_penalty": 1.1, "top_k": 50},
        )
        assert b.extra_params == {"repetition_penalty": 1.1, "top_k": 50}

    def test_is_frozen(self) -> None:
        b = OpenAICompatibleBackend(api_key="sk-test")
        with pytest.raises(AttributeError):
            b.api_key = "other"  # type: ignore[misc]


# =============================================================================
# BackendType enum tests
# =============================================================================


class TestBackendType:
    """Test BackendType enum."""

    def test_unspecified_value(self) -> None:
        assert BackendType.UNSPECIFIED == -1

    def test_local_value(self) -> None:
        assert BackendType.LOCAL == 0

    def test_openai_compatible_value(self) -> None:
        assert BackendType.OPENAI_COMPATIBLE == 1


# =============================================================================
# Ref prefix normalization tests
# =============================================================================


class TestRefPrefixNormalization:
    """Test model reference prefix normalization."""

    def test_openai_colon_normalization(self) -> None:
        handle = spec.normalize_to_handle("openai::gpt-4")
        try:
            view = spec.get_view(handle)
            assert view.backend_type_raw == int(BackendType.OPENAI_COMPATIBLE)
            ref = ctypes.string_at(view.ref).decode("utf-8")
            assert ref == "openai://gpt-4"
        finally:
            get_spec_lib().talu_config_free(handle)

    def test_openai_scheme_preserved(self) -> None:
        handle = spec.normalize_to_handle("openai://gpt-4o")
        try:
            view = spec.get_view(handle)
            assert view.backend_type_raw == int(BackendType.OPENAI_COMPATIBLE)
            ref = ctypes.string_at(view.ref).decode("utf-8")
            assert ref == "openai://gpt-4o"
        finally:
            get_spec_lib().talu_config_free(handle)


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and validation."""

    def test_missing_local_path_returns_model_not_found(self, tmp_path) -> None:
        """Missing local paths (like /tmp/.../missing-model) return ModelNotFound.

        Local paths are detected as local scheme via repository scheme parsing,
        so a missing path returns ModelNotFound rather than AmbiguousBackend.
        """
        from talu.exceptions import ModelNotFoundError

        missing = tmp_path / "missing-model"
        with pytest.raises(ModelNotFoundError):
            spec.normalize_to_handle(str(missing))

    def test_model_not_found_with_local_backend(self, tmp_path) -> None:
        from talu.exceptions import ModelNotFoundError

        missing = tmp_path / "missing-model"
        model_spec = ModelSpec(ref=str(missing), backend=LocalBackend())
        with pytest.raises(ModelNotFoundError):
            spec.normalize_to_handle(model_spec)


# =============================================================================
# ModelSpec to handle conversion tests
# =============================================================================


class TestModelSpecToHandle:
    """Test converting ModelSpec to canonical handle."""

    def test_string_converts_to_handle(self) -> None:
        # openai:// scheme triggers OpenAI backend detection
        handle = spec.normalize_to_handle("openai://gpt-4")
        try:
            assert handle is not None
        finally:
            get_spec_lib().talu_config_free(handle)

    def test_model_spec_with_openai_backend_converts(self) -> None:
        model_spec = ModelSpec(
            ref="gpt-4o",
            backend=OpenAICompatibleBackend(api_key="sk-test"),
        )
        handle = spec.normalize_to_handle(model_spec)
        try:
            view = spec.get_view(handle)
            assert view.backend_type_raw == int(BackendType.OPENAI_COMPATIBLE)
        finally:
            get_spec_lib().talu_config_free(handle)

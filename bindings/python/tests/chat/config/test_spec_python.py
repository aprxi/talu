import ctypes

import pytest

from talu import router as spec
from talu.router import (
    BackendType,
    LocalBackend,
    ModelSpec,
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
# BackendType enum tests
# =============================================================================


class TestBackendType:
    """Test BackendType enum."""

    def test_unspecified_value(self) -> None:
        assert BackendType.UNSPECIFIED == -1

    def test_local_value(self) -> None:
        assert BackendType.LOCAL == 0


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

    def test_existing_local_path_converts_to_handle(self, tmp_path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        handle = spec.normalize_to_handle(str(model_dir))
        try:
            assert handle is not None
            view = spec.get_view(handle)
            assert view.backend_type_raw == int(BackendType.LOCAL)
        finally:
            get_spec_lib().talu_config_free(handle)

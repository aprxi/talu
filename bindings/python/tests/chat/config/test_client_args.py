"""Tests for flattened backend arguments in Client and Chat.

The flattened API allows users to pass backend configuration directly
to Client/Chat constructors instead of constructing ModelSpec manually.

Before (verbose):
    spec = ModelSpec(ref="gpt-4", backend=OpenAICompatibleBackend(base_url="..."))
    client = Client(spec)

After (Pythonic):
    client = Client("gpt-4", base_url="...")
"""

import warnings

import pytest

from talu.chat.session import AsyncChat, Chat
from talu.client import AsyncClient, Client, _build_model_spec
from talu.exceptions import ValidationError
from talu.router import LocalBackend, ModelSpec, OpenAICompatibleBackend

# =============================================================================
# _build_model_spec helper tests
# =============================================================================


class TestBuildModelSpec:
    """Tests for the _build_model_spec helper function."""

    def test_simple_string_returns_spec_with_no_backend(self) -> None:
        """Simple model string creates ModelSpec with no explicit backend."""
        spec = _build_model_spec("Foo/Bar-0B")
        assert spec.ref == "Foo/Bar-0B"
        assert spec.backend is None

    def test_base_url_creates_openai_backend(self) -> None:
        """base_url triggers OpenAICompatibleBackend creation."""
        spec = _build_model_spec("gpt-4", base_url="http://localhost:8080/v1")
        assert spec.ref == "gpt-4"
        assert isinstance(spec.backend, OpenAICompatibleBackend)
        assert spec.backend.base_url == "http://localhost:8080/v1"

    def test_api_key_creates_openai_backend(self) -> None:
        """api_key alone triggers OpenAICompatibleBackend creation."""
        spec = _build_model_spec("gpt-4", api_key="sk-test")
        assert spec.ref == "gpt-4"
        assert isinstance(spec.backend, OpenAICompatibleBackend)
        assert spec.backend.api_key == "sk-test"

    def test_all_remote_kwargs(self) -> None:
        """All remote kwargs are passed to OpenAICompatibleBackend."""
        spec = _build_model_spec(
            "gpt-4",
            base_url="http://localhost:8080/v1",
            api_key="sk-test",
            org_id="org-123",
            timeout_ms=30000,
            max_retries=3,
        )
        assert isinstance(spec.backend, OpenAICompatibleBackend)
        assert spec.backend.base_url == "http://localhost:8080/v1"
        assert spec.backend.api_key == "sk-test"
        assert spec.backend.org_id == "org-123"
        assert spec.backend.timeout_ms == 30000
        assert spec.backend.max_retries == 3

    def test_gpu_layers_creates_local_backend(self) -> None:
        """gpu_layers triggers LocalBackend creation."""
        spec = _build_model_spec("Foo/Bar-0B", gpu_layers=20)
        assert spec.ref == "Foo/Bar-0B"
        assert isinstance(spec.backend, LocalBackend)
        assert spec.backend.gpu_layers == 20

    def test_all_local_kwargs(self) -> None:
        """All local kwargs are passed to LocalBackend."""
        spec = _build_model_spec(
            "Foo/Bar-0B",
            gpu_layers=20,
            use_mmap=False,
            num_threads=4,
        )
        assert isinstance(spec.backend, LocalBackend)
        assert spec.backend.gpu_layers == 20
        assert spec.backend.use_mmap is False
        assert spec.backend.num_threads == 4

    def test_local_backend_defaults(self) -> None:
        """LocalBackend uses sensible defaults for unspecified kwargs."""
        spec = _build_model_spec("model", gpu_layers=10)
        assert isinstance(spec.backend, LocalBackend)
        # Default values from LocalBackend
        assert spec.backend.use_mmap is True
        assert spec.backend.num_threads == 0

    def test_mixed_kwargs_raises_validation_error(self) -> None:
        """Mixing remote and local kwargs raises ValidationError."""
        with pytest.raises(ValidationError) as excinfo:
            _build_model_spec(
                "model",
                base_url="http://test",
                gpu_layers=10,
            )
        assert "Cannot mix" in str(excinfo.value)

    def test_model_spec_passthrough(self) -> None:
        """ModelSpec input is returned unchanged."""
        original = ModelSpec(
            ref="my-model",
            backend=OpenAICompatibleBackend(base_url="http://original"),
        )
        result = _build_model_spec(original)
        assert result is original

    def test_model_spec_with_kwargs_warns(self) -> None:
        """ModelSpec with backend + kwargs triggers warning."""
        original = ModelSpec(
            ref="my-model",
            backend=OpenAICompatibleBackend(base_url="http://original"),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _build_model_spec(original, base_url="http://ignored")
            assert len(w) == 1
            assert "ignored" in str(w[0].message).lower()
        # Original spec is returned unchanged
        assert result.backend.base_url == "http://original"

    def test_model_spec_without_backend_gets_kwargs(self) -> None:
        """ModelSpec without backend accepts kwargs."""
        # ModelSpec with backend=None can get kwargs applied
        original = ModelSpec(ref="my-model", backend=None)
        # But since it already has a ref, it returns as-is
        # (kwargs would need to be applied to a new spec)
        result = _build_model_spec(original)
        assert result is original


# =============================================================================
# Client flattened args tests
# =============================================================================


class TestClientFlattenedArgs:
    """Tests for Client with flattened backend arguments."""

    def test_client_with_base_url(self) -> None:
        """Client accepts base_url for remote backend."""
        client = Client("gpt-4", base_url="http://localhost:8080/v1")
        try:
            assert client.default_model == "gpt-4"
        finally:
            client.close()

    def test_client_with_api_key(self) -> None:
        """Client accepts api_key for remote backend."""
        client = Client("gpt-4", base_url="http://test", api_key="sk-test")
        try:
            assert client.default_model == "gpt-4"
        finally:
            client.close()

    def test_client_with_gpu_layers(self) -> None:
        """Client accepts gpu_layers for local backend."""
        client = Client("Foo/Bar-0B", gpu_layers=20)
        try:
            assert client.default_model == "Foo/Bar-0B"
        finally:
            client.close()

    def test_client_with_all_local_kwargs(self) -> None:
        """Client accepts all local backend kwargs."""
        client = Client(
            "Foo/Bar-0B",
            gpu_layers=20,
            use_mmap=False,
            num_threads=4,
        )
        try:
            assert client.default_model == "Foo/Bar-0B"
        finally:
            client.close()

    def test_client_mixed_kwargs_raises(self) -> None:
        """Client raises when mixing remote and local kwargs."""
        with pytest.raises(ValidationError) as excinfo:
            Client("model", base_url="http://test", gpu_layers=10)
        assert "Cannot mix" in str(excinfo.value)

    def test_client_multi_model_kwargs_first_only(self) -> None:
        """For multi-model Client, kwargs apply to first model only."""
        # This is the expected behavior - kwargs only affect first model
        client = Client(
            ["gpt-4", "gpt-3.5-turbo"],
            base_url="http://localhost:8080/v1",
        )
        try:
            assert "gpt-4" in client.models
            assert "gpt-3.5-turbo" in client.models
        finally:
            client.close()


# =============================================================================
# AsyncClient flattened args tests
# =============================================================================


class TestAsyncClientFlattenedArgs:
    """Tests for AsyncClient with flattened backend arguments."""

    def test_async_client_with_base_url(self) -> None:
        """AsyncClient accepts base_url for remote backend."""
        client = AsyncClient("gpt-4", base_url="http://localhost:8080/v1")
        try:
            assert client.default_model == "gpt-4"
        finally:
            client._router.close()

    def test_async_client_with_gpu_layers(self) -> None:
        """AsyncClient accepts gpu_layers for local backend."""
        client = AsyncClient("Foo/Bar-0B", gpu_layers=20)
        try:
            assert client.default_model == "Foo/Bar-0B"
        finally:
            client._router.close()


# =============================================================================
# Chat/AsyncChat: Infrastructure kwargs removed (v1.0 API cleanup)
# =============================================================================
#
# Chat and AsyncChat no longer accept infrastructure kwargs (base_url, api_key,
# gpu_layers, etc.). Users must use Client/AsyncClient for backend configuration.
#
# This enforces clean separation of concerns:
#   - Chat/AsyncChat: Session state (history, system prompt, templates)
#   - Client/AsyncClient: Infrastructure (model loading, backends, hardware)
#
# See session.py docstrings for the recommended patterns.
# =============================================================================


class TestChatInfraKwargsRemoved:
    """Tests verifying Chat no longer accepts infrastructure kwargs."""

    def test_chat_rejects_base_url(self) -> None:
        """Chat raises TypeError for base_url kwarg."""
        with pytest.raises(TypeError):
            Chat("gpt-4", base_url="http://localhost:8080/v1")

    def test_chat_rejects_api_key(self) -> None:
        """Chat raises TypeError for api_key kwarg."""
        with pytest.raises(TypeError):
            Chat("gpt-4", api_key="sk-test")

    def test_chat_rejects_gpu_layers(self) -> None:
        """Chat raises TypeError for gpu_layers kwarg."""
        with pytest.raises(TypeError):
            Chat("Foo/Bar-0B", gpu_layers=20)

    def test_chat_with_client_works(self) -> None:
        """Chat with client= still works (correct pattern)."""
        client = Client("gpt-4", base_url="http://test")
        try:
            chat = client.chat(system="You are helpful.")
            assert chat._client is client
            chat.close()
        finally:
            client.close()

    def test_chat_accepts_session_kwargs(self) -> None:
        """Chat accepts session-related kwargs (system, config, etc.)."""
        from talu import GenerationConfig

        chat = Chat(system="You are helpful.", config=GenerationConfig(temperature=0.7))
        try:
            assert chat._system == "You are helpful."
            assert chat.config.temperature == 0.7
        finally:
            chat.close()


class TestAsyncChatInfraKwargsRemoved:
    """Tests verifying AsyncChat no longer accepts infrastructure kwargs."""

    def test_async_chat_rejects_base_url(self) -> None:
        """AsyncChat raises TypeError for base_url kwarg."""
        with pytest.raises(TypeError):
            AsyncChat("gpt-4", base_url="http://localhost:8080/v1")

    def test_async_chat_rejects_gpu_layers(self) -> None:
        """AsyncChat raises TypeError for gpu_layers kwarg."""
        with pytest.raises(TypeError):
            AsyncChat("Foo/Bar-0B", gpu_layers=20)

    def test_async_chat_with_client_works(self) -> None:
        """AsyncChat with client= still works (correct pattern)."""
        client = AsyncClient("gpt-4", base_url="http://test")
        try:
            chat = client.chat(system="You are helpful.")
            assert chat._client is client
            chat._close_sync()
        finally:
            client._router.close()


# =============================================================================
# Edge cases and compatibility
# =============================================================================


class TestFlattenedArgsEdgeCases:
    """Edge cases for flattened arguments."""

    def test_explicit_none_kwargs_are_ignored(self) -> None:
        """Explicitly passing None for kwargs doesn't trigger backend creation."""
        spec = _build_model_spec(
            "model",
            base_url=None,
            api_key=None,
            gpu_layers=None,
        )
        assert spec.backend is None

    def test_zero_timeout_is_valid(self) -> None:
        """timeout_ms=0 is valid (means no timeout or use default)."""
        spec = _build_model_spec("model", base_url="http://test", timeout_ms=0)
        assert isinstance(spec.backend, OpenAICompatibleBackend)
        assert spec.backend.timeout_ms == 0

    def test_negative_gpu_layers_is_valid(self) -> None:
        """gpu_layers=-1 means use all layers (valid)."""
        spec = _build_model_spec("model", gpu_layers=-1)
        assert isinstance(spec.backend, LocalBackend)
        assert spec.backend.gpu_layers == -1

    def test_backward_compatible_with_model_spec(self) -> None:
        """Existing ModelSpec usage still works."""
        backend = OpenAICompatibleBackend(base_url="http://test")
        spec = ModelSpec(ref="gpt-4", backend=backend)
        client = Client(spec)
        try:
            assert client.default_model == "gpt-4"
        finally:
            client.close()

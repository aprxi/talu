"""Integration tests for OpenAI-compatible remote inference backends.

These tests verify that talu can connect to and use OpenAI-compatible endpoints
like vLLM, Ollama, or other compatible servers.

The tests require a running server at http://localhost:8000 (configurable via
TALU_VLLM_ENDPOINT environment variable). Tests are skipped if the endpoint
is not available.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass

import pytest

# Import talu spec types
from talu.router import ModelSpec, OpenAICompatibleBackend

# =============================================================================
# Configuration
# =============================================================================
from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM

VLLM_ENDPOINT = os.environ.get("TALU_VLLM_ENDPOINT", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("TALU_VLLM_API_KEY", None)
EXPECTED_MODEL = os.environ.get("TALU_VLLM_MODEL", TEST_MODEL_URI_TEXT_RANDOM)


# =============================================================================
# Fixtures
# =============================================================================


def _endpoint_available() -> bool:
    """Check if the vLLM endpoint is available."""
    try:
        # Parse host and port from endpoint
        url = VLLM_ENDPOINT.replace("http://", "").replace("https://", "")
        if "/" in url:
            url = url.split("/")[0]
        if ":" in url:
            host, port_str = url.split(":")
            port = int(port_str)
        else:
            host = url
            port = 80

        # Try to connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def vllm_available() -> bool:
    """Check if vLLM endpoint is available."""
    return _endpoint_available()


@pytest.fixture(scope="module")
def skip_if_no_vllm(vllm_available: bool) -> None:
    """Skip test if vLLM is not available."""
    if not vllm_available:
        pytest.skip(f"vLLM endpoint not available at {VLLM_ENDPOINT}")


# =============================================================================
# Model Types
# =============================================================================


@dataclass
class ModelInfo:
    """Information about a model from the /v1/models endpoint."""

    id: str
    object: str
    created: int | None
    owned_by: str


# =============================================================================
# Helper Functions (Pure Python HTTP, no native code)
# =============================================================================


def list_models_http(base_url: str, api_key: str | None = None) -> list[ModelInfo]:
    """List available models using Python HTTP (no native code).

    This uses urllib to avoid dependencies on requests/httpx.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to {url}: {e}") from e

    models = []
    for item in data.get("data", []):
        models.append(
            ModelInfo(
                id=item.get("id", ""),
                object=item.get("object", "model"),
                created=item.get("created"),
                owned_by=item.get("owned_by", ""),
            )
        )
    return models


def chat_completion_http(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str | None = None,
    max_tokens: int = 50,
    temperature: float = 0.7,
) -> str:
    """Send a chat completion request using Python HTTP.

    Returns the generated text content.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to {url}: {e}") from e

    choices = result.get("choices", [])
    if not choices:
        raise ValueError("No choices in response")

    message = choices[0].get("message", {})
    return message.get("content", "")


# =============================================================================
# Tests: Model Listing
# =============================================================================


class TestListModels:
    """Tests for listing models from OpenAI-compatible endpoints."""

    def test_list_models_returns_list(self, skip_if_no_vllm: None) -> None:
        """Test that list_models returns a list of models."""
        models = list_models_http(VLLM_ENDPOINT, VLLM_API_KEY)
        assert isinstance(models, list)
        assert len(models) > 0, "Expected at least one model"

    def test_list_models_has_expected_model(self, skip_if_no_vllm: None) -> None:
        """Test that the expected model is available."""
        models = list_models_http(VLLM_ENDPOINT, VLLM_API_KEY)
        model_ids = [m.id for m in models]

        assert EXPECTED_MODEL in model_ids, (
            f"Expected model '{EXPECTED_MODEL}' not found. Available models: {model_ids}"
        )

    def test_model_info_has_required_fields(self, skip_if_no_vllm: None) -> None:
        """Test that model info contains required fields."""
        models = list_models_http(VLLM_ENDPOINT, VLLM_API_KEY)
        assert len(models) > 0

        model = models[0]
        assert model.id, "Model ID should not be empty"
        assert model.object == "model", f"Expected object='model', got '{model.object}'"


# =============================================================================
# Tests: Chat Completions
# =============================================================================


class TestChatCompletions:
    """Tests for chat completions via OpenAI-compatible endpoints."""

    def test_simple_completion(self, skip_if_no_vllm: None) -> None:
        """Test a simple chat completion request."""
        messages = [{"role": "user", "content": "Say 'hello world' and nothing else."}]

        response = chat_completion_http(
            VLLM_ENDPOINT,
            EXPECTED_MODEL,
            messages,
            api_key=VLLM_API_KEY,
            max_tokens=20,
            temperature=0.0,  # Deterministic
        )

        assert response, "Response should not be empty"
        assert "hello" in response.lower(), f"Expected 'hello' in response: {response}"

    def test_multi_turn_conversation(self, skip_if_no_vllm: None) -> None:
        """Test a multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be brief."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        response = chat_completion_http(
            VLLM_ENDPOINT,
            EXPECTED_MODEL,
            messages,
            api_key=VLLM_API_KEY,
            max_tokens=50,
            temperature=0.0,
        )

        assert response, "Response should not be empty"
        # The response should mention "4" somewhere
        assert "4" in response, f"Expected '4' in response: {response}"

    def test_system_message_respected(self, skip_if_no_vllm: None) -> None:
        """Test that system messages are respected."""
        messages = [
            {
                "role": "system",
                "content": "You must respond with exactly the word 'BANANA' and nothing else.",
            },
            {"role": "user", "content": "What should you say?"},
        ]

        response = chat_completion_http(
            VLLM_ENDPOINT,
            EXPECTED_MODEL,
            messages,
            api_key=VLLM_API_KEY,
            max_tokens=20,
            temperature=0.0,
        )

        assert "banana" in response.lower(), f"Expected 'banana' in response: {response}"


# =============================================================================
# Tests: Talu ModelSpec Integration
# =============================================================================


class TestTaluModelSpec:
    """Tests for talu ModelSpec with OpenAI-compatible backends."""

    def test_model_spec_creation(self) -> None:
        """Test creating a ModelSpec with OpenAI backend."""
        backend = OpenAICompatibleBackend(
            base_url=VLLM_ENDPOINT,
            api_key=VLLM_API_KEY,
            timeout_ms=30000,
            max_retries=2,
        )
        spec = ModelSpec(ref=EXPECTED_MODEL, backend=backend)

        assert spec.ref == EXPECTED_MODEL
        assert spec.backend is not None
        assert isinstance(spec.backend, OpenAICompatibleBackend)
        assert spec.backend.base_url == VLLM_ENDPOINT

    def test_model_spec_canonicalize(self, skip_if_no_vllm: None) -> None:
        """Test canonicalizing a ModelSpec with OpenAI backend."""
        from talu.router import get_capabilities, normalize_to_handle
        from talu.router._bindings import get_spec_lib

        backend = OpenAICompatibleBackend(
            base_url=VLLM_ENDPOINT,
            api_key=VLLM_API_KEY or "",
            timeout_ms=30000,
            max_retries=2,
        )
        spec = ModelSpec(ref=EXPECTED_MODEL, backend=backend)

        # This should succeed even without connecting
        handle = normalize_to_handle(spec)
        assert handle is not None

        # Capabilities should indicate streaming support
        caps = get_capabilities(spec)
        assert caps.streaming is True

        # Clean up
        get_spec_lib().talu_config_free(handle)


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in remote backends."""

    def test_invalid_model_returns_error(self, skip_if_no_vllm: None) -> None:
        """Test that requesting an invalid model returns an error."""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ConnectionError):  # URLError -> ConnectionError
            chat_completion_http(
                VLLM_ENDPOINT,
                "nonexistent-model-12345",
                messages,
                api_key=VLLM_API_KEY,
                max_tokens=10,
            )

    def test_connection_refused_error(self) -> None:
        """Test that connection refused is handled gracefully."""
        # Use a port that's unlikely to have a server
        with pytest.raises(ConnectionError):
            list_models_http("http://localhost:59999")

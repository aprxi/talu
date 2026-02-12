"""Integration tests for llama.cpp server OpenAI-compatible backend.

These tests verify that talu can connect to and use llama.cpp server's
OpenAI-compatible API endpoint.

The tests require a running llama.cpp server at http://localhost:8080
(configurable via TALU_LLAMACPP_ENDPOINT environment variable).
Tests are skipped if the endpoint is not available.

Start a llama.cpp server with:
    llama-server -m model.gguf --port 8080

Or with the older name:
    ./server -m model.gguf --port 8080
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass

import pytest

from talu.router import ModelSpec, OpenAICompatibleBackend

# =============================================================================
# Configuration
# =============================================================================

LLAMACPP_ENDPOINT = os.environ.get("TALU_LLAMACPP_ENDPOINT", "http://localhost:8080")
LLAMACPP_API_KEY = os.environ.get("TALU_LLAMACPP_API_KEY", None)
# llama.cpp server typically doesn't have a model name, or uses the filename
# Set this to the model name your server reports, or leave as None to auto-detect
EXPECTED_MODEL = os.environ.get("TALU_LLAMACPP_MODEL", None)


# =============================================================================
# Fixtures
# =============================================================================


def _endpoint_available() -> bool:
    """Check if the llama.cpp endpoint is available."""
    try:
        url = LLAMACPP_ENDPOINT.replace("http://", "").replace("https://", "")
        if "/" in url:
            url = url.split("/")[0]
        if ":" in url:
            host, port_str = url.split(":")
            port = int(port_str)
        else:
            host = url
            port = 80

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def llamacpp_available() -> bool:
    """Check if llama.cpp endpoint is available."""
    return _endpoint_available()


@pytest.fixture(scope="module")
def skip_if_no_llamacpp(llamacpp_available: bool) -> None:
    """Skip test if llama.cpp is not available."""
    if not llamacpp_available:
        pytest.skip(f"llama.cpp endpoint not available at {LLAMACPP_ENDPOINT}")


@pytest.fixture(scope="module")
def model_id(skip_if_no_llamacpp: None) -> str:
    """Get the model ID from the server, or use configured value."""
    if EXPECTED_MODEL:
        return EXPECTED_MODEL

    # Auto-detect from /v1/models
    models = list_models_http(LLAMACPP_ENDPOINT, LLAMACPP_API_KEY)
    if not models:
        pytest.skip("No models available on llama.cpp server")
    return models[0].id


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
# Helper Functions
# =============================================================================


def list_models_http(base_url: str, api_key: str | None = None) -> list[ModelInfo]:
    """List available models using Python HTTP."""
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
    """Send a chat completion request using Python HTTP."""
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


def get_health(base_url: str) -> dict:
    """Check llama.cpp server health endpoint."""
    url = f"{base_url.rstrip('/')}/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to {url}: {e}") from e


# =============================================================================
# Tests: Server Health
# =============================================================================


class TestLlamaCppHealth:
    """Tests for llama.cpp server health and availability."""

    def test_health_endpoint(self, skip_if_no_llamacpp: None) -> None:
        """Test that the health endpoint responds."""
        health = get_health(LLAMACPP_ENDPOINT)
        # llama.cpp returns {"status": "ok"} or similar
        assert "status" in health, f"Expected 'status' in health response: {health}"

    def test_models_endpoint_available(self, skip_if_no_llamacpp: None) -> None:
        """Test that /v1/models endpoint is available."""
        models = list_models_http(LLAMACPP_ENDPOINT, LLAMACPP_API_KEY)
        assert isinstance(models, list)
        # llama.cpp should have at least one model loaded
        assert len(models) > 0, "Expected at least one model"


# =============================================================================
# Tests: Model Listing
# =============================================================================


class TestLlamaCppListModels:
    """Tests for listing models from llama.cpp server."""

    def test_list_models_returns_list(self, skip_if_no_llamacpp: None) -> None:
        """Test that list_models returns a list of models."""
        models = list_models_http(LLAMACPP_ENDPOINT, LLAMACPP_API_KEY)
        assert isinstance(models, list)
        assert len(models) > 0, "Expected at least one model"

    def test_model_info_has_id(self, skip_if_no_llamacpp: None) -> None:
        """Test that model info contains an ID."""
        models = list_models_http(LLAMACPP_ENDPOINT, LLAMACPP_API_KEY)
        assert len(models) > 0

        model = models[0]
        assert model.id, "Model ID should not be empty"

    def test_talu_remote_utilities(self, skip_if_no_llamacpp: None) -> None:
        """Test talu.chat.remote utilities with llama.cpp."""
        from talu.router import check_endpoint, get_model_ids, list_endpoint_models

        # check_endpoint should return True
        assert check_endpoint(LLAMACPP_ENDPOINT) is True

        # list_endpoint_models should return RemoteModelInfo objects
        models = list_endpoint_models(LLAMACPP_ENDPOINT)
        assert len(models) > 0
        assert hasattr(models[0], "id")

        # get_model_ids should return strings
        ids = get_model_ids(LLAMACPP_ENDPOINT)
        assert len(ids) > 0
        assert isinstance(ids[0], str)


# =============================================================================
# Tests: Chat Completions
# =============================================================================


class TestLlamaCppChatCompletions:
    """Tests for chat completions via llama.cpp server."""

    def test_simple_completion(self, skip_if_no_llamacpp: None, model_id: str) -> None:
        """Test a simple chat completion request."""
        messages = [{"role": "user", "content": "Say 'hello' and nothing else."}]

        response = chat_completion_http(
            LLAMACPP_ENDPOINT,
            model_id,
            messages,
            api_key=LLAMACPP_API_KEY,
            max_tokens=20,
            temperature=0.0,
        )

        assert response, "Response should not be empty"
        assert "hello" in response.lower(), f"Expected 'hello' in response: {response}"

    def test_math_question(self, skip_if_no_llamacpp: None, model_id: str) -> None:
        """Test a simple math question."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
            {"role": "user", "content": "What is 2 + 2? Just give the number."},
        ]

        response = chat_completion_http(
            LLAMACPP_ENDPOINT,
            model_id,
            messages,
            api_key=LLAMACPP_API_KEY,
            max_tokens=30,
            temperature=0.0,
        )

        assert response, "Response should not be empty"
        assert "4" in response, f"Expected '4' in response: {response}"

    def test_multi_turn(self, skip_if_no_llamacpp: None, model_id: str) -> None:
        """Test multi-turn conversation sends all messages correctly."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Remember this number: 42"},
            {"role": "assistant", "content": "I'll remember that number: 42."},
            {"role": "user", "content": "Good. Now tell me a short joke."},
        ]

        response = chat_completion_http(
            LLAMACPP_ENDPOINT,
            model_id,
            messages,
            api_key=LLAMACPP_API_KEY,
            max_tokens=50,
            temperature=0.7,
        )

        # Just verify we get a response - multi-turn memory is model-dependent
        assert response, "Response should not be empty"
        assert len(response) > 5, f"Expected substantive response: {response}"


# =============================================================================
# Tests: Talu ModelSpec Integration
# =============================================================================


class TestLlamaCppModelSpec:
    """Tests for talu ModelSpec with llama.cpp backend."""

    def test_model_spec_creation(self, model_id: str) -> None:
        """Test creating a ModelSpec with llama.cpp backend."""
        backend = OpenAICompatibleBackend(
            base_url=LLAMACPP_ENDPOINT,
            api_key=LLAMACPP_API_KEY,
            timeout_ms=30000,
            max_retries=2,
        )
        spec = ModelSpec(ref=model_id, backend=backend)

        assert spec.ref == model_id
        assert spec.backend is not None
        assert isinstance(spec.backend, OpenAICompatibleBackend)
        assert spec.backend.base_url == LLAMACPP_ENDPOINT

    def test_capabilities(self, skip_if_no_llamacpp: None, model_id: str) -> None:
        """Test getting capabilities for llama.cpp backend."""
        from talu.router import get_capabilities

        backend = OpenAICompatibleBackend(
            base_url=LLAMACPP_ENDPOINT,
            api_key=LLAMACPP_API_KEY or "",
            timeout_ms=30000,
            max_retries=2,
        )
        spec = ModelSpec(ref=model_id, backend=backend)

        caps = get_capabilities(spec)
        assert caps.streaming is True  # llama.cpp supports streaming


# =============================================================================
# Tests: Streaming (if supported)
# =============================================================================


class TestLlamaCppStreaming:
    """Tests for streaming responses from llama.cpp server."""

    def test_streaming_completion(self, skip_if_no_llamacpp: None, model_id: str) -> None:
        """Test streaming chat completion."""
        url = f"{LLAMACPP_ENDPOINT.rstrip('/')}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if LLAMACPP_API_KEY:
            headers["Authorization"] = f"Bearer {LLAMACPP_API_KEY}"

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Count from 1 to 3."}],
            "max_tokens": 30,
            "temperature": 0.0,
            "stream": True,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        chunks = []
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    chunks.append(content)
                        except json.JSONDecodeError:
                            continue
        except urllib.error.URLError as e:
            pytest.skip(f"Streaming not available: {e}")

        full_response = "".join(chunks)
        assert full_response, "Expected streaming response content"
        # Should contain numbers 1, 2, 3
        assert any(c in full_response for c in ["1", "2", "3"]), (
            f"Expected numbers in response: {full_response}"
        )

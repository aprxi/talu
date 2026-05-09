"""Contract tests for future-facing Python remote routing API.

Talu supports OpenAI-compatible inbound HTTP through the Rust CLI server.
Python keeps the outbound remote-routing API shape for future support, but
generation through that route is not implemented yet and must fail with a
router-owned error instead of leaking core canonicalization failures.
"""

import pytest

from talu.client import AsyncClient, Client
from talu.exceptions import GenerationError
from talu.router import ModelSpec, OpenAICompatibleBackend, Router
from talu.router.spec import normalize_to_handle


class _ChatStub:
    _chat_ptr = object()


def test_client_accepts_remote_backend_kwargs_for_future_routing() -> None:
    """The public API remains constructible for future Python routing support."""
    client = Client("gpt-4", base_url="http://localhost:8080/v1")
    try:
        assert client.default_model == "gpt-4"
    finally:
        client.close()


def test_async_client_accepts_remote_backend_kwargs_for_future_routing() -> None:
    """AsyncClient preserves the same constructible API shape as Client."""
    client = AsyncClient("gpt-4", api_key="sk-test")
    try:
        assert client.default_model == "gpt-4"
    finally:
        client._router.close()


def test_router_accepts_openai_compatible_backend_at_construction() -> None:
    """Router can store future remote-routing targets."""
    spec = ModelSpec(
        ref="gpt-4",
        backend=OpenAICompatibleBackend(base_url="http://localhost:8080/v1"),
    )

    router = Router([spec])
    try:
        assert router.default_model == "gpt-4"
    finally:
        router.close()


def test_router_generate_remote_backend_fails_before_core_canonicalization() -> None:
    """Using a future remote target should fail clearly at the router boundary."""
    spec = ModelSpec(
        ref="gpt-4",
        backend=OpenAICompatibleBackend(base_url="http://localhost:8080/v1"),
    )
    router = Router([spec])
    try:
        with pytest.raises(GenerationError, match="OpenAI-compatible|remote|not implemented"):
            router.generate(_ChatStub(), "hello")
    finally:
        router.close()


def test_router_stream_remote_backend_fails_before_core_canonicalization() -> None:
    """Streaming should follow the same remote-route failure path."""
    spec = ModelSpec(
        ref="gpt-4",
        backend=OpenAICompatibleBackend(base_url="http://localhost:8080/v1"),
    )
    router = Router([spec])
    try:
        with pytest.raises(GenerationError, match="OpenAI-compatible|remote|not implemented"):
            list(router.stream(_ChatStub(), "hello"))
    finally:
        router.close()


def test_client_capabilities_remote_backend_fails_before_core_canonicalization() -> None:
    """Capability lookup should not treat remote targets as local core specs."""
    client = Client("gpt-4", base_url="http://localhost:8080/v1")
    try:
        with pytest.raises(GenerationError, match="OpenAI-compatible|remote|not implemented"):
            client.capabilities()
    finally:
        client.close()


def test_normalize_to_handle_still_reflects_core_local_only_boundary() -> None:
    """Direct CAPI normalization remains local-only and is not the remote route."""
    spec = ModelSpec(
        ref="gpt-4",
        backend=OpenAICompatibleBackend(base_url="http://localhost:8080/v1"),
    )

    with pytest.raises(Exception, match="canonicalize failed"):
        normalize_to_handle(spec)

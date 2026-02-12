"""Model specification types and API.

This module provides the unified model specification system for configuring
models across different backends (local inference, OpenAI-compatible APIs, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol

from ..exceptions import ValidationError
from . import _bindings as _c

if TYPE_CHECKING:
    pass


# =============================================================================
# User-Facing Types
# =============================================================================


class BackendType(IntEnum):
    """Backend type enumeration for model inference."""

    UNSPECIFIED = -1
    LOCAL = 0
    OPENAI_COMPATIBLE = 1


class BackendSpec(Protocol):
    """Protocol for backend configuration specifications."""

    pass


@dataclass(frozen=True)
class LocalBackend:
    """Configuration for local inference backend."""

    gpu_layers: int = -1
    use_mmap: bool = True
    num_threads: int = 0


@dataclass(frozen=True)
class OpenAICompatibleBackend:
    """Configuration for OpenAI-compatible API backend.

    Attributes
    ----------
        base_url: Base URL for the API (e.g., "http://localhost:8080/v1").
        api_key: API key for authentication.
        org_id: Organization ID for multi-tenant APIs.
        timeout_ms: Request timeout in milliseconds (0 = default).
        max_retries: Maximum retry attempts for failed requests (0 = default).
        extra_params: Additional parameters to include in API requests.
            These are merged into the request body for OpenAI-compatible APIs.
            Useful for provider-specific parameters not covered by GenerationConfig.

            Example::

                backend = OpenAICompatibleBackend(
                    base_url="https://api.together.xyz/v1",
                    api_key="...",
                    extra_params={"repetition_penalty": 1.1}
                )

        headers: Custom HTTP headers to include in every request to this backend.
            Useful for enterprise networking requirements like authentication proxies,
            custom routing, or tracing headers.

            Example::

                backend = OpenAICompatibleBackend(
                    base_url="https://internal-proxy.corp.example.com/llm/v1",
                    api_key="...",
                    headers={
                        "X-Request-ID": "abc123",
                        "X-Team-ID": "ml-team",
                        "X-Proxy-Auth": "secret-token",
                    }
                )
    """

    base_url: str | None = None
    api_key: str | None = None
    org_id: str | None = None
    timeout_ms: int = 0
    max_retries: int = 0
    extra_params: dict[str, Any] | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model with optional backend configuration."""

    ref: str
    backend: BackendSpec | None = None


@dataclass(frozen=True)
class Capabilities:
    """Capability flags for a model backend."""

    streaming: bool
    tool_calling: bool
    logprobs: bool
    embeddings: bool
    json_schema: bool


# =============================================================================
# Internal Helpers
# =============================================================================


def _normalize_ref_prefix(ref: str, allow_mapping: bool) -> str:
    """Normalize model reference prefixes (e.g., 'native::', 'openai::' -> 'openai://')."""
    if ref.startswith("native::"):
        return ref[len("native::") :]
    if not allow_mapping:
        return ref
    if "::" in ref:
        prefix, rest = ref.split("::", 1)
        if prefix in {"openai", "oaic"}:
            return f"{prefix}://{rest}"
    return ref


def _normalize_model_input(
    model_input: str | ModelSpec,
    backend_override: BackendSpec | None,
) -> ModelSpec:
    if isinstance(model_input, ModelSpec):
        spec = model_input
    elif isinstance(model_input, str):
        spec = ModelSpec(ref=model_input, backend=None)
    else:
        raise ValidationError("model_input must be str or ModelSpec")

    backend = backend_override if backend_override is not None else spec.backend
    allow_mapping = backend is None
    ref = _normalize_ref_prefix(spec.ref, allow_mapping)
    return ModelSpec(ref=ref, backend=backend)


def _backend_type(backend: BackendSpec | None) -> BackendType:
    if backend is None:
        return BackendType.UNSPECIFIED
    if isinstance(backend, LocalBackend):
        return BackendType.LOCAL
    if isinstance(backend, OpenAICompatibleBackend):
        return BackendType.OPENAI_COMPATIBLE
    raise ValidationError("Unsupported backend type")


def _build_spec_args(
    model_spec: ModelSpec,
) -> tuple[
    str,
    int,
    tuple[int, bool, int] | None,
    tuple[str | None, str | None, str | None, int, int, str | None] | None,
]:
    """Convert ModelSpec to arguments for build_c_spec in _bindings."""
    backend_type_raw = int(_backend_type(model_spec.backend))

    local_config = None
    openai_config = None

    if isinstance(model_spec.backend, LocalBackend):
        local_config = (
            int(model_spec.backend.gpu_layers),
            model_spec.backend.use_mmap,
            int(model_spec.backend.num_threads),
        )
    elif isinstance(model_spec.backend, OpenAICompatibleBackend):
        backend = model_spec.backend
        headers_json = json.dumps(backend.headers) if backend.headers is not None else None
        openai_config = (
            backend.base_url,
            backend.api_key,
            backend.org_id,
            int(backend.timeout_ms),
            int(backend.max_retries),
            headers_json,
        )

    return model_spec.ref, backend_type_raw, local_config, openai_config


# =============================================================================
# Public API
# =============================================================================


def normalize_to_handle(
    model_input: str | ModelSpec,
    backend_override: BackendSpec | None = None,
) -> _c.TaluCanonicalSpecHandle:
    """
    Normalize model input to a canonical specification handle.

    Args:
        model_input: Model identifier (str or ModelSpec).
        backend_override: Optional backend configuration to override.

    Returns
    -------
        Canonical specification handle for the model.

    Raises
    ------
        ValidationError: If model_input format is invalid.
        TaluError: If canonicalization fails.
    """
    spec = _normalize_model_input(model_input, backend_override)
    ref, backend_type_raw, local_config, openai_config = _build_spec_args(spec)
    c_spec, _buffers = _c.build_c_spec(ref, backend_type_raw, local_config, openai_config)
    return _c.config_canonicalize(c_spec, spec.ref)


def get_view(handle: _c.TaluCanonicalSpecHandle) -> _c.CTaluModelSpec:
    """
    Retrieve model specification view from handle.

    Args:
        handle: Canonical specification handle.

    Returns
    -------
        Model specification view structure.

    Raises
    ------
        TaluError: If view retrieval fails.
    """
    return _c.config_get_view(handle)


def get_capabilities(
    model_input: str | ModelSpec,
    backend_override: BackendSpec | None = None,
) -> Capabilities:
    """
    Retrieve capability flags for a model backend.

    Args:
        model_input: Model identifier (str or ModelSpec).
        backend_override: Optional backend configuration to override.

    Returns
    -------
        Capabilities object with backend feature flags.

    Raises
    ------
        ValidationError: If model_input format is invalid.
        TaluError: If capability retrieval fails.
    """
    handle = normalize_to_handle(model_input, backend_override)
    try:
        view = get_view(handle)
        streaming, tool_calling, logprobs, embeddings, json_schema = _c.backend_get_capabilities(
            view
        )
        return Capabilities(
            streaming=streaming,
            tool_calling=tool_calling,
            logprobs=logprobs,
            embeddings=embeddings,
            json_schema=json_schema,
        )
    finally:
        _c.config_free(handle)


__all__ = [
    # Types
    "BackendType",
    "BackendSpec",
    "LocalBackend",
    "OpenAICompatibleBackend",
    "ModelSpec",
    "Capabilities",
    # API
    "normalize_to_handle",
    "get_view",
    "get_capabilities",
]

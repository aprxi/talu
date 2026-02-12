"""Remote inference utilities for OpenAI-compatible endpoints.

This module provides utilities for working with remote OpenAI-compatible
inference servers like vLLM, Ollama, llama.cpp server, etc.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..exceptions import IOError, ValidationError


@dataclass(frozen=True)
class RemoteModelInfo:
    """Information about a model from a remote endpoint.

    Attributes
    ----------
        id: Model identifier (e.g., "Qwen/Qwen3-4B-Instruct-2507").
        object: Object type (usually "model").
        created: Unix timestamp when the model was created (optional).
        owned_by: Owner/organization of the model.
    """

    id: str
    object: str = "model"
    created: int | None = None
    owned_by: str = ""


def list_endpoint_models(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[RemoteModelInfo]:
    """List available models from an OpenAI-compatible endpoint.

    This function queries the /v1/models endpoint to discover what models
    are available on a remote server.

    Args:
        base_url: Base URL of the server (e.g., "http://localhost:8000").
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.

    Returns
    -------
        List of RemoteModelInfo objects describing available models.

    Raises
    ------
        IOError: If the server is not reachable.
        ValidationError: If the response is invalid.

    Example:
        >>> from talu.router import list_endpoint_models
        >>> models = list_endpoint_models("http://localhost:8000")
        >>> for m in models:
        ...     print(m.id)
        Qwen/Qwen3-4B-Instruct-2507
    """
    # Normalize URL
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        if "/v1" not in base_url:
            base_url = f"{base_url}/v1"

    url = f"{base_url}/models"

    # Build headers
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Make request
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise IOError(f"Failed to connect to {url}: {e}", code="CONNECTION_ERROR") from e
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON response from {url}: {e}") from e

    # Parse response
    if not isinstance(data, dict):
        raise ValidationError(f"Expected dict response, got {type(data)}")

    models: list[RemoteModelInfo] = []
    for item in data.get("data", []):
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if model_id is None:
            continue
        models.append(
            RemoteModelInfo(
                id=str(model_id),
                object=str(item.get("object", "model")),
                created=item.get("created"),
                owned_by=str(item.get("owned_by", "")),
            )
        )

    return models


def check_endpoint(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 5.0,
) -> bool:
    """Check if an OpenAI-compatible endpoint is available.

    This is a lightweight check that attempts to connect to the /v1/models
    endpoint to verify the server is running.

    Args:
        base_url: Base URL of the server (e.g., "http://localhost:8000").
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.

    Returns
    -------
        True if the endpoint is available, False otherwise.

    Example:
        >>> from talu.router import check_endpoint
        >>> if check_endpoint("http://localhost:8000"):
        ...     print("vLLM is running")
    """
    try:
        list_endpoint_models(base_url, api_key, timeout)
        return True
    except (IOError, ValidationError, TimeoutError):
        return False


def get_model_ids(
    base_url: str,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Get just the model IDs from a remote endpoint.

    Convenience function that returns only the model ID strings.

    Args:
        base_url: Base URL of the server (e.g., "http://localhost:8000").
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.

    Returns
    -------
        List of model ID strings.

    Example:
        >>> from talu.router import get_model_ids
        >>> ids = get_model_ids("http://localhost:8000")
        >>> "Qwen/Qwen3-4B-Instruct-2507" in ids
        True
    """
    models = list_endpoint_models(base_url, api_key, timeout)
    return [m.id for m in models]


__all__ = [
    "RemoteModelInfo",
    "list_endpoint_models",
    "check_endpoint",
    "get_model_ids",
]

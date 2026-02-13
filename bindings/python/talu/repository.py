"""
Model repository management.

This module provides operations for managing model repositories:
- Listing cached models and files
- Fetching models from remote sources (HuggingFace Hub, S3 in future)
- Deleting models from cache
- Searching remote sources

The ``list_files()`` function accepts both:

- **Model IDs** (e.g., ``"Qwen/Qwen3-0.6B"``) - Queries remote hub or local cache
- **Local paths** (e.g., ``"./models/my-model"``) - Lists files in that directory

Both are treated as "repositories" - one remote, one local.

Listing:
- list_models() - List cached model IDs
- list_files() - List files in a model repository (local or remote)

Cache Operations:
- is_cached() - Check if model is in local cache
- cache_path() - Get local path for cached model
- delete() - Remove from local cache
- size() - Get cache size

Remote Operations:
- fetch() - Retrieve from remote to local cache
- fetch_file() - Retrieve a single file from remote
- search() - Search for models on remote source

Example
-------
>>> from talu.repository import list_models, fetch
>>> models = list(list_models())  # Returns cached model IDs
>>> path = fetch("Qwen/Qwen3-0.6B")  # Downloads if not cached
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable, Iterator

from ._bindings import get_lib
from ._native import CProgressCallback, DownloadOptions
from .exceptions import IOError

__all__ = [
    "list_models",
    "list_files",
    "resolve_path",
    "cache_path",
    "is_cached",
    "size",
    "delete",
    "clear",
    "fetch",
    "fetch_file",
    "search",
    "is_model_id",
    "cache_dir",
]


# =============================================================================
# FFI Bindings
# =============================================================================

_lib = get_lib()


def _call_repo_resolve_path(
    uri: bytes, offline: bool, token: bytes | None, endpoint_url: bytes | None
) -> tuple[int, str | None]:
    """Resolve model URI to local path."""
    out_ptr = ctypes.c_void_p()
    code = _lib.talu_repo_resolve_path(uri, offline, token, endpoint_url, True, ctypes.byref(out_ptr))
    if code != 0 or not out_ptr.value:
        return (code, None)

    path_bytes = ctypes.cast(out_ptr.value, ctypes.c_char_p).value
    if path_bytes is None:
        _lib.talu_text_free(out_ptr)
        return (code, None)
    path = path_bytes.decode("utf-8")
    _lib.talu_text_free(out_ptr)
    return (0, path)


def _call_repo_list_models(require_weights: bool) -> Iterator[tuple[str, str, str]]:
    """List cached models with source info."""
    out_list = ctypes.c_void_p()
    code = _lib.talu_repo_list_models(require_weights, ctypes.byref(out_list))
    if code != 0 or not out_list.value:
        return

    _source_names = {0: "hub", 1: "managed"}

    try:
        count = _lib.talu_repo_list_count(out_list.value)
        for i in range(count):
            out_id = ctypes.c_void_p()
            out_path = ctypes.c_void_p()
            code = _lib.talu_repo_list_get_id(out_list.value, i, ctypes.byref(out_id))
            if code != 0 or not out_id.value:
                continue
            model_id_bytes = ctypes.cast(out_id.value, ctypes.c_char_p).value
            if not model_id_bytes:
                continue

            _lib.talu_repo_list_get_path(out_list.value, i, ctypes.byref(out_path))
            path_bytes = (
                ctypes.cast(out_path.value, ctypes.c_char_p).value if out_path.value else b""
            )

            source_val = _lib.talu_repo_list_get_source(out_list.value, i)
            source = _source_names.get(source_val, "hub")

            yield (
                model_id_bytes.decode("utf-8"),
                (path_bytes or b"").decode("utf-8"),
                source,
            )
    finally:
        _lib.talu_repo_list_free(out_list.value)


def _call_repo_list_files(model_path: bytes, token: bytes | None) -> Iterator[str]:
    """List files in a model repository."""
    out_list = ctypes.c_void_p()
    code = _lib.talu_repo_list(model_path, token, ctypes.byref(out_list))

    if code != 0 or not out_list.value:
        return

    try:
        count = _lib.talu_repo_string_list_count(out_list.value)
        for i in range(count):
            out_item = ctypes.c_void_p()
            item_code = _lib.talu_repo_string_list_get(out_list.value, i, ctypes.byref(out_item))
            if item_code == 0 and out_item.value:
                filename = ctypes.cast(out_item.value, ctypes.c_char_p).value
                if filename:
                    yield filename.decode("utf-8")
    finally:
        _lib.talu_repo_string_list_free(out_list.value)


def _call_repo_get_cached_path(model_id: bytes) -> str | None:
    """Get local cache path for a model."""
    out_ptr = ctypes.c_void_p()
    code = _lib.talu_repo_get_cached_path(model_id, True, ctypes.byref(out_ptr))
    if code != 0 or not out_ptr.value:
        return None
    path_bytes = ctypes.cast(out_ptr.value, ctypes.c_char_p).value
    if path_bytes is None:
        return None
    path = path_bytes.decode("utf-8")
    _lib.talu_text_free(out_ptr)
    return path


def _call_repo_is_cached(model_id: bytes) -> bool:
    """Check if model is in local cache."""
    return _lib.talu_repo_is_cached(model_id) == 1


def _call_repo_size(model_id: bytes | None) -> int:
    """Get cache size for a specific model or total cache size."""
    if model_id:
        return _lib.talu_repo_size(model_id)
    return _lib.talu_repo_total_size()


def _call_repo_delete(model_id: bytes) -> bool:
    """Delete a model from local cache."""
    return _lib.talu_repo_delete(model_id) == 1


def _call_repo_fetch(
    model_id: bytes,
    force: bool,
    token: bytes | None,
    endpoint_url: bytes | None,
    on_progress: Callable[[int, int, str], None] | None,
) -> str | None:
    """Fetch model from remote to local cache."""
    # Store callback reference to prevent garbage collection
    callback_ref = None

    if on_progress:

        @CProgressCallback
        def progress_cb(update_ptr, user_data):
            if update_ptr:
                update = update_ptr.contents
                filename = update.label.decode("utf-8") if update.label else ""
                on_progress(update.current, update.total, filename)

        callback_ref = progress_cb

    options = DownloadOptions()
    options.token = token
    options.force = force
    options.user_data = None
    options.endpoint_url = endpoint_url
    options.progress_callback = ctypes.cast(callback_ref, ctypes.c_void_p) if callback_ref else None

    out_path = ctypes.c_void_p()
    code = _lib.talu_repo_fetch(model_id, ctypes.byref(options), ctypes.byref(out_path))

    if code != 0 or not out_path.value:
        return None

    path_bytes = ctypes.cast(out_path.value, ctypes.c_char_p).value
    if path_bytes is None:
        return None
    path = path_bytes.decode("utf-8")
    _lib.talu_text_free(out_path)
    return path


def _call_repo_fetch_file(
    model_id: bytes,
    filename: bytes,
    force: bool,
    token: bytes | None,
    endpoint_url: bytes | None,
) -> tuple[int, str | None]:
    """Fetch a single file from a model repository."""
    options = DownloadOptions()
    options.token = token
    options.force = force
    options.user_data = None
    options.endpoint_url = endpoint_url
    options.progress_callback = None

    out_ptr = ctypes.c_void_p()
    code = _lib.talu_repo_fetch_file(
        model_id, filename, ctypes.byref(options), ctypes.byref(out_ptr)
    )

    if code != 0 or not out_ptr.value:
        return (code, None)

    path_bytes = ctypes.cast(out_ptr.value, ctypes.c_char_p).value
    if path_bytes is None:
        _lib.talu_text_free(out_ptr)
        return (code, None)
    path = path_bytes.decode("utf-8")
    _lib.talu_text_free(out_ptr)
    return (0, path)


def _call_repo_search(
    query: bytes, limit: int, token: bytes | None, endpoint_url: bytes | None
) -> Iterator[str]:
    """Search for models on remote source."""
    out_list = ctypes.c_void_p()
    code = _lib.talu_repo_search(query, limit, token, endpoint_url, ctypes.byref(out_list))
    if code != 0 or not out_list.value:
        return

    try:
        count = _lib.talu_repo_string_list_count(out_list.value)
        for i in range(count):
            out_item = ctypes.c_void_p()
            item_code = _lib.talu_repo_string_list_get(out_list.value, i, ctypes.byref(out_item))
            if item_code == 0 and out_item.value:
                model_id = ctypes.cast(out_item.value, ctypes.c_char_p).value
                if model_id:
                    yield model_id.decode("utf-8")
    finally:
        _lib.talu_repo_string_list_free(out_list.value)


def _call_repo_is_model_id(path: bytes) -> bool:
    """Check if string looks like a model ID."""
    return _lib.talu_repo_is_model_id(path) == 1


def _call_repo_get_hf_home() -> str | None:
    """Get HuggingFace home directory."""
    out_ptr = ctypes.c_void_p()
    code = _lib.talu_repo_get_hf_home(ctypes.byref(out_ptr))
    if code != 0 or not out_ptr.value:
        return None
    path_bytes = ctypes.cast(out_ptr.value, ctypes.c_char_p).value
    if path_bytes is None:
        return None
    path = path_bytes.decode("utf-8")
    _lib.talu_text_free(out_ptr)
    return path


# =============================================================================
# Public API (module-level functions)
# =============================================================================


def _get_token(token: str | None = None) -> str | None:
    """Get API token from parameter or HF_TOKEN environment variable."""
    return token or os.environ.get("HF_TOKEN")


def list_models() -> Iterator[str]:
    """
    List all cached model IDs (both Talu-local and HuggingFace).

    Talu-local models are yielded first, then HuggingFace cached models.

    Yields
    ------
    str
        Model IDs in the local cache (e.g., "Qwen/Qwen3-0.6B").

    Example
    -------
    >>> from talu.repository import list_models, size
    >>> models = list(list_models())
    >>> for model_id in models:
    ...     size_mb = size(model_id) / 1e6
    ...     print(f"{model_id}: {size_mb:.1f} MB")  # doctest: +SKIP
    """
    for model_id, _path, _source in _call_repo_list_models(True):
        yield model_id


def list_files(ref: str, token: str | None = None) -> Iterator[str]:
    """
    List files in a model repository.

    Parameters
    ----------
    ref : str
        Repository identifier. Can be:

        - **Model ID string**: ``"Qwen/Qwen3-0.6B"`` - Lists files from remote
          hub or local cache.
        - **Local path**: ``"./models/my-model"`` or ``"/abs/path/to/model"`` -
          Lists files in that directory.

    token : str, optional
        API token for private models. Falls back to ``HF_TOKEN`` env var.

    Yields
    ------
    str
        Filenames in the model repository.

    Example
    -------
    >>> from talu.repository import list_files
    >>> files = list(list_files("Qwen/Qwen3-0.6B"))  # doctest: +SKIP
    >>> "config.json" in files  # doctest: +SKIP
    True
    """
    tok = _get_token(token)
    tok_bytes = tok.encode("utf-8") if tok else None
    yield from _call_repo_list_files(ref.encode("utf-8"), tok_bytes)


def resolve_path(
    uri: str,
    *,
    offline: bool = False,
    token: str | None = None,
    endpoint_url: str | None = None,
) -> str:
    """
    Resolve a model URI to a local filesystem path.

    Parameters
    ----------
    uri : str
        Model URI or identifier (e.g., "Qwen/Qwen3-0.6B").
    offline : bool
        If True, do not use network; requires cached/local availability.
    token : str, optional
        API token for private models. Falls back to ``HF_TOKEN`` env var.
    endpoint_url : str, optional
        Custom HuggingFace endpoint URL (overrides HF_ENDPOINT env var).

    Returns
    -------
    str
        Resolved local filesystem path.

    Raises
    ------
    IOError
        If the path cannot be resolved.

    Example
    -------
    >>> from talu.repository import resolve_path
    >>> path = resolve_path("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    """
    tok = _get_token(token)
    tok_bytes = tok.encode("utf-8") if tok else None
    endpoint_bytes = endpoint_url.encode("utf-8") if endpoint_url else None

    code, path = _call_repo_resolve_path(uri.encode("utf-8"), offline, tok_bytes, endpoint_bytes)
    if code != 0 or path is None:
        raise IOError(f"Failed to resolve model path for '{uri}'")

    return path


def cache_path(model_id: str) -> str | None:
    """
    Get local cache path for a model.

    This is a cache lookup only - does NOT fetch missing models.

    Parameters
    ----------
    model_id : str
        Model ID (e.g., "Qwen/Qwen3-0.6B").

    Returns
    -------
    str or None
        Path to cached model directory, or None if not cached.

    Example
    -------
    >>> from talu.repository import cache_path
    >>> path = cache_path("Qwen/Qwen3-0.6B")
    >>> if path:
    ...     print("Model is cached")  # doctest: +SKIP
    """
    return _call_repo_get_cached_path(model_id.encode("utf-8"))


def is_cached(model_id: str) -> bool:
    """
    Check if a model is in local cache.

    This is an explicit cache-only check. No network requests are made.

    Parameters
    ----------
    model_id : str
        Model ID.

    Returns
    -------
    bool
        True if model is cached locally with valid weights.

    Example
    -------
    >>> from talu.repository import is_cached, cache_path
    >>> if is_cached("Qwen/Qwen3-0.6B"):
    ...     path = cache_path("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    """
    return _call_repo_is_cached(model_id.encode("utf-8"))


def size(model_id: str | None = None) -> int:
    """
    Get size of cached models in bytes.

    Parameters
    ----------
    model_id : str, optional
        Specific model to check. If None, returns total cache size.

    Returns
    -------
    int
        Size in bytes.

    Example
    -------
    >>> from talu.repository import size
    >>> total = size()  # Total cache size
    >>> model_size = size("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    """
    return _call_repo_size(model_id.encode("utf-8") if model_id else None)


def delete(model_id: str) -> bool:
    """
    Delete a model from local cache.

    Parameters
    ----------
    model_id : str
        Model ID.

    Returns
    -------
    bool
        True if model was deleted, False if not cached.

    Example
    -------
    >>> from talu.repository import delete
    >>> deleted = delete("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    """
    return _call_repo_delete(model_id.encode("utf-8"))


def clear() -> int:
    """
    Delete all models from local cache.

    Returns
    -------
    int
        Number of models deleted.

    Example
    -------
    >>> from talu.repository import clear
    >>> count = clear()  # Deletes all cached models  # doctest: +SKIP
    """
    count = 0
    for model_id in list(list_models()):
        if delete(model_id):
            count += 1
    return count


def fetch(
    model_id: str,
    *,
    force: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    token: str | None = None,
    endpoint_url: str | None = None,
) -> str | None:
    """
    Fetch a model from remote to local cache.

    Downloads the model if not already cached (or if force=True).

    Parameters
    ----------
    model_id : str
        Model ID (e.g., "Qwen/Qwen3-0.6B").
    force : bool, optional
        Force re-fetch even if cached. Default False.
    on_progress : callable, optional
        Progress callback: fn(downloaded_bytes, total_bytes, filename).
    token : str, optional
        API token for private models. Falls back to ``HF_TOKEN`` env var.
    endpoint_url : str, optional
        Custom HuggingFace endpoint URL (overrides HF_ENDPOINT env var).

    Returns
    -------
    str or None
        Path to fetched model, or None on error.

    Example
    -------
    >>> from talu.repository import fetch
    >>> path = fetch("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    """
    tok = _get_token(token)
    tok_bytes = tok.encode("utf-8") if tok else None
    endpoint_bytes = endpoint_url.encode("utf-8") if endpoint_url else None

    return _call_repo_fetch(
        model_id.encode("utf-8"), force, tok_bytes, endpoint_bytes, on_progress
    )


def fetch_file(
    model_id: str,
    filename: str,
    *,
    force: bool = False,
    token: str | None = None,
    endpoint_url: str | None = None,
) -> str | None:
    """
    Fetch a single file from a model repository.

    Downloads one file (e.g., ``"config.json"``) without fetching
    the full model weights.

    Parameters
    ----------
    model_id : str
        Model ID (e.g., ``"Qwen/Qwen3-0.6B"``).
    filename : str
        Name of file to fetch (e.g., ``"config.json"``).
    force : bool, optional
        Force re-download even if cached. Default ``False``.
    token : str, optional
        API token for private models. Falls back to ``HF_TOKEN`` env var.
    endpoint_url : str, optional
        Custom HuggingFace endpoint URL.

    Returns
    -------
    str or None
        Path to fetched file, or ``None`` on error.

    Example
    -------
    >>> from talu.repository import fetch_file
    >>> path = fetch_file("Qwen/Qwen3-0.6B", "config.json")  # doctest: +SKIP
    """
    tok = _get_token(token)
    tok_bytes = tok.encode("utf-8") if tok else None
    endpoint_bytes = endpoint_url.encode("utf-8") if endpoint_url else None

    code, path = _call_repo_fetch_file(
        model_id.encode("utf-8"),
        filename.encode("utf-8"),
        force,
        tok_bytes,
        endpoint_bytes,
    )
    return path


def search(
    query: str,
    limit: int = 10,
    token: str | None = None,
    endpoint_url: str | None = None,
) -> Iterator[str]:
    """
    Search for models on the remote source.

    Searches for text-generation models matching the query.

    Parameters
    ----------
    query : str
        Search query (e.g., "qwen", "llama").
    limit : int, optional
        Maximum number of results. Default 10.
    token : str, optional
        API token. Falls back to ``HF_TOKEN`` env var.
    endpoint_url : str, optional
        Custom HuggingFace endpoint URL (overrides HF_ENDPOINT env var).

    Yields
    ------
    str
        Model IDs matching the search query.

    Example
    -------
    >>> from talu.repository import search
    >>> results = list(search("qwen", limit=5))  # doctest: +SKIP
    """
    tok = _get_token(token)
    tok_bytes = tok.encode("utf-8") if tok else None
    endpoint_bytes = endpoint_url.encode("utf-8") if endpoint_url else None

    yield from _call_repo_search(query.encode("utf-8"), limit, tok_bytes, endpoint_bytes)


def is_model_id(path: str) -> bool:
    """
    Check if a string looks like a model ID.

    Parameters
    ----------
    path : str
        String to check.

    Returns
    -------
    bool
        True if it looks like "org/model" format.

    Example
    -------
    >>> from talu.repository import is_model_id
    >>> is_model_id("Qwen/Qwen3-0.6B")
    True
    >>> is_model_id("/path/to/model")
    False
    """
    return _call_repo_is_model_id(path.encode("utf-8"))


def cache_dir() -> str:
    """
    Get the cache directory path.

    Returns
    -------
    str
        Path to the hub cache (e.g., ~/.cache/huggingface/hub).

    Raises
    ------
    IOError
        If the cache home directory cannot be determined.
    """
    path = _call_repo_get_hf_home()
    if path is None:
        raise IOError(
            "Could not determine cache home directory",
            code="IO_READ_FAILED",
        )
    return path + "/hub"

"""
Template loading utilities.

Provides functions for loading chat templates from model directories.
"""

from __future__ import annotations

from ..exceptions import TemplateNotFoundError
from . import _bindings as _c


def get_chat_template_source(model_path: str) -> str:
    """
    Get the raw chat template source from a model directory.

    Args:
        model_path: Path to model directory containing tokenizer_config.json

    Returns
    -------
        The chat template source string.

    Raises
    ------
        TemplateNotFoundError: If model has no chat template.
    """
    source = _c.get_chat_template_source(model_path)

    if source is None:
        raise TemplateNotFoundError(
            f"No chat template found for model '{model_path}'. "
            "Ensure the model has a chat_template in tokenizer_config.json "
            "or a chat_template.jinja file."
        )

    return source


def resolve_model_path(model: str) -> str:
    """
    Resolve model identifier to a local path.

    Handles both local paths and HuggingFace model IDs.

    Args:
        model: Local path or HuggingFace model ID (e.g., "Qwen/Qwen3-0.6B")

    Returns
    -------
        Resolved local path to the model directory.

    Raises
    ------
        FileNotFoundError: If model cannot be found or resolved.
    """
    try:
        from ..repository import resolve_path

        return resolve_path(model, offline=True)
    except (ImportError, OSError):
        pass

    raise FileNotFoundError(
        f"Model '{model}' not found. Provide a local path or a cached "
        "HuggingFace model ID. Download models with: "
        f"talu get {model}"
    )

"""Router module - Model routing, backend specification, and generation config.

This module provides the compute layer for LLM inference:

- Router: Routes generation requests to model backends
- ModelSpec: Unified model specification system
- GenerationConfig: Configuration for text generation
- Grammar: Pre-compiled grammar for structured output

Architecture::

    talu.types   (bottom — pure data, imports nothing from talu)
        ↑
    talu.router  (middle — imports talu.types)
        ↑
    talu.chat    (top — imports talu.router + talu.types)
"""

from ._bindings import (
    SamplingParams as SamplingParams,
)
from ._bindings import (
    SamplingStrategy as SamplingStrategy,
)
from ._bindings import (
    StopFlag,
)
from .config import CompletionOptions, GenerationConfig, Grammar, SchemaStrategy
from .remote import RemoteModelInfo, check_endpoint, get_model_ids, list_endpoint_models
from .router import ModelTarget, Router, StreamToken
from .spec import (
    BackendSpec,
    Capabilities,
    LocalBackend,
    ModelSpec,
    OpenAICompatibleBackend,
)
from .spec import (
    BackendType as BackendType,
)
from .spec import (
    get_capabilities as get_capabilities,
)
from .spec import (
    get_view as get_view,
)
from .spec import (
    normalize_to_handle as normalize_to_handle,
)

__all__ = [
    # Router
    "Router",
    "ModelTarget",
    "StreamToken",
    # Generation Config
    "GenerationConfig",
    "SchemaStrategy",
    "Grammar",
    "CompletionOptions",
    # Cancellation
    "StopFlag",
    # Model Specification
    "BackendSpec",
    "LocalBackend",
    "OpenAICompatibleBackend",
    "ModelSpec",
    "Capabilities",
    # Remote Endpoint Utilities
    "RemoteModelInfo",
    "list_endpoint_models",
    "check_endpoint",
    "get_model_ids",
]

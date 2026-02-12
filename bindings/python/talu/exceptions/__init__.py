"""
Talu exceptions.

This module defines the exception hierarchy for talu:

    TaluError (base)
    ├── GenerationError - Errors during text generation
    │   └── EmptyPromptError - Empty prompt with no BOS token
    ├── ModelError - Errors loading or using models
    │   └── ModelNotFoundError - Model path doesn't exist
    ├── TokenizerError - Errors during tokenization
    ├── TemplateError - Errors during template rendering
    │   ├── TemplateSyntaxError - Invalid template syntax
    │   ├── TemplateUndefinedError - Undefined variable in strict mode
    │   └── TemplateNotFoundError - Template file not found
    ├── ConvertError - Errors during model conversion
    ├── IOError - I/O and network errors
    ├── ResourceError - System resource exhaustion (memory pressure)
    ├── InteropError - DLPack/NumPy interop errors
    ├── StateError - Invalid object state errors
    ├── StorageError - Storage backend failures (items in memory only)
    └── ValidationError - Invalid parameter value
"""

from .exceptions import (
    ConvertError,
    EmptyPromptError,
    GenerationError,
    GrammarError,
    IncompleteJSONError,
    InteropError,
    IOError,
    ModelError,
    ModelNotFoundError,
    ResourceError,
    SchemaValidationError,
    StateError,
    StorageError,
    StreamingValidationError,
    StructuredOutputError,
    TaluError,
    TemplateError,
    TemplateNotFoundError,
    TemplateSyntaxError,
    TemplateUndefinedError,
    TokenizerError,
    ValidationError,
)

# =============================================================================
# Public API - See talu/__init__.py for documentation mapping guidelines
# =============================================================================
__all__ = [
    # Base
    "TaluError",
    # Generation
    "GenerationError",
    "EmptyPromptError",
    # Model
    "ModelError",
    "ModelNotFoundError",
    # Tokenizer
    "TokenizerError",
    # Template
    "TemplateError",
    "TemplateSyntaxError",
    "TemplateUndefinedError",
    "TemplateNotFoundError",
    # Converter
    "ConvertError",
    # I/O
    "IOError",
    # Resource
    "ResourceError",
    # Interop
    "InteropError",
    # State
    "StateError",
    # Storage
    "StorageError",
    # Validation
    "ValidationError",
    # Structured Output
    "StructuredOutputError",
    "IncompleteJSONError",
    "SchemaValidationError",
    "GrammarError",
    "StreamingValidationError",
]

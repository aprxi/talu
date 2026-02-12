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
    ├── InteropError - DLPack/NumPy interop errors
    ├── StateError - Invalid object state errors
    ├── StorageError - Storage backend failures (items in memory only)
    └── ValidationError - Invalid parameter value

Usage:
    try:
        chat("")  # Empty prompt
    except talu.EmptyPromptError:
        print("Prompt cannot be empty for this model")
    except talu.GenerationError as e:
        print(f"Generation failed: {e}")
    except talu.TaluError as e:
        # Catch any talu error with structured details
        print(f"Error {e.code}: {e}")
        print(f"Details: {e.details}")

See Also
--------
    TaluError : Base exception for all talu errors.
"""

from typing import Any

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


class TaluError(Exception):
    """
    Base exception for all talu errors.

    All talu-specific exceptions inherit from this class, enabling:
    - Catch-all handling: ``except talu.TaluError``
    - Stable string-based error codes for programmatic handling
    - Structured details for debugging and logging

    Attributes
    ----------
    message : str
        Human-readable error description.
    code : str
        Stable, string-based error code (e.g., "MODEL_NOT_FOUND").
        Use this for programmatic error handling.
    details : dict[str, Any]
        Structured context (e.g., {"path": "...", "expected": [...]}}).
    original_code : int | None
        The internal Zig integer code (for debugging/logging).

    Example
    -------
    >>> try:
    ...     talu.generate("nonexistent/model", "Hello")
    ... except talu.TaluError as e:
    ...     print(f"Error code: {e.code}")
    ...     print(f"Details: {e.details}")
    Error code: MODEL_NOT_FOUND
    Details: {'path': 'nonexistent/model'}
    """

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.original_code = original_code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.args[0]!r}, code={self.code!r})"


# =============================================================================
# Generation Errors
# =============================================================================


class GenerationError(TaluError, RuntimeError):
    """
    Error during text generation.

    Raised when the generation process fails. Common causes:
    - Context overflow (prompt too long)
    - Invalid generation parameters
    - Internal engine errors
    """

    def __init__(
        self,
        message: str,
        code: str = "GENERATION_FAILED",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 300)


class EmptyPromptError(GenerationError, ValueError):
    """
    Empty prompt provided to a model without a BOS token.

    Some models (like Qwen) don't have a beginning-of-sequence (BOS) token,
    so they require at least one token in the prompt to start generation.

    Solutions:
    - Provide a non-empty prompt
    - Use chat=True to apply the chat template (adds tokens)
    - Use a model with a BOS token
    """

    def __init__(
        self,
        message: str | None = None,
        code: str = "GENERATION_EMPTY_PROMPT",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        if message is None:
            message = (
                "Empty prompt provided but model has no BOS token. "
                "Provide a non-empty prompt or use chat=True."
            )
        super().__init__(message, code, details, original_code or 301)


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(TaluError, RuntimeError):
    """
    Error loading or using a model.

    Raised when model loading fails. Common causes:
    - Invalid model format
    - Unsupported architecture
    - Missing or corrupted config/weights
    """

    def __init__(
        self,
        message: str,
        code: str = "MODEL_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 101)


class ModelNotFoundError(ModelError, FileNotFoundError):
    """
    Model path does not exist or is invalid.

    Raised when the specified model path doesn't exist or
    the HuggingFace model ID cannot be resolved.
    """

    def __init__(
        self,
        message: str,
        code: str = "MODEL_NOT_FOUND",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 100)


# =============================================================================
# Tokenizer Errors
# =============================================================================


class TokenizerError(TaluError, RuntimeError):
    """
    Error during tokenization.

    Raised when encoding or decoding fails. Common causes:
    - Tokenizer not found for model
    - Invalid tokenizer format
    - Encoding/decoding failures
    """

    def __init__(
        self,
        message: str,
        code: str = "TOKENIZER_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 200)


# =============================================================================
# Template Errors
# =============================================================================


class TemplateError(TaluError, RuntimeError):
    """
    Base error for template operations.

    This exception (or its subclasses) is raised when template rendering fails.
    """

    def __init__(
        self,
        message: str,
        code: str = "TEMPLATE_RENDER_FAILED",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 603)


class TemplateSyntaxError(TemplateError, SyntaxError):
    """
    Invalid template syntax.

    Raised when a template has malformed Jinja2 syntax, such as:
    - Unclosed tags: ``{{ name`` without ``}}``
    - Invalid expressions: ``{{ 1 + }}``
    - Unclosed blocks: ``{% if x %}`` without ``{% endif %}``
    """

    def __init__(
        self,
        message: str,
        code: str = "TEMPLATE_SYNTAX_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 600)


class TemplateUndefinedError(TemplateError, NameError):
    """
    Undefined variable accessed in strict mode.

    Raised when a template uses a variable that wasn't provided and
    ``strict=True`` was set. In non-strict mode (default), undefined
    variables silently render as empty strings.

    Solutions:
        - Provide the missing variable
        - Use ``| default(value)`` filter for optional variables
        - Use ``strict=False`` (default) if empty strings are acceptable
    """

    def __init__(
        self,
        message: str,
        code: str = "TEMPLATE_UNDEFINED_VAR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 601)


class TemplateNotFoundError(TemplateError, FileNotFoundError):
    """
    Template file not found.

    Raised when:
    - A model doesn't have a chat_template in tokenizer_config.json
    - No chat_template.jinja file exists in the model directory
    - Template.from_file() is given a non-existent path
    """

    def __init__(
        self,
        message: str,
        code: str = "TEMPLATE_NOT_FOUND",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 604)


# =============================================================================
# Converter Errors
# =============================================================================


class ConvertError(TaluError, RuntimeError):
    """
    Raised when model conversion fails.

    This exception is raised when the conversion process encounters an error,
    such as:

    - Model not found (invalid model ID or local path)
    - Network errors when downloading
    - Unsupported model architecture
    - Disk space issues
    - Invalid model files (corrupted weights, missing config)

    The exception message contains details about what went wrong.

    Examples
    --------
    >>> try:
    ...     talu.convert("nonexistent/model", bits=4)
    ... except talu.ConvertError as e:
    ...     print(f"Conversion failed: {e}")
    Conversion failed: Model not found
    """

    def __init__(
        self,
        message: str,
        code: str = "CONVERT_FAILED",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 400)


# =============================================================================
# I/O Errors
# =============================================================================


class IOError(TaluError, OSError):
    """
    I/O and network errors.

    Raised when file or network operations fail:
    - File not found
    - Permission denied
    - Read/write failures
    - Network errors (downloading from HuggingFace)
    """

    def __init__(
        self,
        message: str,
        code: str = "IO_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 500)


# =============================================================================
# Resource Errors
# =============================================================================


class ResourceError(TaluError, MemoryError):
    """
    System resource exhaustion error.

    Raised when the system cannot allocate required resources:
    - Out of memory during model loading
    - Memory mapping failures (insufficient virtual memory)
    - Too many concurrent model instances

    This error is distinct from OutOfMemory in that it may occur during
    memory-mapped file access when pages cannot be faulted in due to
    system memory pressure.

    Solutions:
    - Reduce parallel test workers (pytest -n 2 instead of -n auto)
    - Close unused model/tokenizer instances
    - Use smaller models or reduce batch sizes
    - Increase system memory or swap space

    Example:
        >>> try:
        ...     # Loading many models in parallel
        ...     tokenizers = [Tokenizer(model) for model in models]
        ... except talu.ResourceError as e:
        ...     print(f"Resource exhaustion: {e}")
        ...     print("Try reducing parallelism or closing unused instances")
    """

    def __init__(
        self,
        message: str,
        code: str = "RESOURCE_EXHAUSTED",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 905)


# =============================================================================
# Interop Errors
# =============================================================================


class InteropError(TaluError, TypeError):
    """
    DLPack/NumPy interop errors.

    Raised when tensor interchange fails or is ambiguous:
    - Ambiguous DLPack export (multiple tensors available)
    - Empty tensor export
    - Device mismatch
    - Invalid tensor state
    """

    def __init__(
        self,
        message: str,
        code: str = "INTEROP_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code)


# =============================================================================
# State Errors
# =============================================================================


class StateError(TaluError, RuntimeError):
    """
    Invalid object state error.

    Raised when an operation is attempted on an object in an invalid state:
    - Using a closed/disposed resource
    - Uninitialized object
    - Object already consumed (e.g., DLPack ownership transferred)
    """

    def __init__(
        self,
        message: str,
        code: str = "STATE_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code)


# =============================================================================
# Storage Errors
# =============================================================================


class StorageError(TaluError, RuntimeError):
    """
    Storage backend operation failed.

    Raised when a storage operation fails during fork/clone operations.
    The items may still exist in memory, but were not persisted to storage.

    This error indicates that:
    - The in-memory chat state is valid and usable
    - The storage backend failed to persist the data
    - On restart, unpersisted items will be lost

    Common causes:
    - Database full or unavailable
    - Network timeout to remote storage
    - Transaction rollback in storage backend
    - Storage backend raised an exception in on_event()

    The fork operation succeeded in memory but the storage layer failed.
    You can continue using the chat, but data is not persisted.

    Example:
        >>> try:
        ...     forked = chat.fork()
        ... except talu.StorageError as e:
        ...     # Fork succeeded in memory, but storage failed
        ...     print(f"Storage error: {e}")
        ...     # Option 1: Continue with unpersisted fork
        ...     # forked = e.details.get("chat")
        ...     # Option 2: Retry or handle failure
    """

    def __init__(
        self,
        message: str,
        code: str = "STORAGE_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 700)


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(TaluError, ValueError):
    """
    Invalid parameter value.

    Raised when a function receives an argument of the correct type
    but an inappropriate value (e.g., bits=3 when only 4, 8, 16 are valid).

    This exception inherits from both TaluError and ValueError, so both work::

        except talu.TaluError:   # catches all talu errors
        except ValueError:       # catches validation errors (Pythonic)

    Example:
        >>> talu.convert("model", bits=3)
        ValidationError: bits must be 4, 5, 6, 8, or 16, got 3
    """

    def __init__(
        self,
        message: str,
        code: str = "INVALID_ARGUMENT",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 901)


# =============================================================================
# Structured Output Errors
# =============================================================================


class StructuredOutputError(TaluError):
    """Base class for all structured output errors."""

    def __init__(
        self,
        message: str,
        code: str = "STRUCTURED_OUTPUT_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 700)


class IncompleteJSONError(StructuredOutputError):
    """
    Generation truncated before JSON was complete.

    Raised when max_tokens is reached mid-generation and the JSON
    structure is incomplete. The partial output may be recoverable.

    Attributes
    ----------
        partial_text: The incomplete JSON string
        finish_reason: Why generation stopped ("length")
    """

    def __init__(
        self,
        partial_text: str,
        finish_reason: str,
        code: str = "INCOMPLETE_JSON",
        original_code: int | None = None,
    ):
        self.partial_text = partial_text
        self.finish_reason = finish_reason
        super().__init__(
            f"JSON incomplete (finish_reason={finish_reason}): {partial_text[:100]}...",
            code,
            {"partial_text": partial_text, "finish_reason": finish_reason},
            original_code or 701,
        )


class SchemaValidationError(StructuredOutputError):
    """
    JSON is syntactically valid but fails schema hydration/validation.

    The grammar enforced structure, but semantic validators failed.
    Common causes:
    - Dataclass missing required fields
    - Type mismatches in nested fields

    This exception enables DATA SALVAGE - the user can access the valid
    JSON data that was generated, even though hydration failed.

    Attributes
    ----------
        raw_text: The generated JSON string
        partial_data: Parsed dict (always available for salvage)
        validation_error: The underlying hydration/validation error
    """

    def __init__(
        self,
        raw_text: str,
        validation_error: Exception,
        code: str = "SCHEMA_VALIDATION_FAILED",
        original_code: int | None = None,
    ):
        import json

        self.raw_text = raw_text
        try:
            self.partial_data = json.loads(raw_text)
        except (json.JSONDecodeError, TypeError):
            self.partial_data = None
        self.validation_error = validation_error
        super().__init__(
            f"Generated valid JSON but validation failed: {validation_error}\n"
            "Raw output available in exception.partial_data for salvage",
            code,
            {"raw_text": raw_text, "partial_data": self.partial_data},
            original_code or 702,
        )


class GrammarError(StructuredOutputError):
    """
    Grammar constraint violation.

    This indicates a bug in the grammar engine - should never happen
    in production. If you see this error, please report it.
    """

    def __init__(
        self,
        message: str,
        code: str = "GRAMMAR_ERROR",
        details: dict[str, Any] | None = None,
        original_code: int | None = None,
    ):
        super().__init__(message, code, details, original_code or 703)


class StreamingValidationError(StructuredOutputError):
    """
    Streaming JSON validation failed.

    Raised when ``Validator.feed(chunk, strict=True)`` encounters invalid data.
    Provides rich diagnostics including the invalid byte, its position, and
    what bytes would have been valid at that position.

    This exception enables immediate debugging without manual buffer inspection:

    - **position**: Byte offset where validation failed
    - **invalid_byte**: The byte that caused the failure
    - **expected**: Human-readable description of valid bytes at this position
    - **context**: Surrounding bytes for visual debugging

    Attributes
    ----------
    position : int
        Byte offset in the stream where validation failed.
    invalid_byte : bytes
        The single byte that was rejected.
    expected : str
        Human-readable description of what was expected (e.g., "digit, quote").
    context : str
        Surrounding bytes for visual debugging (e.g., '..."age": X...').

    Example
    -------
    >>> validator = Validator(schema)
    >>> try:
    ...     validator.feed('{"age": "five"}', strict=True)
    ... except StreamingValidationError as e:
    ...     print(f"Position: {e.position}")
    ...     print(f"Invalid: {e.invalid_byte!r}")
    ...     print(f"Expected: {e.expected}")
    Position: 8
    Invalid: b'"'
    Expected: digit, minus sign
    """

    def __init__(
        self,
        position: int,
        invalid_byte: bytes,
        expected: str,
        context: str = "",
        code: str = "STREAMING_VALIDATION_FAILED",
        original_code: int | None = None,
    ):
        self.position = position
        self.invalid_byte = invalid_byte
        self.expected = expected
        self.context = context

        # Build human-readable message
        invalid_repr = (
            repr(chr(invalid_byte[0]))
            if invalid_byte and invalid_byte[0] < 128
            else repr(invalid_byte)
        )
        message = f"Invalid byte {invalid_repr} at position {position}. Expected: {expected}"
        if context:
            message += f"\nContext: {context}"

        super().__init__(
            message,
            code,
            {
                "position": position,
                "invalid_byte": invalid_byte,
                "expected": expected,
                "context": context,
            },
            original_code or 704,
        )

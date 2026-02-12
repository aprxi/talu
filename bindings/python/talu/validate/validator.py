"""
JSON schema validator with streaming support.

This module provides a Validator class for schema-based JSON validation,
with a unique capability: streaming validation byte-by-byte as data arrives.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from .._bindings import check
from ..exceptions import StreamingValidationError
from ._bindings import (
    call_engine_advance,
    call_engine_can_accept,
    call_engine_create,
    call_engine_destroy,
    call_engine_get_position,
    call_engine_get_valid_bytes,
    call_engine_is_complete,
    call_engine_reset,
    call_engine_validate,
    call_semantic_validator_check,
    call_semantic_validator_create,
    call_semantic_validator_destroy,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


def _format_expected_bytes(valid_bytes: list[bool]) -> str:
    """
    Format a 256-element bool array into a human-readable description.

    Groups related bytes into categories like "digit", "whitespace", etc.
    """
    valid_indices = [i for i, v in enumerate(valid_bytes) if v]

    if not valid_indices:
        return "nothing (grammar complete or error state)"

    # Common byte categories
    categories = []

    # Check for digits
    digits = [i for i in valid_indices if 0x30 <= i <= 0x39]  # '0'-'9'
    if digits:
        if len(digits) == 10:
            categories.append("digit")
        else:
            categories.append(f"digit ({', '.join(chr(d) for d in digits)})")

    # Check for whitespace
    whitespace = [i for i in valid_indices if i in (0x09, 0x0A, 0x0D, 0x20)]
    if whitespace:
        categories.append("whitespace")

    # Check for specific JSON structural characters
    structural = {
        0x7B: "{",
        0x7D: "}",
        0x5B: "[",
        0x5D: "]",
        0x3A: ":",
        0x2C: ",",
        0x22: '"',
    }
    for byte_val, char in structural.items():
        if byte_val in valid_indices:
            categories.append(f"'{char}'")

    # Check for minus sign (for negative numbers)
    if 0x2D in valid_indices:
        categories.append("'-' (minus)")

    # Check for letters (for keywords like true, false, null)
    letters = [i for i in valid_indices if 0x61 <= i <= 0x7A]  # 'a'-'z'
    if letters:
        letter_chars = "".join(chr(i) for i in letters)
        categories.append(f"letter ({letter_chars})")

    # If we haven't categorized everything, show raw
    categorized = set(digits + whitespace + list(structural.keys()) + [0x2D] + letters)
    uncategorized = [i for i in valid_indices if i not in categorized]
    if uncategorized:
        for i in uncategorized:
            if 0x20 <= i < 0x7F:
                categories.append(f"'{chr(i)}'")
            else:
                categories.append(f"0x{i:02x}")

    return ", ".join(categories) if categories else f"bytes: {valid_indices[:10]}..."


class Validator:
    """
    JSON schema validator with streaming support.

    Validates JSON against a schema, with the unique ability to validate
    incrementally as data streams in. This enables early abort when
    schema violations are detected.

    Parameters
    ----------
    schema : type[BaseModel] | dict[str, Any] | str
        The schema to validate against. Can be:
        - A Pydantic model class
        - A JSON schema dictionary
        - A JSON schema string

    Examples
    --------
    Complete validation::

        from pydantic import BaseModel
        from talu.validate import Validator

        class User(BaseModel):
            name: str
            age: int

        validator = Validator(User)
        validator.validate('{"name":"Alice","age":30}')  # True
        validator.validate('{"name":123}')  # False

    Streaming validation::

        validator = Validator(User)
        validator.feed('{"name":')      # True - valid so far
        validator.feed('"Alice",')      # True
        validator.feed('"age":30}')     # True
        validator.is_complete           # True

    Early abort on violation::

        validator = Validator(User)
        validator.feed('{"name":')      # True
        validator.feed('123')           # False - abort here!

    Raises
    ------
    ValueError
        If the schema fails to compile.
    """

    def __init__(self, schema: type[BaseModel] | dict[str, Any] | str):
        # Convert to JSON string
        if isinstance(schema, str):
            schema_json = schema
        elif isinstance(schema, dict):
            schema_json = json.dumps(schema)
        else:
            # Assume Pydantic model
            schema_json = json.dumps(schema.model_json_schema())

        self._handle = call_engine_create(schema_json.encode("utf-8"))
        if not self._handle:
            raise ValueError("Failed to compile schema")

        # Create semantic validator for post-parse validation (number min/max, additionalProperties)
        self._schema_json = schema_json
        self._semantic_handle = call_semantic_validator_create(schema_json.encode("utf-8"))
        # Note: semantic validator creation is optional - if it fails, we still do grammar validation

        # Track state for error reporting
        self._last_chunk: bytes = b""
        self._last_consumed: int = 0
        self._total_fed: int = 0
        self._failed: bool = False
        self._valid_bytes_cache: list[bool] | None = None

    def close(self) -> None:
        """
        Release native validator resources.

        After calling close(), the validator cannot be used. Safe to call
        multiple times (idempotent).
        """
        if hasattr(self, "_handle") and self._handle:
            call_engine_destroy(self._handle)
            self._handle = None
        if hasattr(self, "_semantic_handle") and self._semantic_handle:
            call_semantic_validator_destroy(self._semantic_handle)
            self._semantic_handle = None

    def __enter__(self) -> Validator:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit — calls close()."""
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # =========================================================================
    # Core validation API
    # =========================================================================

    def validate(self, data: bytes | str) -> bool:
        """
        Validate complete JSON data against the schema.

        This is the simple, one-shot validation API. For streaming
        validation, use feed() instead.

        Parameters
        ----------
        data : bytes | str
            The complete JSON data to validate.

        Returns
        -------
        bool
            True if the data is valid according to the schema.

        Examples
        --------
        >>> validator = Validator({"type": "object", "properties": {"x": {"type": "integer"}}})
        >>> validator.validate('{"x": 42}')
        True
        >>> validator.validate('{"x": "not an int"}')
        False
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        # First: grammar validation (structure, types)
        result = call_engine_validate(self._handle, data)
        if result != 1:
            return False

        # Second: semantic validation (number min/max, additionalProperties)
        if self._semantic_handle:
            semantic_result, _ = call_semantic_validator_check(self._semantic_handle, data)
            # 0 = valid, 1 = violation found, negative = error
            if semantic_result == 1:
                return False

        return True

    # =========================================================================
    # Streaming validation API
    # =========================================================================

    def feed(self, chunk: bytes | str, *, strict: bool = False) -> bool:
        """
        Feed a chunk of data for streaming validation.

        Call repeatedly as data arrives. Returns False immediately
        if the chunk violates the schema, enabling early abort.

        Parameters
        ----------
        chunk : bytes | str
            A chunk of JSON data.
        strict : bool, optional
            If True, raises StreamingValidationError on failure with rich
            diagnostics (position, invalid byte, expected bytes, context).
            Default is False for backward compatibility.

        Returns
        -------
        bool
            True if the chunk was valid, False if schema was violated.

        Raises
        ------
        StreamingValidationError
            If strict=True and validation fails.

        Examples
        --------
        >>> validator = Validator(User)
        >>> validator.feed('{"name":')       # True
        >>> validator.feed('"Alice",')       # True
        >>> validator.feed('"age":30}')      # True
        >>> validator.is_complete            # True

        >>> validator.reset()
        >>> validator.feed('{"name":')       # True
        >>> validator.feed('123')            # False - number not allowed

        With strict mode for immediate error diagnostics::

            >>> validator.reset()
            >>> validator.feed('{"name":', strict=True)
            >>> validator.feed('123', strict=True)
            StreamingValidationError: Invalid byte '1' at position 8. Expected: '"' (quote)
        """
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")

        consumed = call_engine_advance(self._handle, chunk)
        valid = consumed == len(chunk)

        if not valid:
            # Track failure state for error property
            self._last_chunk = chunk
            self._last_consumed = consumed
            self._failed = True

            # Query valid bytes AFTER advance - engine state is now at the failure point
            self._valid_bytes_cache = self._get_valid_bytes()

            if strict:
                raise self.error  # type: ignore[misc]
        else:
            self._total_fed += len(chunk)

        return valid

    def reset(self) -> None:
        """
        Reset validator to initial state.

        Call this to reuse the validator for a new validation.

        Raises
        ------
        TaluError
            If the validator state cannot be reset.
        """
        check(call_engine_reset(self._handle))
        # Clear error state
        self._last_chunk = b""
        self._last_consumed = 0
        self._total_fed = 0
        self._failed = False
        self._valid_bytes_cache = None

    @property
    def is_complete(self) -> bool:
        """
        True if the data fed so far is valid AND complete.

        A complete JSON document has been fully parsed and matches
        the schema. Use this after feeding all chunks to confirm
        the document is finished.
        """
        return call_engine_is_complete(self._handle)

    @property
    def position(self) -> int:
        """
        Current byte position in the stream.

        Useful for error reporting - indicates where validation
        stopped or where an error occurred.
        """
        return call_engine_get_position(self._handle)

    def _get_valid_bytes(self) -> list[bool]:
        """
        Query the engine for valid bytes at current position.

        Returns a 256-element list where True means that byte value is valid.
        """
        return call_engine_get_valid_bytes(self._handle)

    @property
    def error(self) -> StreamingValidationError | None:
        """
        Rich error for the last validation failure.

        Returns None if validation hasn't failed. After a failed feed(),
        this property provides detailed diagnostics including:
        - Position where validation failed
        - The invalid byte
        - What bytes would have been valid
        - Context around the failure point

        Example
        -------
        >>> validator.feed('{"name": 123}')  # Returns False
        False
        >>> if validator.error:
        ...     print(validator.error.expected)
        '"' (quote)
        """
        if not self._failed:
            return None

        # Get the invalid byte
        invalid_byte = (
            self._last_chunk[self._last_consumed : self._last_consumed + 1]
            if self._last_consumed < len(self._last_chunk)
            else b""
        )

        # Get what was expected (uses cached value from feed())
        valid_bytes = self._valid_bytes_cache or self._get_valid_bytes()
        expected = _format_expected_bytes(valid_bytes)

        # Build context string around the failure point
        context = ""
        if self._last_chunk:
            start = max(0, self._last_consumed - 10)
            end = min(len(self._last_chunk), self._last_consumed + 10)
            context_bytes = self._last_chunk[start:end]
            try:
                context = context_bytes.decode("utf-8", errors="replace")
                # Mark the invalid position
                marker_pos = self._last_consumed - start
                context = context[:marker_pos] + "→" + context[marker_pos:]
            except (UnicodeDecodeError, IndexError):
                context = repr(context_bytes)

        return StreamingValidationError(
            position=self._total_fed + self._last_consumed,
            invalid_byte=invalid_byte,
            expected=expected,
            context=context,
        )

    # =========================================================================
    # Advanced: inspection API
    # =========================================================================

    def can_continue_with(self, data: bytes | str) -> bool:
        """
        Check if data could be accepted without advancing state.

        Useful for lookahead or testing multiple continuations.

        Parameters
        ----------
        data : bytes | str
            The data to test.

        Returns
        -------
        bool
            True if the data would be valid from current state.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        return call_engine_can_accept(self._handle, data)

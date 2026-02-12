"""
Pre-compiled Grammar for structured output.

This module provides the Grammar class which allows pre-compiling JSON schemas
into reusable grammar handles. This enables zero-latency structured output when
the same schema is used across multiple generations.

Example:
    >>> from dataclasses import dataclass
    >>> from talu.router import Grammar
    >>>
    >>> @dataclass
    ... class Person:
    ...     name: str
    ...     age: int
    ...
    >>> # Compile once at startup
    >>> grammar = Grammar(Person)
    >>>
    >>> # Reuse across multiple requests (no compilation overhead)
    >>> chat = talu.Chat("Qwen/Qwen3-0.6B")
    >>> response = chat.send("Who are you?", response_format=grammar)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from ..._bindings import get_last_error, get_lib
from ...exceptions import StructuredOutputError

if TYPE_CHECKING:
    pass


class Grammar:
    """Pre-compiled grammar for structured output.

    Compiling a JSON schema into a grammar can be expensive. This class allows
    you to compile once and reuse the grammar across multiple generations,
    eliminating the compilation overhead.

    The grammar validates the schema at compile time - if the schema is invalid,
    a StructuredOutputError is raised immediately, not at generation time.

    Args:
        response_format: A dataclass type or JSON schema dict defining the output structure.

    Raises
    ------
        StructuredOutputError: If the schema is invalid or compilation fails.

    Example - Basic usage:
        >>> from dataclasses import dataclass
        >>> from talu.router import Grammar, Chat
        >>>
        >>> @dataclass
        ... class Answer:
        ...     value: int
        ...     reasoning: str
        ...
        >>> # Compile grammar once
        >>> grammar = Grammar(Answer)
        >>>
        >>> # Reuse for multiple generations
        >>> chat = Chat("Qwen/Qwen3-0.6B")
        >>> r1 = chat.send("What is 2+2?", response_format=grammar)
        >>> r2 = chat.send("What is 3+3?", response_format=grammar)

    Example - Server context (compile at startup):
        >>> # At startup
        >>> ANSWER_GRAMMAR = Grammar(Answer)
        >>>
        >>> # In request handler
        >>> async def handle_question(question: str):
        ...     chat = AsyncChat("model")
        ...     response = await chat.send(question, response_format=ANSWER_GRAMMAR)
        ...     return response.parsed

    Note:
        The Grammar class maintains a handle to native compiled grammar data.
        When the Grammar instance is garbage collected, the handle is freed
        automatically. For long-lived grammars (e.g., server context), keep
        a reference to prevent premature collection.
    """

    __slots__ = ("_handle", "_schema_dict", "_response_format", "_closed")

    def __init__(self, response_format: type | dict[str, Any]) -> None:
        """Compile a response format into a reusable grammar.

        Args:
            response_format: A dataclass type or JSON schema dict.

        Raises
        ------
            StructuredOutputError: If the schema is invalid or compilation fails.
        """
        from ..schema.convert import normalize_response_format

        # Store original for .parsed hydration
        self._response_format = response_format

        # Convert to JSON schema dict
        self._schema_dict = normalize_response_format(response_format)
        if self._schema_dict is None:
            raise StructuredOutputError(
                f"Cannot convert {type(response_format).__name__} to JSON schema"
            )

        # Compile the grammar
        schema_json = json.dumps(self._schema_dict).encode("utf-8")
        lib = get_lib()
        handle = lib.talu_grammar_compile(schema_json)

        if handle is None:
            err = get_last_error()
            raise StructuredOutputError(
                f"Failed to compile grammar: {err}" if err else "Failed to compile grammar"
            )

        self._handle = handle
        self._closed = False

    def close(self) -> None:
        """Free the native grammar handle.

        This method is idempotent - calling it multiple times is safe.
        After closing, the grammar cannot be used for generation.
        """
        if not self._closed and hasattr(self, "_handle") and self._handle:
            lib = get_lib()
            lib.talu_grammar_free(self._handle)
            self._handle = None
            self._closed = True

    def __enter__(self) -> Grammar:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, closing grammar."""
        self.close()

    def __del__(self) -> None:
        """Free the native grammar handle (destructor fail-safe)."""
        try:
            self.close()
        except Exception:
            # Suppress errors during interpreter shutdown
            pass

    @property
    def schema(self) -> dict[str, Any]:
        """The compiled JSON schema as a dictionary."""
        return self._schema_dict

    @property
    def response_format(self) -> type | dict[str, Any]:
        """The original response format passed to the constructor.

        This is used to hydrate Response.parsed.
        """
        return self._response_format

    def __repr__(self) -> str:
        if isinstance(self._response_format, type):
            name = self._response_format.__name__
        else:
            name = "dict"
        return f"Grammar({name})"

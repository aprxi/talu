"""
Response - Generation result with metadata.

This module provides two response types for type-safe generation results:

- Response: Completed generation result (non-streaming)
- StreamingResponse: Streaming generation result (iterable)

Both types behave like strings for casual use but expose metadata for power users.

Streaming Behavior (Default for Chat)
-------------------------------------

``Chat()`` and ``Chat.__call__()`` default to ``stream=True``, returning
``StreamingResponse``. This provides immediate feedback and matches the
industry standard for chat interfaces (ChatGPT, Claude, etc.).

**When stream=True (default):**

- Tokens arrive incrementally as the model generates
- Iteration yields tokens as they arrive
- Single-use iterator: cannot iterate twice
- Use ``"".join(response)`` or iterate directly to get full text
- ``response.text`` contains full text after iteration completes

**When stream=False:**

- ``Response`` is returned after generation completes
- ``response.text`` is immediately available
- Use when you need the complete result immediately
- Slightly slower perceived latency (user waits until generation finishes)

When to Use Streaming (stream=True)
-----------------------------------

- Interactive applications (CLIs, chat interfaces)
- Long generations where you want real-time feedback
- Applications showing progress indicators
- User-facing interfaces where latency perception matters
- Reduces perceived latency (tokens appear immediately)

When to Use Non-Streaming (stream=False)
----------------------------------------

- Batch processing (collect all responses at once)
- API endpoints returning JSON with full text
- Simple scripts where you don't need incremental tokens
- Testing/automation where latency doesn't matter
- Cases where you need deterministic timing

Why stream=True is the Default
------------------------------

Streaming provides real-time feedback as tokens arrive:

- Reduces perceived latency (users see progress immediately)
- Prevents confusion about "hanging" during long generations (10+ seconds)
- Matches industry standard for chat interfaces (ChatGPT, Claude, etc.)

For non-streaming one-shot completions, use ``client.ask()`` or
``chat.send(prompt, stream=False)``.

Important: StreamingResponse is Single-Use
------------------------------------------

Once exhausted, you cannot iterate again. If you need full text later,
cache it during iteration::

    >>> response = chat("Tell me a joke")
    >>> for token in response:  # First iteration - yields tokens
    ...     print(token, end="", flush=True)
    >>> # After iteration, use response.text (not re-iterate)
    >>> print(response.text)

Simple Usage
------------

    >>> response = chat.send("Hello!", stream=False)
    >>> print(response)           # Prints text directly
    >>> text = str(response)      # Explicit conversion
    >>> if "yes" in response:     # String operations work
    ...     print("Affirmative")

Streaming Usage (Default)
-------------------------

    >>> chat = Chat("Qwen/Qwen3-0.6B")
    >>> response = chat("Tell me a joke")  # stream=True by default
    >>> for token in response:    # StreamingResponse is iterable
    ...     print(token, end="", flush=True)

Continuing Conversations
------------------------

    >>> response = chat("What is 2+2?")
    >>> response = response.append("Why?")     # Continue the conversation
    >>> response = response.append("Are you sure?")

Accessing the Chat
------------------

    >>> response = chat("Hello!")
    >>> response.chat             # The Chat that generated this response
    >>> response.chat.items       # Full conversation history

Metadata Access
---------------

    >>> response.text             # The generated text
    >>> response.tokens           # List of token IDs
    >>> response.finish_reason    # "eos_token", "length", "stop_sequence"
    >>> response.usage            # Token counts
    >>> response.model            # Which model generated this

Usage Details
-------------

    >>> response.usage.prompt_tokens
    >>> response.usage.completion_tokens
    >>> response.usage.total_tokens

Log Probabilities (if requested)
--------------------------------

    >>> response = chat.send("Hello!", logprobs=True)
    >>> for lp in response.logprobs:
    ...     print(f"{lp.token_str}: {lp.logprob:.3f}")

Tool Calls (for agent applications)
-----------------------------------

    >>> response = chat("Search for Python tutorials")
    >>> if response.tool_calls:
    ...     for tool in response.tool_calls:
    ...         print(f"Call: {tool.name}({tool.arguments})")
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from talu.router.config.grammar import Grammar

from ...exceptions import IncompleteJSONError, SchemaValidationError, StateError, ValidationError
from ..tools import ToolCall
from .hydrate import dict_to_dataclass
from .metadata import ResponseMetadata

if TYPE_CHECKING:
    from talu.types import ContentPart

    from ..hooks import HookManager
    from ..session import AsyncChat, Chat

__all__ = [
    "Token",
    "Usage",
    "Timings",
    "TokenLogprob",
    "FinishReason",
    "Response",
    "AsyncResponse",
    "StreamingResponse",
    "AsyncStreamingResponse",
    "ResponseMetadata",
]


class Token(str):
    """
    Single token from a streaming response.

    Token is returned during streaming iteration. It behaves exactly like a string
    for casual use (print, concatenation, etc.) but also carries per-token
    metadata when logprobs, token IDs, or stop reason detection are needed.

    Attributes
    ----------
        id: The token ID from the tokenizer vocabulary.
        logprob: Log probability of this token (if logprobs were requested), or None.
        is_special: True if this is a special token (EOS, BOS, etc.).
        finish_reason: If this is the last token, why generation stopped. Otherwise None.
            Possible values: "eos_token", "length", "stop_sequence", "tool_calls".

    Example:
        >>> for token in chat("Hello", stream=True):
        ...     print(token, end="", flush=True)

    Example (with metadata):
        >>> for token in chat("Hello", stream=True):
        ...     if token.logprob is not None and token.logprob < -5.0:
        ...         ui.highlight_uncertain(token)
        ...     print(token, end="")

    Note:
        Token instances are immutable (like str). Metadata is set at construction
        and cannot be modified afterward.
    """

    __slots__ = ("id", "logprob", "is_special", "finish_reason")

    def __new__(
        cls,
        text: str,
        *,
        id: int = -1,
        logprob: float | None = None,
        is_special: bool = False,
        finish_reason: str | None = None,
    ) -> Token:
        """Create a new Token instance."""
        instance = super().__new__(cls, text)
        instance.id = id
        instance.logprob = logprob
        instance.is_special = is_special
        instance.finish_reason = finish_reason
        return instance

    def __repr__(self) -> str:
        """Return repr with token text and optional metadata."""
        text_preview = str(self)[:20] + "..." if len(self) > 20 else str(self)
        parts = [f"text={text_preview!r}"]
        if self.id >= 0:
            parts.append(f"id={self.id}")
        if self.logprob is not None:
            parts.append(f"logprob={self.logprob:.3f}")
        if self.is_special:
            parts.append("is_special=True")
        if self.finish_reason:
            parts.append(f"finish_reason={self.finish_reason!r}")
        return f"Token({', '.join(parts)})"


@dataclass
class Usage:
    """
    Token usage statistics.

    Attributes
    ----------
        prompt_tokens: Tokens in the input prompt.
        completion_tokens: Tokens in the generated response.
        total_tokens: Total tokens (prompt + completion).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Timings:
    """
    Generation timing breakdown.

    Provides detailed performance metrics for generation, useful for
    profiling, optimization, and monitoring latency in production.

    Attributes
    ----------
        prefill_ms: Time to process the prompt (milliseconds).
            This is the "time to first token" - how long before generation starts.
        generation_ms: Time to generate all tokens (milliseconds).
            This is the decode phase - actual token generation time.
        tokens_per_second: Generation throughput (tokens/sec).
            Calculated as completion_tokens / (generation_ms / 1000).

    Example:
        >>> response = chat("Tell me a story")
        >>> if response.timings:
        ...     print(f"Prefill: {response.timings.prefill_ms:.1f}ms")
        ...     print(f"Generation: {response.timings.generation_ms:.1f}ms")
        ...     print(f"Speed: {response.timings.tokens_per_second:.1f} tok/s")
    """

    prefill_ms: float
    generation_ms: float
    tokens_per_second: float

    @classmethod
    def from_ns(cls, prefill_ns: int, generation_ns: int, token_count: int) -> Timings:
        """
        Create Timings from nanosecond values.

        Args:
            prefill_ns: Prefill time in nanoseconds.
            generation_ns: Generation time in nanoseconds.
            token_count: Number of tokens generated.

        Returns
        -------
            Timings instance with millisecond values and throughput.
        """
        prefill_ms = prefill_ns / 1_000_000
        generation_ms = generation_ns / 1_000_000
        # Calculate tokens per second (avoid division by zero)
        if generation_ms > 0:
            tokens_per_second = token_count / (generation_ms / 1000)
        else:
            tokens_per_second = 0.0
        return cls(prefill_ms, generation_ms, tokens_per_second)


@dataclass
class TokenLogprob:
    """
    Log probability for a single token.

    Attributes
    ----------
        token: Token ID.
        token_str: Token as string.
        logprob: Log probability.
        top_logprobs: Alternative tokens at this position.
    """

    token: int
    token_str: str
    logprob: float
    top_logprobs: list[tuple[int, str, float]] | None = None


from talu.types import FinishReason  # noqa: E402 — canonical location


class _ResponseBase:
    """
    Base class for response types with shared string-like behavior.

    This class provides common functionality for Response, AsyncResponse,
    StreamingResponse, and AsyncStreamingResponse. Not intended for direct use.

    The ``_stream_mode`` flag tracks whether this response was created in
    streaming mode, enabling ``append()`` to automatically inherit the same
    streaming behavior.

    The ``_msg_index`` tracks which message index this response represents,
    enabling auto-forking when appending to a response that's no longer at
    the conversation tip.
    """

    def __init__(
        self,
        text: str = "",
        *,
        tokens: list[int] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
        timings: Timings | None = None,
        model: str | None = None,
        logprobs: list[TokenLogprob] | None = None,
        tool_calls: list[ToolCall] | None = None,
        chat: Chat | None = None,
        metadata: ResponseMetadata | None = None,
        _stream_mode: bool = False,
        _msg_index: int | None = None,
        _content: list[ContentPart] | None = None,
        _prompt: str | None = None,
    ):
        self._text = text
        self._tokens = tokens or []
        self._finish_reason = finish_reason or FinishReason.EOS_TOKEN
        self._usage = usage
        self._timings = timings
        self._model = model
        self._logprobs = logprobs
        self._tool_calls = tool_calls
        self._tool_registry: dict[str, Callable[..., Any]] | None = None
        self._chat = chat
        self._stream_mode = _stream_mode
        # Store content parts for multimodal output symmetry
        self.__content = _content
        # Store rendered prompt for audit trail
        self._prompt = _prompt
        # Track position in conversation for auto-fork on divergent append
        # If not provided, compute from chat (len - 1 is the last message index)
        if _msg_index is not None:
            self._msg_index = _msg_index
        elif chat is not None:
            self._msg_index = len(chat.items) - 1
        else:
            self._msg_index = -1
        self.metadata = metadata or ResponseMetadata(
            finish_reason=self._finish_reason or FinishReason.EOS_TOKEN
        )

    @property
    def text(self) -> str:
        """The generated text content."""
        return self._text

    @property
    def tokens(self) -> list[int]:
        """List of generated token IDs."""
        return self._tokens

    @property
    def finish_reason(self) -> str:
        """Why generation stopped."""
        return self._finish_reason

    @property
    def usage(self) -> Usage | None:
        """Token usage statistics."""
        return self._usage

    @property
    def timings(self) -> Timings | None:
        """Generation timing breakdown."""
        return self._timings

    @property
    def model(self) -> str | None:
        """Model identifier that generated this response."""
        return self._model

    @property
    def logprobs(self) -> list[TokenLogprob] | None:
        """Token log probabilities (if requested)."""
        return self._logprobs

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        """Tool calls requested by the model (if any)."""
        return self._tool_calls

    @property
    def chat(self) -> Chat | None:
        """The Chat that generated this response."""
        return self._chat

    @property
    def content(self) -> list[ContentPart]:
        """
        Structured content parts for multimodal output symmetry.

        Returns a list of content parts, enabling symmetric handling of input
        and output. For text-only responses, this returns ``[OutputText(text=...)]``.
        Future multimodal models will return additional part types (OutputImage, etc.).

        This property is the source of truth for response content. The ``.text``
        property is a convenience that concatenates all text parts.

        Returns
        -------
            List of content parts (currently OutputText for text responses).

        Example:
            >>> response = chat("Hello!")
            >>> for part in response.content:
            ...     if part.type == ContentType.OUTPUT_TEXT:
            ...         print(part.text)

        Note:
            Currently only returns OutputText. Future versions may include
            OutputImage, OutputAudio, etc. as models evolve.
        """
        if self.__content is not None:
            return self.__content

        # Lazily construct content from text for backward compatibility
        from talu.types import OutputText

        return [OutputText(text=self._text)]

    @property
    def prompt(self) -> str | None:
        """
        The fully rendered prompt sent to the model (audit trail).

        Contains the exact string that was fed to the model engine after all
        templating, system prompt injection, and formatting was applied. Useful
        for debugging template issues and understanding exactly what the model saw.

        Returns
        -------
            The rendered prompt string, or None if not available.

        Example:
            >>> response = chat("Hello!")
            >>> print(response.prompt)
            <|im_start|>system
            You are a helpful assistant.
            <|im_end|>
            <|im_start|>user
            Hello!
            <|im_end|>
            <|im_start|>assistant

        Note:
            Only available for responses generated through Chat. May be None
            for responses from remote APIs or when prompt wasn't captured.
        """
        return self._prompt

    # =========================================================================
    # String-like behavior
    # =========================================================================

    def __str__(self) -> str:
        """Return the text content."""
        return self._text

    def __contains__(self, item: str) -> bool:
        """Enable 'x in response' checks."""
        return item in self._text

    def __eq__(self, other: object) -> bool:
        """Compare with string or another Response."""
        if isinstance(other, str):
            return self._text == other
        if isinstance(other, _ResponseBase):
            return self._text == other._text
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on text content."""
        return hash(self._text)

    def __len__(self) -> int:
        """Length of text content."""
        return len(self._text)

    def __add__(self, other: str) -> str:
        """Concatenate with string."""
        return self._text + other

    def __radd__(self, other: str) -> str:
        """Concatenate with string (reversed)."""
        return other + self._text

    # =========================================================================
    # String method delegation
    # =========================================================================

    def lower(self) -> str:
        """Return text in lowercase."""
        return self._text.lower()

    def upper(self) -> str:
        """Return text in uppercase."""
        return self._text.upper()

    def strip(self, chars: str | None = None) -> str:
        """Return text with leading/trailing chars removed."""
        return self._text.strip(chars)

    def split(self, sep: str | None = None, maxsplit: int = -1) -> list[str]:
        """Split text."""
        return self._text.split(sep, maxsplit)

    def startswith(self, prefix: str) -> bool:
        """Check if text starts with prefix."""
        return self._text.startswith(prefix)

    def endswith(self, suffix: str) -> bool:
        """Check if text ends with suffix."""
        return self._text.endswith(suffix)

    def replace(self, old: str, new: str, count: int = -1) -> str:
        """Replace occurrences in text."""
        return self._text.replace(old, new, count)

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Convert response to a JSON-serializable dictionary.

        This solves the "serialization trap" where Response acts like a string
        but isn't directly JSON serializable. Use this for API responses,
        logging, or any context requiring JSON.

        Returns
        -------
            Dict with text, finish_reason, model, and usage (if available).

        Example - FastAPI endpoint:
            >>> @app.post("/chat")
            >>> async def chat_endpoint(message: str):
            ...     response = await chat(message)
            ...     return response.to_dict()  # JSON serializable

        Example - Logging:
            >>> import json
            >>> response = chat("Hello!")
            >>> json.dumps(response.to_dict())  # Works!

        Example - Custom response structure:
            >>> result = {
            ...     "success": True,
            ...     "data": response.to_dict(),
            ... }
        """
        result: dict[str, Any] = {
            "text": self._text,
            "finish_reason": self._finish_reason,
        }

        if self._model is not None:
            result["model"] = self._model

        if self._usage is not None:
            result["usage"] = {
                "prompt_tokens": self._usage.prompt_tokens,
                "completion_tokens": self._usage.completion_tokens,
                "total_tokens": self._usage.total_tokens,
            }

        if self._timings is not None:
            result["timings"] = {
                "prefill_ms": self._timings.prefill_ms,
                "generation_ms": self._timings.generation_ms,
                "total_ms": self._timings.prefill_ms + self._timings.generation_ms,
                "tokens_per_second": self._timings.tokens_per_second,
            }

        if self._tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self._tool_calls
            ]

        return result

    def _prepare_tool_result(self, tool_call_id: str, result: Any) -> Chat:
        """Serialize tool result, append to conversation, return chat for continuation.

        Shared logic for sync and async submit_tool_result.
        """
        if not isinstance(result, str):
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result)
            else:
                result_str = str(result)
        else:
            result_str = result

        chat = self._chat
        if chat is None:
            raise StateError("Cannot submit tool result: no Chat session.", code="STATE_NO_CHAT")

        chat._append_function_call_output(tool_call_id, result_str)
        return chat


class Response(_ResponseBase):
    """
    Completed generation result.

    Wraps the result of a non-streaming generation. Behaves like a string
    for simple use but exposes rich metadata when needed.

    The ``.text`` property contains the complete generated text, available
    immediately without iteration. Convert to string with ``str(response)``
    or access directly via ``response.text``.

    Attributes
    ----------
        text: The generated text content (always available immediately).
        tokens: List of generated token IDs.
        finish_reason: Why generation stopped (eos_token, length, stop_sequence).
        usage: Token usage statistics.
        timings: Generation timing breakdown.
        model: Model identifier that generated this response.
        logprobs: Token log probabilities (if requested).

    Example - Casual use:
        >>> response = chat("Hello!")
        >>> print(response)  # Works like a string
        Hi there!
        >>> if "hello" in response.lower():
        ...     print("Greeting detected")

    Example - Power user:
        >>> response = chat("Hello!")
        >>> print(f"Used {response.usage.total_tokens} tokens")
        >>> print(f"Finished due to: {response.finish_reason}")
        >>> print(f"Model: {response.model}")
    """

    def __init__(
        self,
        text: str = "",
        *,
        tokens: list[int] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
        timings: Timings | None = None,
        model: str | None = None,
        logprobs: list[TokenLogprob] | None = None,
        tool_calls: list[ToolCall] | None = None,
        chat: Chat | None = None,
        metadata: ResponseMetadata | None = None,
        _response_format: type | dict | Grammar | None = None,
        _stream_mode: bool = False,
        _msg_index: int | None = None,
        _content: list[ContentPart] | None = None,
        _prompt: str | None = None,
    ):
        super().__init__(
            text=text,
            tokens=tokens,
            finish_reason=finish_reason,
            usage=usage,
            timings=timings,
            model=model,
            logprobs=logprobs,
            tool_calls=tool_calls,
            chat=chat,
            metadata=metadata,
            _stream_mode=_stream_mode,
            _msg_index=_msg_index,
            _content=_content,
            _prompt=_prompt,
        )
        self._response_format = _response_format

    @property
    def parsed(self) -> Any:
        """Parse and validate the response against the response_format schema.

        If a response_format was provided during generation, this property
        parses the response text as JSON and validates/hydrates it into the
        specified type (dataclass or Pydantic model).

        Returns
        -------
            The parsed and validated response object, or None if no
            response_format was specified.

        Raises
        ------
            IncompleteJSONError: If finish_reason is "length" and JSON is malformed.
            json.JSONDecodeError: If the response text is not valid JSON.
            SchemaValidationError: If the parsed data doesn't match the schema.
        """
        if self._response_format is None:
            return None

        if self.finish_reason == "length":
            try:
                data = json.loads(self._text)
            except json.JSONDecodeError as err:
                raise IncompleteJSONError(self._text, self.finish_reason) from err
        else:
            data = json.loads(self._text)

        # Hydrate to dataclass or Pydantic model if applicable
        if dataclasses.is_dataclass(self._response_format) or callable(
            getattr(self._response_format, "model_validate", None)
        ):
            try:
                return dict_to_dataclass(self._response_format, data)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                raise SchemaValidationError(self._text, exc) from exc

        return data

    def submit_tool_result(self, tool_call_id: str, result: Any) -> Response:
        """
        Submit a tool result and continue generation.

        When the model requests tool calls (via response.tool_calls), execute
        them and submit the results back using this method. The model will
        then continue generation with the tool results in context.

        Args:
            tool_call_id: The ID from the tool call (tool_call.id).
            result: The result to send back (will be JSON serialized if not str).

        Returns
        -------
            New Response from continued generation.

        Raises
        ------
            StateError: If no Chat session is attached.

        Example:
            >>> response = chat("What's the weather?", tools=[get_weather])
            >>> while response.tool_calls:
            ...     for call in response.tool_calls:
            ...         result = call.execute()
            ...         response = response.submit_tool_result(call.id, result)
            >>> print(response)
        """
        chat = self._prepare_tool_result(tool_call_id, result)
        return chat._continue_generation(tools_registry=self._tool_registry)  # type: ignore[attr-defined]

    @property
    def _schema_overhead(self) -> int:
        """Token cost of the injected JSON schema."""
        return self.metadata.schema_tokens if self.metadata else 0

    def append(
        self,
        message: str,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """
        Continue the conversation with a follow-up message (sync).

        This is the primary way to have multi-turn conversations. The append
        uses the same Chat that generated this response, maintaining context.

        **Auto-Fork Behavior:** If the conversation has moved past this response
        (i.e., more messages were added after this response was generated),
        append() automatically forks the conversation and truncates it back to
        this point before sending the new message. This enables intuitive
        branching where you can append to any previous response without worrying
        about conversation state.

        The append automatically inherits streaming mode from the original response.

        Args:
            message: The follow-up message to send.
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns
        -------
            Response if original was non-streaming, StreamingResponse if streaming.

        Raises
        ------
            StateError: If this response has no associated Chat.

        Example - Linear conversation:
            >>> r1 = chat("What is 2+2?")
            >>> r2 = r1.append("Why?")      # Continues normally
            >>> r3 = r2.append("Thanks!")   # Continues normally

        Example - Branching:
            >>> r1 = chat("Idea 1")
            >>> r2 = r1.append("Critique it")   # chat has [Idea 1, Critique]
            >>> r3 = r1.append("Expand on it")  # Auto-forks! r3.chat is new
            >>> # Original chat unchanged, r3.chat has [Idea 1, Expand]
        """
        if self._chat is None:
            raise StateError(
                "Cannot append to a one-shot response. "
                "Use talu.Chat() for multi-turn conversations:\n\n"
                "    chat = talu.Chat('model')\n"
                "    r1 = chat('Hello!')\n"
                "    r2 = r1.append('Tell me more')",
                code="STATE_NO_CHAT",
            )

        # Check if we're still at the tip of the conversation
        current_last_index = len(self._chat.items) - 1

        if self._msg_index == current_last_index:
            # Case A: Linear - we're at the tip, continue normally
            return self._chat.send(message, stream=self._stream_mode, **kwargs)
        else:
            # Case B: Branching - conversation has moved on, auto-fork
            forked_chat = self._chat._fork_at(self._msg_index)
            return forked_chat.send(message, stream=self._stream_mode, **kwargs)

    def __repr__(self) -> str:
        """Return repr with text preview and metadata."""
        preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
        parts = [f"text={preview!r}"]
        if self._model:
            parts.append(f"model={self._model!r}")
        if self._usage:
            parts.append(f"tokens={self._usage.total_tokens}")
        return f"Response({', '.join(parts)})"


class AsyncResponse(_ResponseBase):
    """
    Async completed generation result.

    Returned by AsyncChat for non-streaming generation. Contains the complete
    generated text and metadata. Behaves like a string for simple use but
    exposes rich metadata when needed.

    The ``append()`` method is async and must be awaited.

    Attributes
    ----------
        text: The generated text content.
        tokens: List of generated token IDs.
        finish_reason: Why generation stopped (eos_token, length, stop_sequence).
        usage: Token usage statistics.
        timings: Generation timing breakdown.
        model: Model identifier that generated this response.
        logprobs: Token log probabilities (if requested).

    Example:
        >>> response = await chat.send("Hello!")
        >>> print(response)  # Works like a string
        >>> print(f"Used {response.usage.total_tokens} tokens")

    Example - Multi-turn:
        >>> response = await chat.send("What is 2+2?")
        >>> response = await response.append("Why?")
        >>> response = await response.append("Are you sure?")
    """

    def __init__(
        self,
        text: str = "",
        *,
        tokens: list[int] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
        timings: Timings | None = None,
        model: str | None = None,
        logprobs: list[TokenLogprob] | None = None,
        tool_calls: list[ToolCall] | None = None,
        chat: AsyncChat | None = None,
        metadata: ResponseMetadata | None = None,
        _response_format: type | dict | Grammar | None = None,
        _stream_mode: bool = False,
        _content: list[ContentPart] | None = None,
        _prompt: str | None = None,
    ):
        super().__init__(
            text=text,
            tokens=tokens,
            finish_reason=finish_reason,
            usage=usage,
            timings=timings,
            model=model,
            logprobs=logprobs,
            tool_calls=tool_calls,
            chat=chat,  # type: ignore[arg-type]
            metadata=metadata,
            _stream_mode=_stream_mode,
            _content=_content,
            _prompt=_prompt,
        )
        self._chat: AsyncChat | None = chat
        self._response_format = _response_format

    @property
    def parsed(self) -> Any:
        """Parse and validate the response against the response_format schema.

        If a response_format was provided during generation, this property
        parses the response text as JSON and validates/hydrates it into the
        specified type (dataclass or Pydantic model).

        Returns
        -------
            The parsed and validated response object, or None if no
            response_format was specified.

        Raises
        ------
            IncompleteJSONError: If finish_reason is "length" and JSON is malformed.
            json.JSONDecodeError: If the response text is not valid JSON.
            SchemaValidationError: If the parsed data doesn't match the schema.
        """
        if self._response_format is None:
            return None

        if self.finish_reason == "length":
            try:
                data = json.loads(self._text)
            except json.JSONDecodeError as err:
                raise IncompleteJSONError(self._text, self.finish_reason) from err
        else:
            data = json.loads(self._text)

        # Hydrate to dataclass or Pydantic model if applicable
        if dataclasses.is_dataclass(self._response_format) or callable(
            getattr(self._response_format, "model_validate", None)
        ):
            try:
                return dict_to_dataclass(self._response_format, data)
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                raise SchemaValidationError(self._text, exc) from exc

        return data

    async def submit_tool_result(self, tool_call_id: str, result: Any) -> AsyncResponse:
        """
        Submit a tool result and continue generation (async).

        Args:
            tool_call_id: The ID from the tool call (tool_call.id).
            result: The result to send back (will be JSON serialized if not str).

        Returns
        -------
            New AsyncResponse from continued generation.

        Raises
        ------
            StateError: If no AsyncChat session is attached.
        """
        chat = self._prepare_tool_result(tool_call_id, result)
        return await chat._continue_generation(tools_registry=self._tool_registry)  # type: ignore[attr-defined]

    @property
    def _schema_overhead(self) -> int:
        """Token cost of the injected JSON schema."""
        return self.metadata.schema_tokens if self.metadata else 0

    async def append(
        self,
        message: str,
        **kwargs: Any,
    ) -> AsyncResponse | AsyncStreamingResponse:
        """
        Continue the conversation with a follow-up message (async).

        This is the async way to have multi-turn conversations. The append
        uses the same AsyncChat that generated this response. Must be awaited.

        **Auto-Fork Behavior:** If the conversation has moved past this response,
        append() automatically forks the conversation and truncates it back to
        this point before sending the new message. See Response.append() for details.

        The append automatically inherits streaming mode from the original response.

        Args:
            message: The follow-up message to send.
            **kwargs: Generation parameters (temperature, max_tokens, etc.)

        Returns
        -------
            AsyncResponse if original was non-streaming, AsyncStreamingResponse if streaming.

        Raises
        ------
            StateError: If this response has no associated AsyncChat.

        Example:
            >>> response = await chat.send("What is 2+2?")
            >>> response = await response.append("Why?")
            >>> response = await response.append("Are you sure?")
        """
        if self._chat is None:
            raise StateError(
                "Cannot append: this response has no associated AsyncChat. "
                "Create an AsyncChat object to have multi-turn conversations.",
                code="STATE_NO_CHAT",
            )

        # Check if we're still at the tip of the conversation
        current_last_index = len(self._chat.items) - 1

        if self._msg_index == current_last_index:
            # Case A: Linear - we're at the tip, continue normally
            return await self._chat.send(message, stream=self._stream_mode, **kwargs)
        else:
            # Case B: Branching - conversation has moved on, auto-fork
            forked_chat = self._chat._fork_at(self._msg_index)
            return await forked_chat.send(message, stream=self._stream_mode, **kwargs)

    def __repr__(self) -> str:
        """Return repr with text preview and metadata."""
        preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
        parts = [f"text={preview!r}"]
        if self._model:
            parts.append(f"model={self._model!r}")
        if self._usage:
            parts.append(f"tokens={self._usage.total_tokens}")
        return f"AsyncResponse({', '.join(parts)})"


class StreamingResponse(_ResponseBase):
    """
    Streaming generation result that yields tokens incrementally.

    Returned when calling ``chat(stream=True)``. Iterate over it to receive
    tokens in real-time. Text accumulates in ``.text`` as you iterate.

    Streaming Behavior
    ------------------
    StreamingResponse objects are **single-use iterators**. Once exhausted,
    you cannot iterate again. If you need the full text later, cache it
    during iteration::

        >>> response = chat("Hello", stream=True)
        >>> full_text = "".join(response)  # Cache during iteration
        >>> print(full_text)

    Calling ``len(response)`` or accessing ``response.text`` after the stream
    is exhausted returns the cached full text. Iterating multiple times on
    the same StreamingResponse will yield no tokens on subsequent iterations.

    Concurrency:
        Single-consumer. Do not iterate from multiple threads/tasks.

    Attributes
    ----------
        text: The accumulated text (grows during iteration, always available after).
        tokens: List of generated token IDs (populated after iteration).
        finish_reason: Why generation stopped (available after iteration).
        usage: Token usage statistics (available after iteration).
        timings: Generation timing breakdown (available after iteration).
        model: Model identifier that generated this response.

    Example:
        >>> response = chat("Tell me a joke", stream=True)
        >>> for token in response:
        ...     print(token, end="", flush=True)
        >>> print()
        >>> print(f"Full text: {response.text}")
        >>> print(f"Tokens used: {response.usage.total_tokens}")

    Example - With callback:
        >>> def on_token(t): print(t, end="")
        >>> response = chat("Hello", stream=True, on_token=on_token)
        >>> for _ in response: pass  # Drain to trigger callbacks

    Note:
        After iteration completes, you can access .text for the full
        accumulated text and .usage/.timings for metadata.
    """

    def __init__(
        self,
        *,
        stream_iterator: Iterator,
        on_token: Callable[[str], None] | None = None,
        on_complete: Callable[[str], None] | None = None,
        tokens: list[int] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
        timings: Timings | None = None,
        model: str | None = None,
        logprobs: list[TokenLogprob] | None = None,
        tool_calls: list[ToolCall] | None = None,
        chat: Chat | None = None,
        metadata: ResponseMetadata | None = None,
        _response_format: type | dict | Grammar | None = None,
        _stream_mode: bool = True,
        _hooks: HookManager | None = None,
        _generation_start_time: float | None = None,
        _prompt: str | None = None,
    ):
        super().__init__(
            text="",
            tokens=tokens,
            finish_reason=finish_reason,
            usage=usage,
            timings=timings,
            model=model,
            logprobs=logprobs,
            tool_calls=tool_calls,
            chat=chat,
            metadata=metadata,
            _stream_mode=_stream_mode,
            _prompt=_prompt,
        )
        self._stream_iterator = stream_iterator
        self._on_token = on_token
        self._on_complete = on_complete
        self._stream_exhausted = False
        self._response_format = _response_format
        self._hooks = _hooks
        self._generation_start_time = _generation_start_time
        self._first_token_dispatched = False

    @property
    def text(self) -> str:
        """
        The generated text content.

        For StreamingResponse, accessing this property will auto-drain the stream
        if it hasn't been consumed yet. This ensures that `.text` always returns
        the complete generated text, regardless of whether the caller explicitly
        iterated over the response.

        Returns
        -------
            The full generated text content.
        """
        if not self._stream_exhausted:
            # Auto-drain the stream to get full text
            for _ in self:
                pass
        return self._text

    @property
    def prompt(self) -> str | None:
        """
        The fully rendered prompt (available after iteration completes).

        For streaming responses, the prompt is captured after iteration finishes
        since messages are added during streaming. Access this property after
        consuming the stream.

        Returns
        -------
            The rendered prompt string, or None if iteration hasn't completed
            or the prompt couldn't be captured.
        """
        # If prompt was explicitly set, return it
        if self._prompt is not None:
            return self._prompt

        # For streaming, try to capture lazily after iteration
        if self._stream_exhausted and self._chat is not None:
            from ...exceptions import StateError, TaluError

            try:
                self._prompt = self._chat.preview_prompt(add_generation_prompt=False)
                return self._prompt
            except (StateError, TaluError):
                pass  # Chat may be closed, have no engine, or template may fail

        return None

    def __iter__(self) -> Iterator[Token]:
        """
        Iterate over tokens as they are generated.

        Each iteration yields a Token object that behaves like a string but
        also carries optional metadata (token ID, logprob, etc.). This enables
        both casual and power user patterns:

        Casual (Token is a str subclass):
            >>> for token in response:
            ...     print(token, end="")  # Works like a string

        Power User:
            >>> for token in response:
            ...     if token.logprob is not None and token.logprob < -5.0:
            ...         highlight(token)

        Yields
        ------
            Token objects (str subclass) as they are generated.

        Example:
            >>> response = chat("Tell me a joke", stream=True)
            >>> for token in response:
            ...     print(token, end="", flush=True)
            >>> print(response.text)  # Full accumulated text
        """
        import time as _time

        if self._stream_exhausted:
            # Already consumed, return empty iterator
            return

        error: BaseException | None = None
        try:
            for stream_token in self._stream_iterator:
                # Dispatch first token hook (TTFT measurement)
                if (
                    not self._first_token_dispatched
                    and self._hooks is not None
                    and self._chat is not None
                    and self._generation_start_time is not None
                ):
                    ttft_ms = (_time.perf_counter() - self._generation_start_time) * 1000
                    self._hooks.dispatch_first_token(self._chat, ttft_ms)
                    self._first_token_dispatched = True

                # Accumulate text — StreamToken has .text, str() handles both
                text = stream_token.text if hasattr(stream_token, "text") else str(stream_token)
                self._text += text

                # Wrap in Token object
                # TODO: Pass token_id and logprob from router when available
                token = Token(text)

                # Call callback if provided (pass Token, which is also a str)
                if self._on_token is not None:
                    self._on_token(token)

                yield token
        except BaseException as e:
            error = e
            raise
        finally:
            self._stream_exhausted = True

            # Dispatch generation end hook
            if self._hooks is not None and self._chat is not None:
                err = error if isinstance(error, Exception) else None
                self._hooks.dispatch_end(self._chat, self if error is None else None, err)  # type: ignore[arg-type]

            # Call completion callback (used for storage notifications)
            if self._on_complete is not None and error is None:
                self._on_complete(self._text)

    def append(
        self,
        message: str,
        **kwargs: Any,
    ) -> StreamingResponse:
        """
        Continue the conversation with a follow-up message.

        Returns a StreamingResponse (inherits streaming mode from this response).
        See Response.append() for full documentation including auto-fork behavior.

        Args:
            message: The follow-up message text.
            **kwargs: Generation overrides (temperature, max_tokens, etc.).

        Raises
        ------
            StateError: If this response has no associated Chat.
        """
        if self._chat is None:
            raise StateError(
                "Cannot append to a one-shot response. "
                "Use talu.Chat() for multi-turn conversations:\n\n"
                "    chat = talu.Chat('model')\n"
                "    r1 = chat('Hello!')\n"
                "    r2 = r1.append('Tell me more')",
                code="STATE_NO_CHAT",
            )

        # Check if we're still at the tip of the conversation
        current_last_index = len(self._chat.items) - 1

        if self._msg_index == current_last_index:
            # Case A: Linear - we're at the tip, continue normally
            return self._chat.send(message, stream=True, **kwargs)
        else:
            # Case B: Branching - conversation has moved on, auto-fork
            forked_chat = self._chat._fork_at(self._msg_index)
            return forked_chat.send(message, stream=True, **kwargs)

    def __str__(self) -> str:
        """Return the text content, auto-draining stream if needed."""
        return self.text  # Uses the property which auto-drains

    def __repr__(self) -> str:
        """Return repr with streaming state."""
        status = "exhausted" if self._stream_exhausted else "pending"
        preview = self._text[:30] + "..." if len(self._text) > 30 else self._text
        return f"StreamingResponse(status={status}, text={preview!r})"


class AsyncStreamingResponse(_ResponseBase):
    """
    Async streaming generation result that yields tokens incrementally.

    Returned when calling ``chat.send(stream=True)`` on AsyncChat. Use
    ``async for`` to receive tokens in real-time. Text accumulates in
    ``.text`` as you iterate.

    Concurrency:
        Single-consumer. Do not iterate from multiple tasks.

    Attributes
    ----------
        text: The accumulated text (grows during iteration).
        tokens: List of generated token IDs (populated after iteration).
        finish_reason: Why generation stopped (available after iteration).
        usage: Token usage statistics (available after iteration).
        timings: Generation timing breakdown (available after iteration).
        model: Model identifier that generated this response.

    Example:
        >>> response = await chat.send("Tell me a joke", stream=True)
        >>> async for token in response:
        ...     print(token, end="", flush=True)
        >>> print()
        >>> print(f"Full text: {response.text}")
    """

    def __init__(
        self,
        *,
        async_stream_iterator: AsyncIterator,
        on_token: Callable[[str], None] | None = None,
        on_complete: Callable[[str], None] | None = None,
        tokens: list[int] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
        timings: Timings | None = None,
        model: str | None = None,
        logprobs: list[TokenLogprob] | None = None,
        tool_calls: list[ToolCall] | None = None,
        chat: AsyncChat | None = None,
        metadata: ResponseMetadata | None = None,
        _response_format: type | dict | Grammar | None = None,
        _stream_mode: bool = True,
        _hooks: HookManager | None = None,
        _generation_start_time: float | None = None,
        _prompt: str | None = None,
    ):
        super().__init__(
            text="",
            tokens=tokens,
            finish_reason=finish_reason,
            usage=usage,
            timings=timings,
            model=model,
            logprobs=logprobs,
            tool_calls=tool_calls,
            chat=chat,  # type: ignore[arg-type]
            metadata=metadata,
            _stream_mode=_stream_mode,
            _prompt=_prompt,
        )
        self._chat: AsyncChat | None = chat
        self._async_stream_iterator = async_stream_iterator
        self._on_token = on_token
        self._on_complete = on_complete
        self._stream_exhausted = False
        self._response_format = _response_format
        self._hooks = _hooks
        self._generation_start_time = _generation_start_time
        self._first_token_dispatched = False

    @property
    def text(self) -> str:
        """
        The generated text content.

        For AsyncStreamingResponse, this returns the text accumulated so far.
        To get the complete text, ensure you have consumed the stream first
        by iterating with ``async for token in response``.

        Note:
            Unlike sync StreamingResponse, AsyncStreamingResponse cannot
            auto-drain because it would require an async context. If you need
            the full text, iterate over the response first::

                async for _ in response:
                    pass
                full_text = response.text

        Returns
        -------
            The accumulated text content (partial if stream not exhausted).
        """
        return self._text

    @property
    def prompt(self) -> str | None:
        """
        The fully rendered prompt (available after iteration completes).

        For async streaming responses, the prompt is captured after iteration
        finishes since messages are added during streaming. Access this property
        after consuming the stream.

        Returns
        -------
            The rendered prompt string, or None if iteration hasn't completed
            or the prompt couldn't be captured.
        """
        # If prompt was explicitly set, return it
        if self._prompt is not None:
            return self._prompt

        # For streaming, try to capture lazily after iteration
        if self._stream_exhausted and self._chat is not None:
            from ...exceptions import StateError, TaluError

            try:
                self._prompt = self._chat.preview_prompt(add_generation_prompt=False)
                return self._prompt
            except (StateError, TaluError):
                pass  # Chat may be closed, have no engine, or template may fail

        return None

    async def __aiter__(self) -> AsyncIterator[Token]:
        """
        Async iterate over tokens as they are generated.

        Each iteration yields a Token object that behaves like a string but
        also carries optional metadata (token ID, logprob, etc.). This enables
        both casual and power user patterns:

        Casual (Token is a str subclass):
            >>> async for token in response:
            ...     print(token, end="")  # Works like a string

        Power User:
            >>> async for token in response:
            ...     if token.logprob is not None and token.logprob < -5.0:
            ...         highlight(token)

        Yields
        ------
            Token objects (str subclass) as they are generated.

        Example:
            >>> response = await chat.send_async("Tell me a joke", stream=True)
            >>> async for token in response:
            ...     print(token, end="", flush=True)
            >>> print(response.text)  # Full accumulated text
        """
        import time as _time

        if self._stream_exhausted:
            # Already consumed, return empty iterator
            return

        error: BaseException | None = None
        try:
            async for stream_token in self._async_stream_iterator:
                # Dispatch first token hook (TTFT measurement)
                if (
                    not self._first_token_dispatched
                    and self._hooks is not None
                    and self._chat is not None
                    and self._generation_start_time is not None
                ):
                    ttft_ms = (_time.perf_counter() - self._generation_start_time) * 1000
                    self._hooks.dispatch_first_token(self._chat, ttft_ms)
                    self._first_token_dispatched = True

                # Accumulate text — StreamToken has .text, str() handles both
                text = stream_token.text if hasattr(stream_token, "text") else str(stream_token)
                self._text += text

                # Wrap in Token object
                # TODO: Pass token_id and logprob from router when available
                token = Token(text)

                # Call callback if provided (pass Token, which is also a str)
                if self._on_token is not None:
                    self._on_token(token)

                yield token
        except BaseException as e:
            error = e
            raise
        finally:
            self._stream_exhausted = True

            # Dispatch generation end hook
            if self._hooks is not None and self._chat is not None:
                err = error if isinstance(error, Exception) else None
                self._hooks.dispatch_end(self._chat, self if error is None else None, err)  # type: ignore[arg-type]

            # Call completion callback (used for storage notifications)
            if self._on_complete is not None and error is None:
                self._on_complete(self._text)

    async def append(
        self,
        message: str,
        **kwargs: Any,
    ) -> AsyncStreamingResponse:
        """
        Continue the conversation with a follow-up message (async streaming).

        Returns AsyncStreamingResponse (inherits streaming mode). Must be awaited.

        See Response.append() for full documentation including auto-fork behavior.

        Args:
            message: The follow-up message text.
            **kwargs: Generation overrides (temperature, max_tokens, etc.).

        Raises
        ------
            StateError: If this response has no associated AsyncChat.

        Example:
            >>> response = await chat("Hello")  # stream=True by default
            >>> async for token in response:
            ...     print(token, end="")
            >>> response2 = await response.append("Continue")
            >>> async for token in response2:
            ...     print(token, end="")
        """
        if self._chat is None:
            raise StateError(
                "Cannot append: this response has no associated AsyncChat. "
                "Create an AsyncChat object to have multi-turn conversations.",
                code="STATE_NO_CHAT",
            )

        # Check if we're still at the tip of the conversation
        current_last_index = len(self._chat.items) - 1

        if self._msg_index == current_last_index:
            # Case A: Linear - we're at the tip, continue normally
            return await self._chat.send(message, stream=True, **kwargs)
        else:
            # Case B: Branching - conversation has moved on, auto-fork
            forked_chat = self._chat._fork_at(self._msg_index)
            return await forked_chat.send(message, stream=True, **kwargs)

    def __repr__(self) -> str:
        """Return repr with streaming state."""
        status = "exhausted" if self._stream_exhausted else "pending"
        preview = self._text[:30] + "..." if len(self._text) > 30 else self._text
        return f"AsyncStreamingResponse(status={status}, text={preview!r})"

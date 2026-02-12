r"""
Chat - Conversational AI made simple.

Chat is the primary user-facing class for talu. It handles
both chat state management and generation in a unified interface.

Simple Usage:

    >>> from talu import Chat
    >>>
    >>> chat = Chat("Qwen/Qwen3-0.6B", system="You are a pirate.")
    >>> response = chat("Hello!")
    >>> print(response)
    Ahoy there, matey!

Multi-turn Conversations:

    >>> response = chat("What is 2+2?")
    >>> response = response.append("Why?")
    >>> response = response.append("Are you sure?")

Streaming:

    >>> response = chat("Tell me about your ship", stream=True)
    >>> for token in response:
    ...     print(token, end="", flush=True)

Multi-User Serving (shared Client):

    >>> from talu import Client
    >>>
    >>> # Load model once
    >>> client = Client("Qwen/Qwen3-0.6B")
    >>>
    >>> # Create lightweight chats per user
    >>> user1 = client.chat(system="You are helpful.")
    >>> user2 = client.chat(system="You are a pirate.")
    >>>
    >>> # Generate responses
    >>> response = user1("Hello!")
    >>> response = user2("Ahoy!")

Message Management:

    Conversation history is accessed via chat.items, a read-only interface
    using the modern Item-based API (Open Responses format).

    >>> chat = Chat(system="You are helpful.")
    >>> chat.items[0]  # First item (usually system message)
    MessageItem(role='system', content=[...])
    >>> chat.items[0].text  # Get text content
    'You are helpful.'
    >>> len(chat.items)
    1

    Note: Items are read-only. To modify conversation history, use
    Chat methods like clear() or reset(), or create a new Chat.

Using GenerationConfig:

    >>> from talu import Chat, GenerationConfig
    >>>
    >>> # Set default config at construction
    >>> config = GenerationConfig(temperature=0.7, max_tokens=100)
    >>> chat = Chat("Qwen/Qwen3-0.6B", config=config)
    >>>
    >>> # Access config directly (single source of truth)
    >>> print(chat.config.temperature)  # 0.7
    >>>
    >>> # Override per-call with kwargs (preferred for simple overrides)
    >>> chat.send("Solve math", temperature=0.1)
    >>>
    >>> # Override per-call with config object (for multiple params)
    >>> chat.send("Be creative", config=GenerationConfig(temperature=1.2, top_p=0.95))

Configuration Precedence:

    When calling send/stream, parameters are resolved in this order:
    1. **kwargs (e.g., temperature=0.1) - highest priority
    2. config parameter (explicit GenerationConfig object)
    3. chat.config (session default) - lowest priority

    Example:
    >>> chat = Chat(config=GenerationConfig(temperature=0.7))
    >>> chat.send("Hi", config=GenerationConfig(temperature=0.5), temperature=0.1)
    >>> # Uses temperature=0.1 (kwargs win over config param)

Debugging with preview_prompt():

    >>> chat = Chat("Qwen/Qwen3-0.6B", system="You are a pirate.")
    >>> _ = chat("Ahoy!")  # Add user message and generate
    >>> print(chat.preview_prompt())
    # Shows the exact formatted prompt that would be sent to the model

Custom Templates:

    >>> from talu import PromptTemplate, Chat
    >>>
    >>> # Override the model's default template
    >>> custom = PromptTemplate(
    ...     "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}\\n{% endfor %}"
    ...     "ASSISTANT:"
    ... )
    >>> chat = Chat("Qwen/Qwen3-0.6B", chat_template=custom)

Custom Templates & Advanced Logic (Pre-Render Pattern):

    The chat_template argument supports Jinja2 strings or PromptTemplate objects.
    If you require complex logic that Jinja2 cannot handle (e.g., calling
    arbitrary Python functions, external lookups during render), use the
    **Pre-Render Pattern**:

    1. Render your prompt to a string using your own Python logic.
    2. Pass the raw string to ``chat.send()``.

    >>> # Pre-render with custom Python logic
    >>> def build_prompt(user_input: str, context: dict) -> str:
    ...     # Complex logic: database lookups, API calls, etc.
    ...     return f"Context: {context}\\nUser: {user_input}"
    >>>
    >>> my_prompt = build_prompt("Hello", {"date": "2024-01-15"})
    >>> chat.send(my_prompt)  # Raw string bypasses internal templating
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Literal, overload

from talu.router.config import GenerationConfig, Grammar
from talu.types import ItemRecord, ItemStatus, ItemType, MessageItem, MessageRole

from ...db import Database
from ...exceptions import (
    SchemaValidationError,
    StateError,
    TaluError,
)
from ...profile import Profile, new_session_id
from ...template import PromptTemplate
from .._chat_base import ChatBase
from .._generate import SCHEMA_PLACEHOLDER
from ..response import Response, ResponseMetadata, StreamingResponse

if TYPE_CHECKING:
    from talu.client import AsyncClient, Client
    from talu.router import Router


def _build_c_storage_records(items: list[ItemRecord]) -> tuple[Any, list[bytes]]:
    from .._bindings import build_c_storage_records

    type_map = {
        "message": int(ItemType.MESSAGE),
        "function_call": int(ItemType.FUNCTION_CALL),
        "function_call_output": int(ItemType.FUNCTION_CALL_OUTPUT),
        "reasoning": int(ItemType.REASONING),
        "item_reference": int(ItemType.ITEM_REFERENCE),
        "unknown": int(ItemType.UNKNOWN),
    }
    role_map = {
        "system": int(MessageRole.SYSTEM),
        "user": int(MessageRole.USER),
        "assistant": int(MessageRole.ASSISTANT),
        "developer": int(MessageRole.DEVELOPER),
        "unknown": int(MessageRole.UNKNOWN),
    }
    status_map = {
        "in_progress": int(ItemStatus.IN_PROGRESS),
        "waiting": int(ItemStatus.WAITING),
        "completed": int(ItemStatus.COMPLETED),
        "incomplete": int(ItemStatus.INCOMPLETE),
        "failed": int(ItemStatus.FAILED),
    }

    return build_c_storage_records(items, type_map, role_map, status_map)


class Chat(ChatBase):
    """
    Stateful multi-turn chat session.

    Chat is the primary interface for talu. Pass a model string
    to get a fully autonomous chat that handles everything, or pass a client
    for efficient multi-user serving.

    Separation of Concerns:
        - **Chat** manages session state: conversation history, system prompt, templates
        - **Client** manages infrastructure: model loading, GPU layers, API keys, threading

        For custom hardware or backend configuration, create a Client first::

            client = Client("model", gpu_layers=20, api_key="...")
            chat = client.chat(system="You are helpful.")

    Concurrency:
        Not intended for concurrent use. Create one Chat per thread/task.
        Sharing across threads can interleave message history unpredictably.

    Note:
        Creating multiple Chat instances for the same model is efficient - they
        share the underlying engine. Only the message history is per-Chat.

    Args:
        model: Model to load (HuggingFace ID or local path). Creates a default Client.
            For custom configuration (GPU layers, API keys, etc.), use Client instead.
        client: Existing Client to use (for multi-user serving or custom config).
        config: Default GenerationConfig for this session. If provided, these
            settings are used for all send/stream calls unless overridden.
        system: Optional system prompt. Stored as the first message with
            role="system" (accessible via ``messages[0]``). This follows the
            HuggingFace chat template convention where system prompts are
            part of the messages list, not a separate template variable.
        profile: Optional storage profile. When provided, chat history is
            persisted under ``~/.talu/db/<profile>/``. If ``session_id`` is
            not provided, a UUIDv4 session ID is generated automatically.
        session_id: Optional session identifier for this conversation. Used by
            storage backends to group messages by session. When persisting to
            TaluDB (when using talu://), session_id is hashed to SESSION_HASH for efficient
            Jump Reads during session restoration.
        parent_session_id: Optional parent session identifier for forks.
        marker: Session marker for storage backends (default: "" = normal/unmarked).
            Values: "pinned", "archived", "deleted", or "" (normal).
        metadata: Optional session metadata dict (tags, UI state, notes).
        chat_template: Custom chat template to use instead of the model's default.
            Can be a PromptTemplate object or a template string. If None (default),
            uses the model's chat_template from tokenizer_config.json.
        storage: Storage for messages. Defaults to Database(":memory:").
            Use Database("talu://<path>") for TaluDB persistence (requires session_id).
            Cannot be combined with ``profile``.
        offline: If True, disallow network access when resolving model URIs.

    Attributes
    ----------
        config: The session's GenerationConfig. This is the single source of truth
            for generation parameters. Can be read or replaced directly.
        messages: List-like access to all messages (including system prompt).
            The system prompt (if set) appears at index 0 with role="system".
        session_id: The session identifier for this conversation, or None.
        client: The Client used for this chat (if any).
        router: The Router used for generation (if any).
        chat_template: The PromptTemplate used for formatting prompts.

    Raises
    ------
        ValidationError: If both ``model`` and ``client`` are provided.
        MemoryError: If Chat creation fails (insufficient memory).

    Note:
        Provide either `model` OR `client`, not both. If neither is provided,
        Chat works as a lightweight state container (for advanced use).

    Configuration Precedence:
        When calling send/stream, parameters are resolved in this order:
        1. **kwargs (e.g., ``temperature=0.1``) - highest priority
        2. ``config`` parameter (explicit GenerationConfig object)
        3. ``self.config`` (session default) - lowest priority

    Example - Simple chat:
        >>> chat = Chat("Qwen/Qwen3-0.6B", system="You are helpful.")
        >>> response = chat("What is 2+2?")
        >>> print(response)
        4

    Example - Remote backend (use Client for backend config):
        >>> client = Client("gpt-4", base_url="http://localhost:8080/v1", api_key="sk-...")
        >>> chat = client.chat()
        >>> response = chat("Hello!")

    Example - Local backend with GPU offload (use Client for hardware config):
        >>> client = Client("Qwen/Qwen3-0.6B", gpu_layers=20, num_threads=4)
        >>> chat = client.chat()

    Example - Multi-turn conversation:
        >>> response = chat("What is Python?")
        >>> response = response.append("What is it used for?")
        >>> response = response.append("Give me an example")

    Example - Streaming:
        >>> chat = Chat("Qwen/Qwen3-0.6B")
        >>> response = chat("Tell me a story", stream=True)
        >>> for token in response:
        ...     print(token, end="", flush=True)

    Example - Multi-user serving:
        >>> client = Client("Qwen/Qwen3-0.6B")
        >>> user1 = client.chat(system="You are helpful.")
        >>> user2 = client.chat(system="You are a pirate.")
        >>> response = user1("Hello!")
        >>> response = user2("Ahoy!")

    Example - Using GenerationConfig:
        >>> config = GenerationConfig(temperature=0.7, max_tokens=100)
        >>> chat = Chat("model", config=config)
        >>> print(chat.config.temperature)  # 0.7
        >>> chat.send("Solve: 2+2")  # Uses temp=0.7 automatically

    Example - Per-call overrides with kwargs (preferred):
        >>> chat = Chat("model", config=GenerationConfig(temperature=0.7))
        >>> chat.send("Solve math", temperature=0.1)  # Uses 0.1 for this call only

    Example - Per-call overrides with config object:
        >>> chat.send("Complex task", config=GenerationConfig(top_k=20))

    Example - Combined overrides (kwargs win):
        >>> chat.send("Hello", config=GenerationConfig(temperature=0.5), temperature=0.1)
        >>> # Uses temperature=0.1 (kwargs override config parameter)

    Example - Message access:
        >>> chat = Chat(system="You are helpful.")
        >>> chat.items[0]  # Access system prompt item
        MessageItem(role='system', content=[...])
        >>> chat.items[0].text  # Get text content
        'You are helpful.'
        >>> chat.clear()  # Clear conversation (keeps system prompt)
        >>> chat.reset()  # Reset everything including system prompt
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        client: Client | None = None,
        config: GenerationConfig | None = None,
        system: str | None = None,
        profile: Profile | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        group_id: str | None = None,
        ttl_ts: int | None = None,
        marker: str = "",
        metadata: dict | None = None,
        source_doc_id: str | None = None,
        prompt_id: str | None = None,
        chat_template: str | PromptTemplate | None = None,
        storage: Database | None = None,
        offline: bool = False,
        _defer_session_update: bool = False,
    ):
        # Validate model/client args
        if model is not None and client is not None:
            from ...exceptions import ValidationError

            raise ValidationError("Provide either 'model' or 'client', not both")

        if profile is not None:
            if storage is not None:
                from ...exceptions import ValidationError

                raise ValidationError("Cannot use both 'profile' and 'storage'")
            storage = profile.database
            if session_id is None:
                session_id = new_session_id()

        # Set up client/router before calling _init_base
        self._client: Client | AsyncClient | None = None
        self._router: Router | None = None
        self._owns_client = False
        self._model_id: str | None = None

        if model is not None:
            from talu.client import Client

            self._client = Client(model)
            self._router = self._client._router
            self._owns_client = True
            self._model_id = model
        elif client is not None:
            self._client = client
            self._router = client._router
            self._owns_client = False
            self._model_id = client.default_model

        # Initialize common base state
        self._init_base(
            system=system,
            session_id=session_id,
            parent_session_id=parent_session_id,
            group_id=group_id,
            ttl_ts=ttl_ts,
            marker=marker,
            metadata=metadata,
            source_doc_id=source_doc_id,
            prompt_id=prompt_id,
            chat_template=chat_template,
            storage=storage,
            config=config,
            offline=offline,
            _defer_session_update=_defer_session_update,
        )
        self._tool_registry: dict[str, Callable[..., Any]] | None = None
        self._tool_config: GenerationConfig | None = None
        self._tool_response_format: type | dict | Grammar | None = None

    def close(self) -> None:
        """
        Close the chat and release resources immediately.

        If this Chat created its own internal Client (via model="..."),
        the Client and its Engine are closed, freeing memory.

        If this Chat uses a shared Client (via client=...), only the
        lightweight chat state is freed. The Client stays alive.

        Safe to call multiple times.

        Example - Explicit cleanup in loops:
            >>> for model in ["Qwen/0.5B", "Qwen/1.5B", "Qwen/4B"]:
            ...     chat = Chat(model)
            ...     print(chat("Hello"))
            ...     chat.close()  # Free memory before loading next model

        Example - Context manager (preferred):
            >>> with Chat("Qwen/0.5B") as chat:
            ...     print(chat("Hello"))
            ... # Memory freed automatically here
        """
        # 1. Free the Zig Chat handle (lightweight state)
        if hasattr(self, "_chat_ptr") and self._chat_ptr:
            self._lib.talu_chat_free(self._chat_ptr)
            self._chat_ptr = None
            self._conversation_ptr = None  # Owned by Chat, now invalid

        # 2. Close the Client ONLY if we created it (model="...")
        # If the user passed client=..., they manage it themselves.
        if hasattr(self, "_owns_client") and self._owns_client and self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Chat:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, closing resources."""
        self.close()

    def __del__(self) -> None:
        """Fallback cleanup if close() wasn't called."""
        # Wrap in try/except because globals might be gone on interpreter shutdown
        try:
            self.close()
        except Exception:
            pass

    def _build_response_from_result(
        self,
        result: dict,
        effective_config: GenerationConfig,
        *,
        _stream_mode: bool = False,
        _response_format: type | dict | Grammar | None = None,
        _prompt: str | None = None,
    ) -> Response:
        """Build a Response from router generation result."""
        from .._generate import build_response

        return build_response(
            self,
            result,
            effective_config,
            Response,
            _stream_mode=_stream_mode,
            _response_format=_response_format,
            _prompt=_prompt,
        )

    # =========================================================================
    # Primary API: __call__ (streaming default), send (non-streaming default)
    # For async operations, use AsyncChat instead.
    # =========================================================================

    @overload
    def __call__(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: Literal[True] = True,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> StreamingResponse: ...

    @overload
    def __call__(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: Literal[False],
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response: ...

    def __call__(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """
        Send a message and get a streaming response (callable syntax).

        This is the primary way to chat. Call the Chat object directly
        with your message. By default, returns a StreamingResponse for
        real-time token display. Use ``stream=False`` for complete response.

        For async usage, use ``send_async()`` instead.

        Args:
            message: The user's message. Can be:
                - A string for simple text messages
                - A list of content parts for multimodal input:
                  [{"type": "text", "text": "..."}, {"type": "image", "data": "...", "mime": "image/png"}]
            config: Generation configuration override for this call only.
                Includes structured output settings (schema_strategy, inject_schema_prompt,
                allow_thinking, max_thinking_tokens).
            stream: If True (default), returns StreamingResponse with tokens arriving
                incrementally. This provides immediate feedback and matches
                industry standard for chat interfaces (ChatGPT, Claude, etc.).

                If False, returns Response after generation completes.
                ``response.text`` is immediately available.

                **Why stream=True is default:**

                Streaming provides real-time feedback as tokens arrive, which:

                - Reduces perceived latency (users see progress immediately)
                - Prevents confusion about "hanging" during long generations
                - Long generations (10+ seconds) with no output appear broken
                - Matches industry standard for chat interfaces

                **Use stream=True for:**

                - Interactive applications (CLIs, chat interfaces)
                - Long generations where users want real-time feedback
                - Applications showing progress indicators
                - Reducing perceived latency for user-facing apps

                **Use stream=False for:**

                - Batch processing (collect all responses at once)
                - Simple scripts where you don't need incremental tokens
                - API endpoints returning JSON with full text
                - Testing/automation where latency doesn't matter
                - Cases requiring deterministic timing

                **Important:** StreamingResponse is single-use. Once exhausted,
                you cannot iterate again. Access ``.text`` after iteration for
                the full accumulated text.

            on_token: Optional callback called for each token (streaming only).
            response_format: Dataclass type or JSON schema dict for structured output.
                When provided, the model output will be constrained to match the schema.
                Use ``response.parsed`` to get a hydrated dataclass instance.
            **kwargs: Individual parameter overrides (temperature, max_tokens, etc.)
                for this call only. Does NOT modify ``chat.config``.

        Returns
        -------
            StreamingResponse: If stream=True (default). **Single-use iterator.**
                Cache tokens during iteration if needed later. Access ``.text``
                after exhaustion for the full accumulated text.
            Response: If stream=False. Complete response with ``.text`` always
                available immediately.

        Raises
        ------
            StateError: If no router is available (Chat created without model/client).
            ValidationError: If an unknown generation parameter is passed.
            StructuredOutputError: If response_format schema setup fails.

        Configuration Precedence:
            Per-call overrides do NOT mutate session state. Priority (high to low):

            1. **kwargs (e.g., ``temperature=0.1``) - this call only
            2. config parameter - this call only
            3. chat.config - session default (unchanged by per-call overrides)

            To permanently change session config: ``chat.config = GenerationConfig(...)``

        Example - Streaming (default):
            >>> chat = Chat("Qwen/Qwen3-0.6B")
            >>> for token in chat("Tell me a joke"):
            ...     print(token, end="", flush=True)

        Example - Non-streaming:
            >>> response = chat("What is 2+2?", stream=False)
            >>> print(response)
            4

        Example - Structured output:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Answer:
            ...     value: int
            >>> response = chat("What is 2+2?", response_format=Answer, stream=False)
            >>> response.parsed.value
            4

        Example - Per-call override (session unchanged):
            >>> chat = Chat("model", config=GenerationConfig(temperature=0.7))
            >>> response = chat("Hi", temperature=0.1)  # Uses 0.1 for this call
            >>> chat.config.temperature  # Still 0.7 (unchanged)
            0.7

        Example - Multi-turn:
            >>> response = chat("What is 2+2?", stream=False)
            >>> response = response.append("Why?")  # Inherits stream=False
        """
        return self._generate_sync(
            message,
            config=config,
            stream=stream,
            on_token=on_token,
            response_format=response_format,
            **kwargs,
        )

    def _generate_sync(
        self,
        message: str
        | list[dict]
        | MessageItem
        | list[MessageItem]
        | MessageItem
        | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """Generate a response synchronously (internal)."""
        from talu.types import normalize_message_input

        from .._generate import extract_json_from_response, prepare_generation

        if self._router is None:
            raise StateError(
                "No router available. Create Chat with model='...' or client=Client(...)",
                code="STATE_NO_ROUTER",
            )

        message = normalize_message_input(message)
        ctx = prepare_generation(self, message, config, response_format, **kwargs)

        response: Response | StreamingResponse
        if stream:
            stream_message = message
            if isinstance(message, str) and ctx.schema_prompt:
                if SCHEMA_PLACEHOLDER in message:
                    stream_message = message.replace(SCHEMA_PLACEHOLDER, ctx.schema_prompt)
                else:
                    stream_message = f"{ctx.schema_prompt}\n\n{message}"

            stream_iterator = self._router.stream(
                self, user_message=stream_message, config=ctx.effective_config
            )
            if ctx.grammar_cleanup is not None:
                base_iterator = stream_iterator
                cleanup = ctx.grammar_cleanup

                def _wrapped_stream() -> Iterator:
                    try:
                        yield from base_iterator
                    finally:
                        cleanup()

                stream_iterator = _wrapped_stream()

            response = StreamingResponse(
                stream_iterator=stream_iterator,
                chat=self,
                on_token=on_token,
                on_complete=ctx.notify_storage,
                metadata=ResponseMetadata(
                    finish_reason="stop",
                    schema_tokens=ctx.schema_tokens,
                    schema_injection=ctx.schema_prompt,
                    prefill_success=ctx.prefill_prefix is not None,
                ),
                _response_format=ctx.actual_response_format,
                _stream_mode=True,
                _hooks=ctx.hooks,
                _generation_start_time=ctx.generation_start_time,
            )
        else:
            generation_error: BaseException | None = None
            try:
                try:
                    if ctx.use_submit and ctx.messages_for_submit is not None:
                        result = self._router.submit(
                            ctx.messages_for_submit,
                            config=ctx.effective_config,
                            response_format=ctx.actual_response_format,
                            allow_thinking=ctx.allow_thinking,
                            max_thinking_tokens=ctx.max_thinking_tokens,
                            stop_tokens=ctx.stop_tokens,
                            prefill_prefix=ctx.prefill_prefix,
                        )
                    else:
                        submit_message = message
                        if (
                            isinstance(message, str)
                            and response_format is not None
                            and ctx.inject_schema_prompt
                            and ctx.messages_for_submit is not None
                        ):
                            submit_message = ctx.messages_for_submit[-1]["content"]
                        result = self._router.generate(
                            self, user_message=submit_message, config=ctx.effective_config
                        )
                finally:
                    semantic_error: str | None = None
                    if ctx.grammar_cleanup is not None:
                        semantic_error = ctx.grammar_cleanup()

                if semantic_error is not None:
                    raw_text = str(result.get("text", "")) if result else ""
                    json_text = extract_json_from_response(raw_text)
                    raise SchemaValidationError(
                        raw_text=json_text,
                        validation_error=ValueError(semantic_error),
                    )
                if ctx.schema_prompt:
                    result["schema_injection"] = ctx.schema_prompt
                    result["schema_tokens"] = ctx.schema_tokens
                if ctx.prefill_prefix is not None:
                    result["prefill_success"] = True

                rendered_prompt: str | None = None
                try:
                    rendered_prompt = self.preview_prompt(add_generation_prompt=False)
                except (StateError, RuntimeError):
                    pass

                response = self._build_response_from_result(
                    result,
                    ctx.effective_config,
                    _stream_mode=False,
                    _response_format=ctx.actual_response_format,
                    _prompt=rendered_prompt,
                )
                ctx.notify_storage(response.text)
            except BaseException as e:
                generation_error = e
                raise
            finally:
                if ctx.hooks:
                    resp = response if generation_error is None else None  # type: ignore[possibly-undefined]
                    err = generation_error if isinstance(generation_error, Exception) else None
                    ctx.hooks.dispatch_end(self, resp, err)

        self._last_response = response  # type: ignore[possibly-undefined]
        return response  # type: ignore[possibly-undefined]

    @overload
    def send(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        tools: list[Callable[..., Any]] | None = None,
        stream: Literal[False] = False,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @overload
    def send(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        tools: list[Callable[..., Any]] | None = None,
        stream: Literal[True],
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> StreamingResponse: ...

    def send(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        tools: list[Callable[..., Any]] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """
        Send a message and get a response (synchronous).

        This is the explicit sync method. For streaming default, use ``chat()``.
        For async, use ``send_async()``.

        Args:
            message: The user's message. Can be:
                - A string for simple text messages
                - A list of content parts for multimodal input:
                  [{"type": "text", "text": "..."}, {"type": "image", "data": "...", "mime": "image/png"}]
            config: Generation configuration override. Includes structured output settings
                (schema_strategy, inject_schema_prompt, allow_thinking, max_thinking_tokens).
            tools: Optional list of @tool-decorated functions to enable tool calling.
            stream: If True, returns StreamingResponse. If False (default), returns Response.
            on_token: Optional callback called for each token (streaming only).
            response_format: Dataclass type or JSON schema dict for structured output.
            **kwargs: Individual parameter overrides (temperature, max_tokens, etc.).

        Returns
        -------
            Response: If stream=False (default).
            StreamingResponse: If stream=True.

        Raises
        ------
            StateError: If no router is available (Chat created without model/client).
            ValidationError: If an unknown generation parameter is passed.
            StructuredOutputError: If response_format schema setup fails.

        Example:
            >>> response = chat.send("What is 2+2?")
            >>> print(response)
            4
            >>> response = response.append("Why?")  # Continues with same mode
        """
        if tools and stream:
            from ...exceptions import ValidationError

            raise ValidationError(
                "Tool calling is not supported with stream=True. "
                "Use chat.send(..., stream=False, tools=[...]).",
                code="INVALID_ARGUMENT",
                details={"param": "stream"},
            )

        tools_registry: dict[str, Callable[..., Any]] | None = None
        tools_json: str | None = None
        if tools:
            from ...exceptions import ValidationError

            seen: set[str] = set()
            tool_defs: list[dict[str, Any]] = []
            tools_registry = {}
            for func in tools:
                if not getattr(func, "_is_tool", False):
                    raise ValidationError(
                        "All tools must be decorated with @tool.",
                        code="INVALID_ARGUMENT",
                        details={"tool": repr(func)},
                    )
                schema = getattr(func, "_tool_schema", None)
                tool_func = getattr(func, "_tool_func", None)
                if not isinstance(schema, dict) or tool_func is None:
                    raise ValidationError(
                        "Tool definition missing schema or function.",
                        code="INVALID_ARGUMENT",
                        details={"tool": repr(func)},
                    )
                name = schema.get("name")
                if not isinstance(name, str) or not name:
                    raise ValidationError(
                        "Tool schema must include a valid name.",
                        code="INVALID_ARGUMENT",
                        details={"tool": repr(func)},
                    )
                if name in seen:
                    raise ValidationError(
                        f"Duplicate tool name '{name}'.",
                        code="INVALID_ARGUMENT",
                        details={"name": name},
                    )
                seen.add(name)
                tool_defs.append({"type": "function", "function": schema})
                tools_registry[name] = tool_func

            import json as json_mod

            tools_json = json_mod.dumps(tool_defs)

        # Build effective config to check validation_retries
        effective_config = self._build_effective_config(config, **kwargs)
        if tools_json is not None:
            effective_config = effective_config.override(tools_json=tools_json)
        retries_remaining = effective_config.validation_retries

        if tools_registry is not None:
            self._tool_registry = tools_registry
            self._tool_config = effective_config
            self._tool_response_format = response_format
        else:
            self._tool_registry = None
            self._tool_config = None
            self._tool_response_format = None

        policy_handle = None
        if tools_registry is not None:
            from .._policy import _PolicyHandle, build_tool_policy

            policy_handle = _PolicyHandle(build_tool_policy(tools_registry.keys()))
            policy_handle.attach(self._chat_ptr)

        def _inject_tool_registry(resp: Response | StreamingResponse) -> None:
            if tools_registry is None:
                return
            tool_calls = getattr(resp, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    call._func = tools_registry.get(call.name)
            resp._tool_registry = tools_registry  # type: ignore[attr-defined]

        # Fast path: no retries or streaming (can't validate until stream exhausted)
        if retries_remaining == 0 or stream or response_format is None:
            try:
                response = self._generate_sync(
                    message,
                    config=effective_config,
                    stream=stream,
                    on_token=on_token,
                    response_format=response_format,
                )
                _inject_tool_registry(response)
                return response
            finally:
                if policy_handle is not None:
                    policy_handle.close()

        # Retry loop for validation errors
        last_error: SchemaValidationError | None = None
        try:
            while True:
                try:
                    response = self._generate_sync(
                        message,
                        config=effective_config,
                        stream=False,
                        on_token=on_token,
                        response_format=response_format,
                    )
                except SchemaValidationError as e:
                    # Semantic validation error from Zig core
                    last_error = e
                    if retries_remaining <= 0:
                        raise  # No retries left

                    # Format error message for self-correction
                    error_msg = self._format_validation_retry_message(e)
                    retries_remaining -= 1

                    # Send error as user message to trigger retry
                    message = error_msg
                    continue

                # stream=False guarantees Response (not StreamingResponse)
                assert isinstance(response, Response)

                # Try to parse/validate the response (Pydantic validation)
                try:
                    _ = response.parsed  # Trigger validation
                    _inject_tool_registry(response)
                    return response  # Success!
                except SchemaValidationError as e:
                    last_error = e
                    if retries_remaining <= 0:
                        raise  # No retries left

                    # Format error message for self-correction
                    error_msg = self._format_validation_retry_message(e)
                    retries_remaining -= 1

                    # Send error as user message (keeps failed response in history as context)
                    # This triggers a new generation with the same response_format
                    message = error_msg
        finally:
            if policy_handle is not None:
                policy_handle.close()

        # Should never reach here, but satisfy type checker
        if last_error is not None:
            raise last_error
        raise TaluError("Unreachable", code="INTERNAL_ERROR")  # pragma: no cover

    def _append_function_call_output(self, call_id: str, content: str) -> None:
        """Append a function call output item to the conversation."""
        if self._conversation_ptr is None:
            raise StateError(
                "Cannot append tool result: chat is closed.",
                code="STATE_NO_CHAT",
            )
        from .. import _bindings as _c

        result = _c.responses_append_function_call_output(
            self._lib, self._conversation_ptr, call_id, content
        )
        if result < 0:
            raise StateError(
                f"Failed to append tool result: error code {result}",
                code="STATE_APPEND_FAILED",
            )

    def _continue_generation(
        self,
        *,
        tools_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> Response:
        """Continue generation after tool output."""
        if self._router is None:
            raise StateError(
                "No router available. Create Chat with model='...' or client=Client(...)",
                code="STATE_NO_ROUTER",
            )
        tools_registry = tools_registry or self._tool_registry or {}
        policy_handle = None
        if tools_registry:
            from .._policy import _PolicyHandle, build_tool_policy

            policy_handle = _PolicyHandle(build_tool_policy(tools_registry.keys()))
            policy_handle.attach(self._chat_ptr)
        try:
            result = self._generate_sync(
                "",
                config=self._tool_config,
                stream=False,
                response_format=self._tool_response_format,
            )
            assert not isinstance(result, StreamingResponse)  # stream=False
            if result.tool_calls:
                for call in result.tool_calls:
                    call._func = tools_registry.get(call.name)
            result._tool_registry = tools_registry  # type: ignore[attr-defined]
            return result
        finally:
            if policy_handle is not None:
                policy_handle.close()

    @overload
    def regenerate(
        self,
        message: str | None = None,
        config: GenerationConfig | None = None,
        *,
        stream: Literal[False] = False,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response: ...

    @overload
    def regenerate(
        self,
        message: str | None = None,
        config: GenerationConfig | None = None,
        *,
        stream: Literal[True],
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> StreamingResponse: ...

    def regenerate(
        self,
        message: str | None = None,
        config: GenerationConfig | None = None,
        *,
        stream: bool = False,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """
        Regenerate the last conversation turn.

        This method unwinds the conversation to the previous user message and
        triggers generation again. Use it to retry a response or edit the last
        user message.

        The operation is atomic: it truncates to the point before the last user
        message and then sends (either the original or new text). This ensures
        fresh item IDs and timestamps for auditability.

        Parameters
        ----------
        message
            Optional new text for the user message.
            If provided: replaces the last user message with this text.
            If None: retries with the existing user message text.
        config
            Generation configuration override.
        stream
            If True, returns StreamingResponse.
        response_format
            Dataclass type or JSON schema dict for structured output.
        **kwargs
            Individual parameter overrides (temperature, max_tokens, etc.)

        Returns
        -------
        Response | StreamingResponse
            The new response from regeneration.

        Raises
        ------
        StateError
            If no user message exists to regenerate from.

        Example
        -------
        >>> chat = Chat("Qwen/Qwen3-0.6B")
        >>> chat("Tell me a joke")
        >>> # Didn't like the joke? Retry:
        >>> chat.regenerate()
        >>> # Or edit and retry:
        >>> chat.regenerate(message="Tell me a better joke")
        >>> # With different parameters:
        >>> chat.regenerate(temperature=1.2)
        """
        # 1. Find the last user message (turn boundary)
        last_user_idx = -1
        for i in range(len(self.items) - 1, -1, -1):
            item = self.items[i]
            if isinstance(item, MessageItem) and item.role == MessageRole.USER:
                last_user_idx = i
                break

        if last_user_idx == -1:
            raise StateError(
                "No user message found to regenerate from",
                code="STATE_REGENERATE_NO_USER",
            )

        # 2. Determine content (new vs existing)
        if message is not None:
            text_to_send = message
        else:
            # Extract text from the user message we're about to delete
            user_item = self.items[last_user_idx]
            text_to_send = user_item.text if isinstance(user_item, MessageItem) else ""

        # 3. Truncate (remove user message and everything after)
        # We truncate to last_user_idx - 1, keeping everything BEFORE the user message
        trunc_target = last_user_idx - 1

        if trunc_target < 0:
            # User message is at index 0 (or 1 with system at 0)
            # Clear everything except system prompt
            self._lib.talu_responses_clear_keeping_system(self._conversation_ptr)
        else:
            self._lib.talu_responses_truncate_after(self._conversation_ptr, trunc_target)

        # 4. Send (appends fresh user message with new ID and generates)
        return self.send(
            text_to_send,
            config=config,
            stream=stream,
            response_format=response_format,
            **kwargs,
        )

    def fork(self) -> Chat:
        """
        Fork this chat to explore alternative conversation paths.

        Creates an independent copy of the chat with the same history,
        config, and client reference. Changes to the forked chat do not
        affect the original.

        Returns
        -------
            New Chat with copied state.

        Raises
        ------
            StateError: If message history cannot be copied.

        Example:
            >>> chat = Chat("Qwen/Qwen3-0.6B")
            >>> response = chat("I have chicken")
            >>>
            >>> # Fork to try different directions
            >>> asian = response.chat.fork()
            >>> italian = response.chat.fork()
            >>>
            >>> asian("Suggest an Asian recipe")
            >>> italian("Suggest an Italian recipe")
            >>>
            >>> # Original unchanged
            >>> print(len(chat.items))  # Same as before forking
        """
        return self._create_fork("talu_responses_clone", (1000,))

    def _fork_at(self, msg_index: int) -> Chat:
        """Fork a chat and keep items up to msg_index (inclusive)."""
        return self._create_fork("talu_responses_clone_prefix", (msg_index, 1000))

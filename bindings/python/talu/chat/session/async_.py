"""
AsyncChat - Async conversational AI interface.

AsyncChat is the async equivalent of Chat. Use it for building async
applications (FastAPI, aiohttp, etc.) where you need non-blocking
generation operations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from talu.router.config import GenerationConfig, Grammar
from talu.types import MessageItem, MessageRole

from ...db import Database
from ...exceptions import (
    SchemaValidationError,
    StateError,
    TaluError,
)
from ...template import PromptTemplate
from .._chat_base import ChatBase
from .._generate import SCHEMA_PLACEHOLDER
from ..response import AsyncStreamingResponse, ResponseMetadata

if TYPE_CHECKING:
    from talu.client import AsyncClient, Client
    from talu.router import Router, StreamToken

    from ..response import AsyncResponse


class AsyncChat(ChatBase):
    """
    Async stateful multi-turn chat session.

    AsyncChat is the async equivalent of Chat. Use it for building async
    applications (FastAPI, aiohttp, etc.) where you need non-blocking
    generation operations.

    All generation methods (``send()``, ``__call__()``) are async and must
    be awaited.

    Separation of Concerns:
        - **AsyncChat** manages session state: conversation history, system prompt, templates
        - **AsyncClient** manages infrastructure: model loading, GPU layers, API keys, threading

        For custom hardware or backend configuration, create an AsyncClient first::

            async with AsyncClient("model", gpu_layers=20, api_key="...") as client:
                chat = client.chat(system="You are helpful.")

    Architecture
    ------------
    AsyncChat shares the same Zig backend as Chat. Model weights are cached
    globally, so creating AsyncChat for the same model as an existing Chat
    shares memory efficiently.

    Concurrency:
        Safe to share across asyncio tasks. Not thread-safe across OS threads.
        Each task should maintain its own conversation flow to avoid interleaving.

    Args:
        model: Model to load (HuggingFace ID or local path). Creates a default AsyncClient.
            For custom configuration (GPU layers, API keys, etc.), use AsyncClient instead.
        client: Existing AsyncClient to use (for multi-user serving or custom config).
        config: Default GenerationConfig for this session.
        system: Optional system prompt.
        session_id: Optional session identifier for this conversation.
        parent_session_id: Optional parent session identifier for forks.
        marker: Session marker for storage backends (default: "" = normal/unmarked).
            Values: "pinned", "archived", "deleted", or "" (normal).
        metadata: Optional session metadata dict (tags, UI state, notes).
        chat_template: Custom chat template to use.
        storage: Storage for messages. Use Database("talu://<path>") for TaluDB
            persistence (requires session_id).
        offline: If True, disallow network access when resolving model URIs.

    Example - Basic async usage:
        >>> chat = AsyncChat("Qwen/Qwen3-0.6B", system="You are helpful.")
        >>> response = await chat("What is 2+2?")
        >>> print(response)

    Example - Remote backend (use AsyncClient for backend config):
        >>> async with AsyncClient("gpt-4", base_url="http://localhost:8080/v1", api_key="sk-...") as client:
        ...     chat = client.chat()
        ...     response = await chat("Hello!")

    Example - Multi-turn async conversation:
        >>> response = await chat("Hello!")
        >>> response = await response.append("Tell me more")

    Example - Async streaming:
        >>> chat = AsyncChat("Qwen/Qwen3-0.6B")
        >>> response = await chat("Tell me a story", stream=True)
        >>> async for chunk in response:
        ...     print(chunk, end="", flush=True)

    Raises
    ------
        ValidationError: If both ``model`` and ``client`` are provided.
        MemoryError: If AsyncChat creation fails (insufficient memory).

    Example - Multi-user async serving:
        >>> async with AsyncClient("Qwen/Qwen3-0.6B") as client:
        ...     user1 = client.chat(system="You are helpful.")
        ...     user2 = client.chat(system="You are a pirate.")
        ...     response = await user1("Hello!")
        ...     response = await user2("Ahoy!")
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        client: AsyncClient | None = None,
        config: GenerationConfig | None = None,
        system: str | None = None,
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

        # Set up client/router before calling _init_base
        self._client: Client | AsyncClient | None = None
        self._router: Router | None = None
        self._owns_client = False
        self._model_id: str | None = None

        if model is not None:
            from talu.client import AsyncClient

            self._client = AsyncClient(model)
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

    async def close(self) -> None:
        """
        Close the chat and release resources immediately.

        If this AsyncChat created its own internal AsyncClient (via model="..."),
        the Client and its Engine are closed, freeing memory.

        If this AsyncChat uses a shared AsyncClient (via client=...), only the
        lightweight chat state is freed. The Client stays alive.

        Safe to call multiple times.

        Example - Explicit cleanup in loops:
            >>> for model in ["Qwen/0.5B", "Qwen/1.5B", "Qwen/4B"]:
            ...     chat = AsyncChat(model)
            ...     print(await chat("Hello"))
            ...     await chat.close()  # Free memory before loading next model

        Example - Context manager (preferred):
            >>> async with AsyncChat("Qwen/0.5B") as chat:
            ...     print(await chat("Hello"))
            ... # Memory freed automatically here
        """
        # Use sync helper since underlying C-API is sync anyway
        self._close_sync()

    def _close_sync(self) -> None:
        """
        Clean up resources synchronously.

        Used by close() (async) and __del__ (sync).
        The underlying C-API calls are blocking/sync, so this is safe.
        """
        # 1. Free the Zig Chat handle (lightweight state)
        if hasattr(self, "_chat_ptr") and self._chat_ptr:
            self._lib.talu_chat_free(self._chat_ptr)
            self._chat_ptr = None
            self._conversation_ptr = None

        # 2. Close the Client ONLY if we created it (model="...")
        if hasattr(self, "_owns_client") and self._owns_client and self._client:
            # AsyncClient.close() is async, but internally it just calls
            # self._router.close(), which is sync. Access router directly.
            if hasattr(self._client, "_router") and self._client._router:
                self._client._router.close()
            self._client = None

    async def __aenter__(self) -> AsyncChat:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager, closing resources."""
        await self.close()

    def __del__(self) -> None:
        """Fallback cleanup if close() wasn't called."""
        # Wrap in try/except because globals might be gone on interpreter shutdown
        try:
            self._close_sync()
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
    ) -> AsyncResponse:
        """Build an AsyncResponse from router generation result."""
        from .._generate import build_response
        from ..response import AsyncResponse

        return build_response(
            self,
            result,
            effective_config,
            AsyncResponse,
            _stream_mode=_stream_mode,
            _response_format=_response_format,
            _prompt=_prompt,
        )

    @overload
    async def __call__(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: Literal[True] = True,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncStreamingResponse: ...

    @overload
    async def __call__(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: Literal[False],
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse: ...

    async def __call__(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: bool = True,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse | AsyncStreamingResponse:
        """
        Send a message and get an async streaming response (callable syntax).

        This is the primary async way to chat. Call the AsyncChat object directly
        with your message. By default, returns an AsyncStreamingResponse.

        Args:
            message: The user's message.
            config: Generation configuration override. Includes structured output settings
                (schema_strategy, inject_schema_prompt, allow_thinking, max_thinking_tokens).
            stream: If True (default), returns AsyncStreamingResponse.
            on_token: Optional callback called for each token.
            response_format: Dataclass type or JSON schema dict for structured output.
            **kwargs: Individual parameter overrides.

        Returns
        -------
            AsyncStreamingResponse: If stream=True (default).
            AsyncResponse: If stream=False.

        Example:
            >>> response = await chat("Tell me a joke")
            >>> async for token in response:
            ...     print(token, end="", flush=True)
        """
        return await self._generate_async(
            message,
            config=config,
            stream=stream,
            on_token=on_token,
            response_format=response_format,
            **kwargs,
        )

    async def _generate_async(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse | AsyncStreamingResponse:
        """Generate a response asynchronously (internal)."""
        import asyncio

        from talu.types import normalize_message_input

        from .._generate import extract_json_from_response, prepare_generation

        if self._router is None:
            raise StateError(
                "No router available. Create AsyncChat with model='...' or client=AsyncClient(...)",
                code="STATE_NO_ROUTER",
            )

        message = normalize_message_input(message)
        ctx = prepare_generation(self, message, config, response_format, **kwargs)

        response: AsyncResponse | AsyncStreamingResponse
        if stream:
            stream_message = message
            if isinstance(message, str) and ctx.schema_prompt:
                if SCHEMA_PLACEHOLDER in message:
                    stream_message = message.replace(SCHEMA_PLACEHOLDER, ctx.schema_prompt)
                else:
                    stream_message = f"{ctx.schema_prompt}\n\n{message}"

            async_stream_iterator = self._router.stream_async(
                self, user_message=stream_message, config=ctx.effective_config
            )
            if ctx.grammar_cleanup is not None:
                base_iterator = async_stream_iterator
                cleanup = ctx.grammar_cleanup

                async def _wrapped_stream() -> AsyncIterator[StreamToken]:
                    try:
                        async for chunk in base_iterator:
                            yield chunk
                    finally:
                        cleanup()

                async_stream_iterator = _wrapped_stream()

            response = AsyncStreamingResponse(
                async_stream_iterator=async_stream_iterator,
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
                        # Wrap blocking FFI in executor to avoid blocking event loop
                        router = cast("Router", self._router)
                        loop = asyncio.get_event_loop()
                        # Capture for lambda (type narrowing doesn't apply in closures)
                        messages = ctx.messages_for_submit
                        result = await loop.run_in_executor(
                            None,
                            lambda: router.submit(
                                messages,
                                config=ctx.effective_config,
                                response_format=ctx.actual_response_format,
                                allow_thinking=ctx.allow_thinking,
                                max_thinking_tokens=ctx.max_thinking_tokens,
                                stop_tokens=ctx.stop_tokens,
                                prefill_prefix=ctx.prefill_prefix,
                            ),
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

                        generate_async = getattr(self._router, "generate_async", None)
                        if generate_async is not None:
                            result = await generate_async(
                                self, user_message=submit_message, config=ctx.effective_config
                            )
                        else:
                            router = cast("Router", self._router)
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None,
                                lambda: router.generate(
                                    self, user_message=submit_message, config=ctx.effective_config
                                ),
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
                    ctx.hooks.dispatch_end(self, resp, err)  # type: ignore[arg-type]

        self._last_response = response  # type: ignore[possibly-undefined]
        return response  # type: ignore[possibly-undefined]

    @overload
    async def send(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        tools: list[Callable[..., Any]] | None = None,
        stream: Literal[False] = False,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse: ...

    @overload
    async def send(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        tools: list[Callable[..., Any]] | None = None,
        stream: Literal[True],
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncStreamingResponse: ...

    async def send(
        self,
        message: str | list[dict] | MessageItem | list[MessageItem],
        config: GenerationConfig | None = None,
        *,
        tools: list[Callable[..., Any]] | None = None,
        stream: bool = False,
        on_token: Callable[[str], None] | None = None,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse | AsyncStreamingResponse:
        """
        Send a message and get a response (async, non-streaming by default).

        Args:
            message: The user's message.
            config: Generation configuration override. Includes structured output settings
                (schema_strategy, inject_schema_prompt, allow_thinking, max_thinking_tokens).
            tools: Optional list of @tool-decorated functions to enable tool calling.
            stream: If True, returns AsyncStreamingResponse. If False (default), AsyncResponse.
            on_token: Optional callback called for each token.
            response_format: Dataclass type or JSON schema dict for structured output.
            **kwargs: Individual parameter overrides.

        Returns
        -------
            AsyncResponse: If stream=False (default).
            AsyncStreamingResponse: If stream=True.

        Raises
        ------
            StateError: If no router is available (AsyncChat created without model/client).
            ValidationError: If an unknown generation parameter is passed.
            StructuredOutputError: If response_format schema setup fails.

        Example:
            >>> response = await chat.send("What is 2+2?")
            >>> print(response)
            4
            >>> response = await response.append("Why?")
        """
        from ..response import AsyncResponse

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

        def _inject_tool_registry(resp: AsyncResponse | AsyncStreamingResponse) -> None:
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
                response = await self._generate_async(
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
                    response = await self._generate_async(
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

                # stream=False guarantees AsyncResponse (not AsyncStreamingResponse)
                assert isinstance(response, AsyncResponse)

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

    async def _continue_generation(
        self,
        *,
        tools_registry: dict[str, Callable[..., Any]] | None = None,
    ) -> AsyncResponse:
        """Continue generation after tool output."""
        if self._router is None:
            raise StateError(
                "No router available. Create AsyncChat with model='...' or client=AsyncClient(...)",
                code="STATE_NO_ROUTER",
            )
        tools_registry = tools_registry or self._tool_registry or {}
        policy_handle = None
        if tools_registry:
            from .._policy import _PolicyHandle, build_tool_policy

            policy_handle = _PolicyHandle(build_tool_policy(tools_registry.keys()))
            policy_handle.attach(self._chat_ptr)
        try:
            result = await self._generate_async(
                "",
                config=self._tool_config,
                stream=False,
                response_format=self._tool_response_format,
            )
            assert not isinstance(result, AsyncStreamingResponse)  # stream=False
            if result.tool_calls:
                for call in result.tool_calls:
                    call._func = tools_registry.get(call.name)
            result._tool_registry = tools_registry  # type: ignore[attr-defined]
            return result
        finally:
            if policy_handle is not None:
                policy_handle.close()

    @overload
    async def regenerate(
        self,
        message: str | None = None,
        config: GenerationConfig | None = None,
        *,
        stream: Literal[False] = False,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse: ...

    @overload
    async def regenerate(
        self,
        message: str | None = None,
        config: GenerationConfig | None = None,
        *,
        stream: Literal[True],
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncStreamingResponse: ...

    async def regenerate(
        self,
        message: str | None = None,
        config: GenerationConfig | None = None,
        *,
        stream: bool = False,
        response_format: type | dict | Grammar | None = None,
        **kwargs: Any,
    ) -> AsyncResponse | AsyncStreamingResponse:
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
            If True, returns AsyncStreamingResponse.
        response_format
            Dataclass type or JSON schema dict for structured output.
        **kwargs
            Individual parameter overrides (temperature, max_tokens, etc.)

        Returns
        -------
        AsyncResponse | AsyncStreamingResponse
            The new response from regeneration.

        Raises
        ------
        StateError
            If no user message exists to regenerate from.

        Example
        -------
        >>> chat = AsyncChat("Qwen/Qwen3-0.6B")
        >>> await chat("Tell me a joke")
        >>> # Didn't like the joke? Retry:
        >>> await chat.regenerate()
        >>> # Or edit and retry:
        >>> await chat.regenerate(message="Tell me a better joke")
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
        return await self.send(
            text_to_send,
            config=config,
            stream=stream,
            response_format=response_format,
            **kwargs,
        )

    def fork(self) -> AsyncChat:
        """
        Fork this chat to explore alternative conversation paths.

        Returns
        -------
            New AsyncChat with copied state.

        Raises
        ------
            StateError: If message copying fails.
        """
        return self._create_fork("talu_responses_clone", (1000,))

    def _fork_at(self, msg_index: int) -> AsyncChat:
        """Fork a chat and keep items up to msg_index (inclusive)."""
        return self._create_fork("talu_responses_clone_prefix", (msg_index, 1000))

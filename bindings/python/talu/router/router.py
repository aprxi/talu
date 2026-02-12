"""
Model routing and endpoint configuration.

Holds model targets and optional custom endpoints for generation requests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from talu.types import ContentType, ItemType

from ..exceptions import GenerationError, StateError, TaluError, ValidationError
from . import _bindings as _c
from ._bindings import TaluInferenceBackendHandle, get_spec_lib
from .spec import ModelSpec, normalize_to_handle

if TYPE_CHECKING:
    from talu.chat.session import AsyncChat, Chat

    from ._bindings import StopFlag
    from .config.generation import GenerationConfig


class StreamToken:
    """Streamed token with content classification metadata.

    Each token carries ``text``, ``item_type``, and ``content_type``
    discriminators from the responses type system, enabling correct SSE
    event routing and display.

    .. note:: Why this lives in ``router``, not ``talu.types``

       StreamToken is an internal wire format between Router (producer) and
       Chat's streaming response (consumer).  It is never persisted, never
       user-facing (users see ``Token``), and is constructed only inside
       ``Router.stream()``/``Router.astream()``.  ``talu.types`` holds
       domain concepts (Items, Records, Events); StreamToken is a transient
       protocol detail that belongs with the code that constructs it.

    Attributes
    ----------
        text: Decoded token text.
        item_type: Item type (e.g. ``ItemType.MESSAGE``, ``ItemType.REASONING``).
        content_type: Content type (e.g. ``ContentType.OUTPUT_TEXT``,
            ``ContentType.REASONING_TEXT``).
    """

    __slots__ = ("text", "item_type", "content_type")

    def __init__(self, text: str, item_type: int, content_type: int) -> None:
        self.text = text
        self.item_type = ItemType(item_type)
        self.content_type = ContentType(content_type)

    def __repr__(self) -> str:
        return (
            f"StreamToken({self.text!r}, "
            f"item_type={self.item_type.name}, "
            f"content_type={self.content_type.name})"
        )

    def __str__(self) -> str:
        return self.text


@dataclass
class ModelTarget:
    """
    Model target with optional custom endpoint.

    Attributes
    ----------
        model: Model identifier (e.g., "Qwen/Qwen3-0.6B", "openai::gpt-4o").
        endpoint: Optional custom endpoint URL (overrides provider defaults).
        spec: The ModelSpec used to create this target.
    """

    model: str
    endpoint: str | None = None
    spec: ModelSpec | None = field(default=None, repr=False)
    backend_handle: TaluInferenceBackendHandle | None = field(default=None, init=False, repr=False)


class Router:
    """
    Routes generation requests to models.

    Router holds model targets (names + optional endpoints) and submits
    generation requests. This is typically created by Client, not by users directly.

    Concurrency:
        Not thread-safe. Create one Router per thread, or use Client which
        manages Router instances internally.

    Example:
        >>> # Usually created via Client
        >>> client = Client("Qwen/Qwen3-0.6B")
        >>> # client._router is the Router instance
    """

    def __init__(
        self,
        models: list[str] | list[ModelTarget] | list[ModelSpec] | list[str | ModelSpec],
        default_model: str | None = None,
    ):
        """
        Initialize Router with model targets.

        Args:
            models: List of model identifiers, ModelTarget instances, or ModelSpec.
            default_model: The default model to use when none specified.
                Defaults to the first model in the list.

        Raises
        ------
            ValidationError: If no models provided, invalid model type, or default
                model not in models list.
        """
        if not models:
            raise ValidationError("At least one model must be provided")

        # Get cached Zig library with configured signatures
        from ._bindings import get_router_lib

        self._lib = get_router_lib()
        self._spec_lib = get_spec_lib()

        # Normalize to ModelTarget instances (backend handles created lazily)
        self._targets: dict[str, ModelTarget] = {}
        self._canonical_handles: list[Any] = []  # Keep canonical handles for cleanup
        self._backend_handles: list[
            TaluInferenceBackendHandle
        ] = []  # Track backend handles for cleanup

        for m in models:
            if isinstance(m, str):
                spec = ModelSpec(ref=m)
            elif isinstance(m, ModelSpec):
                spec = m
            elif isinstance(m, ModelTarget):
                # Already a ModelTarget, but may not have engine
                if m.spec is None:
                    spec = ModelSpec(ref=m.model)
                else:
                    spec = m.spec
            else:
                raise ValidationError(f"Invalid model type: {type(m)}")

            # Store target with spec but no backend handle yet (lazy creation)
            target = ModelTarget(
                model=spec.ref,
                endpoint=m.endpoint if isinstance(m, ModelTarget) else None,
                spec=spec,
            )
            self._targets[target.model] = target

        # Set default model
        if default_model is None:
            self._default_model = list(self._targets.keys())[0]
        elif default_model not in self._targets:
            raise ValidationError(f"Default model '{default_model}' not in models")
        else:
            self._default_model = default_model

        self._closed = False

        # Track active generation threads for safe shutdown
        import threading

        self._active_threads: set[threading.Thread] = set()
        self._active_threads_lock = threading.Lock()

        # Track active iterators for safe shutdown (sync stream calls)
        self._active_iterators: set[int] = set()  # Iterator handle addresses
        self._active_iterators_lock = threading.Lock()
        self._iterator_done = threading.Condition(self._active_iterators_lock)

    def _is_external_api(self, model: str) -> bool:
        """Check if model is an external API (openai::, vllm::, etc.).

        The :: separator identifies the backend. native:: is the native
        backend (talu's inference engine), not an external API.
        """
        if "::" not in model:
            return False
        # native:: is the native backend, not external API
        if model.startswith("native::"):
            return False
        return True

    @property
    def default_model(self) -> str:
        """The default model used when none is specified."""
        return self._default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        """Set the default model."""
        if model not in self._targets:
            raise ValidationError(
                f"Model '{model}' not available. Available: {list(self._targets.keys())}"
            )
        self._default_model = model

    @property
    def models(self) -> list[str]:
        """List of available model names."""
        return list(self._targets.keys())

    def get_endpoint(self, model: str | None = None) -> str | None:
        """Get custom endpoint for a model, if any.

        Args:
            model: Model name, or None for the default model.
        """
        model = model or self._default_model
        if model not in self._targets:
            return None
        return self._targets[model].endpoint

    def set_endpoint(self, model: str, endpoint: str | None) -> None:
        """
        Set custom endpoint for a model.

        Args:
            model: Model identifier.
            endpoint: Custom endpoint URL, or None to use default.

        Raises
        ------
            ValidationError: If model is not available.
        """
        if model not in self._targets:
            raise ValidationError(f"Model '{model}' not available")
        self._targets[model].endpoint = endpoint

    def _cleanup_handles(self) -> None:
        """Free all backend and canonical handles."""
        # Free backend handles
        for backend_handle in self._backend_handles:
            if backend_handle:
                self._spec_lib.talu_backend_free(backend_handle)
        self._backend_handles.clear()

        # Free canonical handles
        for canonical_handle in self._canonical_handles:
            if canonical_handle:
                self._spec_lib.talu_config_free(canonical_handle)
        self._canonical_handles.clear()

    def _get_or_create_backend(self, model: str) -> TaluInferenceBackendHandle:
        """Get or lazily create an inference backend handle for a model.

        Backend handles are created on first use to avoid validation errors
        during Router construction with non-existent model paths (e.g., in tests).
        """
        target = self._targets[model]

        # Return existing handle if already created
        if target.backend_handle is not None:
            return target.backend_handle

        # Create backend from spec
        spec = target.spec
        if spec is None:
            raise ValidationError(f"No spec for model '{model}'")

        # Canonicalize the spec
        canonical_handle = normalize_to_handle(spec)
        self._canonical_handles.append(canonical_handle)

        # Create backend from canonical spec
        try:
            backend_handle = _c.router_create_backend(self._spec_lib, canonical_handle)
        except TaluError as e:
            # Re-raise with model context
            raise type(e)(str(e) + f" (model: {spec.ref})", code=e.code) from e

        # Store for reuse and cleanup
        target.backend_handle = backend_handle
        self._backend_handles.append(backend_handle)

        return backend_handle

    def _check_closed(self) -> None:
        """Raise if router has been closed."""
        if self._closed:
            raise StateError(
                "Router has been closed. Create a new Router instance.",
                code="STATE_CLOSED",
            )

    def _resolve_model(self, model: str | None = None) -> str:
        """Resolve model name, defaulting if None."""
        self._check_closed()
        model = model or self._default_model
        if model not in self._targets:
            raise ValidationError(
                f"Model '{model}' not available. Available: {list(self._targets.keys())}"
            )
        return model

    def _resolve_chat_template(self, chat_template: Any) -> str | None:
        """Convert chat_template (PromptTemplate | str | None) to string for C API."""
        if chat_template is None:
            return None
        if isinstance(chat_template, str):
            return chat_template
        # PromptTemplate - extract the source template string
        if hasattr(chat_template, "source"):
            return chat_template.source
        # Fallback: try to get template string
        return str(chat_template)

    def generate(
        self,
        chat: Chat | AsyncChat,
        user_message: str | list[dict],
        config: GenerationConfig | None = None,
        model: str | None = None,
        stop_flag: StopFlag | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response for a Chat.

        Submits request to Zig C API. Zig handles:
        - Adding user_message to Messages
        - Routing to correct engine
        - Running inference
        - Adding assistant response to Messages

        Args:
            chat: The Chat instance with conversation history.
            user_message: The user's message. Can be:
                - A string for simple text messages
                - A list of content parts for multimodal input (Open Responses format):
                  [{"type": "input_text", "text": "..."},
                   {"type": "input_image", "image_url": "data:image/png;base64,..."}]
                  Or use InputImage/InputAudio/InputVideo classes and normalize_content().
            config: Generation configuration.
            model: Model to use, or None for default.
            stop_flag: Optional StopFlag for cancellation. When signalled, Zig stops
                generation gracefully on its next decode loop iteration.

        Returns
        -------
            dict with 'text', 'token_count', 'prompt_tokens', 'completion_tokens',
            'prefill_ns', 'generation_ns'.

        Raises
        ------
            GenerationError: If generation fails. Error message includes detailed
                context from Zig (model path, specific failure reason, etc.).
            ValidationError: If the specified model is not found in targets.
            StateError: If the router has been closed.
        """
        from ._bindings import RouterGenerateConfig

        model = self._resolve_model(model)

        # Build C config struct from GenerationConfig
        c_config = None
        if config is not None or stop_flag is not None:
            # Convert chat_template to string (can be PromptTemplate or str)
            chat_template_str = (
                self._resolve_chat_template(config.chat_template) if config else None
            )
            c_config = RouterGenerateConfig(
                max_tokens=config.max_tokens if config else 0,
                temperature=config.temperature if config else -1.0,
                top_k=config.top_k if config else 0,
                top_p=config.top_p if config else -1.0,
                min_p=config.min_p if config else -1.0,
                repetition_penalty=config.repetition_penalty if config else 0.0,
                stop_sequences=config.stop_sequences if config else None,
                logit_bias=config.logit_bias if config else None,
                seed=(config.seed if config.seed is not None else 0) if config else 0,
                chat_template=chat_template_str,
                extra_context=config.extra_context if config else None,
                tools_json=config.tools_json if config else None,
                tool_choice=config.tool_choice if config else None,
                stop_flag=stop_flag,
                extra_body=config.extra_body if config else None,
            )

        # Convert string to content parts if needed
        if isinstance(user_message, str):
            parts_list = [{"type": "text", "text": user_message}]
        else:
            # Warn if non-text content types are used (not yet supported for inference)
            self._warn_unsupported_content_types(user_message)
            parts_list = user_message

        # Build content parts and call generate with backend-based API
        parts, data_refs = _c.build_router_content_parts(parts_list)

        # Get or create backend handle for this model (lazy initialization)
        backend_handle = self._get_or_create_backend(model)

        result = _c.router_generate_with_backend(
            self._lib,
            chat._chat_ptr,
            parts,
            len(parts_list),
            backend_handle,
            c_config,
        )
        # Keep data_refs alive until after the call
        del data_refs

        # Check for errors
        if result.error_code != 0:
            from talu._bindings import get_last_error

            # Get detailed error from Zig (includes context like model path, etc.)
            zig_error = get_last_error()
            if zig_error:
                raise GenerationError(
                    f"Router.generate() failed: {zig_error}",
                    code="GENERATION_FAILED",
                )

            # Fallback to generic messages if Zig didn't set an error
            error_messages = {
                -1: "Invalid chat handle",
                -2: "Invalid user message",
                -3: "Invalid model",
                -4: "Engine creation failed",
                -5: "Failed to add user message",
                -6: "Generation failed",
                -8: "Memory allocation failed",
                -10: f"External API model '{model}' not yet supported",
            }
            msg = error_messages.get(result.error_code, f"Unknown error: {result.error_code}")
            raise GenerationError(
                f"Router.generate() failed: {msg}",
                code="GENERATION_FAILED",
                details={"error_code": result.error_code},
            )

        # Extract result
        text = _c.router_result_extract_text(result)
        token_count = result.token_count
        prompt_tokens = result.prompt_tokens
        completion_tokens = result.completion_tokens
        prefill_ns = result.prefill_ns
        generation_ns = result.generation_ns
        tool_calls_raw = _c.router_result_extract_tool_calls(result)
        finish_reason = {
            0: "stop",
            1: "length",
            2: "stop_sequence",
            3: "tool_calls",
            4: "content_filter",
            5: "cancelled",
        }.get(result.finish_reason)

        # Free result memory
        _c.router_result_free(self._lib, result)

        tool_calls = None
        if tool_calls_raw:
            from talu.chat.tools import ToolCall

            tool_calls = [
                ToolCall.create(call["id"], call["name"], call["arguments"])
                for call in tool_calls_raw
            ]

        # Return result dict (will be wrapped by Chat into Response)
        return {
            "text": text,
            "token_count": token_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_ns": prefill_ns,
            "generation_ns": generation_ns,
            "finish_reason": finish_reason,
            "tool_calls": tool_calls,
        }

    def submit(
        self,
        messages: list[dict],
        config: GenerationConfig | None = None,
        response_format: type | dict | None = None,
        *,
        chat: Chat | None = None,
        model: str | None = None,
        stop_flag: StopFlag | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Submit a prepared message list for generation.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            config: Generation configuration.
            response_format: Dataclass type or JSON schema dict for structured output.
            chat: The Chat instance (required for native backend).
            model: Model to use, or None for default.
            stop_flag: Optional StopFlag for cancellation.

        Returns
        -------
            dict with 'text', 'token_count', 'prompt_tokens', 'completion_tokens',
            'prefill_ns', 'generation_ns'.

        Raises
        ------
            ValidationError: If chat is None or no user message found in messages.
            GenerationError: If generation fails.
            StateError: If the router has been closed.
        """
        if chat is None:
            raise ValidationError(
                "Router.submit requires a Chat instance for native backend",
                code="INVALID_ARGUMENT",
                details={"param": "chat", "hint": "Pass a Chat instance"},
            )

        user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if user_message is None:
            raise ValidationError("No user message found in submitted messages")

        return self.generate(
            chat, user_message=user_message, config=config, model=model, stop_flag=stop_flag
        )

    def _warn_unsupported_content_types(self, parts: list[dict]) -> None:
        """
        Warn if non-text content types are used.

        Multimodal content (images, audio, video) can be stored in messages but
        is not yet processed by the inference engine. This method emits a warning
        if such content is detected.
        """
        import warnings

        unsupported = set()
        for part in parts:
            content_type = part.get("type", "text")
            if content_type != "text":
                unsupported.add(content_type)

        if unsupported:
            types_str = ", ".join(sorted(unsupported))
            warnings.warn(
                f"Content type(s) [{types_str}] not yet supported for inference. "
                f"Non-text content will be stored but not processed by the model.",
                UserWarning,
                stacklevel=4,
            )

    def stream(
        self,
        chat: Chat,
        user_message: str | list[dict],
        config: GenerationConfig | None = None,
        model: str | None = None,
        *,
        stop_flag: StopFlag | None = None,
    ) -> Iterator[StreamToken]:
        """
        Stream a response for a Chat.

        Submits request to Zig C API. Zig handles all message management.
        Tokens are yielded in real-time as they are generated.

        Uses a pull-based iterator API internally for reliable streaming
        without callback lifetime issues.

        Each yielded ``StreamToken`` carries ``text``, ``item_type``, and
        ``content_type`` metadata for content classification.

        Args:
            chat: The Chat instance with conversation history.
            user_message: The user's message. Can be:
                - A string for simple text messages
                - A list of content parts for multimodal input (Open Responses format):
                  [{"type": "input_text", "text": "..."},
                   {"type": "input_image", "image_url": "data:image/png;base64,..."}]
                  Or use InputImage/InputAudio/InputVideo classes and normalize_content().
            config: Generation configuration.
            model: Model to use, or None for default.
            stop_flag: Optional flag to cancel generation mid-stream.

        Yields
        ------
            StreamToken instances with text, item_type, content_type.

        Raises
        ------
            GenerationError: If streaming fails. Error message includes detailed
                context from Zig (model path, specific failure reason, etc.).
            ValidationError: If the specified model is not found in targets.
            StateError: If the router has been closed.
        """
        from ._bindings import RouterGenerateConfig

        model = self._resolve_model(model)

        # Build C config struct
        chat_template_str = self._resolve_chat_template(config.chat_template) if config else None
        c_config = RouterGenerateConfig(
            max_tokens=config.max_tokens if config else 0,
            temperature=config.temperature if config else -1.0,
            top_k=config.top_k if config else 0,
            top_p=config.top_p if config else -1.0,
            min_p=config.min_p if config else -1.0,
            repetition_penalty=config.repetition_penalty if config else 0.0,
            stop_sequences=config.stop_sequences if config else None,
            logit_bias=config.logit_bias if config else None,
            seed=(config.seed if config.seed is not None else 0) if config else 0,
            chat_template=chat_template_str,
            extra_context=config.extra_context if config else None,
            tools_json=config.tools_json if config else None,
            tool_choice=config.tool_choice if config else None,
            stop_flag=stop_flag,  # Allow async callers to pass stop_flag for cancellation
            extra_body=config.extra_body if config else None,
        )

        # Warn if non-text content types are used
        if isinstance(user_message, list):
            self._warn_unsupported_content_types(user_message)

        # Build content parts
        if isinstance(user_message, str):
            parts_list = [{"type": "text", "text": user_message}]
        else:
            parts_list = user_message
        parts, data_refs = _c.build_router_content_parts(parts_list)

        # Get or create backend handle for this model
        backend_handle = self._get_or_create_backend(model)

        # Create iterator
        iterator = _c.iterator_create(
            self._lib,
            chat._chat_ptr,
            parts,
            len(parts_list),
            backend_handle,
            c_config,
        )

        if not iterator:
            from talu._bindings import get_last_error

            zig_error = get_last_error()
            raise GenerationError(
                f"Router.stream() failed to create iterator: {zig_error or 'unknown error'}",
                code="ITERATOR_CREATE_FAILED",
            )

        # Track this iterator so close() can wait for it
        iterator_id = id(iterator)
        with self._active_iterators_lock:
            self._active_iterators.add(iterator_id)

        try:
            # Poll for tokens
            while True:
                text = _c.iterator_next(self._lib, iterator)
                if text is None:
                    # Check for errors
                    if _c.iterator_has_error(self._lib, iterator):
                        error_code = _c.iterator_error_code(self._lib, iterator)
                        raise GenerationError(
                            f"Router.stream() failed with error code {error_code}",
                            code="GENERATION_FAILED",
                            details={"error_code": error_code},
                        )
                    break
                # Read content classification for this token
                item_type = _c.iterator_item_type(self._lib, iterator)
                content_type = _c.iterator_content_type(self._lib, iterator)
                yield StreamToken(text, item_type, content_type)
        except GeneratorExit:
            # Generator was closed early (e.g., break in for loop)
            _c.iterator_cancel(self._lib, iterator)
            raise
        finally:
            # Always free the iterator
            _c.iterator_free(self._lib, iterator)
            # Keep data_refs alive until here
            del data_refs
            # Untrack iterator and notify waiters
            with self._active_iterators_lock:
                self._active_iterators.discard(iterator_id)
                self._iterator_done.notify_all()

    async def stream_async(
        self,
        chat: Chat | AsyncChat,
        user_message: str | list[dict],
        config: GenerationConfig | None = None,
        model: str | None = None,
    ) -> AsyncIterator[StreamToken]:
        """
        Async stream a response for a Chat.

        True async streaming using a background thread with cancellation support.
        Tokens are yielded as they are generated, not buffered.

        When the async iterator is cancelled (e.g., client disconnect, CancelledError),
        generation is stopped gracefully via the stop flag mechanism.

        Args:
            chat: The Chat instance with conversation history.
            user_message: The user's message to add and respond to.
            config: Generation configuration.
            model: Model to use, or None for default.

        Yields
        ------
            StreamToken instances with text, item_type, content_type.

        Raises
        ------
            StateError: If router has no default model and none specified.
            GenerationError: If generation fails (Zig error, model error, etc.).
        """
        import asyncio
        import threading

        from ._bindings import StopFlag

        # Create async queue and stop flag
        token_queue: asyncio.Queue[StreamToken | None | BaseException] = asyncio.Queue()
        stop_flag = StopFlag()
        loop = asyncio.get_running_loop()

        def run_generation():
            """Run sync streaming in background thread, putting tokens on async queue."""
            try:
                # Call the sync stream method with stop_flag for Zig-level cancellation
                for token in self.stream(chat, user_message, config, model, stop_flag=stop_flag):  # type: ignore[arg-type]
                    # Put token on async queue from sync thread
                    loop.call_soon_threadsafe(token_queue.put_nowait, token)
            except BaseException as e:
                # BaseException intentional: thread boundary must forward ALL exceptions
                # (including KeyboardInterrupt/SystemExit) to async context for re-raise
                loop.call_soon_threadsafe(token_queue.put_nowait, e)
            finally:
                # Signal end of stream
                loop.call_soon_threadsafe(token_queue.put_nowait, None)
                # Unregister thread now that all work is complete
                with self._active_threads_lock:
                    self._active_threads.discard(threading.current_thread())

        # Start generation in background thread
        gen_thread = threading.Thread(target=run_generation, daemon=True)

        # Track this thread so close() can wait for it
        with self._active_threads_lock:
            self._active_threads.add(gen_thread)

        gen_thread.start()

        generator_exhausted = False
        try:
            # Yield tokens as they arrive (true async streaming)
            while True:
                item = await token_queue.get()
                if item is None:
                    generator_exhausted = True
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        except asyncio.CancelledError:
            # Client disconnected or task cancelled - signal Zig to stop
            stop_flag.signal()
            raise
        finally:
            # If generator wasn't exhausted, signal stop so thread can finish
            if not generator_exhausted:
                stop_flag.signal()

            # Wait for thread to complete before returning.
            # This is critical for correctness: we must not return while
            # the background thread is still running, as subsequent operations
            # (like starting a new stream or calling close()) could race.
            # Use run_in_executor to avoid blocking the event loop.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, gen_thread.join)

    def embed(
        self,
        text: str,
        model: str | None = None,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[float]:
        """
        Extract embeddings from text.

        Runs the full transformer forward pass and returns pooled hidden states
        as a dense vector embedding. Uses the same cached engine as generation.

        Args:
            text: Input text to embed.
            model: Model to use, or None for default.
            pooling: Pooling strategy:
                - "last" (default): Last token's hidden state (best for decoder models)
                - "mean": Average of all token hidden states
                - "first": First token (CLS token for BERT-style models)
            normalize: Whether to L2-normalize the output embedding. Default True.

        Returns
        -------
            List of floats representing the embedding vector.
            Length equals the model's hidden dimension (d_model).

        Raises
        ------
            GenerationError: If embedding extraction fails.
            ValidationError: If invalid pooling strategy.

        Example:
            >>> router = Router(["Qwen/Qwen3-0.6B"])
            >>> embedding = router.embed("Hello, world!")
            >>> len(embedding)  # d_model (e.g., 1024)
        """
        model = self._resolve_model(model)

        # Convert pooling strategy to enum value
        pooling_map = {"last": 0, "mean": 1, "first": 2}
        if pooling not in pooling_map:
            raise ValidationError(
                f"Invalid pooling strategy '{pooling}'. Use 'last', 'mean', or 'first'."
            )
        pooling_value = pooling_map[pooling]

        return _c.router_embed(self._lib, model, text, pooling_value, normalize)

    def embedding_dim(self, model: str | None = None) -> int:
        """
        Get the embedding dimension for a model.

        Args:
            model: Model to query, or None for default.

        Returns
        -------
            The embedding dimension (d_model).

        Raises
        ------
            GenerationError: If the model cannot be loaded or doesn't support embeddings.
        """
        model_str = self._resolve_model(model)
        dim = self._lib.talu_router_embedding_dim(model_str.encode("utf-8"))
        if dim == 0:
            raise GenerationError(
                f"Failed to get embedding dimension for model '{model_str}': "
                "model may not be loaded or doesn't support embeddings"
            )
        return dim

    def close(self) -> None:
        """Close router and release resources.

        Waits for any active generation threads to complete before releasing
        resources to prevent use-after-free crashes.

        Note: This only frees this Router's backend handles. The global engine
        cache in Zig is intentionally NOT cleared, as other Chat/Router instances
        may still be using those engines. The engine cache is process-level
        and shared across all instances for performance.
        """
        if not self._closed:
            # Wait for all active generation threads to complete
            # This prevents closing handles while Zig is still using them
            with self._active_threads_lock:
                threads_to_wait = list(self._active_threads)
            for thread in threads_to_wait:
                thread.join()  # Blocking wait - generation must complete

            # Wait for all active sync stream iterators to complete
            with self._active_iterators_lock:
                while self._active_iterators:
                    self._iterator_done.wait()

            # Free this router's backend and canonical handles only.
            # Do NOT call talu_router_close_all() - that clears the GLOBAL
            # engine cache which would corrupt other Chat/Router instances.
            self._cleanup_handles()
            self._targets.clear()
            self._closed = True

    def __enter__(self) -> Router:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, closing router."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure cleanup as fail-safe."""
        try:
            self.close()
        except Exception:
            pass  # Suppress all exceptions in destructor

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"Router(models={self.models}, default={self._default_model!r}, status={status})"

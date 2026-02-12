"""
Entry point for LLM inference.

Manages model connections and creates Chat instances for both
casual and power user workflows.

Casual Usage (module-level functions):

    >>> import talu
    >>>
    >>> # Chat-formatted one-shot
    >>> response = talu.ask("Qwen/Qwen3-0.6B", "What is 2+2?")
    >>>
    >>> # Raw completion
    >>> response = talu.raw_complete(
    ...     "Qwen/Qwen3-0.6B",
    ...     "The sky is blue because"
    ... )

Power User (explicit client):

    >>> from talu import Client
    >>>
    >>> client = Client("Qwen/Qwen3-0.6B")
    >>>
    >>> # One-shot completions (efficient, reuses Client)
    >>> response = client.ask("What is 2+2?")
    >>>
    >>> # For multi-turn conversations, use client.chat()
    >>> chat = client.chat(system="You are helpful.")
    >>> response = chat("Hello!")
    >>> response = response.append("Tell me more")

Multi-Model Routing (coming soon):

    >>> client = Client(["Qwen/Qwen3-0.6B", "openai::gpt-4o"])
    >>>
    >>> chat = client.chat()
    >>>
    >>> # Default uses first model
    >>> response = chat("Hello!")
    >>>
    >>> # Switch for one call
    >>> response = chat("Hello!", model="openai::gpt-4o")
    >>>
    >>> # Fan-out to multiple models
    >>> responses = chat("Hello!", model=["Qwen/Qwen3-0.6B", "openai::gpt-4o"])

Model Backends:

    The :: separator identifies the backend (who runs inference).
    Bare model IDs use talu's native inference engine.

    Native backend (talu's inference engine):
    - "Qwen/Qwen3-0.6B" - Implicit native, model from HuggingFace
    - "native::Qwen/Qwen3-0.6B" - Explicit native backend
    - "native::./my-model" - Native backend, local directory

    External API backends (coming soon):
    - "openai::gpt-4o" - OpenAI API
    - "anthropic::claude-3-sonnet" - Anthropic API
    - "vllm::Qwen/Qwen3-0.6B" - vLLM server (endpoint configured separately)
    - "ollama::llama3" - Ollama server
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from .exceptions import StateError
from .router import (
    LocalBackend,
    ModelSpec,
    OpenAICompatibleBackend,
)

if TYPE_CHECKING:
    from .chat.hooks import Hook, HookManager
    from .chat.session import AsyncChat
    from .profile import Profile
    from .router import BackendSpec, Capabilities, Router
    from .router.config import CompletionOptions, GenerationConfig
    from .template import PromptTemplate

from .chat.response import AsyncResponse, Response
from .chat.session import Chat

__all__ = [
    "Client",
    "AsyncClient",
    "ModelInput",
]

# Type alias for model input (str or structured ModelSpec)
ModelInput = str | ModelSpec


def _build_model_spec(
    model: str | ModelSpec,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    org_id: str | None = None,
    timeout_ms: int | None = None,
    max_retries: int | None = None,
    gpu_layers: int | None = None,
    use_mmap: bool | None = None,
    num_threads: int | None = None,
) -> ModelSpec:
    """
    Build a ModelSpec from a model string or existing spec, applying backend kwargs.

    If any OpenAI-compatible kwargs (base_url, api_key, org_id, timeout_ms, max_retries)
    are provided, creates an OpenAICompatibleBackend.

    If any local kwargs (gpu_layers, use_mmap, num_threads) are provided, creates
    a LocalBackend.

    If both local and remote kwargs are provided, raises ValidationError.

    If no backend kwargs are provided and model is a string, returns a simple ModelSpec
    with no explicit backend (letting Zig auto-detect).

    If model is already a ModelSpec with a backend, returns it as-is (backend kwargs
    are ignored with a warning if any were provided).
    """
    from .exceptions import ValidationError

    # If already a ModelSpec with a backend, return as-is
    if isinstance(model, ModelSpec):
        # Check if user tried to override backend kwargs
        has_remote_kwargs = any(
            x is not None for x in [base_url, api_key, org_id, timeout_ms, max_retries]
        )
        has_local_kwargs = any(x is not None for x in [gpu_layers, use_mmap, num_threads])
        if (has_remote_kwargs or has_local_kwargs) and model.backend is not None:
            import warnings

            warnings.warn(
                "Backend kwargs (base_url, api_key, etc.) are ignored when "
                "model is a ModelSpec with an existing backend",
                UserWarning,
                stacklevel=3,
            )
        return model

    # Determine which backend type based on kwargs
    has_remote_kwargs = any(
        x is not None for x in [base_url, api_key, org_id, timeout_ms, max_retries]
    )
    has_local_kwargs = any(x is not None for x in [gpu_layers, use_mmap, num_threads])

    if has_remote_kwargs and has_local_kwargs:
        raise ValidationError(
            "Cannot mix remote backend kwargs (base_url, api_key, etc.) "
            "with local backend kwargs (gpu_layers, use_mmap, num_threads)"
        )

    backend: BackendSpec | None = None

    if has_remote_kwargs:
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_key=api_key,
            org_id=org_id,
            timeout_ms=timeout_ms or 0,
            max_retries=max_retries or 0,
        )
    elif has_local_kwargs:
        backend = LocalBackend(
            gpu_layers=gpu_layers if gpu_layers is not None else -1,
            use_mmap=use_mmap if use_mmap is not None else True,
            num_threads=num_threads if num_threads is not None else 0,
        )

    return ModelSpec(ref=model, backend=backend)


class Client:
    """
    Entry point for LLM inference.

    Concurrency:
        Not thread-safe. Create one Client per thread, or use separate Chat
        instances per thread (Chat instances are independent).

    Manages model connections and creates conversations. Supports single
    models, multiple models for routing, and various backends (local,
    OpenAI, Anthropic, vLLM, etc.).

    Note:
        Multiple Client instances for the same model share the underlying
        engine. Subsequent Client creation after the first model load is
        inexpensive.

    Args:
        model: Model identifier(s). Can be:
            - **str**: Simple model ID ("Qwen/Qwen3-0.6B", "openai::gpt-4o")
            - **ModelSpec**: Structured config for advanced backend settings
            - **list**: Multiple models for routing/fan-out (coming soon)
        profile: Optional storage profile inherited by all chats created by
            this client. If provided, chats persist to
            ``~/.talu/db/<profile>/`` by default.
        hooks: List of Hook instances for observability (metrics, logging, tracing).
            Hooks receive callbacks at generation start, first token (TTFT), and end.
        base_url: API endpoint URL. When provided, uses OpenAI-compatible backend.
        api_key: API key for remote backends. Falls back to environment
            variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
        org_id: Organization ID for remote backends.
        timeout_ms: Request timeout in milliseconds for remote backends.
        max_retries: Maximum retry attempts for remote backends.
        gpu_layers: Number of layers to offload to GPU (-1 for all). Local backend only.
        use_mmap: Use memory-mapped files for model loading. Local backend only.
        num_threads: Number of threads for inference (0 for auto). Local backend only.

    Attributes
    ----------
        models: List of available model identifiers.
        default_model: The default model used for generation.
        hooks: The HookManager for adding/removing hooks after construction.

    Example - Single model:
        >>> client = Client("Qwen/Qwen3-0.6B")
        >>> response = client.ask("What is 2+2?")
        >>> print(response)
        4

    Example - Remote backend (Pythonic):
        >>> client = Client("gpt-4", base_url="http://localhost:8080/v1", api_key="sk-...")
        >>> response = client.ask("Hello!")

    Example - Local backend with GPU offload:
        >>> client = Client("Qwen/Qwen3-0.6B", gpu_layers=20, num_threads=4)

    Example - Advanced config via ModelSpec (power users):
        >>> from talu import Client
        >>> from talu.router import ModelSpec, OpenAICompatibleBackend
        >>>
        >>> spec = ModelSpec(
        ...     ref="my-model",
        ...     backend=OpenAICompatibleBackend(
        ...         base_url="http://localhost:8080/v1",
        ...         timeout_ms=5000
        ...     )
        ... )
        >>> client = Client(spec)

    Example - Multi-user serving:
        >>> client = Client("Qwen/Qwen3-0.6B")
        >>> alice = client.chat(system="You are helpful.")
        >>> bob = client.chat(system="You are a pirate.")
        >>> response = alice("Hello!")
        >>> response = bob("Ahoy!")
        >>> client.close()

    Raises
    ------
        ValidationError: If no models are provided.

    Example - Context manager:
        >>> with Client("Qwen/Qwen3-0.6B") as client:
        ...     chat = client.chat()
        ...     response = chat("Hello!")
    """

    def __init__(
        self,
        model: ModelInput | list[ModelInput],
        *,
        profile: Profile | None = None,
        hooks: list[Hook] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        timeout_ms: int | None = None,
        max_retries: int | None = None,
        gpu_layers: int | None = None,
        use_mmap: bool | None = None,
        num_threads: int | None = None,
    ):
        from .chat.hooks import HookManager

        # Normalize to list
        if isinstance(model, (str, ModelSpec)):
            models_raw = [model]
        else:
            models_raw = list(model)

        if len(models_raw) == 0:
            from .exceptions import ValidationError

            raise ValidationError("At least one model must be provided")

        # Build ModelSpecs from flattened kwargs
        # Note: kwargs only apply to the first model (for single-model convenience)
        # Multi-model routing with different backends requires explicit ModelSpec per model
        models: list[ModelSpec] = []
        for i, m in enumerate(models_raw):
            if i == 0:
                # Apply flattened kwargs to first model
                spec = _build_model_spec(
                    m,
                    base_url=base_url,
                    api_key=api_key,
                    org_id=org_id,
                    timeout_ms=timeout_ms,
                    max_retries=max_retries,
                    gpu_layers=gpu_layers,
                    use_mmap=use_mmap,
                    num_threads=num_threads,
                )
            else:
                # Other models use default backend detection
                spec = _build_model_spec(m)
            models.append(spec)

        self._closed = False
        self._chats: list[Chat] = []
        self._profile = profile

        # Initialize hook manager for observability
        self._hooks = HookManager(hooks)

        # Create router with model targets
        # Router holds model strings; actual engine creation happens in Zig
        from talu.router import Router

        self._router = Router(models)

    @property
    def models(self) -> list[str]:
        """List of available model identifiers."""
        return self._router.models

    @property
    def default_model(self) -> str:
        """The default model used when none is specified."""
        return self._router.default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        """Set the default model."""
        self._router.default_model = model

    @property
    def router(self) -> Router:
        """The Router instance (for advanced use)."""
        return self._router

    @property
    def hooks(self) -> HookManager:
        """
        The HookManager for observability.

        Use this to add or remove hooks after Client construction:

            >>> client = Client("model")
            >>> client.hooks.add(MetricsHook())
            >>> client.hooks.remove(my_hook)

        Returns
        -------
            HookManager instance.
        """
        return self._hooks

    def _check_closed(self) -> None:
        """Raise if client has been closed."""
        if self._closed:
            raise StateError(
                "Client has been closed. Create a new Client instance.",
                code="STATE_CLOSED",
            )

    def ask(
        self,
        prompt: str,
        *,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Stateless text completion (non-streaming).

        Generates a complete response for a single prompt without conversation
        history. This method always returns Response (not StreamingResponse)
        and waits for generation to finish before returning.

        This is the efficient way to do one-shot generation when you have
        an existing Client instance (e.g., in production server or batch
        processing). No Chat object overhead - just calls chat.send() directly.

        For multi-turn conversations with history and append(), use ``client.chat()``.

        Streaming vs Non-Streaming
        --------------------------

        This method always returns ``Response`` (non-streaming). For streaming
        behavior with real-time token feedback, use:

        - ``client.chat()`` with default ``stream=True`` (Chat object, multi-turn)
        - ``chat.send(prompt, stream=True)`` (one-shot streaming via Chat)

        **Use client.ask() when:**

        - You need the full response immediately (batch processing, API endpoints)
        - You don't care about intermediate tokens
        - You want Response object with metadata access
        - You're doing multiple one-shot calls efficiently (reuses Client)

        **Use client.chat() with stream=True when:**

        - You want real-time feedback (interactive applications)
        - You're building a chat interface with live updates
        - Long generations where progress matters
        - Reducing perceived latency for user-facing apps

        **Why client.ask() is non-streaming:**

        One-shot completions are typically used for batch processing or API
        endpoints where you need the complete result immediately. Adding streaming
        adds complexity (iterators, caching) without benefit for these use cases.

        Args:
            prompt: The input prompt.
            model: Model to use (overrides default_model).
            config: Generation configuration.
            **kwargs: Generation overrides.

        Returns
        -------
            Response object (str-able, with metadata access).

        Raises
        ------
            StateError: If client has been closed.
            GenerationError: If generation fails.

        Example:
            >>> from talu import Client
            >>> client = Client("Qwen/Qwen3-0.6B")
            >>> # One-shot completions (efficient, reuses Client)
            >>> response = client.ask("What is 2+2?")
            >>> print(response)
            4
            >>> # Multiple one-shot calls (no history, fast)
            >>> for task in ["What is 3+3?", "What is 4+4?"]:
            ...     print(client.ask(task))
            >>> # For streaming with real-time feedback, use chat()
            >>> chat = client.chat()
            >>> for token in chat("Tell me a story"):
            ...     print(token, end="", flush=True)
        """
        self._check_closed()

        # Create a temporary chat for the completion
        chat = self.chat(model=model, config=config)
        return chat.send(prompt, stream=False, **kwargs)

    def chat(
        self,
        *,
        system: str | None = None,
        messages: list[dict] | None = None,
        model: str | None = None,
        config: GenerationConfig | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        marker: str = "",
        metadata: dict | None = None,
        chat_template: str | PromptTemplate | None = None,
        offline: bool = False,
    ) -> Chat:
        """
        Create a Chat session for multi-turn conversations.

        Returns a Chat object that stores conversation history and allows
        generation with append() support.

        Streaming Behavior
        ------------------

        Chat instances default to streaming (``stream=True``) when called via
        ``chat(prompt)`` or ``chat()(prompt)``. This returns ``StreamingResponse``
        with tokens arriving incrementally, providing immediate feedback and
        matching industry standard for chat interfaces.

        To disable streaming and get complete Response after generation finishes,
        pass ``stream=False``::

            >>> chat = client.chat()
            >>> response = chat("Hello!", stream=False)  # Response (non-streaming)
            >>> print(response.text)  # Immediately available

        For streaming (default)::

            >>> for token in chat("Hello!"):  # StreamingResponse
            ...     print(token, end="", flush=True)

        Args:
            system: System prompt.
            messages: Initial message history (for restoring sessions).
            model: Default model for this chat (not yet supported).
            config: Default generation config for this chat.
                Profile persistence (if configured on Client) is also applied.
            session_id: Optional session identifier for this chat.
            parent_session_id: Optional parent session identifier for forks.
            marker: Session marker for storage backends (default: "" = normal/unmarked).
                Values: "pinned", "archived", "deleted", or "" (normal).
            metadata: Optional session metadata dict.
            chat_template: Custom chat template (None uses model default, use
                explicit None for raw completion without template formatting).
            offline: If True, disallow network access when resolving model URIs.

        Returns
        -------
            Chat object for multi-turn interaction.

        Raises
        ------
            StateError: If the client has been closed.
            NotImplementedError: If messages parameter is provided (not yet supported).

        Example:
            >>> chat = client.chat(system="You are a pirate.")
            >>> response = chat("Hello!")
            >>> response = response.append("Tell me about your ship")
            >>> print(chat.items)  # Full history
        """
        self._check_closed()

        # Create chat with client reference (Chat gets Router from Client)
        chat_instance = Chat(
            client=self,
            profile=self._profile,
            system=system,
            model=model,
            config=config,
            session_id=session_id,
            parent_session_id=parent_session_id,
            marker=marker,
            metadata=metadata,
            chat_template=chat_template,
            offline=offline,
        )

        # Load initial messages if provided
        # TODO: Implement message loading when talu_completions_load_messages C API exists
        if messages:
            raise NotImplementedError(
                "Loading initial messages is not yet implemented. "
                "Use system= for system prompt instead."
            )

        # Track chat for lifecycle management
        self._chats.append(chat_instance)

        return chat_instance

    def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream a stateless completion.

        Args:
            prompt: The input prompt.
            model: Model to use (not yet supported).
            config: Generation configuration.
            **kwargs: Generation overrides.

        Yields
        ------
            Text chunks as they are generated.

        Raises
        ------
            StateError: If the client has been closed.
            GenerationError: If generation fails.

        Example:
            >>> for chunk in client.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        self._check_closed()

        # Create a temporary chat and stream
        chat = self.chat(model=model, config=config)
        response = chat.send(prompt, stream=True, **kwargs)
        yield from response

    def raw_complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        config: GenerationConfig | None = None,
        completion_opts: CompletionOptions | None = None,
        **kwargs: Any,
    ) -> Response:
        """
        Raw completion without chat templates.

        Sends prompt directly to model without formatting. This is a technical
        use case for prompt engineering and advanced control. Most users should
        use ``client.chat().send()`` with chat templates instead.

        The ONLY difference from chat-based completion:

        - Chat uses the model's chat template (adds role markers)
        - ``raw_complete()`` does NOT apply any template (sends raw prompt)

        **Raw-only options** (available ONLY via ``completion_opts`` parameter):

        - ``token_ids``: Send pre-tokenized input, bypassing tokenizer.
        - ``continue_from_token_id``: Force continuation from a specific token ID.
        - ``echo_prompt``: Return input + output combined.

        These options don't make sense with chat-formatted prompts and are
        intentionally EXCLUDED from chat-based APIs to keep them clean.

        Args:
            prompt: The raw prompt (sent exactly as-is, no formatting).
            system: Optional system prompt.
            model: Model to use (overrides default_model).
            config: Generation configuration.
            completion_opts: Raw-completion options (CompletionOptions).
            **kwargs: Additional generation overrides (for any backend options
                not yet in CompletionOptions).

        Returns
        -------
            Response object.

        Raises
        ------
            StateError: If client has been closed.
            GenerationError: If generation fails.

        Example:
            >>> client = Client("Qwen/Qwen3-0.6B")
            >>> # Raw completion
            >>> response = client.raw_complete("Continue: The sky is")
            >>> print(response)
            blue due to Rayleigh scattering.

            >>> # With CompletionOptions
            >>> from talu.router import CompletionOptions
            >>> opts = CompletionOptions(
            ...     token_ids=[1234, 5678],
            ...     continue_from_token_id=151645
            ... )
            >>> response = client.raw_complete("Continue: ", completion_opts=opts)
        """
        self._check_closed()
        # Extract CompletionOptions into request kwargs
        request_kwargs = dict(kwargs)
        if completion_opts is not None:
            if completion_opts.token_ids is not None:
                request_kwargs["token_ids"] = completion_opts.token_ids
            if completion_opts.continue_from_token_id is not None:
                request_kwargs["continue_from_token_id"] = completion_opts.continue_from_token_id
            if completion_opts.echo_prompt:
                request_kwargs["echo_prompt"] = completion_opts.echo_prompt
        # Create temporary chat with explicit passthrough template to disable chat formatting.
        chat = self.chat(
            system=system,
            model=model,
            config=config,
            chat_template="{{ messages[-1].content }}",
        )
        return chat.send(prompt, stream=False, **request_kwargs)

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[float]:
        """
        Extract embedding from text.

        Runs the full transformer forward pass and returns pooled hidden states
        as a dense vector embedding. Useful for semantic search, RAG, and
        document similarity.

        Args:
            text: Input text to embed.
            model: Model to use (overrides default_model).
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
            StateError: If the client has been closed.
            ValidationError: If pooling strategy is invalid.
            GenerationError: If embedding extraction fails.

        Example:
            >>> client = Client("Qwen/Qwen3-0.6B")
            >>> embedding = client.embed("Hello, world!")
            >>> len(embedding)  # d_model (e.g., 1024)

        Example - Semantic similarity:
            >>> emb1 = client.embed("The cat sat on the mat")
            >>> emb2 = client.embed("A feline rested on the rug")
            >>> similarity = sum(a*b for a, b in zip(emb1, emb2))  # cosine sim (if normalized)
        """
        self._check_closed()
        return self._router.embed(text, model=model, pooling=pooling, normalize=normalize)

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Extract embeddings from multiple texts.

        Args:
            texts: List of input texts to embed.
            model: Model to use (overrides default_model).
            pooling: Pooling strategy ("last", "mean", "first").
            normalize: Whether to L2-normalize the output embeddings.

        Returns
        -------
            List of embedding vectors, one per input text.

        Raises
        ------
            StateError: If the client has been closed.

        Example:
            >>> embeddings = client.embed_batch([
            ...     "First document",
            ...     "Second document",
            ... ])
        """
        self._check_closed()
        return [
            self.embed(text, model=model, pooling=pooling, normalize=normalize) for text in texts
        ]

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
            StateError: If the client has been closed.
        """
        self._check_closed()
        return self._router.embedding_dim(model=model)

    def capabilities(self, model: str | ModelSpec | None = None) -> Capabilities:
        """
        Return backend capabilities for a model.

        Args:
            model: Model to query, or None for default.

        Returns
        -------
            Capabilities object with backend feature flags.

        Raises
        ------
            StateError: If the client has been closed.
            ValidationError: If model_input format is invalid.
            TaluError: If capability retrieval fails.
        """
        self._check_closed()
        from talu.router import get_capabilities

        model_input = model if model is not None else self.default_model
        return get_capabilities(model_input)

    def close(self) -> None:
        """
        Close client and release resources.

        Closes all model connections. After calling close(), the client
        cannot be used for generation. Safe to call multiple times.
        """
        if not self._closed:
            self._router.close()
            self._chats.clear()
            self._closed = True

    def __enter__(self) -> Client:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure cleanup as fail-safe."""
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"Client(models={self.models}, default={self.default_model!r}, status={status})"


class AsyncClient:
    """
    Async entry point for LLM inference.

    Async equivalent of Client for non-blocking inference (FastAPI,
    aiohttp, etc.). All generation methods are async and must be awaited.

    Wraps the same engine as Client. Model weights are cached globally,
    so creating AsyncClient for the same model as an existing Client
    shares the underlying engine.

    Concurrency:
        Safe to share across asyncio tasks. Not thread-safe across OS threads.

    Args:
        model: Model identifier(s). Can be:
            - **str**: Simple model ID ("Qwen/Qwen3-0.6B", "openai::gpt-4o")
            - **ModelSpec**: Structured config for advanced backend settings
            - **list**: Multiple models for routing/fan-out (coming soon)
        base_url: API endpoint URL. When provided, uses OpenAI-compatible backend.
        api_key: API key for remote backends. Falls back to environment
            variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
        org_id: Organization ID for remote backends.
        timeout_ms: Request timeout in milliseconds for remote backends.
        max_retries: Maximum retry attempts for remote backends.
        gpu_layers: Number of layers to offload to GPU (-1 for all). Local backend only.
        use_mmap: Use memory-mapped files for model loading. Local backend only.
        num_threads: Number of threads for inference (0 for auto). Local backend only.

    Example - Basic async usage:
        >>> async with AsyncClient("Qwen/Qwen3-0.6B") as client:
        ...     response = await client.ask("What is 2+2?")
        ...     print(response)

    Example - Remote backend (Pythonic):
        >>> async with AsyncClient("gpt-4", base_url="http://localhost:8080/v1") as client:
        ...     response = await client.ask("Hello!")

    Example - Advanced config via ModelSpec (power users):
        >>> from talu import AsyncClient
        >>> from talu.router import ModelSpec, OpenAICompatibleBackend
        >>>
        >>> spec = ModelSpec(
        ...     ref="my-model",
        ...     backend=OpenAICompatibleBackend(
        ...         base_url="http://localhost:8080/v1",
        ...         timeout_ms=5000
        ...     )
        ... )
        >>> async with AsyncClient(spec) as client:
        ...     response = await client.ask("Hello!")

    Example - Async streaming:
        >>> async with AsyncClient("Qwen/Qwen3-0.6B") as client:
        ...     async for chunk in client.stream("Tell me a story"):
        ...         print(chunk, end="", flush=True)

    Raises
    ------
        ValidationError: If no models are provided.

    Example - Multi-user async serving:
        >>> client = AsyncClient("Qwen/Qwen3-0.6B")
        >>> alice = client.chat(system="You are helpful.")
        >>> bob = client.chat(system="You are a pirate.")
        >>> response = await alice("Hello!")
        >>> response = await bob("Ahoy!")
        >>> await client.close()
    """

    def __init__(
        self,
        model: ModelInput | list[ModelInput],
        *,
        hooks: list[Hook] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        org_id: str | None = None,
        timeout_ms: int | None = None,
        max_retries: int | None = None,
        gpu_layers: int | None = None,
        use_mmap: bool | None = None,
        num_threads: int | None = None,
    ):
        from .chat.hooks import HookManager

        # Normalize to list
        if isinstance(model, (str, ModelSpec)):
            models_raw = [model]
        else:
            models_raw = list(model)

        if len(models_raw) == 0:
            from .exceptions import ValidationError

            raise ValidationError("At least one model must be provided")

        # Build ModelSpecs from flattened kwargs
        models: list[ModelSpec] = []
        for i, m in enumerate(models_raw):
            if i == 0:
                spec = _build_model_spec(
                    m,
                    base_url=base_url,
                    api_key=api_key,
                    org_id=org_id,
                    timeout_ms=timeout_ms,
                    max_retries=max_retries,
                    gpu_layers=gpu_layers,
                    use_mmap=use_mmap,
                    num_threads=num_threads,
                )
            else:
                spec = _build_model_spec(m)
            models.append(spec)

        self._closed = False
        self._chats: list[AsyncChat] = []

        # Initialize hook manager for observability
        self._hooks = HookManager(hooks)

        # Create router with model targets
        from talu.router import Router

        self._router = Router(models)

    @property
    def models(self) -> list[str]:
        """List of available model identifiers."""
        return self._router.models

    @property
    def default_model(self) -> str:
        """The default model used when none is specified."""
        return self._router.default_model

    @default_model.setter
    def default_model(self, model: str) -> None:
        """Set the default model."""
        self._router.default_model = model

    @property
    def router(self) -> Router:
        """The Router instance (for advanced use)."""
        return self._router

    @property
    def hooks(self) -> HookManager:
        """
        The HookManager for observability.

        Use this to add or remove hooks after AsyncClient construction.

        Returns
        -------
            HookManager instance.
        """
        return self._hooks

    def _check_closed(self) -> None:
        """Raise if client has been closed."""
        if self._closed:
            raise StateError(
                "AsyncClient has been closed. Create a new AsyncClient instance.",
                code="STATE_CLOSED",
            )

    async def ask(
        self,
        prompt: str,
        *,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncResponse:
        """
        Async stateless text completion.

        Generates a response for a single prompt without conversation history.
        For multi-turn conversations, use chat() instead.

        Args:
            prompt: The input prompt.
            model: Model to use (overrides default_model).
            config: Generation configuration.
            **kwargs: Generation overrides.

        Returns
        -------
            Response object (str-able, with metadata access).

        Raises
        ------
            StateError: If the client has been closed.
            GenerationError: If generation fails.

        Example:
            >>> response = await client.ask("What is 2+2?")
            >>> print(response)
        """
        self._check_closed()

        chat = self.chat(model=model, config=config)
        return await chat.send(prompt, stream=False, **kwargs)

    def chat(
        self,
        *,
        system: str | None = None,
        messages: list[dict] | None = None,
        model: str | None = None,
        config: GenerationConfig | None = None,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        marker: str = "",
        metadata: dict | None = None,
        chat_template: str | PromptTemplate | None = None,
        offline: bool = False,
    ) -> AsyncChat:
        """
        Create a new AsyncChat instance.

        Note: This method itself is synchronous - it just creates the AsyncChat
        object. The generation methods on AsyncChat are async.

        Args:
            system: System prompt.
            messages: Initial message history (for restoring sessions).
            model: Default model for this chat (not yet supported).
            config: Default generation config for this chat.
            session_id: Optional session identifier for this chat.
            parent_session_id: Optional parent session identifier for forks.
            marker: Session marker for storage backends (default: "" = normal/unmarked).
                Values: "pinned", "archived", "deleted", or "" (normal).
            metadata: Optional session metadata dict.
            chat_template: Custom chat template (None uses model default, use
                explicit None for raw completion without template formatting).
            offline: If True, disallow network access when resolving model URIs.

        Returns
        -------
            AsyncChat object for async multi-turn interaction.

        Raises
        ------
            StateError: If the client has been closed.
            NotImplementedError: If messages parameter is provided (not yet supported).

        Example:
            >>> chat = client.chat(system="You are helpful.")
            >>> response = await chat("Hello!")
            >>> response = await response.append("Tell me more")
        """
        self._check_closed()

        from .chat.session import AsyncChat

        chat_instance = AsyncChat(
            client=self,
            system=system,
            model=model,
            config=config,
            session_id=session_id,
            parent_session_id=parent_session_id,
            marker=marker,
            metadata=metadata,
            chat_template=chat_template,
            offline=offline,
        )

        if messages:
            raise NotImplementedError(
                "Loading initial messages is not yet implemented. "
                "Use system= for system prompt instead."
            )

        self._chats.append(chat_instance)
        return chat_instance

    async def stream(
        self,
        prompt: str,
        *,
        model: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Async stream a stateless completion.

        Args:
            prompt: The input prompt.
            model: Model to use (not yet supported).
            config: Generation configuration.
            **kwargs: Generation overrides.

        Yields
        ------
            Text chunks as they are generated.

        Raises
        ------
            StateError: If the client has been closed.
            GenerationError: If generation fails.

        Example:
            >>> async for chunk in client.stream("Tell me a story"):
            ...     print(chunk, end="", flush=True)
        """
        self._check_closed()

        chat = self.chat(model=model, config=config)
        response = await chat.send(prompt, stream=True, **kwargs)
        async for chunk in response:
            yield chunk

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[float]:
        """
        Extract embedding from text (synchronous).

        Note: This method is synchronous as embedding extraction
        is typically fast enough to not require async.

        Args:
            text: Input text to embed.
            model: Model to use (overrides default_model).
            pooling: Pooling strategy ("last", "mean", "first").
            normalize: Whether to L2-normalize the output embedding.

        Returns
        -------
            List of floats representing the embedding vector.

        Raises
        ------
            StateError: If the client has been closed.
            ValidationError: If pooling strategy is invalid.
            GenerationError: If embedding extraction fails.
        """
        self._check_closed()
        return self._router.embed(text, model=model, pooling=pooling, normalize=normalize)

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        pooling: str = "last",
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Extract embeddings from multiple texts (synchronous).

        Args:
            texts: List of input texts to embed.
            model: Model to use (overrides default_model).
            pooling: Pooling strategy.
            normalize: Whether to L2-normalize.

        Returns
        -------
            List of embedding vectors.

        Raises
        ------
            StateError: If the client has been closed.
        """
        self._check_closed()
        return [
            self.embed(text, model=model, pooling=pooling, normalize=normalize) for text in texts
        ]

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
            StateError: If the client has been closed.
        """
        self._check_closed()
        return self._router.embedding_dim(model=model)

    def capabilities(self, model: str | ModelSpec | None = None) -> Capabilities:
        """
        Return backend capabilities for a model.

        Args:
            model: Model to query, or None for default.

        Returns
        -------
            Capabilities object with backend feature flags.

        Raises
        ------
            StateError: If the client has been closed.
            ValidationError: If model_input format is invalid.
            TaluError: If capability retrieval fails.
        """
        self._check_closed()
        from talu.router import get_capabilities

        model_input = model if model is not None else self.default_model
        return get_capabilities(model_input)

    async def close(self) -> None:
        """
        Close client and release resources.

        Closes all model connections. After calling close(), the client
        cannot be used for generation. Safe to call multiple times.
        """
        if not self._closed:
            self._router.close()
            self._chats.clear()
            self._closed = True

    async def __aenter__(self) -> AsyncClient:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def __del__(self) -> None:
        """Destructor - ensure cleanup as fail-safe."""
        try:
            # Use sync close path since __del__ cannot be async
            if not self._closed:
                self._router.close()
                self._closed = True
        except Exception:
            pass

    def __repr__(self) -> str:
        status = "closed" if self._closed else "open"
        return f"AsyncClient(models={self.models}, default={self.default_model!r}, status={status})"

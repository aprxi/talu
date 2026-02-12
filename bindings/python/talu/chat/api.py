"""
Module-level convenience functions.

These functions provide the simplest possible API for casual users.
They wrap Client internally, so users don't need to manage instances.

**Primary function** (use this for one-shot tasks):

    >>> import talu
    >>> response = talu.ask("Qwen/Qwen3-0.6B", "What is 2+2?")
    >>> print(response)
    4

**With system prompt**:

    >>> response = talu.ask(
    ...     "Qwen/Qwen3-0.6B",
    ...     "Hello!",
    ...     system="You are a helpful assistant."
    ... )
    >>> print(response)
    Hi! How can I help you today?

**Raw completion** (power users only):

    >>> response = talu.raw_complete(
    ...     "Qwen/Qwen3-0.6B",
    ...     "The sky is blue because"
    ... )
    >>> print(response)
    of Rayleigh scattering.

**Streaming**:

    >>> for chunk in talu.stream("Qwen/Qwen3-0.6B", "Tell me a story"):
    ...     print(chunk, end="", flush=True)

For repeated use or multi-turn conversations, use Chat (loads model once):

    >>> chat = talu.Chat("Qwen/Qwen3-0.6B", system="You are helpful.")
    >>> response = chat("Hello!")
    >>> response = response.append("Tell me more")
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from talu.router.config import CompletionOptions, GenerationConfig

    from .response import Response


def raw_complete(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    config: GenerationConfig | None = None,
    completion_opts: CompletionOptions | None = None,
    **kwargs: Any,
) -> Response:
    """
    Raw completion without chat templates.

    Sends prompt directly to model without any formatting. This is a technical
    use case for prompt engineering and advanced control. Most users should
    use ``talu.ask()`` instead.

    The ONLY difference from ``talu.ask()``:

    - ``ask()`` applies the model's chat template (adds role markers)
    - ``raw_complete()`` does NOT apply any template (sends raw prompt)

    **Raw-only options** (available ONLY via ``completion_opts`` parameter):

    - ``token_ids``: Send pre-tokenized input, bypassing tokenizer.
    - ``continue_from_token_id``: Force continuation from a specific token ID.
    - ``echo_prompt``: Return input + output combined.

    These options don't make sense with chat-formatted prompts and are
    intentionally EXCLUDED from ``talu.ask()`` to keep the casual API clean.

    Args:
        model: Model identifier.
        prompt: The raw prompt (sent exactly as-is, no formatting).
        system: Optional system prompt.
        config: Generation configuration.
        completion_opts: Raw-completion options (CompletionOptions).
        **kwargs: Additional generation overrides (for any backend options
            not yet in CompletionOptions).

    Returns
    -------
        Response object (str-able, with metadata access).

    Raises
    ------
        ModelError: If the model cannot be loaded.
        GenerationError: If generation fails.

    Example:
        >>> import talu
        >>> # Raw continuation (no chat template)
        >>> response = talu.raw_complete(
        ...     "Qwen/Qwen3-0.6B",
        ...     "The sky is blue because"
        ... )
        >>> print(response)
        of Rayleigh scattering.

        >>> # With CompletionOptions
        >>> from talu.router import CompletionOptions
        >>> opts = CompletionOptions(
        ...     token_ids=[1234, 5678],
        ...     continue_from_token_id=151645
        ... )
        >>> response = talu.raw_complete(
        ...     "Qwen/Qwen3-0.6B",
        ...     "Continue: ",
        ...     completion_opts=opts
        ... )
    """
    from talu.client import Client

    with Client(model) as client:
        return client.raw_complete(
            prompt,
            system=system,
            config=config,
            completion_opts=completion_opts,
            **kwargs,
        )


def stream(
    model: str,
    prompt: str,
    *,
    config: GenerationConfig | None = None,
    **kwargs: Any,
) -> Iterator[str]:
    """
    Stream a stateless completion.

    .. warning::

        **Performance: This function loads the model on every call.**

        For repeated queries, use ``Client`` instead::

            # SLOW: Loads model each time
            for p in prompts:
                for chunk in talu.stream("Qwen/Qwen3-0.6B", p):
                    print(chunk, end="")

            # FAST: Loads model once
            client = talu.Client("Qwen/Qwen3-0.6B")
            for p in prompts:
                for chunk in client.stream(p):
                    print(chunk, end="")

    Args:
        model: Model identifier.
        prompt: The input prompt.
        config: Generation configuration.
        **kwargs: Generation overrides.

    Yields
    ------
        Text chunks as they are generated.

    Raises
    ------
        ModelError: If the model cannot be loaded.
        GenerationError: If generation fails.

    Example:
        >>> import talu
        >>> for chunk in talu.stream("Qwen/Qwen3-0.6B", "Tell me a story"):
        ...     print(chunk, end="", flush=True)
    """
    from talu.client import Client

    with Client(model) as client:
        yield from client.stream(prompt, config=config, **kwargs)


def ask(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    config: GenerationConfig | None = None,
    **kwargs: Any,
) -> Response:
    """
    One-shot question-answer with automatic resource cleanup.

    This is the simplest way to get a single response from a model. Resources
    are automatically released when the function returns, preventing memory
    leaks in loops.

    .. warning::

        **Performance: This function loads the model on every call.**

        For repeated queries, use ``Client`` or ``Chat`` instead to avoid
        paying model load time on each call::

            # SLOW: Loads model 100 times
            for q in questions:
                r = talu.ask("Qwen/Qwen3-0.6B", q)  # ~2-5s load + generation

            # FAST: Loads model once
            chat = talu.Chat("Qwen/Qwen3-0.6B")
            for q in questions:
                r = chat(q)  # Just generation time

    The returned Response is "detached" - calling ``.append()`` on it will raise
    an error. For multi-turn conversations, use ``talu.Chat()`` instead.

    Args:
        model: Model identifier (local path, HuggingFace ID, or URI).
        prompt: The message to send (required).
        system: Optional system prompt.
        config: Generation configuration.
        **kwargs: Generation overrides (temperature, max_tokens, etc.).

    Returns
    -------
        Response object (str-able, with metadata access). Cannot be replied to.

    Raises
    ------
        ModelError: If the model cannot be loaded.
        GenerationError: If generation fails.

    Example - Simple question:
        >>> response = talu.ask("Qwen/Qwen3-0.6B", "What is 2+2?")
        >>> print(response)
        4

    Example - With system prompt:
        >>> response = talu.ask(
        ...     "Qwen/Qwen3-0.6B",
        ...     "Hello!",
        ...     system="You are a pirate.",
        ... )
        >>> print(response)
        Ahoy there, matey!

    Example - Safe in loops (no resource leak):
        >>> for question in questions:
        ...     response = talu.ask(model, question)  # Auto-cleans each iteration
        ...     results.append(str(response))

    For repeated use, prefer Chat (loads model once):
        >>> chat = talu.Chat("Qwen/Qwen3-0.6B")
        >>> for question in questions:
        ...     response = chat(question)  # Fast: model already loaded
        ...     results.append(str(response))

    For conversations with append:
        >>> chat = talu.Chat("Qwen/Qwen3-0.6B")
        >>> r1 = chat("Hello!")
        >>> r2 = r1.append("Tell me more")  # Works because Chat is attached
    """
    from talu.client import Client

    # Use context manager to guarantee cleanup
    with Client(model) as client:
        chat_instance = client.chat(system=system, config=config)
        # Use stream=False for one-shot to return Response (not StreamingResponse)
        response = chat_instance.send(prompt, stream=False, **kwargs)
        # Detach response from chat so .append() fails with clear error
        response._chat = None
        response._msg_index = -1
        return response


# =============================================================================
# Async API Design Decision
# =============================================================================
#
# .. design-decision:: No Top-Level Async Convenience Functions
#     :status: DELIBERATE
#     :rationale: Async users are power users who should manage client lifecycle explicitly.
#
# We intentionally do NOT provide top-level async convenience functions like:
#   - talu.achat()
#   - talu.complete_async()
#   - talu.stream_async()
#
# Why?
# 1. **Async users are power users**: They understand `async with` and want
#    explicit control over client lifecycle for connection pooling, etc.
#
# 2. **Resource management risk**: One-shot async functions that spin up and
#    tear down a client per call are wasteful and can leak resources if the
#    coroutine is cancelled mid-execution.
#
# 3. **Namespace pollution**: Doubling the API surface (sync + async variants
#    of every function) adds cognitive overhead without real benefit.
#
# 4. **Naming inconsistency**: Having `AsyncClient` (prefix) alongside
#    `stream_async()` (suffix) is inconsistent. Removing the suffixed
#    functions resolves this.
#
# The "perfect" async API is simply:
#
#     from talu import AsyncClient
#
#     async with AsyncClient("model") as client:
#         response = await client.complete("Hello")
#         async for chunk in client.stream("Hi"): ...
#
# Or for chat:
#
#     from talu import AsyncChat
#
#     chat = AsyncChat("model", system="You are helpful.")
#     response = await chat("Hello!")
#
# This keeps the API clean:
#   - talu.ask(), talu.stream() -> Sync convenience
#   - talu.raw_complete() -> Power user (no chat template)
#   - talu.AsyncChat, talu.AsyncClient -> Async power tools

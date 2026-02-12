"""
Hooks - Observability and instrumentation for LLM inference.

Hooks provide a clean way to add logging, metrics, and tracing to your
LLM applications without modifying business logic. They are called at
key points during generation:

- on_generation_start: Before generation begins
- on_first_token: When the first token arrives (TTFT measurement)
- on_generation_end: After generation completes

Example - Basic metrics collection:
    >>> from talu import Client
    >>> from talu.chat import Hook
    >>>
    >>> class MetricsHook(Hook):
    ...     def on_generation_start(self, chat, input_text):
    ...         self.start_time = time.time()
    ...
    ...     def on_first_token(self, chat, time_ms):
    ...         print(f"TTFT: {time_ms:.1f}ms")
    ...
    ...     def on_generation_end(self, chat, response):
    ...         latency = (time.time() - self.start_time) * 1000
    ...         print(f"Total: {latency:.1f}ms, {response.usage.total_tokens} tokens")
    >>>
    >>> client = Client("Qwen/Qwen3-0.6B", hooks=[MetricsHook()])
    >>> chat = client.chat()
    >>> response = chat("Hello!")

Example - Langfuse/Datadog integration:
    >>> class LangfuseHook(Hook):
    ...     def __init__(self, langfuse):
    ...         self.langfuse = langfuse
    ...
    ...     def on_generation_start(self, chat, input_text):
    ...         self.trace = self.langfuse.trace(name="llm-call")
    ...         self.generation = self.trace.generation(input=input_text)
    ...
    ...     def on_generation_end(self, chat, response):
    ...         self.generation.end(output=str(response))

Example - Multiple hooks:
    >>> client = Client("model", hooks=[MetricsHook(), LoggingHook()])

Note:
    Hooks are called from the Python layer (not Zig), so they measure
    client-side latency which includes Python overhead. For most use cases,
    this is what users care about anyway.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .response import Response
    from .session import AsyncChat, Chat

_hook_logger = logging.getLogger("talu.hooks")


class Hook:
    """
    Base class for generation hooks.

    Implement any subset of these methods to receive callbacks during generation.
    All methods have default no-op implementations, so you only need to override
    the ones you care about.

    Methods are called in this order:
    1. on_generation_start - Before Zig generation begins
    2. on_first_token - When first token arrives (streaming) or N/A (non-streaming)
    3. on_generation_end - After generation completes (success or error)

    Thread Safety:
        Hook methods may be called from different threads for concurrent generations.
        If your hook maintains state, ensure it's thread-safe.
    """

    def on_generation_start(
        self,
        chat: Chat | AsyncChat,
        input_text: str,
        *,
        config: Any = None,
    ) -> None:
        """
        Handle generation start event.

        Args:
            chat: The Chat instance initiating generation.
            input_text: The user's input message.
            config: The GenerationConfig for this request (if any).

        Example:
            >>> def on_generation_start(self, chat, input_text, config=None):
            ...     self.start_time = time.perf_counter()
            ...     self.input_tokens = len(input_text.split())  # Rough estimate
        """

    def on_first_token(
        self,
        chat: Chat | AsyncChat,
        time_ms: float,
    ) -> None:
        """
        Handle first token event (streaming only).

        This is the Time-To-First-Token (TTFT) measurement point, critical for
        perceived latency in interactive applications.

        Args:
            chat: The Chat instance.
            time_ms: Milliseconds since generation_start.

        Note:
            Only called for streaming responses. For non-streaming, TTFT is
            effectively the same as total latency.

        Example:
            >>> def on_first_token(self, chat, time_ms):
            ...     metrics.histogram("llm.ttft", time_ms)
        """

    def on_generation_end(
        self,
        chat: Chat | AsyncChat,
        response: Response | None,
        *,
        error: Exception | None = None,
    ) -> None:
        """
        Handle generation end event (success or error).

        Args:
            chat: The Chat instance.
            response: The Response object (if successful), or None if error.
            error: The exception (if generation failed), or None if successful.

        Example:
            >>> def on_generation_end(self, chat, response, error=None):
            ...     if error:
            ...         metrics.counter("llm.errors", 1)
            ...     else:
            ...         metrics.counter("llm.tokens", response.usage.total_tokens)
        """


class HookManager:
    """
    Hook dispatcher for generation lifecycle events.

    Used internally by Client to dispatch hook calls. Users register
    hooks on the Client rather than interacting with this class directly.
    """

    def __init__(self, hooks: list[Hook] | None = None) -> None:
        self._hooks = hooks or []

    def add(self, hook: Hook) -> None:
        """Add a hook to the manager.

        Args:
            hook: Hook instance to register.
        """
        self._hooks.append(hook)

    def remove(self, hook: Hook) -> None:
        """Remove a hook from the manager.

        Args:
            hook: Hook instance to unregister.
        """
        self._hooks.remove(hook)

    @property
    def hooks(self) -> list[Hook]:
        """Return the list of registered hooks."""
        return self._hooks.copy()

    def dispatch_start(
        self,
        chat: Chat | AsyncChat,
        input_text: str,
        config: Any = None,
    ) -> None:
        """Dispatch on_generation_start to all hooks.

        Args:
            chat: The Chat or AsyncChat instance.
            input_text: The user's input text.
            config: Generation configuration, if any.
        """
        for hook in self._hooks:
            try:
                hook.on_generation_start(chat, input_text, config=config)
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                _hook_logger.debug("Hook %r.on_generation_start raised: %s", hook, exc)

    def dispatch_first_token(
        self,
        chat: Chat | AsyncChat,
        time_ms: float,
    ) -> None:
        """Dispatch on_first_token to all hooks.

        Args:
            chat: The Chat or AsyncChat instance.
            time_ms: Time to first token in milliseconds.
        """
        for hook in self._hooks:
            try:
                hook.on_first_token(chat, time_ms)
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                _hook_logger.debug("Hook %r.on_first_token raised: %s", hook, exc)

    def dispatch_end(
        self,
        chat: Chat | AsyncChat,
        response: Response | None,
        error: Exception | None = None,
    ) -> None:
        """Dispatch on_generation_end to all hooks.

        Args:
            chat: The Chat or AsyncChat instance.
            response: The completed response, or None on error.
            error: The exception that occurred, if any.
        """
        for hook in self._hooks:
            try:
                hook.on_generation_end(chat, response, error=error)
            except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as exc:
                _hook_logger.debug("Hook %r.on_generation_end raised: %s", hook, exc)

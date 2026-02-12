"""Hooks - Observability and instrumentation for LLM inference.

Primary API: talu.chat.Hook, talu.chat.HookManager
Scope: Single

This example demonstrates the Hook system for adding logging, metrics, and
tracing to your LLM applications without modifying business logic.

Key points:
- Hooks receive callbacks at generation start, first token (TTFT), and end
- Multiple hooks can be registered and are called in order
- Hook errors are isolated - one failing hook won't break others
- Perfect for Langfuse, Datadog, Prometheus, or custom metrics

Related:
    - examples/developers/chat/streaming_tokens.py (Token metadata)
"""

import time

import talu
from talu.chat import Hook, HookManager, Client


# =============================================================================
# Basic Metrics Hook
# =============================================================================

class MetricsHook(Hook):
    """Collect timing and token metrics."""

    def __init__(self):
        self.start_time = None
        self.ttft_ms = None
        self.total_ms = None
        self.request_count = 0

    def on_generation_start(self, chat, input_text, *, config=None):
        self.start_time = time.perf_counter()
        self.request_count += 1
        print(f"[MetricsHook] Request #{self.request_count} started")
        print(f"[MetricsHook] Input length: {len(input_text)} chars")

    def on_first_token(self, chat, time_ms):
        self.ttft_ms = time_ms
        print(f"[MetricsHook] TTFT: {time_ms:.1f}ms")

    def on_generation_end(self, chat, response, *, error=None):
        if self.start_time:
            self.total_ms = (time.perf_counter() - self.start_time) * 1000

        if error:
            print(f"[MetricsHook] Error: {error}")
        else:
            print(f"[MetricsHook] Total latency: {self.total_ms:.1f}ms")
            if response and response.usage:
                print(f"[MetricsHook] Tokens: {response.usage.total_tokens}")


# =============================================================================
# Logging Hook
# =============================================================================

class LoggingHook(Hook):
    """Log all generation events for debugging."""

    def __init__(self, prefix="[Log]"):
        self.prefix = prefix

    def on_generation_start(self, chat, input_text, *, config=None):
        preview = input_text[:50] + "..." if len(input_text) > 50 else input_text
        print(f"{self.prefix} START: {preview!r}")

    def on_first_token(self, chat, time_ms):
        print(f"{self.prefix} FIRST_TOKEN at {time_ms:.0f}ms")

    def on_generation_end(self, chat, response, *, error=None):
        if error:
            print(f"{self.prefix} ERROR: {type(error).__name__}: {error}")
        else:
            preview = str(response)[:50] + "..." if len(str(response)) > 50 else str(response)
            print(f"{self.prefix} END: {preview!r}")


# =============================================================================
# Usage with Client
# =============================================================================

print("=== Single Hook ===")
metrics = MetricsHook()

# Note: In real usage, you'd pass hooks to Client:
# client = Client("Qwen/Qwen3-0.6B", hooks=[metrics])
# chat = client.chat()

# For this example, we'll demonstrate HookManager directly
manager = HookManager([metrics])

# Simulate hook dispatch (in real code, this happens internally)
manager.dispatch_start(None, "What is 2+2?")
manager.dispatch_first_token(None, 150.0)
manager.dispatch_end(None, None)


# =============================================================================
# Multiple Hooks
# =============================================================================

print("\n=== Multiple Hooks ===")

logger = LoggingHook("[Debug]")
metrics2 = MetricsHook()

manager = HookManager([logger, metrics2])

manager.dispatch_start(None, "Tell me a joke about programming")
manager.dispatch_first_token(None, 200.0)
manager.dispatch_end(None, None)


# =============================================================================
# Adding/Removing Hooks Dynamically
# =============================================================================

print("\n=== Dynamic Hook Management ===")

manager = HookManager()
print(f"Initial hooks: {len(manager.hooks)}")

# Add hooks
verbose_logger = LoggingHook("[Verbose]")
manager.add(verbose_logger)
print(f"After add: {len(manager.hooks)}")

# Remove hooks
manager.remove(verbose_logger)
print(f"After remove: {len(manager.hooks)}")


# =============================================================================
# Error Isolation
# =============================================================================

print("\n=== Error Isolation ===")

class FailingHook(Hook):
    """A hook that always fails."""
    def on_generation_start(self, chat, input_text, *, config=None):
        raise RuntimeError("Hook failed!")

class SafeHook(Hook):
    """A hook that works correctly."""
    def on_generation_start(self, chat, input_text, *, config=None):
        print("[SafeHook] Called successfully!")

manager = HookManager([FailingHook(), SafeHook()])

# SafeHook still gets called even though FailingHook raises
manager.dispatch_start(None, "test")


# =============================================================================
# Production Pattern: Langfuse/Datadog Integration
# =============================================================================

print("\n=== Production Integration Pattern ===")

class LangfuseHook(Hook):
    """
    Example of Langfuse integration (pseudocode).

    In production, you'd use the actual Langfuse SDK:
        from langfuse import Langfuse
        langfuse = Langfuse()
    """

    def __init__(self, langfuse_client=None):
        self.client = langfuse_client
        self.trace = None
        self.generation = None

    def on_generation_start(self, chat, input_text, *, config=None):
        # self.trace = self.client.trace(name="llm-generation")
        # self.generation = self.trace.generation(
        #     name="chat",
        #     input=input_text,
        #     model=config.model if config else "unknown",
        # )
        print("[Langfuse] Trace started")

    def on_first_token(self, chat, time_ms):
        # self.generation.update(metadata={"ttft_ms": time_ms})
        print(f"[Langfuse] TTFT recorded: {time_ms}ms")

    def on_generation_end(self, chat, response, *, error=None):
        # if error:
        #     self.generation.end(error=str(error))
        # else:
        #     self.generation.end(
        #         output=str(response),
        #         usage={"total_tokens": response.usage.total_tokens}
        #     )
        print("[Langfuse] Generation ended")


# Simulate usage
langfuse_hook = LangfuseHook()
manager = HookManager([langfuse_hook])
manager.dispatch_start(None, "Hello!")
manager.dispatch_first_token(None, 100.0)
manager.dispatch_end(None, None)


"""
Topics covered:
* chat.streaming
* stream.tokens

Related:
* examples/developers/chat/streaming_tokens.py
"""

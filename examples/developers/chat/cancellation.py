"""Cancellation - Stop generation early and handle client disconnects.

Primary API: talu.router.StopFlag
Scope: Single

When streaming responses, you may need to stop generation early:
- User clicks "Stop" button
- Client disconnects (HTTP connection closed)
- Timeout reached
- Error in downstream processing

talu provides cooperative cancellation via StopFlag for sync code
and native asyncio cancellation support for async code.

Key concepts:
- StopFlag: Thread-safe signal to stop generation
- Breaking from stream loop: Automatically signals cancellation
- asyncio.CancelledError: Native async cancellation
- Router cleanup: Active threads are tracked for safe shutdown

Related:
    - examples/recipes/http_server_fastapi.py (disconnect handling)
    - examples/developers/chat/streaming.py
    - examples/developers/chat/async_usage.py
"""

import asyncio
import threading
import time

import talu
from talu.router import StopFlag


# =============================================================================
# StopFlag Basics
# =============================================================================


def stop_flag_basics():
    """Basic StopFlag usage for cooperative cancellation."""
    print("=== StopFlag Basics ===\n")

    chat = talu.Chat("Qwen/Qwen3-0.6B")

    # Create a stop flag
    stop_flag = StopFlag()

    # Signal stop from another thread after a short delay
    def signal_stop():
        time.sleep(0.1)  # Let a few tokens generate
        stop_flag.signal()
        print("\n[Stop signaled!]")

    threading.Thread(target=signal_stop, daemon=True).start()

    # Stream with stop flag - generation stops when signaled
    print("Generating (will stop early): ", end="")
    response = chat("Count from 1 to 100", stream=True, stop_flag=stop_flag)
    for token in response:
        print(token, end="", flush=True)
    print("\n")

    # Reset for reuse
    stop_flag.reset()
    print(f"Stop flag reset, is_set: {stop_flag.is_set}")


# =============================================================================
# Breaking from Stream Loop
# =============================================================================


def break_from_loop():
    """Breaking from stream loop automatically signals cancellation."""
    print("\n=== Break from Loop ===\n")

    chat = talu.Chat("Qwen/Qwen3-0.6B")

    # When you break from a stream, talu automatically:
    # 1. Signals an internal stop flag to stop generation
    # 2. Waits for the generation thread to complete
    # 3. Cleans up resources safely

    print("Generating (will break after 3 tokens): ", end="")
    token_count = 0
    response = chat("Count from 1 to 100", stream=True)
    for token in response:
        print(token, end="", flush=True)
        token_count += 1
        if token_count >= 3:
            break  # This triggers cancellation automatically!

    print(f"\n[Broke after {token_count} tokens - generation stopped cleanly]\n")


# =============================================================================
# Async Cancellation
# =============================================================================


async def async_cancellation():
    """Native asyncio cancellation support."""
    print("=== Async Cancellation ===\n")

    from talu import AsyncChat

    chat = AsyncChat("Qwen/Qwen3-0.6B")

    async def stream_with_timeout():
        """Stream that gets cancelled by timeout."""
        response = await chat("Write a very long essay about the history of computing")
        async for token in response:
            print(token, end="", flush=True)

    # asyncio.timeout cancels the task after the specified time
    print("Generating (2 second timeout): ", end="")
    try:
        async with asyncio.timeout(2.0):
            await stream_with_timeout()
    except asyncio.TimeoutError:
        print("\n[Timed out - generation cancelled cleanly]\n")


async def async_task_cancellation():
    """Cancel an async task programmatically."""
    print("=== Async Task Cancellation ===\n")

    from talu import AsyncChat

    chat = AsyncChat("Qwen/Qwen3-0.6B")

    async def long_generation():
        """A long-running generation task."""
        response = await chat("Count from 1 to 1000, one number per line")
        async for token in response:
            print(token, end="", flush=True)

    # Create a task
    task = asyncio.create_task(long_generation())

    # Let it run for a bit
    await asyncio.sleep(0.5)

    # Cancel the task
    task.cancel()
    print("\n[Task cancelled]")

    # Wait for cancellation to complete
    try:
        await task
    except asyncio.CancelledError:
        print("[CancelledError caught - cleanup complete]\n")


# =============================================================================
# StopFlag with Timeout Pattern
# =============================================================================


def timeout_pattern():
    """Using StopFlag with a timeout thread."""
    print("=== Timeout Pattern ===\n")

    chat = talu.Chat("Qwen/Qwen3-0.6B")
    stop_flag = StopFlag()
    timeout_seconds = 1.0

    # Timeout thread
    def timeout_handler():
        time.sleep(timeout_seconds)
        if not stop_flag.is_set:
            stop_flag.signal()
            print("\n[Timeout reached]")

    timeout_thread = threading.Thread(target=timeout_handler, daemon=True)
    timeout_thread.start()

    print(f"Generating ({timeout_seconds}s timeout): ", end="")
    response = chat("Write a very long story", stream=True, stop_flag=stop_flag)
    for token in response:
        print(token, end="", flush=True)
    print("\n")


# =============================================================================
# Router Close Waits for Active Generations
# =============================================================================


def safe_router_close():
    """Demonstrate that router.close() waits for active generations."""
    print("=== Safe Router Close ===\n")

    from talu.chat import Chat
    from talu.router import Router, GenerationConfig

    router = Router(models=["Qwen/Qwen3-0.6B"])

    # Start a generation in another thread
    tokens_generated = []

    def generate():
        # Router.stream requires a Chat instance and user message
        chat = Chat(router=router)
        config = GenerationConfig(max_tokens=50)
        stream = router.stream(chat, "Count to 10", config=config)
        for token in stream:
            tokens_generated.append(str(token))

    gen_thread = threading.Thread(target=generate)
    gen_thread.start()

    # Small delay to let generation start
    time.sleep(0.1)

    # Close the router - this will wait for the generation to complete
    print("Closing router (will wait for active generation)...")
    router.close()

    print(f"Router closed. Tokens generated: {len(tokens_generated)}")
    print(f"Text: {''.join(tokens_generated)[:50]}...\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all cancellation examples."""
    stop_flag_basics()
    break_from_loop()
    timeout_pattern()
    safe_router_close()

    # Async examples
    print("=== Running Async Examples ===\n")
    asyncio.run(async_cancellation())
    asyncio.run(async_task_cancellation())

    print("=== All cancellation examples complete ===")


if __name__ == "__main__":
    main()

"""
Topics covered:
* chat.streaming
* stream.tokens
"""

"""Async - Non-blocking generation for async applications.

Primary API: talu.chat.AsyncChat, talu.AsyncClient
Scope: Single

talu provides a complete async API through AsyncChat and AsyncClient.
Use these for web servers (FastAPI, aiohttp), async scripts, or any
application using asyncio.

Note: Unlike the sync API which provides convenience functions (talu.ask(),
talu.raw_complete(), talu.stream()), the async API intentionally does NOT provide
top-level async convenience functions. Async users are expected to manage
client lifecycle explicitly via AsyncChat or AsyncClient. This design prevents
resource leaks from one-shot async operations that might be cancelled mid-execution.

Key differences from sync API:
- AsyncChat instead of Chat
- AsyncClient instead of Client
- All generation methods are async (await chat.send(), await chat())
- append() on AsyncResponse/AsyncStreamingResponse is also async
- Use async context managers (async with) for resource cleanup

See recipes/http_server_fastapi.py for a complete web server example.

Related:
    - examples/basics/01_chat.py
    - examples/recipes/http_server_fastapi.py
"""

import asyncio

from talu import AsyncChat, AsyncClient


# =============================================================================
# Basic AsyncChat Usage
# =============================================================================


async def basic_async_generation():
    """Basic async generation with AsyncChat."""
    print("=== Basic Async Generation ===\n")

    # Create an AsyncChat - same interface as Chat
    chat = AsyncChat("Qwen/Qwen3-0.6B", system="You are helpful and concise.")

    # send() is async and non-streaming by default
    response = await chat.send("What is the capital of France?")
    print(f"Response: {response}")
    print(f"Tokens used: {response.usage.total_tokens}\n")


async def async_streaming():
    """Async streaming with AsyncChat."""
    print("=== Async Streaming ===\n")

    chat = AsyncChat("Qwen/Qwen3-0.6B")

    # __call__() streams by default (like sync Chat)
    print("Streaming response: ", end="")
    response = await chat("Count from 1 to 5")
    async for token in response:
        print(token, end="", flush=True)
    print("\n")

    # Or explicitly request streaming with send()
    print("Explicit streaming: ", end="")
    response = await chat.send("Now count backwards from 5 to 1", stream=True)
    async for token in response:
        print(token, end="", flush=True)
    print("\n")


async def async_multi_turn():
    """Multi-turn async conversation with append()."""
    print("=== Async Multi-Turn Conversation ===\n")

    chat = AsyncChat("Qwen/Qwen3-0.6B", system="You are a math tutor.")

    # First message
    response = await chat.send("What is 2 + 2?")
    print(f"Q: What is 2 + 2?")
    print(f"A: {response}\n")

    # append() on AsyncResponse is async - must await
    response = await response.append("And what is that multiplied by 3?")
    print(f"Q: And what is that multiplied by 3?")
    print(f"A: {response}\n")

    # Can chain appends
    response = await response.append("Thanks! One more: divide by 2?")
    print(f"Q: Thanks! One more: divide by 2?")
    print(f"A: {response}\n")


# =============================================================================
# AsyncClient for Multi-User / Resource Management
# =============================================================================


async def async_client_basics():
    """AsyncClient for explicit resource management."""
    print("=== AsyncClient Basics ===\n")

    # AsyncClient with async context manager ensures cleanup
    async with AsyncClient("Qwen/Qwen3-0.6B") as client:
        # Create chats from the client
        chat = client.chat(system="You are helpful.")

        response = await chat.send("Hello!")
        print(f"Response: {response}\n")

        # Client also has direct ask() for one-shot generation
        response = await client.ask("What is 1 + 1?")
        print(f"One-shot: {response}\n")


async def async_multi_user():
    """Multiple users sharing the same model with AsyncClient."""
    print("=== Async Multi-User ===\n")

    async with AsyncClient("Qwen/Qwen3-0.6B") as client:
        # Each user gets their own chat with separate history
        alice = client.chat(system="You are a helpful assistant named Alice.")
        bob = client.chat(system="You are a pirate assistant named Bob.")

        # Users can interact independently
        alice_response = await alice.send("What's your name?")
        print(f"Alice's chat: {alice_response}")

        bob_response = await bob.send("What's your name?")
        print(f"Bob's chat: {bob_response}\n")


# =============================================================================
# Concurrent Execution
# =============================================================================


async def concurrent_requests():
    """Run multiple async requests concurrently."""
    print("=== Concurrent Requests ===\n")

    async with AsyncClient("Qwen/Qwen3-0.6B") as client:
        # Create multiple independent chats
        chats = [
            client.chat(system="Answer in one word."),
            client.chat(system="Answer in one word."),
            client.chat(system="Answer in one word."),
        ]

        questions = [
            "What color is the sky?",
            "What color is grass?",
            "What color is the sun?",
        ]

        # Run all requests concurrently with asyncio.gather()
        responses = await asyncio.gather(*[
            chat.send(question) for chat, question in zip(chats, questions)
        ])

        for question, response in zip(questions, responses):
            print(f"Q: {question}")
            print(f"A: {response}\n")


async def concurrent_streaming():
    """Concurrent streaming responses."""
    print("=== Concurrent Streaming ===\n")

    async with AsyncClient("Qwen/Qwen3-0.6B") as client:

        async def stream_response(name: str, prompt: str):
            """Helper to stream and collect a response."""
            chat = client.chat()
            response = await chat(prompt)  # Streaming
            tokens = []
            async for token in response:
                tokens.append(token)
            return name, "".join(tokens)

        # Start multiple streams concurrently
        results = await asyncio.gather(
            stream_response("Task 1", "Say 'hello' in French"),
            stream_response("Task 2", "Say 'hello' in Spanish"),
            stream_response("Task 3", "Say 'hello' in German"),
        )

        for name, text in results:
            print(f"{name}: {text}")
        print()


# =============================================================================
# Error Handling
# =============================================================================


async def async_error_handling():
    """Proper error handling in async context."""
    print("=== Async Error Handling ===\n")

    try:
        async with AsyncClient("Qwen/Qwen3-0.6B") as client:
            chat = client.chat()
            response = await chat.send("Hello!")
            print(f"Success: {response}\n")

    except Exception as e:
        print(f"Error occurred: {e}\n")
        # Resources are still cleaned up thanks to async context manager


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all async examples."""
    await basic_async_generation()
    await async_streaming()
    await async_multi_turn()
    await async_client_basics()
    await async_multi_user()
    await concurrent_requests()
    await concurrent_streaming()
    await async_error_handling()

    print("=== All async examples complete ===")


if __name__ == "__main__":
    asyncio.run(main())

"""
Topics covered:
* chat.session
* chat.streaming
"""

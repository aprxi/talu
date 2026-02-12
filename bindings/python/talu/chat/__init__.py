"""
Chat module - Stateful multi-turn conversation sessions.

This module provides a layered API for LLM chat:

1. Casual User (module-level functions):

    >>> import talu
    >>> response = talu.ask("Qwen/Qwen3-0.6B", "What is 2+2?")
    >>> response = talu.ask("Qwen/Qwen3-0.6B", "Hello!", system="Pirate")

2. Power User (explicit Client):

    >>> from talu import Client
    >>> client = Client("Qwen/Qwen3-0.6B")
    >>> chat = client.chat(system="You are helpful.")
    >>> response = chat("Hello!")
    >>> response = response.append("Tell me more")

3. Async User (AsyncClient for FastAPI, etc.):

    >>> from talu import AsyncClient
    >>> async with AsyncClient("Qwen/Qwen3-0.6B") as client:
    ...     chat = client.chat(system="You are helpful.")
    ...     response = await chat("Hello!")
    ...     response = await response.append("Tell me more")

Classes:
    Sync Stack (blocking I/O):
    - Client: Entry point for sync LLM inference
    - Chat: Stateful multi-turn chat (sync)
    - Response: Completed generation result with metadata
    - StreamingResponse: Streaming generation result (iterable)

    Async Stack (non-blocking I/O):
    - AsyncClient: Entry point for async LLM inference
    - AsyncChat: Stateful multi-turn chat (async)
    - AsyncResponse: Async completed generation result
    - AsyncStreamingResponse: Async streaming generation result

    - ConversationItems: Read-only view into conversation history (Item-based API)

Module-level functions (convenience wrappers, sync only):
    - ask(): One-shot generation with optional system prompt
    - raw_complete(): Raw completion without chat templates
    - stream(): Streaming stateless generation

Data model types (Items, Content Parts, Enums) live in ``talu.types``.
"""

from ._message_list import MessageList as MessageList
from .api import ask as ask
from .api import raw_complete as raw_complete
from .api import stream as stream
from .hooks import Hook, HookManager
from .items import ConversationItems
from .response import (
    AsyncResponse,
    AsyncStreamingResponse,
    FinishReason,
    Response,
    ResponseFormat,
    ResponseMetadata,
    StreamingResponse,
    Timings,
    Token,
    TokenLogprob,
    Usage,
)
from .session import AsyncChat, Chat
from .tools import (
    ToolCall,
    ToolCallFunction,
    ToolResult,
    ToolState,
    ToolStatus,
)

# =============================================================================
# Public API - See talu/__init__.py for documentation mapping guidelines
# =============================================================================
__all__ = [
    # Sync Core
    "Chat",
    "Response",
    "StreamingResponse",
    # Async Core
    "AsyncChat",
    "AsyncResponse",
    "AsyncStreamingResponse",
    # Shared (Items API)
    "ConversationItems",
    # Hooks (observability)
    "Hook",
    "HookManager",
    # Response Metadata
    "Token",
    "Usage",
    "Timings",
    "FinishReason",
    "TokenLogprob",
    "ResponseMetadata",
    # Response Format
    "ResponseFormat",
    # Tool Calling
    "ToolCall",
    "ToolCallFunction",
    "ToolResult",
    "ToolState",
    "ToolStatus",
]

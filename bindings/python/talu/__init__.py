"""
Talu - Fast LLM inference in pure Python.

Talu runs large language models locally with a simple Python API.
No PyTorch, no dependencies beyond the standard library.

Quick Start
-----------

Chat is all you need:

    >>> from talu import Chat
    >>>
    >>> chat = Chat("Qwen/Qwen3-0.6B", system="You are a pirate.")
    >>> response = chat("Hello!")
    >>> print(response)
    Ahoy there, matey!

Multi-turn conversations with append():

    >>> response = chat("What is 2+2?")
    >>> response = response.append("Why?")
    >>> response = response.append("Are you sure?")

Streaming:

    >>> response = chat("Tell me about your ship", stream=True)
    >>> for chunk in response:
    ...     print(chunk, end="", flush=True)

One-liners (for single questions):

    >>> import talu
    >>> response = talu.ask("Qwen/Qwen3-0.6B", "What is 2+2?")


Multi-User Serving
------------------

For production with many concurrent users, use Client:

    >>> from talu import Client
    >>>
    >>> # Load model once (at server startup)
    >>> client = Client("Qwen/Qwen3-0.6B")
    >>>
    >>> # Create lightweight chats per user
    >>> user1 = client.chat(system="You are helpful.")
    >>> user2 = client.chat(system="You are a pirate.")
    >>>
    >>> # Generate responses (uses batched inference internally)
    >>> response = user1("Hello!")
    >>> response = user2("Ahoy!")
    >>>
    >>> client.close()


Async Support
-------------

For async applications (FastAPI, aiohttp, etc.), use AsyncClient:

    >>> from talu import AsyncClient
    >>>
    >>> async with AsyncClient("Qwen/Qwen3-0.6B") as client:
    ...     chat = client.chat(system="You are helpful.")
    ...     response = await chat("Hello!")
    ...     response = await response.append("Tell me more")


Core Classes
------------

Sync Stack (blocking I/O):
- `Chat` - Primary interface for casual users (pass model to load)
- `Client` - Power user interface (manages model, creates multiple chats)

Async Stack (non-blocking I/O):
- `AsyncChat` - Async interface for async applications
- `AsyncClient` - Async power user interface for async apps

Utilities:
- `Tokenizer` - Tokenize text without loading model weights
- `PromptTemplate` - Prompt templates with Jinja2 syntax
- `convert` - Convert models to optimized formats


When to Use What
----------------

**Use `Chat(model=...)`** for:
- Simple scripts and prototyping
- Single-user applications
- The easiest possible API

**Use `Client(model=...)`** for:
- Serving multiple users (one client, many chats)
- Efficient batched inference
- Building a sync chat server or API

**Use `AsyncClient(model=...)`** for:
- Async applications (FastAPI, aiohttp)
- Non-blocking inference
- Building an async chat server or API


Examples
--------
Simple chat:

    >>> chat = Chat("Qwen/Qwen3-0.6B", system="You are a math tutor.")
    >>> response = chat("What is 2+2?")
    >>> print(response)
    4
    >>> response = response.append("And 3+3?")
    >>> print(response)
    6

Restoring history from saved data:

    >>> # Load saved messages from database/file
    >>> saved_messages = [
    ...     {"role": "user", "content": "Hello"},
    ...     {"role": "assistant", "content": "Hi there!"},
    ... ]
    >>> chat = Chat.from_dict({"messages": saved_messages}, model="Qwen/Qwen3-0.6B")
    >>> response = chat("How are you?")  # Continues from history

Multi-user server pattern:

    >>> client = Client("Qwen/Qwen3-0.6B")
    >>> user_chats = {}  # Store in Redis/DB in production
    >>>
    >>> def handle_message(user_id: str, message: str) -> str:
    ...     if user_id not in user_chats:
    ...         user_chats[user_id] = client.chat()
    ...     return str(user_chats[user_id](message))

Async server pattern:

    >>> async with AsyncClient("Qwen/Qwen3-0.6B") as client:
    ...     async def handle_message(user_id: str, message: str) -> str:
    ...         chat = client.chat(system="You are helpful.")
    ...         response = await chat(message)
    ...         return str(response)

Convert a model to 4-bit quantized format:

    >>> path = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_64")
    >>> chat = Chat(path)


Supported Models
----------------

talu supports many popular model architectures:

- Qwen2, Qwen2.5, Qwen3
- LLaMA 2, LLaMA 3, LLaMA 3.2
- Mistral, Ministral
- Gemma 2, Gemma 3
- Phi-3, Phi-4
- And more (see documentation)

Models can be specified as:
- Model IDs: "Qwen/Qwen3-0.6B" (downloaded automatically)
- Local paths: "./models/my-model"
"""

# Version from _version.py (synced from VERSION file at build time)
# Logging configuration
from talu._bindings import get_lib as _get_lib
from talu._version import __version__ as __version__

# Chat - Sync
from talu.chat import Chat

# Chat - Sync convenience functions
from talu.chat.api import ask
from talu.chat.api import raw_complete as raw_complete
from talu.chat.api import stream as stream
from talu.chat.session import AsyncChat

# Chat - Async (classes only - async users should manage client lifecycle explicitly)
from talu.client import AsyncClient, Client
from talu.profile import Profile

# Converter
from talu.converter import convert

# Validate
from talu.db import Database

# Exceptions (commonly-used exceptions at root; all via talu.exceptions)
from talu.exceptions import (
    ConvertError as ConvertError,
)
from talu.exceptions import (
    GenerationError as GenerationError,
)
from talu.exceptions import (
    InteropError as InteropError,
)
from talu.exceptions import (
    IOError as IOError,
)
from talu.exceptions import (
    ModelError as ModelError,
)
from talu.exceptions import (
    StateError as StateError,
)
from talu.exceptions import (
    TaluError,
)
from talu.exceptions import (
    TemplateError as TemplateError,
)
from talu.exceptions import (
    TokenizerError as TokenizerError,
)
from talu.exceptions import (
    ValidationError as ValidationError,
)

# Configuration / Model Specification
from talu.router import (
    CompletionOptions as CompletionOptions,
)
from talu.router import (
    GenerationConfig as GenerationConfig,
)
from talu.router import (
    Grammar as Grammar,
)
from talu.router import (
    LocalBackend as LocalBackend,
)
from talu.router import (
    ModelSpec as ModelSpec,
)
from talu.router import (
    OpenAICompatibleBackend as OpenAICompatibleBackend,
)

# Template
from talu.template import PromptTemplate, TemplateEnvironment

# Tokenizer
from talu.tokenizer import Tokenizer
from talu.validate import Validator


def list_sessions(
    *,
    profile: str | None = None,
    search: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List sessions in a profile.

    Args:
        profile: Profile name, or None for the default profile.
        search: Filter sessions by text content.
        limit: Maximum number of sessions to return.
    """
    return Profile(profile).sessions(search=search, limit=limit)


def set_log_level(level: str) -> None:
    """Set logging verbosity level.

    Args:
        level: One of 'trace', 'debug', 'info', 'warn', 'error', 'off'.
               Default is 'warn' (silent operation).

    Example:
        >>> import talu
        >>> talu.set_log_level('debug')  # Enable debug output
        >>> talu.set_log_level('warn')   # Back to silent (default)
    """
    levels = {"trace": 1, "debug": 5, "info": 9, "warn": 13, "error": 17, "off": 255}
    level_int = levels.get(level.lower(), 13)  # Default to warn
    _get_lib().talu_set_log_level(level_int)


def set_log_format(fmt: str) -> None:
    """Set logging output format.

    Args:
        fmt: Either 'json' (machine-readable) or 'human' (readable).
             Default is 'json'.

    Example:
        >>> import talu
        >>> talu.set_log_format('human')  # Pretty output
        >>> talu.set_log_format('json')   # Structured JSON
    """
    format_int = 0 if fmt.lower() == "json" else 1
    _get_lib().talu_set_log_format(format_int)


# __version__ imported from _version.py above

# =============================================================================
# Public API - Mapped 1:1 to Documentation
# =============================================================================
#
# This __all__ defines what appears in the docs sidebar navigation.
# The docs generator (docs/scripts/build.py) parses this list preserving:
#   - Exact order of exports
#   - Comments as section headers (e.g., "# Chat" becomes a nav label)
#
# Guidelines for maintainers:
#   - Only add symbols that deserve top-level documentation
#   - Use comments to group related exports into sections
#   - Sections should match examples/ structure: chat, tokenizer, template, converter
#   - Other symbols remain importable via submodules (e.g., from talu.router import Router)
#
__all__ = [
    # Chat
    "Chat",
    "Client",
    "AsyncChat",
    "AsyncClient",
    "ask",
    "Profile",
    "list_sessions",
    # Tokenizer
    "Tokenizer",
    # Template
    "PromptTemplate",
    "TemplateEnvironment",
    # Converter
    "convert",
    # Validate
    "Validator",
    # Database
    "Database",
    # Exceptions
    "TaluError",
]

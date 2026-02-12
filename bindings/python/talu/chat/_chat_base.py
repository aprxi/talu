"""ChatBase - Shared implementation for Chat and AsyncChat.

This module contains the base class with all methods shared between
the sync Chat and async AsyncChat classes. Both classes inherit from
ChatBase to avoid code duplication.

The generation methods (send, __call__, _generate_sync/_generate_async)
remain in the subclasses since they have fundamentally different
sync/async signatures.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any, TypeVar, overload

from . import _bindings as _c

if TYPE_CHECKING:
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self
else:
    Self = TypeVar("Self")

from talu.router.config import GenerationConfig, Grammar, SchemaStrategy
from talu.types import MessageItem

from .._bindings import check
from .._native import ChatCreateOptions
from ..db import Database
from ..exceptions import (
    GenerationError,
    SchemaValidationError,
    StateError,
    ValidationError,
)
from ..template import PromptTemplate
from ._bindings import get_chat_lib
from .items import ConversationItems
from .response import AsyncStreamingResponse, Response, StreamingResponse

if TYPE_CHECKING:
    from talu.client import AsyncClient, Client
    from talu.router import Router
    from talu.types import ItemRecord

SCHEMA_PLACEHOLDER = "{{ schema }}"


class ChatBase:
    """Base class containing shared Chat/AsyncChat implementation.

    This class is not meant to be instantiated directly. Use Chat or AsyncChat.

    Contains all methods that are identical between sync and async versions:
    - Message manipulation: pop, remove, insert, append, clear, reset
    - Properties: items, messages, system, session_id, client, router, etc.
    - Serialization: to_dict, from_dict, to_json
    - Utilities: count_tokens, preview_prompt, fork, archive

    Subclasses must implement:
    - Generation methods (send, __call__)
    - Async context management (AsyncChat)
    - Sync context management (Chat)
    """

    # These are set by subclass __init__
    _lib: Any
    _chat_ptr: Any
    _conversation_ptr: Any
    _client: Client | AsyncClient | None
    _router: Router | None
    _owns_client: bool
    _model_id: str | None
    _system: str | None
    _session_id: str | None
    _parent_session_id: str | None
    _group_id: str | None
    _ttl_ts: int | None
    _marker: str
    _metadata: dict
    _source_doc_id: str | None
    _prompt_id: str | None
    _defer_session_update: bool
    _chat_template: PromptTemplate | None
    _storage: Database
    _last_response: Response | StreamingResponse | AsyncStreamingResponse | None
    config: GenerationConfig

    # =========================================================================
    # Initialization helper (called from subclass __init__)
    # =========================================================================

    def _init_base(
        self,
        *,
        system: str | None,
        session_id: str | None,
        parent_session_id: str | None,
        group_id: str | None,
        ttl_ts: int | None,
        marker: str,
        metadata: dict | None,
        source_doc_id: str | None,
        prompt_id: str | None,
        chat_template: str | PromptTemplate | None,
        storage: Database | None,
        config: GenerationConfig | None,
        offline: bool,
        _defer_session_update: bool,
    ) -> None:
        """Initialize base state shared by Chat and AsyncChat.

        Called from subclass __init__ after client/router setup.
        """
        use_taludb = isinstance(storage, Database) and storage.location.startswith("talu://")
        self._lib = get_chat_lib()

        # Session state
        self._system = system
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._group_id = group_id
        self._ttl_ts = ttl_ts
        self._marker = marker
        self._metadata = metadata or {}
        self._source_doc_id = source_doc_id
        self._prompt_id = prompt_id
        self._defer_session_update = _defer_session_update

        # Create Zig Chat handle
        options = ChatCreateOptions(offline=offline)
        create_system = None if use_taludb else system
        self._chat_ptr = _c.chat_create(self._lib, create_system, session_id, options)

        if not self._chat_ptr:
            raise MemoryError("Failed to create Chat")

        if ttl_ts is not None:
            result = self._lib.talu_chat_set_ttl_ts(self._chat_ptr, int(ttl_ts))
            if result != 0:
                raise StateError(
                    f"Failed to set ttl_ts: {result}",
                    code="STATE_SET_TTL_FAILED",
                )

        # Get Conversation handle from Chat
        self._conversation_ptr = self._lib.talu_chat_get_conversation(self._chat_ptr)
        if not self._conversation_ptr:
            self._lib.talu_chat_free(self._chat_ptr)
            raise MemoryError("Failed to get Conversation from Chat")

        # Config
        if config is not None:
            from dataclasses import replace

            self.config = replace(config)
        else:
            self.config = GenerationConfig()

        # Chat template
        if isinstance(chat_template, str):
            self._chat_template = PromptTemplate(chat_template)
        else:
            self._chat_template = chat_template

        # Storage
        self._storage = storage if storage is not None else Database()

        if self._storage.location.startswith("talu://"):
            if not self._session_id:
                raise ValidationError("TaluDB requires session_id to be set.")

            db_path = self._storage.location[len("talu://") :]
            if not db_path:
                raise ValidationError("TaluDB location must include a path after 'talu://'.")

            result = self._lib.talu_chat_set_storage_db(
                self._chat_ptr,
                db_path.encode("utf-8"),
                self._session_id.encode("utf-8"),
            )
            check(result, {"db_path": db_path, "session_id": self._session_id})

            if len(self.items) == 0:
                if system is not None:
                    self.system = system
            else:
                self._system = self.items.system

        # Set prompt_id if provided
        if prompt_id is not None:
            rc = self._lib.talu_chat_set_prompt_id(self._chat_ptr, prompt_id.encode("utf-8"))
            if rc != 0:
                from .._bindings import get_last_error

                error = get_last_error() or "unknown error"
                self._lib.talu_chat_free(self._chat_ptr)
                raise StateError(f"Failed to set prompt_id: {error}", code="SET_PROMPT_ID_FAILED")

        # Persist session metadata when using TaluDB
        if use_taludb and not _defer_session_update:
            rc = self._lib.talu_chat_notify_session_update(
                self._chat_ptr,
                None,  # model
                None,  # title
                None,  # system_prompt
                None,  # config_json
                None,  # marker
                None,  # parent_session_id
                None,  # group_id
                None,  # metadata_json
                source_doc_id.encode("utf-8") if source_doc_id else None,
            )
            if rc != 0:
                from .._bindings import get_last_error

                error = get_last_error() or "unknown error"
                self._lib.talu_chat_free(self._chat_ptr)
                raise StateError(
                    f"Failed to persist session: {error}", code="PERSIST_SESSION_FAILED"
                )

        # Track last response
        self._last_response = None

    # =========================================================================
    # Config building
    # =========================================================================

    def _build_effective_config(
        self,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> GenerationConfig:
        """Build effective config by merging session default, config override, and kwargs.

        Priority (highest to lowest):
        1. **kwargs (per-call overrides like temperature=0.1)
        2. config parameter (explicit GenerationConfig object)
        3. self.config (session default)
        """
        from dataclasses import fields

        effective = self.config

        if config is not None:
            effective = config

        if kwargs:
            config_dict = {f.name: getattr(effective, f.name) for f in fields(effective)}
            for key, value in kwargs.items():
                if key in config_dict:
                    config_dict[key] = value
                else:
                    raise ValidationError(
                        f"Unknown generation parameter: {key}",
                        code="INVALID_ARGUMENT",
                        details={"param": key},
                    )
            effective = GenerationConfig(**config_dict)

        if effective.chat_template is None and self._chat_template is not None:
            effective = effective.override(chat_template=self._chat_template)

        return effective

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def items(self) -> ConversationItems:
        """Read-only access to conversation as typed Items."""
        if self._conversation_ptr is None:
            raise StateError("Chat is closed", code="CHAT_CLOSED")
        return ConversationItems(self._lib, self._conversation_ptr)

    @property
    def messages(self) -> list:
        """Read-only view of conversation as standard OpenAI-format messages."""
        from ._message_list import MessageList

        return MessageList(
            item.to_message_dict() for item in self.items if isinstance(item, MessageItem)
        )

    @property
    def session_id(self) -> str | None:
        """The session identifier for this conversation."""
        return self._session_id

    @property
    def source_doc_id(self) -> str | None:
        """The source document ID for lineage tracking.

        Links this conversation to the prompt/persona document that spawned it.
        Used for tracking which document was used to create the conversation.
        """
        return self._source_doc_id

    @source_doc_id.setter
    def source_doc_id(self, doc_id: str | None) -> None:
        """Set the source document ID for lineage tracking.

        Args:
            doc_id: Document ID to link, or None to clear lineage.

        Raises
        ------
            StateError: If chat is closed.
        """
        if self._chat_ptr is None:
            raise StateError("Chat is closed", code="CHAT_CLOSED")

        # Notify session update with source_doc_id
        rc = self._lib.talu_chat_notify_session_update(
            self._chat_ptr,
            None,  # model
            None,  # title
            None,  # system_prompt
            None,  # config_json
            None,  # marker
            None,  # parent_session_id
            None,  # group_id
            None,  # metadata_json
            doc_id.encode("utf-8") if doc_id else None,  # source_doc_id
        )
        if rc != 0:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise StateError(f"Failed to set source_doc_id: {error}", code="SET_SOURCE_DOC_FAILED")

        self._source_doc_id = doc_id

    @property
    def prompt_id(self) -> str | None:
        """The prompt document ID for this conversation.

        When set, this links the conversation to a prompt/persona document.
        The prompt document can provide the system prompt content and tags
        that can be inherited via inherit_tags().

        This is stored on the Chat object and used by inherit_tags().
        For persistent lineage tracking in session records, use source_doc_id.
        """
        if self._chat_ptr is None:
            return self._prompt_id  # Return cached value if closed

        result_ptr = self._lib.talu_chat_get_prompt_id(self._chat_ptr)
        if not result_ptr:
            return None
        try:
            import ctypes

            value = ctypes.cast(result_ptr, ctypes.c_char_p).value
            return value.decode("utf-8") if value else None
        finally:
            self._lib.talu_text_free(result_ptr)

    @prompt_id.setter
    def prompt_id(self, doc_id: str | None) -> None:
        """Set the prompt document ID.

        Args:
            doc_id: Document ID to link, or None to clear.

        Raises
        ------
            StateError: If chat is closed.
        """
        if self._chat_ptr is None:
            raise StateError("Chat is closed", code="CHAT_CLOSED")

        rc = self._lib.talu_chat_set_prompt_id(
            self._chat_ptr, doc_id.encode("utf-8") if doc_id else None
        )
        if rc != 0:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise StateError(f"Failed to set prompt_id: {error}", code="SET_PROMPT_ID_FAILED")

        self._prompt_id = doc_id

    def inherit_tags(self) -> None:
        """Copy tags from the prompt document to this conversation.

        Requires both prompt_id and session_id to be set, and requires
        TaluDB storage to be configured.

        Raises
        ------
            StateError: If chat is closed.
            ValidationError: If prompt_id or session_id is not set.
            IOError: If tag inheritance fails.
        """
        if self._chat_ptr is None:
            raise StateError("Chat is closed", code="CHAT_CLOSED")

        if not self._storage.location.startswith("talu://"):
            raise ValidationError("inherit_tags requires TaluDB storage (talu://...)")

        db_path = self._storage.location[len("talu://") :]
        rc = self._lib.talu_chat_inherit_tags(self._chat_ptr, db_path.encode("utf-8"))
        if rc != 0:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise StateError(f"Failed to inherit tags: {error}", code="INHERIT_TAGS_FAILED")

    @property
    def system(self) -> str | None:
        """Get the system prompt."""
        if self._chat_ptr is None:
            raise StateError("Chat is closed", code="CHAT_CLOSED")

        result_ptr = self._lib.talu_chat_get_system(self._chat_ptr)
        return _c.read_c_text_result(self._lib, result_ptr)

    @system.setter
    def system(self, content: str | None) -> None:
        """Set or replace the system prompt."""
        if self._chat_ptr is None:
            raise StateError("Chat is closed", code="CHAT_CLOSED")

        if content is None:
            content_bytes = b""
        else:
            content_bytes = content.encode("utf-8")

        rc = self._lib.talu_chat_set_system(self._chat_ptr, content_bytes)
        if rc != 0:
            from .._bindings import get_last_error

            error = get_last_error() or "unknown error"
            raise StateError(f"Failed to set system prompt: {error}", code="SET_SYSTEM_FAILED")

        # Refresh _conversation_ptr after setSystem
        self._conversation_ptr = self._lib.talu_chat_get_conversation(self._chat_ptr)
        self._system = content

    @property
    def owns_client(self) -> bool:
        """True if this Chat owns its Client (standalone mode)."""
        return self._owns_client

    @property
    def chat_template(self) -> PromptTemplate | None:
        """Get the custom chat template, if any."""
        return self._chat_template

    @property
    def last_response(self) -> Response | StreamingResponse | AsyncStreamingResponse | None:
        """Get the last response from generation."""
        return self._last_response

    @property
    def client(self) -> Client | AsyncClient | None:
        """Get the client used by this chat."""
        return self._client

    @property
    def router(self) -> Router | None:
        """Get the router used by this chat."""
        return self._router

    # =========================================================================
    # Token counting
    # =========================================================================

    def count_tokens(self, message: str | None = None) -> int:
        """Count tokens in current history or a hypothetical message.

        Args:
            message: Optional message to count. If None, counts current history.

        Returns
        -------
            Token count.

        Raises
        ------
            StateError: If no model configured.
            GenerationError: If token counting fails.
        """
        from .._bindings import get_last_error

        model_path = self._router.default_model if self._router else None
        if model_path is None:
            raise StateError(
                "Cannot count tokens: no model configured. "
                "Create Chat with model='...' to enable token counting.",
                code="STATE_NO_MODEL",
            )

        model_bytes = model_path.encode("utf-8")
        msg_bytes = message.encode("utf-8") if message else None
        msg_len = len(msg_bytes) if msg_bytes else 0

        result = self._lib.talu_chat_count_tokens(
            self._chat_ptr,
            model_bytes,
            msg_bytes,
            msg_len,
        )

        if result < 0:
            error = get_last_error() or "unknown error"
            raise GenerationError(
                f"Failed to count tokens: {error}",
                code="TOKEN_COUNT_FAILED",
            )

        return result

    @property
    def max_context_length(self) -> int | None:
        """Get the model's maximum context length."""
        model_path = self._router.default_model if self._router else None
        if model_path is None:
            return None

        model_bytes = model_path.encode("utf-8")
        result = self._lib.talu_chat_max_context_length(model_bytes)
        return result if result > 0 else None

    # =========================================================================
    # Message manipulation
    # =========================================================================

    def clear(self) -> Self:
        """Clear conversation history (keeps system prompt and settings).

        Returns
        -------
            self, for chaining.
        """
        self._lib.talu_responses_clear_keeping_system(self._conversation_ptr)
        return self

    def reset(self) -> Self:
        """Reset everything including system prompt.

        Returns
        -------
            self, for chaining.
        """
        self._lib.talu_responses_clear(self._conversation_ptr)
        return self

    def pop(self) -> Self:
        """Remove and discard the last message.

        Returns
        -------
            self, for chaining.

        Raises
        ------
            StateError: If no messages to remove.
        """
        item_count = len(self.items)
        if item_count == 0:
            raise StateError(
                "No messages to remove (chat is empty or only has system)",
                code="STATE_POP_FAILED",
            )

        result = self._lib.talu_responses_pop(self._conversation_ptr)
        if result != 0:
            raise StateError(
                "No messages to remove (chat is empty or only has system)",
                code="STATE_POP_FAILED",
            )
        return self

    def remove(self, index: int) -> Self:
        """Remove message at the specified index.

        Args:
            index: Index of message to remove (0-based).

        Returns
        -------
            self, for chaining.

        Raises
        ------
            StateError: If index is out of bounds.
        """
        item_count = len(self.items)
        if index < 0 or index >= item_count:
            raise StateError(
                f"Index {index} out of bounds",
                code="STATE_REMOVE_FAILED",
            )

        result = self._lib.talu_responses_remove(self._conversation_ptr, index)
        if result != 0:
            raise StateError(
                f"Index {index} out of bounds",
                code="STATE_REMOVE_FAILED",
            )
        return self

    def set_item_parent(self, item_index: int, parent_item_id: int | None) -> None:
        """Set parent_item_id for an item by index.

        Raises
        ------
            StateError: If the operation fails.
        """
        parent_id = parent_item_id if parent_item_id is not None else 0
        has_parent = 1 if parent_item_id is not None else 0
        result = self._lib.talu_responses_set_item_parent(
            self._conversation_ptr, item_index, parent_id, has_parent
        )
        if result != 0:
            raise StateError(
                f"Failed to set parent for item {item_index}: {result}",
                code="STATE_SET_PARENT_FAILED",
            )

    def set_item_validation_flags(
        self,
        item_index: int,
        *,
        json_valid: bool,
        schema_valid: bool,
        repaired: bool = False,
    ) -> None:
        """Set structured validation flags for an item by index.

        Use this after structured parsing/validation to mark JSON/schema validity.

        Raises
        ------
            StateError: If the operation fails.
        """
        result = self._lib.talu_responses_set_item_validation_flags(
            self._conversation_ptr,
            int(item_index),
            bool(json_valid),
            bool(schema_valid),
            bool(repaired),
        )
        if result != 0:
            raise StateError(
                f"Failed to set validation flags: {result}",
                code="STATE_SET_VALIDATION_FAILED",
            )

    def insert(self, index: int, role: str, content: str, *, hidden: bool = False) -> Self:
        """Insert a message at the specified index.

        Args:
            index: Position to insert at (0-based).
            role: Message role ("system", "user", "assistant", "developer").
            content: Message text content.
            hidden: Hide from UI history while keeping in LLM context.

        Returns
        -------
            self, for chaining.

        Raises
        ------
            ValidationError: If role is invalid.
            StateError: If index is out of bounds or insert fails.
        """
        role_map = {"system": 0, "user": 1, "assistant": 2, "developer": 3}
        role_int = role_map.get(role.lower())
        if role_int is None:
            raise ValidationError(
                f"Invalid role: {role}. Must be one of: system, user, assistant, developer",
                code="VALIDATION_INVALID_ROLE",
            )

        result = _c.responses_insert_message(
            self._lib, self._conversation_ptr, index, role_int, content, hidden
        )
        if result < 0:
            raise StateError(
                f"Failed to insert message at index {index}: error code {result}",
                code="STATE_INSERT_FAILED",
            )
        return self

    def append_hidden(self, role: str, content: str) -> Self:
        """Append a hidden message to the conversation.

        Hidden messages are included in LLM context but omitted from UI history.

        Args:
            role: Message role ("system", "user", "assistant", "developer").
            content: Message text content.

        Returns
        -------
            Self for method chaining.

        Raises
        ------
            ValidationError: If role is not one of the valid roles.
            StateError: If the append operation fails.
        """
        role_map = {"system": 0, "user": 1, "assistant": 2, "developer": 3}
        role_int = role_map.get(role.lower())
        if role_int is None:
            raise ValidationError(
                f"Invalid role: {role}. Must be one of: system, user, assistant, developer",
                code="VALIDATION_INVALID_ROLE",
            )

        result = _c.responses_append_message(
            self._lib, self._conversation_ptr, role_int, content, hidden=True
        )
        if result < 0:
            raise StateError(
                f"Failed to append hidden message: error code {result}",
                code="STATE_APPEND_FAILED",
            )
        return self

    # =========================================================================
    # Fork
    # =========================================================================

    def _truncate_to(self, msg_index: int) -> Self:
        """Truncate message history to keep only messages up to and including msg_index."""
        current_len = len(self)
        messages_to_remove = current_len - (msg_index + 1)
        if messages_to_remove > 0:
            result = self._lib.talu_responses_truncate_after(self._conversation_ptr, msg_index)
            if result != 0:
                raise StateError(
                    f"Failed to truncate messages: {result}",
                    code="STATE_TRUNCATE_FAILED",
                )
        return self

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        config_dict = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "min_p": self.config.min_p,
            "repetition_penalty": self.config.repetition_penalty,
        }
        if self.config.chat_template is not None:
            chat_template = self.config.chat_template
            source = getattr(chat_template, "source", None)
            if source is not None:
                config_dict["chat_template"] = source
            else:
                config_dict["chat_template"] = chat_template
        if self.config.extra_context is not None:
            config_dict["extra_context"] = self.config.extra_context
        return {
            "config": config_dict,
            "messages": self._get_completions_messages(),
        }

    def to_json(self) -> str:
        """Get messages as JSON string (from Zig).

        Returns
        -------
            JSON string of messages array in OpenAI Completions format.
            This is an interchange format and does not include storage-only metadata.
        """
        result_ptr = self._lib.talu_responses_to_completions_json(self._conversation_ptr)
        return _c.read_c_json_result(self._lib, result_ptr, default="[]")

    def _get_completions_messages(self) -> list[dict]:
        """Get messages in OpenAI completions format.

        Internal helper for template rendering and serialization.
        """
        return json.loads(self.to_json())

    def _load_storage_records(self, items: list[ItemRecord]) -> None:
        """Load items from storage records."""
        from .session import _build_c_storage_records

        if not items:
            return

        records, keepalive = _build_c_storage_records(items)
        result = self._lib.talu_responses_load_storage_records(
            self._conversation_ptr,
            records,
            len(items),
        )
        if result != 0:
            raise StateError(
                f"Failed to load storage records: {result}",
                code="STATE_LOAD_FAILED",
            )
        _ = keepalive

    # =========================================================================
    # Append methods
    # =========================================================================

    @overload
    def append(self, role_or_item: str, content: str, *, hidden: bool = False) -> Self: ...

    @overload
    def append(
        self, role_or_item: MessageItem, content: None = None, *, hidden: bool = False
    ) -> Self: ...

    def append(
        self,
        role_or_item: str | MessageItem,
        content: str | None = None,
        *,
        hidden: bool = False,
    ) -> Self:
        """Append a message to the conversation.

        Can be called with either:
        - Two arguments: ``append(role, content)`` - role string and content string
        - One argument: ``append(item)`` - a MessageItem object

        Args:
            role_or_item: Either a role string ("system", "user", "assistant", "developer")
                or a MessageItem object.
            content: Message content (required when first arg is a role string).
            hidden: Hide from UI history while keeping in LLM context.

        Returns
        -------
            self, for chaining.

        Raises
        ------
            ValidationError: If role is invalid or arguments are malformed.
            StateError: If append fails.
        """
        # Handle MessageItem overload
        if isinstance(role_or_item, MessageItem):
            role_int = int(role_or_item.role)
            text_content = role_or_item.text
        else:
            # Handle (role, content) overload
            if content is None:
                raise ValidationError(
                    "content is required when appending with role string",
                    code="VALIDATION_MISSING_CONTENT",
                )

            role_map = {"system": 0, "user": 1, "assistant": 2, "developer": 3}
            role_int = role_map.get(role_or_item.lower())
            if role_int is None:
                raise ValidationError(
                    f"Invalid role: {role_or_item}. Must be one of: system, user, assistant, developer",
                    code="VALIDATION_INVALID_ROLE",
                )
            text_content = content

        # Get fresh conversation pointer to avoid stale pointers
        conv_ptr = self._lib.talu_chat_get_conversation(self._chat_ptr)
        if conv_ptr is None:
            raise StateError("Chat has been closed", code="STATE_CLOSED")
        self._conversation_ptr = conv_ptr

        result = _c.responses_append_message(self._lib, conv_ptr, role_int, text_content, hidden)
        if result < 0:
            raise StateError(
                f"Failed to append message: error code {result}",
                code="STATE_APPEND_FAILED",
            )
        return self

    # =========================================================================
    # Preview prompt
    # =========================================================================

    def preview_prompt(
        self,
        add_generation_prompt: bool = True,
        config: GenerationConfig | None = None,
    ) -> str:
        r"""Return the exact formatted prompt that would be sent to the model.

        This is a read-only inspection tool for debugging template logic
        or verifying system prompts. It does NOT send anything to the engine
        or affect the conversation state.

        Args:
            add_generation_prompt: If True (default), include the assistant
                turn marker at the end (e.g., "<|im_start|>assistant\n").
            config: Optional GenerationConfig. If provided and contains a
                chat_template, that template will be used instead of the
                session-level or model default template.

        Returns
        -------
            The formatted prompt string.

        Raises
        ------
            StateError: If no engine is available and no custom template is set.
        """
        # Check for per-request chat_template override in config
        effective_config = self._build_effective_config(config)
        if effective_config.chat_template is not None:
            template = effective_config.chat_template
            if isinstance(template, str):
                template = PromptTemplate(template)
            return template.render(
                messages=self._get_completions_messages(),
                add_generation_prompt=add_generation_prompt,
            )

        # Use session-level custom template
        if self._chat_template is not None:
            return self._chat_template.render(
                messages=self._get_completions_messages(),
                add_generation_prompt=add_generation_prompt,
            )

        # Use model's default template via router
        if self._router is not None:
            get_engine = getattr(self._router, "_get_engine", None)
            if get_engine is None:
                raise StateError(
                    "Router does not support prompt preview. "
                    "Provide a custom chat_template to use preview_prompt().",
                    code="STATE_NO_ENGINE",
                )
            engine = get_engine()
            return engine.tokenizer.apply_chat_template(
                self._get_completions_messages(),
                add_generation_prompt=add_generation_prompt,
            )

        raise StateError(
            "Cannot preview prompt: no router or custom template available. "
            "Create Chat with model='...' or provide a chat_template.",
            code="STATE_NO_ROUTER",
        )

    # =========================================================================
    # Generation helpers
    # =========================================================================

    def _prepare_messages(
        self,
        content: str,
        response_format: type | dict | Grammar | None = None,
        allow_thinking: bool = False,
        inject_schema_prompt: bool = True,
        schema_strategy: SchemaStrategy = "auto",
        model_name: str | None = None,
        model_type: str | None = None,
    ) -> list[dict]:
        """Prepare messages list, optionally injecting schema."""
        from talu.router.schema.convert import normalize_response_format
        from talu.template.schema.injection import schema_to_prompt_description

        def _extract_text(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                for part in value:
                    if part.get("type") == "text":
                        return part.get("text", "")
            return ""

        # Get existing messages from conversation
        existing_messages = self._get_completions_messages()

        # Find system content
        system_content = self._system or ""
        system_index: int | None = None
        if existing_messages and existing_messages[0].get("role") == "system":
            system_index = 0
            if not system_content:
                system_content = _extract_text(existing_messages[0].get("content"))
        elif not system_content:
            system_content = self.items.system or ""

        schema_prompt = None
        if (
            response_format is not None
            and inject_schema_prompt
            and not isinstance(response_format, Grammar)
        ):
            schema = normalize_response_format(response_format)
            if schema is not None:
                schema = self._apply_numeric_const(schema, content)

                schema_prompt = schema_to_prompt_description(
                    schema,
                    allow_thinking=allow_thinking,
                    strategy=schema_strategy,
                    model_name=model_name,
                    model_type=model_type,
                )

        if schema_prompt:
            if SCHEMA_PLACEHOLDER in system_content:
                system_content = system_content.replace(SCHEMA_PLACEHOLDER, schema_prompt)
            elif SCHEMA_PLACEHOLDER in content:
                content = content.replace(SCHEMA_PLACEHOLDER, schema_prompt)
            elif system_content:
                system_content = f"{system_content}\n\n{schema_prompt}"
            elif content:
                content = f"{schema_prompt}\n\n{content}"
            elif not system_content and not content:
                system_content = schema_prompt
            elif not system_content:
                system_content = schema_prompt

        messages: list[dict] = []
        if system_index is not None:
            for idx, msg in enumerate(existing_messages):
                if idx == system_index:
                    updated = dict(msg)
                    updated["content"] = system_content
                    messages.append(updated)
                else:
                    messages.append(dict(msg))
        else:
            if system_content:
                messages.append({"role": "system", "content": system_content})
            for msg in existing_messages:
                messages.append(dict(msg))

        messages.append({"role": "user", "content": content})

        return messages

    def _resolve_stop_tokens(self, model_name: str | None) -> set[int]:
        """Resolve EOS/EOT stop tokens as a set of token IDs."""
        if not model_name:
            return set()
        try:
            from ..tokenizer import Tokenizer

            tokenizer = Tokenizer(model_name)
            return set(tokenizer.eos_token_ids)
        except (ImportError, ValueError, RuntimeError, OSError):
            return set()

    def _apply_numeric_const(self, schema: dict, message: str | None) -> dict:
        """Inject a const number from the prompt when schema is numeric-only."""
        if not message:
            return schema
        if schema.get("const") is not None or schema.get("enum") is not None:
            return schema
        if schema.get("type") not in ("number", "integer"):
            return schema

        import re

        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", message)
        if not match:
            return schema

        value_str = match.group(0)
        schema_copy = dict(schema)
        schema_copy["x-talu-const-literal"] = value_str
        if schema.get("type") == "integer":
            try:
                schema_copy["const"] = int(value_str)
                return schema_copy
            except ValueError:
                return schema

        try:
            schema_copy["const"] = float(value_str)
            return schema_copy
        except ValueError:
            return schema

    def _detect_prefill_prefix(self, allow_thinking: bool) -> str | None:
        """Detect assistant prefix for prompt prefilling."""
        if allow_thinking:
            return None
        try:
            prompt_without = self.preview_prompt(add_generation_prompt=False)
            prompt_with = self.preview_prompt(add_generation_prompt=True)
        except (ImportError, ValueError, RuntimeError, OSError, StateError):
            return None
        if prompt_with.startswith(prompt_without):
            prefix = prompt_with[len(prompt_without) :]
            return prefix or None
        return None

    # =========================================================================
    # Validation retry helper
    # =========================================================================

    def _format_validation_retry_message(self, error: SchemaValidationError) -> str:
        """Format a validation error into a retry prompt for self-correction."""
        validation_msg = str(error.validation_error)
        return (
            f"Your previous response was valid JSON but failed validation:\n"
            f"{validation_msg}\n\n"
            f"Please try again with a corrected response that satisfies the constraints."
        )

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __len__(self) -> int:
        """Get number of messages in conversation."""
        return len(self.items)

    def __repr__(self) -> str:
        """Get string representation."""
        parts = []
        if self._client is not None:
            parts.append(f"model={self._client.default_model!r}")
        mode = "standalone" if self._owns_client else "attached"
        parts.append(f"mode={mode!r}")
        parts.append(f"items={len(self)}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    # =========================================================================
    # Serialization class methods
    # =========================================================================

    @classmethod
    def from_dict(
        cls,
        data: dict,
        model: str | None = None,
    ) -> Self:
        """
        Deserialize from dictionary.

        Restores a Chat/AsyncChat from a dict created by to_dict(). Use this
        to resume conversations from a database or file.

        Args:
            data: Dict from to_dict(). If ``items`` is provided, full ItemRecord
                data is loaded; otherwise only OpenAI-format messages are restored.
            model: Model to load (HuggingFace ID or local path).

        Returns
        -------
            New instance with restored state.

        Raises
        ------
            StateError: If message loading fails.
        """
        cfg = data.get("config", {})
        config = GenerationConfig(
            max_tokens=cfg.get("max_tokens", 256),
            temperature=cfg.get("temperature", 0.7),
            top_k=cfg.get("top_k", 50),
            top_p=cfg.get("top_p", 0.9),
            min_p=cfg.get("min_p", 0.0),
            repetition_penalty=cfg.get("repetition_penalty", 1.0),
            chat_template=cfg.get("chat_template"),
            extra_context=cfg.get("extra_context"),
        )

        chat = cls(model=model, config=config)  # type: ignore[call-arg]

        items = data.get("items")
        if items is not None:
            chat._load_storage_records(items)
        else:
            messages = data.get("messages", [])
            if messages:
                messages_json = json.dumps(messages)
                result = chat._lib.talu_responses_load_completions_json(
                    chat._conversation_ptr, messages_json.encode("utf-8")
                )
                if result != 0:
                    raise StateError(
                        f"Failed to load messages: {result}",
                        code="STATE_LOAD_FAILED",
                    )

        chat._system = chat.items.system
        return chat

    @classmethod
    def _from_items(
        cls,
        items: list[ItemRecord],
        model: str | None = None,
        *,
        config: GenerationConfig | None = None,
    ) -> Self:
        """Restore from ItemRecord dictionaries (full fidelity)."""
        chat = cls(model=model, config=config)  # type: ignore[call-arg]
        chat._load_storage_records(items)
        chat._system = chat.items.system
        return chat

    # =========================================================================
    # Fork helpers
    # =========================================================================

    def _create_fork(self, clone_fn_name: str, clone_args: tuple) -> Self:
        """Create a forked copy using the specified clone function.

        Args:
            clone_fn_name: Name of the C API clone function to call
            clone_args: Arguments to pass after (dest_ptr, src_ptr)
        """
        from dataclasses import replace

        new = self.__class__(  # type: ignore[reportCallIssue]
            client=self._client,  # type: ignore[reportCallIssue]
            config=replace(self.config),  # type: ignore[reportCallIssue]
            chat_template=self._chat_template,  # type: ignore[reportCallIssue]
            storage=self._storage,  # type: ignore[reportCallIssue]
            session_id=None,  # type: ignore[reportCallIssue]
            parent_session_id=self._session_id,  # type: ignore[reportCallIssue]
            group_id=self._group_id,  # type: ignore[reportCallIssue]
            ttl_ts=self._ttl_ts,  # type: ignore[reportCallIssue]
            marker="",  # type: ignore[reportCallIssue]
            metadata=dict(self._metadata),  # type: ignore[reportCallIssue]
            _defer_session_update=True,  # type: ignore[reportCallIssue]
        )
        clone_fn = getattr(self._lib, clone_fn_name)
        result = clone_fn(new._conversation_ptr, self._conversation_ptr, *clone_args)
        check(result)
        return new

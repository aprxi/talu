"""
Type definitions for Open Responses storage records.

This module provides type definitions for the portable storage format:

- ItemRecord: Portable item format for storage (Items/Responses API)
- SessionRecord: Portable session metadata
- ContentPart variants: Various content types for items
- ItemVariant records: Polymorphic item data types
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

__all__ = [
    # Type aliases
    "RecordItemType",
    "RecordItemStatus",
    "RecordMessageRole",
    "RecordContentType",
    # Content part variants
    "InputTextContent",
    "InputImageContent",
    "InputAudioContent",
    "InputVideoContent",
    "InputFileContent",
    "OutputTextContent",
    "RefusalContent",
    "TextContent",
    "ReasoningTextContent",
    "SummaryTextContent",
    "UnknownContent",
    "ContentPart",
    # Item variant records
    "MessageItemVariant",
    "FunctionCallVariant",
    "FunctionCallOutputVariant",
    "ReasoningVariant",
    "ItemReferenceVariant",
    # Main records
    "ItemRecord",
    "SessionRecord",
]


# =============================================================================
# Type Aliases (matching Zig enums as string literals)
# =============================================================================

RecordItemType = Literal[
    "message",
    "function_call",
    "function_call_output",
    "reasoning",
    "item_reference",
    "unknown",
]

RecordItemStatus = Literal[
    "in_progress",
    "waiting",
    "completed",
    "incomplete",
    "failed",
]

RecordMessageRole = Literal[
    "system",
    "user",
    "assistant",
    "developer",
    "unknown",
]

RecordContentType = Literal[
    "input_text",
    "input_image",
    "input_audio",
    "input_video",
    "input_file",
    "output_text",
    "refusal",
    "text",
    "reasoning_text",
    "summary_text",
    "unknown",
]


# =============================================================================
# ContentPart - Content part variants for Items
# =============================================================================


class InputTextContent(TypedDict):
    """Input text content (user/system/developer messages)."""

    type: Literal["input_text"]
    text: str


class InputImageContent(TypedDict, total=False):
    """Input image content."""

    type: Literal["input_image"]  # Required
    image_url: str  # Required
    detail: Literal["auto", "low", "high"]  # Optional, default "auto"


class InputAudioContent(TypedDict):
    """Input audio content."""

    type: Literal["input_audio"]
    audio_data: str


class InputVideoContent(TypedDict):
    """Input video content."""

    type: Literal["input_video"]
    video_url: str


class InputFileContent(TypedDict, total=False):
    """Input file content."""

    type: Literal["input_file"]  # Required
    filename: str
    file_data: str
    file_url: str


class OutputTextContent(TypedDict, total=False):
    """Output text content (assistant responses)."""

    type: Literal["output_text"]  # Required
    text: str  # Required
    logprobs_json: str  # JSON-encoded logprobs array
    annotations_json: str  # JSON-encoded annotations array


class RefusalContent(TypedDict):
    """Refusal content (content filter triggered)."""

    type: Literal["refusal"]
    refusal: str


class TextContent(TypedDict):
    """Generic text content (used in reasoning)."""

    type: Literal["text"]
    text: str


class ReasoningTextContent(TypedDict):
    """Reasoning text content (chain-of-thought)."""

    type: Literal["reasoning_text"]
    text: str


class SummaryTextContent(TypedDict):
    """Summary text content (reasoning summary)."""

    type: Literal["summary_text"]
    text: str


class UnknownContent(TypedDict):
    """Unknown content type (forward compatibility)."""

    type: Literal["unknown"]
    raw_type: str
    raw_data: str


# Union of all content part types
ContentPart = (
    InputTextContent
    | InputImageContent
    | InputAudioContent
    | InputVideoContent
    | InputFileContent
    | OutputTextContent
    | RefusalContent
    | TextContent
    | ReasoningTextContent
    | SummaryTextContent
    | UnknownContent
)


# =============================================================================
# ItemVariant Records - Polymorphic item data for storage
# =============================================================================


class MessageItemVariant(TypedDict):
    """
    Message item variant (user, assistant, system, developer).

    Matches Zig's MessageItemRecord.
    """

    role: RecordMessageRole
    status: RecordItemStatus
    content: Sequence[ContentPart]


class FunctionCallVariant(TypedDict):
    """
    Function/tool call intent from assistant.

    Matches Zig's FunctionCallRecord.
    """

    call_id: str
    name: str
    arguments: str
    status: RecordItemStatus


class FunctionCallOutputVariant(TypedDict):
    """
    Function/tool call output/result.

    Matches Zig's FunctionCallOutputRecord.
    """

    call_id: str
    output: Sequence[ContentPart]
    status: RecordItemStatus


class ReasoningVariant(TypedDict, total=False):
    """
    Reasoning item (chain-of-thought, o1/o3 models).

    Matches Zig's ReasoningRecord.
    """

    content: Sequence[ContentPart]  # Required
    summary: Sequence[ContentPart]  # Required
    encrypted_content: str  # Optional (for encrypted reasoning)
    status: RecordItemStatus  # Required


class ItemReferenceVariant(TypedDict, total=False):
    """
    Reference to a previous item (for context replay).

    Matches Zig's ItemReferenceRecord.
    """

    id: str  # Required
    status: RecordItemStatus  # Required, default "completed"


# =============================================================================
# ItemRecord - Portable item format for storage
# =============================================================================


class ItemRecord(TypedDict, total=False):
    """
    Portable item snapshot for storage backends.

    This is the format used in StorageEvent for persistence operations.
    It's a frozen, self-contained snapshot of an item that can be
    serialized, stored, and restored.

    Matches Zig's ItemRecord struct and the Open Responses API format.

    Required Fields:
        item_id: Session-scoped monotonic item identity (never changes after creation)
            Chosen for fast array indexing in Zig and clustered storage by (session_id, item_id).
        created_at_ms: Unix timestamp in milliseconds when item was finalized
        status: Item status ("completed", "incomplete", "failed", "in_progress", "waiting")
        item_type: Item type discriminator
        variant: Polymorphic item data (one of the variant types)

    Optional Fields:
        input_tokens: Tokens in the input prompt (if available)
        output_tokens: Tokens in the generated output (if available)
        hidden: If true, hide from UI history but keep for LLM context
        pinned: If true, prioritize for context retention
        json_valid: True if output parsed as valid JSON
        schema_valid: True if output passed schema validation
        repaired: True if output was repaired before validation
        parent_item_id: Parent item ID for edits/regenerations
        origin_session_id: Origin session ID for fork lineage (see note below)
        origin_item_id: Origin item ID for fork lineage (see note below)
        finish_reason: Reason generation stopped (e.g., "stop", "length")
        prefill_ns: Prefill time in nanoseconds
        generation_ns: Generation time in nanoseconds
        ttl_ts: Expiration timestamp (Unix ms). 0 = no expiry
        metadata: JSON object for developer metadata

    Fork Lineage (origin_session_id, origin_item_id):
        These fields track where an item originated when forking conversations:

        - If the source chat had a session_id, origin points to the original item
        - If the source chat was ephemeral (no session_id), origin is null/None
        - If the item was previously forked, the original origin is preserved

        Null origin means the item is "original" to this session (not forked from
        a persistent source). This is intentional for ephemeral-to-persistent
        transitions where no prior identity exists to reference.

    Example:
        >>> # Simple text message
        >>> record: ItemRecord = {
        ...     "item_id": 0,
        ...     "created_at_ms": 1705123456789,
        ...     "status": "completed",
        ...     "hidden": False,
        ...     "pinned": False,
        ...     "parent_item_id": None,
        ...     "finish_reason": "stop",
        ...     "prefill_ns": 123456,
        ...     "generation_ns": 654321,
        ...     "item_type": "message",
        ...     "variant": {
        ...         "role": "user",
        ...         "status": "completed",
        ...         "content": [{"type": "input_text", "text": "Hello!"}],
        ...     },
        ... }
        >>> # Function call
        >>> record: ItemRecord = {
        ...     "item_id": 1,
        ...     "created_at_ms": 1705123456800,
        ...     "status": "completed",
        ...     "input_tokens": 42,
        ...     "output_tokens": 128,
        ...     "item_type": "function_call",
        ...     "variant": {
        ...         "call_id": "call_abc123",
        ...         "name": "get_weather",
        ...         "arguments": '{"city": "London"}',
        ...         "status": "completed",
        ...     },
        ... }
        >>> # With metadata
        >>> record: ItemRecord = {
        ...     "item_id": 2,
        ...     "created_at_ms": 1705123456900,
        ...     "status": "completed",
        ...     "item_type": "message",
        ...     "variant": {
        ...         "role": "assistant",
        ...         "status": "completed",
        ...         "content": [{"type": "output_text", "text": "Response"}],
        ...     },
        ...     "metadata": {"custom_key": "custom_value"},
        ... }
    """

    item_id: int  # Stable item identity (monotonic, set by Zig)
    created_at_ms: int  # Unix timestamp in milliseconds (set by Zig at finalize)
    status: str
    hidden: bool
    pinned: bool
    json_valid: bool
    schema_valid: bool
    repaired: bool
    parent_item_id: int | None
    origin_session_id: str | None
    origin_item_id: int | None
    finish_reason: str | None
    prefill_ns: int
    generation_ns: int
    ttl_ts: int
    input_tokens: int
    output_tokens: int
    item_type: RecordItemType  # Required: discriminator
    variant: (
        MessageItemVariant
        | FunctionCallVariant
        | FunctionCallOutputVariant
        | ReasoningVariant
        | ItemReferenceVariant
        | dict  # For forward compatibility
    )
    metadata: dict  # Optional: developer metadata


# =============================================================================
# SessionRecord - Portable session metadata for storage
# =============================================================================


class SessionRecord(TypedDict, total=False):
    """
    Portable session metadata for storage backends.

    This is the format used in PutSession storage events. It contains
    session-level metadata that should be stored separately from
    individual items.

    Zig is the single source of truth for session metadata. This ensures
    that GenerationConfig (temperature, max_tokens, etc.) is portable
    across Python, Rust, and any future bindings.

    Fields:
        session_id: Session identifier (user-provided or generated)
        model: Model identifier (optional)
        title: Human-readable title (optional)
        system_prompt: System prompt text (optional)
        config: GenerationConfig as dict (temperature, max_tokens, etc.)
        marker: Session marker (optional: "pinned", "archived", "deleted"; empty = normal)
        parent_session_id: Parent session identifier (optional)
        group_id: Group identifier for multi-tenant listing (optional)
        head_item_id: Latest item_id in the session (0 when no items yet)
        ttl_ts: Expiration timestamp (Unix ms). 0 = no expiry
        metadata: Session metadata dict (optional)
        search_snippet: Search result snippet text (optional, present in search results)
        source_doc_id: Document ID that spawned this session (optional, for lineage)
        created_at_ms: Unix timestamp in milliseconds when session was created
        updated_at_ms: Unix timestamp in milliseconds when session was last updated

    Example:
        >>> record: SessionRecord = {
        ...     "session_id": "user_123",
        ...     "title": "Weather Chat",
        ...     "system_prompt": "You are a helpful weather assistant.",
        ...     "config": {"temperature": 0.7, "max_tokens": 1024},
        ...     "created_at_ms": 1705123456000,
        ...     "updated_at_ms": 1705123456789,
        ... }
    """

    session_id: str
    model: str
    title: str
    system_prompt: str
    config: dict  # GenerationConfig fields
    marker: str
    parent_session_id: str
    group_id: str
    head_item_id: int
    ttl_ts: int
    metadata: dict
    search_snippet: str
    source_doc_id: str
    created_at_ms: int
    updated_at_ms: int

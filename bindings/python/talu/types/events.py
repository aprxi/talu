"""
Event types for storage operations.

This module provides event types for storage backend operations:

- StorageEvent: Event union for storage operations (PutItems, DeleteItem, ClearItems, etc.)
- DeleteItemEvent: Event for item deletion
- ClearItemsEvent: Event for clearing items
- ForkEvent: Event for fork transaction boundaries
"""

from __future__ import annotations

from typing import TypedDict

from .records import ItemRecord, SessionRecord

__all__ = [
    "DeleteItemEvent",
    "ClearItemsEvent",
    "ForkEvent",
    "StorageEvent",
]


class DeleteItemEvent(TypedDict):
    """Event for deleting an item."""

    item_id: int  # Stable item identity to delete
    deleted_at_ms: int  # Unix timestamp in milliseconds


class ClearItemsEvent(TypedDict):
    """Event for clearing items."""

    cleared_at_ms: int  # Unix timestamp in milliseconds
    keep_context: bool  # Whether to keep system/developer messages


class ForkEvent(TypedDict):
    """Event for fork transaction boundaries."""

    fork_id: int
    session_id: str


class StorageEvent(TypedDict, total=False):
    """
    Event union for storage operations.

    Exactly one of the event type fields will be present.

    Matches Zig's StorageEvent union.

    Event Types:
        PutItems: Items were finalized (list of ItemRecord - enables transaction batching)
        DeleteItem: An item was deleted (contains item_id, deleted_at_ms)
        ClearItems: All items were cleared (contains cleared_at_ms, keep_context)
        PutSession: Session metadata was set/updated (contains SessionRecord)
        BeginFork: Begin a fork transaction boundary (contains fork_id, session_id)
        EndFork: End a fork transaction boundary (contains fork_id, session_id)

    Note: PutItems is always a list, even for single items. This design enables
    efficient transaction batching when handling parallel tool calls (e.g., when
    a model calls 3 functions simultaneously, all 3 can be wrapped in a single
    database transaction instead of 3 separate commits).

    Fork Performance:
        During fork operations, items are deep-copied in Zig memory, then emitted
        as PutItems events in batches (default 1000 items per batch). Cost is O(n)
        where n is total content size including multimodal data (images, audio).
        For large conversations with many images, consider the memory/CPU cost.

        BeginFork/EndFork events enable transactional semantics in your storage:
        wrap all PutItems between them in a database transaction for atomicity.

    Example:
        >>> # PutItems event (single item - most common case)
        >>> event: StorageEvent = {
        ...     "PutItems": [{
        ...         "item_id": 0,
        ...         "created_at_ms": 1705123456789,
        ...         "item_type": "message",
        ...         "variant": {
        ...             "role": "user",
        ...             "status": "completed",
        ...             "content": [{"type": "input_text", "text": "Hello!"}],
        ...         },
        ...     }]
        ... }
        >>> # PutItems event (parallel tool calls - future)
        >>> event: StorageEvent = {
        ...     "PutItems": [
        ...         {"item_id": 1, "item_type": "function_call", ...},
        ...         {"item_id": 2, "item_type": "function_call", ...},
        ...         {"item_id": 3, "item_type": "function_call", ...},
        ...     ]
        ... }
        >>> # DeleteItem event
        >>> event: StorageEvent = {
        ...     "DeleteItem": {"item_id": 0, "deleted_at_ms": 1705123456900}
        ... }
        >>> # ClearItems event
        >>> event: StorageEvent = {
        ...     "ClearItems": {"cleared_at_ms": 1705123457000, "keep_context": True}
        ... }
        >>> # PutSession event
        >>> event: StorageEvent = {
        ...     "PutSession": {
        ...         "session_id": "user_123",
        ...         "title": "Weather Chat",
        ...         "config": {"temperature": 0.7},
        ...         "created_at_ms": 1705123456000,
        ...         "updated_at_ms": 1705123456789,
        ...     }
        ... }
        >>> # Begin/End fork events (transaction boundary)
        >>> event = {"BeginFork": {"fork_id": 1, "session_id": "user_123"}}
        >>> event = {"EndFork": {"fork_id": 1, "session_id": "user_123"}}
    """

    PutItems: list[ItemRecord]
    DeleteItem: DeleteItemEvent
    ClearItems: ClearItemsEvent
    PutSession: SessionRecord
    BeginFork: ForkEvent
    EndFork: ForkEvent

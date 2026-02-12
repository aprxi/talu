"""
Open Responses data model.

This module provides the data definitions for conversation history and storage
records. It is the Python equivalent of Zig's ``core/src/responses/`` module.

The types here define the "shape" of conversations, enabling ``talu.db`` and
``talu.chat`` to share a common language without importing each other.

Three layers:

1. **Runtime Items** - Live objects in memory (frozen dataclasses):
   ``Item``, ``MessageItem``, ``FunctionCallItem``, content parts, enums.

2. **Storage Records** - Portable snapshots for persistence (TypedDicts):
   ``ItemRecord``, ``SessionRecord``, variant records.

3. **Storage Events** - Persistence operations:
   ``StorageEvent``, ``DeleteItemEvent``, ``ClearItemsEvent``, ``ForkEvent``.

Example:
    >>> from talu.types import MessageItem, MessageRole, InputText
    >>> item = MessageItem.create("user", "Hello!")
    >>> item.role == MessageRole.USER
    True
"""

# 1. Runtime Items (Live objects in memory)
# Zig: items.zig
# 3. Storage Events
# Zig: backend.zig (StorageEvent)
from .events import (
    ClearItemsEvent as ClearItemsEvent,
)
from .events import (
    DeleteItemEvent as DeleteItemEvent,
)
from .events import (
    ForkEvent as ForkEvent,
)
from .events import (
    StorageEvent as StorageEvent,
)
from .items import (
    # MIME maps
    AUDIO_MIME_MAP as AUDIO_MIME_MAP,
)
from .items import (
    IMAGE_MIME_MAP as IMAGE_MIME_MAP,
)
from .items import (
    VIDEO_MIME_MAP as VIDEO_MIME_MAP,
)
from .items import (
    CodeBlock,
    ContentPart,
    ContentType,
    ConversationItem,
    FinishReason,
    FunctionCallItem,
    FunctionCallOutputItem,
    ImageDetail,
    InputAudio,
    InputFile,
    InputImage,
    InputText,
    InputVideo,
    Item,
    ItemReferenceItem,
    ItemStatus,
    ItemType,
    MessageItem,
    MessageRole,
    OutputText,
    ReasoningItem,
    ReasoningText,
    Refusal,
    SummaryText,
    Text,
    UnknownContent,
    UnknownItem,
)
from .items import (
    # Helper functions
    normalize_content as normalize_content,
)
from .items import (
    normalize_message_input as normalize_message_input,
)

# 2. Storage Records (Portable snapshots)
# Zig: backend.zig (ItemRecord, SessionRecord)
from .records import (  # noqa: F401
    ContentPart as RecordContentPart,
)
from .records import (
    FunctionCallOutputVariant as FunctionCallOutputVariant,
)
from .records import (
    FunctionCallVariant as FunctionCallVariant,
)
from .records import (
    InputAudioContent as InputAudioContent,
)
from .records import (
    InputFileContent as InputFileContent,
)
from .records import (
    InputImageContent as InputImageContent,
)
from .records import (
    InputTextContent as InputTextContent,
)
from .records import (
    InputVideoContent as InputVideoContent,
)
from .records import (
    ItemRecord as ItemRecord,
)
from .records import (
    ItemReferenceVariant as ItemReferenceVariant,
)
from .records import (
    MessageItemVariant as MessageItemVariant,
)
from .records import (
    OutputTextContent as OutputTextContent,
)
from .records import (
    ReasoningTextContent as ReasoningTextContent,
)
from .records import (
    ReasoningVariant as ReasoningVariant,
)
from .records import (
    RecordContentType as RecordContentType,
)
from .records import (
    RecordItemStatus as RecordItemStatus,
)
from .records import (
    RecordItemType as RecordItemType,
)
from .records import (
    RecordMessageRole as RecordMessageRole,
)
from .records import (
    RefusalContent as RefusalContent,
)
from .records import (
    SessionRecord,
)
from .records import (
    SummaryTextContent as SummaryTextContent,
)
from .records import (
    TextContent as TextContent,
)
from .records import (  # noqa: F401
    UnknownContent as RecordUnknownContent,
)

__all__ = [
    # Enums
    "ItemType",
    "ItemStatus",
    "MessageRole",
    "ContentType",
    "ImageDetail",
    "FinishReason",
    # Content Parts
    "ContentPart",
    "InputText",
    "InputImage",
    "InputAudio",
    "InputVideo",
    "InputFile",
    "OutputText",
    "Refusal",
    "Text",
    "ReasoningText",
    "SummaryText",
    "UnknownContent",
    "CodeBlock",
    # Items
    "Item",
    "MessageItem",
    "FunctionCallItem",
    "FunctionCallOutputItem",
    "ReasoningItem",
    "ItemReferenceItem",
    "UnknownItem",
    # Type Aliases
    "ConversationItem",
    # Storage Records (returned by Database.list_sessions())
    "SessionRecord",
]

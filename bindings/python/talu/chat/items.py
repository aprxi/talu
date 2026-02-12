"""
Items - FFI bridge and container for Open Responses Items.

This module provides:
- ConversationItems: Read-only Sequence view into Zig conversation memory.
- FFI helpers for reading C structs into Python dataclasses.

Data types (Item, MessageItem, ContentPart, enums, etc.) live in
``talu.types.items``.

See Also
--------
talu.types : The canonical home for all data model types.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any, overload

# Data types used by FFI helpers and ConversationItems
from talu.types.items import (
    CodeBlock,
    ContentPart,
    ContentType,
    ConversationItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    ImageDetail,
    InputAudio,
    InputFile,
    InputImage,
    InputText,
    InputVideo,
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

# Import FFI functions from bindings module (ctypes usage is encapsulated there)
from . import _bindings as _c

__all__ = [
    "ConversationItems",
]


# =============================================================================
# FFI Helpers
# =============================================================================


def _read_content_part_dict(part_dict: dict[str, Any]) -> ContentPart:
    """Convert a content part dict (from _bindings) to a Python content part object.

    Args:
        part_dict: Dict with content_type, data, secondary, tertiary, image_detail.

    Returns
    -------
        The appropriate ContentPart subclass instance.
    """
    content_type = ContentType(part_dict["content_type"])
    data = part_dict["data"]
    secondary = part_dict["secondary"]
    tertiary = part_dict["tertiary"]
    image_detail = part_dict["image_detail"]

    match content_type:
        case ContentType.INPUT_TEXT:
            return InputText(text=data)
        case ContentType.INPUT_IMAGE:
            return InputImage(image_url=data, detail=ImageDetail(image_detail))
        case ContentType.INPUT_AUDIO:
            return InputAudio(audio_data=data)
        case ContentType.INPUT_VIDEO:
            return InputVideo(video_url=data)
        case ContentType.INPUT_FILE:
            return InputFile(
                filename=secondary if secondary else None,
                file_data=data if data else None,
            )
        case ContentType.OUTPUT_TEXT:
            # Parse annotations, logprobs, and code_blocks from JSON
            annotations = None
            if secondary:
                try:
                    annotations = json.loads(secondary)
                except json.JSONDecodeError:
                    pass
            logprobs = None
            if tertiary:
                try:
                    logprobs = json.loads(tertiary)
                except json.JSONDecodeError:
                    pass
            code_blocks: list[CodeBlock] | None = None
            quaternary = part_dict.get("quaternary")
            if quaternary:
                try:
                    raw_blocks = json.loads(quaternary)
                    if isinstance(raw_blocks, list):
                        code_blocks = [CodeBlock.from_dict(b) for b in raw_blocks]
                except json.JSONDecodeError:
                    pass
            return OutputText(
                text=data, annotations=annotations, logprobs=logprobs, code_blocks=code_blocks
            )
        case ContentType.REFUSAL:
            return Refusal(refusal=data)
        case ContentType.TEXT:
            return Text(text=data)
        case ContentType.REASONING_TEXT:
            return ReasoningText(text=data)
        case ContentType.SUMMARY_TEXT:
            return SummaryText(text=data)
        case _:
            return UnknownContent(raw_type=secondary, raw_data=data)


def _read_content_part(part: _c.CContentPart) -> ContentPart:
    """Read a CContentPart struct into a Python content part object.

    This function is provided for backwards compatibility with tests.
    The FFI layer now uses _read_content_part_dict with dicts from _bindings.

    Args:
        part: CContentPart struct from _native.

    Returns
    -------
        The appropriate ContentPart subclass instance.
    """
    # Convert C struct to dict and delegate to _read_content_part_dict
    part_dict = {
        "content_type": part.content_type,
        "data": _c.read_c_string(part.data_ptr, part.data_len),
        "secondary": _c.read_c_string(part.secondary_ptr, part.secondary_len),
        "tertiary": _c.read_c_string(part.tertiary_ptr, part.tertiary_len),
        "image_detail": part.image_detail,
    }
    return _read_content_part_dict(part_dict)


# =============================================================================
# ConversationItems - Native Conversation C API backed Items
# =============================================================================


class ConversationItems(Sequence["ConversationItem"]):
    """
    Read-only view into conversation history.

    Reads items directly from the Conversation C API, providing typed
    access without the Messages adapter layer. Items are read on-demand
    using zero-copy access to the underlying storage.

    Example:
        >>> # Access items via chat.items
        >>> for item in chat.items:
        ...     if isinstance(item, MessageItem):
        ...         print(f"{item.role.name}: {item.text}")
        ...     elif isinstance(item, FunctionCallItem):
        ...         print(f"Tool call: {item.name}({item.arguments})")
    """

    def __init__(self, lib: Any, conversation_ptr: int) -> None:
        """
        Initialize ConversationItems view from a Conversation handle.

        Args:
            lib: The loaded talu shared library.
            conversation_ptr: Pointer to the Conversation handle.
        """
        self._lib = lib
        self._conversation_ptr = conversation_ptr

    def __len__(self) -> int:
        """Return the number of items."""
        return _c.items_get_count(self._lib, self._conversation_ptr)

    @overload
    def __getitem__(self, index: int) -> ConversationItem: ...

    @overload
    def __getitem__(self, index: slice) -> list[ConversationItem]: ...

    def __getitem__(self, index: int | slice) -> ConversationItem | list[ConversationItem]:
        """Get item(s) by index or slice."""
        if isinstance(index, slice):
            # Handle slice
            length = len(self)
            indices = range(*index.indices(length))
            return [self._read_item(i) for i in indices]

        # Handle negative index
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Item index {index} out of range")

        return self._read_item(index)

    def __iter__(self) -> Iterator[ConversationItem]:
        """Iterate over all items."""
        for i in range(len(self)):
            yield self._read_item(i)

    def _read_item(self, index: int) -> ConversationItem:
        """Read a single item from the Conversation at the given index."""
        # Get item header
        header = _c.items_get_item_header(self._lib, self._conversation_ptr, index)
        if header is None:
            return UnknownItem(
                id=0,
                status=ItemStatus.FAILED,
                created_at_ms=0,
                raw_type="error",
                payload=f"Failed to read item at index {index}",
            )

        item_type = ItemType(header["item_type"])
        item_id = header["id"]
        status = ItemStatus(header["status"])
        created_at_ms = header["created_at_ms"]

        match item_type:
            case ItemType.MESSAGE:
                return self._read_message_item(index, item_id, status, created_at_ms)
            case ItemType.FUNCTION_CALL:
                return self._read_function_call_item(index, item_id, status, created_at_ms)
            case ItemType.FUNCTION_CALL_OUTPUT:
                return self._read_function_call_output_item(index, item_id, status, created_at_ms)
            case ItemType.REASONING:
                return self._read_reasoning_item(index, item_id, status, created_at_ms)
            case ItemType.ITEM_REFERENCE:
                return self._read_item_reference_item(index, item_id, status, created_at_ms)
            case _:
                return UnknownItem(
                    id=item_id,
                    status=status,
                    created_at_ms=created_at_ms,
                    raw_type=str(header["item_type"]),
                    payload="",
                )

    def _read_message_item(
        self, index: int, item_id: int, status: ItemStatus, created_at_ms: int
    ) -> MessageItem:
        """Read a message item variant."""
        msg_data = _c.items_get_message(self._lib, self._conversation_ptr, index)
        if msg_data is None:
            return MessageItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                role=MessageRole.UNKNOWN,
                content=(),
            )

        role = MessageRole(msg_data["role"])
        raw_role = msg_data["raw_role"]

        # Read content parts
        content_parts: list[ContentPart] = []
        for part_idx in range(msg_data["content_count"]):
            part_dict = _c.items_get_message_content(
                self._lib, self._conversation_ptr, index, part_idx
            )
            if part_dict is not None:
                content_parts.append(_read_content_part_dict(part_dict))

        # Fetch generation params (only for assistant messages typically)
        generation = _c.items_get_generation_json(self._lib, self._conversation_ptr, index)

        return MessageItem(
            id=item_id,
            status=status,
            created_at_ms=created_at_ms,
            role=role,
            content=tuple(content_parts),
            raw_role=raw_role,
            generation=generation,
        )

    def _read_function_call_item(
        self, index: int, item_id: int, status: ItemStatus, created_at_ms: int
    ) -> FunctionCallItem:
        """Read a function call item variant."""
        fc_data = _c.items_get_function_call(self._lib, self._conversation_ptr, index)
        if fc_data is None:
            return FunctionCallItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                name="",
                call_id="",
                arguments="",
            )

        return FunctionCallItem(
            id=item_id,
            status=status,
            created_at_ms=created_at_ms,
            name=fc_data["name"],
            call_id=fc_data["call_id"],
            arguments=fc_data["arguments"],
        )

    def _read_function_call_output_item(
        self, index: int, item_id: int, status: ItemStatus, created_at_ms: int
    ) -> FunctionCallOutputItem:
        """Read a function call output item variant."""
        fco_data = _c.items_get_function_call_output(self._lib, self._conversation_ptr, index)
        if fco_data is None:
            return FunctionCallOutputItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                call_id="",
            )

        call_id = fco_data["call_id"]

        # Check if text output or parts output
        if fco_data["is_text_output"]:
            return FunctionCallOutputItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                call_id=call_id,
                output_text=fco_data["output_text"],
            )
        else:
            # Read output parts
            output_parts: list[ContentPart] = []
            for part_idx in range(fco_data["output_parts_count"]):
                part_dict = _c.items_get_fco_part(
                    self._lib, self._conversation_ptr, index, part_idx
                )
                if part_dict is not None:
                    output_parts.append(_read_content_part_dict(part_dict))
            return FunctionCallOutputItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                call_id=call_id,
                output_parts=tuple(output_parts) if output_parts else None,
            )

    def _read_reasoning_item(
        self, index: int, item_id: int, status: ItemStatus, created_at_ms: int
    ) -> ReasoningItem:
        """Read a reasoning item variant."""
        reasoning_data = _c.items_get_reasoning(self._lib, self._conversation_ptr, index)
        if reasoning_data is None:
            return ReasoningItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                content=(),
                summary=(),
            )

        # Read content parts
        content_parts: list[ContentPart] = []
        for part_idx in range(reasoning_data["content_count"]):
            part_dict = _c.items_get_reasoning_content(
                self._lib, self._conversation_ptr, index, part_idx
            )
            if part_dict is not None:
                content_parts.append(_read_content_part_dict(part_dict))

        # Read summary parts
        summary_parts: list[ContentPart] = []
        for part_idx in range(reasoning_data["summary_count"]):
            part_dict = _c.items_get_reasoning_summary(
                self._lib, self._conversation_ptr, index, part_idx
            )
            if part_dict is not None:
                summary_parts.append(_read_content_part_dict(part_dict))

        return ReasoningItem(
            id=item_id,
            status=status,
            created_at_ms=created_at_ms,
            content=tuple(content_parts),
            summary=tuple(summary_parts),
            encrypted_content=reasoning_data["encrypted_content"],
        )

    def _read_item_reference_item(
        self, index: int, item_id: int, status: ItemStatus, created_at_ms: int
    ) -> ItemReferenceItem:
        """Read an item reference item variant."""
        ref_data = _c.items_get_item_reference(self._lib, self._conversation_ptr, index)
        if ref_data is None:
            return ItemReferenceItem(
                id=item_id,
                status=status,
                created_at_ms=created_at_ms,
                ref_id="",
            )

        return ItemReferenceItem(
            id=item_id,
            status=status,
            created_at_ms=created_at_ms,
            ref_id=ref_data["ref_id"],
        )

    @property
    def last(self) -> ConversationItem | None:
        """Get the last item, or None if empty."""
        if len(self) == 0:
            return None
        return self[-1]

    @property
    def first(self) -> ConversationItem | None:
        """Get the first item, or None if empty."""
        if len(self) == 0:
            return None
        return self[0]

    @property
    def system(self) -> str | None:
        """Get the system message content, or None if no system message."""
        for item in self:
            if isinstance(item, MessageItem) and item.role == MessageRole.SYSTEM:
                return item.text
        return None

    def filter_by_type(self, item_type: type[ConversationItem]) -> list[ConversationItem]:
        """Get all items of a specific type.

        Args:
            item_type: The ConversationItem subclass to filter by.
        """
        return [item for item in self if isinstance(item, item_type)]

    def filter_by_role(self, role: MessageRole) -> list[MessageItem]:
        """Get all message items with a specific role.

        Args:
            role: The message role to filter by.
        """
        return [item for item in self if isinstance(item, MessageItem) and item.role == role]

"""
FFI bindings for chat module.

Provides chat handle creation/lifecycle FFI, storage callback and record
building functions that manage C struct lifecycle, conversation items FFI
functions (items_get_count, items_get_item_header, items_get_message, etc.)
that read C structs and convert to Python dicts, and message insertion
functions for the Responses API.
"""

import ctypes
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._bindings import check, get_lib
from .._logging import scoped_logger

# C structs from auto-generated bindings
from .._native import (
    CContentPart,
    CSessionRecord,
    CStorageEvent,
    CStorageRecord,
)
from ..exceptions import StateError

logger = scoped_logger("chat._bindings")

# Get the library handle (signatures are set up by _native.py at import time)
_lib = get_lib()


def get_chat_lib():
    """Get the chat library with all signatures configured.

    Note: Signatures are automatically set up by _native.py (auto-generated
    from Zig C API) when the library is first loaded via get_lib().
    """
    return _lib


# =============================================================================
# Storage Callback System (merged from _storage_bindings.py)
# =============================================================================

if TYPE_CHECKING:
    from talu.types import ItemRecord, SessionRecord, StorageEvent


def c_storage_record_to_item_record(record: CStorageRecord) -> "ItemRecord":
    """
    Convert C struct to Python ItemRecord dict.

    Parses the content_json to extract the variant data and combines
    with the indexed header fields.

    Parameters
    ----------
    record : CStorageRecord
        C struct from Zig callback.

    Returns
    -------
    ItemRecord
        Dict ready for storage.on_event()
    """
    # Parse the content JSON
    content_json = record.content_json
    if content_json:
        try:
            content = json.loads(content_json.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            content = {}
    else:
        content = {}

    # Map item_type to string
    item_type_map = {
        0: "message",
        1: "function_call",
        2: "function_call_output",
        3: "reasoning",
        4: "item_reference",
        255: "unknown",
    }
    item_type = item_type_map.get(record.item_type, "unknown")

    # Build variant from content JSON
    # The content JSON is the full Open Responses object
    variant: dict[str, Any] = {}

    if item_type == "message":
        variant = {
            "role": content.get("role", "unknown"),
            "status": content.get("status", "completed"),
            "content": content.get("content", []),
        }
    elif item_type == "function_call":
        variant = {
            "call_id": content.get("call_id", ""),
            "name": content.get("name", ""),
            "arguments": content.get("arguments", ""),
            "status": content.get("status", "completed"),
        }
    elif item_type == "function_call_output":
        variant = {
            "call_id": content.get("call_id", ""),
            "output": content.get("output", []),
            "status": content.get("status", "completed"),
        }
    elif item_type == "reasoning":
        variant = {
            "content": content.get("content", []),
            "summary": content.get("summary", []),
            "status": "completed",
        }
        if "encrypted_content" in content:
            variant["encrypted_content"] = content["encrypted_content"]
    elif item_type == "item_reference":
        variant = {
            "id": content.get("id", ""),
            "status": "completed",
        }
    else:
        variant = content

    # Build the ItemRecord
    status_map = {
        0: "in_progress",
        1: "waiting",
        2: "completed",
        3: "incomplete",
        4: "failed",
    }
    result: dict[str, Any] = {
        "item_id": record.item_id,
        "created_at_ms": record.created_at_ms,
        "status": status_map.get(record.status, "unknown"),
        "hidden": bool(record.hidden),
        "pinned": bool(record.pinned),
        "json_valid": bool(record.json_valid),
        "schema_valid": bool(record.schema_valid),
        "repaired": bool(record.repaired),
        "parent_item_id": record.parent_item_id if record.has_parent else None,
        "origin_session_id": (
            record.origin_session_id.decode("utf-8")
            if record.has_origin and record.origin_session_id
            else None
        ),
        "origin_item_id": record.origin_item_id if record.has_origin else None,
        "finish_reason": record.finish_reason.decode("utf-8") if record.finish_reason else None,
        "prefill_ns": record.prefill_ns,
        "generation_ns": record.generation_ns,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "ttl_ts": record.ttl_ts,
        "item_type": item_type,
        "variant": variant,
    }

    # Add metadata if present
    metadata_json = record.metadata_json
    if metadata_json:
        try:
            result["metadata"] = json.loads(metadata_json.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    return result  # type: ignore[return-value]


def c_session_record_to_session_record(record: CSessionRecord) -> "SessionRecord":
    """
    Convert C struct to Python SessionRecord dict.

    Parameters
    ----------
    record : CSessionRecord
        C struct from Zig callback.

    Returns
    -------
    SessionRecord
        Dict ready for storage.on_event()
    """
    result: SessionRecord = {  # type: ignore[assignment]
        "created_at_ms": record.created_at_ms,
        "updated_at_ms": record.updated_at_ms,
        "ttl_ts": record.ttl_ts,
    }

    # Decode optional string fields
    if record.session_id:
        result["session_id"] = record.session_id.decode("utf-8")

    if record.model:
        result["model"] = record.model.decode("utf-8")

    if record.title:
        result["title"] = record.title.decode("utf-8")

    if record.system_prompt:
        result["system_prompt"] = record.system_prompt.decode("utf-8")

    if record.config_json:
        try:
            result["config"] = json.loads(record.config_json.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if record.marker:
        result["marker"] = record.marker.decode("utf-8")

    if record.parent_session_id:
        result["parent_session_id"] = record.parent_session_id.decode("utf-8")

    if record.group_id:
        result["group_id"] = record.group_id.decode("utf-8")

    result["head_item_id"] = int(record.head_item_id)

    if record.metadata_json:
        try:
            result["metadata"] = json.loads(record.metadata_json.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if record.search_snippet:
        result["search_snippet"] = record.search_snippet.decode("utf-8")

    if record.source_doc_id:
        result["source_doc_id"] = record.source_doc_id.decode("utf-8")

    return result


def c_storage_event_to_storage_event(event: CStorageEvent) -> "StorageEvent":
    """
    Convert C struct to Python StorageEvent dict.

    Parameters
    ----------
    event : CStorageEvent
        C struct from Zig callback.

    Returns
    -------
    StorageEvent
        Dict with exactly one key (PutItems/DeleteItem/ClearItems/PutSession).

    Note: PutItems is always a list, even for single items. This enables
    efficient transaction batching when handling parallel tool calls.
    """
    event_type = event.event_type

    if event_type == 0:  # PUT_ITEMS (batched)
        # Convert C array to list of ItemRecords
        records = []
        for i in range(event.items_count):
            c_record = event.items[i]
            records.append(c_storage_record_to_item_record(c_record))
        return {"PutItems": records}
    elif event_type == 1:  # DELETE_ITEM
        return {
            "DeleteItem": {
                "item_id": event.deleted_item_id,
                "deleted_at_ms": event.deleted_at_ms,
            }
        }
    elif event_type == 2:  # CLEAR_ITEMS
        return {
            "ClearItems": {
                "cleared_at_ms": event.cleared_at_ms,
                "keep_context": event.keep_context,
            }
        }
    elif event_type == 3:  # PUT_SESSION
        return {"PutSession": c_session_record_to_session_record(event.session)}
    elif event_type == 4:  # BEGIN_FORK
        return {
            "BeginFork": {
                "fork_id": event.fork_id,
                "session_id": event.fork_session_id.decode("utf-8")
                if event.fork_session_id
                else "",
            }
        }
    elif event_type == 5:  # END_FORK
        return {
            "EndFork": {
                "fork_id": event.fork_id,
                "session_id": event.fork_session_id.decode("utf-8")
                if event.fork_session_id
                else "",
            }
        }
    else:
        # Unknown event type - return empty event
        return {}


# =============================================================================
# Chat Base FFI Functions
# =============================================================================


def chat_create(
    lib: Any,
    system: str | None,
    session_id: str | None,
    options: Any,
) -> Any:
    """Call talu_chat_create* C APIs.

    Args:
        lib: The loaded talu shared library.
        system: Optional system prompt.
        session_id: Optional session ID.
        options: ChatCreateOptions struct.

    Returns
    -------
        Chat pointer.
    """
    options_ptr = ctypes.byref(options)
    session_id_bytes = session_id.encode("utf-8") if session_id else None

    if system is not None:
        return lib.talu_chat_create_with_system_and_session(
            system.encode("utf-8"), session_id_bytes, options_ptr
        )
    elif session_id is not None:
        return lib.talu_chat_create_with_session(session_id_bytes, options_ptr)
    else:
        return lib.talu_chat_create(options_ptr)


def policy_create(lib: Any, json_bytes: bytes) -> ctypes.c_void_p:
    """Create a policy from JSON bytes."""
    handle = ctypes.c_void_p()
    rc = lib.talu_policy_create(json_bytes, len(json_bytes), ctypes.byref(handle))
    check(rc, {"operation": "policy_create"})
    if not handle:
        raise StateError(
            "Policy handle is null after creation.",
            code="STATE_INVALID_POLICY",
        )
    return handle


def chat_set_policy(lib: Any, chat_ptr: Any, policy_handle: Any) -> None:
    """Attach a policy handle to a chat."""
    rc = lib.talu_chat_set_policy(chat_ptr, policy_handle)
    check(rc, {"operation": "chat_set_policy"})


def policy_free(lib: Any, policy_handle: Any) -> None:
    """Free a policy handle."""
    if policy_handle:
        lib.talu_policy_free(policy_handle)


def read_c_text_result(lib: Any, result_ptr: Any) -> str | None:
    """Read a C text result and free it.

    Args:
        lib: The loaded talu shared library.
        result_ptr: Pointer from C API (to be freed).

    Returns
    -------
        Decoded string, or None if pointer is null or C string is empty.
    """
    if not result_ptr:
        return None
    try:
        value = ctypes.cast(result_ptr, ctypes.c_char_p).value
        return value.decode("utf-8") if value else None
    finally:
        lib.talu_text_free(result_ptr)


def read_c_json_result(lib: Any, result_ptr: Any, default: str = "[]") -> str:
    """Read a C JSON result and free it.

    Args:
        lib: The loaded talu shared library.
        result_ptr: Pointer from C API (to be freed).
        default: Default value if pointer is null.

    Returns
    -------
        Decoded JSON string or default.
    """
    if not result_ptr:
        return default
    try:
        json_str = ctypes.cast(result_ptr, ctypes.c_char_p).value
        return json_str.decode("utf-8") if json_str else default
    finally:
        lib.talu_text_free(result_ptr)


def responses_insert_message(
    lib: Any,
    conversation_ptr: Any,
    index: int,
    role_int: int,
    content: str,
    hidden: bool,
) -> int:
    """Call talu_responses_insert_message* C APIs.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the conversation handle.
        index: Position to insert at.
        role_int: Role enum value.
        content: Message content.
        hidden: Whether message should be hidden.

    Returns
    -------
        Result code from C API.
    """
    content_bytes = content.encode("utf-8")
    content_ptr = ctypes.c_char_p(content_bytes)

    if hidden:
        return lib.talu_responses_insert_message_hidden(
            conversation_ptr,
            index,
            role_int,
            content_ptr,
            len(content_bytes),
            True,
        )
    else:
        return lib.talu_responses_insert_message(
            conversation_ptr,
            index,
            role_int,
            content_ptr,
            len(content_bytes),
        )


def responses_append_message(
    lib: Any,
    conversation_ptr: Any,
    role_int: int,
    content: str,
    hidden: bool,
) -> int:
    """Call talu_responses_append_message* C APIs.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the conversation handle.
        role_int: Role enum value.
        content: Message content.
        hidden: Whether message should be hidden.

    Returns
    -------
        Result code from C API.
    """
    content_bytes = content.encode("utf-8")
    content_ptr = ctypes.c_char_p(content_bytes)

    if hidden:
        return lib.talu_responses_append_message_hidden(
            conversation_ptr,
            role_int,
            content_ptr,
            len(content_bytes),
            True,
        )
    else:
        return lib.talu_responses_append_message(
            conversation_ptr,
            role_int,
            content_ptr,
            len(content_bytes),
        )


def responses_append_function_call_output(
    lib: Any,
    conversation_ptr: Any,
    call_id: str,
    content: str,
) -> int:
    """Call talu_responses_append_function_call_output C API."""
    call_id_bytes = call_id.encode("utf-8")
    call_id_ptr = ctypes.c_char_p(call_id_bytes)
    content_bytes = content.encode("utf-8")
    content_ptr = ctypes.c_char_p(content_bytes)

    return lib.talu_responses_append_function_call_output(
        conversation_ptr,
        call_id_ptr,
        content_ptr,
        len(content_bytes),
    )


# =============================================================================
# Conversation Items FFI Functions
# =============================================================================


def read_c_string(ptr: Any, length: int) -> str:
    """Read a C string from pointer and length, returning Python str.

    Args:
        ptr: Pointer to C string data.
        length: Number of bytes to read.

    Returns
    -------
        Decoded UTF-8 string (with replacement for invalid bytes).
    """
    if not ptr or length <= 0:
        return ""
    data_bytes = ctypes.string_at(ptr, length)
    return data_bytes.decode("utf-8", errors="replace")


def read_c_string_ptr(ptr: Any) -> str | None:
    """Read a null-terminated C string from pointer.

    Args:
        ptr: Pointer to null-terminated C string.

    Returns
    -------
        Decoded UTF-8 string (with replacement for invalid bytes), or None if ptr is null.
    """
    if not ptr:
        return None
    return ptr.decode("utf-8", errors="replace")


def items_get_count(lib: Any, conversation_ptr: int) -> int:
    """Get number of items in conversation.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.

    Returns
    -------
        Number of items.
    """
    return lib.talu_responses_item_count(conversation_ptr)


def items_get_item_header(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get item header (type, id, status, created_at) at index.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Dict with item_type, id, status, created_at_ms or None on error.
    """
    from .._native import CItem

    c_item = CItem()
    result = lib.talu_responses_get_item(conversation_ptr, index, ctypes.byref(c_item))
    if result != 0:
        return None
    return {
        "item_type": c_item.item_type,
        "id": c_item.id,
        "status": c_item.status,
        "created_at_ms": c_item.created_at_ms,
    }


def items_get_message(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get message item data at index.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Dict with role, raw_role, content_count or None on error.
    """
    from .._native import CMessageItem

    c_msg = CMessageItem()
    result = lib.talu_responses_item_as_message(conversation_ptr, index, ctypes.byref(c_msg))
    if result != 0:
        return None
    return {
        "role": c_msg.role,
        "raw_role": read_c_string_ptr(c_msg.raw_role_ptr),
        "content_count": c_msg.content_count,
    }


def items_get_message_content(
    lib: Any,
    conversation_ptr: int,
    item_index: int,
    part_index: int,
) -> dict[str, Any] | None:
    """Get content part from message item.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        item_index: Item index.
        part_index: Content part index.

    Returns
    -------
        Dict with content_type, data, secondary, tertiary, image_detail or None on error.
    """
    c_part = CContentPart()
    result = lib.talu_responses_item_message_get_content(
        conversation_ptr, item_index, part_index, ctypes.byref(c_part)
    )
    if result != 0:
        return None
    return _content_part_to_dict(c_part)


def _content_part_to_dict(c_part: Any) -> dict[str, Any]:
    """Convert CContentPart to dict with all fields.

    Args:
        c_part: CContentPart struct.

    Returns
    -------
        Dict with content_type, data, secondary (annotations_json),
        tertiary (logprobs_json), quaternary (code_blocks_json), image_detail.
    """
    return {
        "content_type": c_part.content_type,
        "data": read_c_string(c_part.data_ptr, c_part.data_len),
        "secondary": read_c_string(c_part.secondary_ptr, c_part.secondary_len),
        "tertiary": read_c_string(c_part.tertiary_ptr, c_part.tertiary_len),
        "quaternary": read_c_string(c_part.quaternary_ptr, c_part.quaternary_len),
        "image_detail": c_part.image_detail,
    }


def items_get_function_call(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get function call item data at index.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Dict with name, call_id, arguments or None on error.
    """
    from .._native import CFunctionCallItem

    c_fc = CFunctionCallItem()
    result = lib.talu_responses_item_as_function_call(conversation_ptr, index, ctypes.byref(c_fc))
    if result != 0:
        return None
    return {
        "name": read_c_string_ptr(c_fc.name_ptr) or "",
        "call_id": read_c_string_ptr(c_fc.call_id_ptr) or "",
        "arguments": read_c_string(c_fc.arguments_ptr, c_fc.arguments_len),
    }


def items_get_function_call_output(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get function call output item data at index.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Dict with call_id, is_text_output, output_text, output_parts_count or None on error.
    """
    from .._native import CFunctionCallOutputItem

    c_fco = CFunctionCallOutputItem()
    result = lib.talu_responses_item_as_function_call_output(
        conversation_ptr, index, ctypes.byref(c_fco)
    )
    if result != 0:
        return None
    return {
        "call_id": read_c_string_ptr(c_fco.call_id_ptr) or "",
        "is_text_output": c_fco.is_text_output,
        "output_text": read_c_string(c_fco.output_text_ptr, c_fco.output_text_len),
        "output_parts_count": c_fco.output_parts_count,
    }


def items_get_fco_part(
    lib: Any,
    conversation_ptr: int,
    item_index: int,
    part_index: int,
) -> dict[str, Any] | None:
    """Get output part from function call output item.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        item_index: Item index.
        part_index: Output part index.

    Returns
    -------
        Dict with content_type, data, secondary, tertiary, image_detail or None on error.
    """
    c_part = CContentPart()
    result = lib.talu_responses_item_fco_get_part(
        conversation_ptr, item_index, part_index, ctypes.byref(c_part)
    )
    if result != 0:
        return None
    return _content_part_to_dict(c_part)


def items_get_reasoning(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get reasoning item data at index.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Dict with content_count, summary_count, encrypted_content or None on error.
    """
    from .._native import CReasoningItem

    c_reasoning = CReasoningItem()
    result = lib.talu_responses_item_as_reasoning(
        conversation_ptr, index, ctypes.byref(c_reasoning)
    )
    if result != 0:
        return None
    return {
        "content_count": c_reasoning.content_count,
        "summary_count": c_reasoning.summary_count,
        "encrypted_content": read_c_string(
            c_reasoning.encrypted_content_ptr, c_reasoning.encrypted_content_len
        )
        or None,
    }


def items_get_reasoning_content(
    lib: Any,
    conversation_ptr: int,
    item_index: int,
    part_index: int,
) -> dict[str, Any] | None:
    """Get content part from reasoning item.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        item_index: Item index.
        part_index: Content part index.

    Returns
    -------
        Dict with content_type, data, secondary, tertiary, image_detail or None on error.
    """
    c_part = CContentPart()
    result = lib.talu_responses_item_reasoning_get_content(
        conversation_ptr, item_index, part_index, ctypes.byref(c_part)
    )
    if result != 0:
        return None
    return _content_part_to_dict(c_part)


def items_get_reasoning_summary(
    lib: Any,
    conversation_ptr: int,
    item_index: int,
    part_index: int,
) -> dict[str, Any] | None:
    """Get summary part from reasoning item.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        item_index: Item index.
        part_index: Summary part index.

    Returns
    -------
        Dict with content_type, data, secondary, tertiary, image_detail or None on error.
    """
    c_part = CContentPart()
    result = lib.talu_responses_item_reasoning_get_summary(
        conversation_ptr, item_index, part_index, ctypes.byref(c_part)
    )
    if result != 0:
        return None
    return _content_part_to_dict(c_part)


def items_get_item_reference(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get item reference data at index.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Dict with ref_id or None on error.
    """
    from .._native import CItemReferenceItem

    c_ref = CItemReferenceItem()
    result = lib.talu_responses_item_as_item_reference(conversation_ptr, index, ctypes.byref(c_ref))
    if result != 0:
        return None
    return {
        "ref_id": read_c_string_ptr(c_ref.id_ptr) or "",
    }


def items_get_generation_json(
    lib: Any,
    conversation_ptr: int,
    index: int,
) -> dict[str, Any] | None:
    """Get generation parameters JSON for an item.

    Args:
        lib: The loaded talu shared library.
        conversation_ptr: Pointer to the Conversation handle.
        index: Item index.

    Returns
    -------
        Parsed generation dict or None if not set.
    """
    import json

    out_ptr = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    result = lib.talu_responses_item_get_generation_json(
        conversation_ptr,
        index,
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
    )
    if result != 0 or not out_ptr.value or out_len.value == 0:
        return None

    # Read the bytes from the pointer
    buf = ctypes.cast(out_ptr.value, ctypes.POINTER(ctypes.c_char * out_len.value)).contents
    json_str = bytes(buf).decode("utf-8")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# =============================================================================
# Storage Record FFI Functions
# =============================================================================


def build_c_storage_records(
    items: "Sequence[Mapping[str, Any]]",
    type_map: dict[str, int],
    role_map: dict[str, int],
    status_map: dict[str, int],
) -> tuple[ctypes.Array, list[bytes]]:
    """Build a ctypes array of CStorageRecord from Python dicts.

    Args:
        items: List of ItemRecord dicts.
        type_map: Mapping from type string to int enum value.
        role_map: Mapping from role string to int enum value.
        status_map: Mapping from status string to int enum value.

    Returns
    -------
        Tuple of (CStorageRecord array, keepalive list of bytes).
    """
    keepalive: list[bytes] = []
    array_type = CStorageRecord * len(items)
    records = array_type()

    for idx, item in enumerate(items):
        item_type_str = item.get("item_type", "message")
        item_type = type_map.get(item_type_str, 255)  # 255 = UNKNOWN
        variant = item.get("variant", {})
        role_str = variant.get("role", "unknown")
        role = role_map.get(role_str, 255)  # 255 = UNKNOWN
        status_str = item.get("status", "completed")
        status = status_map.get(status_str, 2)  # 2 = COMPLETED

        content_obj = dict(variant)
        content_obj["type"] = item_type_str
        content_json = json.dumps(content_obj).encode("utf-8")
        keepalive.append(content_json)
        metadata_json = None
        if item.get("metadata") is not None:
            metadata_json = json.dumps(item.get("metadata")).encode("utf-8")
            keepalive.append(metadata_json)

        finish_reason = item.get("finish_reason")
        finish_bytes = finish_reason.encode("utf-8") if finish_reason else None
        if finish_bytes is not None:
            keepalive.append(finish_bytes)

        origin_session = item.get("origin_session_id")
        origin_bytes = origin_session.encode("utf-8") if origin_session else None
        if origin_bytes is not None:
            keepalive.append(origin_bytes)

        record = CStorageRecord()
        record.item_id = int(item.get("item_id", 0))
        record.session_id = None
        record.item_type = int(item_type)
        record.role = int(role)
        record.status = int(status)
        record.hidden = bool(item.get("hidden", False))
        record.pinned = bool(item.get("pinned", False))
        record.json_valid = bool(item.get("json_valid", False))
        record.schema_valid = bool(item.get("schema_valid", False))
        record.repaired = bool(item.get("repaired", False))
        record.parent_item_id = int(item.get("parent_item_id", 0) or 0)
        record.has_parent = bool(item.get("parent_item_id") is not None)
        record.origin_item_id = int(item.get("origin_item_id", 0) or 0)
        record.has_origin = bool(item.get("origin_item_id") is not None)
        record.origin_session_id = origin_bytes
        record.finish_reason = finish_bytes
        record.prefill_ns = int(item.get("prefill_ns", 0))
        record.generation_ns = int(item.get("generation_ns", 0))
        record.input_tokens = int(item.get("input_tokens", 0))
        record.output_tokens = int(item.get("output_tokens", 0))
        record.ttl_ts = int(item.get("ttl_ts", 0))
        record.created_at_ms = int(item.get("created_at_ms", 0))
        record.content_json = content_json
        record.metadata_json = metadata_json
        records[idx] = record

    return records, keepalive

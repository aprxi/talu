"""
FFI bindings for chat module.

Provides chat handle creation/lifecycle FFI, conversation items FFI
functions (items_get_count, items_get_item_header, items_get_message, etc.)
that read C structs and convert to Python dicts, and message insertion
functions for the Responses API.
"""

import ctypes
import json
from typing import Any

from .._bindings import check, get_lib
from .._logging import scoped_logger

# C structs from auto-generated bindings
from .._native import (
    CContentPart,
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


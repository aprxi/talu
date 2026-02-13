"""
FFI bindings for tokenizer.

Justification: Provides C API call wrappers that handle ctypes memory management
(c_void_p output params, byref, cast, string_at, array creation). Also provides
PyCapsule creation for tokenizer handle export via ctypes.pythonapi.
"""

import ctypes
from typing import Any

from .._bindings import get_lib

# C structs from auto-generated bindings (used in function signatures below)
from .._native import EncodeOptions, PaddedTensorOptions, PaddedTensorResult


def create_dlpack_capsule(dlpack_ptr: Any) -> Any:
    """
    Create a PyCapsule wrapping a DLManagedTensor pointer.

    Args:
        dlpack_ptr: Pointer to DLManagedTensor from Zig.

    Returns
    -------
        PyCapsule with name "dltensor" for DLPack protocol.
    """
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    PyCapsule_New.restype = ctypes.py_object
    return PyCapsule_New(dlpack_ptr, b"dltensor", None)


# Get the library handle (signatures are set up by _native.py at import time)
_lib = get_lib()


def call_apply_chat_template(
    model_path: bytes, messages_json: bytes, add_generation_prompt: bool
) -> tuple[int, str | None]:
    """Call talu_apply_chat_template and return (error_code, result_string).

    Returns
    -------
        Tuple of (error_code, result_string). If error_code != 0, result_string is None.
    """
    out_prompt = ctypes.c_void_p()
    code = _lib.talu_apply_chat_template(
        model_path,
        messages_json,
        1 if add_generation_prompt else 0,
        ctypes.byref(out_prompt),
    )

    if code != 0:
        return (code, None)

    if not out_prompt.value:
        return (0, None)  # Success but null result

    text_bytes = ctypes.cast(out_prompt.value, ctypes.c_char_p).value
    result = text_bytes.decode("utf-8") if text_bytes else None
    _lib.talu_text_free(out_prompt)
    return (0, result)


def call_apply_chat_template_string(
    template: bytes,
    messages_json: bytes,
    add_generation_prompt: bool,
    bos_token: bytes,
    eos_token: bytes,
) -> tuple[int, str | None]:
    """Call talu_apply_chat_template_string with a template string directly.

    This renders a chat template without needing a model directory.

    Returns
    -------
        Tuple of (error_code, result_string). If error_code != 0, result_string is None.
    """
    out_prompt = ctypes.c_void_p()
    template_array = (ctypes.c_char * len(template)).from_buffer_copy(template)
    code = _lib.talu_apply_chat_template_string(
        template_array,
        len(template),
        messages_json,
        1 if add_generation_prompt else 0,
        bos_token,
        eos_token,
        ctypes.byref(out_prompt),
    )

    if code != 0:
        return (code, None)

    if not out_prompt.value:
        return (0, None)  # Success but null result

    text_bytes = ctypes.cast(out_prompt.value, ctypes.c_char_p).value
    # text_bytes can be b'' (empty string) which is valid, only None means error
    result = text_bytes.decode("utf-8") if text_bytes is not None else ""
    _lib.talu_text_free(out_prompt)
    return (0, result)


def call_resolve_model_path(model: bytes) -> tuple[int, str | None]:
    """Call talu_resolve_model_path and return (error_code, path_string)."""
    out_path = ctypes.c_void_p()
    code = _lib.talu_resolve_model_path(model, ctypes.byref(out_path))
    if code != 0 or not out_path.value:
        return (code, None)
    path_bytes = ctypes.cast(out_path.value, ctypes.c_char_p).value
    if path_bytes is None:
        return (0, None)
    result = path_bytes.decode("utf-8")
    _lib.talu_text_free(out_path)
    return (0, result)


def call_tokenizer_create(model_dir: bytes) -> tuple[int, int]:
    """Call talu_tokenizer_create and return (error_code, handle_ptr).

    Returns handle as int (pointer value) for storage in Python class.
    """
    out_ptr = ctypes.c_void_p()
    code = _lib.talu_tokenizer_create(model_dir, ctypes.byref(out_ptr))
    if code != 0:
        return (code, 0)
    return (0, out_ptr.value or 0)


def call_tokenizer_create_from_json(json_bytes: bytes) -> tuple[int, int]:
    """Call talu_tokenizer_create_from_json and return (error_code, handle_ptr).

    Creates a minimal tokenizer from JSON content without a model directory.
    Returns handle as int (pointer value) for storage in Python class.
    """
    out_ptr = ctypes.c_void_p()
    json_array = (ctypes.c_char * len(json_bytes)).from_buffer_copy(json_bytes)
    code = _lib.talu_tokenizer_create_from_json(json_array, len(json_bytes), ctypes.byref(out_ptr))
    if code != 0:
        return (code, 0)
    return (0, out_ptr.value or 0)


def call_tokenizer_encode(ptr: int, text_bytes: bytes, options: "EncodeOptions") -> "Any":
    """Call talu_tokenizer_encode and return the result struct.

    The caller must handle the result struct fields.
    """
    text_array = (ctypes.c_char * len(text_bytes)).from_buffer_copy(text_bytes)
    return _lib.talu_tokenizer_encode(ptr, text_array, len(text_bytes), ctypes.byref(options))


def call_tokenizer_encode_batch(ptr: int, texts: list[bytes], options: "EncodeOptions") -> "Any":
    """Call talu_tokenizer_encode_batch and return the result struct."""
    num_texts = len(texts)
    c_char_p_array = (ctypes.POINTER(ctypes.c_char) * num_texts)()
    lengths_array = (ctypes.c_size_t * num_texts)()

    # Keep references to prevent GC
    char_arrays = []
    for i, text_bytes in enumerate(texts):
        char_array = (ctypes.c_char * len(text_bytes)).from_buffer_copy(text_bytes)
        char_arrays.append(char_array)
        c_char_p_array[i] = ctypes.cast(char_array, ctypes.POINTER(ctypes.c_char))
        lengths_array[i] = len(text_bytes)

    result = _lib.talu_tokenizer_encode_batch(
        ptr,
        ctypes.cast(c_char_p_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))),
        ctypes.cast(lengths_array, ctypes.POINTER(ctypes.c_size_t)),
        num_texts,
        ctypes.byref(options),
    )
    # Keep char_arrays alive until after the call
    _ = char_arrays
    return result


def call_tokenizer_decode(
    ptr: int, tokens: list[int], options: "Any"
) -> tuple[str | None, str | None]:
    """Call talu_tokenizer_decode and return (result_text, error_msg).

    Returns (text, None) on success, (None, error_msg) on error.
    """
    num_tokens = len(tokens)
    if num_tokens > 0:
        arr = (ctypes.c_uint32 * num_tokens)(*tokens)
        tokens_ptr = ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32))
    else:
        tokens_ptr = None

    result = _lib.talu_tokenizer_decode(ptr, tokens_ptr, num_tokens, ctypes.byref(options))

    if result.error_msg:
        return (None, result.error_msg.decode("utf-8"))

    if not result.text or result.text_len == 0:
        return ("", None)

    text_bytes = ctypes.string_at(result.text, result.text_len)
    text = text_bytes.decode("utf-8")
    _lib.talu_decode_result_free(result.text, result.text_len)
    return (text, None)


def call_tokenizer_id_to_token(ptr: int, token_id: int) -> str | None:
    """Call talu_tokenizer_id_to_token and return token string or None."""
    out_token = ctypes.c_void_p()
    code = _lib.talu_tokenizer_id_to_token(ptr, token_id, ctypes.byref(out_token))
    if code != 0 or not out_token.value:
        return None
    token_bytes = ctypes.cast(out_token.value, ctypes.c_char_p).value
    if token_bytes is None:
        return None
    token = token_bytes.decode("utf-8")
    _lib.talu_text_free(out_token)
    return token


def call_tokenizer_token_to_id(ptr: int, token: bytes) -> int:
    """Call talu_tokenizer_token_to_id and return token ID (-1 if not found)."""
    token_array = (ctypes.c_char * len(token)).from_buffer_copy(token)
    return _lib.talu_tokenizer_token_to_id(ptr, token_array, len(token))


def call_tokenizer_tokenize(ptr: int, text_bytes: bytes) -> "Any":
    """Call talu_tokenizer_tokenize and return the result struct."""
    text_array = (ctypes.c_char * len(text_bytes)).from_buffer_copy(text_bytes)
    return _lib.talu_tokenizer_tokenize(ptr, text_array, len(text_bytes))


def call_tokenizer_tokenize_bytes(ptr: int, text_bytes: bytes) -> "Any":
    """Call talu_tokenizer_tokenize_bytes and return the result struct."""
    text_array = (ctypes.c_char * len(text_bytes)).from_buffer_copy(text_bytes)
    return _lib.talu_tokenizer_tokenize_bytes(ptr, text_array, len(text_bytes))


def free_tokenize_bytes_result(result: "Any") -> None:
    """Free the result from talu_tokenizer_tokenize_bytes."""
    _lib.talu_tokenize_bytes_result_free(
        result.data, result.data_len, result.offsets, result.num_tokens
    )


def free_tokenize_result(result: "Any") -> None:
    """Free the result from talu_tokenizer_tokenize."""
    _lib.talu_tokenize_result_free(result.tokens, result.num_tokens)


def call_tokenizer_free(ptr: int) -> None:
    """Free the tokenizer handle."""
    _lib.talu_tokenizer_free(ptr)


def call_tokenizer_get_vocab_size(ptr: int) -> int:
    """Get vocabulary size."""
    return _lib.talu_tokenizer_get_vocab_size(ptr)


def call_tokenizer_get_model_max_length(ptr: int) -> int:
    """Get model max length."""
    return _lib.talu_tokenizer_get_model_max_length(ptr)


def call_tokenizer_get_eos_tokens(ptr: int) -> tuple[tuple[int, ...], bool]:
    """Get EOS token IDs.

    Returns
    -------
        Tuple of (token_ids_tuple, success). token_ids_tuple may be empty.
    """
    result = _lib.talu_tokenizer_get_eos_tokens(ptr)
    if result.tokens and result.num_tokens > 0:
        # Preserve order, deduplicate, return as immutable tuple
        seen: set[int] = set()
        ordered: list[int] = []
        for i in range(result.num_tokens):
            token_id = result.tokens[i]
            if token_id not in seen:
                seen.add(token_id)
                ordered.append(token_id)
        _lib.talu_tokens_free(result.tokens, result.num_tokens)
        return (tuple(ordered), True)
    return ((), True)


def call_tokenizer_get_special_tokens(ptr: int) -> tuple[int | None, int | None, int | None]:
    """Get special token IDs.

    Returns
    -------
        Tuple of (bos_token_id, unk_token_id, pad_token_id). None if not set (<0).
    """
    result = _lib.talu_tokenizer_get_special_tokens(ptr)
    bos = result.bos_token_id if result.bos_token_id >= 0 else None
    unk = result.unk_token_id if result.unk_token_id >= 0 else None
    pad = result.pad_token_id if result.pad_token_id >= 0 else None
    return (bos, unk, pad)


def call_buffer_create_from_owned(tokens_ptr: "Any", num_tokens: int) -> int:
    """Create a SharedBuffer from owned token data.

    Returns buffer handle as int, or 0 on failure.
    """
    handle = _lib.talu_buffer_create_from_owned(tokens_ptr, num_tokens)
    return handle if handle else 0


def call_buffer_get_data_ptr(buffer_handle: int) -> "Any":
    """Get data pointer from buffer handle."""
    return _lib.talu_buffer_get_data_ptr(buffer_handle)


def call_tokens_free(tokens_ptr: "Any", num_tokens: int) -> None:
    """Free token array."""
    _lib.talu_tokens_free(tokens_ptr, num_tokens)


def call_tokenizer_get_vocab(ptr: int) -> tuple[dict[str, int], str | None]:
    """Get vocabulary as dict.

    Returns
    -------
        Tuple of (vocab_dict, error_msg). Error msg is None on success.
    """
    result = _lib.talu_tokenizer_get_vocab(ptr)

    if result.error_msg:
        return ({}, result.error_msg.decode("utf-8"))

    if result.num_entries == 0:
        return ({}, None)

    vocab = {}
    for i in range(result.num_entries):
        if result.tokens[i]:
            token_str = result.tokens[i][: result.lengths[i]].decode("utf-8")
            vocab[token_str] = result.ids[i]

    _lib.talu_vocab_result_free(result.tokens, result.lengths, result.ids, result.num_entries)
    return (vocab, None)


def call_buffer_create_from_copy(token_list: list[int]) -> tuple[int, "Any"]:
    """Create a SharedBuffer by copying token data.

    Returns
    -------
        Tuple of (buffer_handle, data_ptr). Handle is 0 on failure.
    """
    if not token_list:
        return (0, None)
    arr = (ctypes.c_uint32 * len(token_list))(*token_list)
    buffer_handle = _lib.talu_buffer_create_from_copy(
        ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32)),
        len(token_list),
    )
    if not buffer_handle:
        return (0, None)
    data_ptr = _lib.talu_buffer_get_data_ptr(buffer_handle)
    return (buffer_handle, data_ptr)


def call_buffer_release(buffer_handle: int) -> None:
    """Release a buffer handle."""
    _lib.talu_buffer_release(buffer_handle)


def call_buffer_to_dlpack(buffer_handle: int, offset: int, length: int) -> "Any":
    """Create DLPack tensor from buffer.

    Returns pointer to DLManagedTensor, or None on failure.
    """
    return _lib.talu_buffer_to_dlpack(buffer_handle, offset, length)


def call_tokens_concat(ptr1: "Any", len1: int, ptr2: "Any", len2: int) -> "Any":
    """Concatenate two token arrays.

    Returns pointer to concatenated tokens, or None on failure.
    """
    return _lib.talu_tokens_concat(ptr1, len1, ptr2, len2)


def call_tokens_concat_with_list(ptr1: "Any", len1: int, token_list: list[int]) -> "Any":
    """Concatenate token array with list.

    Returns pointer to concatenated tokens, or None on failure.
    """
    if not token_list:
        return _lib.talu_tokens_concat(ptr1, len1, None, 0)
    arr = (ctypes.c_uint32 * len(token_list))(*token_list)
    return _lib.talu_tokens_concat(
        ptr1, len1, ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32)), len(token_list)
    )


def call_list_concat_with_tokens(token_list: list[int], ptr2: "Any", len2: int) -> "Any":
    """Concatenate list with token array.

    Returns pointer to concatenated tokens, or None on failure.
    """
    if not token_list:
        return _lib.talu_tokens_concat(None, 0, ptr2, len2)
    arr = (ctypes.c_uint32 * len(token_list))(*token_list)
    return _lib.talu_tokens_concat(
        ctypes.cast(arr, ctypes.POINTER(ctypes.c_uint32)), len(token_list), ptr2, len2
    )


def call_encode_result_free(result: "Any") -> None:
    """Free an EncodeResult.

    Null out result.ids first if ownership was transferred to a SharedBuffer.
    """
    _lib.talu_encode_result_free(result)


def get_ptr_address(ptr: "Any") -> int:
    """Get address of pointer contents for array interface."""
    return ctypes.addressof(ptr.contents)


def get_ptr_address_offset(ptr: "Any", offset: int) -> int:
    """Get address of pointer contents plus byte offset for array interface."""
    return ctypes.addressof(ptr.contents) + offset


def call_batch_to_dlpack(
    ids_ptr: "Any",
    offsets_ptr: "Any",
    num_sequences: int,
    pad_id: int,
    max_length: int,
    left_pad: bool,
) -> "Any":
    """Create input_ids DLPack tensor from batch.

    Returns pointer to DLManagedTensor, or None on failure.
    """
    return _lib.talu_batch_to_dlpack(
        ids_ptr, offsets_ptr, num_sequences, pad_id, max_length, 1 if left_pad else 0
    )


def call_batch_mask_to_dlpack(
    ids_ptr: "Any", offsets_ptr: "Any", num_sequences: int, max_length: int, left_pad: bool
) -> "Any":
    """Create attention mask DLPack tensor from batch.

    Returns pointer to DLManagedTensor, or None on failure.
    """
    return _lib.talu_batch_mask_to_dlpack(
        ids_ptr, offsets_ptr, num_sequences, max_length, 1 if left_pad else 0
    )


def call_batch_to_padded_tensor(
    ids_ptr: "Any", offsets_ptr: "Any", num_sequences: int, options: "PaddedTensorOptions"
) -> "PaddedTensorResult":
    """Convert batch to padded tensor result."""
    return _lib.talu_batch_to_padded_tensor(
        ids_ptr, offsets_ptr, num_sequences, ctypes.byref(options)
    )


def call_padded_tensor_result_free(
    input_ids: "Any", attention_mask: "Any", num_sequences: int, padded_length: int
) -> None:
    """Free padded tensor result."""
    _lib.talu_padded_tensor_result_free(input_ids, attention_mask, num_sequences, padded_length)


def call_batch_encode_result_free(
    ids_ptr: "Any", offsets_ptr: "Any", total_tokens: int, num_sequences: int
) -> None:
    """Free batch encoding result."""
    _lib.talu_batch_encode_result_free(ids_ptr, offsets_ptr, total_tokens, num_sequences)

"""
FFI bindings for template module.

Justification: Provides FFI boundary functions for template rendering that handle
ctypes conversions (byref, cast) for C API calls. Defines COutputSpan structure
and CustomFilterFunc CFUNCTYPE for debug rendering and custom filter callbacks.
"""

import ctypes
from collections.abc import Callable
from typing import Any

from .._bindings import check, get_lib
from ..exceptions import TaluError

# =============================================================================
# C Struct Definitions
# =============================================================================


class COutputSpan(ctypes.Structure):
    """C-compatible output span structure for debug rendering."""

    _fields_ = [
        ("start", ctypes.c_uint32),
        ("end", ctypes.c_uint32),
        ("source_type", ctypes.c_uint8),
        ("variable_path", ctypes.c_char_p),
    ]


# Span source type constants
SPAN_STATIC = 0
SPAN_VARIABLE = 1
SPAN_EXPRESSION = 2


# Custom filter callback type for ctypes
# Signature: (value_json, args_json, user_data) -> result_json or NULL
# Returns void* because the Zig side expects a pointer that can be NULL
CustomFilterFunc = ctypes.CFUNCTYPE(
    ctypes.c_void_p,  # return: result_json pointer (or NULL on error)
    ctypes.c_char_p,  # value_json
    ctypes.c_char_p,  # args_json
    ctypes.c_void_p,  # user_data
)


def get_chat_template_source(model_path: str) -> str:
    """
    Get chat template source from model path.

    Returns the template source string.
    Raises appropriate exceptions on error via check().
    """
    lib = get_lib()
    out_source = ctypes.c_void_p()
    code = lib.talu_get_chat_template_source(
        model_path.encode("utf-8"),
        ctypes.byref(out_source),
    )
    check(code)

    # C API guarantees non-null on success after check() passes
    if not out_source.value:  # pragma: no cover - C API invariant
        raise TaluError(
            "unexpected null pointer from get_chat_template_source", code="INTERNAL_ERROR"
        )

    try:
        raw_value = ctypes.cast(out_source.value, ctypes.c_char_p).value
        if raw_value is None:  # pragma: no cover - C API invariant
            raise TaluError(
                "unexpected null string from get_chat_template_source", code="INTERNAL_ERROR"
            )
        return raw_value.decode("utf-8")
    finally:
        lib.talu_text_free(out_source)


def template_compile_check(source: str) -> int:
    """
    Check if template compiles (syntax check only).

    Returns error code (0 = success, non-zero = error).
    Does NOT raise - caller handles error code interpretation.
    """
    lib = get_lib()
    out_rendered = ctypes.c_void_p()
    code = lib.talu_template_render(
        source.encode("utf-8"),
        b"{}",  # Empty JSON object
        False,  # strict mode off for compile check
        ctypes.byref(out_rendered),
    )

    if out_rendered.value is not None:
        lib.talu_text_free(out_rendered)

    return code


def template_render(source: str, json_vars: str, strict: bool) -> str | None:
    """
    Render a template with JSON variables.

    Returns rendered string, or None if output is empty.
    Raises appropriate exceptions on error via check().
    """
    lib = get_lib()
    out_rendered = ctypes.c_void_p()
    code = lib.talu_template_render(
        source.encode("utf-8"),
        json_vars.encode("utf-8"),
        strict,
        ctypes.byref(out_rendered),
    )
    check(code)

    # C API guarantees non-null on success after check() passes
    if not out_rendered.value:  # pragma: no cover - C API invariant
        raise TaluError("unexpected null pointer from template render", code="INTERNAL_ERROR")

    try:
        raw_text = ctypes.cast(out_rendered.value, ctypes.c_char_p).value
        if raw_text is None:  # pragma: no cover - C API invariant
            raise TaluError("unexpected null string from template render", code="INTERNAL_ERROR")
        return raw_text.decode("utf-8")
    finally:
        lib.talu_text_free(out_rendered)


def template_render_with_filters(
    source: str,
    json_vars: str,
    strict: bool,
    custom_filters: dict[str, tuple[Callable, Any]],
) -> str | None:
    """
    Render a template with custom filters.

    Args:
        source: Template source string
        json_vars: JSON-encoded variables
        strict: Whether to use strict mode
        custom_filters: Dict of filter_name -> (python_func, ctypes_callback)

    Returns rendered string, or None if output is empty.
    Raises appropriate exceptions on error via check().
    """
    lib = get_lib()

    # Build arrays for C API
    num_filters = len(custom_filters)
    filter_names = (ctypes.c_char_p * num_filters)()
    filter_callbacks = (CustomFilterFunc * num_filters)()
    filter_user_data = (ctypes.c_void_p * num_filters)()

    for i, (name, (_, callback)) in enumerate(custom_filters.items()):
        filter_names[i] = name.encode("utf-8")
        filter_callbacks[i] = callback
        filter_user_data[i] = None  # We don't use user_data (closure captures state)

    out_rendered = ctypes.c_void_p()
    code = lib.talu_template_render_with_filters(
        source.encode("utf-8"),
        json_vars.encode("utf-8"),
        strict,
        filter_names,
        ctypes.cast(filter_callbacks, ctypes.c_void_p),
        filter_user_data,
        num_filters,
        ctypes.byref(out_rendered),
    )
    check(code)

    # C API guarantees non-null on success after check() passes
    if not out_rendered.value:  # pragma: no cover - C API invariant
        raise TaluError("unexpected null pointer from template render", code="INTERNAL_ERROR")

    try:
        raw_text = ctypes.cast(out_rendered.value, ctypes.c_char_p).value
        if raw_text is None:  # pragma: no cover - C API invariant
            raise TaluError("unexpected null string from template render", code="INTERNAL_ERROR")
        return raw_text.decode("utf-8")
    finally:
        lib.talu_text_free(out_rendered)


def template_validate_raw(source: str, json_vars: str) -> tuple[int, str | None]:
    """
    Validate a template with JSON variables (raw version).

    Returns (error_code, result_json). Does NOT raise.
    """
    lib = get_lib()
    out_result = ctypes.c_void_p()
    code = lib.talu_template_validate(
        source.encode("utf-8"),
        json_vars.encode("utf-8"),
        ctypes.byref(out_result),
    )

    if not out_result.value:
        return (code, None)

    raw_result = ctypes.cast(out_result.value, ctypes.c_char_p).value
    if raw_result is None:
        lib.talu_text_free(out_result)
        return (code, None)

    result_json = raw_result.decode("utf-8")
    lib.talu_text_free(out_result)
    return (code, result_json)


def template_validate(source: str, json_vars: str) -> str | None:
    """
    Validate a template with JSON variables.

    Returns result JSON string, or None if output is empty.
    Raises appropriate exceptions on error via check().
    """
    lib = get_lib()
    out_result = ctypes.c_void_p()
    code = lib.talu_template_validate(
        source.encode("utf-8"),
        json_vars.encode("utf-8"),
        ctypes.byref(out_result),
    )
    check(code)

    # C API guarantees non-null on success after check() passes
    if not out_result.value:  # pragma: no cover - C API invariant
        raise TaluError("unexpected null pointer from template validate", code="INTERNAL_ERROR")

    try:
        raw_result = ctypes.cast(out_result.value, ctypes.c_char_p).value
        if raw_result is None:  # pragma: no cover - C API invariant
            raise TaluError("unexpected null string from template validate", code="INTERNAL_ERROR")
        return raw_result.decode("utf-8")
    finally:
        lib.talu_text_free(out_result)


def template_render_debug(
    source: str, json_vars: str, strict: bool
) -> tuple[str | None, list[tuple[int, int, str, str]]]:
    """
    Render a template with debug span information.

    Returns (rendered_string, spans). Each span is (start, end, source, text).
    Raises appropriate exceptions on error via check().
    """
    lib = get_lib()
    out_rendered = ctypes.c_void_p()
    out_spans = ctypes.c_void_p()
    out_span_count = ctypes.c_uint32()

    code = lib.talu_template_render_debug(
        source.encode("utf-8"),
        json_vars.encode("utf-8"),
        strict,
        ctypes.byref(out_rendered),
        ctypes.byref(out_spans),
        ctypes.byref(out_span_count),
    )
    check(code)

    if not out_rendered.value:
        return (None, [])

    raw_output = ctypes.cast(out_rendered.value, ctypes.c_char_p).value
    if raw_output is None:
        lib.talu_text_free(out_rendered)
        return (None, [])

    output = raw_output.decode("utf-8")
    lib.talu_text_free(out_rendered)

    # Extract spans as tuples
    spans: list[tuple[int, int, str, str]] = []
    span_count = out_span_count.value

    if span_count > 0 and out_spans.value:
        c_spans = ctypes.cast(out_spans.value, ctypes.POINTER(COutputSpan * span_count)).contents

        for c_span in c_spans:
            start = c_span.start
            end = c_span.end
            text = output[start:end]

            # Determine source type
            if c_span.source_type == SPAN_STATIC:
                source_str = "static"
            elif c_span.source_type == SPAN_VARIABLE:
                source_str = (
                    c_span.variable_path.decode("utf-8") if c_span.variable_path else "variable"
                )
            else:  # SPAN_EXPRESSION
                source_str = "expression"

            spans.append((start, end, source_str, text))

        # Free the spans
        lib.talu_free_spans(out_spans, span_count)

    return (output, spans)


def alloc_string_for_callback(data: bytes) -> int:
    """
    Allocate C memory for a string to return from a callback.

    Uses talu_alloc_string which allocates via c_allocator, allowing Zig
    to free the memory correctly.

    Returns the pointer as an int.

    Raises
    ------
        MemoryError: If allocation fails.
    """
    lib = get_lib()
    length = len(data) + 1  # +1 for null terminator

    ptr = lib.talu_alloc_string(length)
    if not ptr:  # pragma: no cover - allocation failure
        raise MemoryError(f"failed to allocate {length} bytes for callback string")

    # Copy the string to C memory
    ctypes.memmove(ptr, data, len(data))
    # Add null terminator
    ctypes.memset(ptr + len(data), 0, 1)

    return ptr

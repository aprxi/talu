"""
FFI bindings for validation engine.

Justification: Provides TokenInfoCallback CFUNCTYPE for the validation engine
to query token bytes from Python. Callback types require manual CFUNCTYPE
definition that cannot be auto-generated. Also provides wrappers for C API
calls that require ctypes.byref() or array creation.
"""

import ctypes
from typing import Any

from .._bindings import get_lib
from .._native import SemanticViolationC

_lib = get_lib()

# Note: Function signatures (argtypes/restype) are set up automatically by
# _native.py (auto-generated from Zig C API) when the library is first loaded.

# Handle types for documentation
EngineHandle = ctypes.c_void_p
TokenMaskHandle = ctypes.c_void_p

# Callback type for token info (still needed for Python callback signature)
TokenInfoCallback = ctypes.CFUNCTYPE(
    ctypes.c_char_p,  # return: pointer to token bytes
    ctypes.c_uint32,  # token_id
    ctypes.POINTER(ctypes.c_size_t),  # out_len
    ctypes.c_void_p,  # ctx
)


def call_engine_create(schema_json: bytes) -> "Any":
    """Create validation engine from JSON schema."""
    return _lib.talu_validate_engine_create(schema_json)


def call_semantic_validator_create(schema_json: bytes) -> "Any":
    """Create semantic validator from JSON schema."""
    return _lib.talu_semantic_validator_create(schema_json)


def call_engine_destroy(handle: "Any") -> None:
    """Destroy validation engine."""
    _lib.talu_validate_engine_destroy(handle)


def call_semantic_validator_destroy(handle: "Any") -> None:
    """Destroy semantic validator."""
    _lib.talu_semantic_validator_destroy(handle)


def call_engine_validate(handle: "Any", data: bytes) -> int:
    """Validate complete JSON data. Returns 1 if valid."""
    return _lib.talu_validate_engine_validate(handle, data, len(data))


def call_semantic_validator_check(handle: "Any", data: bytes) -> tuple[int, dict | None]:
    """Check semantic constraints.

    Returns
    -------
        Tuple of (result_code, violation_info).
        result_code: 0=valid, 1=violation, negative=error
        violation_info: dict with violation details if result_code==1, else None
    """
    violation = SemanticViolationC()
    result = _lib.talu_semantic_validator_check(handle, data, len(data), ctypes.byref(violation))

    if result == 1:
        # Violation found - extract info
        return (
            result,
            {
                "path": violation.path.decode("utf-8") if violation.path else "",
                "message": violation.message.decode("utf-8") if violation.message else "",
                "constraint_type": violation.constraint_type,
            },
        )

    return (result, None)


def call_engine_advance(handle: "Any", chunk: bytes) -> int:
    """Advance engine with chunk. Returns number of bytes consumed."""
    return _lib.talu_validate_engine_advance(handle, chunk, len(chunk))


def call_engine_reset(handle: "Any") -> int:
    """Reset engine to initial state. Returns 0 on success."""
    return _lib.talu_validate_engine_reset(handle)


def call_engine_is_complete(handle: "Any") -> bool:
    """Check if engine is in complete state."""
    return _lib.talu_validate_engine_is_complete(handle)


def call_engine_get_position(handle: "Any") -> int:
    """Get current position in stream."""
    return _lib.talu_validate_engine_get_position(handle)


def call_engine_get_valid_bytes(handle: "Any") -> list[bool]:
    """Get valid bytes at current position.

    Returns 256-element list where True means that byte value is valid.
    """
    valid_bytes_array = (ctypes.c_bool * 256)()
    result = _lib.talu_validate_engine_get_valid_bytes(handle, ctypes.byref(valid_bytes_array))
    if result != 0:
        return [False] * 256
    return list(valid_bytes_array)


def call_engine_can_accept(handle: "Any", data: bytes) -> bool:
    """Check if data can be accepted without advancing state."""
    return _lib.talu_validate_engine_can_accept(handle, data, len(data))

"""
Model Conversion API.

Provides tools to convert models into optimized formats for Talu or external use.

Schemes
-------
Each scheme encodes all conversion parameters (method, bits, group_size).
This eliminates invalid parameter combinations.

**Grouped Affine (GAF):**

- ``gaf4_32``: 4-bit, group_size=32 (highest accuracy)
- ``gaf4_64``: 4-bit, group_size=64 (DEFAULT, balanced)
- ``gaf4_128``: 4-bit, group_size=128 (smallest)
- ``gaf8_32``: 8-bit, group_size=32
- ``gaf8_64``: 8-bit, group_size=64
- ``gaf8_128``: 8-bit, group_size=128

**Hardware Float (Not Yet Implemented):**

- ``fp8_e4m3``: FP8 E4M3 for inference (H100/vLLM)
- ``fp8_e5m2``: FP8 E5M2 for training
- ``mxfp4``: OCP Microscaling 4-bit
- ``nvfp4``: NVIDIA Blackwell 4-bit

Memory Usage
------------
The converter uses **mmap** for zero-copy file reading, which means:

- Source model weights are memory-mapped, not loaded into RAM
- Only the tensor currently being quantized is in memory at a time
- Peak RAM usage ≈ size of largest single tensor × 2-3

For a 70B model with typical ~1GB tensors, expect ~3-4GB peak RAM usage regardless
of total model size. This makes conversion feasible on constrained hardware (e.g.,
Apple Silicon Macs with 16GB relying on swap).

**Note:** For sharded source models (multiple .safetensors files), shards are
loaded on-demand as tensors are accessed, further reducing memory pressure.

Usage
-----
>>> import talu

>>> # Default: 4-bit grouped affine (gaf4_64)
>>> talu.convert("Qwen/Qwen3-7B")

>>> # Higher accuracy with smaller groups
>>> talu.convert("Qwen/Qwen3-7B", scheme="gaf4_32")

>>> # 8-bit near-lossless
>>> talu.convert("Qwen/Qwen3-7B", scheme="gaf8_64")

>>> # List all available schemes
>>> talu.list_schemes()

See Also
--------
convert : Convert models using one of these schemes.
list_schemes : List all available quantization schemes with details.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Literal

from ._bindings import get_lib
from ._logging import logger
from .exceptions import ConvertError, ModelError, ValidationError

__all__ = [
    # Primary API
    "convert",
    "verify",
    "list_schemes",
    "VerificationResult",
    # Inspection
    "describe",
    "ModelInfo",
]


# =============================================================================
# Progress Types (Unified Progress API)
# =============================================================================


class ProgressAction:
    """Action to perform on a progress line.

    Values match ProgressAction enum in core/src/capi/progress.zig.
    """

    ADD = 0  # Add a new progress line (or reset existing)
    UPDATE = 1  # Update an existing line's progress/message
    COMPLETE = 2  # Mark line as complete and remove from display


# =============================================================================
# Scheme Enum
# =============================================================================


class Scheme:
    """
    Unified quantization scheme.

    Each scheme encodes all conversion parameters (method, bits, group_size).
    This eliminates invalid parameter combinations.

    Grouped Affine (MLX Compatible)
    -------------------------------
    - GAF4_32: 4-bit, group_size=32 (highest accuracy)
    - GAF4_64: 4-bit, group_size=64 (balanced, DEFAULT)
    - GAF4_128: 4-bit, group_size=128 (smallest)
    - GAF8_32: 8-bit, group_size=32
    - GAF8_64: 8-bit, group_size=64
    - GAF8_128: 8-bit, group_size=128

    Hardware Float (Not Yet Implemented)
    ------------------------------------
    - FP8_E4M3: FP8 E4M3 for inference (H100/vLLM)
    - FP8_E5M2: FP8 E5M2 for training
    - MXFP4: OCP Microscaling 4-bit
    - NVFP4: NVIDIA Blackwell 4-bit

    See Also
    --------
    convert : Convert models using these schemes.
    """

    # Grouped Affine (MLX compatible) - values 10-15
    GAF4_32 = 10  # 4-bit, group_size=32 (highest accuracy)
    GAF4_64 = 11  # 4-bit, group_size=64 (balanced, DEFAULT)
    GAF4_128 = 12  # 4-bit, group_size=128 (smallest)
    GAF8_32 = 13  # 8-bit, group_size=32
    GAF8_64 = 14  # 8-bit, group_size=64
    GAF8_128 = 15  # 8-bit, group_size=128

    # Hardware float (not yet implemented) - values 20-23
    FP8_E4M3 = 20  # FP8 E4M3 for inference (H100/vLLM)
    FP8_E5M2 = 21  # FP8 E5M2 for training
    MXFP4 = 22  # OCP Microscaling 4-bit (group_size=32 fixed)
    NVFP4 = 23  # NVIDIA Blackwell 4-bit (group_size=32 fixed)


# =============================================================================
# Scheme Constants and Helpers
# =============================================================================

# Scheme name to enum value mapping
SCHEME_NAME_TO_ENUM: dict[str, int] = {
    # GAF schemes
    "gaf4_32": Scheme.GAF4_32,
    "gaf4_64": Scheme.GAF4_64,
    "gaf4_128": Scheme.GAF4_128,
    "gaf8_32": Scheme.GAF8_32,
    "gaf8_64": Scheme.GAF8_64,
    "gaf8_128": Scheme.GAF8_128,
    # Hardware float schemes
    "fp8_e4m3": Scheme.FP8_E4M3,
    "fp8_e5m2": Scheme.FP8_E5M2,
    "mxfp4": Scheme.MXFP4,
    "nvfp4": Scheme.NVFP4,
}

# All scheme names
ALL_SCHEMES = frozenset(SCHEME_NAME_TO_ENUM.keys())

# Scheme categories
GAF_SCHEMES = frozenset({"gaf4_32", "gaf4_64", "gaf4_128", "gaf8_32", "gaf8_64", "gaf8_128"})
HARDWARE_SCHEMES = frozenset({"fp8_e4m3", "fp8_e5m2", "mxfp4", "nvfp4"})

# Currently implemented schemes
IMPLEMENTED_SCHEMES = GAF_SCHEMES

# Type alias for scheme literals
SchemeLiteral = Literal[
    "gaf4_32",
    "gaf4_64",
    "gaf4_128",
    "gaf8_32",
    "gaf8_64",
    "gaf8_128",
    "fp8_e4m3",
    "fp8_e5m2",
    "mxfp4",
    "nvfp4",
]


def scheme_to_enum(scheme: str) -> int:
    """
    Convert scheme name (or alias) to enum value via Zig.

    Zig is the single source of truth for aliases like "4bit" -> GAF4_64.
    This ensures Python and Zig always agree on alias resolution.

    Parameters
    ----------
    scheme : str
        Scheme name or alias (e.g., "gaf4_64", "4bit", "mlx").

    Returns
    -------
    int
        Enum value for the scheme.

    Raises
    ------
    ValidationError
        If the scheme name is not recognized.
    """
    lib = _get_convert_lib()
    code = lib.talu_convert_parse_scheme(scheme.encode("utf-8"))

    if code == -1:
        # Fetch valid schemes to show a helpful error
        try:
            scheme_data = json.loads(get_schemes_json())
            valid = sorted(scheme_data.keys())
            # Flatten aliases for the error message
            aliases = []
            for v in scheme_data.values():
                if v:
                    aliases.extend(v)
            all_valid = sorted(set(valid) | set(aliases))
        except (json.JSONDecodeError, KeyError, TypeError):
            all_valid = sorted(ALL_SCHEMES)

        raise ValidationError(
            f"Unknown scheme: {scheme!r}. Valid options: {', '.join(all_valid[:10])}...",
            code="INVALID_ARGUMENT",
            details={"param": "scheme", "value": scheme},
        )

    return code


def is_gaf_scheme(scheme: str) -> bool:
    """Check if scheme is a grouped affine (MLX compatible) scheme."""
    return scheme.lower() in GAF_SCHEMES


def is_implemented(scheme: str) -> bool:
    """Check if scheme is currently implemented."""
    return scheme.lower() in IMPLEMENTED_SCHEMES


# =============================================================================
# Platform Enum
# =============================================================================


class Platform:
    """
    Target platform for conversion.

    Used with ``quant`` parameter to resolve the appropriate scheme.

    Attributes
    ----------
    CPU : int
        Resolves to CPU-optimized GAF formats (gaf4_64, gaf8_64).
    METAL : int
        Resolves to Apple Silicon Metal formats (gaf4_64, gaf8_64).
    CUDA : int
        Resolves to NVIDIA CUDA formats (gaf4_64, gaf8_64).
    """

    CPU = 0
    METAL = 1
    CUDA = 2


# Platform name to enum value mapping
PLATFORM_NAME_TO_ENUM: dict[str, int] = {
    "cpu": Platform.CPU,
    "metal": Platform.METAL,
    "mps": Platform.METAL,
    "apple": Platform.METAL,
    "cuda": Platform.CUDA,
    "gpu": Platform.CUDA,
    "nvidia": Platform.CUDA,
}


def platform_to_enum(platform: str) -> int:
    """
    Convert platform name to enum value.

    Parameters
    ----------
    platform : str
        Platform name (e.g., "cpu", "metal", "cuda").

    Returns
    -------
    int
        Enum value for the platform.

    Raises
    ------
    ValidationError
        If the platform name is not recognized.
    """
    key = platform.lower()
    if key not in PLATFORM_NAME_TO_ENUM:
        valid = sorted(set(PLATFORM_NAME_TO_ENUM.keys()))
        raise ValidationError(
            f"Unknown platform: {platform!r}. Valid options: {', '.join(valid)}",
            code="INVALID_ARGUMENT",
            details={"param": "platform", "value": platform},
        )
    return PLATFORM_NAME_TO_ENUM[key]


# =============================================================================
# QuantLevel Enum
# =============================================================================


class QuantLevel:
    """
    Quantization level (bit precision).

    Used with ``platform`` parameter to resolve the appropriate scheme.

    Attributes
    ----------
    Q4 : int
        4-bit quantization (gaf4_64 on all platforms).
    Q8 : int
        8-bit quantization (gaf8_64 on all platforms).
    """

    Q4 = 0
    Q8 = 1


# QuantLevel name to enum value mapping
QUANT_NAME_TO_ENUM: dict[str, int] = {
    "4bit": QuantLevel.Q4,
    "q4": QuantLevel.Q4,
    "int4": QuantLevel.Q4,
    "8bit": QuantLevel.Q8,
    "q8": QuantLevel.Q8,
    "int8": QuantLevel.Q8,
}


def quant_to_enum(quant: str) -> int:
    """
    Convert quant level name to enum value.

    Parameters
    ----------
    quant : str
        Quant level name (e.g., "4bit", "8bit", "16bit").

    Returns
    -------
    int
        Enum value for the quant level.

    Raises
    ------
    ValidationError
        If the quant level name is not recognized.
    """
    key = quant.lower()
    if key not in QUANT_NAME_TO_ENUM:
        valid = sorted(set(QUANT_NAME_TO_ENUM.keys()))
        raise ValidationError(
            f"Unknown quant level: {quant!r}. Valid options: {', '.join(valid)}",
            code="INVALID_ARGUMENT",
            details={"param": "quant", "value": quant},
        )
    return QUANT_NAME_TO_ENUM[key]


# =============================================================================
# Constants
# =============================================================================

# Maximum number of override rules (must match MAX_OVERRIDES in Zig)
MAX_OVERRIDES = 32


# =============================================================================
# Ctypes Structs
# =============================================================================


class OverrideRule(ctypes.Structure):
    """
    Override rule for per-tensor quantization.

    This struct must match the layout of OverrideRule in
    core/src/capi/convert.zig.
    """

    _fields_ = [
        ("pattern", ctypes.c_char_p),  # Glob pattern (e.g., "model.layers.*.mlp.experts.*")
        ("scheme", ctypes.c_uint32),  # Scheme enum value (block schemes only)
    ]


class ConvertOptions(ctypes.Structure):
    """
    Conversion configuration passed to Zig.

    This struct must match the layout of ConvertOptions in
    core/src/converter/scheme.zig.
    """

    _fields_ = [
        ("scheme", ctypes.c_uint32),  # Scheme enum value
        ("force", ctypes.c_bool),  # Overwrite existing output
        ("offline", ctypes.c_bool),  # Disallow network access
        ("destination", ctypes.c_char_p),  # Explicit output path
        # Override rules for per-tensor quantization (block schemes only)
        ("overrides", OverrideRule * MAX_OVERRIDES),
        ("num_overrides", ctypes.c_uint32),
        # Maximum shard size in bytes (0 = no limit)
        ("max_shard_size", ctypes.c_uint64),
        # Dry run mode - estimate conversion without writing files
        ("dry_run", ctypes.c_bool),
        # Platform for scheme resolution (cpu=0, metal=1, cuda=2)
        ("platform", ctypes.c_uint32),
        # Quant level for scheme resolution (q4=0, q8=1, q16=2)
        ("quant", ctypes.c_uint32),
        # If true, resolve scheme from platform/quant instead of using scheme
        ("use_platform_quant", ctypes.c_bool),
        # Progress callback: fn(current, total, message, user_data) or NULL
        # Using c_void_p allows null values (CFUNCTYPE doesn't accept None)
        ("progress_callback", ctypes.c_void_p),
        # User-provided context pointer passed to progress_callback
        ("progress_user_data", ctypes.c_void_p),
    ]


class ConvertResult(ctypes.Structure):
    """Conversion result returned from the C API.

    Note: We use c_void_p instead of c_char_p for string fields to avoid
    ctypes auto-converting to bytes. This is necessary because:
    1. c_char_p auto-converts to Python bytes when accessed
    2. When we call talu_convert_free_string, ctypes would create a
       NEW temporary buffer for the bytes, not the original pointer
    3. The Zig allocator would then try to free an invalid pointer

    With c_void_p, we keep the original pointer and can safely free it.
    """

    _fields_ = [
        ("output_path", ctypes.c_void_p),
        ("error_msg", ctypes.c_void_p),
        ("success", ctypes.c_bool),
    ]


# =============================================================================
# C API Setup
# =============================================================================

# Note: Function signatures (argtypes/restype) are set up automatically by
# _native.py (auto-generated from Zig C API) when the library is first loaded.
#
# EXCEPTION: We override ConvertResult's restype here because we need c_void_p
# for string fields (not c_char_p as generated) to properly manage memory when
# calling talu_convert_free_string. See ConvertResult docstring for details.

_lib: ctypes.CDLL | None = None


def _get_convert_lib() -> ctypes.CDLL:
    """Get the library handle."""
    global _lib
    if _lib is None:
        _lib = get_lib()
        # Override with our ConvertResult (uses c_void_p for proper memory management)
        _lib.talu_convert.restype = ConvertResult
    return _lib


def get_schemes_json() -> str:
    """
    Fetch the JSON string of schemes and aliases from Zig.

    Returns a JSON object mapping canonical scheme names to their aliases:
    ``{"gaf4_64": ["mlx", "mlx4", "gaf4", "4bit"], ...}``

    Returns
    -------
    str
        JSON string of schemes and aliases.
    """
    lib = _get_convert_lib()
    out_ptr = ctypes.c_void_p()

    if lib.talu_convert_schemes(ctypes.byref(out_ptr)) != 0 or not out_ptr.value:
        return "{}"

    try:
        json_bytes = ctypes.cast(out_ptr.value, ctypes.c_char_p).value
        return json_bytes.decode("utf-8") if json_bytes else "{}"
    finally:
        # Cast to c_char_p since that's the argtype
        lib.talu_convert_free_string(ctypes.cast(out_ptr.value, ctypes.c_char_p))


def _call_convert(
    model: bytes, target_dir: bytes, opts: ConvertOptions
) -> tuple[bool, str | None, str | None]:
    """Call talu_convert and return (success, output_path, error_msg).

    Handles memory management for result strings.

    Returns
    -------
        Tuple of (success, output_path, error_msg).
        - On success: (True, path_string, None)
        - On failure: (False, None, error_string)
    """
    lib = _get_convert_lib()
    result = lib.talu_convert(model, target_dir, ctypes.byref(opts))

    output_path = None
    error_msg = None

    if result.output_path:
        path_bytes = ctypes.cast(result.output_path, ctypes.c_char_p).value
        if path_bytes is not None:
            output_path = path_bytes.decode("utf-8")
        lib.talu_convert_free_string(ctypes.cast(result.output_path, ctypes.c_char_p))

    if result.error_msg:
        error_bytes = ctypes.cast(result.error_msg, ctypes.c_char_p).value
        if error_bytes is not None:
            error_msg = error_bytes.decode("utf-8")
        lib.talu_convert_free_string(ctypes.cast(result.error_msg, ctypes.c_char_p))

    return (result.success, output_path, error_msg)


def _call_describe(model: bytes) -> tuple[dict | None, str | None]:
    """Call talu_describe and return (info_dict, error_msg).

    Extracts all fields from ModelInfo result and returns as dict.
    Handles memory management for result strings.

    Returns
    -------
        Tuple of (info_dict, error_msg).
        - On success: (dict with model info, None)
        - On failure: (None, error_string)
    """
    lib = get_lib()
    result = lib.talu_describe(model)

    if result.error_msg:
        error = result.error_msg.decode("utf-8")
        return (None, error)

    # Extract strings before freeing
    model_type = None
    if result.model_type:
        model_type_bytes = ctypes.cast(result.model_type, ctypes.c_char_p).value
        if model_type_bytes is not None:
            model_type = model_type_bytes.decode("utf-8")

    architecture = None
    if result.architecture:
        arch_bytes = ctypes.cast(result.architecture, ctypes.c_char_p).value
        if arch_bytes is not None:
            architecture = arch_bytes.decode("utf-8")

    info = {
        "vocab_size": result.vocab_size,
        "hidden_size": result.hidden_size,
        "num_layers": result.num_layers,
        "num_heads": result.num_heads,
        "num_kv_heads": result.num_kv_heads,
        "intermediate_size": result.intermediate_size,
        "max_seq_len": result.max_seq_len,
        "head_dim": result.head_dim,
        "rope_theta": result.rope_theta,
        "norm_eps": result.norm_eps,
        "quant_bits": result.quant_bits,
        "quant_group_size": result.quant_group_size,
        "model_type": model_type,
        "architecture": architecture,
        "tie_word_embeddings": result.tie_word_embeddings,
        "use_gelu": result.use_gelu,
        "num_experts": result.num_experts,
        "experts_per_token": result.experts_per_token,
    }

    # Free the C strings
    lib.talu_model_info_free(ctypes.byref(result))

    return (info, None)


def _call_describe_and_execution_plan(model: bytes) -> tuple[dict | None, str | None]:
    """Call talu_describe then talu_execution_plan, returning the plan dict.

    Combines describe + execution_plan + free into a single binding call so
    callers don't need ctypes for the intermediate ModelInfo struct.

    Returns
    -------
        Tuple of (plan_dict, error_msg).
        - On success: (dict with execution plan, None)
        - On failure: (None, error_string)
    """
    lib = get_lib()
    model_info = lib.talu_describe(model)

    if model_info.error_msg:
        error = model_info.error_msg.decode("utf-8")
        return (None, error)

    plan_dict, error = _call_execution_plan(model_info)

    # Free the model info strings
    lib.talu_model_info_free(ctypes.byref(model_info))

    return (plan_dict, error)


def _call_execution_plan(model_info) -> tuple[dict | None, str | None]:
    """Call talu_execution_plan and return (plan_dict, error_msg).

    Takes a ModelInfo result from talu_describe and returns the execution plan
    showing which kernels will be used.

    Returns
    -------
        Tuple of (plan_dict, error_msg).
        - On success: (dict with execution plan, None)
        - On failure: (None, error_string)
    """
    lib = get_lib()
    result = lib.talu_execution_plan(ctypes.byref(model_info))

    if result.error_msg:
        error = result.error_msg.decode("utf-8")
        return (None, error)

    plan = {
        "matmul_kernel": result.matmul_kernel.decode("utf-8")
        if result.matmul_kernel
        else "unknown",
        "attention_type": result.attention_type.decode("utf-8")
        if result.attention_type
        else "unknown",
        "ffn_type": result.ffn_type.decode("utf-8") if result.ffn_type else "unknown",
        "num_layers": result.num_layers,
        "hidden_size": result.hidden_size,
        "num_heads": result.num_heads,
        "num_kv_heads": result.num_kv_heads,
        "head_dim": result.head_dim,
        "num_experts": result.num_experts,
        "experts_per_token": result.experts_per_token,
        "quant_bits": result.quant_bits,
        "quant_group_size": result.quant_group_size,
        "uses_gqa": result.uses_gqa,
        "uses_moe": result.uses_moe,
        "uses_quantization": result.uses_quantization,
        "uses_gelu": result.uses_gelu,
    }

    return (plan, None)


# =============================================================================
# Scheme Metadata (formats)
# =============================================================================

# Descriptive info for all schemes
SCHEME_INFO: dict[str, dict] = {
    # Grouped Affine (MLX compatible)
    "gaf4_32": {
        "category": "gaf",
        "bits": 4,
        "group_size": 32,
        "description": "4-bit grouped affine (highest accuracy)",
        "quality": "good",
        "size": "~2.5 GB/7B",
        "status": "stable",
        "mlx_compatible": True,
    },
    "gaf4_64": {
        "category": "gaf",
        "bits": 4,
        "group_size": 64,
        "description": "4-bit grouped affine (balanced, DEFAULT)",
        "quality": "good",
        "size": "~2.5 GB/7B",
        "status": "stable",
        "mlx_compatible": True,
        "default": True,
    },
    "gaf4_128": {
        "category": "gaf",
        "bits": 4,
        "group_size": 128,
        "description": "4-bit grouped affine (smallest)",
        "quality": "fair",
        "size": "~2.4 GB/7B",
        "status": "stable",
        "mlx_compatible": True,
    },
    "gaf8_32": {
        "category": "gaf",
        "bits": 8,
        "group_size": 32,
        "description": "8-bit grouped affine (highest accuracy)",
        "quality": "high",
        "size": "~4.5 GB/7B",
        "status": "stable",
        "mlx_compatible": True,
    },
    "gaf8_64": {
        "category": "gaf",
        "bits": 8,
        "group_size": 64,
        "description": "8-bit grouped affine (balanced)",
        "quality": "high",
        "size": "~4.5 GB/7B",
        "status": "stable",
        "mlx_compatible": True,
    },
    "gaf8_128": {
        "category": "gaf",
        "bits": 8,
        "group_size": 128,
        "description": "8-bit grouped affine (smallest)",
        "quality": "near-lossless",
        "size": "~4.4 GB/7B",
        "status": "stable",
        "mlx_compatible": True,
    },
    # Hardware float (not yet implemented)
    "fp8_e4m3": {
        "category": "hardware",
        "bits": 8,
        "description": "FP8 E4M3 for inference (H100/vLLM)",
        "quality": "good",
        "status": "not implemented",
    },
    "fp8_e5m2": {
        "category": "hardware",
        "bits": 8,
        "description": "FP8 E5M2 for training",
        "quality": "good",
        "status": "not implemented",
    },
    "mxfp4": {
        "category": "hardware",
        "bits": 4,
        "group_size": 32,
        "description": "OCP Microscaling 4-bit",
        "quality": "good",
        "status": "not implemented",
    },
    "nvfp4": {
        "category": "hardware",
        "bits": 4,
        "group_size": 32,
        "description": "NVIDIA Blackwell 4-bit",
        "quality": "good",
        "status": "not implemented",
    },
}


def list_schemes(
    include_unimplemented: bool = False,
    category: str | None = None,
) -> dict[str, dict]:
    """
    List available quantization schemes with descriptions and aliases.

    Returns detailed information about each scheme to help users make
    informed decisions about which scheme to use. Fetches live alias
    information from the core runtime (Zig) to ensure consistency.

    Parameters
    ----------
    include_unimplemented : bool, default False
        If True, include schemes that are not yet implemented
        (fp8_e4m3, fp8_e5m2, mxfp4, nvfp4).
    category : str, optional
        Filter by category: "gaf" or "hardware".
        If None, returns all categories.

    Returns
    -------
    dict
        Dictionary mapping scheme names to their metadata:

        - ``category``: "gaf" or "hardware"
        - ``bits``: Bit width
        - ``group_size``: (gaf/hardware only) Group size
        - ``description``: What this scheme does
        - ``quality``: Relative quality (fair/good/better/high/near-lossless/lossless)
        - ``size``: Approximate size for a 7B model
        - ``status``: "stable" or "not implemented"
        - ``mlx_compatible``: (gaf only) True if compatible with MLX
        - ``aliases``: List of user-friendly aliases (e.g., ["4bit", "q4"])

    Example
    -------
    >>> import talu
    >>> schemes = talu.list_schemes()
    >>> "gaf4_64" in schemes
    True
    >>> "description" in schemes["gaf4_64"]
    True
    >>> "4bit" in schemes["gaf4_64"]["aliases"]
    True

    See Also
    --------
    convert : Convert a model using one of these schemes.
    """
    result = {}

    # Fetch aliases from Zig (Single Source of Truth)
    zig_aliases: dict[str, list[str]] = {}
    try:
        zig_json = get_schemes_json()
        zig_aliases = json.loads(zig_json)
    except (OSError, json.JSONDecodeError):
        pass  # Fallback to no aliases if Zig not ready

    for name, info in SCHEME_INFO.items():
        # Filter by implementation status
        if not include_unimplemented and info.get("status") == "not implemented":
            continue

        # Filter by category
        if category and info.get("category") != category:
            continue

        entry = info.copy()
        # Merge aliases from Zig
        entry["aliases"] = zig_aliases.get(name, [])

        result[name] = entry

    return result


def schemes() -> list[str]:
    """
    Get all available scheme names (implemented only).

    Returns
    -------
    list[str]
        List of scheme names sorted alphabetically.

    Example
    -------
    >>> schemes()
    ['gaf4_128', 'gaf4_32', 'gaf4_64', 'gaf8_128', 'gaf8_32', 'gaf8_64']
    """
    return sorted(IMPLEMENTED_SCHEMES)


# =============================================================================
# Model Inspection
# =============================================================================


class ModelInfo:
    """
    Model architecture and configuration information.

    This class provides a read-only view of model metadata extracted from
    config.json without loading the model weights. Useful for:

    - Pre-flight checks before conversion
    - Comparing model architectures
    - Determining quantization status
    - Checking MoE configuration

    Attributes
    ----------
    vocab_size : int
        Vocabulary size
    hidden_size : int
        Hidden dimension (d_model)
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of attention heads
    num_kv_heads : int
        Number of key-value heads (for GQA)
    intermediate_size : int
        FFN intermediate dimension
    max_seq_len : int
        Maximum sequence length
    head_dim : int
        Dimension per attention head
    rope_theta : float
        RoPE base frequency
    norm_eps : float
        Layer norm epsilon
    quant_bits : int
        Quantization bits (4, 8, or 16)
    quant_group_size : int
        Quantization group size
    model_type : str or None
        Model type string (e.g., "qwen3", "llama")
    architecture : str or None
        Architecture class name
    tie_word_embeddings : bool
        Whether embeddings are tied
    use_gelu : bool
        Whether GELU activation is used
    num_experts : int
        Number of MoE experts (0 if not MoE)
    experts_per_token : int
        Experts used per token

    Example
    -------
    >>> from talu.converter import describe
    >>> info = describe("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    >>> info.num_layers  # doctest: +SKIP
    28
    >>> info.is_quantized  # doctest: +SKIP
    False
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        head_dim: int,
        rope_theta: float,
        norm_eps: float,
        quant_bits: int,
        quant_group_size: int,
        model_type: str | None,
        architecture: str | None,
        tie_word_embeddings: bool,
        use_gelu: bool,
        num_experts: int,
        experts_per_token: int,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.model_type = model_type
        self.architecture = architecture
        self.tie_word_embeddings = tie_word_embeddings
        self.use_gelu = use_gelu
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    @property
    def is_quantized(self) -> bool:
        """Whether the model is quantized (not fp16)."""
        return self.quant_bits < 16

    @property
    def is_moe(self) -> bool:
        """Whether the model uses Mixture of Experts."""
        return self.num_experts > 0

    def __repr__(self) -> str:
        quant_str = f"Q{self.quant_bits}" if self.is_quantized else "FP16"
        return (
            f"ModelInfo({self.architecture or self.model_type or 'unknown'}, "
            f"layers={self.num_layers}, hidden={self.hidden_size}, "
            f"heads={self.num_heads}, {quant_str})"
        )


def describe(model: str) -> ModelInfo:
    """
    Get model architecture and configuration information.

    Reads config.json without loading model weights. This is useful for
    pre-flight checks before conversion or understanding model structure.

    Parameters
    ----------
    model : str
        Path to model directory or HuggingFace model ID.

    Returns
    -------
    ModelInfo
        Object containing model configuration details.

    Raises
    ------
    ModelError
        If model cannot be loaded or parsed.

    Example
    -------
    >>> from talu.converter import describe
    >>> info = describe("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    >>> info.num_layers > 0  # doctest: +SKIP
    True
    """
    info_dict, error = _call_describe(model.encode("utf-8"))

    if error:
        raise ModelError(f"Failed to describe model '{model}': {error}")

    if info_dict is None:
        raise ModelError(f"Failed to describe model '{model}': unknown error")

    return ModelInfo(
        vocab_size=info_dict["vocab_size"],
        hidden_size=info_dict["hidden_size"],
        num_layers=info_dict["num_layers"],
        num_heads=info_dict["num_heads"],
        num_kv_heads=info_dict["num_kv_heads"],
        intermediate_size=info_dict["intermediate_size"],
        max_seq_len=info_dict["max_seq_len"],
        head_dim=info_dict["head_dim"],
        rope_theta=info_dict["rope_theta"],
        norm_eps=info_dict["norm_eps"],
        quant_bits=info_dict["quant_bits"],
        quant_group_size=info_dict["quant_group_size"],
        model_type=info_dict["model_type"],
        architecture=info_dict["architecture"],
        tie_word_embeddings=info_dict["tie_word_embeddings"],
        use_gelu=info_dict["use_gelu"],
        num_experts=info_dict["num_experts"],
        experts_per_token=info_dict["experts_per_token"],
    )


class ExecutionPlan:
    """
    Execution plan showing which kernels will be used for a model.

    This provides static analysis of kernel selection based on model config.
    Use this to understand which code paths need optimization for a given model.

    Attributes
    ----------
    matmul_kernel : str
        Matmul kernel that will be used (e.g., "matmul_bf16", "matmul_grouped_affine_u4")
    attention_type : str
        Attention implementation (e.g., "MultiHeadAttention", "GroupedQueryAttention")
    ffn_type : str
        FFN type (e.g., "SwiGLU(SiLU)", "MoE(SiLU)")
    num_layers : int
        Number of transformer layers
    hidden_size : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    num_kv_heads : int
        Number of key-value heads
    head_dim : int
        Head dimension
    num_experts : int
        Number of MoE experts (0 if not MoE)
    experts_per_token : int
        Experts per token for MoE
    quant_bits : int
        Quantization bits
    quant_group_size : int
        Quantization group size
    uses_gqa : bool
        Whether model uses grouped-query attention
    uses_moe : bool
        Whether model uses mixture of experts
    uses_quantization : bool
        Whether model is quantized
    uses_gelu : bool
        Whether model uses GELU activation
    is_supported : bool
        Whether model type is supported by talu's inference engine

    Example
    -------
    >>> from talu.converter import execution_plan
    >>> plan = execution_plan("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    >>> plan.matmul_kernel  # doctest: +SKIP
    'matmul_bf16'
    """

    def __init__(
        self,
        matmul_kernel: str,
        attention_type: str,
        ffn_type: str,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_experts: int,
        experts_per_token: int,
        quant_bits: int,
        quant_group_size: int,
        uses_gqa: bool,
        uses_moe: bool,
        uses_quantization: bool,
        uses_gelu: bool,
    ):
        self.matmul_kernel = matmul_kernel
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.quant_bits = quant_bits
        self.quant_group_size = quant_group_size
        self.uses_gqa = uses_gqa
        self.uses_moe = uses_moe
        self.uses_quantization = uses_quantization
        self.uses_gelu = uses_gelu

    def __repr__(self) -> str:
        features = []
        if self.uses_gqa:
            features.append("GQA")
        if self.uses_moe:
            features.append(f"MoE({self.num_experts})")
        if self.uses_quantization:
            features.append(f"Q{self.quant_bits}")
        features_str = ", ".join(features) if features else "FP16"
        return (
            f"ExecutionPlan({self.matmul_kernel}, {self.attention_type}, "
            f"{self.ffn_type}, {features_str})"
        )

    def print_plan(self) -> None:
        """Print a detailed execution plan to stdout."""
        print("=" * 64)
        print("EXECUTION PLAN")
        print("=" * 64)
        print()
        print("ARCHITECTURE")
        print(f"  Layers:         {self.num_layers}")
        print(f"  Hidden Size:    {self.hidden_size}")
        print(f"  Attention Heads:{self.num_heads}")
        print(f"  KV Heads:       {self.num_kv_heads}")
        print(f"  Head Dimension: {self.head_dim}")
        print()
        print("KERNEL SELECTION")
        print(f"  Matmul:         {self.matmul_kernel}")
        print(f"  Attention:      {self.attention_type}")
        print(f"  FFN:            {self.ffn_type}")
        if self.uses_moe:
            print()
            print("MIXTURE OF EXPERTS")
            print(f"  Num Experts:    {self.num_experts}")
            print(f"  Top-K:          {self.experts_per_token}")
        if self.uses_quantization:
            print()
            print("QUANTIZATION")
            print(f"  Bits:           {self.quant_bits}")
            print(f"  Group Size:     {self.quant_group_size}")
        print()
        print("CODE PATHS TO OPTIMIZE")
        print(f"  • compute/ops/matmul_primitives.zig → {self.matmul_kernel}")
        print(f"  • inference/backend/cpu/kernels/attention.zig → {self.attention_type}")
        if self.uses_moe:
            print(f"  • inference/backend/cpu/kernels/moe.zig → {self.ffn_type}")
        else:
            print(f"  • inference/backend/cpu/kernels/ffn.zig → {self.ffn_type}")
        print("  • inference/backend/cpu/kernels/norm.zig → rmsnormForward")
        print("=" * 64)


def execution_plan(model: str) -> ExecutionPlan:
    """
    Get execution plan showing which kernels will be used for a model.

    This is static analysis based on config.json - no model loading required.
    Use this to understand which code paths to optimize for a given model.

    Parameters
    ----------
    model : str
        Path to model directory or HuggingFace model ID.

    Returns
    -------
    ExecutionPlan
        Object containing kernel selection details.

    Raises
    ------
    ModelError
        If model cannot be loaded or parsed.

    Example
    -------
    >>> from talu.converter import execution_plan
    >>> plan = execution_plan("Qwen/Qwen3-0.6B")  # doctest: +SKIP
    >>> plan.matmul_kernel is not None  # doctest: +SKIP
    True
    """
    plan_dict, error = _call_describe_and_execution_plan(model.encode("utf-8"))

    if error:
        raise ModelError(f"Failed to get execution plan for '{model}': {error}")

    if plan_dict is None:
        raise ModelError(f"Failed to get execution plan for '{model}': unknown error")

    return ExecutionPlan(
        matmul_kernel=plan_dict["matmul_kernel"],
        attention_type=plan_dict["attention_type"],
        ffn_type=plan_dict["ffn_type"],
        num_layers=plan_dict["num_layers"],
        hidden_size=plan_dict["hidden_size"],
        num_heads=plan_dict["num_heads"],
        num_kv_heads=plan_dict["num_kv_heads"],
        head_dim=plan_dict["head_dim"],
        num_experts=plan_dict["num_experts"],
        experts_per_token=plan_dict["experts_per_token"],
        quant_bits=plan_dict["quant_bits"],
        quant_group_size=plan_dict["quant_group_size"],
        uses_gqa=plan_dict["uses_gqa"],
        uses_moe=plan_dict["uses_moe"],
        uses_quantization=plan_dict["uses_quantization"],
        uses_gelu=plan_dict["uses_gelu"],
    )


# =============================================================================
# Verification
# =============================================================================

# Default prompt for verification (simple, works with any model)
_VERIFY_PROMPT = "The capital of France is"


class VerificationResult:
    """
    Result of model verification.

    Attributes
    ----------
    success : bool
        Whether verification passed.
    model_path : str
        Path to the verified model.
    output : str
        Generated output from the test prompt.
    tokens_generated : int
        Number of tokens successfully generated.
    error : str | None
        Error message if verification failed.

    Example
    -------
    >>> result = VerificationResult(success=True, model_path="/path/to/model")
    >>> bool(result)
    True
    >>> result.success
    True
    """

    __slots__ = ("success", "model_path", "output", "tokens_generated", "error")

    def __init__(
        self,
        *,
        success: bool,
        model_path: str,
        output: str = "",
        tokens_generated: int = 0,
        error: str | None = None,
    ):
        self.success = success
        self.model_path = model_path
        self.output = output
        self.tokens_generated = tokens_generated
        self.error = error

    def __repr__(self) -> str:
        if self.success:
            return f"VerificationResult(success=True, tokens={self.tokens_generated})"
        return f"VerificationResult(success=False, error={self.error!r})"

    def __bool__(self) -> bool:
        return self.success


def _verify_model_impl(
    model_path: str,
    *,
    prompt: str | None = None,
    max_tokens: int = 5,
) -> VerificationResult:
    """Verify a model after conversion."""
    # Import here to avoid circular imports
    from .chat import Chat
    from .tokenizer import Tokenizer

    test_prompt = prompt or _VERIFY_PROMPT

    try:
        # Try to load and generate
        with Chat(model_path) as chat:
            response = chat(test_prompt, max_tokens=max_tokens, stream=False)
            output = str(response)

        # Count tokens accurately using the tokenizer
        with Tokenizer(model_path) as tokenizer:
            tokens_generated = tokenizer.count_tokens(output, special_tokens=False) if output else 0

        return VerificationResult(
            success=True,
            model_path=model_path,
            output=output,
            tokens_generated=tokens_generated,
        )

    except (ImportError, FileNotFoundError, ValueError, TypeError, RuntimeError, OSError) as e:
        return VerificationResult(
            success=False,
            model_path=model_path,
            error=str(e),
        )


# =============================================================================
# Converter Class
# =============================================================================


def _get_default_talu_dir() -> str:
    """Get default Talu models directory.

    Resolution: $TALU_HOME/models > ~/.cache/talu/models
    """
    talu_home = os.environ.get("TALU_HOME")
    if talu_home:
        base = Path(talu_home)
    else:
        base = Path.home() / ".cache" / "talu"
    return str(base / "models")


def _parse_size(size: int | str) -> int:
    """
    Parse a size value to bytes.

    Parameters
    ----------
    size : int | str
        Size as bytes (int) or human-readable string (e.g., "5GB", "500MB").

    Returns
    -------
    int
        Size in bytes.

    Raises
    ------
    ValueError
        If string format is invalid.
    """
    if isinstance(size, int):
        return size

    size_str = size.strip().upper()

    # Parse numeric part and unit (ordered longest-first to avoid partial matches)
    units = [("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)]

    for unit, multiplier in units:
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)].strip())
                return int(value * multiplier)
            except ValueError as err:
                raise ValueError(f"Invalid size format: {size!r}") from err

    # Try parsing as plain number
    try:
        return int(size_str)
    except ValueError as err:
        raise ValueError(
            f"Invalid size format: {size!r}. "
            "Use bytes (int) or human-readable string (e.g., '5GB', '500MB')."
        ) from err


def verify(
    model_path: str,
    *,
    prompt: str | None = None,
    max_tokens: int = 5,
) -> VerificationResult:
    """
    Verify a model can load and generate text.

    Performs a quick sanity check by loading the model and generating
    a few tokens. This catches corruption, missing files, and basic
    inference failures early.

    Parameters
    ----------
    model_path : str
        Path to the model directory to verify.
    prompt : str, optional
        Custom prompt to use. Defaults to "The capital of France is".
    max_tokens : int, default 5
        Number of tokens to generate. Keep small for speed.

    Returns
    -------
    VerificationResult
        Result with success status, output, and any error message.

    Examples
    --------
    Basic verification::

        >>> result = talu.verify("./models/qwen3-q4")
        >>> if result:
        ...     print(f"OK: generated {result.tokens_generated} tokens")
        ... else:
        ...     print(f"FAILED: {result.error}")

    With custom prompt::

        >>> result = talu.verify(
        ...     "./models/qwen3-q4",
        ...     prompt="2 + 2 =",
        ...     max_tokens=3,
        ... )
        >>> print(result.output)  # Should be " 4" or similar
    """
    return _verify_model_impl(model_path, prompt=prompt, max_tokens=max_tokens)


# =============================================================================
# Convert Function
# =============================================================================


def convert(
    model: str,
    *,
    scheme: str | None = None,
    platform: str | None = None,
    quant: str | None = None,
    output_dir: str | None = None,
    destination: str | None = None,
    force: bool = False,
    offline: bool = False,
    verify: bool = False,
    overrides: dict[str, str] | None = None,
    max_shard_size: int | str | None = None,
    dry_run: bool = False,
) -> str | dict:
    """
    Convert a model to an optimized format for efficient inference.

    Parameters
    ----------
    model : str
        Model to convert. Can be:

        - **Model ID**: ``"Qwen/Qwen3-0.6B"``, ``"meta-llama/Llama-3-8B"``
        - **Local path**: ``"./my-model"`` or ``"/path/to/model"``

    scheme : str, optional
        Explicit quantization scheme. Each scheme encodes all necessary parameters
        (method, bits, group_size). You can use specific keys or aliases.

        **If not set**, uses ``platform`` and ``quant`` for automatic resolution.
        When neither ``scheme`` nor ``platform``/``quant`` are specified,
        defaults to ``gaf4_64``.

        **User-Friendly Aliases (Recommended):**

        - ``"4bit"`` / ``"q4"`` / ``"int4"``: Maps to ``gaf4_64`` (balanced 4-bit)
        - ``"8bit"`` / ``"q8"`` / ``"int8"``: Maps to ``gaf8_64`` (near-lossless)
        - ``"mlx"`` / ``"mlx4"`` / ``"gaf4"``: Maps to ``gaf4_64`` (Apple Silicon optimized)
        - ``"mlx8"`` / ``"gaf8"``: Maps to ``gaf8_64`` (8-bit MLX)
        - ``"fp8"``: Maps to ``fp8_e4m3`` (H100/vLLM inference)

        **Grouped Affine (MLX compatible):**
        ``gaf4_32``, ``gaf4_64``, ``gaf4_128``, ``gaf8_32``, ``gaf8_64``, ``gaf8_128``

        **Hardware float (not yet implemented):**
        ``fp8_e4m3``, ``fp8_e5m2``, ``mxfp4``, ``nvfp4``

    platform : str, optional
        Target platform for scheme resolution (``"cpu"``, ``"metal"``, ``"cuda"``).
        When set, resolves to the appropriate scheme for that platform and quant level.
        Ignored if ``scheme`` is explicitly set.
    quant : str, optional
        Quantization level (``"4bit"``, ``"8bit"``). Used with ``platform``.
        Defaults to ``"4bit"`` if platform is set but quant is not.
    output_dir : str, optional
        Parent directory for auto-named output.
        Defaults to ``~/.cache/talu/models`` (or ``$TALU_HOME/models``).
        Ignored if ``destination`` is set.
    destination : str, optional
        Explicit output path (overrides ``output_dir``).
    force : bool, default False
        Overwrite existing output directory.
    offline : bool, default False
        If True, do not use network access when resolving model URIs.
    verify : bool, default False
        After conversion, verify the model by loading it and generating
        a few tokens. Catches corruption, missing files, and basic
        inference failures early.
    overrides : dict, optional
        Reserved for future use. Not currently supported.
    max_shard_size : int | str, optional
        Maximum size per shard file. When set, splits large models into
        multiple SafeTensors files. Can be bytes (int) or human-readable
        string (e.g., ``"5GB"``, ``"500MB"``).
    dry_run : bool, default False
        If True, estimate conversion without writing files. Returns a dict
        with estimation results (total_params, estimated_size_bytes,
        shard_count, scheme, bits_per_param).

    Returns
    -------
    str | dict
        When ``dry_run=False``: Absolute path to the converted model directory.
        When ``dry_run=True``: Dict with estimation results.

    Raises
    ------
    ConvertError
        If conversion fails (network error, unsupported format, etc.),
        or if verification fails when ``verify=True``.
    ValueError
        If invalid scheme or override is provided.

    Examples
    --------
    >>> import talu
    >>> path = talu.convert("Qwen/Qwen3-0.6B")  # Uses gaf4_64 by default

    >>> path = talu.convert("Qwen/Qwen3-0.6B", scheme="gaf4_32")  # Higher quality

    >>> # Platform-aware conversion
    >>> path = talu.convert("Qwen/Qwen3-0.6B", platform="metal")  # → gaf4_64

    >>> # Sharded output for large models
    >>> path = talu.convert("meta-llama/Llama-3-70B", max_shard_size="5GB")

    See Also
    --------
    list_schemes : List all available quantization schemes.
    verify : Verify a converted model.
    """
    # Determine mode: explicit scheme vs platform/quant resolution
    use_platform_quant = scheme is None and platform is not None

    # Build configuration
    opts = ConvertOptions()
    opts.force = force
    opts.offline = offline
    opts.destination = destination.encode("utf-8") if destination else None
    opts.max_shard_size = _parse_size(max_shard_size) if max_shard_size else 0
    opts.dry_run = dry_run

    if use_platform_quant:
        # Platform/quant mode: let Zig resolve the scheme
        assert platform is not None  # Guaranteed by use_platform_quant check
        opts.platform = platform_to_enum(platform)
        opts.quant = quant_to_enum(quant) if quant else QuantLevel.Q4
        opts.use_platform_quant = True
        # scheme field is ignored when use_platform_quant is True
        opts.scheme = 0
        log_scheme = f"platform={platform}, quant={quant or '4bit'}"

        # Cannot use overrides with platform/quant (scheme is resolved by Zig)
        if overrides:
            raise ValidationError(
                "Per-tensor overrides cannot be used with platform/quant mode. "
                "Use an explicit scheme parameter instead."
            )
    else:
        # Explicit scheme mode (default: gaf4_64)
        effective_scheme = scheme or "gaf4_64"

        # Validate scheme (delegates to Zig for alias resolution)
        scheme_enum = scheme_to_enum(effective_scheme)

        # Get the canonical name for validation (reverse lookup)
        scheme_lower = effective_scheme.lower()
        canonical_scheme = scheme_lower
        for name, enum_val in SCHEME_NAME_TO_ENUM.items():
            if enum_val == scheme_enum:
                canonical_scheme = name
                break

        if not is_implemented(canonical_scheme):
            raise ConvertError(
                f"Scheme '{effective_scheme}' is not yet implemented. "
                f"Available schemes: {sorted(IMPLEMENTED_SCHEMES)}"
            )

        # Overrides are not supported for GAF schemes
        if overrides:
            raise ValidationError(
                "Per-tensor overrides are not supported for GAF schemes. "
                "All tensors use uniform quantization."
            )

        opts.scheme = scheme_enum
        opts.use_platform_quant = False
        log_scheme = effective_scheme

    # Log start of conversion
    logger.info(
        "Converting model",
        extra={"scope": "converter", "model_id": model, "scheme": log_scheme},
    )

    # Execute C conversion
    target_dir = output_dir or _get_default_talu_dir()
    success, final_path, error_msg = _call_convert(
        model.encode("utf-8"),
        target_dir.encode("utf-8"),
        opts,
    )

    if not success:
        error = error_msg or "Unknown error"
        # Add Python-specific hint for "output exists" error
        if "Output directory already exists" in error:
            error = f"{error} (use force=True to overwrite)"
        raise ConvertError(error)

    final_path = final_path or ""

    # Handle dry run response (JSON)
    if dry_run:
        return json.loads(final_path)

    output_path = str(Path(final_path).absolute())

    # Verify if requested
    if verify:
        verification = _verify_model_impl(output_path)
        if not verification:
            raise ConvertError(
                f"Verification failed: {verification.error}. "
                f"The converted model at {output_path} may be corrupted."
            )

    return output_path

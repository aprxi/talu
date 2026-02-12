"""
FFI bindings for X-Ray tensor inspection.

Justification: Provides wrapper functions and dataclasses for tensor inspection
that require ctypes pointer handling and struct field access beyond what
_native.py exports.
"""

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from .._bindings import get_lib

# Import C structs from auto-generated bindings
from .._native import CapturedTensorInfo, TensorStats

_lib = get_lib()


# =============================================================================
# Constants
# =============================================================================


class CaptureMode(IntEnum):
    """What data to capture from each tensor."""

    STATS = 0  # Statistics only: min, max, mean, rms, nan_count, inf_count
    SAMPLE = 1  # Statistics + first N tensor values
    FULL = 2  # Complete tensor data copy


class Point(IntEnum):
    """Trace points in the inference pipeline.

    Names correspond to actual method signatures in the codebase.
    """

    EMBED = 0  # EmbeddingKernel.forward()
    EMBED_POS = 1
    LAYER_INPUT = 2
    LAYER_ATTN_NORM = 3
    ATTN_Q = 4  # AttentionKernel.forward() - Q projection
    ATTN_K = 5  # AttentionKernel.forward() - K projection
    ATTN_V = 6  # AttentionKernel.forward() - V projection
    ATTN_QK = 7
    ATTN_WEIGHTS = 8
    ATTN_OUT = 9  # AttentionKernel.forward() - output projection
    LAYER_FFN_NORM = 10
    FFN_GATE = 11  # FfnKernel.forward() - gate/up projection
    FFN_UP = 12
    FFN_ACT = 13
    FFN_DOWN = 14  # FfnKernel.forward() - down projection
    BLOCK_OUT = 15  # model.forward() - after residual add
    MAMBA_OUT = 16  # MambaKernel.forward()
    CONV_IN_PROJ = 17  # ShortConvKernel - input projection
    CONV_CONV = 18  # ShortConvKernel - depthwise convolution
    CONV_OUT_PROJ = 19  # ShortConvKernel - output projection
    FINAL_NORM = 20
    LM_HEAD = 21  # lm_head matmul
    LOGITS_SCALED = 22


# Point bitmasks for configuration
POINT_EMBED = 1 << 0
POINT_EMBED_POS = 1 << 1
POINT_LAYER_INPUT = 1 << 2
POINT_LAYER_ATTN_NORM = 1 << 3
POINT_ATTN_Q = 1 << 4
POINT_ATTN_K = 1 << 5
POINT_ATTN_V = 1 << 6
POINT_ATTN_QK = 1 << 7
POINT_ATTN_WEIGHTS = 1 << 8
POINT_ATTN_OUT = 1 << 9
POINT_LAYER_FFN_NORM = 1 << 10
POINT_FFN_GATE = 1 << 11
POINT_FFN_UP = 1 << 12
POINT_FFN_ACT = 1 << 13
POINT_FFN_DOWN = 1 << 14
POINT_BLOCK_OUT = 1 << 15
POINT_MAMBA_OUT = 1 << 16
POINT_CONV_IN_PROJ = 1 << 17
POINT_CONV_CONV = 1 << 18
POINT_CONV_OUT_PROJ = 1 << 19
POINT_FINAL_NORM = 1 << 20
POINT_LM_HEAD = 1 << 21
POINT_LOGITS_SCALED = 1 << 22
POINT_ALL = 0x7FFFFF  # All 23 points


def points_to_mask(points: Sequence[str | Point]) -> int:
    """Convert list of point names/enums to bitmask."""
    mask = 0
    for p in points:
        if isinstance(p, str):
            # Convert string to Point enum
            p = Point[p.upper()]
        mask |= 1 << p.value
    return mask


# =============================================================================
# C Structures (imported from auto-generated _native.py)
# =============================================================================
# TensorStats and CapturedTensorInfo are imported at the top of this file


# =============================================================================
# Python dataclasses for results
# =============================================================================


@dataclass
class Stats:
    """Statistics for a captured tensor."""

    count: int
    min: float
    max: float
    mean: float
    rms: float
    nan_count: int
    inf_count: int

    @property
    def has_anomalies(self) -> bool:
        """Check if tensor has NaN or Inf values."""
        return self.nan_count > 0 or self.inf_count > 0

    @classmethod
    def from_c(cls, c_stats: TensorStats) -> "Stats":
        """
        Create a Stats instance from a C TensorStats structure.

        Args:
            c_stats: The C structure containing tensor statistics.

        Returns
        -------
            A new Stats instance populated from the C data.
        """
        return cls(
            count=c_stats.count,
            min=c_stats.min,
            max=c_stats.max,
            mean=c_stats.mean,
            rms=c_stats.rms,
            nan_count=c_stats.nan_count,
            inf_count=c_stats.inf_count,
        )


@dataclass
class CapturedTensor:
    """A captured tensor record."""

    point: Point
    layer: int  # 0xFFFF means not applicable
    token: int
    position: int
    shape: tuple[int, ...]
    ndim: int
    dtype: int
    stats: Stats
    timestamp_ns: int = 0
    samples: list[float] | None = None

    @property
    def point_name(self) -> str:
        """Get the name of the trace point."""
        return self.point.name.lower()

    @property
    def is_layer_point(self) -> bool:
        """Check if this is a per-layer point."""
        return self.layer != 0xFFFF

    @classmethod
    def from_c(
        cls, info: CapturedTensorInfo, samples: list[float] | None = None
    ) -> "CapturedTensor":
        """
        Create a CapturedTensor instance from a C CapturedTensorInfo structure.

        Args:
            info: The C structure containing captured tensor metadata.
            samples: Optional list of sample values from the tensor.

        Returns
        -------
            A new CapturedTensor instance populated from the C data.
        """
        shape = tuple(info.shape[i] for i in range(info.ndim))
        return cls(
            point=Point(info.point),
            layer=info.layer,
            token=info.token,
            position=info.position,
            shape=shape,
            ndim=info.ndim,
            dtype=info.dtype,
            stats=Stats.from_c(info.stats),
            timestamp_ns=info.timestamp_ns,
            samples=samples,
        )


@dataclass
class AnomalyLocation:
    """Location where an anomaly (NaN/Inf) was found."""

    point: Point
    layer: int
    token: int

    @property
    def point_name(self) -> str:
        """Lowercase name of the capture point (e.g., 'attention', 'mlp')."""
        return self.point.name.lower()


# Note: Function signatures (argtypes/restype) are set up automatically by
# _native.py (auto-generated from Zig C API) when the library is first loaded.


# =============================================================================
# Wrapper functions
# =============================================================================


def capture_create(
    points: int | list[str | Point] = POINT_ALL,
    mode: CaptureMode = CaptureMode.STATS,
    sample_count: int = 8,
) -> ctypes.c_void_p:
    """Create a new capture handle.

    Args:
        points: Bitmask of points to capture, or list of point names/enums.
                Use POINT_ALL for all points.
        mode: What data to capture (STATS, SAMPLE, or FULL).
        sample_count: Number of sample values to capture (for SAMPLE mode).

    Returns
    -------
        Opaque capture handle.

    Raises
    ------
        MemoryError: If allocation fails.
    """
    if isinstance(points, list):
        points = points_to_mask(points)

    if points == POINT_ALL:
        handle = _lib.talu_xray_capture_create_all(mode, sample_count)
    else:
        handle = _lib.talu_xray_capture_create(points, mode, sample_count)

    if not handle:  # pragma: no cover
        raise MemoryError("Failed to create capture")
    return handle


def capture_enable(handle: ctypes.c_void_p) -> None:
    """Enable capturing (start receiving trace emissions)."""
    _lib.talu_xray_capture_enable(handle)


def capture_disable() -> None:
    """Disable capturing (stop receiving trace emissions)."""
    _lib.talu_xray_capture_disable()


def capture_is_enabled() -> bool:
    """Check if capturing is currently enabled."""
    return _lib.talu_xray_capture_is_enabled()


def capture_clear(handle: ctypes.c_void_p) -> None:
    """Clear all captured data (keep configuration)."""
    _lib.talu_xray_capture_clear(handle)


def capture_count(handle: ctypes.c_void_p) -> int:
    """Get number of captured records."""
    return _lib.talu_xray_capture_count(handle)


def capture_overflow(handle: ctypes.c_void_p) -> bool:
    """Check if capture overflowed (memory limit exceeded)."""
    return _lib.talu_xray_capture_overflow(handle)


def capture_destroy(handle: ctypes.c_void_p) -> None:
    """Destroy capture and free memory."""
    _lib.talu_xray_capture_destroy(handle)


def capture_get(handle: ctypes.c_void_p, index: int, max_samples: int = 8) -> CapturedTensor | None:
    """Get captured tensor info by index.

    Args:
        handle: Capture handle.
        index: Index of the record (0 to count-1).
        max_samples: Maximum number of sample values to retrieve.

    Returns
    -------
        CapturedTensor or None if index out of range.
    """
    info = CapturedTensorInfo()
    if not _lib.talu_xray_get(handle, index, ctypes.byref(info)):
        return None

    # Get samples if available
    samples = None
    if max_samples > 0:
        sample_buf = (ctypes.c_float * max_samples)()
        n = _lib.talu_xray_get_samples(handle, index, sample_buf, max_samples)
        if n > 0:
            samples = list(sample_buf[:n])

    return CapturedTensor.from_c(info, samples)


def capture_get_data(handle: ctypes.c_void_p, index: int) -> bytes | None:
    """Get full captured tensor data as bytes (available in FULL mode)."""
    size = _lib.talu_xray_get_data_size(handle, index)
    if size <= 0:
        return None
    buf = (ctypes.c_ubyte * size)()
    copied = _lib.talu_xray_get_data(handle, index, buf, size)
    if copied <= 0:
        return None
    return bytes(buf[:copied])


def capture_find_anomaly(handle: ctypes.c_void_p) -> AnomalyLocation | None:
    """Find first tensor with anomalies (NaN or Inf).

    Returns
    -------
        AnomalyLocation or None if no anomalies found.
    """
    out_point = ctypes.c_uint8()
    out_layer = ctypes.c_uint16()
    out_token = ctypes.c_uint32()

    if _lib.talu_xray_find_anomaly(
        handle,
        ctypes.byref(out_point),
        ctypes.byref(out_layer),
        ctypes.byref(out_token),
    ):
        return AnomalyLocation(
            point=Point(out_point.value),
            layer=out_layer.value,
            token=out_token.value,
        )
    return None


def capture_count_matching(
    handle: ctypes.c_void_p,
    point: Point | str | None = None,
    layer: int | None = None,
    token: int | None = None,
) -> int:
    """Count captured tensors matching criteria.

    Args:
        handle: Capture handle.
        point: Filter by point (None = any).
        layer: Filter by layer (None = any).
        token: Filter by token (None = any).

    Returns
    -------
        Number of matching records.
    """
    # Convert to C sentinel values
    c_point = 0xFF if point is None else (Point[point.upper()] if isinstance(point, str) else point)
    c_layer = 0xFFFF if layer is None else layer
    c_token = 0xFFFFFFFF if token is None else token

    return _lib.talu_xray_count_matching(handle, c_point, c_layer, c_token)


def point_name(point: int | Point) -> str:
    """Get the name of a trace point."""
    if isinstance(point, Point):
        point = point.value
    name = _lib.talu_xray_point_name(point)
    return name.decode("utf-8") if name else "unknown"

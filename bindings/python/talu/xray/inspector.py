"""
Inspector - High-level API for tensor inspection during inference.

The Inspector captures tensor values at trace points during inference,
allowing debugging of numerical issues and validation of implementations.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from ..exceptions import StateError, ValidationError
from . import _bindings as _b
from ._bindings import (
    AnomalyLocation,
    CapturedTensor,
    CaptureMode,
    Point,
    Stats,
)

if TYPE_CHECKING:
    import ctypes


class Inspector:
    """Captures and queries tensor data during inference.

    The Inspector is configured before inference runs, then enabled to
    capture tensor values at specified trace points. After inference,
    captured data can be queried.

    Example:
        >>> inspector = Inspector(
        ...     points=["logits", "layer_residual"],
        ...     mode=CaptureMode.STATS,
        ... )
        >>> inspector.enable()
        >>> # Run inference...
        >>> inspector.disable()
        >>>
        >>> # Query results
        >>> for tensor in inspector.iter():
        ...     print(f"{tensor.point_name}: min={tensor.stats.min:.4f}")
        >>>
        >>> # Check for anomalies
        >>> if anomaly := inspector.find_anomaly():
        ...     print(f"NaN found at {anomaly.point_name} layer {anomaly.layer}")
    """

    def __init__(
        self,
        points: list[str] | list[Point] | str | int = "all",
        mode: CaptureMode = CaptureMode.STATS,
        sample_count: int = 8,
    ):
        """Create an inspector.

        Args:
            points: Which trace points to capture. Can be:
                - "all" to capture all points
                - A list of point names: ["logits", "layer_residual"]
                - A list of Point enums
                - A bitmask integer
            mode: What data to capture:
                - STATS: Statistics only (min, max, mean, rms, nan_count)
                - SAMPLE: Statistics + first N values
                - FULL: Complete tensor data
            sample_count: Number of sample values to capture (for SAMPLE mode).

        Raises
        ------
            ValidationError: If points is not a valid type.
            MemoryError: If capture allocation fails.
        """
        # Convert points to bitmask
        if points == "all":
            points_mask = _b.POINT_ALL
        elif isinstance(points, int):
            points_mask = points
        elif isinstance(points, list):
            points_mask = _b.points_to_mask(points)
        else:
            raise ValidationError(
                f"Invalid points: {points}. Expected int mask, list of Points, or CaptureAllPoints.",
                code="INVALID_ARGUMENT",
                details={"param": "points", "type": type(points).__name__},
            )

        self._handle = _b.capture_create(points_mask, mode, sample_count)
        self._mode = mode
        self._sample_count = sample_count
        self._enabled = False

    def close(self) -> None:
        """
        Release native capture resources.

        Disables capturing if active, then destroys the handle. Safe to call
        multiple times (idempotent).
        """
        if hasattr(self, "_handle") and self._handle:
            if self._enabled:
                _b.capture_disable()
                self._enabled = False
            _b.capture_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> Inspector:
        """Enable capturing on context entry."""
        self.enable()
        return self

    def __exit__(self, *args) -> None:
        """Disable capturing on context exit.

        Only disables; does not destroy. Call close() for final cleanup,
        or rely on __del__. This allows re-entering the context manager.
        """
        self.disable()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _require_handle(self) -> ctypes.c_void_p:
        """Return the native handle, raising StateError if closed."""
        handle = self._handle
        if handle is None:
            raise StateError(
                "Inspector has been closed. Create a new Inspector instance.",
                code="STATE_CLOSED",
            )
        return handle

    def enable(self) -> None:
        """Enable capturing (start receiving trace emissions)."""
        _b.capture_enable(self._require_handle())
        self._enabled = True

    def disable(self) -> None:
        """Disable capturing (stop receiving trace emissions)."""
        _b.capture_disable()
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if capturing is currently enabled."""
        return _b.capture_is_enabled()

    def clear(self) -> None:
        """Clear all captured data (keep configuration)."""
        _b.capture_clear(self._require_handle())

    @property
    def count(self) -> int:
        """Number of captured records."""
        return _b.capture_count(self._require_handle())

    @property
    def overflow(self) -> bool:
        """Check if capture overflowed (memory limit exceeded)."""
        return _b.capture_overflow(self._require_handle())

    def __len__(self) -> int:
        """Return the number of captured records."""
        return self.count

    # =========================================================================
    # Query methods
    # =========================================================================

    def get(self, index: int) -> CapturedTensor | None:
        """Get captured tensor by index.

        Args:
            index: Index of the record (0 to count-1).

        Returns
        -------
            CapturedTensor or None if index out of range.
        """
        return _b.capture_get(self._require_handle(), index, self._sample_count)

    def __getitem__(self, index: int) -> CapturedTensor:
        """Get captured tensor by index (raises IndexError if out of range)."""
        if index < 0:
            index = self.count + index
        result = self.get(index)
        if result is None:
            raise IndexError(f"Index {index} out of range (count={self.count})")
        return result

    def get_data(self, index: int) -> bytes | None:
        """Get full captured tensor data as bytes (FULL mode only)."""
        return _b.capture_get_data(self._require_handle(), index)

    def iter(
        self,
        point: Point | str | None = None,
        layer: int | None = None,
        token: int | None = None,
    ) -> Iterator[CapturedTensor]:
        """Iterate over captured tensors, optionally filtering.

        Args:
            point: Filter by trace point (None = all).
            layer: Filter by layer (None = all).
            token: Filter by token (None = all).

        Yields
        ------
            CapturedTensor objects matching the filter.
        """
        # Convert string point to enum
        if isinstance(point, str):
            point = Point[point.upper()]

        for i in range(self.count):
            tensor = self.get(i)
            if tensor is None:
                continue
            if point is not None and tensor.point != point:
                continue
            if layer is not None and tensor.layer != layer:
                continue
            if token is not None and tensor.token != token:
                continue
            yield tensor

    def __iter__(self) -> Iterator[CapturedTensor]:
        """Iterate over all captured tensors."""
        return self.iter()

    def find(
        self,
        point: Point | str | None = None,
        layer: int | None = None,
        token: int | None = None,
    ) -> CapturedTensor | None:
        """Find first tensor matching criteria.

        Args:
            point: Filter by trace point (None = any).
            layer: Filter by layer (None = any).
            token: Filter by token (None = any).

        Returns
        -------
            First matching CapturedTensor, or None if not found.
        """
        for tensor in self.iter(point=point, layer=layer, token=token):
            return tensor
        return None

    def find_anomaly(self) -> AnomalyLocation | None:
        """Find first tensor with NaN or Inf values.

        Returns
        -------
            AnomalyLocation if found, None otherwise.
        """
        return _b.capture_find_anomaly(self._require_handle())

    def count_matching(
        self,
        point: Point | str | None = None,
        layer: int | None = None,
        token: int | None = None,
    ) -> int:
        """Count tensors matching criteria.

        Args:
            point: Filter by trace point (None = any).
            layer: Filter by layer (None = any).
            token: Filter by token (None = any).

        Returns
        -------
            Number of matching records.
        """
        return _b.capture_count_matching(self._require_handle(), point, layer, token)

    # =========================================================================
    # Analysis methods
    # =========================================================================

    def layer_stats(self, token: int = 0) -> list[Stats]:
        """Get statistics for all layer outputs for a specific token.

        Args:
            token: Token index to get stats for.

        Returns
        -------
            List of Stats for each layer (ordered by layer index).
        """
        results = []
        for tensor in self.iter(point=Point.BLOCK_OUT, token=token):
            results.append((tensor.layer, tensor.stats))
        # Sort by layer index
        results.sort(key=lambda x: x[0])
        return [stats for _, stats in results]

    def logits(self, token: int = 0) -> CapturedTensor | None:
        """Get logits tensor for a specific token.

        Args:
            token: Token index.

        Returns
        -------
            CapturedTensor for logits, or None if not captured.
        """
        return self.find(point=Point.LM_HEAD, token=token)

    def summary(self) -> dict:
        """Get summary of captured data.

        Returns
        -------
            Dictionary with capture summary.
        """
        total = self.count
        by_point: dict[str, int] = {}
        nan_count = 0
        inf_count = 0

        for tensor in self.iter():
            name = tensor.point_name
            by_point[name] = by_point.get(name, 0) + 1
            nan_count += tensor.stats.nan_count
            inf_count += tensor.stats.inf_count

        return {
            "total_captures": total,
            "by_point": by_point,
            "total_nan": nan_count,
            "total_inf": inf_count,
            "has_anomalies": nan_count > 0 or inf_count > 0,
            "overflow": self.overflow,
        }

    def print_summary(self) -> None:
        """Print a summary of captured data."""
        s = self.summary()
        print(f"Captured {s['total_captures']} tensors")
        if s["by_point"]:
            print("By point:")
            for name, count in sorted(s["by_point"].items()):
                print(f"  {name}: {count}")
        if s["has_anomalies"]:
            print(f"Anomalies: {s['total_nan']} NaN, {s['total_inf']} Inf")
            if loc := self.find_anomaly():
                print(f"First anomaly at: {loc.point_name} layer={loc.layer} token={loc.token}")
        if s["overflow"]:
            print("WARNING: Capture buffer overflowed")

    # =========================================================================
    # Export methods
    # =========================================================================

    def export_npz(self, path: str) -> None:
        """Export captured tensors to NPZ file for offline comparison.

        This method exports all captured tensors to an NPZ file that can be
        compared against a PyTorch reference NPZ (from your reference pipeline)
        to find the first point of divergence.

        Requires CaptureMode.FULL to have actual tensor data.

        Args:
            path: Output .npz file path

        Raises
        ------
            ValueError: If capture mode is not FULL.

        Example:
            >>> with Inspector(points="all", mode=CaptureMode.FULL) as insp:
            ...     talu.raw_complete(model_path, "Hello", max_tokens=1)
            ...     insp.export_npz("talu.npz")

        The exported NPZ can then be compared:
            $ uv run python -m talu.xray.compare _reference/qwen3.npz /tmp/talu.npz
        """
        import numpy as np

        if self._mode != CaptureMode.FULL:
            raise ValueError(
                "export_npz() requires CaptureMode.FULL to capture tensor data. "
                "Create Inspector with mode=CaptureMode.FULL"
            )

        # DType mapping: Zig DType enum values to numpy dtypes
        # See core/src/dtype.zig for enum values
        dtype_map = {
            0: np.float32,  # f32
            1: np.float64,  # f64
            2: np.int32,  # i32
            3: np.int64,  # i64
            4: np.float16,  # f16
            5: np.float32,  # bf16 -> convert to f32 for compatibility
            6: np.int8,  # i8
            7: np.int16,  # i16
            8: np.uint8,  # u8
            9: np.uint16,  # u16
            10: np.uint32,  # u32
            11: np.uint64,  # u64
        }

        tensors = {}
        for i in range(self.count):
            t = self.get(i)
            if t is None:
                continue

            data = self.get_data(i)
            if data is None:
                continue

            # Build key: "layer{N}.{point}" or just "{point}" for non-layer points
            # This matches the format expected by talu.xray.compare
            point_name = t.point_name
            if t.layer == 0xFFFF:
                key = point_name
            else:
                key = f"layer{t.layer}.{point_name}"

            # Get numpy dtype
            np_dtype = dtype_map.get(t.dtype, np.float32)

            # Handle bf16: read as uint16, reinterpret, then convert to float32
            if t.dtype == 5:  # bf16
                bf16_data = np.frombuffer(data, dtype=np.uint16)
                # Pad bf16 to f32: bf16 has same exponent as f32, just fewer mantissa bits
                # bf16 bits: seeeeeeemmmmmmm (16 bits)
                # f32 bits:  seeeeeeeemmmmmmmmmmmmmmmmmmmmmmm (32 bits)
                # Simply shift bf16 left by 16 bits to get f32
                f32_int = bf16_data.astype(np.uint32) << 16
                array = f32_int.view(np.float32).reshape(t.shape[: t.ndim])
            else:
                shape = tuple(t.shape[: t.ndim])
                array = np.frombuffer(data, dtype=np_dtype).reshape(shape)

            # Convert to float32 for consistent comparison
            tensors[key] = array.astype(np.float32)

        np.savez_compressed(path, **tensors)

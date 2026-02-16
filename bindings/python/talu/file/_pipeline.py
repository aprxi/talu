"""Pipeline configuration and ModelBuffer for image-to-tensor conversion."""

from __future__ import annotations

from typing import Any

from ..exceptions import StateError, ValidationError
from . import _bindings as ffi

# =============================================================================
# String-to-int mappings for Pipeline parameters
# =============================================================================

_FIT_MODES = {"stretch": 0, "contain": 1, "cover": 2}
_FILTERS = {"nearest": 0, "bilinear": 1, "bicubic": 2}
_DTYPES = {"uint8": 0, "float32": 1}
_LAYOUTS = {"nhwc": 0, "nchw": 1}
_NORMALIZE = {"none": 0, "zero_one": 1, "imagenet": 2}

# Reverse mappings for display
_LAYOUT_NAMES = {0: "nhwc", 1: "nchw"}
_DTYPE_NAMES = {0: "uint8", 1: "float32"}


# =============================================================================
# ModelBuffer
# =============================================================================


class ModelBuffer:
    """Model-ready tensor buffer with pixel data owned by the Zig core.

    Access the raw data as standard Python bytes::

        with pipeline(image) as buf:
            raw = buf.tobytes()
            print(buf.shape, buf.dtype)

    For zero-copy interop, ``ModelBuffer`` exposes the
    ``__array_interface__`` protocol so third-party libraries that
    support it (e.g. numpy) can read the memory directly without
    copying.

    Args:
        c_buffer: The native TaluModelBuffer struct (takes ownership).
    """

    def __init__(self, c_buffer: Any) -> None:
        self._c_buffer = c_buffer

    def _check_open(self) -> None:
        if self._c_buffer is None:
            raise StateError("ModelBuffer is closed.", code="STATE_ERROR")

    @property
    def width(self) -> int:
        """Tensor width in pixels."""
        self._check_open()
        return self._c_buffer.width

    @property
    def height(self) -> int:
        """Tensor height in pixels."""
        self._check_open()
        return self._c_buffer.height

    @property
    def channels(self) -> int:
        """Number of channels (typically 3 for RGB)."""
        self._check_open()
        return self._c_buffer.channels

    @property
    def layout(self) -> str:
        """Tensor layout ("nhwc" or "nchw")."""
        self._check_open()
        return _LAYOUT_NAMES.get(self._c_buffer.layout, "unknown")

    @property
    def dtype(self) -> str:
        """Data type ("uint8" or "float32")."""
        self._check_open()
        return _DTYPE_NAMES.get(self._c_buffer.dtype, "unknown")

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape as a tuple.

        Returns ``(height, width, channels)`` for nhwc layout,
        ``(channels, height, width)`` for nchw layout.
        """
        self._check_open()
        h, w, c = self._c_buffer.height, self._c_buffer.width, self._c_buffer.channels
        if self._c_buffer.layout == 1:  # nchw
            return (c, h, w)
        return (h, w, c)

    def tobytes(self) -> bytes:
        """Copy the tensor data to a Python bytes object.

        Returns:
            Raw tensor bytes in the buffer's layout and dtype.

        Raises:
            StateError: If the buffer is closed.
        """
        self._check_open()
        return ffi.read_buffer_raw(self._c_buffer.data, self._c_buffer.len)

    @property
    def __array_interface__(self) -> dict:
        """Array interface for zero-copy access (protocol v3).

        Third-party array libraries that support this protocol can
        read the Zig-owned memory directly without copying.

        Returns:
            Dictionary conforming to the array interface protocol v3.
        """
        self._check_open()
        buf = self._c_buffer

        # Determine typestr and element size
        if buf.dtype == 1:  # f32
            typestr = "<f4"
            elem_size = 4
        else:  # u8
            typestr = "|u1"
            elem_size = 1

        h, w, c = buf.height, buf.width, buf.channels

        if buf.layout == 1:  # nchw
            shape = (c, h, w)
            strides = (h * w * elem_size, w * elem_size, elem_size)
        else:  # nhwc
            shape = (h, w, c)
            strides = (w * c * elem_size, c * elem_size, elem_size)

        return {
            "version": 3,
            "shape": shape,
            "typestr": typestr,
            "data": (ffi.get_ptr_address(buf.data), False),
            "strides": strides,
        }

    def close(self) -> None:
        """Release the tensor buffer. Idempotent."""
        if self._c_buffer is not None:
            ffi.model_buffer_free(self._c_buffer)
            self._c_buffer = None

    def __enter__(self) -> ModelBuffer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._c_buffer is None:
            return "ModelBuffer(closed)"
        return f"ModelBuffer(shape={self.shape}, dtype={self.dtype}, layout={self.layout})"


# =============================================================================
# Pipeline
# =============================================================================


def _validate_choice(name: str, value: str, choices: dict[str, int]) -> int:
    """Validate a string parameter against allowed choices."""
    result = choices.get(value)
    if result is None:
        allowed = ", ".join(f"'{k}'" for k in choices)
        raise ValidationError(
            f"Invalid {name}: {value!r}. Must be one of: {allowed}.",
            code="INVALID_ARGUMENT",
        )
    return result


class Pipeline:
    """Image-to-tensor transformation pipeline.

    Defines how images are resized, normalized, and laid out in memory
    for model consumption. Reusable across multiple images.

    Args:
        size: Target (width, height) tuple.
        fit: Resize strategy. "contain" (default) preserves aspect ratio
            with padding; "cover" crops to fill; "stretch" ignores aspect ratio.
        normalize: Normalization mode. "imagenet" (default) applies ImageNet
            mean/std; "zero_one" scales to [0,1]; "none" keeps raw values.
        layout: Tensor memory layout. "nchw" (default, PyTorch style) or
            "nhwc" (TensorFlow/PIL style).
        dtype: Output data type. "float32" (default) or "uint8".
        filter: Resize interpolation. "bilinear" (default), "bicubic",
            or "nearest".
        pad_color: RGB padding color as (r, g, b) tuple. Default (0, 0, 0).

    Example:
        >>> pipeline = Pipeline(size=(384, 384), normalize="imagenet")
        >>> with talu.file.open("photo.jpg") as img:
        ...     buffer = pipeline(img)
        ...     print(buffer.shape, buffer.dtype)
    """

    def __init__(
        self,
        size: tuple[int, int],
        *,
        fit: str = "contain",
        normalize: str = "imagenet",
        layout: str = "nchw",
        dtype: str = "float32",
        filter: str = "bilinear",
        pad_color: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        if (
            not isinstance(size, tuple)
            or len(size) != 2
            or not all(isinstance(s, int) and s > 0 for s in size)
        ):
            raise ValidationError(
                f"size must be a (width, height) tuple of positive integers, got {size!r}.",
                code="INVALID_ARGUMENT",
            )

        self._size = size
        self._fit = fit
        self._normalize = normalize
        self._layout = layout
        self._dtype = dtype
        self._filter = filter
        self._pad_color = pad_color

        # Validate all string params and build the C struct
        fit_int = _validate_choice("fit", fit, _FIT_MODES)
        norm_int = _validate_choice("normalize", normalize, _NORMALIZE)
        layout_int = _validate_choice("layout", layout, _LAYOUTS)
        dtype_int = _validate_choice("dtype", dtype, _DTYPES)
        filter_int = _validate_choice("filter", filter, _FILTERS)

        from .._native import TaluModelInputSpec

        spec = TaluModelInputSpec()
        spec.width = size[0]
        spec.height = size[1]
        spec.dtype = dtype_int
        spec.layout = layout_int
        spec.normalize = norm_int
        spec.fit_mode = fit_int
        spec.filter = filter_int
        spec.pad_r = pad_color[0]
        spec.pad_g = pad_color[1]
        spec.pad_b = pad_color[2]
        self._spec = spec

    def __call__(self, source: Any) -> ModelBuffer:
        """Process an image or bytes into a model-ready tensor buffer.

        Args:
            source: An :class:`Image` object, raw image bytes (``bytes``),
                or a ``memoryview`` of encoded image data.

        Returns:
            A :class:`ModelBuffer` containing the processed tensor.

        Raises:
            ValidationError: If source type is not supported.
            StateError: If source Image is closed.
        """
        # Import here to avoid circular imports at module level
        from ._types import Image

        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
        elif isinstance(source, Image):
            source._check_open()
            if source._source_bytes is not None:
                data = source._source_bytes
            else:
                # PDF-rendered page: encode to PNG first, then process
                data = source.encode("png")
        else:
            raise ValidationError(
                f"Pipeline accepts Image or bytes, got {type(source).__name__}.",
                code="INVALID_ARGUMENT",
            )

        c_buffer = ffi.image_to_model_input(data, self._spec)
        return ModelBuffer(c_buffer)

    def __repr__(self) -> str:
        return (
            f"Pipeline(size={self._size}, fit={self._fit!r}, "
            f"normalize={self._normalize!r}, layout={self._layout!r}, "
            f"dtype={self._dtype!r})"
        )

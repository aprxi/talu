"""Image, Document, Audio, Video, Text, Binary, and the open() entry point."""

from __future__ import annotations

import builtins
import os
from typing import Any, Callable, Iterator

from ..exceptions import StateError, ValidationError
from . import _bindings as ffi


# =============================================================================
# Format Mappings
# =============================================================================

_PIXEL_FORMAT_NAMES = {
    ffi.PIXEL_FORMAT_GRAY8: "gray8",
    ffi.PIXEL_FORMAT_RGB8: "rgb8",
    ffi.PIXEL_FORMAT_RGBA8: "rgba8",
}

_IMAGE_FORMAT_NAMES = {
    ffi.IMAGE_FORMAT_UNKNOWN: "unknown",
    ffi.IMAGE_FORMAT_JPEG: "jpeg",
    ffi.IMAGE_FORMAT_PNG: "png",
    ffi.IMAGE_FORMAT_WEBP: "webp",
}

_ENCODE_FORMAT_MAP = {
    "jpeg": ffi.ENCODE_FORMAT_JPEG,
    "png": ffi.ENCODE_FORMAT_PNG,
}


# =============================================================================
# Image
# =============================================================================


class Image:
    """A raster image — from a file or a rendered document page.

    For file-based images (from :func:`open`), header metadata (width,
    height, format, mime) is available immediately without decoding.
    Pixel data is decoded lazily on first :meth:`encode` call or when
    pixel-level properties are accessed.

    For rendered pages (from :class:`Document` iteration), dimensions
    come from the render output.

    Use as a context manager::

        with talu.file.open("photo.jpg") as img:
            print(img.width, img.height)
            buf = pipeline(img)

    Args:
        c_image: Decoded TaluImage struct (takes ownership), or None
            for file-based images that haven't been decoded yet.
        source_bytes: Original encoded bytes (enables fast Pipeline path).
        mime: MIME type string (e.g. "image/jpeg").
        format: Image file format ("jpeg", "png", "webp").
        width: Image width in pixels.
        height: Image height in pixels.
        orientation: EXIF orientation tag (1-8). 0 if unknown.
    """

    def __init__(
        self,
        c_image: Any = None,
        *,
        source_bytes: bytes | None = None,
        mime: str | None = None,
        format: str | None = None,
        width: int | None = None,
        height: int | None = None,
        orientation: int = 0,
    ) -> None:
        self._c_image = c_image
        self._source_bytes = source_bytes
        self._mime = mime
        self._format = format
        self._orientation = orientation
        if width is not None:
            self._width = width
            self._height = height or 0
        elif c_image is not None:
            self._width = c_image.width
            self._height = c_image.height
        else:
            self._width = 0
            self._height = 0

    def _check_open(self) -> None:
        if self._c_image is None and self._source_bytes is None:
            raise StateError("Image is closed.", code="STATE_ERROR")

    def _ensure_decoded(self) -> None:
        """Lazily decode source bytes into pixel data."""
        if self._c_image is None:
            if self._source_bytes is None:
                raise StateError("Image is closed.", code="STATE_ERROR")
            self._c_image = ffi.image_decode(self._source_bytes)

    @property
    def width(self) -> int:
        """Image width in pixels."""
        self._check_open()
        return self._width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        self._check_open()
        return self._height

    @property
    def mime(self) -> str | None:
        """MIME type string (e.g. "image/jpeg"), or None for rendered pages."""
        self._check_open()
        return self._mime

    @property
    def format(self) -> str | None:
        """Image file format ("jpeg", "png", "webp"), or None for rendered pages."""
        self._check_open()
        return self._format

    @property
    def orientation(self) -> int:
        """EXIF orientation tag (1-8). 0 if unknown or not applicable."""
        self._check_open()
        return self._orientation

    @property
    def pixel_format(self) -> str:
        """Pixel format name ("gray8", "rgb8", or "rgba8").

        Triggers decode if pixel data has not been loaded yet.
        """
        self._ensure_decoded()
        return _PIXEL_FORMAT_NAMES.get(self._c_image.format, "unknown")

    @property
    def stride(self) -> int:
        """Bytes per row (may exceed width * bytes_per_pixel due to alignment).

        Triggers decode if pixel data has not been loaded yet.
        """
        self._ensure_decoded()
        return self._c_image.stride

    @property
    def channels(self) -> int:
        """Number of channels (1 for gray8, 3 for rgb8, 4 for rgba8).

        Triggers decode if pixel data has not been loaded yet.
        """
        self._ensure_decoded()
        fmt = self._c_image.format
        if fmt == ffi.PIXEL_FORMAT_GRAY8:
            return 1
        if fmt == ffi.PIXEL_FORMAT_RGB8:
            return 3
        return 4  # rgba8

    def encode(self, fmt: str = "jpeg", quality: int = 85) -> bytes:
        """Encode the image to JPEG or PNG bytes.

        Triggers decode if pixel data has not been loaded yet.

        Args:
            fmt: Output format, "jpeg" or "png".
            quality: JPEG quality (1-100). Ignored for PNG.

        Returns:
            Encoded image bytes.

        Raises:
            StateError: If the image is closed.
            ValidationError: If the format is invalid.
        """
        self._ensure_decoded()
        encode_fmt = _ENCODE_FORMAT_MAP.get(fmt)
        if encode_fmt is None:
            raise ValidationError(
                f"Unsupported encode format: {fmt!r}. Use 'jpeg' or 'png'.",
                code="INVALID_ARGUMENT",
            )
        from .._native import TaluImageEncodeOptions

        opts = TaluImageEncodeOptions()
        opts.format = encode_fmt
        opts.jpeg_quality = quality
        return ffi.image_encode(self._c_image, opts)

    def close(self) -> None:
        """Release resources. Idempotent."""
        if self._c_image is not None:
            ffi.image_free(self._c_image)
            self._c_image = None
        self._source_bytes = None

    def __enter__(self) -> Image:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._c_image is None and self._source_bytes is None:
            return "Image(closed)"
        parts = [f"{self._width}x{self._height}"]
        if self._format:
            parts.append(self._format)
        if self._mime:
            parts.append(repr(self._mime))
        return f"Image({', '.join(parts)})"


# =============================================================================
# Document
# =============================================================================


class Document:
    """A rendered document (PDF, and future: DOCX, PPTX).

    Has no intrinsic pixel dimensions — those depend on the DPI chosen at
    render time. Iterating lazily renders each page to an :class:`Image`.

    Use as a context manager::

        with talu.file.open("report.pdf") as doc:
            print(f"{len(doc)} pages")
            for page_image in doc:
                buf = pipeline(page_image)

    Args:
        data: Raw file bytes.
        mime: MIME type string.
        page_count_fn: Callable that returns the page count given the bytes.
        render_page_fn: Callable that renders a page given (bytes, page_index).
    """

    def __init__(
        self,
        data: bytes,
        mime: str,
        page_count_fn: Callable[[bytes], int],
        render_page_fn: Callable[[bytes, int], Any],
    ) -> None:
        self._data = data
        self._mime = mime
        self._count_fn = page_count_fn
        self._render_fn = render_page_fn
        self._page_count: int | None = None

    @property
    def mime(self) -> str:
        """MIME type string (e.g. "application/pdf")."""
        return self._mime

    def __len__(self) -> int:
        self._check_open()
        if self._page_count is None:
            self._page_count = self._count_fn(self._data)
        return self._page_count

    def __iter__(self) -> Iterator[Image]:
        self._check_open()
        count = len(self)
        for i in range(count):
            if self._data is None:
                return
            c_image = self._render_fn(self._data, i)
            yield Image(c_image)

    def _check_open(self) -> None:
        if self._data is None:
            raise StateError("Document is closed.", code="STATE_ERROR")

    def close(self) -> None:
        """Release resources. Idempotent."""
        self._data = None

    def __enter__(self) -> Document:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._data is None:
            return "Document(closed)"
        pages = self._page_count if self._page_count is not None else "?"
        return f"Document({pages} pages, {self._mime!r})"


# =============================================================================
# Audio
# =============================================================================


class Audio:
    """An audio file (MP3, WAV, OGG, FLAC, etc.).

    Use as a context manager::

        with talu.file.open("song.mp3") as audio:
            print(audio.mime)

    Args:
        data: Raw file bytes.
        mime: MIME type string (e.g. "audio/mpeg").
    """

    def __init__(self, data: bytes, mime: str) -> None:
        self._data: bytes | None = data
        self._mime = mime

    def _check_open(self) -> None:
        if self._data is None:
            raise StateError("Audio is closed.", code="STATE_ERROR")

    @property
    def mime(self) -> str:
        """MIME type string (e.g. "audio/mpeg")."""
        self._check_open()
        return self._mime

    @property
    def data(self) -> bytes:
        """Raw file bytes."""
        self._check_open()
        return self._data

    def close(self) -> None:
        """Release resources. Idempotent."""
        self._data = None

    def __enter__(self) -> Audio:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._data is None:
            return "Audio(closed)"
        return f"Audio({self._mime!r})"


# =============================================================================
# Video
# =============================================================================


class Video:
    """A video file (MP4, WebM, AVI, etc.).

    Use as a context manager::

        with talu.file.open("clip.mp4") as video:
            print(video.mime)

    Args:
        data: Raw file bytes.
        mime: MIME type string (e.g. "video/mp4").
    """

    def __init__(self, data: bytes, mime: str) -> None:
        self._data: bytes | None = data
        self._mime = mime

    def _check_open(self) -> None:
        if self._data is None:
            raise StateError("Video is closed.", code="STATE_ERROR")

    @property
    def mime(self) -> str:
        """MIME type string (e.g. "video/mp4")."""
        self._check_open()
        return self._mime

    @property
    def data(self) -> bytes:
        """Raw file bytes."""
        self._check_open()
        return self._data

    def close(self) -> None:
        """Release resources. Idempotent."""
        self._data = None

    def __enter__(self) -> Video:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._data is None:
            return "Video(closed)"
        return f"Video({self._mime!r})"


# =============================================================================
# Text
# =============================================================================


class Text:
    """A text file (plain text, JSON, XML, HTML, YAML, CSV, etc.).

    Use as a context manager::

        with talu.file.open("config.json") as txt:
            print(txt.mime)
            print(txt.text)

    Args:
        data: Raw file bytes.
        mime: MIME type string (e.g. "text/plain", "application/json").
    """

    def __init__(self, data: bytes, mime: str) -> None:
        self._data: bytes | None = data
        self._mime = mime

    def _check_open(self) -> None:
        if self._data is None:
            raise StateError("Text is closed.", code="STATE_ERROR")

    @property
    def mime(self) -> str:
        """MIME type string (e.g. "text/plain", "application/json")."""
        self._check_open()
        return self._mime

    @property
    def data(self) -> bytes:
        """Raw file bytes."""
        self._check_open()
        return self._data

    @property
    def text(self) -> str:
        """Decoded text content (UTF-8, lossy)."""
        self._check_open()
        return self._data.decode("utf-8", errors="replace")

    def close(self) -> None:
        """Release resources. Idempotent."""
        self._data = None

    def __enter__(self) -> Text:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._data is None:
            return "Text(closed)"
        return f"Text({self._mime!r})"


# =============================================================================
# Binary
# =============================================================================


class Binary:
    """An unrecognized or non-text file (executables, archives, etc.).

    Use as a context manager::

        with talu.file.open("archive.zip") as blob:
            print(blob.mime)

    Args:
        data: Raw file bytes.
        mime: MIME type string (e.g. "application/octet-stream").
    """

    def __init__(self, data: bytes, mime: str) -> None:
        self._data: bytes | None = data
        self._mime = mime

    def _check_open(self) -> None:
        if self._data is None:
            raise StateError("Binary is closed.", code="STATE_ERROR")

    @property
    def mime(self) -> str:
        """MIME type string (e.g. "application/octet-stream")."""
        self._check_open()
        return self._mime

    @property
    def data(self) -> bytes:
        """Raw file bytes."""
        self._check_open()
        return self._data

    def close(self) -> None:
        """Release resources. Idempotent."""
        self._data = None

    def __enter__(self) -> Binary:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        if self._data is None:
            return "Binary(closed)"
        return f"Binary({self._mime!r})"


# =============================================================================
# Entry Points
# =============================================================================


def _read_source(source: str | os.PathLike | bytes) -> bytes:
    """Read file bytes from any supported source type."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with builtins.open(path, "rb") as f:
            return f.read()
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        data = source.read()
        if not isinstance(data, bytes):
            raise ValidationError(
                "File-like object must return bytes from read().",
                code="INVALID_ARGUMENT",
            )
        return data
    raise ValidationError(
        f"Unsupported source type: {type(source).__name__}. "
        "Pass a file path, bytes, or file-like object.",
        code="INVALID_ARGUMENT",
    )


def open(
    source: str | os.PathLike | bytes,
) -> Image | Document | Audio | Video | Text | Binary:
    """Open a file and return a typed wrapper.

    Inspects the file header and returns the appropriate class:

    - :class:`Image` for raster images (JPEG, PNG, WebP).
    - :class:`Document` for rendered formats (PDF).
    - :class:`Audio` for audio files (MP3, WAV, OGG, etc.).
    - :class:`Video` for video files (MP4, WebM, AVI, etc.).
    - :class:`Text` for human-readable text (plain, JSON, XML, etc.).
    - :class:`Binary` for unrecognized files.

    Args:
        source: File path (str or PathLike), raw bytes, or a file-like
            object with a ``read()`` method.

    Returns:
        A typed file wrapper.

    Raises:
        FileNotFoundError: If a path is given and the file does not exist.

    Example:
        >>> with talu.file.open("photo.jpg") as img:
        ...     print(img.width, img.height)
        ...     buf = pipeline(img)
    """
    data = _read_source(source)
    info_dict = ffi.file_inspect(data)
    mime = info_dict["mime"]
    kind = info_dict["kind"]

    if kind == ffi.FILE_KIND_IMAGE:
        return Image(
            source_bytes=data,
            mime=mime,
            format=_IMAGE_FORMAT_NAMES.get(info_dict["image_format"], "unknown"),
            width=info_dict["width"],
            height=info_dict["height"],
            orientation=info_dict["orientation"],
        )
    if kind == ffi.FILE_KIND_DOCUMENT:
        return Document(
            data,
            mime=mime,
            page_count_fn=ffi.pdf_page_count,
            render_page_fn=ffi.pdf_render_page,
        )
    if kind == ffi.FILE_KIND_AUDIO:
        return Audio(data, mime=mime)
    if kind == ffi.FILE_KIND_VIDEO:
        return Video(data, mime=mime)
    if kind == ffi.FILE_KIND_TEXT:
        return Text(data, mime=mime)
    return Binary(data, mime=mime)

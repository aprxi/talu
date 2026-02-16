"""
FFI bindings for file/image module.

Justification: Provides ctypes wrappers for file inspection, image
decode/convert/encode, PDF page rendering, and image-to-model-input
tensor conversion. These operations require direct FFI interaction
with the Zig core for performance-critical image processing.
"""

import ctypes
from typing import Any

from .._bindings import check, get_lib, read_buffer
from .._native import (
    TaluFileInfo,
    TaluFileTransformOptions,
    TaluImage,
    TaluImageConvertOptions,
    TaluImageDecodeOptions,
    TaluImageEncodeOptions,
    TaluImageInfo,
    TaluModelBuffer,
    TaluModelInputSpec,
)

# =============================================================================
# Constants
# =============================================================================

# Image formats (TaluImageInfo.format) — raster formats only.
IMAGE_FORMAT_UNKNOWN = 0
IMAGE_FORMAT_JPEG = 1
IMAGE_FORMAT_PNG = 2
IMAGE_FORMAT_WEBP = 3

# File kinds (TaluFileInfo.kind):
#   0 = binary — unrecognized or non-text file.
#   1 = image — raster image (JPEG, PNG, WebP). Has intrinsic dimensions.
#   2 = document — rendered format (PDF). No intrinsic dimensions.
#   3 = audio — audio file (MP3, WAV, OGG, FLAC, etc.).
#   4 = video — video file (MP4, WebM, AVI, etc.).
#   5 = text — human-readable text (plain, JSON, XML, HTML, YAML, etc.).
FILE_KIND_BINARY = 0
FILE_KIND_IMAGE = 1
FILE_KIND_DOCUMENT = 2
FILE_KIND_AUDIO = 3
FILE_KIND_VIDEO = 4
FILE_KIND_TEXT = 5

# Pixel formats
PIXEL_FORMAT_GRAY8 = 0
PIXEL_FORMAT_RGB8 = 1
PIXEL_FORMAT_RGBA8 = 2

# Encode formats
ENCODE_FORMAT_JPEG = 0
ENCODE_FORMAT_PNG = 1

# =============================================================================
# File Inspection
# =============================================================================


def file_inspect(data: bytes) -> dict[str, Any]:
    """Inspect file bytes and return metadata.

    Args:
        data: Raw file bytes to inspect.

    Returns:
        Dictionary with keys: kind, mime, description, image_format,
        width, height, orientation.
    """
    lib = get_lib()
    info = TaluFileInfo()
    img = TaluImageInfo()
    rc = lib.talu_file_inspect(data, len(data), ctypes.byref(info), ctypes.byref(img))
    check(rc, {"operation": "file_inspect"})

    mime = ""
    if info.mime_ptr and info.mime_len > 0:
        mime = read_buffer(info.mime_ptr, info.mime_len).decode("utf-8", errors="replace")

    description = ""
    if info.description_ptr and info.description_len > 0:
        description = read_buffer(info.description_ptr, info.description_len).decode(
            "utf-8", errors="replace"
        )

    result = {
        "kind": info.kind,
        "mime": mime,
        "description": description,
        "image_format": img.format,
        "width": img.width,
        "height": img.height,
        "orientation": img.orientation,
    }

    lib.talu_file_info_free(ctypes.byref(info))
    return result


# =============================================================================
# Image Decode / Convert / Encode
# =============================================================================


def image_decode(data: bytes, opts: "TaluImageDecodeOptions | None" = None) -> TaluImage:
    """Decode image bytes into pixel data.

    The caller is responsible for calling image_free() on the returned
    TaluImage when done.

    Args:
        data: Encoded image bytes (JPEG, PNG, WebP).
        opts: Decode options, or None for defaults.

    Returns:
        TaluImage struct with decoded pixel data owned by Zig.
    """
    lib = get_lib()
    out = TaluImage()
    opts_ptr = ctypes.byref(opts) if opts is not None else None
    rc = lib.talu_image_decode(data, len(data), opts_ptr, ctypes.byref(out))
    check(rc, {"operation": "image_decode"})
    return out


def image_convert(src: TaluImage, opts: TaluImageConvertOptions) -> TaluImage:
    """Convert an image (format, resize, alpha).

    The caller is responsible for calling image_free() on the returned
    TaluImage when done. The source image is NOT freed.

    Args:
        src: Source image.
        opts: Conversion options.

    Returns:
        TaluImage struct with converted pixel data owned by Zig.
    """
    lib = get_lib()
    out = TaluImage()
    rc = lib.talu_image_convert(ctypes.byref(src), ctypes.byref(opts), ctypes.byref(out))
    check(rc, {"operation": "image_convert"})
    return out


def image_encode(src: TaluImage, opts: TaluImageEncodeOptions) -> bytes:
    """Encode an image to JPEG or PNG bytes.

    Returns a copy of the encoded data as Python bytes.
    The C buffer is freed before returning.

    Args:
        src: Source image with pixel data.
        opts: Encode options (format, quality).

    Returns:
        Encoded image bytes.
    """
    lib = get_lib()
    out_ptr = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    rc = lib.talu_image_encode(
        ctypes.byref(src), ctypes.byref(opts), ctypes.byref(out_ptr), ctypes.byref(out_len)
    )
    check(rc, {"operation": "image_encode"})

    result = ctypes.string_at(out_ptr, out_len.value)
    lib.talu_image_encode_free(out_ptr, out_len)
    return result


def image_free(img: TaluImage) -> None:
    """Free a decoded image's pixel data.

    Args:
        img: The TaluImage struct to free.
    """
    get_lib().talu_image_free(ctypes.byref(img))


# =============================================================================
# Model Input
# =============================================================================


def image_to_model_input(data: bytes, spec: TaluModelInputSpec) -> TaluModelBuffer:
    """Decode image and convert to model-ready tensor buffer.

    The caller is responsible for calling model_buffer_free() on the
    returned TaluModelBuffer when done.

    Args:
        data: Encoded image bytes (JPEG, PNG, WebP).
        spec: Model input specification (size, layout, dtype, normalization).

    Returns:
        TaluModelBuffer struct with tensor data owned by Zig.
    """
    lib = get_lib()
    out = TaluModelBuffer()
    rc = lib.talu_image_to_model_input(data, len(data), ctypes.byref(spec), ctypes.byref(out))
    check(rc, {"operation": "image_to_model_input"})
    return out


def model_buffer_free(buf: TaluModelBuffer) -> None:
    """Free a model buffer's tensor data.

    Args:
        buf: The TaluModelBuffer struct to free.
    """
    get_lib().talu_model_buffer_free(ctypes.byref(buf))


# =============================================================================
# File Transform
# =============================================================================


def file_transform(
    data: bytes, opts: "TaluFileTransformOptions | None" = None
) -> tuple[bytes, dict[str, Any]]:
    """Transform a file (decode, resize, re-encode) in one pass.

    Args:
        data: Input file bytes.
        opts: Transform options, or None for defaults.

    Returns:
        Tuple of (encoded_bytes, info_dict).
    """
    lib = get_lib()
    out_ptr = ctypes.c_void_p()
    out_len = ctypes.c_size_t()
    out_image = TaluImageInfo()
    opts_ptr = ctypes.byref(opts) if opts is not None else None

    rc = lib.talu_file_transform(
        data,
        len(data),
        opts_ptr,
        ctypes.byref(out_ptr),
        ctypes.byref(out_len),
        ctypes.byref(out_image),
    )
    check(rc, {"operation": "file_transform"})

    result_bytes = ctypes.string_at(out_ptr, out_len.value)
    lib.talu_file_bytes_free(out_ptr, out_len)

    info_dict = {
        "image_format": out_image.format,
        "width": out_image.width,
        "height": out_image.height,
    }
    return result_bytes, info_dict


# =============================================================================
# PDF
# =============================================================================


def pdf_page_count(data: bytes) -> int:
    """Return the number of pages in a PDF.

    Args:
        data: PDF file bytes.

    Returns:
        Page count.
    """
    lib = get_lib()
    count = ctypes.c_uint32()
    rc = lib.talu_pdf_page_count(data, len(data), ctypes.byref(count))
    check(rc, {"operation": "pdf_page_count"})
    return count.value


def pdf_render_page(data: bytes, page_index: int, dpi: int = 0) -> TaluImage:
    """Render a PDF page to an image.

    The caller is responsible for calling image_free() on the returned
    TaluImage when done.

    Args:
        data: PDF file bytes.
        page_index: Zero-based page index.
        dpi: Render DPI (0 = default 150).

    Returns:
        TaluImage struct with rendered pixel data owned by Zig.
    """
    lib = get_lib()
    out = TaluImage()
    rc = lib.talu_pdf_render_page(data, len(data), page_index, dpi, ctypes.byref(out))
    check(rc, {"operation": "pdf_render_page", "page_index": page_index})
    return out


# =============================================================================
# Pointer Helpers
# =============================================================================


def get_ptr_address(ptr: Any) -> int:
    """Get the memory address of pointer contents for __array_interface__.

    Args:
        ptr: A ctypes POINTER whose contents address we need.

    Returns:
        Integer memory address.
    """
    return ctypes.addressof(ptr.contents)


def read_buffer_raw(ptr: Any, length: int) -> bytes:
    """Copy native buffer contents to Python bytes.

    Args:
        ptr: A ctypes POINTER to the data.
        length: Number of bytes to copy.

    Returns:
        Copied bytes.
    """
    return ctypes.string_at(ptr, length)

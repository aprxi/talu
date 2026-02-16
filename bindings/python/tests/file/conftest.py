"""Fixtures for talu.file tests."""

import ctypes
from pathlib import Path

import pytest

# Repository root (where test.jpeg lives)
REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def jpeg_path() -> Path:
    """Path to the test JPEG image (417x417)."""
    path = REPO_ROOT / "test.jpeg"
    if not path.exists():
        pytest.skip("test.jpeg not found at repo root")
    return path


@pytest.fixture
def jpeg_bytes(jpeg_path: Path) -> bytes:
    """Raw bytes of the test JPEG image."""
    return jpeg_path.read_bytes()


@pytest.fixture
def small_jpeg_bytes() -> bytes:
    """A minimal valid JPEG for fast tests.

    Creates a tiny 2x2 red JPEG via the Zig core.
    """
    import talu.file._bindings as ffi
    from talu._native import TaluImage, TaluImageEncodeOptions

    # Build a 2x2 RGB8 image (red pixels)
    pixel_data = bytes([255, 0, 0] * 4)
    buf = (ctypes.c_uint8 * len(pixel_data))(*pixel_data)

    img = TaluImage()
    img.data = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
    img.len = len(pixel_data)
    img.width = 2
    img.height = 2
    img.stride = 6  # 2 pixels * 3 bytes
    img.format = 1  # rgb8

    opts = TaluImageEncodeOptions()
    opts.format = 0  # jpeg
    opts.jpeg_quality = 85

    encoded = ffi.image_encode(img, opts)
    return encoded

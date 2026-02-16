"""Tests for Image decode, encode, and lifecycle."""

import gc

import pytest

from talu.exceptions import StateError
from talu.file import Image, open


class TestImageDecode:
    """Image decoding from file bytes."""

    def test_decode_jpeg_dimensions(self, jpeg_bytes: bytes) -> None:
        """Image has expected dimensions from header."""
        with open(jpeg_bytes) as img:
            assert img.width == 417
            assert img.height == 417

    def test_decode_jpeg_pixel_format(self, jpeg_bytes: bytes) -> None:
        """Decoded JPEG has valid pixel format."""
        with open(jpeg_bytes) as img:
            assert img.pixel_format in ("rgb8", "rgba8", "gray8")
            assert img.channels in (1, 3, 4)

    def test_decode_jpeg_stride(self, jpeg_bytes: bytes) -> None:
        """Decoded JPEG has valid stride."""
        with open(jpeg_bytes) as img:
            assert img.stride >= img.width * img.channels

    def test_decode_small_jpeg(self, small_jpeg_bytes: bytes) -> None:
        """Decode a tiny synthetic JPEG."""
        with open(small_jpeg_bytes) as img:
            assert img.width == 2
            assert img.height == 2


class TestImageEncode:
    """Image encoding to bytes."""

    def test_encode_to_jpeg(self, jpeg_bytes: bytes) -> None:
        """Encode image to JPEG."""
        with open(jpeg_bytes) as img:
            encoded = img.encode("jpeg", quality=90)
            assert isinstance(encoded, bytes)
            assert len(encoded) > 0
            # JPEG magic bytes
            assert encoded[:2] == b"\xff\xd8"

    def test_encode_to_png(self, jpeg_bytes: bytes) -> None:
        """Encode image to PNG."""
        with open(jpeg_bytes) as img:
            encoded = img.encode("png")
            assert isinstance(encoded, bytes)
            # PNG magic bytes
            assert encoded[:4] == b"\x89PNG"

    def test_encode_invalid_format(self, jpeg_bytes: bytes) -> None:
        """Encoding with unsupported format raises ValidationError."""
        with open(jpeg_bytes) as img:
            with pytest.raises(Exception, match="Unsupported encode format"):
                img.encode("bmp")


class TestImageLifecycle:
    """Image resource lifecycle (close, context manager, __del__)."""

    def test_close_idempotent(self, jpeg_bytes: bytes) -> None:
        """close() can be called multiple times safely."""
        img = open(jpeg_bytes)
        img.close()
        img.close()  # Second call should not raise

    def test_use_after_close_raises(self, jpeg_bytes: bytes) -> None:
        """Accessing properties after close raises StateError."""
        img = open(jpeg_bytes)
        img.close()
        with pytest.raises(StateError):
            _ = img.width

    def test_context_manager(self, jpeg_bytes: bytes) -> None:
        """Image works as context manager."""
        with open(jpeg_bytes) as img:
            assert img.width > 0
        with pytest.raises(StateError):
            _ = img.width

    def test_repr_open_and_closed(self, jpeg_bytes: bytes) -> None:
        """Image repr changes after close."""
        with open(jpeg_bytes) as img:
            r = repr(img)
            assert "417" in r
        assert "closed" in repr(img)

    def test_del_cleanup(self, small_jpeg_bytes: bytes) -> None:
        """__del__ cleans up without error."""
        img = open(small_jpeg_bytes)
        del img
        gc.collect()

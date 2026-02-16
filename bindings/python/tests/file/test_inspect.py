"""Tests for low-level file inspection via talu.file._bindings."""

import pytest

from talu.file._bindings import file_inspect


class TestFileInspect:
    """Low-level file_inspect() tests."""

    def test_inspect_jpeg(self, jpeg_bytes: bytes) -> None:
        """Inspect JPEG bytes returns correct metadata."""
        info = file_inspect(jpeg_bytes)
        assert info["kind"] == 1  # image
        assert info["mime"] == "image/jpeg"
        assert info["image_format"] == 1  # jpeg
        assert info["width"] == 417
        assert info["height"] == 417

    def test_inspect_small_jpeg(self, small_jpeg_bytes: bytes) -> None:
        """Inspect a small synthetic JPEG."""
        info = file_inspect(small_jpeg_bytes)
        assert info["kind"] == 1
        assert "jpeg" in info["mime"]
        assert info["image_format"] == 1

    def test_inspect_unknown_bytes(self) -> None:
        """Inspect non-image bytes returns kind=5 (text) for text content."""
        info = file_inspect(b"this is plain text content, not an image")
        assert info["kind"] == 5  # text
        assert info["image_format"] == 0

    def test_inspect_binary_bytes(self) -> None:
        """Inspect binary bytes returns kind=0 (binary)."""
        info = file_inspect(bytes(range(128)))
        assert info["kind"] == 0  # binary
        assert info["image_format"] == 0

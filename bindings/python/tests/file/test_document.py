"""Tests for open(), Image, Document, Audio, Video, Text, Binary."""

from pathlib import Path

import pytest

from talu.exceptions import StateError, ValidationError
from talu.file import (
    Audio,
    Binary,
    Document,
    Image,
    ModelBuffer,
    Pipeline,
    Text,
    Video,
    open,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).resolve().parents[4] / "core" / "tests" / "image" / "corpus"


@pytest.fixture
def pdf_bytes() -> bytes:
    """Raw bytes of the 1-page test PDF."""
    path = CORPUS_DIR / "1x1_page.pdf"
    if not path.exists():
        pytest.skip("1x1_page.pdf not found in core corpus")
    return path.read_bytes()


# ---------------------------------------------------------------------------
# open()
# ---------------------------------------------------------------------------


class TestOpenFunction:
    """talu.file.open() dispatch and input types."""

    def test_open_path(self, jpeg_path) -> None:
        """open() with a file path returns Image."""
        with open(jpeg_path) as img:
            assert isinstance(img, Image)

    def test_open_bytes(self, jpeg_bytes: bytes) -> None:
        """open() with bytes returns Image."""
        with open(jpeg_bytes) as img:
            assert isinstance(img, Image)

    def test_open_string_path(self, jpeg_path) -> None:
        """open() with a string path works."""
        with open(str(jpeg_path)) as img:
            assert isinstance(img, Image)

    def test_open_nonexistent_file(self, tmp_path) -> None:
        """open() with nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            open(tmp_path / "nonexistent.jpg")

    def test_open_text_returns_text(self) -> None:
        """open() with text bytes returns Text."""
        with open(b"this is plain text content") as txt:
            assert isinstance(txt, Text)

    def test_open_binary_returns_binary(self) -> None:
        """open() with binary bytes returns Binary."""
        with open(bytes(range(128))) as blob:
            assert isinstance(blob, Binary)

    def test_open_invalid_source_type(self) -> None:
        """open() with invalid source type raises ValidationError."""
        with pytest.raises(ValidationError, match="Unsupported source type"):
            open(12345)

    def test_open_pdf_returns_document(self, pdf_bytes: bytes) -> None:
        """open() with PDF bytes returns Document."""
        with open(pdf_bytes) as doc:
            assert isinstance(doc, Document)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class TestImage:
    """Image properties, encoding, and lifecycle."""

    def test_properties(self, jpeg_bytes: bytes) -> None:
        """Image from open() exposes header metadata."""
        with open(jpeg_bytes) as img:
            assert isinstance(img, Image)
            assert img.mime == "image/jpeg"
            assert img.format == "jpeg"
            assert img.width == 417
            assert img.height == 417
            assert img.orientation >= 0

    def test_close_idempotent(self, jpeg_bytes: bytes) -> None:
        """Image close is idempotent."""
        img = open(jpeg_bytes)
        img.close()
        img.close()

    def test_use_after_close(self, jpeg_bytes: bytes) -> None:
        """Accessing closed image raises StateError."""
        img = open(jpeg_bytes)
        img.close()
        with pytest.raises(StateError):
            _ = img.width

    def test_context_manager(self, jpeg_bytes: bytes) -> None:
        """Image works as context manager."""
        with open(jpeg_bytes) as img:
            assert img.width == 417
        with pytest.raises(StateError):
            _ = img.width

    def test_repr(self, jpeg_bytes: bytes) -> None:
        """Image repr includes format and dimensions."""
        with open(jpeg_bytes) as img:
            r = repr(img)
            assert "Image" in r
            assert "417" in r
        assert "closed" in repr(img)


# ---------------------------------------------------------------------------
# Document (PDF)
# ---------------------------------------------------------------------------


class TestDocument:
    """Document (rendered format) iteration and lifecycle."""

    def test_page_count(self, pdf_bytes: bytes) -> None:
        """PDF Document reports correct page count."""
        with open(pdf_bytes) as doc:
            assert isinstance(doc, Document)
            assert len(doc) == 1

    def test_iteration_yields_images(self, pdf_bytes: bytes) -> None:
        """Iterating a PDF yields Image objects."""
        with open(pdf_bytes) as doc:
            images = list(doc)
            assert len(images) == 1
            assert isinstance(images[0], Image)
            assert images[0].width > 0
            assert images[0].height > 0

    def test_mime(self, pdf_bytes: bytes) -> None:
        """PDF Document has correct MIME type."""
        with open(pdf_bytes) as doc:
            assert "pdf" in doc.mime

    def test_close_idempotent(self, pdf_bytes: bytes) -> None:
        """Document close is idempotent."""
        doc = open(pdf_bytes)
        doc.close()
        doc.close()

    def test_use_after_close(self, pdf_bytes: bytes) -> None:
        """Iterating closed document raises StateError."""
        doc = open(pdf_bytes)
        doc.close()
        with pytest.raises(StateError):
            list(doc)

    def test_context_manager(self, pdf_bytes: bytes) -> None:
        """Document works as context manager."""
        with open(pdf_bytes) as doc:
            assert len(doc) >= 1
        with pytest.raises(StateError):
            list(doc)

    def test_repr(self, pdf_bytes: bytes) -> None:
        """Document repr shows page count."""
        with open(pdf_bytes) as doc:
            r = repr(doc)
            assert "Document" in r
        assert "closed" in repr(doc)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TestText:
    """Text file lifecycle and properties."""

    def test_properties(self) -> None:
        """Text from open() exposes mime and text content."""
        with open(b"hello world") as txt:
            assert isinstance(txt, Text)
            assert txt.mime == "text/plain"
            assert txt.text == "hello world"
            assert txt.data == b"hello world"

    def test_close_idempotent(self) -> None:
        """Text close is idempotent."""
        txt = open(b"hello")
        txt.close()
        txt.close()

    def test_use_after_close(self) -> None:
        """Accessing closed Text raises StateError."""
        txt = open(b"hello")
        txt.close()
        with pytest.raises(StateError):
            _ = txt.text

    def test_context_manager(self) -> None:
        """Text works as context manager."""
        with open(b"hello") as txt:
            assert txt.text == "hello"
        with pytest.raises(StateError):
            _ = txt.text

    def test_repr(self) -> None:
        """Text repr shows MIME type."""
        with open(b"hello") as txt:
            r = repr(txt)
            assert "Text" in r
        assert "closed" in repr(txt)


# ---------------------------------------------------------------------------
# Binary
# ---------------------------------------------------------------------------


class TestBinary:
    """Binary file lifecycle and properties."""

    def test_properties(self) -> None:
        """Binary from open() exposes mime and data."""
        data = bytes(range(128))
        with open(data) as blob:
            assert isinstance(blob, Binary)
            assert blob.data == data

    def test_close_idempotent(self) -> None:
        """Binary close is idempotent."""
        blob = open(bytes(range(128)))
        blob.close()
        blob.close()

    def test_use_after_close(self) -> None:
        """Accessing closed Binary raises StateError."""
        blob = open(bytes(range(128)))
        blob.close()
        with pytest.raises(StateError):
            _ = blob.data

    def test_context_manager(self) -> None:
        """Binary works as context manager."""
        data = bytes(range(128))
        with open(data) as blob:
            assert blob.data == data
        with pytest.raises(StateError):
            _ = blob.data

    def test_repr(self) -> None:
        """Binary repr shows MIME type."""
        with open(bytes(range(128))) as blob:
            r = repr(blob)
            assert "Binary" in r
        assert "closed" in repr(blob)


# ---------------------------------------------------------------------------
# Audio / Video (class contract tests — no real audio/video files needed)
# ---------------------------------------------------------------------------


class TestAudioClass:
    """Audio class lifecycle (constructed directly, no real audio files)."""

    def test_lifecycle(self) -> None:
        """Audio supports context manager and close."""
        audio = Audio(b"fake-audio-data", mime="audio/mpeg")
        assert audio.mime == "audio/mpeg"
        assert audio.data == b"fake-audio-data"
        audio.close()
        with pytest.raises(StateError):
            _ = audio.mime

    def test_repr(self) -> None:
        """Audio repr shows MIME type."""
        audio = Audio(b"data", mime="audio/wav")
        assert "Audio" in repr(audio)
        audio.close()
        assert "closed" in repr(audio)


class TestVideoClass:
    """Video class lifecycle (constructed directly, no real video files)."""

    def test_lifecycle(self) -> None:
        """Video supports context manager and close."""
        video = Video(b"fake-video-data", mime="video/mp4")
        assert video.mime == "video/mp4"
        assert video.data == b"fake-video-data"
        video.close()
        with pytest.raises(StateError):
            _ = video.mime

    def test_repr(self) -> None:
        """Video repr shows MIME type."""
        video = Video(b"data", mime="video/webm")
        assert "Video" in repr(video)
        video.close()
        assert "closed" in repr(video)


# ---------------------------------------------------------------------------
# Full roundtrip
# ---------------------------------------------------------------------------


class TestFullRoundtrip:
    """End-to-end: open -> pipeline -> ModelBuffer."""

    def test_jpeg_to_tensor(self, jpeg_bytes: bytes) -> None:
        """Full roundtrip from JPEG bytes to model tensor."""
        pipeline = Pipeline(size=(224, 224))
        with open(jpeg_bytes) as img:
            with pipeline(img) as buf:
                assert isinstance(buf, ModelBuffer)
                assert buf.shape == (3, 224, 224)
                assert buf.dtype == "float32"

    def test_jpeg_path_to_tensor(self, jpeg_path) -> None:
        """Full roundtrip from JPEG path to model tensor."""
        pipeline = Pipeline(size=(112, 112), layout="nhwc", dtype="uint8", normalize="none")
        with open(jpeg_path) as img:
            with pipeline(img) as buf:
                assert buf.shape == (112, 112, 3)
                assert buf.dtype == "uint8"

    def test_multiple_pipelines_same_image(self, jpeg_bytes: bytes) -> None:
        """Multiple pipelines can process the same image."""
        p1 = Pipeline(size=(224, 224), layout="nchw")
        p2 = Pipeline(size=(384, 384), layout="nhwc")
        with open(jpeg_bytes) as img:
            with p1(img) as b1, p2(img) as b2:
                assert b1.shape == (3, 224, 224)
                assert b2.shape == (384, 384, 3)

    def test_pdf_to_tensor(self, pdf_bytes: bytes) -> None:
        """Full roundtrip from PDF to model tensor."""
        pipeline = Pipeline(size=(64, 64))
        with open(pdf_bytes) as doc:
            for image in doc:
                with pipeline(image) as buf:
                    assert buf.shape == (3, 64, 64)
                    assert buf.dtype == "float32"

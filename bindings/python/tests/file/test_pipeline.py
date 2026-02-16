"""Tests for Pipeline and ModelBuffer."""

import gc

import pytest

from talu.exceptions import StateError, ValidationError
from talu.file import ModelBuffer, Pipeline, open


class TestPipelineConstruction:
    """Pipeline parameter validation."""

    def test_valid_pipeline(self) -> None:
        """Pipeline with valid parameters constructs successfully."""
        p = Pipeline(size=(224, 224))
        assert repr(p)

    def test_all_options(self) -> None:
        """Pipeline accepts all documented options."""
        p = Pipeline(
            size=(384, 384),
            fit="cover",
            normalize="zero_one",
            layout="nhwc",
            dtype="uint8",
            filter="bicubic",
            pad_color=(128, 128, 128),
        )
        assert "384" in repr(p)

    def test_invalid_size_type(self) -> None:
        """Non-tuple size raises ValidationError."""
        with pytest.raises(ValidationError):
            Pipeline(size=224)

    def test_invalid_size_negative(self) -> None:
        """Negative size raises ValidationError."""
        with pytest.raises(ValidationError):
            Pipeline(size=(-1, 224))

    def test_invalid_fit(self) -> None:
        """Unknown fit mode raises ValidationError."""
        with pytest.raises(ValidationError):
            Pipeline(size=(224, 224), fit="squish")

    def test_invalid_normalize(self) -> None:
        """Unknown normalize mode raises ValidationError."""
        with pytest.raises(ValidationError):
            Pipeline(size=(224, 224), normalize="custom")

    def test_invalid_layout(self) -> None:
        """Unknown layout raises ValidationError."""
        with pytest.raises(ValidationError):
            Pipeline(size=(224, 224), layout="chwn")

    def test_invalid_dtype(self) -> None:
        """Unknown dtype raises ValidationError."""
        with pytest.raises(ValidationError):
            Pipeline(size=(224, 224), dtype="float16")


class TestPipelineCall:
    """Pipeline __call__ with various inputs."""

    def test_pipeline_from_bytes(self, jpeg_bytes: bytes) -> None:
        """Pipeline processes raw JPEG bytes directly."""
        pipeline = Pipeline(size=(224, 224))
        buf = pipeline(jpeg_bytes)
        assert isinstance(buf, ModelBuffer)
        assert buf.shape == (3, 224, 224)
        assert buf.dtype == "float32"
        assert buf.layout == "nchw"
        buf.close()

    def test_pipeline_from_image(self, jpeg_bytes: bytes) -> None:
        """Pipeline processes an Image object."""
        pipeline = Pipeline(size=(112, 112))
        with open(jpeg_bytes) as img:
            with pipeline(img) as buf:
                assert buf.shape == (3, 112, 112)

    def test_pipeline_nhwc(self, small_jpeg_bytes: bytes) -> None:
        """NHWC layout produces (H, W, C) shape."""
        pipeline = Pipeline(size=(32, 32), layout="nhwc")
        buf = pipeline(small_jpeg_bytes)
        assert buf.shape == (32, 32, 3)
        assert buf.layout == "nhwc"
        buf.close()

    def test_pipeline_nchw(self, small_jpeg_bytes: bytes) -> None:
        """NCHW layout produces (C, H, W) shape."""
        pipeline = Pipeline(size=(32, 32), layout="nchw")
        buf = pipeline(small_jpeg_bytes)
        assert buf.shape == (3, 32, 32)
        assert buf.layout == "nchw"
        buf.close()

    def test_pipeline_uint8(self, small_jpeg_bytes: bytes) -> None:
        """uint8 dtype produces uint8 buffer."""
        pipeline = Pipeline(size=(32, 32), dtype="uint8", normalize="none")
        buf = pipeline(small_jpeg_bytes)
        assert buf.dtype == "uint8"
        buf.close()

    def test_pipeline_float32(self, small_jpeg_bytes: bytes) -> None:
        """float32 dtype produces float32 buffer."""
        pipeline = Pipeline(size=(32, 32), dtype="float32")
        buf = pipeline(small_jpeg_bytes)
        assert buf.dtype == "float32"
        buf.close()

    def test_pipeline_invalid_source(self) -> None:
        """Pipeline with invalid source type raises ValidationError."""
        pipeline = Pipeline(size=(224, 224))
        with pytest.raises(ValidationError):
            pipeline(12345)


class TestModelBufferTobytes:
    """ModelBuffer.tobytes() — raw data access without third-party libs."""

    def test_tobytes_float32(self, small_jpeg_bytes: bytes) -> None:
        """tobytes() returns correct length for float32 NCHW."""
        pipeline = Pipeline(size=(32, 32), layout="nchw", dtype="float32")
        with pipeline(small_jpeg_bytes) as buf:
            raw = buf.tobytes()
            assert isinstance(raw, bytes)
            assert len(raw) == 3 * 32 * 32 * 4  # C*H*W * sizeof(f32)

    def test_tobytes_uint8(self, small_jpeg_bytes: bytes) -> None:
        """tobytes() returns correct length for uint8 NHWC."""
        pipeline = Pipeline(size=(16, 16), layout="nhwc", dtype="uint8", normalize="none")
        with pipeline(small_jpeg_bytes) as buf:
            raw = buf.tobytes()
            assert len(raw) == 16 * 16 * 3  # H*W*C * sizeof(u8)

    def test_tobytes_after_close_raises(self, small_jpeg_bytes: bytes) -> None:
        """tobytes() on closed buffer raises StateError."""
        pipeline = Pipeline(size=(32, 32))
        buf = pipeline(small_jpeg_bytes)
        buf.close()
        with pytest.raises(StateError):
            buf.tobytes()


class TestModelBuffer:
    """ModelBuffer properties and lifecycle."""

    def test_buffer_properties(self, small_jpeg_bytes: bytes) -> None:
        """ModelBuffer exposes correct properties."""
        pipeline = Pipeline(size=(64, 64), layout="nchw", dtype="float32")
        buf = pipeline(small_jpeg_bytes)
        assert buf.width == 64
        assert buf.height == 64
        assert buf.channels == 3
        assert buf.layout == "nchw"
        assert buf.dtype == "float32"
        assert buf.shape == (3, 64, 64)
        buf.close()

    def test_buffer_close_idempotent(self, small_jpeg_bytes: bytes) -> None:
        """close() can be called multiple times safely."""
        pipeline = Pipeline(size=(32, 32))
        buf = pipeline(small_jpeg_bytes)
        buf.close()
        buf.close()

    def test_buffer_use_after_close(self, small_jpeg_bytes: bytes) -> None:
        """Accessing closed buffer raises StateError."""
        pipeline = Pipeline(size=(32, 32))
        buf = pipeline(small_jpeg_bytes)
        buf.close()
        with pytest.raises(StateError):
            _ = buf.shape

    def test_buffer_context_manager(self, small_jpeg_bytes: bytes) -> None:
        """ModelBuffer works as context manager."""
        pipeline = Pipeline(size=(32, 32))
        with pipeline(small_jpeg_bytes) as buf:
            assert buf.width == 32
        with pytest.raises(StateError):
            _ = buf.width

    def test_buffer_repr(self, small_jpeg_bytes: bytes) -> None:
        """ModelBuffer repr shows shape and dtype."""
        pipeline = Pipeline(size=(32, 32))
        buf = pipeline(small_jpeg_bytes)
        r = repr(buf)
        assert "32" in r
        assert "float32" in r
        buf.close()
        assert "closed" in repr(buf)

    def test_buffer_del_cleanup(self, small_jpeg_bytes: bytes) -> None:
        """__del__ cleans up without error."""
        pipeline = Pipeline(size=(32, 32))
        buf = pipeline(small_jpeg_bytes)
        del buf
        gc.collect()


class TestModelBufferArrayInterface:
    """ModelBuffer __array_interface__ protocol."""

    def test_array_interface_nchw_f32(self, small_jpeg_bytes: bytes) -> None:
        """NCHW float32 buffer has correct array interface."""
        pipeline = Pipeline(size=(32, 32), layout="nchw", dtype="float32")
        with pipeline(small_jpeg_bytes) as buf:
            ai = buf.__array_interface__
            assert ai["version"] == 3
            assert ai["shape"] == (3, 32, 32)
            assert ai["typestr"] == "<f4"
            assert isinstance(ai["data"], tuple)
            assert ai["data"][0] > 0  # Non-null pointer
            assert ai["data"][1] is False  # Not read-only
            # Strides: C * H * W for NCHW
            assert ai["strides"] == (32 * 32 * 4, 32 * 4, 4)

    def test_array_interface_nhwc_u8(self, small_jpeg_bytes: bytes) -> None:
        """NHWC uint8 buffer has correct array interface."""
        pipeline = Pipeline(size=(16, 16), layout="nhwc", dtype="uint8", normalize="none")
        with pipeline(small_jpeg_bytes) as buf:
            ai = buf.__array_interface__
            assert ai["shape"] == (16, 16, 3)
            assert ai["typestr"] == "|u1"
            assert ai["strides"] == (16 * 3, 3, 1)

    def test_array_interface_closed_raises(self, small_jpeg_bytes: bytes) -> None:
        """Accessing array interface on closed buffer raises StateError."""
        pipeline = Pipeline(size=(32, 32))
        buf = pipeline(small_jpeg_bytes)
        buf.close()
        with pytest.raises(StateError):
            _ = buf.__array_interface__

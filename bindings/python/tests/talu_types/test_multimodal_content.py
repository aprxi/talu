"""Tests for multimodal content types (InputImage, InputAudio, InputVideo).

Tests the unified content API where factory methods live on the dataclasses:
- InputImage.from_file(), .from_bytes(), .from_url()
- InputAudio.from_file(), .from_bytes(), .from_url()
- InputVideo.from_file(), .from_bytes(), .from_url()
"""

import base64

import pytest

from talu.types import (
    AUDIO_MIME_MAP,
    IMAGE_MIME_MAP,
    VIDEO_MIME_MAP,
    InputAudio,
    InputImage,
    InputVideo,
    normalize_content,
)


class TestInputImage:
    """Tests for InputImage factory methods."""

    def test_from_bytes(self):
        """Create image from raw bytes."""
        data = b"fake image data"
        img = InputImage.from_bytes(data, mime_type="image/png")

        assert img.image_url == f"data:image/png;base64,{base64.b64encode(data).decode()}"
        assert img.detail.name == "AUTO"

    def test_from_bytes_default_mime(self):
        """Default MIME type for bytes is image/png."""
        img = InputImage.from_bytes(b"data")
        assert "image/png" in img.image_url

    def test_from_bytes_with_detail(self):
        """Detail parameter works with from_bytes."""
        img = InputImage.from_bytes(b"data", detail="high")
        assert img.detail.name == "HIGH"

        img = InputImage.from_bytes(b"data", detail="low")
        assert img.detail.name == "LOW"

        img = InputImage.from_bytes(b"data", detail="auto")
        assert img.detail.name == "AUTO"

    def test_from_file(self, tmp_path):
        """Create image from file path."""
        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"jpeg data")

        img = InputImage.from_file(img_file)

        assert "image/jpeg" in img.image_url
        assert base64.b64encode(b"jpeg data").decode() in img.image_url

    def test_from_file_string_path(self, tmp_path):
        """Create image from string path."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"png data")

        img = InputImage.from_file(str(img_file))

        assert "image/png" in img.image_url

    def test_from_file_with_detail(self, tmp_path):
        """Detail parameter works with from_file."""
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"data")

        img = InputImage.from_file(img_file, detail="high")
        assert img.detail.name == "HIGH"

    def test_mime_detection(self, tmp_path):
        """MIME type is detected from file extension."""
        cases = [
            ("test.jpg", "image/jpeg"),
            ("test.jpeg", "image/jpeg"),
            ("test.png", "image/png"),
            ("test.webp", "image/webp"),
            ("test.gif", "image/gif"),
            ("test.bmp", "image/bmp"),
            ("test.svg", "image/svg+xml"),
        ]
        for filename, expected_mime in cases:
            img_file = tmp_path / filename
            img_file.write_bytes(b"data")
            img = InputImage.from_file(img_file)
            assert expected_mime in img.image_url, f"Failed for {filename}"

    def test_from_url(self):
        """Create image from URL."""
        url = "https://example.com/image.png"
        img = InputImage.from_url(url)

        assert img.image_url == url
        assert img.detail.name == "AUTO"

    def test_from_url_with_detail(self):
        """Detail parameter works with from_url."""
        img = InputImage.from_url("https://example.com/image.png", detail="high")
        assert img.detail.name == "HIGH"

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Image not found"):
            InputImage.from_file(tmp_path / "nonexistent.png")

    def test_to_dict(self):
        """to_dict returns proper format for Zig bridge."""
        img = InputImage.from_bytes(b"test", mime_type="image/png", detail="high")
        d = img.to_dict()

        assert d["type"] == "input_image"
        assert "data:image/png;base64," in d["image_url"]
        assert d["detail"] == "high"

    def test_frozen_dataclass(self):
        """InputImage is immutable."""
        img = InputImage.from_bytes(b"test")
        with pytest.raises(AttributeError):
            img.image_url = "new_url"


class TestInputAudio:
    """Tests for InputAudio factory methods."""

    def test_from_bytes(self):
        """Create audio from raw bytes."""
        data = b"fake audio data"
        audio = InputAudio.from_bytes(data, mime_type="audio/wav")

        assert audio.audio_data == f"data:audio/wav;base64,{base64.b64encode(data).decode()}"

    def test_from_bytes_default_mime(self):
        """Default MIME type for bytes is audio/wav."""
        audio = InputAudio.from_bytes(b"data")
        assert "audio/wav" in audio.audio_data

    def test_from_file(self, tmp_path):
        """Create audio from file path."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"mp3 data")

        audio = InputAudio.from_file(audio_file)

        assert "audio/mpeg" in audio.audio_data

    def test_mime_detection(self, tmp_path):
        """MIME type is detected from file extension."""
        cases = [
            ("test.wav", "audio/wav"),
            ("test.mp3", "audio/mpeg"),
            ("test.ogg", "audio/ogg"),
            ("test.flac", "audio/flac"),
            ("test.aac", "audio/aac"),
            ("test.m4a", "audio/mp4"),
        ]
        for filename, expected_mime in cases:
            audio_file = tmp_path / filename
            audio_file.write_bytes(b"data")
            audio = InputAudio.from_file(audio_file)
            assert expected_mime in audio.audio_data, f"Failed for {filename}"

    def test_from_url(self):
        """Create audio from URL."""
        url = "https://example.com/audio.mp3"
        audio = InputAudio.from_url(url)

        assert audio.audio_data == url

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Audio not found"):
            InputAudio.from_file(tmp_path / "nonexistent.wav")

    def test_to_dict(self):
        """to_dict returns proper format for Zig bridge."""
        audio = InputAudio.from_bytes(b"test", mime_type="audio/wav")
        d = audio.to_dict()

        assert d["type"] == "input_audio"
        assert "data:audio/wav;base64," in d["audio_data"]


class TestInputVideo:
    """Tests for InputVideo factory methods."""

    def test_from_bytes(self):
        """Create video from raw bytes."""
        data = b"fake video data"
        video = InputVideo.from_bytes(data, mime_type="video/mp4")

        assert video.video_url == f"data:video/mp4;base64,{base64.b64encode(data).decode()}"

    def test_from_bytes_default_mime(self):
        """Default MIME type for bytes is video/mp4."""
        video = InputVideo.from_bytes(b"data")
        assert "video/mp4" in video.video_url

    def test_from_file(self, tmp_path):
        """Create video from file path."""
        video_file = tmp_path / "test.webm"
        video_file.write_bytes(b"webm data")

        video = InputVideo.from_file(video_file)

        assert "video/webm" in video.video_url

    def test_mime_detection(self, tmp_path):
        """MIME type is detected from file extension."""
        cases = [
            ("test.mp4", "video/mp4"),
            ("test.webm", "video/webm"),
            ("test.avi", "video/x-msvideo"),
            ("test.mov", "video/quicktime"),
            ("test.mkv", "video/x-matroska"),
        ]
        for filename, expected_mime in cases:
            video_file = tmp_path / filename
            video_file.write_bytes(b"data")
            video = InputVideo.from_file(video_file)
            assert expected_mime in video.video_url, f"Failed for {filename}"

    def test_from_url(self):
        """Create video from URL."""
        url = "https://example.com/video.mp4"
        video = InputVideo.from_url(url)

        assert video.video_url == url

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Video not found"):
            InputVideo.from_file(tmp_path / "nonexistent.mp4")

    def test_to_dict(self):
        """to_dict returns proper format for Zig bridge."""
        video = InputVideo.from_bytes(b"test", mime_type="video/mp4")
        d = video.to_dict()

        assert d["type"] == "input_video"
        assert "data:video/mp4;base64," in d["video_url"]


class TestNormalizeContent:
    """Tests for normalize_content helper."""

    def test_string_input(self):
        """String input becomes input_text dict."""
        result = normalize_content("Hello")
        assert result == [{"type": "input_text", "text": "Hello"}]

    def test_dict_passthrough(self):
        """Dict input is passed through."""
        d = {"type": "custom", "data": "value"}
        result = normalize_content([d])
        assert result == [d]

    def test_content_part_conversion(self):
        """ContentPart objects are converted via to_dict()."""
        img = InputImage.from_bytes(b"test", mime_type="image/png")
        result = normalize_content([img])

        assert len(result) == 1
        assert result[0]["type"] == "input_image"

    def test_mixed_content(self):
        """Mixed content types are normalized."""
        img = InputImage.from_bytes(b"test", mime_type="image/png")
        result = normalize_content(
            [
                "Describe this:",
                img,
                {"type": "custom", "value": 123},
            ]
        )

        assert len(result) == 3
        assert result[0]["type"] == "input_text"
        assert result[1]["type"] == "input_image"
        assert result[2]["type"] == "custom"

    def test_invalid_type_raises(self):
        """Invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Cannot normalize content"):
            normalize_content([123])  # int has no to_dict


class TestContentIntegration:
    """Integration tests for multimodal content lists."""

    def test_mixed_content_list(self, tmp_path):
        """Content can be mixed with strings in a list."""
        img_file = tmp_path / "photo.png"
        img_file.write_bytes(b"image")

        content_list = [
            "What's in this image?",  # text is just a string
            InputImage.from_file(img_file),
        ]

        assert isinstance(content_list[0], str)
        assert isinstance(content_list[1], InputImage)

        # Normalize for API consumption
        normalized = normalize_content(content_list)
        assert normalized[0]["type"] == "input_text"
        assert normalized[1]["type"] == "input_image"

    def test_multiple_media_types(self, tmp_path):
        """Multiple media types can be combined."""
        img = tmp_path / "photo.png"
        img.write_bytes(b"img")
        audio = tmp_path / "voice.wav"
        audio.write_bytes(b"audio")

        content_list = [
            "Describe the image and transcribe the audio:",
            InputImage.from_file(img),
            InputAudio.from_file(audio),
        ]

        normalized = normalize_content(content_list)
        assert len(normalized) == 3
        assert normalized[1]["type"] == "input_image"
        assert normalized[2]["type"] == "input_audio"


class TestMimeMaps:
    """Tests for MIME type map exports."""

    def test_image_mime_map(self):
        """IMAGE_MIME_MAP is exported and has expected entries."""
        assert ".jpg" in IMAGE_MIME_MAP
        assert ".png" in IMAGE_MIME_MAP
        assert IMAGE_MIME_MAP[".jpg"] == "image/jpeg"

    def test_audio_mime_map(self):
        """AUDIO_MIME_MAP is exported and has expected entries."""
        assert ".wav" in AUDIO_MIME_MAP
        assert ".mp3" in AUDIO_MIME_MAP
        assert AUDIO_MIME_MAP[".mp3"] == "audio/mpeg"

    def test_video_mime_map(self):
        """VIDEO_MIME_MAP is exported and has expected entries."""
        assert ".mp4" in VIDEO_MIME_MAP
        assert ".webm" in VIDEO_MIME_MAP
        assert VIDEO_MIME_MAP[".mp4"] == "video/mp4"

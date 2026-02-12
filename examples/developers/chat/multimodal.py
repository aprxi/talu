"""Multimodal input - Send images, audio, and video with text.

Primary API: talu.types.InputImage, talu.types.InputAudio, talu.types.InputVideo
Scope: Single

This example demonstrates:
- Creating image, audio, and video content using factory methods
- Composing multimodal messages
- MIME type auto-detection from file extensions

Related:
    - examples/developers/chat/standard_interfaces.py
"""

import os
import tempfile
from pathlib import Path

from talu.types import InputAudio, InputImage, InputVideo, normalize_content

# =============================================================================
# Design: Text is a string, media uses factory methods on dataclasses
# =============================================================================

# Text is the 99% use case - it should be frictionless.
# Binary media (images, audio, video) need encoding, so they use factory methods.

# The factory methods live on the dataclasses themselves:
# - InputImage.from_file(), .from_bytes(), .from_url()
# - InputAudio.from_file(), .from_bytes(), .from_url()
# - InputVideo.from_file(), .from_bytes(), .from_url()

# =============================================================================
# Creating image content
# =============================================================================

# From bytes (e.g., downloaded or generated)
png_bytes = b"\x89PNG\r\n\x1a\n..."  # fake PNG data
image_from_bytes = InputImage.from_bytes(png_bytes, mime_type="image/png")
print(f"Image type: {image_from_bytes.to_dict()['type']}")
print(f"Image URL starts with: {image_from_bytes.image_url[:30]}...")

# From file path (MIME auto-detected from extension)
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
    f.write(b"fake jpeg data")
    jpg_path = f.name

image_from_file = InputImage.from_file(jpg_path)
print(f"Auto-detected MIME in URL: {'image/jpeg' in image_from_file.image_url}")

# From Path object
image_from_path = InputImage.from_file(Path(jpg_path))
print(f"Path object works too: {'image/jpeg' in image_from_path.image_url}")

# With detail control for vision models
high_detail = InputImage.from_bytes(png_bytes, mime_type="image/png", detail="high")
low_detail = InputImage.from_bytes(png_bytes, mime_type="image/png", detail="low")
print(f"High detail for analysis: {high_detail.detail.name}")
print(f"Low detail for thumbnails: {low_detail.detail.name}")

# From URL (for remote images)
remote_image = InputImage.from_url("https://example.com/image.png", detail="auto")
print(f"Remote image URL: {remote_image.image_url}")

# =============================================================================
# Creating audio content
# =============================================================================

# From bytes
wav_bytes = b"RIFF..."  # fake WAV data
audio_from_bytes = InputAudio.from_bytes(wav_bytes, mime_type="audio/wav")
print(f"Audio type: {audio_from_bytes.to_dict()['type']}")

# From file (MIME auto-detected)
with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
    f.write(b"fake mp3 data")
    mp3_path = f.name

audio_from_file = InputAudio.from_file(mp3_path)
print(f"Audio MIME in URL: {'audio/mpeg' in audio_from_file.audio_data}")

# From URL
remote_audio = InputAudio.from_url("https://example.com/audio.mp3")
print(f"Remote audio URL: {remote_audio.audio_data}")

# =============================================================================
# Creating video content
# =============================================================================

# From bytes
mp4_bytes = b"\x00\x00\x00..."  # fake MP4 data
video_from_bytes = InputVideo.from_bytes(mp4_bytes, mime_type="video/mp4")
print(f"Video type: {video_from_bytes.to_dict()['type']}")

# From file (MIME auto-detected)
with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
    f.write(b"fake webm data")
    webm_path = f.name

video_from_file = InputVideo.from_file(webm_path)
print(f"Video MIME in URL: {'video/webm' in video_from_file.video_url}")

# From URL
remote_video = InputVideo.from_url("https://example.com/video.mp4")
print(f"Remote video URL: {remote_video.video_url}")

# =============================================================================
# Composing multimodal messages
# =============================================================================

# Mix text (strings) with media (InputImage/Audio/Video) in a list
content_list = [
    "What's in this image?",  # Text is just a string
    InputImage.from_file(jpg_path),  # Image from file
]
print(f"Message has {len(content_list)} parts")
print(f"Part 1 is string: {isinstance(content_list[0], str)}")
print(f"Part 2 is InputImage: {isinstance(content_list[1], InputImage)}")

# Multiple media types in one message
multi_media = [
    "Describe the image and transcribe the audio:",
    InputImage.from_file(jpg_path),
    InputAudio.from_file(mp3_path),
]
print(f"Multi-media message has {len(multi_media)} parts")

# Normalize for API consumption (converts to dict format)
normalized = normalize_content(multi_media)
print(f"Normalized types: {[p['type'] for p in normalized]}")

# =============================================================================
# Using with Chat (commented - requires model)
# =============================================================================

# import talu
#
# chat = talu.Chat("Qwen/Qwen3-0.6B")  # Use a vision model for images
#
# # Send multimodal message
# response = chat([
#     "What do you see in this image?",
#     InputImage.from_file("photo.jpg"),
# ])
# print(response)
#
# # Follow up about the image
# response = response.append("What colors are prominent?")
# print(response)

# =============================================================================
# Supported formats
# =============================================================================

print("\n=== Supported Formats ===")
print("Images: jpg, jpeg, png, webp, gif, bmp, svg, tiff, ico, heic, heif")
print("Audio:  wav, mp3, ogg, flac, aac, m4a, webm")
print("Video:  mp4, webm, avi, mov, mkv, wmv, flv, ogv")

# Cleanup temp files
os.unlink(jpg_path)
os.unlink(mp3_path)
os.unlink(webm_path)

"""
Topics covered:

* chat.multimodal
* content.mime
* multimodal.image
* multimodal.audio
* multimodal.video

Related:

* examples/developers/chat/standard_interfaces.py
"""

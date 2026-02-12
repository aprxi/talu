"""
Typed Python objects for the Open Responses API.

This module provides typed dataclasses that mirror the Open Responses API schema,
enabling zero-copy access to conversation Items from Zig memory.

Architecture:
    Zig Memory (Item-based)          Python (Typed Objects)
    ========================================================
    Item { type: message }      <--  MessageItem
    Item { type: function_call } <-- FunctionCallItem
    Item { type: reasoning }    <--  ReasoningItem

Content Types:
    Messages and reasoning items contain content parts:
    - InputText, InputImage, InputAudio, InputVideo, InputFile (inputs)
    - OutputText (with logprobs and annotations), Refusal (outputs)
    - ReasoningText, SummaryText (reasoning)

Factory Methods:
    InputImage, InputAudio, and InputVideo provide factory class methods for
    convenient creation from files, bytes, or URLs:

    >>> # From file (MIME auto-detected)
    >>> img = InputImage.from_file("photo.jpg")

    >>> # From bytes with explicit MIME
    >>> img = InputImage.from_bytes(png_data, mime_type="image/png")

    >>> # From URL (passed through as-is)
    >>> img = InputImage.from_url("https://example.com/image.png")

Example:
    >>> # Access items via chat.items (polymorphic)
    >>> for item in chat.items:
    ...     if isinstance(item, MessageItem):
    ...         print(f"{item.role}: {item.text}")
    ...     elif isinstance(item, ReasoningItem):
    ...         print(f"Thinking: {item.summary}")
    ...     elif isinstance(item, FunctionCallItem):
    ...         print(f"Tool call: {item.name}({item.arguments})")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

__all__ = [
    # Enums
    "ItemType",
    "ItemStatus",
    "MessageRole",
    "ContentType",
    "ImageDetail",
    # MIME Maps (for advanced usage)
    "IMAGE_MIME_MAP",
    "AUDIO_MIME_MAP",
    "VIDEO_MIME_MAP",
    # Helper functions
    "normalize_content",
    "normalize_message_input",
    # Content Parts
    "ContentPart",
    "CodeBlock",
    "InputText",
    "InputImage",
    "InputAudio",
    "InputVideo",
    "InputFile",
    "OutputText",
    "Refusal",
    "Text",
    "ReasoningText",
    "SummaryText",
    "UnknownContent",
    # Items
    "Item",
    "MessageItem",
    "FunctionCallItem",
    "FunctionCallOutputItem",
    "ReasoningItem",
    "ItemReferenceItem",
    "UnknownItem",
    # Type aliases
    "ConversationItem",
    "ContentPart",
]

# =============================================================================
# MIME Type Mappings
# =============================================================================

IMAGE_MIME_MAP: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".ico": "image/x-icon",
    ".heic": "image/heic",
    ".heif": "image/heif",
}

AUDIO_MIME_MAP: dict[str, str] = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".weba": "audio/webm",
    ".webm": "audio/webm",
}

VIDEO_MIME_MAP: dict[str, str] = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".ogv": "video/ogg",
}


def _encode_file(
    path: str | Path,
    mime_map: dict[str, str],
    default_mime: str,
    content_name: str,
) -> tuple[str, str]:
    """
    Read file and encode to base64, with MIME type auto-detection.

    Returns (base64_data, mime_type).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{content_name} not found: {p}")

    suffix = p.suffix.lower()
    mime = mime_map.get(suffix, default_mime)

    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    return data, mime


def _encode_bytes(data: bytes, mime_type: str | None, default_mime: str) -> tuple[str, str]:
    """
    Encode bytes to base64.

    Returns (base64_data, mime_type).
    """
    encoded = base64.b64encode(data).decode("utf-8")
    return encoded, mime_type if mime_type else default_mime


# =============================================================================
# Enums (must match Zig enums)
# =============================================================================


class ItemType(IntEnum):
    """Item type discriminator."""

    MESSAGE = 0
    FUNCTION_CALL = 1
    FUNCTION_CALL_OUTPUT = 2
    REASONING = 3
    ITEM_REFERENCE = 4
    UNKNOWN = 255


class ItemStatus(IntEnum):
    """Item processing status."""

    IN_PROGRESS = 0
    WAITING = 1
    COMPLETED = 2
    INCOMPLETE = 3
    FAILED = 4


class MessageRole(IntEnum):
    """Message role discriminator."""

    SYSTEM = 0
    USER = 1
    ASSISTANT = 2
    DEVELOPER = 3
    UNKNOWN = 255


class ContentType(IntEnum):
    """Content type discriminator."""

    INPUT_TEXT = 0
    INPUT_IMAGE = 1
    INPUT_AUDIO = 2
    INPUT_VIDEO = 3
    INPUT_FILE = 4
    OUTPUT_TEXT = 5
    REFUSAL = 6
    TEXT = 7
    REASONING_TEXT = 8
    SUMMARY_TEXT = 9
    UNKNOWN = 255


class ImageDetail(IntEnum):
    """Image detail level."""

    AUTO = 0
    LOW = 1
    HIGH = 2


class FinishReason:
    """Constants for generation stop reasons.

    Attributes
    ----------
        EOS_TOKEN: End-of-sequence token generated.
        LENGTH: Maximum token limit reached.
        STOP_SEQUENCE: User-defined stop sequence matched.
        TOOL_CALLS: Model requested tool execution.
        CANCELLED: Request was cancelled (client disconnect, stop flag set).
    """

    EOS_TOKEN = "eos_token"
    LENGTH = "length"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_CALLS = "tool_calls"
    CANCELLED = "cancelled"


def _parse_image_detail(detail: str | ImageDetail | None) -> ImageDetail:
    """Parse detail string to ImageDetail enum."""
    if detail is None:
        return ImageDetail.AUTO
    if isinstance(detail, ImageDetail):
        return detail
    detail_map = {"auto": ImageDetail.AUTO, "low": ImageDetail.LOW, "high": ImageDetail.HIGH}
    return detail_map.get(detail.lower(), ImageDetail.AUTO)


def normalize_content(
    content: str | list[str | dict[str, Any] | ContentPart],
) -> list[dict[str, Any]]:
    """
    Normalize content to a list of dictionaries.

    This function converts various content formats to the dictionary format
    expected by the Zig bridge and storage backends.

    Parameters
    ----------
    content : str or list
        Content to normalize. Can be:
        - A string (becomes [{"type": "input_text", "text": ...}])
        - A list of strings, dicts, or ContentPart objects

    Returns
    -------
    list[dict]
        List of content dictionaries suitable for Zig bridge.

    Examples
    --------
    >>> normalize_content("Hello")
    [{'type': 'input_text', 'text': 'Hello'}]

    >>> normalize_content([
    ...     "Describe this:",
    ...     InputImage.from_file("photo.jpg")
    ... ])
    [{'type': 'input_text', 'text': 'Describe this:'},
     {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,...', 'detail': 'auto'}]
    """
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    result: list[dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            result.append({"type": "input_text", "text": item})
        elif isinstance(item, dict):
            result.append(item)
        elif hasattr(item, "to_dict"):
            # ContentPart dataclass with to_dict() method
            result.append(item.to_dict())  # type: ignore[union-attr]
        else:
            # Fallback: try to convert dataclass fields
            raise TypeError(
                f"Cannot normalize content of type {type(item).__name__}. "
                "Expected str, dict, or ContentPart with to_dict() method."
            )

    return result


def normalize_message_input(
    message: Any,
) -> str | list[dict[str, Any]]:
    """
    Normalize message input to str or list[dict].

    This function enables symmetric input/output for chat.send(). Items retrieved
    from chat.items can be passed back to chat.send() without conversion.

    Parameters
    ----------
    message : str | list[dict] | MessageItem | list[MessageItem]
        Message to normalize. Can be:
        - A string (returned as-is)
        - A list of dicts (returned as-is)
        - A MessageItem (content extracted and normalized)
        - A list of MessageItems (contents extracted and flattened)

    Returns
    -------
    str | list[dict]
        Normalized message suitable for Router.stream() or Router.generate().

    Examples
    --------
    >>> normalize_message_input("Hello")
    'Hello'

    >>> item = MessageItem.create("user", "Hello!")
    >>> normalize_message_input(item)
    [{'type': 'input_text', 'text': 'Hello!'}]

    >>> normalize_message_input([item1, item2])
    [{'type': 'input_text', 'text': '...'}, ...]
    """
    # String: pass through
    if isinstance(message, str):
        return message

    # List of dicts: pass through
    if isinstance(message, list) and message and isinstance(message[0], dict):
        return message

    # MessageItem: extract and normalize content
    # Check by duck typing to avoid circular import
    if hasattr(message, "content") and hasattr(message, "role"):
        # Single MessageItem
        content_parts = message.content  # type: ignore[union-attr]  # tuple[ContentPart, ...]
        result: list[dict[str, Any]] = []
        for part in content_parts:
            if hasattr(part, "to_dict"):
                result.append(part.to_dict())
            elif hasattr(part, "text"):
                # Simple text content (InputText, OutputText, Text)
                result.append({"type": "input_text", "text": part.text})
        return result if result else [{"type": "input_text", "text": ""}]

    # List of MessageItems
    if isinstance(message, list):
        result = []
        for item in message:
            if hasattr(item, "content") and hasattr(item, "role"):
                for part in item.content:
                    if hasattr(part, "to_dict"):
                        result.append(part.to_dict())
                    elif hasattr(part, "text"):
                        result.append({"type": "input_text", "text": part.text})
            elif isinstance(item, dict):
                result.append(item)
        return result

    # Fallback: convert to string
    return str(message)


# =============================================================================
# Content Part Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class _ContentPartBase:
    """Base class for content parts."""

    type: ContentType


@dataclass(frozen=True, slots=True)
class InputText(_ContentPartBase):
    """Input text content (user/system/developer messages)."""

    type: ContentType = field(default=ContentType.INPUT_TEXT, init=False)
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for Zig bridge serialization."""
        return {"type": "input_text", "text": self.text}


@dataclass(frozen=True, slots=True)
class InputImage(_ContentPartBase):
    """
    Input image content.

    Stores an image as a data URI (base64-encoded) or external URL.
    Use factory methods for convenient creation:

    Examples
    --------
    From file (MIME auto-detected)::

        >>> img = InputImage.from_file("photo.jpg")
        >>> img = InputImage.from_file("scan.png", detail="high")

    From bytes with explicit MIME::

        >>> img = InputImage.from_bytes(png_data, mime_type="image/png")

    From URL (passed through as-is)::

        >>> img = InputImage.from_url("https://example.com/image.png")

    With detail control for vision models::

        >>> img = InputImage.from_file("document.png", detail="high")

    Attributes
    ----------
    image_url : str
        Base64 data URI (``data:<mime>;base64,<data>``) or external URL.
    detail : ImageDetail
        Vision model detail level (AUTO, LOW, HIGH).
    """

    type: ContentType = field(default=ContentType.INPUT_IMAGE, init=False)
    image_url: str = ""
    detail: ImageDetail = ImageDetail.AUTO

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        detail: Literal["auto", "low", "high"] | ImageDetail | None = None,
    ) -> InputImage:
        """
        Create InputImage from a file.

        MIME type is auto-detected from the file extension.

        Parameters
        ----------
        path : str or Path
            Path to the image file.
        detail : {'auto', 'low', 'high'} or ImageDetail, optional
            Vision model detail level:
            - 'low': Faster, fewer tokens, lower resolution
            - 'high': Slower, more tokens, full resolution
            - 'auto': Let the model decide (default)

        Returns
        -------
        InputImage
            Image content with base64-encoded data URI.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> img = InputImage.from_file("photo.jpg")
        >>> img = InputImage.from_file("document.png", detail="high")
        """
        data, mime = _encode_file(path, IMAGE_MIME_MAP, "image/png", "Image")
        return cls(
            image_url=f"data:{mime};base64,{data}",
            detail=_parse_image_detail(detail),
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        mime_type: str | None = None,
        detail: Literal["auto", "low", "high"] | ImageDetail | None = None,
    ) -> InputImage:
        """
        Create InputImage from raw bytes.

        Parameters
        ----------
        data : bytes
            Raw image bytes.
        mime_type : str, optional
            MIME type (e.g., 'image/jpeg'). Defaults to 'image/png'.
        detail : {'auto', 'low', 'high'} or ImageDetail, optional
            Vision model detail level.

        Returns
        -------
        InputImage
            Image content with base64-encoded data URI.

        Examples
        --------
        >>> img = InputImage.from_bytes(png_data)
        >>> img = InputImage.from_bytes(jpeg_data, mime_type="image/jpeg")
        """
        encoded, mime = _encode_bytes(data, mime_type, "image/png")
        return cls(
            image_url=f"data:{mime};base64,{encoded}",
            detail=_parse_image_detail(detail),
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        detail: Literal["auto", "low", "high"] | ImageDetail | None = None,
    ) -> InputImage:
        """
        Create InputImage from a URL.

        The URL is stored as-is and will be fetched by the model.

        Parameters
        ----------
        url : str
            URL to the image (http/https).
        detail : {'auto', 'low', 'high'} or ImageDetail, optional
            Vision model detail level.

        Returns
        -------
        InputImage
            Image content with external URL.

        Examples
        --------
        >>> img = InputImage.from_url("https://example.com/photo.jpg")
        """
        return cls(
            image_url=url,
            detail=_parse_image_detail(detail),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for Zig bridge serialization.

        Returns
        -------
        dict
            Dictionary with 'type', 'image_url', and 'detail' keys.
            The 'detail' is converted to string ('auto', 'low', 'high').

        Examples
        --------
        >>> img = InputImage.from_file("photo.jpg")
        >>> img.to_dict()
        {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,...', 'detail': 'auto'}
        """
        detail_map = {ImageDetail.AUTO: "auto", ImageDetail.LOW: "low", ImageDetail.HIGH: "high"}
        return {
            "type": "input_image",
            "image_url": self.image_url,
            "detail": detail_map.get(self.detail, "auto"),
        }


@dataclass(frozen=True, slots=True)
class InputAudio(_ContentPartBase):
    """
    Input audio content.

    Stores audio as a data URI (base64-encoded) or external URL.
    Use factory methods for convenient creation:

    Examples
    --------
    From file (MIME auto-detected)::

        >>> audio = InputAudio.from_file("voice.wav")
        >>> audio = InputAudio.from_file("podcast.mp3")

    From bytes with explicit MIME::

        >>> audio = InputAudio.from_bytes(wav_data, mime_type="audio/wav")

    From URL::

        >>> audio = InputAudio.from_url("https://example.com/audio.mp3")

    Attributes
    ----------
    audio_data : str
        Base64 data URI (``data:<mime>;base64,<data>``) or external URL.
    """

    type: ContentType = field(default=ContentType.INPUT_AUDIO, init=False)
    audio_data: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> InputAudio:
        """
        Create InputAudio from a file.

        MIME type is auto-detected from the file extension.

        Parameters
        ----------
        path : str or Path
            Path to the audio file.

        Returns
        -------
        InputAudio
            Audio content with base64-encoded data URI.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> audio = InputAudio.from_file("voice.wav")
        >>> audio = InputAudio.from_file("podcast.mp3")
        """
        data, mime = _encode_file(path, AUDIO_MIME_MAP, "audio/wav", "Audio")
        return cls(audio_data=f"data:{mime};base64,{data}")

    @classmethod
    def from_bytes(cls, data: bytes, *, mime_type: str | None = None) -> InputAudio:
        """
        Create InputAudio from raw bytes.

        Parameters
        ----------
        data : bytes
            Raw audio bytes.
        mime_type : str, optional
            MIME type (e.g., 'audio/wav'). Defaults to 'audio/wav'.

        Returns
        -------
        InputAudio
            Audio content with base64-encoded data URI.

        Examples
        --------
        >>> audio = InputAudio.from_bytes(wav_data)
        >>> audio = InputAudio.from_bytes(mp3_data, mime_type="audio/mpeg")
        """
        encoded, mime = _encode_bytes(data, mime_type, "audio/wav")
        return cls(audio_data=f"data:{mime};base64,{encoded}")

    @classmethod
    def from_url(cls, url: str) -> InputAudio:
        """
        Create InputAudio from a URL.

        The URL is stored as-is and will be fetched by the model.

        Parameters
        ----------
        url : str
            URL to the audio file (http/https).

        Returns
        -------
        InputAudio
            Audio content with external URL.

        Examples
        --------
        >>> audio = InputAudio.from_url("https://example.com/voice.wav")
        """
        return cls(audio_data=url)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for Zig bridge serialization.

        Returns
        -------
        dict
            Dictionary with 'type' and 'audio_data' keys.

        Examples
        --------
        >>> audio = InputAudio.from_file("voice.wav")
        >>> audio.to_dict()
        {'type': 'input_audio', 'audio_data': 'data:audio/wav;base64,...'}
        """
        return {
            "type": "input_audio",
            "audio_data": self.audio_data,
        }


@dataclass(frozen=True, slots=True)
class InputVideo(_ContentPartBase):
    """
    Input video content.

    Stores video as a data URI (base64-encoded) or external URL.
    Use factory methods for convenient creation:

    Examples
    --------
    From file (MIME auto-detected)::

        >>> video = InputVideo.from_file("clip.mp4")
        >>> video = InputVideo.from_file("recording.webm")

    From bytes with explicit MIME::

        >>> video = InputVideo.from_bytes(mp4_data, mime_type="video/mp4")

    From URL::

        >>> video = InputVideo.from_url("https://example.com/video.mp4")

    Attributes
    ----------
    video_url : str
        Base64 data URI (``data:<mime>;base64,<data>``) or external URL.
    """

    type: ContentType = field(default=ContentType.INPUT_VIDEO, init=False)
    video_url: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> InputVideo:
        """
        Create InputVideo from a file.

        MIME type is auto-detected from the file extension.

        Parameters
        ----------
        path : str or Path
            Path to the video file.

        Returns
        -------
        InputVideo
            Video content with base64-encoded data URI.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        Examples
        --------
        >>> video = InputVideo.from_file("clip.mp4")
        >>> video = InputVideo.from_file("recording.webm")
        """
        data, mime = _encode_file(path, VIDEO_MIME_MAP, "video/mp4", "Video")
        return cls(video_url=f"data:{mime};base64,{data}")

    @classmethod
    def from_bytes(cls, data: bytes, *, mime_type: str | None = None) -> InputVideo:
        """
        Create InputVideo from raw bytes.

        Parameters
        ----------
        data : bytes
            Raw video bytes.
        mime_type : str, optional
            MIME type (e.g., 'video/mp4'). Defaults to 'video/mp4'.

        Returns
        -------
        InputVideo
            Video content with base64-encoded data URI.

        Examples
        --------
        >>> video = InputVideo.from_bytes(mp4_data)
        >>> video = InputVideo.from_bytes(webm_data, mime_type="video/webm")
        """
        encoded, mime = _encode_bytes(data, mime_type, "video/mp4")
        return cls(video_url=f"data:{mime};base64,{encoded}")

    @classmethod
    def from_url(cls, url: str) -> InputVideo:
        """
        Create InputVideo from a URL.

        The URL is stored as-is and will be fetched by the model.

        Parameters
        ----------
        url : str
            URL to the video file (http/https).

        Returns
        -------
        InputVideo
            Video content with external URL.

        Examples
        --------
        >>> video = InputVideo.from_url("https://example.com/clip.mp4")
        """
        return cls(video_url=url)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary format for Zig bridge serialization.

        Returns
        -------
        dict
            Dictionary with 'type' and 'video_url' keys.

        Examples
        --------
        >>> video = InputVideo.from_file("clip.mp4")
        >>> video.to_dict()
        {'type': 'input_video', 'video_url': 'data:video/mp4;base64,...'}
        """
        return {
            "type": "input_video",
            "video_url": self.video_url,
        }


@dataclass(frozen=True, slots=True)
class InputFile(_ContentPartBase):
    """Input file content."""

    type: ContentType = field(default=ContentType.INPUT_FILE, init=False)
    filename: str | None = None
    file_data: str | None = None
    file_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for Zig bridge serialization."""
        result: dict[str, Any] = {"type": "input_file"}
        if self.filename is not None:
            result["filename"] = self.filename
        if self.file_data is not None:
            result["file_data"] = self.file_data
        if self.file_url is not None:
            result["file_url"] = self.file_url
        return result


@dataclass(frozen=True, slots=True)
class CodeBlock:
    r"""Detected code block with position metadata.

    Positions are byte offsets into the source text. All ranges are half-open: [start, end).

    Attributes
    ----------
        index: Sequential index of this block (0-based).
        fence_start: Byte offset where opening fence begins.
        fence_end: Byte offset after closing fence (or current position if incomplete).
        language_start: Byte offset where language identifier begins.
        language_end: Byte offset after language identifier.
        content_start: Byte offset where code content begins.
        content_end: Byte offset where code content ends.
        complete: True if closing fence was found.

    Example:
        >>> # Given text: "```python\\nprint('hi')\\n```"
        >>> block = output.code_blocks[0]
        >>> lang = block.get_language(output.text)  # "python"
        >>> code = block.get_content(output.text)   # "print('hi')"
    """

    index: int
    fence_start: int
    fence_end: int
    language_start: int
    language_end: int
    content_start: int
    content_end: int
    complete: bool

    def get_language(self, source: str) -> str:
        """Extract the language string from the source text.

        Args:
            source: The full text containing this code block.

        Returns
        -------
            Language identifier (e.g., "python", "rust"), or empty string if none.
        """
        return source[self.language_start : self.language_end]

    def get_content(self, source: str) -> str:
        """Extract the code content from the source text.

        Args:
            source: The full text containing this code block.

        Returns
        -------
            The code content (without fences or language identifier).
        """
        return source[self.content_start : self.content_end]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeBlock:
        """Create a CodeBlock from a dictionary.

        Args:
            data: Dictionary with code block fields.

        Returns
        -------
            CodeBlock instance.
        """
        return cls(
            index=data.get("index", 0),
            fence_start=data.get("fence_start", 0),
            fence_end=data.get("fence_end", 0),
            language_start=data.get("language_start", 0),
            language_end=data.get("language_end", 0),
            content_start=data.get("content_start", 0),
            content_end=data.get("content_end", 0),
            complete=data.get("complete", False),
        )


@dataclass(frozen=True, slots=True)
class OutputText(_ContentPartBase):
    """Output text content (assistant responses).

    Attributes
    ----------
        text: The generated text content.
        logprobs: Token log probabilities (if requested).
        annotations: Content annotations (citations, etc.).
        code_blocks: Detected code blocks with position metadata.
    """

    type: ContentType = field(default=ContentType.OUTPUT_TEXT, init=False)
    text: str = ""
    logprobs: list[dict[str, Any]] | None = None
    annotations: list[dict[str, Any]] | None = None
    code_blocks: list[CodeBlock] | None = None


@dataclass(frozen=True, slots=True)
class Refusal(_ContentPartBase):
    """Refusal content (model declined to respond)."""

    type: ContentType = field(default=ContentType.REFUSAL, init=False)
    refusal: str = ""


@dataclass(frozen=True, slots=True)
class Text(_ContentPartBase):
    """Generic text content."""

    type: ContentType = field(default=ContentType.TEXT, init=False)
    text: str = ""


@dataclass(frozen=True, slots=True)
class ReasoningText(_ContentPartBase):
    """Reasoning text content (chain-of-thought)."""

    type: ContentType = field(default=ContentType.REASONING_TEXT, init=False)
    text: str = ""


@dataclass(frozen=True, slots=True)
class SummaryText(_ContentPartBase):
    """Summary text content (reasoning summary)."""

    type: ContentType = field(default=ContentType.SUMMARY_TEXT, init=False)
    text: str = ""


@dataclass(frozen=True, slots=True)
class UnknownContent(_ContentPartBase):
    """Unknown content type (forward compatibility)."""

    type: ContentType = field(default=ContentType.UNKNOWN, init=False)
    raw_type: str = ""
    raw_data: str = ""


# Type alias for any content part
ContentPart = (
    InputText
    | InputImage
    | InputAudio
    | InputVideo
    | InputFile
    | OutputText
    | Refusal
    | Text
    | ReasoningText
    | SummaryText
    | UnknownContent
)

# =============================================================================
# Item Dataclasses
# =============================================================================


@dataclass(frozen=True, slots=True)
class Item:
    """Base class for items."""

    id: int
    type: ItemType
    status: ItemStatus
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class MessageItem(Item):
    """Message item (user, assistant, system, developer)."""

    type: ItemType = field(default=ItemType.MESSAGE, init=False)
    role: MessageRole = MessageRole.USER
    content: tuple[ContentPart, ...] = ()
    raw_role: str | None = None
    generation: dict[str, Any] | None = None
    """Generation parameters (model, temperature, etc.) for assistant messages."""

    @classmethod
    def create(
        cls,
        role: MessageRole | str,
        content: str | tuple[ContentPart, ...],
    ) -> MessageItem:
        """
        Create a MessageItem with auto-generated metadata.

        This is the ergonomic way to construct a MessageItem for use with
        ``chat.append()``. The id, status, and created_at_ms fields are
        auto-populated.

        Args:
            role: Message role. Can be a MessageRole enum or string
                ("system", "user", "assistant", "developer").
            content: Message content. Can be a string (converted to InputText)
                or a tuple of content parts.

        Returns
        -------
            A new MessageItem ready for use with chat.append().

        Raises
        ------
        ValueError
            If role string is not one of the valid roles.

        Example:
            >>> item = MessageItem.create("user", "Hello!")
            >>> chat.append(item)
            >>>
            >>> # With enum and explicit content
            >>> item = MessageItem.create(
            ...     MessageRole.ASSISTANT,
            ...     (OutputText(text="Hi there!"),)
            ... )
        """
        import time

        # Parse role
        if isinstance(role, str):
            role_map = {
                "system": MessageRole.SYSTEM,
                "user": MessageRole.USER,
                "assistant": MessageRole.ASSISTANT,
                "developer": MessageRole.DEVELOPER,
            }
            parsed_role = role_map.get(role.lower())
            if parsed_role is None:
                raise ValueError(
                    f"Invalid role: {role}. Must be one of: system, user, assistant, developer"
                )
        else:
            parsed_role = role

        # Parse content
        if isinstance(content, str):
            parsed_content: tuple[ContentPart, ...] = (InputText(text=content),)
        else:
            parsed_content = content

        return cls(
            id=0,  # Will be assigned by Zig backend on append
            status=ItemStatus.COMPLETED,
            created_at_ms=int(time.time() * 1000),
            role=parsed_role,
            content=parsed_content,
        )

    @property
    def text(self) -> str:
        """Get first text content (convenience for simple messages)."""
        for part in self.content:
            if isinstance(part, (InputText, OutputText, Text)):
                return part.text
        return ""

    def to_message_dict(self) -> dict[str, str]:
        """
        Convert to OpenAI-compatible message dict.

        Returns the standard ``{"role": "...", "content": "..."}`` format
        used by OpenAI, Anthropic, HuggingFace, and most LLM APIs.

        This is useful for:
        - Debugging ("What does the LLM see?")
        - Exporting conversations to JSON
        - Sending to other APIs
        - Interoperability with other libraries

        Returns
        -------
            Dict with 'role' and 'content' keys.

        Example:
            >>> item = MessageItem(role=MessageRole.USER, content=(InputText(text="Hello"),))
            >>> item.to_message_dict()
            {'role': 'user', 'content': 'Hello'}
        """
        role_map = {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.DEVELOPER: "developer",
            MessageRole.UNKNOWN: self.raw_role or "unknown",
        }
        return {
            "role": role_map.get(self.role, "unknown"),
            "content": self.text,
        }


@dataclass(frozen=True, slots=True)
class FunctionCallItem(Item):
    """Function/tool call item."""

    type: ItemType = field(default=ItemType.FUNCTION_CALL, init=False)
    name: str = ""
    call_id: str = ""
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class FunctionCallOutputItem(Item):
    """Function/tool call output item."""

    type: ItemType = field(default=ItemType.FUNCTION_CALL_OUTPUT, init=False)
    call_id: str = ""
    output_text: str | None = None
    output_parts: tuple[ContentPart, ...] | None = None

    @property
    def output(self) -> str:
        """Get output as text."""
        if self.output_text is not None:
            return self.output_text
        if self.output_parts:
            for part in self.output_parts:
                if isinstance(part, (InputText, Text, OutputText)):
                    return part.text
        return ""


@dataclass(frozen=True, slots=True)
class ReasoningItem(Item):
    """Reasoning item (chain-of-thought)."""

    type: ItemType = field(default=ItemType.REASONING, init=False)
    content: tuple[ContentPart, ...] = ()
    summary: tuple[ContentPart, ...] = ()
    encrypted_content: str | None = None

    @property
    def text(self) -> str:
        """Get reasoning text (from content parts)."""
        texts = []
        for part in self.content:
            if isinstance(part, (ReasoningText, Text)):
                texts.append(part.text)
        return "".join(texts)

    @property
    def summary_text(self) -> str:
        """Get summary text."""
        for part in self.summary:
            if isinstance(part, (SummaryText, Text)):
                return part.text
        return ""


@dataclass(frozen=True, slots=True)
class ItemReferenceItem(Item):
    """Item reference (pointer to another item)."""

    type: ItemType = field(default=ItemType.ITEM_REFERENCE, init=False)
    ref_id: str = ""


@dataclass(frozen=True, slots=True)
class UnknownItem(Item):
    """Unknown item type (forward compatibility)."""

    type: ItemType = field(default=ItemType.UNKNOWN, init=False)
    raw_type: str = ""
    payload: str = ""


# Type alias for any item
ConversationItem = (
    MessageItem
    | FunctionCallItem
    | FunctionCallOutputItem
    | ReasoningItem
    | ItemReferenceItem
    | UnknownItem
)

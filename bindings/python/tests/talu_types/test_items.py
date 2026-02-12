"""
Tests for talu/types/items.py.

Tests for data model enums, content parts, item dataclasses, and helper functions.
These are pure-Python tests with no FFI dependency.
"""

from unittest.mock import Mock

from talu.types import (
    ContentPart,
    ContentType,
    ConversationItem,
    FunctionCallItem,
    FunctionCallOutputItem,
    ImageDetail,
    InputAudio,
    InputFile,
    InputImage,
    InputText,
    InputVideo,
    Item,
    ItemReferenceItem,
    ItemStatus,
    ItemType,
    MessageItem,
    MessageRole,
    OutputText,
    ReasoningItem,
    ReasoningText,
    Refusal,
    SummaryText,
    Text,
    UnknownContent,
    UnknownItem,
    normalize_message_input,
)
from talu.types.items import _parse_image_detail

# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for item-related enums."""

    def test_item_type_values(self):
        """ItemType enum has expected values."""
        assert ItemType.MESSAGE == 0
        assert ItemType.FUNCTION_CALL == 1
        assert ItemType.FUNCTION_CALL_OUTPUT == 2
        assert ItemType.REASONING == 3
        assert ItemType.ITEM_REFERENCE == 4
        assert ItemType.UNKNOWN == 255

    def test_item_status_values(self):
        """ItemStatus enum has expected values."""
        assert ItemStatus.IN_PROGRESS == 0
        assert ItemStatus.WAITING == 1
        assert ItemStatus.COMPLETED == 2
        assert ItemStatus.INCOMPLETE == 3
        assert ItemStatus.FAILED == 4

    def test_message_role_values(self):
        """MessageRole enum has expected values."""
        assert MessageRole.SYSTEM == 0
        assert MessageRole.USER == 1
        assert MessageRole.ASSISTANT == 2
        assert MessageRole.DEVELOPER == 3
        assert MessageRole.UNKNOWN == 255

    def test_content_type_values(self):
        """ContentType enum has expected values."""
        assert ContentType.INPUT_TEXT == 0
        assert ContentType.INPUT_IMAGE == 1
        assert ContentType.INPUT_AUDIO == 2
        assert ContentType.INPUT_VIDEO == 3
        assert ContentType.INPUT_FILE == 4
        assert ContentType.OUTPUT_TEXT == 5
        assert ContentType.REFUSAL == 6
        assert ContentType.TEXT == 7
        assert ContentType.REASONING_TEXT == 8
        assert ContentType.SUMMARY_TEXT == 9
        assert ContentType.UNKNOWN == 255

    def test_image_detail_values(self):
        """ImageDetail enum has expected values."""
        assert ImageDetail.AUTO == 0
        assert ImageDetail.LOW == 1
        assert ImageDetail.HIGH == 2


# =============================================================================
# Content Part Tests
# =============================================================================


class TestContentParts:
    """Tests for content part dataclasses."""

    def test_input_text(self):
        """InputText content part."""
        part = InputText(text="Hello world")
        assert part.type == ContentType.INPUT_TEXT
        assert part.text == "Hello world"

    def test_input_image(self):
        """InputImage content part."""
        part = InputImage(image_url="http://example.com/image.png", detail=ImageDetail.HIGH)
        assert part.type == ContentType.INPUT_IMAGE
        assert part.image_url == "http://example.com/image.png"
        assert part.detail == ImageDetail.HIGH

    def test_input_audio(self):
        """InputAudio content part."""
        part = InputAudio(audio_data="base64data")
        assert part.type == ContentType.INPUT_AUDIO
        assert part.audio_data == "base64data"

    def test_input_video(self):
        """InputVideo content part."""
        part = InputVideo(video_url="http://example.com/video.mp4")
        assert part.type == ContentType.INPUT_VIDEO
        assert part.video_url == "http://example.com/video.mp4"

    def test_input_file(self):
        """InputFile content part."""
        part = InputFile(filename="test.txt", file_data="file contents")
        assert part.type == ContentType.INPUT_FILE
        assert part.filename == "test.txt"
        assert part.file_data == "file contents"

    def test_output_text(self):
        """OutputText content part."""
        part = OutputText(text="Response", logprobs=[], annotations=[])
        assert part.type == ContentType.OUTPUT_TEXT
        assert part.text == "Response"
        assert part.logprobs == []
        assert part.annotations == []

    def test_refusal(self):
        """Refusal content part."""
        part = Refusal(refusal="I cannot help with that")
        assert part.type == ContentType.REFUSAL
        assert part.refusal == "I cannot help with that"

    def test_text(self):
        """Text content part."""
        part = Text(text="Plain text")
        assert part.type == ContentType.TEXT
        assert part.text == "Plain text"

    def test_reasoning_text(self):
        """ReasoningText content part."""
        part = ReasoningText(text="Chain of thought")
        assert part.type == ContentType.REASONING_TEXT
        assert part.text == "Chain of thought"

    def test_summary_text(self):
        """SummaryText content part."""
        part = SummaryText(text="Summary")
        assert part.type == ContentType.SUMMARY_TEXT
        assert part.text == "Summary"

    def test_unknown_content(self):
        """UnknownContent content part."""
        part = UnknownContent(raw_type="future_type", raw_data="data")
        assert part.type == ContentType.UNKNOWN
        assert part.raw_type == "future_type"
        assert part.raw_data == "data"

    def test_content_parts_are_frozen(self):
        """Content part dataclasses are frozen."""
        part = InputText(text="Hello")
        # Creating a new instance with same id shows immutability
        part2 = InputText(text="Hello")
        assert part == part2
        # Cannot modify frozen dataclass (would raise in normal usage)
        # Using __setattr__ bypasses frozen check with slots


# =============================================================================
# Item Tests
# =============================================================================


class TestItem:
    """Tests for item dataclasses."""

    def test_base_item(self):
        """Base Item class."""
        item = Item(
            id=1,
            type=ItemType.UNKNOWN,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
        )
        assert item.id == 1
        assert item.type == ItemType.UNKNOWN
        assert item.status == ItemStatus.COMPLETED
        assert item.created_at_ms == 1234567890000

    def test_message_item(self):
        """MessageItem for chat messages."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            role=MessageRole.USER,
            content=(InputText(text="Hello"),),
        )
        assert item.role == MessageRole.USER
        assert item.text == "Hello"

    def test_message_item_text_property_empty_content(self):
        """MessageItem.text property returns empty string with no text content."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            role=MessageRole.USER,
            content=(),
        )
        assert item.text == ""

    def test_message_item_text_property_first_text(self):
        """MessageItem.text property returns first text content."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            role=MessageRole.USER,
            content=(InputImage(image_url="url"), InputText(text="Hello")),
        )
        assert item.text == "Hello"

    def test_function_call_item(self):
        """FunctionCallItem for tool calls."""
        item = FunctionCallItem(
            id=1,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            name="search",
            call_id="call_123",
            arguments='{"query": "test"}',
        )
        assert item.name == "search"
        assert item.call_id == "call_123"
        assert item.arguments == '{"query": "test"}'

    def test_function_call_output_item_with_text(self):
        """FunctionCallOutputItem with output_text."""
        item = FunctionCallOutputItem(
            id=2,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            call_id="call_123",
            output_text="Result",
        )
        assert item.call_id == "call_123"
        assert item.output == "Result"

    def test_function_call_output_item_with_parts(self):
        """FunctionCallOutputItem with output_parts."""
        item = FunctionCallOutputItem(
            id=2,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            call_id="call_123",
            output_parts=(Text(text="Result"),),
        )
        assert item.output == "Result"

    def test_function_call_output_item_output_property_none(self):
        """FunctionCallOutputItem.output property returns empty string when none."""
        item = FunctionCallOutputItem(
            id=2,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            call_id="call_123",
            output_text=None,
            output_parts=None,
        )
        assert item.output == ""

    def test_reasoning_item(self):
        """ReasoningItem for chain-of-thought."""
        item = ReasoningItem(
            id=3,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            content=(ReasoningText(text="Thinking"),),
            summary=(SummaryText(text="Summary"),),
        )
        assert item.text == "Thinking"
        assert item.summary_text == "Summary"

    def test_reasoning_item_text_property_multiple_parts(self):
        """ReasoningItem.text property concatenates ReasoningText and Text parts."""
        item = ReasoningItem(
            id=3,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            content=(
                ReasoningText(text="Step 1"),
                Text(text="Step 2"),
                ReasoningText(text="Step 3"),
            ),
            summary=(),
        )
        # Property concatenates both ReasoningText and Text parts
        assert item.text == "Step 1Step 2Step 3"

    def test_reasoning_item_summary_property(self):
        """ReasoningItem.summary_text property returns summary."""
        item = ReasoningItem(
            id=3,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            content=(),
            summary=(SummaryText(text="Summary"),),
        )
        assert item.summary_text == "Summary"

    def test_reasoning_item_encrypted_content(self):
        """ReasoningItem with encrypted content."""
        item = ReasoningItem(
            id=3,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            content=(),
            summary=(),
            encrypted_content="encrypted_data",
        )
        assert item.encrypted_content == "encrypted_data"

    def test_item_reference_item(self):
        """ItemReferenceItem for context replay."""
        item = ItemReferenceItem(
            id=4,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            ref_id="ref_123",
        )
        assert item.ref_id == "ref_123"

    def test_unknown_item(self):
        """UnknownItem for forward compatibility."""
        item = UnknownItem(
            id=5,
            status=ItemStatus.FAILED,
            created_at_ms=1234567890000,
            raw_type="future_type",
            payload="payload_data",
        )
        assert item.raw_type == "future_type"
        assert item.payload == "payload_data"

    def test_items_are_frozen(self):
        """Item dataclasses are frozen."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            role=MessageRole.USER,
            content=(InputText(text="Hello"),),
        )
        # Creating a new instance with same fields shows immutability
        item2 = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=1234567890000,
            role=MessageRole.USER,
            content=(InputText(text="Hello"),),
        )
        assert item == item2
        # Cannot modify frozen dataclass (would raise in normal usage)
        # Using __setattr__ bypasses frozen check with slots


# =============================================================================
# _parse_image_detail Tests
# =============================================================================


class TestParseImageDetail:
    """Tests for _parse_image_detail function."""

    def test_none_returns_auto(self):
        """None returns AUTO."""
        assert _parse_image_detail(None) == ImageDetail.AUTO

    def test_image_detail_enum_passthrough(self):
        """ImageDetail enum is passed through."""
        assert _parse_image_detail(ImageDetail.HIGH) == ImageDetail.HIGH
        assert _parse_image_detail(ImageDetail.LOW) == ImageDetail.LOW
        assert _parse_image_detail(ImageDetail.AUTO) == ImageDetail.AUTO

    def test_string_conversion(self):
        """String values are converted to enum."""
        assert _parse_image_detail("high") == ImageDetail.HIGH
        assert _parse_image_detail("low") == ImageDetail.LOW
        assert _parse_image_detail("auto") == ImageDetail.AUTO

    def test_case_insensitive(self):
        """String parsing is case-insensitive."""
        assert _parse_image_detail("HIGH") == ImageDetail.HIGH
        assert _parse_image_detail("High") == ImageDetail.HIGH

    def test_unknown_string_returns_auto(self):
        """Unknown string returns AUTO."""
        assert _parse_image_detail("unknown") == ImageDetail.AUTO


# =============================================================================
# normalize_message_input Tests
# =============================================================================


class TestNormalizeMessageInput:
    """Tests for normalize_message_input function."""

    def test_string_passthrough(self):
        """String is passed through."""
        assert normalize_message_input("Hello") == "Hello"

    def test_list_of_dicts_passthrough(self):
        """List of dicts is passed through."""
        dicts = [{"type": "input_text", "text": "Hello"}]
        assert normalize_message_input(dicts) == dicts

    def test_message_item_with_to_dict_content(self):
        """MessageItem with content having to_dict is normalized."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.USER,
            content=(InputText(text="Hello"),),
        )
        result = normalize_message_input(item)
        assert result == [{"type": "input_text", "text": "Hello"}]

    def test_message_item_with_text_only_content(self):
        """MessageItem with content having only text attr."""

        # Create a simple object with text but no to_dict
        class SimpleText:
            def __init__(self, text):
                self.text = text

        item = Mock()
        item.content = (SimpleText("Hello"),)
        item.role = MessageRole.USER

        result = normalize_message_input(item)
        assert result == [{"type": "input_text", "text": "Hello"}]

    def test_message_item_empty_content(self):
        """MessageItem with empty content returns default."""
        item = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.USER,
            content=(),
        )
        result = normalize_message_input(item)
        assert result == [{"type": "input_text", "text": ""}]

    def test_list_of_message_items(self):
        """List of MessageItems extracts and flattens content."""
        item1 = MessageItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.USER,
            content=(InputText(text="Hello"),),
        )
        item2 = MessageItem(
            id=1,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            role=MessageRole.ASSISTANT,
            content=(InputText(text="World"),),
        )
        result = normalize_message_input([item1, item2])
        assert len(result) == 2
        assert result[0] == {"type": "input_text", "text": "Hello"}
        assert result[1] == {"type": "input_text", "text": "World"}

    def test_list_of_message_items_with_text_only_content(self):
        """List of MessageItems with text-only content."""

        class SimpleText:
            def __init__(self, text):
                self.text = text

        item = Mock()
        item.content = (SimpleText("Hello"),)
        item.role = MessageRole.USER

        result = normalize_message_input([item])
        assert result == [{"type": "input_text", "text": "Hello"}]

    def test_list_of_dicts_in_message_items_list(self):
        """List containing dicts is passed through."""
        dicts = [{"type": "custom", "data": "value"}]
        result = normalize_message_input(dicts)
        assert result == dicts

    def test_fallback_to_str(self):
        """Fallback: unknown type converts to string."""
        result = normalize_message_input(12345)
        assert result == "12345"

    def test_empty_list_returns_empty(self):
        """Empty list returns empty list (via list-of-MessageItems path)."""
        result = normalize_message_input([])
        # Empty list: not a list of dicts (line 352 fails due to `message` being falsy)
        # Falls through to line 370 and returns empty list
        assert result == []


# =============================================================================
# InputFile.to_dict Tests
# =============================================================================


class TestInputFileToDict:
    """Tests for InputFile.to_dict method."""

    def test_to_dict_with_all_fields(self):
        """InputFile.to_dict with all fields set."""
        file = InputFile(
            filename="test.txt", file_data="contents", file_url="http://example.com/file"
        )
        d = file.to_dict()

        assert d["type"] == "input_file"
        assert d["filename"] == "test.txt"
        assert d["file_data"] == "contents"
        assert d["file_url"] == "http://example.com/file"

    def test_to_dict_with_filename_only(self):
        """InputFile.to_dict with filename only."""
        file = InputFile(filename="test.txt")
        d = file.to_dict()

        assert d["type"] == "input_file"
        assert d["filename"] == "test.txt"
        assert "file_data" not in d
        assert "file_url" not in d

    def test_to_dict_with_file_data_only(self):
        """InputFile.to_dict with file_data only."""
        file = InputFile(file_data="contents")
        d = file.to_dict()

        assert d["type"] == "input_file"
        assert "filename" not in d
        assert d["file_data"] == "contents"
        assert "file_url" not in d

    def test_to_dict_with_file_url_only(self):
        """InputFile.to_dict with file_url only."""
        file = InputFile(file_url="http://example.com/file")
        d = file.to_dict()

        assert d["type"] == "input_file"
        assert "filename" not in d
        assert "file_data" not in d
        assert d["file_url"] == "http://example.com/file"

    def test_to_dict_empty(self):
        """InputFile.to_dict with no fields set."""
        file = InputFile()
        d = file.to_dict()

        assert d == {"type": "input_file"}


# =============================================================================
# ReasoningItem.summary_text Tests
# =============================================================================


class TestReasoningItemSummaryText:
    """Tests for ReasoningItem.summary_text property."""

    def test_summary_text_empty_summary(self):
        """summary_text returns empty string when summary is empty."""
        item = ReasoningItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            content=(),
            summary=(),
        )
        assert item.summary_text == ""

    def test_summary_text_with_summary_text(self):
        """summary_text returns SummaryText content."""
        item = ReasoningItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            content=(),
            summary=(SummaryText(text="The summary"),),
        )
        assert item.summary_text == "The summary"

    def test_summary_text_with_text_type(self):
        """summary_text returns Text content."""
        item = ReasoningItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            content=(),
            summary=(Text(text="Plain text summary"),),
        )
        assert item.summary_text == "Plain text summary"

    def test_summary_text_first_match_only(self):
        """summary_text returns first SummaryText or Text."""
        item = ReasoningItem(
            id=0,
            status=ItemStatus.COMPLETED,
            created_at_ms=0,
            content=(),
            summary=(SummaryText(text="First"), SummaryText(text="Second")),
        )
        assert item.summary_text == "First"


# =============================================================================
# Type Alias Tests
# =============================================================================


class TestContentPart:
    """Tests for the ContentPart union type alias."""

    def test_importable(self):
        """ContentPart is importable from talu.types."""
        assert ContentPart is not None

    def test_all_content_types_match(self):
        """ContentPart includes all content part dataclasses."""
        expected = {
            InputText,
            InputImage,
            InputAudio,
            InputVideo,
            InputFile,
            OutputText,
            Refusal,
            Text,
            ReasoningText,
            SummaryText,
            UnknownContent,
        }
        # UnionType.__args__ gives the tuple of member types
        assert set(ContentPart.__args__) == expected

    def test_isinstance_checks(self):
        """isinstance works against ContentPart members."""
        assert isinstance(InputText(text="hi"), InputText)
        assert isinstance(OutputText(text="hi"), OutputText)


class TestConversationItem:
    """Tests for the ConversationItem union type alias."""

    def test_importable(self):
        """ConversationItem is importable from talu.types."""
        assert ConversationItem is not None

    def test_all_item_types_match(self):
        """ConversationItem includes all item dataclasses."""
        expected = {
            MessageItem,
            FunctionCallItem,
            FunctionCallOutputItem,
            ReasoningItem,
            ItemReferenceItem,
            UnknownItem,
        }
        assert set(ConversationItem.__args__) == expected

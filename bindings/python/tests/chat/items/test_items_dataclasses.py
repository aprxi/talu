"""
Tests for talu/chat/items.py.

Tests for ConversationItems FFI bridge and C struct compatibility.
Pure data model tests (enums, content parts, items) live in tests/talu_types/test_items.py.
"""

import ctypes
from unittest.mock import Mock

import pytest

from talu._native import (
    CContentPart,
    CFunctionCallItem,
    CItem,
    CMessageItem,
)
from talu.chat.items import ConversationItems, _read_content_part
from talu.types import (
    ContentType,
    FunctionCallItem,
    FunctionCallOutputItem,
    ImageDetail,
    InputAudio,
    InputFile,
    InputImage,
    InputText,
    InputVideo,
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
)

# =============================================================================
# C Struct Tests
# =============================================================================


class TestCStructs:
    """Tests for C-compatible structs."""

    def test_c_item_structure(self):
        """CItem struct has expected fields."""
        c_item = CItem()
        c_item.id = 123
        c_item.item_type = 1
        c_item.status = 2
        c_item.created_at_ms = 456789

        assert c_item.id == 123
        assert c_item.item_type == 1
        assert c_item.status == 2
        assert c_item.created_at_ms == 456789

    def test_c_message_item_structure(self):
        """CMessageItem struct has expected fields."""
        c_msg = CMessageItem()
        c_msg.role = 1
        c_msg.content_count = 2

        assert c_msg.role == 1
        assert c_msg.content_count == 2

    def test_c_function_call_item_structure(self):
        """CFunctionCallItem struct has expected fields."""
        c_fc = CFunctionCallItem()
        c_fc.name_ptr = ctypes.c_char_p(b"function_name")
        c_fc.call_id_ptr = ctypes.c_char_p(b"call_id")

        assert c_fc.name_ptr is not None
        assert c_fc.call_id_ptr is not None

    def test_c_content_part_structure(self):
        """CContentPart struct has expected fields."""
        c_part = CContentPart()
        c_part.content_type = 0
        c_part.image_detail = 1

        assert c_part.content_type == 0
        assert c_part.image_detail == 1


# =============================================================================
# ConversationItems Tests (basic functionality only)
# =============================================================================


class TestConversationItems:
    """Tests for ConversationItems class."""

    def test_conversation_items_requires_lib_and_ptr(self):
        """ConversationItems requires lib and conversation_ptr."""
        mock_lib = Mock()

        items = ConversationItems(mock_lib, 12345)

        assert items._lib == mock_lib
        assert items._conversation_ptr == 12345

    def test_len_returns_count_from_lib(self):
        """__len__ returns item count from lib."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 5

        items = ConversationItems(mock_lib, 12345)
        count = len(items)

        assert count == 5
        mock_lib.talu_responses_item_count.assert_called_once_with(12345)

    def test_getitem_invalid_index_raises(self):
        """__getitem__ raises IndexError for invalid index."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        items = ConversationItems(mock_lib, 12345)

        with pytest.raises(IndexError):
            _ = items[10]

    def test_last_property_none_when_empty(self):
        """last property returns None when empty."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 0

        items = ConversationItems(mock_lib, 12345)
        last = items.last

        assert last is None

    def test_first_property_none_when_empty(self):
        """first property returns None when empty."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 0

        items = ConversationItems(mock_lib, 12345)
        first = items.first

        assert first is None


# =============================================================================
# _read_content_part Tests
# =============================================================================


class TestReadContentPart:
    """Tests for _read_content_part function.

    These tests create CContentPart structs with proper ctypes memory management.
    The key is to keep references to the string buffers alive while the struct is used.
    """

    def _make_c_part(
        self,
        content_type: ContentType,
        data: bytes = b"",
        secondary: bytes = b"",
        tertiary: bytes = b"",
        image_detail: int = 0,
    ) -> tuple[CContentPart, list]:
        """Helper to create a CContentPart with ctypes pointers.

        Returns the part and a list of buffers that must be kept alive.
        """
        part = CContentPart()
        part.content_type = content_type.value
        part.image_detail = image_detail

        # Keep references to prevent garbage collection
        buffers = []

        # Set primary data - use ctypes.POINTER(c_uint8) as per struct definition
        if data:
            data_buf = ctypes.create_string_buffer(data)
            buffers.append(data_buf)
            part.data_ptr = ctypes.cast(data_buf, ctypes.POINTER(ctypes.c_uint8))
            part.data_len = len(data)
        else:
            part.data_ptr = None
            part.data_len = 0

        # Set secondary data
        if secondary:
            sec_buf = ctypes.create_string_buffer(secondary)
            buffers.append(sec_buf)
            part.secondary_ptr = ctypes.cast(sec_buf, ctypes.POINTER(ctypes.c_uint8))
            part.secondary_len = len(secondary)
        else:
            part.secondary_ptr = None
            part.secondary_len = 0

        # Set tertiary data
        if tertiary:
            tert_buf = ctypes.create_string_buffer(tertiary)
            buffers.append(tert_buf)
            part.tertiary_ptr = ctypes.cast(tert_buf, ctypes.POINTER(ctypes.c_uint8))
            part.tertiary_len = len(tertiary)
        else:
            part.tertiary_ptr = None
            part.tertiary_len = 0

        return part, buffers

    def test_input_text(self):
        """_read_content_part for INPUT_TEXT."""
        part, _ = self._make_c_part(ContentType.INPUT_TEXT, b"Hello")
        result = _read_content_part(part)

        assert isinstance(result, InputText)
        assert result.text == "Hello"

    def test_input_image(self):
        """_read_content_part for INPUT_IMAGE."""
        part, _ = self._make_c_part(
            ContentType.INPUT_IMAGE, b"http://example.com/img.png", image_detail=2
        )
        result = _read_content_part(part)

        assert isinstance(result, InputImage)
        assert result.image_url == "http://example.com/img.png"
        assert result.detail == ImageDetail.HIGH

    def test_input_audio(self):
        """_read_content_part for INPUT_AUDIO."""
        part, _ = self._make_c_part(ContentType.INPUT_AUDIO, b"audio_data")
        result = _read_content_part(part)

        assert isinstance(result, InputAudio)
        assert result.audio_data == "audio_data"

    def test_input_video(self):
        """_read_content_part for INPUT_VIDEO."""
        part, _ = self._make_c_part(ContentType.INPUT_VIDEO, b"http://example.com/video.mp4")
        result = _read_content_part(part)

        assert isinstance(result, InputVideo)
        assert result.video_url == "http://example.com/video.mp4"

    def test_input_file(self):
        """_read_content_part for INPUT_FILE."""
        part, _ = self._make_c_part(
            ContentType.INPUT_FILE, b"file_contents", secondary=b"filename.txt"
        )
        result = _read_content_part(part)

        assert isinstance(result, InputFile)
        assert result.file_data == "file_contents"
        assert result.filename == "filename.txt"

    def test_input_file_no_secondary(self):
        """_read_content_part for INPUT_FILE with no secondary data."""
        part, _ = self._make_c_part(ContentType.INPUT_FILE, b"file_contents")
        result = _read_content_part(part)

        assert isinstance(result, InputFile)
        assert result.file_data == "file_contents"
        assert result.filename is None

    def test_output_text_with_annotations_and_logprobs(self):
        """_read_content_part for OUTPUT_TEXT with annotations and logprobs."""
        annotations = b'[{"type": "citation"}]'
        logprobs = b'[{"token": "hi", "logprob": -0.5}]'
        part, _ = self._make_c_part(
            ContentType.OUTPUT_TEXT, b"Hello", secondary=annotations, tertiary=logprobs
        )
        result = _read_content_part(part)

        assert isinstance(result, OutputText)
        assert result.text == "Hello"
        assert result.annotations == [{"type": "citation"}]
        assert result.logprobs == [{"token": "hi", "logprob": -0.5}]

    def test_output_text_invalid_json_annotations(self):
        """_read_content_part for OUTPUT_TEXT with invalid JSON annotations."""
        part, _ = self._make_c_part(ContentType.OUTPUT_TEXT, b"Hello", secondary=b"not json")
        result = _read_content_part(part)

        assert isinstance(result, OutputText)
        assert result.text == "Hello"
        assert result.annotations is None

    def test_output_text_invalid_json_logprobs(self):
        """_read_content_part for OUTPUT_TEXT with invalid JSON logprobs."""
        part, _ = self._make_c_part(ContentType.OUTPUT_TEXT, b"Hello", tertiary=b"not json")
        result = _read_content_part(part)

        assert isinstance(result, OutputText)
        assert result.text == "Hello"
        assert result.logprobs is None

    def test_refusal(self):
        """_read_content_part for REFUSAL."""
        part, _ = self._make_c_part(ContentType.REFUSAL, b"I cannot help with that")
        result = _read_content_part(part)

        assert isinstance(result, Refusal)
        assert result.refusal == "I cannot help with that"

    def test_text(self):
        """_read_content_part for TEXT."""
        part, _ = self._make_c_part(ContentType.TEXT, b"Plain text")
        result = _read_content_part(part)

        assert isinstance(result, Text)
        assert result.text == "Plain text"

    def test_reasoning_text(self):
        """_read_content_part for REASONING_TEXT."""
        part, _ = self._make_c_part(ContentType.REASONING_TEXT, b"Chain of thought")
        result = _read_content_part(part)

        assert isinstance(result, ReasoningText)
        assert result.text == "Chain of thought"

    def test_summary_text(self):
        """_read_content_part for SUMMARY_TEXT."""
        part, _ = self._make_c_part(ContentType.SUMMARY_TEXT, b"Summary")
        result = _read_content_part(part)

        assert isinstance(result, SummaryText)
        assert result.text == "Summary"

    def test_unknown_content_type(self):
        """_read_content_part for UNKNOWN."""
        part, _ = self._make_c_part(ContentType.UNKNOWN, b"raw_data", secondary=b"custom_type")
        result = _read_content_part(part)

        assert isinstance(result, UnknownContent)
        assert result.raw_data == "raw_data"
        assert result.raw_type == "custom_type"


# =============================================================================
# ConversationItems._read_item Error Paths and Item Type Dispatch
# =============================================================================


class TestConversationItemsReadItem:
    """Tests for ConversationItems._read_item and item type dispatch."""

    def test_read_item_error_returns_unknown_item(self):
        """_read_item returns UnknownItem when talu_responses_get_item fails."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1
        mock_lib.talu_responses_get_item.return_value = -1  # Error

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, UnknownItem)
        assert result.id == 0
        assert result.status == ItemStatus.FAILED
        assert "Failed to read item" in result.payload

    def test_read_item_reasoning_dispatch(self):
        """_read_item dispatches to _read_reasoning_item."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.REASONING.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_reasoning(ptr, idx, c_reasoning_ref):
            c_reasoning = c_reasoning_ref._obj
            c_reasoning.content_count = 0
            c_reasoning.summary_count = 0
            c_reasoning.encrypted_content_ptr = None
            c_reasoning.encrypted_content_len = 0
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_reasoning.side_effect = mock_as_reasoning

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ReasoningItem)

    def test_read_item_item_reference_dispatch(self):
        """_read_item dispatches to _read_item_reference_item."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.ITEM_REFERENCE.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_item_ref(ptr, idx, c_ref_ref):
            c_ref = c_ref_ref._obj
            c_ref.id_ptr = ctypes.c_char_p(b"ref_456")
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_item_reference.side_effect = mock_as_item_ref

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ItemReferenceItem)
        assert result.ref_id == "ref_456"

    def test_read_item_unknown_type_fallback(self):
        """_read_item returns UnknownItem for UNKNOWN item type."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 99
            c_item.item_type = ItemType.UNKNOWN.value  # 255 - valid UNKNOWN type
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, UnknownItem)
        assert result.id == 99
        assert result.raw_type == "255"


# =============================================================================
# ConversationItems._read_message_item Error and raw_role
# =============================================================================


class TestConversationItemsReadMessageItem:
    """Tests for _read_message_item."""

    def test_read_message_item_error_returns_default(self):
        """_read_message_item returns default MessageItem on error."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.MESSAGE.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_message.return_value = -1  # Error

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, MessageItem)
        assert result.role == MessageRole.UNKNOWN
        assert result.content == ()

    def test_read_message_item_with_raw_role(self):
        """_read_message_item extracts raw_role when set."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.MESSAGE.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_message(ptr, idx, c_msg_ref):
            c_msg = c_msg_ref._obj
            c_msg.role = MessageRole.UNKNOWN.value
            c_msg.raw_role_ptr = ctypes.c_char_p(b"custom_role")
            c_msg.content_count = 0
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_message.side_effect = mock_as_message

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, MessageItem)
        assert result.raw_role == "custom_role"


# =============================================================================
# ConversationItems._read_function_call_item Error and Arguments
# =============================================================================


class TestConversationItemsReadFunctionCallItem:
    """Tests for _read_function_call_item."""

    def test_read_function_call_item_error_returns_default(self):
        """_read_function_call_item returns default FunctionCallItem on error."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.FUNCTION_CALL.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_function_call.return_value = -1  # Error

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, FunctionCallItem)
        assert result.name == ""
        assert result.call_id == ""
        assert result.arguments == ""


# =============================================================================
# ConversationItems._read_function_call_output_item
# =============================================================================


class TestConversationItemsReadFunctionCallOutputItem:
    """Tests for _read_function_call_output_item."""

    def test_read_fco_error_returns_default(self):
        """_read_function_call_output_item returns default on error."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.FUNCTION_CALL_OUTPUT.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_function_call_output.return_value = -1  # Error

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, FunctionCallOutputItem)
        assert result.call_id == ""

    def test_read_fco_with_output_parts(self):
        """_read_function_call_output_item with output parts."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        text_buf = ctypes.create_string_buffer(b"Part text")

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.FUNCTION_CALL_OUTPUT.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_fco(ptr, idx, c_fco_ref):
            c_fco = c_fco_ref._obj
            c_fco.call_id_ptr = ctypes.c_char_p(b"call_456")
            c_fco.is_text_output = False
            c_fco.output_text_ptr = None
            c_fco.output_text_len = 0
            c_fco.output_parts_count = 1
            return 0

        def mock_get_part(ptr, idx, part_idx, c_part_ref):
            c_part = c_part_ref._obj
            c_part.content_type = ContentType.TEXT.value
            c_part.data_ptr = ctypes.cast(text_buf, ctypes.POINTER(ctypes.c_uint8))
            c_part.data_len = 9
            c_part.secondary_ptr = None
            c_part.secondary_len = 0
            c_part.tertiary_ptr = None
            c_part.tertiary_len = 0
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_function_call_output.side_effect = mock_as_fco
        mock_lib.talu_responses_item_fco_get_part.side_effect = mock_get_part

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, FunctionCallOutputItem)
        assert result.call_id == "call_456"
        assert result.output_parts is not None
        assert len(result.output_parts) == 1
        assert isinstance(result.output_parts[0], Text)


# =============================================================================
# ConversationItems._read_reasoning_item
# =============================================================================


class TestConversationItemsReadReasoningItem:
    """Tests for _read_reasoning_item."""

    def test_read_reasoning_error_returns_default(self):
        """_read_reasoning_item returns default on error."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.REASONING.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_reasoning.return_value = -1  # Error

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ReasoningItem)
        assert result.content == ()
        assert result.summary == ()

    def test_read_reasoning_with_content_and_summary(self):
        """_read_reasoning_item with content and summary parts."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        content_buf = ctypes.create_string_buffer(b"Thinking...")
        summary_buf = ctypes.create_string_buffer(b"Summary text")

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.REASONING.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_reasoning(ptr, idx, c_reasoning_ref):
            c_reasoning = c_reasoning_ref._obj
            c_reasoning.content_count = 1
            c_reasoning.summary_count = 1
            c_reasoning.encrypted_content_ptr = None
            c_reasoning.encrypted_content_len = 0
            return 0

        def mock_get_content(ptr, idx, part_idx, c_part_ref):
            c_part = c_part_ref._obj
            c_part.content_type = ContentType.REASONING_TEXT.value
            c_part.data_ptr = ctypes.cast(content_buf, ctypes.POINTER(ctypes.c_uint8))
            c_part.data_len = 11
            c_part.secondary_ptr = None
            c_part.secondary_len = 0
            c_part.tertiary_ptr = None
            c_part.tertiary_len = 0
            return 0

        def mock_get_summary(ptr, idx, part_idx, c_part_ref):
            c_part = c_part_ref._obj
            c_part.content_type = ContentType.SUMMARY_TEXT.value
            c_part.data_ptr = ctypes.cast(summary_buf, ctypes.POINTER(ctypes.c_uint8))
            c_part.data_len = 12
            c_part.secondary_ptr = None
            c_part.secondary_len = 0
            c_part.tertiary_ptr = None
            c_part.tertiary_len = 0
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_reasoning.side_effect = mock_as_reasoning
        mock_lib.talu_responses_item_reasoning_get_content.side_effect = mock_get_content
        mock_lib.talu_responses_item_reasoning_get_summary.side_effect = mock_get_summary

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ReasoningItem)
        assert len(result.content) == 1
        assert isinstance(result.content[0], ReasoningText)
        assert len(result.summary) == 1
        assert isinstance(result.summary[0], SummaryText)

    def test_read_reasoning_with_encrypted_content(self):
        """_read_reasoning_item with encrypted content."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        encrypted_buf = ctypes.create_string_buffer(b"encrypted_data_here")

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.REASONING.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_reasoning(ptr, idx, c_reasoning_ref):
            c_reasoning = c_reasoning_ref._obj
            c_reasoning.content_count = 0
            c_reasoning.summary_count = 0
            c_reasoning.encrypted_content_ptr = ctypes.cast(
                encrypted_buf, ctypes.POINTER(ctypes.c_uint8)
            )
            c_reasoning.encrypted_content_len = 19
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_reasoning.side_effect = mock_as_reasoning

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ReasoningItem)
        assert result.encrypted_content == "encrypted_data_here"


# =============================================================================
# ConversationItems._read_item_reference_item
# =============================================================================


class TestConversationItemsReadItemReferenceItem:
    """Tests for _read_item_reference_item."""

    def test_read_item_reference_error_returns_default(self):
        """_read_item_reference_item returns default on error."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.ITEM_REFERENCE.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_item_reference.return_value = -1  # Error

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ItemReferenceItem)
        assert result.ref_id == ""

    def test_read_item_reference_success(self):
        """_read_item_reference_item extracts ref_id."""
        mock_lib = Mock()
        mock_lib.talu_responses_item_count.return_value = 1

        def mock_get_item(ptr, idx, c_item_ref):
            c_item = c_item_ref._obj
            c_item.id = 1
            c_item.item_type = ItemType.ITEM_REFERENCE.value
            c_item.status = ItemStatus.COMPLETED.value
            c_item.created_at_ms = 1000
            return 0

        def mock_as_item_ref(ptr, idx, c_ref_ref):
            c_ref = c_ref_ref._obj
            c_ref.id_ptr = ctypes.c_char_p(b"item_ref_789")
            return 0

        mock_lib.talu_responses_get_item.side_effect = mock_get_item
        mock_lib.talu_responses_item_as_item_reference.side_effect = mock_as_item_ref

        items = ConversationItems(mock_lib, 12345)
        result = items[0]

        assert isinstance(result, ItemReferenceItem)
        assert result.ref_id == "item_ref_789"

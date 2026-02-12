"""
Tests for talu/types/records.py.

Tests for storage record TypedDicts: ItemRecord, SessionRecord, and variants.
"""

from talu.types import (
    FunctionCallOutputVariant,
    FunctionCallVariant,
    InputTextContent,
    ItemRecord,
    ItemReferenceVariant,
    MessageItemVariant,
    OutputTextContent,
    ReasoningVariant,
    RecordContentType,
    RecordItemStatus,
    RecordItemType,
    RecordMessageRole,
    SessionRecord,
)


class TestRecordTypeAliases:
    """Tests for record Literal type aliases."""

    def test_record_item_type_is_literal(self):
        """RecordItemType is a Literal type alias."""
        # Verify the type alias accepts valid values
        value: RecordItemType = "message"
        assert value == "message"

    def test_record_item_status_is_literal(self):
        """RecordItemStatus is a Literal type alias."""
        value: RecordItemStatus = "completed"
        assert value == "completed"

    def test_record_message_role_is_literal(self):
        """RecordMessageRole is a Literal type alias."""
        value: RecordMessageRole = "user"
        assert value == "user"

    def test_record_content_type_is_literal(self):
        """RecordContentType is a Literal type alias."""
        value: RecordContentType = "input_text"
        assert value == "input_text"


class TestContentPartRecords:
    """Tests for content part TypedDicts."""

    def test_input_text_content(self):
        """InputTextContent TypedDict."""
        content: InputTextContent = {"type": "input_text", "text": "Hello"}
        assert content["type"] == "input_text"
        assert content["text"] == "Hello"

    def test_output_text_content(self):
        """OutputTextContent TypedDict."""
        content: OutputTextContent = {"type": "output_text", "text": "Response"}
        assert content["type"] == "output_text"
        assert content["text"] == "Response"


class TestItemVariantRecords:
    """Tests for item variant TypedDicts."""

    def test_message_item_variant(self):
        """MessageItemVariant TypedDict."""
        variant: MessageItemVariant = {
            "role": "user",
            "status": "completed",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
        assert variant["role"] == "user"
        assert variant["status"] == "completed"
        assert len(variant["content"]) == 1

    def test_function_call_variant(self):
        """FunctionCallVariant TypedDict."""
        variant: FunctionCallVariant = {
            "call_id": "call_123",
            "name": "search",
            "arguments": '{"q": "test"}',
            "status": "completed",
        }
        assert variant["name"] == "search"
        assert variant["call_id"] == "call_123"

    def test_function_call_output_variant(self):
        """FunctionCallOutputVariant TypedDict."""
        variant: FunctionCallOutputVariant = {
            "call_id": "call_123",
            "output": [{"type": "text", "text": "Result"}],
            "status": "completed",
        }
        assert variant["call_id"] == "call_123"

    def test_reasoning_variant(self):
        """ReasoningVariant TypedDict."""
        variant: ReasoningVariant = {
            "content": [{"type": "reasoning_text", "text": "Thinking"}],
            "summary": [{"type": "summary_text", "text": "Summary"}],
            "status": "completed",
        }
        assert len(variant["content"]) == 1
        assert len(variant["summary"]) == 1

    def test_item_reference_variant(self):
        """ItemReferenceVariant TypedDict."""
        variant: ItemReferenceVariant = {
            "id": "ref_123",
            "status": "completed",
        }
        assert variant["id"] == "ref_123"


class TestItemRecord:
    """Tests for ItemRecord TypedDict."""

    def test_message_item_record(self):
        """ItemRecord with message variant."""
        record: ItemRecord = {
            "item_id": 0,
            "created_at_ms": 1705123456789,
            "status": "completed",
            "item_type": "message",
            "variant": {
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": "Hello!"}],
            },
        }
        assert record["item_id"] == 0
        assert record["item_type"] == "message"
        assert record["variant"]["role"] == "user"

    def test_function_call_item_record(self):
        """ItemRecord with function call variant."""
        record: ItemRecord = {
            "item_id": 1,
            "created_at_ms": 1705123456800,
            "status": "completed",
            "item_type": "function_call",
            "variant": {
                "call_id": "call_abc123",
                "name": "get_weather",
                "arguments": '{"city": "London"}',
                "status": "completed",
            },
        }
        assert record["item_type"] == "function_call"
        assert record["variant"]["name"] == "get_weather"

    def test_item_record_with_metadata(self):
        """ItemRecord with optional metadata."""
        record: ItemRecord = {
            "item_id": 2,
            "created_at_ms": 1705123456900,
            "status": "completed",
            "item_type": "message",
            "variant": {
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Response"}],
            },
            "metadata": {"custom_key": "custom_value"},
        }
        assert record["metadata"] == {"custom_key": "custom_value"}

    def test_item_record_with_token_counts(self):
        """ItemRecord with input/output token counts."""
        record: ItemRecord = {
            "item_id": 3,
            "created_at_ms": 1705123457000,
            "status": "completed",
            "item_type": "message",
            "variant": {
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Response"}],
            },
            "input_tokens": 42,
            "output_tokens": 128,
        }
        assert record["input_tokens"] == 42
        assert record["output_tokens"] == 128


class TestSessionRecord:
    """Tests for SessionRecord TypedDict."""

    def test_session_record_basic(self):
        """SessionRecord with basic fields."""
        record: SessionRecord = {
            "session_id": "user_123",
            "title": "Weather Chat",
            "system_prompt": "You are a weather assistant.",
            "config": {"temperature": 0.7, "max_tokens": 1024},
            "created_at_ms": 1705123456000,
            "updated_at_ms": 1705123456789,
        }
        assert record["session_id"] == "user_123"
        assert record["title"] == "Weather Chat"
        assert record["config"]["temperature"] == 0.7

    def test_session_record_minimal(self):
        """SessionRecord with minimal fields (total=False)."""
        record: SessionRecord = {
            "session_id": "s1",
        }
        assert record["session_id"] == "s1"

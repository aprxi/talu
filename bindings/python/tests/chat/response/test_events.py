"""Tests for talu.chat.events - Streaming event dataclasses."""

from talu.chat.response import DataEvent, ErrorEvent, ResponseMetadata


class TestDataEvent:
    """Test DataEvent dataclass."""

    def test_creation_with_dict_snapshot(self):
        event = DataEvent(snapshot={"key": "value"})
        assert event.snapshot == {"key": "value"}
        assert event.delta is None

    def test_creation_with_list_snapshot(self):
        event = DataEvent(snapshot=[1, 2, 3])
        assert event.snapshot == [1, 2, 3]

    def test_creation_with_delta(self):
        event = DataEvent(snapshot={"a": 1, "b": 2}, delta={"b": 2})
        assert event.snapshot == {"a": 1, "b": 2}
        assert event.delta == {"b": 2}


class TestErrorEvent:
    """Test ErrorEvent dataclass."""

    def test_creation(self):
        event = ErrorEvent(error="Parse failed")
        assert event.error == "Parse failed"


class TestResponseMetadata:
    """Test ResponseMetadata dataclass."""

    def test_minimal_creation(self):
        meta = ResponseMetadata(finish_reason="stop")
        assert meta.finish_reason == "stop"
        assert meta.schema_tokens == 0
        assert meta.schema_injection is None
        assert meta.grammar_gbnf is None
        assert meta.grammar_trace is None
        assert meta.prefill_success is None

    def test_full_creation(self):
        meta = ResponseMetadata(
            finish_reason="length",
            schema_tokens=42,
            schema_injection="<schema>",
            grammar_gbnf="root ::= ...",
            grammar_trace=["step1", "step2"],
            prefill_success=True,
        )
        assert meta.finish_reason == "length"
        assert meta.schema_tokens == 42
        assert meta.schema_injection == "<schema>"
        assert meta.grammar_gbnf == "root ::= ..."
        assert meta.grammar_trace == ["step1", "step2"]
        assert meta.prefill_success is True

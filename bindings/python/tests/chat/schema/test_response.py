"""Tests for Response.parsed and error handling with data salvage."""

import pytest
from pydantic import BaseModel, Field, field_validator

from talu.chat.response import Response
from talu.exceptions import IncompleteJSONError, SchemaValidationError


class Weather(BaseModel):
    location: str
    temperature: float


class ValidatedModel(BaseModel):
    age: int = Field(ge=0, le=150)

    @field_validator("age")
    @classmethod
    def check_reasonable(cls, v):
        if v > 120:
            raise ValueError("Unrealistic age")
        return v


class TestResponseParsed:
    """Test Response.parsed property."""

    def test_success_pydantic(self):
        resp = Response(
            text='{"location": "NYC", "temperature": 20.5}',
            finish_reason="stop",
            _response_format=Weather,
        )

        result = resp.parsed
        assert isinstance(result, Weather)
        assert result.location == "NYC"
        assert result.temperature == 20.5

    def test_success_dict(self):
        resp = Response(
            text='{"key": "value"}',
            finish_reason="stop",
            _response_format={"type": "json_object"},
        )

        result = resp.parsed
        assert result == {"key": "value"}

    def test_no_format_returns_none(self):
        resp = Response(
            text="Just plain text",
            finish_reason="stop",
            _response_format=None,
        )

        assert resp.parsed is None


class TestSchemaValidationError:
    """Test SchemaValidationError with data salvage capabilities."""

    def test_type_mismatch_preserves_data(self):
        """SchemaValidationError should preserve the raw data for salvage."""
        resp = Response(
            text='{"location": "NYC", "temperature": "hot"}',
            finish_reason="stop",
            _response_format=Weather,
        )

        with pytest.raises(SchemaValidationError) as exc_info:
            _ = resp.parsed

        err = exc_info.value

        assert err.raw_text == '{"location": "NYC", "temperature": "hot"}'
        assert err.partial_data == {"location": "NYC", "temperature": "hot"}
        assert err.validation_error is not None

    def test_custom_validator_failure_with_salvage(self):
        """Custom validator failures should still provide salvageable data."""
        resp = Response(
            text='{"age": 200}',
            finish_reason="stop",
            _response_format=ValidatedModel,
        )

        with pytest.raises(SchemaValidationError) as exc_info:
            _ = resp.parsed

        err = exc_info.value

        assert err.partial_data == {"age": 200}
        assert "Unrealistic age" in str(err.validation_error) or "less than or equal" in str(
            err.validation_error
        )

    def test_data_salvage_workflow(self):
        """Demonstrate the data salvage pattern from the DX feedback."""
        resp = Response(
            text='{"age": 150, "name": "John"}',
            finish_reason="stop",
            _response_format=ValidatedModel,
        )

        try:
            result = resp.parsed
        except SchemaValidationError as e:
            assert e.partial_data == {"age": 150, "name": "John"}
            assert e.validation_error is not None

            if e.partial_data.get("age", 0) > 120:
                e.partial_data["age"] = 120
                result = ValidatedModel(**e.partial_data)

            assert result.age == 120


class TestIncompleteJSONError:
    """Test IncompleteJSONError for truncated output."""

    def test_truncation_detected(self):
        resp = Response(
            text='{"location": "NYC", "temper',
            finish_reason="length",
            _response_format=Weather,
        )

        with pytest.raises(IncompleteJSONError) as exc_info:
            _ = resp.parsed

        assert exc_info.value.partial_text == '{"location": "NYC", "temper'
        assert exc_info.value.finish_reason == "length"

    def test_complete_json_with_length_ok(self):
        """If JSON happens to be complete despite length finish, no error."""
        resp = Response(
            text='{"location": "NYC", "temperature": 20}',
            finish_reason="length",
            _response_format=Weather,
        )

        result = resp.parsed
        assert result.location == "NYC"

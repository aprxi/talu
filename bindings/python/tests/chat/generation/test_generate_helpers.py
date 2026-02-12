"""
Unit tests for shared generation helper functions.

Tests the helper functions in talu.chat._generate without needing
a real model or router.
"""

from unittest.mock import MagicMock, patch

from talu.chat._generate import (
    GenerationContext,
    _build_schema_prompt,
    _get_model_type,
    _make_notify_storage,
    build_response,
    extract_json_from_response,
)


class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response()."""

    def test_no_thinking_block(self):
        """Returns text unchanged when no thinking block."""
        text = '{"value": 42}'
        assert extract_json_from_response(text) == '{"value": 42}'

    def test_with_think_end_tag(self):
        """Extracts JSON after </think> tag."""
        text = '<think>reasoning here</think>{"value": 42}'
        assert extract_json_from_response(text) == '{"value": 42}'

    def test_with_think_end_tag_pipe_variant(self):
        """Extracts JSON after <|/think|> tag."""
        text = '<|think|>reasoning here<|/think|>{"value": 42}'
        assert extract_json_from_response(text) == '{"value": 42}'

    def test_multiple_thinking_blocks(self):
        """Uses last thinking block end."""
        text = '<think>first</think>middle<think>second</think>{"final": true}'
        assert extract_json_from_response(text) == '{"final": true}'

    def test_strips_whitespace_after_think(self):
        """Strips whitespace after thinking block."""
        text = '<think>reasoning</think>  \n  {"value": 42}'
        assert extract_json_from_response(text) == '{"value": 42}'

    def test_empty_after_think(self):
        """Handles empty content after thinking block."""
        text = "<think>reasoning</think>"
        assert extract_json_from_response(text) == ""

    def test_no_closing_bracket_in_tag(self):
        """Handles malformed tag without closing >."""
        # If the tag doesn't have >, it won't find tag_end properly
        # This tests the edge case
        text = '</think{"value": 42}'  # Malformed - no > after /think
        result = extract_json_from_response(text)
        # Since rfind finds '</think' but find('>', think_end) finds the next >
        # which is in the JSON, this is an edge case
        assert "42" in result


class TestGetModelType:
    """Tests for _get_model_type()."""

    def test_returns_none_when_no_response_format(self):
        """Returns None when response_format is None."""
        result = _get_model_type("model", None, "auto")
        assert result is None

    def test_returns_none_when_not_auto_strategy(self):
        """Returns None when schema_strategy is not 'auto'."""
        result = _get_model_type("model", {"type": "object"}, "json_schema")
        assert result is None

    def test_returns_none_when_no_model_name(self):
        """Returns None when model_name is None."""
        result = _get_model_type(None, {"type": "object"}, "auto")
        assert result is None

    def test_returns_none_on_import_error(self):
        """Returns None when describe() raises ImportError."""
        # describe is imported inside the function, so patch at the converter module
        with patch("talu.converter.describe", side_effect=ImportError()):
            result = _get_model_type("model", {"type": "object"}, "auto")
        assert result is None

    def test_returns_none_on_value_error(self):
        """Returns None when describe() raises ValueError."""
        with patch("talu.converter.describe", side_effect=ValueError()):
            result = _get_model_type("model", {"type": "object"}, "auto")
        assert result is None

    def test_returns_model_type_on_success(self):
        """Returns model_type from describe() on success."""
        mock_info = MagicMock()
        mock_info.model_type = "qwen2"
        with patch("talu.converter.describe", return_value=mock_info):
            result = _get_model_type("model", {"type": "object"}, "auto")
        assert result == "qwen2"


class TestBuildSchemaPrompt:
    """Tests for _build_schema_prompt()."""

    def test_returns_none_when_no_response_format(self):
        """Returns (None, 0) when response_format is None."""
        result = _build_schema_prompt(
            "hello", None, True, False, "auto", "model", None, lambda x, _: x
        )
        assert result == (None, 0)

    def test_returns_none_when_inject_disabled(self):
        """Returns (None, 0) when inject_schema_prompt is False."""
        result = _build_schema_prompt(
            "hello", {"type": "object"}, False, False, "auto", "model", None, lambda x, _: x
        )
        assert result == (None, 0)

    def test_returns_none_when_message_is_list(self):
        """Returns (None, 0) when message is not a string."""
        result = _build_schema_prompt(
            [{"role": "user", "content": "hello"}],
            {"type": "object"},
            True,
            False,
            "auto",
            "model",
            None,
            lambda x, _: x,
        )
        assert result == (None, 0)

    def test_returns_none_for_grammar_response_format(self):
        """Returns (None, 0) when response_format is a Grammar."""
        from talu.router.config import Grammar

        # Create a mock Grammar
        mock_grammar = MagicMock(spec=Grammar)

        result = _build_schema_prompt(
            "hello", mock_grammar, True, False, "auto", "model", None, lambda x, _: x
        )
        assert result == (None, 0)

    def test_builds_prompt_for_valid_schema(self):
        """Builds schema prompt for valid inputs."""
        schema = {"type": "object", "properties": {"value": {"type": "integer"}}}

        # Patch at the source modules (lazy imports)
        with patch("talu.router.schema.convert.normalize_response_format", return_value=schema):
            with patch(
                "talu.template.schema.injection.schema_to_prompt_description",
                return_value="Please respond with JSON",
            ):
                result = _build_schema_prompt(
                    "hello",
                    schema,
                    True,
                    False,
                    "auto",
                    None,  # No model = no token counting
                    None,
                    lambda x, _: x,
                )

        assert result[0] == "Please respond with JSON"
        assert result[1] == 0  # No tokens counted without model


class TestMakeNotifyStorage:
    """Tests for _make_notify_storage()."""

    def test_returns_callable(self):
        """Returns a callable."""
        mock_chat = MagicMock()
        callback = _make_notify_storage(mock_chat, "hello")
        assert callable(callback)

    def test_callback_is_noop(self):
        """Callback doesn't raise and has no side effects."""
        mock_chat = MagicMock()
        callback = _make_notify_storage(mock_chat, "hello")
        # Should not raise
        callback("assistant response")


class TestBuildResponse:
    """Tests for build_response()."""

    def test_builds_response_with_minimal_result(self):
        """Builds response from minimal result dict."""
        mock_chat = MagicMock()
        mock_chat._router.default_model = "test-model"

        mock_response_class = MagicMock()

        result = {
            "text": "Hello world",
            "completion_tokens": 5,
            "prompt_tokens": 10,
        }

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
        )

        # Verify response class was called
        mock_response_class.assert_called_once()
        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["text"] == "Hello world"
        assert call_kwargs["usage"].completion_tokens == 5
        assert call_kwargs["usage"].prompt_tokens == 10

    def test_builds_response_with_finish_reason_stop(self):
        """Sets finish_reason to 'stop' when not at max tokens."""
        mock_chat = MagicMock()
        mock_chat._router.default_model = "test-model"

        mock_response_class = MagicMock()

        result = {
            "text": "Hello",
            "completion_tokens": 2,
            "prompt_tokens": 5,
        }

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
        )

        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["finish_reason"] == "stop"

    def test_builds_response_with_finish_reason_length(self):
        """Sets finish_reason to 'length' when at max tokens."""
        mock_chat = MagicMock()
        mock_chat._router.default_model = "test-model"

        mock_response_class = MagicMock()

        result = {
            "text": "Hello",
            "completion_tokens": 100,
            "prompt_tokens": 5,
        }

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
        )

        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["finish_reason"] == "length"

    def test_preserves_finish_reason_from_result(self):
        """Uses finish_reason from result if provided."""
        mock_chat = MagicMock()
        mock_chat._router.default_model = "test-model"

        mock_response_class = MagicMock()

        result = {
            "text": "Hello",
            "completion_tokens": 50,
            "prompt_tokens": 5,
            "finish_reason": "tool_calls",
        }

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
        )

        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["finish_reason"] == "tool_calls"

    def test_uses_token_count_fallback(self):
        """Uses 'token_count' when 'completion_tokens' not present."""
        mock_chat = MagicMock()
        mock_chat._router.default_model = "test-model"

        mock_response_class = MagicMock()

        result = {
            "text": "Hello",
            "token_count": 42,  # Old field name
            "prompt_tokens": 5,
        }

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
        )

        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["usage"].completion_tokens == 42

    def test_handles_no_router(self):
        """Handles chat with no router."""
        mock_chat = MagicMock()
        mock_chat._router = None

        mock_response_class = MagicMock()

        result = {
            "text": "Hello",
            "completion_tokens": 5,
            "prompt_tokens": 10,
        }

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
        )

        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["model"] is None

    def test_passes_response_format(self):
        """Passes _response_format through."""
        mock_chat = MagicMock()
        mock_chat._router.default_model = "test-model"

        mock_response_class = MagicMock()
        response_format = {"type": "object"}

        result = {"text": "Hello", "completion_tokens": 5, "prompt_tokens": 10}

        build_response(
            mock_chat,
            result,
            MagicMock(max_tokens=100),
            mock_response_class,
            _response_format=response_format,
        )

        call_kwargs = mock_response_class.call_args.kwargs
        assert call_kwargs["_response_format"] == response_format


class TestGenerationContext:
    """Tests for GenerationContext dataclass."""

    def test_creation(self):
        """GenerationContext can be created with all fields."""
        ctx = GenerationContext(
            effective_config=MagicMock(),
            hooks=None,
            generation_start_time=1.0,
            allow_thinking=False,
            max_thinking_tokens=0,
            inject_schema_prompt=True,
            schema_strategy="auto",
            model_name="test",
            model_type="llama",
            schema_prompt="respond with JSON",
            schema_tokens=10,
            messages_for_submit=None,
            stop_tokens=set(),
            prefill_prefix=None,
            grammar_cleanup=None,
            actual_response_format=None,
            use_submit=False,
            notify_storage=lambda x: None,
        )
        assert ctx.model_name == "test"
        assert ctx.allow_thinking is False

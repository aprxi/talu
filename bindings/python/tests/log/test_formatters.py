"""
Tests for OpenTelemetry-compliant log formatters.

Tests for JsonFormatter, HumanFormatter, and scoped_logger.
"""

import json
import logging
import os
from io import StringIO
from unittest.mock import patch


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def test_produces_valid_json(self):
        """JsonFormatter output is valid JSON."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert isinstance(parsed, dict)

    def test_otel_structure(self):
        """Output follows OpenTelemetry Logging Data Model."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        # Required OpenTelemetry fields
        assert "timestamp" in parsed
        assert "severityText" in parsed
        assert "body" in parsed
        assert "attributes" in parsed
        assert "resource" in parsed

    def test_timestamp_format(self):
        """Timestamp is RFC3339 with nanoseconds."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        # Should look like: 2026-01-09T11:10:47.008550065Z
        ts = parsed["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts
        assert len(ts.split(".")[-1]) == 10  # 9 digits + Z

    def test_severity_text_mapping(self):
        """Python log levels map to OpenTelemetry severity text."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()

        test_cases = [
            (logging.DEBUG, "DEBUG"),
            (logging.INFO, "INFO"),
            (logging.WARNING, "WARN"),
            (logging.ERROR, "ERROR"),
            (logging.CRITICAL, "FATAL"),
        ]

        for level, expected_severity in test_cases:
            record = logging.LogRecord(
                name="talu.test",
                level=level,
                pathname="test.py",
                lineno=42,
                msg="Test",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["severityText"] == expected_severity, f"Level {level}"

    def test_body_contains_message(self):
        """Body contains the log message."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Hello %s",
            args=("world",),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["body"] == "Hello world"

    def test_scope_from_extra(self):
        """Scope comes from extra dict if provided."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.scope = "fetch"

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["attributes"]["scope"] == "fetch"

    def test_extra_attributes_included(self):
        """Extra attributes are included in attributes dict."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.model_id = "Foo/Bar-0B"
        record.size_bytes = 12345

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["attributes"]["model_id"] == "Foo/Bar-0B"
        assert parsed["attributes"]["size_bytes"] == 12345

    def test_resource_contains_service_info(self):
        """Resource contains service name and version."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["resource"]["service.name"] == "talu"
        assert "service.version" in parsed["resource"]

    def test_code_location_for_debug(self):
        """DEBUG level includes code location."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.DEBUG,
            pathname="/path/to/talu/module.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "code.filepath" in parsed["attributes"]
        assert parsed["attributes"]["code.lineno"] == 42

    def test_code_location_for_error(self):
        """ERROR level includes code location."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.ERROR,
            pathname="/path/to/talu/module.py",
            lineno=99,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "code.filepath" in parsed["attributes"]
        assert parsed["attributes"]["code.lineno"] == 99

    def test_no_code_location_for_info(self):
        """INFO level does not include code location."""
        from talu._logging import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="/path/to/talu/module.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "code.filepath" not in parsed["attributes"]
        assert "code.lineno" not in parsed["attributes"]


class TestHumanFormatter:
    """Tests for HumanFormatter."""

    def test_format_basic(self):
        """HumanFormatter produces readable output."""
        from talu._logging import HumanFormatter

        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        assert "INFO" in output
        assert "Test message" in output

    def test_time_format(self):
        """Time is formatted as HH:MM:SS."""
        from talu._logging import HumanFormatter

        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should start with time like "12:34:56"
        time_part = output.split()[0]
        assert len(time_part) == 8
        assert time_part.count(":") == 2

    def test_scope_in_brackets(self):
        """Scope appears in square brackets."""
        from talu._logging import HumanFormatter

        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.scope = "fetch"

        output = formatter.format(record)

        assert "[fetch]" in output

    def test_model_id_in_parentheses(self):
        """model_id appears in parentheses."""
        from talu._logging import HumanFormatter

        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="talu.test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Using cached model",
            args=(),
            exc_info=None,
        )
        record.model_id = "Foo/Bar-0B"

        output = formatter.format(record)

        assert "(Foo/Bar-0B)" in output

    def test_colors_disabled(self):
        """Colors can be disabled."""
        from talu._logging import HumanFormatter

        formatter = HumanFormatter(use_colors=False)
        record = logging.LogRecord(
            name="talu.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # No ANSI escape codes
        assert "\x1b[" not in output

    def test_colors_enabled(self):
        """Colors are included when enabled."""
        from talu._logging import HumanFormatter

        formatter = HumanFormatter(use_colors=True)
        record = logging.LogRecord(
            name="talu.test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        output = formatter.format(record)

        # Should have ANSI escape codes
        assert "\x1b[" in output


class TestScopedLogger:
    """Tests for scoped_logger."""

    def test_creates_logger_adapter(self):
        """scoped_logger returns a LoggerAdapter."""
        from talu._logging import scoped_logger

        log = scoped_logger("test")

        assert isinstance(log, logging.LoggerAdapter)

    def test_scope_in_extra(self):
        """scoped_logger includes scope in all messages."""
        from talu._logging import JsonFormatter, scoped_logger

        log = scoped_logger("fetch")

        # Create a handler to capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        log.logger.addHandler(handler)
        old_level = log.logger.level
        log.logger.setLevel(logging.INFO)

        try:
            log.info("Test message")

            output = stream.getvalue()
            parsed = json.loads(output.strip())

            assert parsed["attributes"]["scope"] == "fetch"
        finally:
            log.logger.removeHandler(handler)
            log.logger.setLevel(old_level)

    def test_extra_merges_with_scope(self):
        """Extra attributes merge with scope."""
        from talu._logging import JsonFormatter, scoped_logger

        log = scoped_logger("fetch")

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        log.logger.addHandler(handler)
        old_level = log.logger.level
        log.logger.setLevel(logging.INFO)

        try:
            log.info("Test message", extra={"model_id": "test-model"})

            output = stream.getvalue()
            parsed = json.loads(output.strip())

            # Both scope and model_id should be present
            assert parsed["attributes"]["scope"] == "fetch"
            assert parsed["attributes"]["model_id"] == "test-model"
        finally:
            log.logger.removeHandler(handler)
            log.logger.setLevel(old_level)


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_talu_log_level_debug(self):
        """TALU_LOG_LEVEL=debug sets DEBUG level."""
        from talu._logging import _get_log_level

        with patch.dict(os.environ, {"TALU_LOG_LEVEL": "debug"}):
            level = _get_log_level()
            assert level == logging.DEBUG

    def test_talu_log_level_warn(self):
        """TALU_LOG_LEVEL=warn sets WARNING level."""
        from talu._logging import _get_log_level

        with patch.dict(os.environ, {"TALU_LOG_LEVEL": "warn"}):
            level = _get_log_level()
            assert level == logging.WARNING

    def test_talu_log_level_off(self):
        """TALU_LOG_LEVEL=off disables logging."""
        from talu._logging import _get_log_level

        with patch.dict(os.environ, {"TALU_LOG_LEVEL": "off"}):
            level = _get_log_level()
            assert level > logging.CRITICAL

    def test_talu_log_format_json(self):
        """TALU_LOG_FORMAT=json returns json."""
        from talu._logging import _get_log_format

        with patch.dict(os.environ, {"TALU_LOG_FORMAT": "json"}):
            fmt = _get_log_format()
            assert fmt == "json"

    def test_talu_log_format_human(self):
        """TALU_LOG_FORMAT=human returns human."""
        from talu._logging import _get_log_format

        with patch.dict(os.environ, {"TALU_LOG_FORMAT": "human"}):
            fmt = _get_log_format()
            assert fmt == "human"

    def test_legacy_talu_log_env(self):
        """Legacy TALU_LOG env var still works."""
        from talu._logging import _get_log_level

        with patch.dict(os.environ, {"TALU_LOG": "error"}, clear=False):
            # Remove TALU_LOG_LEVEL if present
            env = dict(os.environ)
            env.pop("TALU_LOG_LEVEL", None)
            env["TALU_LOG"] = "error"
            with patch.dict(os.environ, env, clear=True):
                level = _get_log_level()
                assert level == logging.ERROR


class TestSetupLoggingFormat:
    """Tests for setup_logging with format parameter."""

    def test_setup_logging_with_json_format(self):
        """setup_logging(format='json') uses JsonFormatter."""
        from talu._logging import JsonFormatter, logger, setup_logging

        setup_logging("INFO", format="json")

        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JsonFormatter)

    def test_setup_logging_with_human_format(self):
        """setup_logging(format='human') uses HumanFormatter."""
        from talu._logging import HumanFormatter, logger, setup_logging

        setup_logging("INFO", format="human")

        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, HumanFormatter)

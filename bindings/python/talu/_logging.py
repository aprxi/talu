"""
Structured logging (OpenTelemetry-compliant).

Produces structured log output following the OpenTelemetry Logging Data Model.
Both Python and Zig (core) produce identical output formats.

Usage::

    from ._logging import logger

    # Info message (no code location)
    logger.info("Fetching file list", extra={"scope": "fetch", "model_id": model_id})

    # Error message (includes code location automatically)
    logger.error("Model not found", extra={"scope": "fetch", "model_id": model_id})

Environment::

    TALU_LOG_LEVEL=trace|debug|info|warn|error|fatal|off (default: info)
    TALU_LOG_FORMAT=json|human (default: human if tty, json if piped)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import MutableMapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from typing import Any

__all__ = ["logger", "setup_logging", "scoped_logger"]

# =============================================================================
# Version
# =============================================================================


def _get_version() -> str:
    """Get talu version from package metadata."""
    try:
        return get_version("talu")
    except (ImportError, PackageNotFoundError, AttributeError):
        return "0.0.0"


# =============================================================================
# Level Mapping
# =============================================================================

# Map Python levels to OpenTelemetry severity text
_LEVEL_TO_SEVERITY = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "FATAL",
}

# Map string names to Python levels (case-insensitive)
_NAME_TO_LEVEL = {
    "trace": logging.DEBUG,  # Python doesn't have TRACE, use DEBUG
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "err": logging.ERROR,
    "fatal": logging.CRITICAL,
    "critical": logging.CRITICAL,
    "off": logging.CRITICAL + 10,  # Higher than any level
    "none": logging.CRITICAL + 10,
}

# Levels that include code location
_CODE_LOCATION_LEVELS = {logging.DEBUG, logging.ERROR, logging.CRITICAL}


def _infer_scope(logger_name: str) -> str:
    """Infer scope from logger name when not explicitly provided."""
    if "fetch" in logger_name or "hf" in logger_name or "download" in logger_name:
        return "fetch"
    if "load" in logger_name or "model" in logger_name:
        return "load"
    if "token" in logger_name:
        return "tokenizer"
    if "template" in logger_name or "chat" in logger_name:
        return "template"
    if "infer" in logger_name or "generate" in logger_name:
        return "inference"
    if "convert" in logger_name:
        return "converter"
    if "cli" in logger_name or "__main__" in logger_name:
        return "cli"
    return logger_name.split(".")[-1] if logger_name else "talu"


def _strip_path_prefix(filepath: str) -> str:
    """Strip common prefixes from filepath for cleaner log output."""
    for prefix in ("talu/", "src/"):
        if prefix in filepath:
            return filepath[filepath.index(prefix) + len(prefix) :]
    return filepath


# =============================================================================
# Formatters
# =============================================================================


class JsonFormatter(logging.Formatter):
    """OpenTelemetry-compliant JSON formatter."""

    def __init__(self) -> None:
        super().__init__()
        self._version = _get_version()

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp with nanosecond precision (RFC3339)
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        # Python's timestamp has microsecond precision, pad to nanoseconds
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(dt.microsecond * 1000):09d}Z"

        # Severity
        severity = _LEVEL_TO_SEVERITY.get(record.levelno, "INFO")

        # Build attributes
        attributes: dict[str, Any] = {}

        # Scope from extra or infer from logger name
        scope = getattr(record, "scope", None) or _infer_scope(record.name)
        attributes["scope"] = scope

        # Add extra attributes (excluding standard LogRecord fields)
        _standard_fields = {
            "name",
            "msg",
            "args",
            "created",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "exc_info",
            "exc_text",
            "thread",
            "threadName",
            "taskName",
            "scope",
            "message",
        }
        for key, value in record.__dict__.items():
            if key not in _standard_fields and not key.startswith("_"):
                attributes[key] = value

        # Code location for DEBUG/ERROR/FATAL
        if record.levelno in _CODE_LOCATION_LEVELS:
            attributes["code.filepath"] = _strip_path_prefix(record.pathname)
            attributes["code.lineno"] = record.lineno

        # Build the record
        log_record = {
            "timestamp": timestamp,
            "severityText": severity,
            "body": record.getMessage(),
            "attributes": attributes,
            "resource": {
                "service.name": "talu",
                "service.version": self._version,
            },
        }

        return json.dumps(log_record, separators=(",", ":"))


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for terminal output."""

    # ANSI color codes
    _RESET = "\x1b[0m"
    _DIM = "\x1b[2m"
    _RED = "\x1b[31m"
    _YELLOW = "\x1b[33m"
    _CYAN = "\x1b[36m"

    def __init__(self, use_colors: bool = True) -> None:
        super().__init__()
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        # Time (HH:MM:SS)
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        time_str = dt.strftime("%H:%M:%S")

        # Level with padding
        severity = _LEVEL_TO_SEVERITY.get(record.levelno, "INFO")

        # Color for level
        if self._use_colors:
            if record.levelno <= logging.DEBUG:
                level_color = self._DIM
            elif record.levelno >= logging.ERROR:
                level_color = self._RED
            elif record.levelno >= logging.WARNING:
                level_color = self._YELLOW
            else:
                level_color = ""
        else:
            level_color = ""

        # Scope from extra or infer from logger name
        scope = getattr(record, "scope", None) or _infer_scope(record.name)

        # Build message
        parts = [time_str, " "]

        # Level with color
        if level_color:
            parts.append(level_color)
        parts.append(f"{severity:<5} ")
        if level_color:
            parts.append(self._RESET)

        # Scope with color
        if self._use_colors:
            parts.append(self._CYAN)
        parts.append(f"[{scope}] ")
        if self._use_colors:
            parts.append(self._RESET)

        # Message body
        parts.append(record.getMessage())

        # Key attributes inline (model_id shown in parentheses)
        model_id = getattr(record, "model_id", None)
        if model_id:
            parts.append(f" ({model_id})")

        # Code location for DEBUG/ERROR/FATAL
        if record.levelno in _CODE_LOCATION_LEVELS:
            filepath = _strip_path_prefix(record.pathname)
            if self._use_colors:
                parts.append(self._DIM)
            parts.append(f" [{filepath}:{record.lineno}]")
            if self._use_colors:
                parts.append(self._RESET)

        return "".join(parts)


# =============================================================================
# Logger Setup
# =============================================================================


def _get_log_level() -> int:
    """Get log level from environment."""
    level_name = os.environ.get("TALU_LOG_LEVEL") or os.environ.get("TALU_LOG", "info")
    return _NAME_TO_LEVEL.get(level_name.lower(), logging.INFO)


def _get_log_format() -> str:
    """Get log format from environment or auto-detect."""
    fmt = os.environ.get("TALU_LOG_FORMAT")
    if fmt:
        return fmt.lower()
    # Auto-detect: human for TTY, json for pipe
    return "human" if sys.stderr.isatty() else "json"


def _create_handler() -> logging.Handler:
    """Create appropriate handler based on format."""
    handler = logging.StreamHandler(sys.stderr)
    fmt = _get_log_format()

    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        use_colors = sys.stderr.isatty()
        handler.setFormatter(HumanFormatter(use_colors=use_colors))

    return handler


# Single logger for all of talu
logger = logging.getLogger("talu")


def _setup_default_handler() -> None:
    """Configure default logging based on environment."""
    # Don't add handler if user already configured logging
    if logger.handlers:
        return

    handler = _create_handler()
    logger.addHandler(handler)
    logger.setLevel(_get_log_level())


def setup_logging(
    level: str | int = "INFO",
    format: str | None = None,
) -> None:
    """
    Configure talu logging.

    Parameters
    ----------
    level : str or int, default "INFO"
        Log level. Can be "DEBUG", "INFO", "WARNING", "ERROR", "FATAL",
        or a logging constant like ``logging.DEBUG``.

    format : str, optional
        Log format. Either "json" or "human". If not specified,
        uses TALU_LOG_FORMAT env var or auto-detects based on TTY.

    Examples
    --------
    JSON output for machine parsing::

        >>> import talu
        >>> talu.setup_logging("INFO", format="json")

    Human-readable output::

        >>> talu.setup_logging("DEBUG", format="human")
    """
    # Parse level
    if isinstance(level, str):
        level = _NAME_TO_LEVEL.get(level.lower(), logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set format in environment for child processes
    if format:
        os.environ["TALU_LOG_FORMAT"] = format

    # Add new handler
    handler = _create_handler()
    logger.addHandler(handler)
    logger.setLevel(level)


class _ScopedLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that merges extra attributes with scope."""

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> tuple[str, MutableMapping[str, Any]]:
        # Merge extra from call with our defaults (scope)
        extra: dict[str, Any] = dict(self.extra) if self.extra else {}
        if "extra" in kwargs:
            extra.update(kwargs["extra"])
        kwargs["extra"] = extra
        return msg, kwargs


def scoped_logger(scope: str) -> logging.LoggerAdapter:
    """
    Create a logger adapter with a fixed scope.

    This is useful for modules that always log with the same scope.

    Parameters
    ----------
    scope : str
        The scope name (e.g., "fetch", "load", "tokenizer").

    Returns
    -------
    logging.LoggerAdapter
        A logger adapter that automatically adds scope to all messages.

    Examples
    --------
    ::

        from talu._logging import scoped_logger
        log = scoped_logger("fetch")
        log.info("Downloading model", extra={"model_id": "Qwen/Qwen3-0.6B"})
    """
    return _ScopedLoggerAdapter(logger, {"scope": scope})


# Initialize on import
_setup_default_handler()

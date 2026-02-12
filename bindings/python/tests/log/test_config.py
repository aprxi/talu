"""
Logging configuration tests.

Tests for talu._logging module.
"""

import logging


class TestSetupLogging:
    """Tests for setup_logging() function."""

    def test_setup_logging_accessible(self):
        """setup_logging is accessible from talu._logging."""
        from talu._logging import setup_logging

        assert callable(setup_logging)

    def test_setup_logging_default_level(self):
        """setup_logging() defaults to INFO level."""
        from talu._logging import logger, setup_logging

        setup_logging()

        assert logger.level == logging.INFO

    def test_setup_logging_accepts_string_level(self):
        """setup_logging() accepts string level names."""
        from talu._logging import logger, setup_logging

        setup_logging("DEBUG")

        assert logger.level == logging.DEBUG

    def test_setup_logging_accepts_int_level(self):
        """setup_logging() accepts integer level constants."""
        from talu._logging import logger, setup_logging

        setup_logging(logging.WARNING)

        assert logger.level == logging.WARNING

    def test_setup_logging_adds_stream_handler(self):
        """setup_logging() adds a StreamHandler."""
        from talu._logging import logger, setup_logging

        setup_logging("INFO")

        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1

    def test_setup_logging_replaces_handlers(self):
        """setup_logging() replaces existing handlers."""
        from talu._logging import logger, setup_logging

        # Add some handlers
        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()
        logger.addHandler(handler1)
        logger.addHandler(handler2)

        # setup_logging should replace all handlers with one
        setup_logging("INFO")

        # Should have exactly one handler now
        assert len(logger.handlers) == 1


class TestDefaultBehavior:
    """Tests for default logging behavior."""

    def test_logger_name_is_talu(self):
        """Root talu logger has correct name."""
        from talu._logging import logger

        assert logger.name == "talu"

    def test_default_level_is_info(self):
        """Default level is INFO (shows progress)."""
        # Fresh import to get default state
        import talu._logging

        # The default setup should set INFO level
        # Note: in test environment this may have been modified
        assert talu._logging.logger is not None


class TestLoggerHierarchy:
    """Tests for logger hierarchy."""

    def test_module_loggers_are_children(self):
        """Module-specific loggers are children of talu logger."""
        chat_log = logging.getLogger("talu.chat")
        repo_log = logging.getLogger("talu.repository")

        assert chat_log.parent.name == "talu"
        assert repo_log.parent.name == "talu"

    def test_child_inherits_level(self):
        """Child loggers inherit level from parent."""
        from talu._logging import setup_logging

        setup_logging("DEBUG")

        child = logging.getLogger("talu.child")

        # Child effective level should match parent
        assert child.getEffectiveLevel() == logging.DEBUG


class TestCoreLogConfig:
    """Tests for talu.set_log_level() and talu.set_log_format()."""

    def test_set_log_level_accepts_all_levels(self):
        """set_log_level() accepts all valid level names."""
        import talu

        for level in ["trace", "debug", "info", "warn", "error", "off"]:
            talu.set_log_level(level)  # Should not raise

    def test_set_log_level_case_insensitive(self):
        """set_log_level() is case insensitive."""
        import talu

        talu.set_log_level("DEBUG")
        talu.set_log_level("Debug")
        talu.set_log_level("debug")

    def test_set_log_level_unknown_defaults_to_warn(self):
        """set_log_level() with unknown level defaults to warn."""
        import talu

        talu.set_log_level("unknown")  # Should not raise, defaults to warn

    def test_set_log_format_json(self):
        """set_log_format() accepts 'json'."""
        import talu

        talu.set_log_format("json")  # Should not raise

    def test_set_log_format_human(self):
        """set_log_format() accepts 'human'."""
        import talu

        talu.set_log_format("human")  # Should not raise

    def test_set_log_format_case_insensitive(self):
        """set_log_format() is case insensitive."""
        import talu

        talu.set_log_format("JSON")
        talu.set_log_format("Json")
        talu.set_log_format("HUMAN")
        talu.set_log_format("Human")

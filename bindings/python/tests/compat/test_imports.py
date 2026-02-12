"""
Python 3.10 import compatibility tests.

These tests verify that all talu modules can be imported on Python 3.10+.
This catches accidental use of Python 3.11+ features like:
- tomllib (3.11+)
- ExceptionGroup (3.11+)
- typing.Self (3.11+)
- asyncio.TaskGroup (3.11+)
- match statement improvements (3.10 has basic match, but some patterns are 3.11+)

These tests run in both test-api (Python 3.10) and test-all (Python 3.14).
"""

import sys


class TestTaluImports:
    """Test that all talu modules import successfully."""

    def test_import_talu(self):
        """Import main talu package."""
        import talu

        assert talu is not None

    def test_import_public_api(self):
        """Import all public API classes."""
        from talu import Chat, Client

        assert Chat is not None
        assert Client is not None

    def test_import_chat_session(self):
        """Import chat subpackage."""
        from talu.chat import Chat, Response

        assert Chat is not None
        assert Response is not None

    def test_import_generation_config(self):
        """Import generation config."""
        from talu.router import GenerationConfig

        assert GenerationConfig is not None

    def test_import_tokenizer(self):
        """Import tokenizer module."""
        from talu.tokenizer import Tokenizer

        assert Tokenizer is not None

    def test_import_repository(self):
        """Import repository module."""
        from talu import repository

        assert repository is not None

    def test_import_lib(self):
        """Import internal library loader."""
        from talu._bindings import get_lib

        assert get_lib is not None


class TestNoForbiddenImports:
    """Verify no 3.11+ stdlib modules are imported."""

    def test_no_wsgiref_types(self):
        """wsgiref.types is 3.11+ only."""
        import talu  # noqa: F401

        assert "wsgiref.types" not in sys.modules, "talu should not import wsgiref.types (3.11+)"

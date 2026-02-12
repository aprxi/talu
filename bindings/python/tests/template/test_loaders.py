"""Tests for talu.template.loaders module.

Tests for template loading utilities without requiring model inference.
Uses mocking to test error paths and edge cases.
"""

from unittest.mock import MagicMock, patch

import pytest

from talu.exceptions import TemplateNotFoundError
from talu.template.loaders import get_chat_template_source, resolve_model_path


class TestGetChatTemplateSource:
    """Tests for get_chat_template_source function."""

    def test_returns_source_when_found(self):
        """Returns template source when found."""
        expected_source = "{% for message in messages %}{{ message['content'] }}{% endfor %}"

        with patch("talu.template.loaders._c.get_chat_template_source") as mock:
            mock.return_value = expected_source

            result = get_chat_template_source("/path/to/model")

            assert result == expected_source
            mock.assert_called_once_with("/path/to/model")

    def test_raises_template_not_found_when_none(self):
        """Raises TemplateNotFoundError when C API returns None."""
        with patch("talu.template.loaders._c.get_chat_template_source") as mock:
            mock.return_value = None

            with pytest.raises(TemplateNotFoundError) as exc_info:
                get_chat_template_source("/path/to/model")

            assert "No chat template found" in str(exc_info.value)
            assert "/path/to/model" in str(exc_info.value)
            assert "tokenizer_config.json" in str(exc_info.value)
            assert "chat_template.jinja" in str(exc_info.value)

    def test_error_message_includes_model_path(self):
        """Error message includes the model path for debugging."""
        model_path = "/home/user/models/my-custom-model"

        with patch("talu.template.loaders._c.get_chat_template_source") as mock:
            mock.return_value = None

            with pytest.raises(TemplateNotFoundError) as exc_info:
                get_chat_template_source(model_path)

            assert model_path in str(exc_info.value)


class TestResolveModelPath:
    """Tests for resolve_model_path function."""

    def test_returns_path_when_repository_resolves(self):
        """Returns resolved path when resolve_path succeeds."""
        expected_path = "/home/user/.cache/talu/models/Foo/Bar-0B"

        with patch("talu.repository.resolve_path", return_value=expected_path) as mock:
            result = resolve_model_path("Foo/Bar-0B")

            assert result == expected_path
            mock.assert_called_once_with("Foo/Bar-0B", offline=True)

    def test_raises_file_not_found_on_import_error(self):
        """Raises FileNotFoundError when repository import fails."""
        import sys

        original_modules = sys.modules.copy()

        modules_to_remove = [k for k in sys.modules if k.startswith("talu.repository")]
        for mod in modules_to_remove:
            sys.modules.pop(mod, None)

        try:
            with patch.dict(sys.modules, {"talu.repository": None}):
                original_import = (
                    __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
                )

                def mock_import(name, *args, **kwargs):
                    if "repository" in name:
                        raise ImportError("mocked import error")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    with pytest.raises(FileNotFoundError) as exc_info:
                        resolve_model_path("some/model")

                    assert "some/model" in str(exc_info.value)
                    assert "not found" in str(exc_info.value)
        finally:
            sys.modules.update(original_modules)

    def test_raises_file_not_found_on_os_error(self):
        """Raises FileNotFoundError when resolve_path raises OSError."""
        with patch("talu.repository.resolve_path", side_effect=OSError("Path not found")):
            with pytest.raises(FileNotFoundError) as exc_info:
                resolve_model_path("nonexistent/model")

            assert "nonexistent/model" in str(exc_info.value)
            assert "not found" in str(exc_info.value)

    def test_error_message_includes_download_hint(self):
        """Error message includes hint about downloading with talu get."""
        model_id = "meta-llama/Llama-2-7b"

        with patch("talu.repository.resolve_path", side_effect=OSError("Path not found")):
            with pytest.raises(FileNotFoundError) as exc_info:
                resolve_model_path(model_id)

            error_msg = str(exc_info.value)
            assert model_id in error_msg
            assert f"talu get {model_id}" in error_msg

    def test_accepts_local_path_when_repository_succeeds(self):
        """Local paths work when resolve_path returns them."""
        local_path = "/home/user/my-local-model"

        with patch("talu.repository.resolve_path", return_value=local_path):
            result = resolve_model_path(local_path)

            assert result == local_path

    def test_uses_offline_mode(self):
        """Always uses offline=True to avoid network access."""
        with patch("talu.repository.resolve_path", return_value="/some/path") as mock:
            resolve_model_path("model/id")

            mock.assert_called_with("model/id", offline=True)

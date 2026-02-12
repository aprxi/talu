"""
Error path cleanup tests.

Verifies that resources are properly cleaned up when:
1. Operations fail mid-way
2. Exceptions are raised
3. Invalid input is provided
"""

from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM


class TestTokenizerErrorCleanup:
    """Tokenizer error path cleanup."""

    def test_invalid_model_path_no_leak(self, memory_tracker):
        """Invalid model path doesn't leak memory."""
        from talu import Tokenizer
        from talu.exceptions import TaluError

        memory_tracker.capture_baseline()

        for _ in range(50):
            try:
                Tokenizer("/nonexistent/path/model.gguf")
            except (TaluError, FileNotFoundError, OSError):
                pass

        memory_tracker.assert_no_leak(threshold_mb=2, context="50 invalid tokenizer path attempts")

    def test_encode_empty_string_no_leak(self, memory_tracker):
        """Encoding empty string doesn't leak."""
        from talu import Tokenizer

        tokenizer = Tokenizer(TEST_MODEL_URI_TEXT_RANDOM)

        memory_tracker.capture_baseline()

        for _ in range(100):
            result = tokenizer.encode("")
            del result

        memory_tracker.assert_no_leak(threshold_mb=2, context="100 empty string encodes")

    def test_decode_empty_list_no_leak(self, memory_tracker):
        """Decoding empty list doesn't leak."""
        from talu import Tokenizer

        tokenizer = Tokenizer(TEST_MODEL_URI_TEXT_RANDOM)

        memory_tracker.capture_baseline()

        for _ in range(100):
            result = tokenizer.decode([])
            del result

        memory_tracker.assert_no_leak(threshold_mb=2, context="100 empty list decodes")


class TestChatErrorCleanup:
    """Chat error path cleanup."""

    def test_closed_chat_operations_no_leak(self, memory_tracker):
        """Operations on closed chat don't leak."""
        from talu import Chat
        from talu.exceptions import StateError

        memory_tracker.capture_baseline()

        for _ in range(100):
            chat = Chat()
            chat.close()

            try:
                chat.append("user", "test")
            except StateError:
                pass

            try:
                _ = chat.items
            except StateError:
                pass

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 closed chat operation attempts")

    def test_chat_close_idempotent_no_leak(self, memory_tracker):
        """Multiple close() calls don't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        for _ in range(100):
            chat = Chat()
            chat.append("user", "hello")

            # Close multiple times
            chat.close()
            chat.close()
            chat.close()

        memory_tracker.assert_no_leak(
            threshold_mb=5, context="100 chat instances with multiple close()"
        )

    def test_chat_context_manager_exception_no_leak(self, memory_tracker):
        """Exception in context manager doesn't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        for _ in range(100):
            try:
                with Chat() as chat:
                    chat.append("user", "hello")
                    raise ValueError("Intentional error")
            except ValueError:
                pass

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 chat context manager exceptions")


class TestTemplateErrorCleanup:
    """Template error path cleanup."""

    def test_invalid_template_syntax_no_leak(self, memory_tracker):
        """Invalid template syntax doesn't leak."""
        from talu.exceptions import TemplateError
        from talu.template import PromptTemplate

        memory_tracker.capture_baseline()

        for _ in range(100):
            try:
                PromptTemplate("{{ unclosed")
            except TemplateError:
                pass

        memory_tracker.assert_no_leak(
            threshold_mb=2, context="100 invalid template syntax attempts"
        )

    def test_missing_variable_no_leak(self, memory_tracker):
        """Missing template variable doesn't leak."""
        from talu.exceptions import TemplateError
        from talu.template import PromptTemplate

        template = PromptTemplate("Hello {{ name }}!")

        memory_tracker.capture_baseline()

        for _ in range(100):
            try:
                # Render without required variable (strict mode)
                template.render()
            except (TemplateError, KeyError, TypeError):
                pass

        memory_tracker.assert_no_leak(threshold_mb=2, context="100 missing variable renders")


class TestRepositoryErrorCleanup:
    """Repository error path cleanup."""

    def test_repository_list_cleanup(self, memory_tracker):
        """Repository list operations cleanup properly."""
        from talu.repository import (
    cache_dir,
    cache_path,
    clear,
    delete,
    fetch,
    fetch_file,
    is_cached,
    is_model_id,
    list_files,
    list_models,
    resolve_path,
    search,
    size,
)

        memory_tracker.capture_baseline()

        for _ in range(50):
            # Iterate and discard
            models = list(list_models())
            del models

        memory_tracker.assert_no_leak(threshold_mb=2, context="50 repository list operations")


class TestConfigErrorCleanup:
    """Config error path cleanup."""

    def test_invalid_config_values_no_leak(self, memory_tracker):
        """Invalid config values don't leak."""
        from talu import GenerationConfig

        memory_tracker.capture_baseline()

        for _ in range(200):
            # Create configs with edge case values
            config = GenerationConfig(
                temperature=-1.0,  # Invalid but may be accepted
                max_tokens=0,
                stop_sequences=[],
                logit_bias={},
            )
            del config

        memory_tracker.assert_no_leak(threshold_mb=2, context="200 edge case configs")

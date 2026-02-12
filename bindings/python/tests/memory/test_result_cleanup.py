"""
Result cleanup completeness tests.

Verifies that every talu_* function that returns allocated memory
has a corresponding free function that is called.
"""

import gc

import pytest

from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM


@pytest.fixture
def tokenizer():
    """Create tokenizer for result tests."""

    from talu import Tokenizer

    return Tokenizer(TEST_MODEL_URI_TEXT_RANDOM)


class TestTokenizerResultCleanup:
    """Tokenizer result cleanup tests."""

    def test_encode_result_freed(self, tokenizer, memory_tracker):
        """Encode result memory is freed."""
        memory_tracker.capture_baseline()

        # Many encode operations
        for i in range(100):
            result = tokenizer.encode(f"test string number {i}" * 10)
            del result

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 encode results")

    def test_decode_result_freed(self, tokenizer, memory_tracker):
        """Decode result memory is freed."""
        tokens = tokenizer.encode("hello world")

        memory_tracker.capture_baseline()

        for _ in range(100):
            text = tokenizer.decode(tokens)
            del text

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 decode results")

    def test_batch_encode_result_freed(self, tokenizer, memory_tracker):
        """Batch encode result memory is freed."""
        memory_tracker.capture_baseline()

        texts = [f"text {i}" for i in range(10)]
        for _ in range(50):
            batch = tokenizer.encode(texts)
            del batch

        memory_tracker.assert_no_leak(threshold_mb=10, context="50 batch encode results")

    def test_vocab_result_freed(self, tokenizer, memory_tracker):
        """Vocabulary result is freed."""
        memory_tracker.capture_baseline()

        for _ in range(20):
            vocab = tokenizer.get_vocab()
            _ = len(vocab)
            del vocab

        memory_tracker.assert_no_leak(threshold_mb=20, context="20 vocab accesses")

    def test_id_to_token_result_freed(self, tokenizer, memory_tracker):
        """id_to_token result is freed."""
        memory_tracker.capture_baseline()

        for _ in range(500):
            for token_id in range(100):
                token = tokenizer.id_to_token(token_id)
                del token

        memory_tracker.assert_no_leak(threshold_mb=5, context="50000 id_to_token calls")


class TestRepositoryResultCleanup:
    """Repository result cleanup tests."""

    def test_model_list_iteration_cleanup(self, memory_tracker):
        """Model list iteration cleans up properly."""
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

        # Iterate multiple times
        for _ in range(20):
            models = list(list_models())
            del models

        memory_tracker.assert_no_leak(threshold_mb=5, context="20 model list iterations")

    def test_model_list_partial_iteration_cleanup(self):
        """Partial iteration still cleans up (generator cleanup)."""
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

        # Only iterate first 2 items
        count = 0
        for _model in list_models():
            count += 1
            if count >= 2:
                break

        # Force GC
        gc.collect()
        gc.collect()
        gc.collect()

        # No assertion needed - test passes if no crash/leak

    def test_model_list_stress_cleanup(self, memory_tracker):
        """Repeated model listing cleans up properly."""
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
            # Full iteration
            models = list(list_models())
            _ = len(models)
            del models

        memory_tracker.assert_no_leak(threshold_mb=5, context="50 full model list iterations")


class TestChatResultCleanup:
    """Chat result cleanup tests."""

    def test_chat_items_result_freed(self, memory_tracker):
        """Chat items access results are freed."""
        from talu import Chat

        chat = Chat()
        chat.append("user", "hello")
        chat.append("assistant", "hi there")

        memory_tracker.capture_baseline()

        for _ in range(100):
            items = chat.items
            _ = len(items)
            del items

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 chat.items accesses")

        chat.close()

    def test_chat_to_json_result_freed(self, memory_tracker):
        """Chat to_json results are freed."""
        from talu import Chat

        chat = Chat()
        chat.append("user", "hello")
        chat.append("assistant", "hi there")

        memory_tracker.capture_baseline()

        for _ in range(100):
            json_str = chat.to_json()
            _ = len(json_str)
            del json_str

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 chat.to_json() calls")

        chat.close()

    def test_chat_to_dict_result_freed(self, memory_tracker):
        """Chat to_dict results are freed."""
        from talu import Chat

        chat = Chat()
        chat.append("user", "hello")
        chat.append("assistant", "hi there")

        memory_tracker.capture_baseline()

        for _ in range(100):
            d = chat.to_dict()
            _ = len(d)
            del d

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 chat.to_dict() calls")

        chat.close()


class TestTemplateResultCleanup:
    """Template result cleanup tests."""

    def test_template_render_result_freed(self, memory_tracker):
        """Template render results are freed."""
        from talu.template import PromptTemplate

        template = PromptTemplate("Hello {{ name }}!")

        memory_tracker.capture_baseline()

        for i in range(200):
            result = template.render(name=f"User{i}")
            _ = len(result)
            del result

        memory_tracker.assert_no_leak(threshold_mb=5, context="200 template renders")

    def test_template_compile_cleanup(self, memory_tracker):
        """Template compilation resources are cleaned up."""
        from talu.template import PromptTemplate

        memory_tracker.capture_baseline()

        for i in range(100):
            template = PromptTemplate(f"Hello {{{{ name_{i} }}}}!")
            _ = template.render(**{f"name_{i}": "test"})
            del template

        memory_tracker.assert_no_leak(threshold_mb=5, context="100 template compile/render cycles")

"""
Memory stress tests.

High-volume tests to detect:
1. Slow memory leaks
2. Fragmentation issues
3. Reference counting edge cases

These tests are marked with @pytest.mark.slow and are excluded from
normal CI runs. Run with: pytest tests/memory/test_stress.py -v
"""

import pytest

from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM


@pytest.fixture
def tokenizer():
    """Create tokenizer for stress tests."""

    from talu import Tokenizer

    return Tokenizer(TEST_MODEL_URI_TEXT_RANDOM)


@pytest.mark.slow
class TestTokenizerStress:
    """Tokenizer memory stress tests."""

    def test_rapid_encode_decode_cycles(self, tokenizer, memory_tracker):
        """Rapid encode/decode cycles don't leak."""
        # Warmup
        for _ in range(10):
            tokens = tokenizer.encode("warmup")
            _ = tokenizer.decode(tokens)

        memory_tracker.capture_baseline()

        # Stress
        for i in range(1000):
            text = f"stress test iteration {i} with some content"
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            assert text in decoded or decoded in text
            del tokens, decoded

        memory_tracker.assert_no_leak(threshold_mb=20, context="1000 encode/decode cycles")

    def test_large_text_encoding(self, tokenizer, memory_tracker):
        """Large text encoding doesn't leak."""
        memory_tracker.capture_baseline()

        large_text = "This is a test sentence. " * 500  # ~12KB

        for _ in range(100):
            tokens = tokenizer.encode(large_text)
            assert len(tokens) > 100
            del tokens

        memory_tracker.assert_no_leak(threshold_mb=20, context="100 large text encodes")

    def test_batch_encode_stress(self, tokenizer, memory_tracker):
        """Heavy batch encoding doesn't leak."""
        memory_tracker.capture_baseline()

        texts = [f"text number {i} with some content" for i in range(50)]

        for _ in range(100):
            batch = tokenizer.encode(texts)
            _ = len(batch)
            del batch

        memory_tracker.assert_no_leak(threshold_mb=30, context="100 batch encodes of 50 texts")

    def test_concurrent_tokenarray_dlpack(self, tokenizer, memory_tracker):
        """Many TokenArrays with DLPack exports don't leak."""
        torch = pytest.importorskip("torch")

        memory_tracker.capture_baseline()

        tensors = []
        for i in range(100):
            tokens = tokenizer.encode(f"test {i}")
            tensor = torch.from_dlpack(tokens)
            tensors.append(tensor)
            del tokens  # Delete array, tensor should survive

        # Verify all tensors valid
        for t in tensors:
            assert len(t) > 0

        del tensors
        memory_tracker.assert_no_leak(threshold_mb=10, context="100 DLPack exports")


@pytest.mark.slow
class TestChatStress:
    """Chat memory stress tests."""

    def test_many_chat_instances(self, memory_tracker):
        """Creating many Chat instances doesn't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        for _ in range(500):
            chat = Chat()
            chat.append("user", "hello")
            chat.append("assistant", "hi there")
            chat.close()

        memory_tracker.assert_no_leak(threshold_mb=20, context="500 Chat instances")

    def test_chat_with_large_messages(self, memory_tracker):
        """Chat with large messages doesn't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        large_message = "x" * 10000  # 10KB message

        for _ in range(100):
            chat = Chat()
            for _ in range(10):
                chat.append("user", large_message)
                chat.append("assistant", large_message)
            chat.close()

        memory_tracker.assert_no_leak(threshold_mb=50, context="100 chats with 10KB messages")

    def test_chat_with_many_turns(self, memory_tracker):
        """Chat with many conversation turns doesn't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        for _ in range(50):
            chat = Chat()
            for j in range(100):
                chat.append("user", f"message {j}")
                chat.append("assistant", f"response {j}")
            chat.close()

        memory_tracker.assert_no_leak(threshold_mb=30, context="50 chats with 100 turns each")

    def test_chat_context_manager_stress(self, memory_tracker):
        """Chat context manager stress test."""
        from talu import Chat

        memory_tracker.capture_baseline()

        for i in range(300):
            with Chat() as chat:
                chat.append("user", f"hello {i}")
                chat.append("assistant", f"hi {i}")
                _ = chat.items

        memory_tracker.assert_no_leak(threshold_mb=20, context="300 chat context managers")


@pytest.mark.slow
class TestConfigStress:
    """Config stress tests."""

    def test_config_creation_stress(self, memory_tracker):
        """Creating many configs doesn't leak."""
        from talu import GenerationConfig

        memory_tracker.capture_baseline()

        for i in range(2000):
            config = GenerationConfig(
                temperature=0.7,
                max_tokens=100,
                stop_sequences=[f"STOP{i}", f"END{i}"],
                logit_bias={i % 1000: 1.0},
            )
            del config

        memory_tracker.assert_no_leak(threshold_mb=10, context="2000 GenerationConfig instances")

    def test_config_merge_stress(self, memory_tracker):
        """Config merging stress test."""
        from talu import GenerationConfig

        memory_tracker.capture_baseline()

        base = GenerationConfig(temperature=0.5)

        for i in range(1000):
            override = GenerationConfig(
                max_tokens=i % 500 + 1,
                stop_sequences=[f"S{i}"],
            )
            merged = base | override
            _ = merged.max_tokens
            del override, merged

        memory_tracker.assert_no_leak(threshold_mb=10, context="1000 config merges")


@pytest.mark.slow
class TestTemplateStress:
    """Template stress tests."""

    def test_template_render_stress(self, memory_tracker):
        """Template rendering stress test."""
        from talu.template import PromptTemplate

        template = PromptTemplate("Hello {{ name }}! You have {{ count }} messages.")

        memory_tracker.capture_baseline()

        for i in range(5000):
            result = template.render(name=f"User{i}", count=i)
            _ = len(result)
            del result

        memory_tracker.assert_no_leak(threshold_mb=10, context="5000 template renders")

    def test_template_compilation_stress(self, memory_tracker):
        """Template compilation stress test."""
        from talu.template import PromptTemplate

        memory_tracker.capture_baseline()

        for i in range(500):
            template = PromptTemplate(f"Template {{{{ var_{i} }}}} number {i}")
            _ = template.render(**{f"var_{i}": "value"})
            del template

        memory_tracker.assert_no_leak(threshold_mb=10, context="500 template compilations")


@pytest.mark.slow
class TestMixedStress:
    """Mixed workload stress tests."""

    def test_interleaved_operations(self, memory_tracker):
        """Interleaved operations don't leak."""
        from talu import Chat, GenerationConfig
        from talu.template import PromptTemplate

        memory_tracker.capture_baseline()

        template = PromptTemplate("User says: {{ msg }}")

        for i in range(200):
            # Create chat
            chat = Chat()

            # Create config
            config = GenerationConfig(
                temperature=0.7,
                stop_sequences=[f"END{i}"],
            )

            # Render template
            rendered = template.render(msg=f"hello {i}")

            # Use in chat
            chat.append("user", rendered)
            chat.append("assistant", f"response {i}")

            # Access items
            _ = chat.items

            # Cleanup
            del config, rendered
            chat.close()

        memory_tracker.assert_no_leak(threshold_mb=20, context="200 interleaved operations")

    def test_parallel_chat_simulation(self, memory_tracker):
        """Simulated parallel chat sessions don't leak."""
        from talu import Chat

        memory_tracker.capture_baseline()

        # Simulate multiple concurrent sessions
        sessions = []
        for _ in range(20):
            chat = Chat()
            sessions.append(chat)

        # Interleave operations across sessions
        for turn in range(50):
            for i, chat in enumerate(sessions):
                chat.append("user", f"turn {turn} msg {i}")
                chat.append("assistant", f"response {turn} {i}")

        # Close all
        for chat in sessions:
            chat.close()
        del sessions

        memory_tracker.assert_no_leak(threshold_mb=30, context="20 parallel sessions with 50 turns")

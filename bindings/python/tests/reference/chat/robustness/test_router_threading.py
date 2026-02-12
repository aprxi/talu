"""Tests for Router streaming thread lifecycle management.

These tests verify that threads spawned by stream() are properly tracked
and cleaned up after stream consumption completes.

Requires TEST_MODEL_URI_TEXT.
"""

import pytest

import talu
from talu.router import Router


class TestRouterThreadCleanup:
    """Tests for Router streaming thread lifecycle management.

    These tests verify that threads spawned by stream() are properly tracked
    and cleaned up after stream consumption completes.
    """

    @pytest.mark.requires_model
    @pytest.mark.integration
    def test_stream_thread_cleanup(self, test_model_path):
        """Streaming threads are removed from _active_threads after consumption."""
        router = Router(models=[test_model_path])
        chat = talu.Chat(test_model_path)

        # Track initial thread count
        with router._active_threads_lock:
            initial_count = len(router._active_threads)

        # Consume the stream completely
        tokens = list(router.stream(chat, "Say hello"))
        assert len(tokens) > 0

        # After complete consumption, thread should be cleaned up
        # The finally block in stream() joins the thread and removes it
        with router._active_threads_lock:
            final_count = len(router._active_threads)

        assert final_count == initial_count, (
            f"Thread not cleaned up: started with {initial_count}, ended with {final_count}"
        )

        router.close()
        chat.close()

    @pytest.mark.requires_model
    @pytest.mark.integration
    def test_stream_thread_cleanup_on_early_break(self, test_model_path):
        """Streaming threads are cleaned up even when breaking early."""
        router = Router(models=[test_model_path])
        chat = talu.Chat(test_model_path)

        with router._active_threads_lock:
            initial_count = len(router._active_threads)

        # Break after first token (partial consumption)
        for _token in router.stream(chat, "Count to 100"):
            break  # Exit after first token

        # Thread should still be cleaned up (finally block runs on generator close)
        with router._active_threads_lock:
            final_count = len(router._active_threads)

        assert final_count == initial_count, (
            f"Thread not cleaned up after early break: started with {initial_count}, ended with {final_count}"
        )

        router.close()
        chat.close()

    @pytest.mark.requires_model
    @pytest.mark.integration
    def test_stream_thread_cleanup_on_exception(self, test_model_path):
        """Streaming threads are cleaned up when exception occurs during iteration."""
        router = Router(models=[test_model_path])
        chat = talu.Chat(test_model_path)

        with router._active_threads_lock:
            initial_count = len(router._active_threads)

        # Raise exception during iteration
        with pytest.raises(ValueError, match="test exception"):
            for i, _token in enumerate(router.stream(chat, "Count to 100")):
                if i >= 2:
                    raise ValueError("test exception")

        # Thread should still be cleaned up (finally block runs)
        with router._active_threads_lock:
            final_count = len(router._active_threads)

        assert final_count == initial_count, (
            f"Thread not cleaned up after exception: started with {initial_count}, ended with {final_count}"
        )

        router.close()
        chat.close()

    @pytest.mark.requires_model
    @pytest.mark.integration
    def test_multiple_sequential_streams_cleanup(self, test_model_path):
        """Multiple sequential streams properly clean up threads.

        This test currently crashes (segfault) due to memory corruption
        in the Zig core during sequential engine lifecycle. Same root cause
        as test_engine_lifecycle.py::test_sequential_chat_instances_with_close.
        """
        router = Router(models=[test_model_path])
        chat = talu.Chat(test_model_path)

        with router._active_threads_lock:
            initial_count = len(router._active_threads)

        # Run multiple streams sequentially
        for i in range(3):
            tokens = list(router.stream(chat, f"Say {i}"))
            assert len(tokens) > 0

            # Check cleanup after each stream
            with router._active_threads_lock:
                current_count = len(router._active_threads)
            assert current_count == initial_count, f"Thread leak on iteration {i}"

        router.close()
        chat.close()

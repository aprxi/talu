"""
Engine lifecycle memory tests.

Tests for memory safety when creating/destroying multiple Chat instances
that use real inference backends (LocalEngine).

These tests detect:
1. Use-after-free from engine cleanup
2. Double-free from shared resources
3. Global state corruption from cleanup

Issue: Creating multiple Chat instances sequentially with explicit close()
can cause segfault during the second Chat's generation due to memory
corruption in the Zig core.
"""

import gc

import pytest


@pytest.mark.integration
class TestEngineLifecycle:
    """Tests for engine lifecycle memory safety."""

    def test_sequential_chat_instances_with_close(self, test_model_path):
        """Sequential Chat instances with explicit close should not segfault.

        This is a regression test for a memory corruption bug where
        creating Chat instances, generating, and closing them sequentially
        causes segfault during later generations.

        Root cause: TBD - suspected global state corruption during
        InferenceBackend.deinit() / LocalEngine.deinit().
        """
        from talu import Chat

        for i in range(3):
            chat = Chat(test_model_path)
            response = chat(f"Say the number {i + 1}")
            assert response is not None, f"No response on iteration {i}"
            chat.close()

    def test_sequential_chat_instances_without_close(self, test_model_path):
        """Sequential Chat instances relying on GC should not segfault.

        Similar to test_sequential_chat_instances_with_close but relies
        on Python's garbage collector for cleanup instead of explicit close().
        """
        from talu import Chat

        for i in range(3):
            chat = Chat(test_model_path)
            response = chat(f"Say the number {i + 1}")
            assert response is not None, f"No response on iteration {i}"
            # No explicit close - rely on GC

        # Force GC to ensure all cleanup runs
        gc.collect()
        gc.collect()
        gc.collect()

    def test_overlapping_chat_instances(self, test_model_path):
        """Overlapping Chat instance lifetimes should not segfault.

        Creates a second Chat while the first is still alive, then
        destroys them in various orders.
        """
        from talu import Chat

        # Create two chats
        chat1 = Chat(test_model_path)
        chat2 = Chat(test_model_path)

        # Use both
        r1 = chat1("Say hello")
        r2 = chat2("Say world")

        assert r1 is not None
        assert r2 is not None

        # Close in order
        chat1.close()
        chat2.close()

    def test_overlapping_chat_instances_reverse_close(self, test_model_path):
        """Overlapping Chat instances closed in reverse order should not segfault."""
        from talu import Chat

        # Create two chats
        chat1 = Chat(test_model_path)
        chat2 = Chat(test_model_path)

        # Use both
        r1 = chat1("Say hello")
        r2 = chat2("Say world")

        assert r1 is not None
        assert r2 is not None

        # Close in reverse order
        chat2.close()
        chat1.close()

    def test_chat_reuse_after_generation(self, test_model_path):
        """Chat instance can be reused for multiple generations."""
        from talu import Chat

        chat = Chat(test_model_path)

        for i in range(3):
            response = chat(f"Say the number {i + 1}")
            assert response is not None, f"No response on iteration {i}"

        chat.close()

    def test_context_manager_sequential(self, test_model_path):
        """Sequential Chat instances via context manager should not segfault."""
        from talu import Chat

        for i in range(3):
            with Chat(test_model_path) as chat:
                response = chat(f"Say the number {i + 1}")
                assert response is not None, f"No response on iteration {i}"


@pytest.mark.integration
@pytest.mark.slow
class TestEngineLifecycleStress:
    """Stress tests for engine lifecycle."""

    def test_many_sequential_chat_instances(self, test_model_path, memory_tracker):
        """Many sequential Chat instances should not leak or segfault."""
        from talu import Chat

        # Warmup
        chat = Chat(test_model_path)
        _ = chat("warmup")
        chat.close()

        memory_tracker.capture_baseline()

        for _i in range(10):
            chat = Chat(test_model_path)
            response = chat("Say hello")
            assert response is not None
            chat.close()

        memory_tracker.assert_no_leak(
            threshold_mb=100,  # Model loading is expensive, allow some growth
            context="10 sequential Chat instances with generation",
        )

    def test_rapid_chat_create_close(self, test_model_path, memory_tracker):
        """Rapid create/close cycles should not leak or segfault."""
        from talu import Chat

        # Warmup
        chat = Chat(test_model_path)
        chat.close()

        memory_tracker.capture_baseline()

        for _ in range(20):
            chat = Chat(test_model_path)
            # No generation - just create/close
            chat.close()

        memory_tracker.assert_no_leak(
            threshold_mb=50,
            context="20 rapid create/close cycles",
        )

    def test_rapid_chat_creation_with_streaming(self, test_model_path) -> None:
        """Rapidly create/destroy Chat instances with streaming."""
        from talu import Chat

        for i in range(10):
            chat = Chat(test_model_path)
            response = chat(f"Count to {i + 1}", stream=True)
            text = "".join(response)  # Force stream consumption
            assert text is not None, f"No response on iteration {i}"
            chat.close()

    def test_concurrent_chats_with_streaming(self, test_model_path) -> None:
        """Multiple concurrent Chat instances with streaming should not crash."""
        import threading
        from queue import Queue

        from talu import Chat

        results: Queue[int] = Queue()
        errors: Queue[tuple[int, str]] = Queue()

        def run_chat(model_path: str, iteration: int) -> None:
            try:
                chat = Chat(model_path)
                response = chat(f"Say number {iteration}", stream=True)
                _ = "".join(response)
                chat.close()
                results.put(iteration)
            except Exception as e:
                errors.put((iteration, str(e)))

        # Create 5 concurrent chats
        threads = []
        for i in range(5):
            t = threading.Thread(target=run_chat, args=(test_model_path, i))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join(timeout=30)  # DEADLOCK_GUARD

        # Check results
        successful = list(results.queue)
        error_list = list(errors.queue)

        assert len(successful) == 5, f"Only {len(successful)}/5 succeeded, errors: {error_list}"
        assert len(error_list) == 0, f"Errors: {error_list}"

"""
Callback lifetime safety tests.

Verifies that Python callbacks passed to Zig functions:
1. Survive garbage collection while Zig may call them
2. Are properly prevented from early collection
3. Don't cause segfaults when GC'd prematurely (caught by design)

These tests verify the IMPLEMENTATION patterns, not runtime behavior,
since actual segfaults would crash the test process.
"""

import gc
import weakref


class TestTokenCallbackLifetime:
    """Token streaming callback survival tests."""

    def test_callback_stored_in_instance_survives(self):
        """Callback stored in self._ field survives GC."""
        from talu._native import CProgressCallback

        call_count = 0

        @CProgressCallback
        def progress_cb(update_ptr, user_data):
            nonlocal call_count
            call_count += 1

        # Simulate proper storage pattern
        class MockHandler:
            def __init__(self):
                self._callback = None

            def setup(self, callback):
                self._callback = callback  # CORRECT: stored

        handler = MockHandler()
        handler.setup(progress_cb)

        # Force GC - callback should survive
        del progress_cb
        gc.collect()
        gc.collect()
        gc.collect()

        # Verify callback still exists
        assert handler._callback is not None

    def test_callback_not_stored_can_be_collected(self, callback_ref_tracker):
        """Callback NOT stored can be GC'd (documents the risk)."""
        from talu._native import CProgressCallback

        @CProgressCallback
        def progress_cb(update_ptr, user_data):
            pass

        callback_ref_tracker.track_weak(progress_cb)

        # Delete only reference
        del progress_cb

        # Should be collected (THIS IS THE RISK we're documenting)
        callback_ref_tracker.assert_weak_collected()

    def test_progress_renderer_stores_callback(self):
        """ProgressRenderer correctly stores callback reference."""
        from talu._progress import ProgressRenderer

        renderer = ProgressRenderer()

        # Verify callback is stored
        assert renderer._c_callback is not None

        # Store weak ref before forcing GC
        weak_cb = weakref.ref(renderer._c_callback)

        # Force GC - callback should survive because renderer holds it
        gc.collect()
        gc.collect()
        gc.collect()

        # Callback should still exist
        assert weak_cb() is not None


class TestChatCallbackRefs:
    """Chat/Router callback reference tests."""

    def test_chat_context_manager_cleanup(self):
        """Chat context manager properly cleans up callbacks."""
        from talu import Chat

        # Use context manager
        with Chat() as chat:
            chat.append("user", "hello")
            # Callbacks should be active here

        # After exit, chat is closed
        # No assertion needed - test passes if no crash

    def test_multiple_chat_instances_independent(self):
        """Multiple Chat instances have independent callbacks."""
        from talu import Chat

        chats = []
        for i in range(5):
            chat = Chat()
            chat.append("user", f"message {i}")
            chats.append(chat)

        # All should be independent
        for chat in chats:
            assert len(chat.items) == 1
            chat.close()

        # Force GC after closing
        del chats
        gc.collect()
        gc.collect()
        gc.collect()


class TestCallbackPreventionPatterns:
    """Tests for callback GC prevention patterns."""

    def test_keepalive_list_pattern(self):
        """Keepalive list prevents callback collection."""
        import ctypes

        # Common pattern: keep references in a list
        keepalive = []

        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int)

        def my_callback(x):
            pass

        cb = callback_type(my_callback)
        keepalive.append(cb)

        weak_ref = weakref.ref(cb)
        del cb

        # Should NOT be collected because keepalive holds reference
        gc.collect()
        gc.collect()
        gc.collect()

        assert weak_ref() is not None

        # Clear keepalive
        keepalive.clear()
        gc.collect()

        # Now it can be collected
        assert weak_ref() is None

    def test_instance_attribute_pattern(self):
        """Instance attribute prevents callback collection."""
        import ctypes

        callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int)

        class Handler:
            def __init__(self):
                self._cb = callback_type(self._on_event)

            def _on_event(self, x):
                pass

        handler = Handler()
        weak_ref = weakref.ref(handler._cb)

        # Force GC - callback should survive with handler
        gc.collect()
        gc.collect()
        gc.collect()

        assert weak_ref() is not None

        # Delete handler
        del handler
        gc.collect()
        gc.collect()
        gc.collect()

        # Now callback can be collected
        assert weak_ref() is None

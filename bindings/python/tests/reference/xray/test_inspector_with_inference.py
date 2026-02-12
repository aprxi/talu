"""
Reference tests for XRay Inspector capture mechanics.

NOTE: XRay capture is a global mechanism that captures tensor data during
inference. The capture must be enabled BEFORE inference starts and the
inference engine must be instrumented with XRay hooks.

Currently, XRay capture may not be integrated with Chat/Router - it's
primarily designed for lower-level ops debugging. These tests verify
the Inspector API works correctly even when no data is captured.
"""

from talu import Chat
from talu.xray import CaptureMode, Inspector, Point
from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM as MODEL_URI

# =============================================================================
# Inspector Enable/Disable with Real Chat
# =============================================================================


class TestInspectorWithChat:
    """Tests for Inspector enable/disable during Chat operations."""

    def test_inspector_enable_disable_cycle(self):
        """Inspector enable/disable works during Chat lifecycle."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            # Enable before generation
            inspector.enable()
            assert inspector.is_enabled

            # Generate
            response = chat.send("Hello", max_tokens=3)
            assert response is not None

            # Disable after
            inspector.disable()
            assert not inspector.is_enabled

            # Count may be 0 if XRay not integrated with Router
            # This is expected behavior
            assert inspector.count >= 0
        finally:
            del chat

    def test_context_manager_with_generation(self):
        """Context manager works around generation."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                assert inspector.is_enabled
                response = chat.send("Test", max_tokens=2)
                assert response is not None

            assert not inspector.is_enabled
        finally:
            del chat

    def test_multiple_generations_with_inspector(self):
        """Inspector handles multiple generation calls."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("First", max_tokens=2)
                chat.send("Second", max_tokens=2)
                chat.send("Third", max_tokens=2)

            # Should not crash
            assert inspector.count >= 0
        finally:
            del chat


# =============================================================================
# Inspector Query Methods (Empty State)
# =============================================================================


class TestInspectorQueriesEmpty:
    """Tests for query methods when no data is captured."""

    def test_iter_after_generation_no_crash(self):
        """iter() doesn't crash after generation."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Hello", max_tokens=2)

            # Should not crash, may be empty
            tensors = list(inspector.iter())
            assert isinstance(tensors, list)
        finally:
            del chat

    def test_find_after_generation_no_crash(self):
        """find() doesn't crash after generation."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)

            # May return None if nothing captured
            result = inspector.find(point=Point.LOGITS_SCALED)
            assert result is None or hasattr(result, "point")
        finally:
            del chat

    def test_summary_after_generation(self):
        """summary() returns valid dict after generation."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)

            summary = inspector.summary()
            assert "total_captures" in summary
            assert "by_point" in summary
            assert "has_anomalies" in summary
            assert summary["total_captures"] >= 0
        finally:
            del chat


# =============================================================================
# Clear and Reuse
# =============================================================================


class TestInspectorClearReuse:
    """Tests for clear() and reusing Inspector."""

    def test_clear_after_generation(self):
        """clear() works after generation."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Hello", max_tokens=2)

            inspector.clear()
            assert inspector.count == 0
        finally:
            del chat

    def test_reuse_inspector_multiple_sessions(self):
        """Inspector can be reused across multiple Chat sessions."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        for i in range(3):
            chat = Chat(MODEL_URI)
            try:
                with inspector:
                    chat.send(f"Message {i}", max_tokens=2)

                inspector.clear()
            finally:
                del chat

        # Should not have accumulated data
        assert inspector.count == 0


# =============================================================================
# Different Capture Modes
# =============================================================================


class TestCaptureModes:
    """Tests for different capture modes with Chat."""

    def test_stats_mode_with_chat(self):
        """STATS mode works with Chat."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)
            # Should not crash
        finally:
            del chat

    def test_sample_mode_with_chat(self):
        """SAMPLE mode works with Chat."""
        inspector = Inspector(
            points=[Point.LOGITS_SCALED],
            mode=CaptureMode.SAMPLE,
            sample_count=8,
        )

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)
            # Should not crash
        finally:
            del chat

    def test_full_mode_with_chat(self):
        """FULL mode works with Chat."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.FULL)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)
            # Should not crash
        finally:
            del chat


# =============================================================================
# Different Point Configurations
# =============================================================================


class TestPointConfigurations:
    """Tests for different point configurations with Chat."""

    def test_all_points(self):
        """Capturing all points doesn't crash."""
        inspector = Inspector(points="all", mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)
        finally:
            del chat

    def test_multiple_layer_points(self):
        """Capturing multiple layer points works."""
        inspector = Inspector(
            points=[Point.BLOCK_OUT, Point.FFN_DOWN, Point.ATTN_OUT],
            mode=CaptureMode.STATS,
        )

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)
        finally:
            del chat

    def test_bitmask_points(self):
        """Capturing with bitmask works."""
        from talu.xray import POINT_BLOCK_OUT, POINT_LOGITS_SCALED

        inspector = Inspector(
            points=POINT_LOGITS_SCALED | POINT_BLOCK_OUT,
            mode=CaptureMode.STATS,
        )

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                chat.send("Test", max_tokens=2)
        finally:
            del chat


# =============================================================================
# Streaming with Inspector
# =============================================================================


class TestStreamingWithInspector:
    """Tests for Inspector with streaming generation."""

    def test_streaming_with_inspector(self):
        """Inspector works during streaming generation."""
        inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            with inspector:
                response = chat("Test", max_tokens=5)
                # Consume stream
                text = "".join(response)
                assert len(text) > 0

            # Should not crash
            assert inspector.count >= 0
        finally:
            del chat


# =============================================================================
# Resource Cleanup
# =============================================================================


class TestInspectorResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_inspector_deleted_during_generation(self):
        """Inspector can be deleted after generation."""
        import gc

        chat = Chat(MODEL_URI)
        try:
            inspector = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)
            with inspector:
                chat.send("Test", max_tokens=2)

            del inspector
            gc.collect()
            # Should not crash
        finally:
            del chat

    def test_multiple_inspectors_with_chat(self):
        """Multiple inspectors can be used with Chat."""

        inspector1 = Inspector(points=[Point.LOGITS_SCALED], mode=CaptureMode.STATS)
        inspector2 = Inspector(points=[Point.BLOCK_OUT], mode=CaptureMode.STATS)

        chat = Chat(MODEL_URI)
        try:
            # Use inspector1
            with inspector1:
                chat.send("First", max_tokens=2)

            # Use inspector2
            with inspector2:
                chat.send("Second", max_tokens=2)

            # Both should be usable
            assert inspector1.count >= 0
            assert inspector2.count >= 0
        finally:
            del chat

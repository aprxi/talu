"""
Mock-based tests for XRay Inspector.

These tests verify the Inspector API without requiring actual model inference.
They cover error paths, edge cases, and Python-side logic to ensure stability
under error conditions.
"""

import pytest

from talu.xray import CaptureMode, Inspector, Point

# =============================================================================
# Inspector State Tests
# =============================================================================


class TestInspectorState:
    """Tests for Inspector state management."""

    def test_inspector_initial_state(self):
        """Inspector starts disabled."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        assert not inspector.is_enabled
        assert inspector.count == 0

    def test_inspector_enable_disable(self):
        """Inspector can be enabled and disabled."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.enable()
        assert inspector.is_enabled

        inspector.disable()
        assert not inspector.is_enabled

    def test_inspector_context_manager(self):
        """Inspector works as context manager."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        assert not inspector.is_enabled

        with inspector:
            assert inspector.is_enabled

        assert not inspector.is_enabled

    def test_inspector_clear(self):
        """Inspector clear resets state."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.clear()
        assert inspector.count == 0


# =============================================================================
# Inspector Configuration Tests
# =============================================================================


class TestInspectorConfiguration:
    """Tests for Inspector configuration options."""

    def test_inspector_stats_mode(self):
        """Inspector with STATS mode initializes without error."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        # Should not crash - mode is internal
        assert inspector.count == 0

    def test_inspector_sample_mode(self):
        """Inspector with SAMPLE mode initializes without error."""
        inspector = Inspector(
            points=[Point.LM_HEAD],
            mode=CaptureMode.SAMPLE,
            sample_count=16,
        )
        # Should not crash - mode is internal
        assert inspector.count == 0

    def test_inspector_full_mode(self):
        """Inspector with FULL mode initializes without error."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.FULL)
        # Should not crash - mode is internal
        assert inspector.count == 0

    def test_inspector_multiple_points(self):
        """Inspector can capture multiple points."""
        inspector = Inspector(
            points=[Point.LM_HEAD, Point.BLOCK_OUT, Point.FFN_DOWN],
            mode=CaptureMode.STATS,
        )
        # Should not crash
        assert inspector.count == 0

    def test_inspector_all_points_string(self):
        """Inspector accepts 'all' string for points."""
        inspector = Inspector(points="all", mode=CaptureMode.STATS)
        # Should not crash
        assert inspector.count == 0

    def test_inspector_bitmask_points(self):
        """Inspector accepts bitmask for points."""
        from talu.xray import POINT_BLOCK_OUT, POINT_LM_HEAD

        inspector = Inspector(
            points=POINT_LM_HEAD | POINT_BLOCK_OUT,
            mode=CaptureMode.STATS,
        )
        # Should not crash
        assert inspector.count == 0


# =============================================================================
# Inspector Query Tests (Empty State)
# =============================================================================


class TestInspectorQueriesEmpty:
    """Tests for query methods when no data is captured."""

    def test_iter_empty(self):
        """iter() returns empty iterator when nothing captured."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        tensors = list(inspector.iter())
        assert tensors == []

    def test_iter_with_filter_empty(self):
        """iter() with filter returns empty when nothing captured."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        tensors = list(inspector.iter(point=Point.LM_HEAD, layer=0, token=0))
        assert tensors == []

    def test_find_empty(self):
        """find() returns None when nothing captured."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        result = inspector.find(point=Point.LM_HEAD)
        assert result is None

    def test_find_with_filter_empty(self):
        """find() with filter returns None when nothing captured."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        result = inspector.find(point=Point.LM_HEAD, layer=0, token=0)
        assert result is None

    def test_summary_empty(self):
        """summary() returns valid dict when empty."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        summary = inspector.summary()

        assert "total_captures" in summary
        assert "by_point" in summary
        assert "has_anomalies" in summary
        assert summary["total_captures"] == 0
        assert summary["has_anomalies"] is False

    def test_count_matching_empty(self):
        """count_matching() returns 0 when empty."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        count = inspector.count_matching(point=Point.LM_HEAD)
        assert count == 0

    def test_logits_empty(self):
        """logits() returns None when empty."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        result = inspector.logits(token=0)
        assert result is None


# =============================================================================
# Inspector Index Access Tests
# =============================================================================


class TestInspectorIndexAccess:
    """Tests for index-based access."""

    def test_getitem_out_of_range(self):
        """__getitem__ raises IndexError for out of range."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        with pytest.raises(IndexError):
            _ = inspector[0]

    def test_getitem_negative_out_of_range(self):
        """__getitem__ with negative index raises IndexError when empty."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        with pytest.raises(IndexError):
            _ = inspector[-1]

    def test_get_returns_none_for_invalid(self):
        """get() returns None for invalid index."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        result = inspector.get(0)
        assert result is None

        result = inspector.get(100)
        assert result is None


# =============================================================================
# Inspector Iteration Tests
# =============================================================================


class TestInspectorIteration:
    """Tests for iteration behavior."""

    def test_iter_is_iterator(self):
        """__iter__ returns iterator."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        iterator = iter(inspector)
        assert hasattr(iterator, "__next__")

    def test_iter_exhausted_is_empty(self):
        """Exhausted iterator produces no items."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        # First iteration
        items1 = list(inspector)
        # Second iteration
        items2 = list(inspector)

        assert items1 == items2 == []

    def test_iter_with_point_string(self):
        """iter() accepts point as string."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        # Should accept string and convert to enum
        tensors = list(inspector.iter(point="lm_head"))
        assert tensors == []


# =============================================================================
# Inspector Lifecycle Tests
# =============================================================================


class TestInspectorLifecycle:
    """Tests for Inspector lifecycle management."""

    def test_inspector_reusable(self):
        """Inspector can be reused after clear."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        with inspector:
            pass

        inspector.clear()

        with inspector:
            pass

        assert inspector.count == 0

    def test_inspector_multiple_enable_disable(self):
        """Multiple enable/disable cycles work correctly."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        for _ in range(5):
            inspector.enable()
            assert inspector.is_enabled
            inspector.disable()
            assert not inspector.is_enabled

    def test_inspector_enable_when_enabled(self):
        """Enable when already enabled is safe."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.enable()
        inspector.enable()  # Should not crash
        assert inspector.is_enabled

    def test_inspector_disable_when_disabled(self):
        """Disable when already disabled is safe."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.disable()  # Already disabled
        inspector.disable()  # Should not crash
        assert not inspector.is_enabled


# =============================================================================
# Inspector Print Summary Tests
# =============================================================================


class TestInspectorPrintSummary:
    """Tests for print_summary() output."""

    def test_print_summary_empty(self, capsys):
        """print_summary() works when empty."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.print_summary()
        captured = capsys.readouterr()
        assert "Captured 0 tensors" in captured.out


class TestInspectorPrintSummaryExtra:
    """Additional print_summary tests."""

    def test_print_summary_no_anomalies(self, capsys):
        """print_summary() doesn't show anomalies section when none."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.print_summary()

        captured = capsys.readouterr()
        assert "Anomalies" not in captured.out


# =============================================================================
# Point Enum Tests
# =============================================================================


class TestPointEnum:
    """Tests for Point enum values."""

    def test_point_logits(self):
        """Point.LM_HEAD has expected value."""
        assert Point.LM_HEAD is not None
        assert Point.LM_HEAD.name == "LM_HEAD"

    def test_point_layer_residual(self):
        """Point.BLOCK_OUT has expected value."""
        assert Point.BLOCK_OUT is not None
        assert Point.BLOCK_OUT.name == "BLOCK_OUT"

    def test_point_from_string(self):
        """Point can be created from string."""
        point = Point["LM_HEAD"]
        assert point == Point.LM_HEAD


# =============================================================================
# CaptureMode Enum Tests
# =============================================================================


class TestCaptureModeEnum:
    """Tests for CaptureMode enum values."""

    def test_capture_mode_stats(self):
        """CaptureMode.STATS has expected value."""
        assert CaptureMode.STATS is not None
        assert CaptureMode.STATS.name == "STATS"

    def test_capture_mode_sample(self):
        """CaptureMode.SAMPLE has expected value."""
        assert CaptureMode.SAMPLE is not None
        assert CaptureMode.SAMPLE.name == "SAMPLE"

    def test_capture_mode_full(self):
        """CaptureMode.FULL has expected value."""
        assert CaptureMode.FULL is not None
        assert CaptureMode.FULL.name == "FULL"


# =============================================================================
# Inspector Closed State Tests
# =============================================================================


class TestInspectorClosedState:
    """Tests for Inspector after close()."""

    def test_close_is_idempotent(self):
        """close() can be called multiple times."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        inspector.close()
        inspector.close()  # Second call should not raise
        inspector.close()  # Third call should not raise

    def test_count_raises_after_close(self):
        """count raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        inspector.close()

        with pytest.raises(StateError) as exc_info:
            _ = inspector.count

        assert "closed" in str(exc_info.value).lower()
        assert exc_info.value.code == "STATE_CLOSED"

    def test_enable_raises_after_close(self):
        """enable() raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        inspector.close()

        with pytest.raises(StateError):
            inspector.enable()

    def test_get_raises_after_close(self):
        """get() raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        inspector.close()

        with pytest.raises(StateError):
            inspector.get(0)

    def test_clear_raises_after_close(self):
        """clear() raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        inspector.close()

        with pytest.raises(StateError):
            inspector.clear()

    def test_overflow_raises_after_close(self):
        """overflow raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        inspector.close()

        with pytest.raises(StateError):
            _ = inspector.overflow

    def test_find_anomaly_raises_after_close(self):
        """find_anomaly() raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        inspector.close()

        with pytest.raises(StateError):
            inspector.find_anomaly()

    def test_del_suppresses_exceptions(self):
        """__del__ suppresses exceptions during cleanup."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)
        # Close first to force __del__ to try to close again
        inspector.close()
        # Should not raise
        inspector.__del__()


# =============================================================================
# Inspector get_data Tests
# =============================================================================


class TestInspectorGetData:
    """Tests for get_data() method."""

    def test_get_data_returns_none_for_invalid_index(self):
        """get_data() returns None for invalid index."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.FULL)

        result = inspector.get_data(0)
        assert result is None

        result = inspector.get_data(100)
        assert result is None

    def test_get_data_raises_after_close(self):
        """get_data() raises StateError after close."""
        from talu.exceptions import StateError

        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.FULL)
        inspector.close()

        with pytest.raises(StateError):
            inspector.get_data(0)


# =============================================================================
# Inspector len Tests
# =============================================================================


class TestInspectorLen:
    """Tests for __len__ method."""

    def test_len_returns_count(self):
        """len() returns the same as count property."""
        inspector = Inspector(points=[Point.LM_HEAD], mode=CaptureMode.STATS)

        assert len(inspector) == inspector.count
        assert len(inspector) == 0


# =============================================================================
# Inspector Invalid Points Tests
# =============================================================================


class TestInspectorInvalidPoints:
    """Tests for invalid points parameter."""

    def test_invalid_points_type_raises(self):
        """Invalid points type raises ValidationError."""
        from talu.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Inspector(points={"invalid": "dict"}, mode=CaptureMode.STATS)

        assert "Invalid points" in str(exc_info.value)
        assert exc_info.value.code == "INVALID_ARGUMENT"

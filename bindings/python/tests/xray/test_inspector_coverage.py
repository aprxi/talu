"""
Additional tests for talu/xray/inspector.py coverage.

Targets uncovered edge cases without requiring a model for generation.
"""

import pytest

from talu.exceptions import ValidationError
from talu.xray import CaptureMode, Inspector, Point

# =============================================================================
# Constructor Tests
# =============================================================================


class TestInspectorConstruction:
    """Tests for Inspector construction."""

    def test_construction_with_all_points(self):
        """Inspector accepts 'all' for points."""
        inspector = Inspector(points="all")
        assert inspector._mode == CaptureMode.STATS

    def test_construction_with_int_mask(self):
        """Inspector accepts integer bitmask."""
        inspector = Inspector(points=0xFF)
        assert inspector._handle is not None

    def test_construction_with_point_list(self):
        """Inspector accepts list of Point enums."""
        inspector = Inspector(points=[Point.LM_HEAD, Point.BLOCK_OUT])
        assert inspector._handle is not None

    def test_construction_with_string_list(self):
        """Inspector accepts list of point name strings."""
        inspector = Inspector(points=["lm_head", "block_out"])
        assert inspector._handle is not None

    def test_construction_with_sample_mode(self):
        """Inspector accepts SAMPLE mode with sample_count."""
        inspector = Inspector(mode=CaptureMode.SAMPLE, sample_count=16)
        assert inspector._mode == CaptureMode.SAMPLE
        assert inspector._sample_count == 16

    def test_construction_invalid_points_raises(self):
        """Inspector raises for invalid points type."""
        with pytest.raises(ValidationError, match="Invalid points"):
            Inspector(points=3.14)  # type: ignore[arg-type]


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestInspectorContextManager:
    """Tests for Inspector context manager protocol."""

    def test_context_manager_enables_on_enter(self):
        """__enter__ enables capturing."""
        inspector = Inspector()
        with inspector as i:
            assert i is inspector
            assert inspector._enabled is True

    def test_context_manager_disables_on_exit(self):
        """__exit__ disables capturing."""
        inspector = Inspector()
        with inspector:
            pass
        assert inspector._enabled is False


# =============================================================================
# Enable/Disable Tests
# =============================================================================


class TestEnableDisable:
    """Tests for enable/disable methods."""

    def test_enable_sets_flag(self):
        """enable() sets _enabled flag."""
        inspector = Inspector()
        inspector.enable()
        assert inspector._enabled is True
        inspector.disable()

    def test_disable_clears_flag(self):
        """disable() clears _enabled flag."""
        inspector = Inspector()
        inspector.enable()
        inspector.disable()
        assert inspector._enabled is False

    def test_is_enabled_property(self):
        """is_enabled property queries capture state."""
        inspector = Inspector()
        # Note: is_enabled queries global state, not instance state
        initial = inspector.is_enabled
        assert isinstance(initial, bool)


# =============================================================================
# Count and Clear Tests
# =============================================================================


class TestCountAndClear:
    """Tests for count and clear methods."""

    def test_count_initially_zero(self):
        """count starts at 0."""
        inspector = Inspector()
        assert inspector.count == 0

    def test_len_returns_count(self):
        """__len__ returns count."""
        inspector = Inspector()
        assert len(inspector) == 0

    def test_clear_resets_count(self):
        """clear() resets count to 0."""
        inspector = Inspector()
        inspector.clear()
        assert inspector.count == 0

    def test_overflow_initially_false(self):
        """overflow starts as False."""
        inspector = Inspector()
        assert inspector.overflow is False


# =============================================================================
# Get and Indexing Tests
# =============================================================================


class TestGetAndIndexing:
    """Tests for get() and __getitem__ methods."""

    def test_get_out_of_range_returns_none(self):
        """get() returns None for out of range index."""
        inspector = Inspector()
        result = inspector.get(0)
        assert result is None

    def test_getitem_out_of_range_raises(self):
        """__getitem__ raises IndexError for out of range."""
        inspector = Inspector()
        with pytest.raises(IndexError, match="out of range"):
            _ = inspector[0]

    def test_getitem_negative_index(self):
        """__getitem__ handles negative index."""
        inspector = Inspector()
        # With count=0, negative index also out of range
        with pytest.raises(IndexError):
            _ = inspector[-1]


# =============================================================================
# Iterator Tests
# =============================================================================


class TestIterator:
    """Tests for iter() and __iter__ methods."""

    def test_iter_empty_yields_nothing(self):
        """iter() on empty inspector yields nothing."""
        inspector = Inspector()
        result = list(inspector.iter())
        assert result == []

    def test_iter_is_default_iterator(self):
        """__iter__ calls iter() with no filters."""
        inspector = Inspector()
        result = list(inspector)
        assert result == []

    def test_iter_with_string_point_filter(self):
        """iter() accepts string point name."""
        inspector = Inspector()
        result = list(inspector.iter(point="lm_head"))
        assert result == []

    def test_iter_with_point_enum_filter(self):
        """iter() accepts Point enum."""
        inspector = Inspector()
        result = list(inspector.iter(point=Point.LM_HEAD))
        assert result == []

    def test_iter_with_layer_filter(self):
        """iter() accepts layer filter."""
        inspector = Inspector()
        result = list(inspector.iter(layer=0))
        assert result == []

    def test_iter_with_token_filter(self):
        """iter() accepts token filter."""
        inspector = Inspector()
        result = list(inspector.iter(token=0))
        assert result == []

    def test_iter_with_combined_filters(self):
        """iter() accepts multiple filters."""
        inspector = Inspector()
        result = list(inspector.iter(point=Point.LM_HEAD, layer=0, token=0))
        assert result == []


# =============================================================================
# Find Tests
# =============================================================================


class TestFind:
    """Tests for find() method."""

    def test_find_empty_returns_none(self):
        """find() on empty inspector returns None."""
        inspector = Inspector()
        result = inspector.find()
        assert result is None

    def test_find_with_filters(self):
        """find() accepts filters."""
        inspector = Inspector()
        result = inspector.find(point=Point.LM_HEAD, layer=0)
        assert result is None


# =============================================================================
# Analysis Methods Tests
# =============================================================================


class TestAnalysisMethods:
    """Tests for analysis methods."""

    def test_find_anomaly_empty_returns_none(self):
        """find_anomaly() on empty inspector returns None."""
        inspector = Inspector()
        result = inspector.find_anomaly()
        assert result is None

    def test_count_matching_empty(self):
        """count_matching() on empty inspector returns 0."""
        inspector = Inspector()
        count = inspector.count_matching()
        assert count == 0

    def test_count_matching_with_filters(self):
        """count_matching() accepts filters."""
        inspector = Inspector()
        count = inspector.count_matching(point=Point.LM_HEAD)
        assert count == 0

    def test_layer_stats_empty(self):
        """layer_stats() on empty inspector returns empty list."""
        inspector = Inspector()
        stats = inspector.layer_stats()
        assert stats == []

    def test_logits_empty(self):
        """logits() on empty inspector returns None."""
        inspector = Inspector()
        result = inspector.logits()
        assert result is None

    def test_summary_empty(self):
        """summary() on empty inspector returns valid dict."""
        inspector = Inspector()
        s = inspector.summary()
        assert s["total_captures"] == 0
        assert s["by_point"] == {}
        assert s["total_nan"] == 0
        assert s["total_inf"] == 0
        assert s["has_anomalies"] is False
        assert s["overflow"] is False


# =============================================================================
# Print Summary Test
# =============================================================================


class TestPrintSummary:
    """Tests for print_summary method."""

    def test_print_summary_no_error(self, capsys):
        """print_summary() runs without error."""
        inspector = Inspector()
        inspector.print_summary()
        captured = capsys.readouterr()
        assert "Captured 0 tensors" in captured.out


# =============================================================================
# Resource Cleanup Tests
# =============================================================================


class TestResourceCleanup:
    """Tests for Inspector resource cleanup and lifecycle."""

    def test_del_without_disable(self):
        """Inspector can be deleted without explicit disable()."""
        import gc

        inspector = Inspector()
        inspector.enable()
        # _enabled is True, __del__ should handle this

        del inspector
        gc.collect()
        # Should not crash

    def test_del_after_disable(self):
        """Inspector can be deleted after disable()."""
        import gc

        inspector = Inspector()
        inspector.enable()
        inspector.disable()

        del inspector
        gc.collect()
        # Should not crash

    def test_del_never_enabled(self):
        """Inspector can be deleted without ever being enabled."""
        import gc

        inspector = Inspector()
        # Never enable

        del inspector
        gc.collect()
        # Should not crash

    def test_repeated_enable_disable_cycles(self):
        """Multiple enable/disable cycles work correctly."""
        inspector = Inspector()
        for _ in range(10):
            inspector.enable()
            assert inspector._enabled is True
            inspector.disable()
            assert inspector._enabled is False

    def test_enable_idempotent(self):
        """Multiple enable() calls are safe."""
        inspector = Inspector()
        inspector.enable()
        inspector.enable()  # Should not cause issues
        inspector.disable()

    def test_disable_idempotent(self):
        """Multiple disable() calls are safe."""
        inspector = Inspector()
        inspector.enable()
        inspector.disable()
        inspector.disable()  # Should not cause issues

    def test_disable_without_enable(self):
        """disable() without prior enable() is safe."""
        inspector = Inspector()
        inspector.disable()  # Should not crash

    def test_clear_multiple_times(self):
        """clear() can be called multiple times."""
        inspector = Inspector()
        for _ in range(10):
            inspector.clear()
            assert inspector.count == 0

    def test_context_manager_multiple_entries(self):
        """Context manager can be used multiple times on same instance."""
        inspector = Inspector()

        with inspector:
            assert inspector._enabled is True
        assert inspector._enabled is False

        # Re-enter
        with inspector:
            assert inspector._enabled is True
        assert inspector._enabled is False


class TestMultipleInspectors:
    """Tests for multiple Inspector instances."""

    def test_multiple_inspectors_created(self):
        """Multiple inspectors can be created."""
        inspectors = [Inspector() for _ in range(5)]
        for _i, insp in enumerate(inspectors):
            assert insp._handle is not None

    def test_multiple_inspectors_cleanup(self):
        """Multiple inspectors clean up correctly."""
        import gc

        inspectors = [Inspector() for _ in range(5)]
        for insp in inspectors:
            insp.enable()
            insp.disable()

        # Delete all
        del inspectors
        gc.collect()
        # Should not crash

    def test_inspector_with_different_modes(self):
        """Inspectors with different modes coexist."""
        stats_inspector = Inspector(mode=CaptureMode.STATS)
        sample_inspector = Inspector(mode=CaptureMode.SAMPLE, sample_count=16)
        full_inspector = Inspector(mode=CaptureMode.FULL)

        assert stats_inspector._mode == CaptureMode.STATS
        assert sample_inspector._mode == CaptureMode.SAMPLE
        assert full_inspector._mode == CaptureMode.FULL

    def test_inspector_with_different_points(self):
        """Inspectors with different point configurations."""
        all_inspector = Inspector(points="all")
        logits_only = Inspector(points=[Point.LM_HEAD])
        mask_inspector = Inspector(points=0x01)

        assert all_inspector._handle is not None
        assert logits_only._handle is not None
        assert mask_inspector._handle is not None


class TestInspectorEdgeCases:
    """Edge case tests for Inspector."""

    def test_get_with_large_index(self):
        """get() with very large index returns None."""
        inspector = Inspector()
        result = inspector.get(999999)
        assert result is None

    def test_getitem_with_very_negative_index(self):
        """__getitem__ with very negative index raises IndexError."""
        inspector = Inspector()
        with pytest.raises(IndexError):
            _ = inspector[-999999]

    def test_iter_with_string_point_case_insensitive(self):
        """iter() point string is case-insensitive via Point enum."""
        inspector = Inspector()
        # Point[...upper()] is used internally
        result = list(inspector.iter(point="LM_HEAD"))
        assert result == []

    def test_layer_stats_with_specific_token(self):
        """layer_stats() accepts token parameter."""
        inspector = Inspector()
        stats = inspector.layer_stats(token=5)
        assert stats == []

    def test_logits_with_specific_token(self):
        """logits() accepts token parameter."""
        inspector = Inspector()
        result = inspector.logits(token=5)
        assert result is None

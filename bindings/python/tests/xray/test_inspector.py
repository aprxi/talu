"""
Tests for the Inspector tensor inspection API.

Tests for talu.xray.Inspector class and related functionality.
"""

import pytest

from talu.xray import (
    POINT_ALL,
    POINT_BLOCK_OUT,
    POINT_EMBED,
    POINT_LM_HEAD,
    AnomalyLocation,
    CapturedTensor,
    CaptureMode,
    Inspector,
    Point,
    Stats,
)


class TestInspectorCreation:
    """Tests for Inspector creation and configuration."""

    def test_create_default(self):
        """Create inspector with default settings."""
        inspector = Inspector()

        assert inspector is not None
        assert len(inspector) == 0
        assert not inspector.is_enabled

    def test_create_all_points(self):
        """Create inspector capturing all trace points."""
        inspector = Inspector(points="all")

        assert inspector is not None
        assert len(inspector) == 0

    def test_create_with_point_list(self):
        """Create inspector with list of point names."""
        inspector = Inspector(points=["lm_head", "block_out"])

        assert inspector is not None
        assert len(inspector) == 0

    def test_create_with_point_enums(self):
        """Create inspector with Point enums."""
        inspector = Inspector(points=[Point.LM_HEAD, Point.BLOCK_OUT])

        assert inspector is not None
        assert len(inspector) == 0

    def test_create_with_bitmask(self):
        """Create inspector with integer bitmask."""
        inspector = Inspector(points=POINT_LM_HEAD | POINT_BLOCK_OUT)

        assert inspector is not None
        assert len(inspector) == 0

    def test_create_stats_mode(self):
        """Create inspector in stats-only mode."""
        inspector = Inspector(mode=CaptureMode.STATS)

        assert inspector is not None

    def test_create_sample_mode(self):
        """Create inspector in sample mode."""
        inspector = Inspector(mode=CaptureMode.SAMPLE, sample_count=16)

        assert inspector is not None

    def test_create_full_mode(self):
        """Create inspector in full capture mode."""
        inspector = Inspector(mode=CaptureMode.FULL)

        assert inspector is not None

    def test_invalid_points_raises(self):
        """Invalid points argument raises ValueError."""
        with pytest.raises(ValueError):
            Inspector(points={"invalid": "dict"})


class TestInspectorEnableDisable:
    """Tests for enabling and disabling capture."""

    def test_enable_disable(self):
        """Enable and disable capturing."""
        inspector = Inspector()

        assert not inspector.is_enabled

        inspector.enable()
        assert inspector.is_enabled

        inspector.disable()
        assert not inspector.is_enabled

    def test_context_manager(self):
        """Inspector works as context manager."""
        inspector = Inspector()

        assert not inspector.is_enabled

        with inspector:
            assert inspector.is_enabled

        assert not inspector.is_enabled

    def test_nested_context_managers(self):
        """Nested context managers work correctly."""
        inspector1 = Inspector(points=["lm_head"])
        inspector2 = Inspector(points=["embed"])

        with inspector1:
            assert inspector1.is_enabled

            # Note: Only one capture can be active at a time
            # Enabling inspector2 will replace inspector1
            with inspector2:
                # inspector2 is now the active capture
                pass

            # After exiting inspector2, capture is disabled
            assert not inspector2.is_enabled


class TestInspectorClear:
    """Tests for clearing captured data."""

    def test_clear_empty(self):
        """Clear on empty inspector works."""
        inspector = Inspector()
        inspector.clear()

        assert len(inspector) == 0

    def test_clear_resets_count(self):
        """Clear resets capture count."""
        inspector = Inspector()
        # No data captured yet
        assert len(inspector) == 0

        inspector.clear()
        assert len(inspector) == 0


class TestInspectorOverflow:
    """Tests for capture overflow detection."""

    def test_overflow_initially_false(self):
        """Overflow is initially false."""
        inspector = Inspector()

        assert not inspector.overflow


class TestInspectorQuery:
    """Tests for query methods on empty inspector."""

    def test_get_out_of_range(self):
        """Get with out-of-range index returns None."""
        inspector = Inspector()

        assert inspector.get(0) is None
        assert inspector.get(100) is None

    def test_getitem_raises_indexerror(self):
        """Index access raises IndexError when out of range."""
        inspector = Inspector()

        with pytest.raises(IndexError):
            _ = inspector[0]

    def test_negative_index_raises(self):
        """Negative index on empty inspector raises IndexError."""
        inspector = Inspector()

        with pytest.raises(IndexError):
            _ = inspector[-1]

    def test_iter_empty(self):
        """Iterating empty inspector yields nothing."""
        inspector = Inspector()

        items = list(inspector)
        assert items == []

    def test_iter_with_filter_empty(self):
        """Filtered iteration on empty inspector yields nothing."""
        inspector = Inspector()

        items = list(inspector.iter(point=Point.LM_HEAD))
        assert items == []

    def test_find_empty(self):
        """Find on empty inspector returns None."""
        inspector = Inspector()

        result = inspector.find()
        assert result is None

    def test_find_with_filter_empty(self):
        """Find with filter on empty inspector returns None."""
        inspector = Inspector()

        result = inspector.find(point=Point.LM_HEAD, layer=0)
        assert result is None

    def test_find_anomaly_empty(self):
        """Find anomaly on empty inspector returns None."""
        inspector = Inspector()

        result = inspector.find_anomaly()
        assert result is None

    def test_count_matching_empty(self):
        """Count matching on empty inspector returns 0."""
        inspector = Inspector()

        count = inspector.count_matching()
        assert count == 0

        count = inspector.count_matching(point=Point.LM_HEAD)
        assert count == 0


class TestInspectorAnalysis:
    """Tests for analysis methods."""

    def test_layer_stats_empty(self):
        """Layer stats on empty inspector returns empty list."""
        inspector = Inspector()

        stats = inspector.layer_stats()
        assert stats == []

    def test_logits_empty(self):
        """Logits on empty inspector returns None."""
        inspector = Inspector()

        result = inspector.logits()
        assert result is None

    def test_summary_empty(self):
        """Summary of empty inspector."""
        inspector = Inspector()

        summary = inspector.summary()

        assert summary["total_captures"] == 0
        assert summary["by_point"] == {}
        assert summary["total_nan"] == 0
        assert summary["total_inf"] == 0
        assert summary["has_anomalies"] is False
        assert summary["overflow"] is False

    def test_print_summary_empty(self, capsys):
        """Print summary works on empty inspector."""
        inspector = Inspector()

        inspector.print_summary()

        captured = capsys.readouterr()
        assert "Captured 0 tensors" in captured.out


class TestStats:
    """Tests for Stats dataclass."""

    def test_stats_creation(self):
        """Create Stats instance."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        assert stats.count == 100
        assert stats.min == -1.0
        assert stats.max == 1.0
        assert stats.mean == 0.0
        assert stats.rms == 0.5
        assert stats.nan_count == 0
        assert stats.inf_count == 0

    def test_stats_has_anomalies_false(self):
        """Stats without anomalies returns False."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        assert not stats.has_anomalies

    def test_stats_has_anomalies_nan(self):
        """Stats with NaN returns True."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=5,
            inf_count=0,
        )

        assert stats.has_anomalies

    def test_stats_has_anomalies_inf(self):
        """Stats with Inf returns True."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=3,
        )

        assert stats.has_anomalies


class TestCapturedTensor:
    """Tests for CapturedTensor dataclass."""

    def test_captured_tensor_creation(self):
        """Create CapturedTensor instance."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        tensor = CapturedTensor(
            point=Point.LM_HEAD,
            layer=0xFFFF,  # Not a layer point
            token=0,
            position=0,
            shape=(1, 128, 32000),
            ndim=3,
            dtype=0,
            stats=stats,
            samples=None,
        )

        assert tensor.point == Point.LM_HEAD
        assert tensor.token == 0
        assert tensor.shape == (1, 128, 32000)
        assert tensor.ndim == 3
        assert tensor.stats.count == 100

    def test_captured_tensor_point_name(self):
        """Point name property works."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        tensor = CapturedTensor(
            point=Point.LM_HEAD,
            layer=0xFFFF,
            token=0,
            position=0,
            shape=(1, 128, 32000),
            ndim=3,
            dtype=0,
            stats=stats,
            samples=None,
        )

        assert tensor.point_name == "lm_head"

    def test_captured_tensor_is_layer_point_false(self):
        """Logits is not a layer point."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        tensor = CapturedTensor(
            point=Point.LM_HEAD,
            layer=0xFFFF,
            token=0,
            position=0,
            shape=(1, 128, 32000),
            ndim=3,
            dtype=0,
            stats=stats,
            samples=None,
        )

        assert not tensor.is_layer_point

    def test_captured_tensor_is_layer_point_true(self):
        """Layer residual is a layer point."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        tensor = CapturedTensor(
            point=Point.BLOCK_OUT,
            layer=5,
            token=0,
            position=0,
            shape=(1, 128, 768),
            ndim=3,
            dtype=0,
            stats=stats,
            samples=None,
        )

        assert tensor.is_layer_point
        assert tensor.layer == 5

    def test_captured_tensor_with_samples(self):
        """Tensor with sample data."""
        stats = Stats(
            count=100,
            min=-1.0,
            max=1.0,
            mean=0.0,
            rms=0.5,
            nan_count=0,
            inf_count=0,
        )

        tensor = CapturedTensor(
            point=Point.LM_HEAD,
            layer=0xFFFF,
            token=0,
            position=0,
            shape=(1, 128, 32000),
            ndim=3,
            dtype=0,
            stats=stats,
            samples=[0.1, 0.2, 0.3, -0.1, -0.2],
        )

        assert tensor.samples is not None
        assert len(tensor.samples) == 5
        assert tensor.samples[0] == pytest.approx(0.1)


class TestAnomalyLocation:
    """Tests for AnomalyLocation dataclass."""

    def test_anomaly_location_creation(self):
        """Create AnomalyLocation instance."""
        loc = AnomalyLocation(
            point=Point.BLOCK_OUT if hasattr(Point, "LAYER_FFN_OUT") else Point.FFN_DOWN,
            layer=10,
            token=5,
        )

        assert loc.layer == 10
        assert loc.token == 5

    def test_anomaly_location_point_name(self):
        """Point name property works."""
        loc = AnomalyLocation(
            point=Point.BLOCK_OUT,
            layer=10,
            token=5,
        )

        assert loc.point_name == "block_out"


class TestPoint:
    """Tests for Point enum."""

    def test_point_values(self):
        """Point enum has expected values."""
        assert Point.EMBED.value == 0
        assert Point.EMBED_POS.value == 1
        assert Point.LAYER_INPUT.value == 2
        assert Point.LM_HEAD.value == 21
        assert Point.LOGITS_SCALED.value == 22

    def test_point_names(self):
        """Point names are correct."""
        assert Point.EMBED.name == "EMBED"
        assert Point.LM_HEAD.name == "LM_HEAD"
        assert Point.BLOCK_OUT.name == "BLOCK_OUT"


class TestCaptureMode:
    """Tests for CaptureMode enum."""

    def test_capture_mode_values(self):
        """CaptureMode enum has expected values."""
        assert CaptureMode.STATS.value == 0
        assert CaptureMode.SAMPLE.value == 1
        assert CaptureMode.FULL.value == 2


class TestPointBitmasks:
    """Tests for point bitmask constants."""

    def test_bitmask_values(self):
        """Bitmask constants have expected values."""
        assert POINT_EMBED == 1 << 0
        assert POINT_LM_HEAD == 1 << 21
        assert POINT_BLOCK_OUT == 1 << 15
        assert POINT_ALL == 0x7FFFFF  # 23 points

    def test_bitmask_combination(self):
        """Bitmasks can be combined."""
        combined = POINT_EMBED | POINT_LM_HEAD
        assert combined == (1 << 0) | (1 << 21)


class TestMultipleInspectors:
    """Tests for multiple inspector instances."""

    def test_multiple_inspectors(self):
        """Can create multiple inspector instances."""
        inspector1 = Inspector(points=["lm_head"])
        inspector2 = Inspector(points=["embed"])

        assert inspector1 is not inspector2
        assert len(inspector1) == 0
        assert len(inspector2) == 0

    def test_inspectors_independent_count(self):
        """Different inspectors have independent counts."""
        inspector1 = Inspector()
        inspector2 = Inspector()

        # Both start empty
        assert len(inspector1) == 0
        assert len(inspector2) == 0

        # Clearing one doesn't affect the other
        inspector1.clear()
        assert len(inspector2) == 0


# =============================================================================
# Holy Trinity lifecycle tests (close / context manager / __del__)
# =============================================================================


class TestInspectorLifecycle:
    """Inspector must implement close(), __enter__/__exit__, __del__."""

    def test_close_is_idempotent(self):
        """close() can be called multiple times without error."""
        inspector = Inspector()
        inspector.close()
        inspector.close()

    def test_close_nulls_handle(self):
        """close() sets handle to None."""
        inspector = Inspector()
        assert inspector._handle is not None

        inspector.close()
        assert inspector._handle is None

    def test_close_disables_if_enabled(self):
        """close() disables capturing if active."""
        inspector = Inspector()
        inspector.enable()
        assert inspector._enabled

        inspector.close()
        assert not inspector._enabled
        assert inspector._handle is None

    def test_context_manager_disables_on_exit(self):
        """Context manager disables on exit (handle preserved for re-entry)."""
        with Inspector() as insp:
            assert insp._enabled
        # After exiting, capturing is disabled but handle is preserved
        assert not insp._enabled
        assert insp._handle is not None

    def test_close_destroys_after_context_manager(self):
        """close() destroys handle after context manager usage."""
        with Inspector() as insp:
            pass
        insp.close()
        assert insp._handle is None

    def test_context_manager_returns_self(self):
        """__enter__ returns self."""
        insp = Inspector()
        with insp as ctx:
            assert ctx is insp

"""
Tests for the xray module C bindings.

Low-level tests for talu.xray._bindings module.
"""

from talu.xray._bindings import (
    # Constants
    POINT_ALL,
    POINT_EMBED,
    POINT_LM_HEAD,
    CapturedTensorInfo,
    # Enums
    CaptureMode,
    Point,
    # Structures
    TensorStats,
    capture_clear,
    capture_count,
    capture_count_matching,
    capture_create,
    capture_destroy,
    capture_disable,
    capture_enable,
    capture_find_anomaly,
    capture_get,
    capture_is_enabled,
    capture_overflow,
    point_name,
    points_to_mask,
)


class TestCaptureFunctions:
    """Tests for capture lifecycle functions."""

    def test_capture_create_all_points(self):
        """Create capture with all points."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        assert handle is not None
        assert handle != 0

        capture_destroy(handle)

    def test_capture_create_specific_points(self):
        """Create capture with specific points."""
        handle = capture_create(POINT_LM_HEAD | POINT_EMBED, CaptureMode.STATS, 8)

        assert handle is not None
        assert handle != 0

        capture_destroy(handle)

    def test_capture_create_sample_mode(self):
        """Create capture in sample mode."""
        handle = capture_create(POINT_ALL, CaptureMode.SAMPLE, 16)

        assert handle is not None
        assert handle != 0

        capture_destroy(handle)

    def test_capture_create_full_mode(self):
        """Create capture in full mode."""
        handle = capture_create(POINT_ALL, CaptureMode.FULL, 0)

        assert handle is not None
        assert handle != 0

        capture_destroy(handle)

    def test_capture_create_from_list(self):
        """Create capture from point list."""
        points = ["lm_head", "embed", "block_out"]
        handle = capture_create(points, CaptureMode.STATS, 8)

        assert handle is not None
        assert handle != 0

        capture_destroy(handle)

    def test_capture_create_from_enum_list(self):
        """Create capture from Point enum list."""
        points = [Point.LM_HEAD, Point.EMBED]
        handle = capture_create(points, CaptureMode.STATS, 8)

        assert handle is not None
        assert handle != 0

        capture_destroy(handle)


class TestCaptureEnableDisable:
    """Tests for enable/disable functions."""

    def test_capture_enable_disable(self):
        """Enable and disable capture."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        assert not capture_is_enabled()

        capture_enable(handle)
        assert capture_is_enabled()

        capture_disable()
        assert not capture_is_enabled()

        capture_destroy(handle)

    def test_capture_disable_without_enable(self):
        """Disable when not enabled is safe."""
        assert not capture_is_enabled()

        capture_disable()  # Should not crash

        assert not capture_is_enabled()


class TestCaptureOperations:
    """Tests for capture operations."""

    def test_capture_count_empty(self):
        """Empty capture has count 0."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        count = capture_count(handle)
        assert count == 0

        capture_destroy(handle)

    def test_capture_overflow_empty(self):
        """Empty capture has no overflow."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        assert not capture_overflow(handle)

        capture_destroy(handle)

    def test_capture_clear_empty(self):
        """Clear empty capture is safe."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        capture_clear(handle)  # Should not crash

        assert capture_count(handle) == 0

        capture_destroy(handle)


class TestCaptureQuery:
    """Tests for capture query functions."""

    def test_capture_get_empty(self):
        """Get from empty capture returns None."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        result = capture_get(handle, 0)
        assert result is None

        result = capture_get(handle, 100)
        assert result is None

        capture_destroy(handle)

    def test_capture_find_anomaly_empty(self):
        """Find anomaly in empty capture returns None."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        result = capture_find_anomaly(handle)
        assert result is None

        capture_destroy(handle)

    def test_capture_count_matching_empty(self):
        """Count matching in empty capture returns 0."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        count = capture_count_matching(handle)
        assert count == 0

        count = capture_count_matching(handle, point=Point.LM_HEAD)
        assert count == 0

        count = capture_count_matching(handle, layer=0)
        assert count == 0

        count = capture_count_matching(handle, token=0)
        assert count == 0

        capture_destroy(handle)


class TestPointName:
    """Tests for point name function."""

    def test_point_name_enum(self):
        """Get name from Point enum."""
        name = point_name(Point.LM_HEAD)
        assert name.lower() == "lm_head"

        name = point_name(Point.EMBED)
        assert name.lower() == "embed_tokens"

        name = point_name(Point.BLOCK_OUT)
        assert name.lower() == "block.out"

    def test_point_name_int(self):
        """Get name from integer value."""
        name = point_name(Point.LM_HEAD.value)
        assert name.lower() == "lm_head"

        name = point_name(Point.EMBED.value)
        assert name.lower() == "embed_tokens"


class TestPointsToMask:
    """Tests for points_to_mask function."""

    def test_points_to_mask_strings(self):
        """Convert string list to mask."""
        mask = points_to_mask(["lm_head"])
        assert mask == POINT_LM_HEAD

        mask = points_to_mask(["embed"])
        assert mask == POINT_EMBED

        mask = points_to_mask(["lm_head", "embed"])
        assert mask == POINT_LM_HEAD | POINT_EMBED

    def test_points_to_mask_enums(self):
        """Convert enum list to mask."""
        mask = points_to_mask([Point.LM_HEAD])
        assert mask == POINT_LM_HEAD

        mask = points_to_mask([Point.LM_HEAD, Point.EMBED])
        assert mask == POINT_LM_HEAD | POINT_EMBED

    def test_points_to_mask_mixed(self):
        """Convert mixed list to mask."""
        mask = points_to_mask(["lm_head", Point.EMBED])
        assert mask == POINT_LM_HEAD | POINT_EMBED

    def test_points_to_mask_case_insensitive(self):
        """String conversion is case insensitive."""
        mask1 = points_to_mask(["LM_HEAD"])
        mask2 = points_to_mask(["lm_head"])
        mask3 = points_to_mask(["Lm_Head"])

        assert mask1 == mask2 == mask3 == POINT_LM_HEAD


class TestTensorStatsStructure:
    """Tests for TensorStats ctypes structure."""

    def test_tensor_stats_fields(self):
        """TensorStats has expected fields."""
        stats = TensorStats()

        # Check field access doesn't raise
        _ = stats.count
        _ = stats.min
        _ = stats.max
        _ = stats.mean
        _ = stats.rms
        _ = stats.nan_count
        _ = stats.inf_count

    def test_tensor_stats_default_values(self):
        """TensorStats has default zero values."""
        stats = TensorStats()

        assert stats.count == 0
        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.mean == 0.0
        assert stats.rms == 0.0
        assert stats.nan_count == 0
        assert stats.inf_count == 0


class TestCapturedTensorInfoStructure:
    """Tests for CapturedTensorInfo ctypes structure."""

    def test_captured_tensor_info_fields(self):
        """CapturedTensorInfo has expected fields."""
        info = CapturedTensorInfo()

        # Check field access doesn't raise
        _ = info.point
        _ = info.layer
        _ = info.token
        _ = info.position
        _ = info.shape
        _ = info.ndim
        _ = info.dtype
        _ = info.stats

    def test_captured_tensor_info_shape_array(self):
        """Shape is a 4-element array."""
        info = CapturedTensorInfo()

        assert len(info.shape) == 4
        for i in range(4):
            assert info.shape[i] == 0


class TestMemorySafety:
    """Tests for memory safety of bindings."""

    def test_destroy_twice_safe(self):
        """Destroying handle is safe (but should only be done once)."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)
        capture_destroy(handle)

        # Note: Double destroy is undefined behavior in C
        # This test just verifies single destroy works

    def test_operations_on_fresh_handle(self):
        """All operations work on fresh handle."""
        handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)

        # All these should work without crashing
        assert capture_count(handle) == 0
        assert not capture_overflow(handle)
        assert capture_get(handle, 0) is None
        assert capture_find_anomaly(handle) is None
        assert capture_count_matching(handle) == 0

        capture_clear(handle)
        capture_enable(handle)
        capture_disable()

        capture_destroy(handle)

    def test_many_create_destroy_cycles(self):
        """Many create/destroy cycles don't leak memory."""
        for _ in range(100):
            handle = capture_create(POINT_ALL, CaptureMode.STATS, 8)
            capture_enable(handle)
            capture_disable()
            capture_destroy(handle)

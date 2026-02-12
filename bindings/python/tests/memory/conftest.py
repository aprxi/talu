"""
Memory safety test fixtures.

Provides tools for:
- Memory leak detection via RSS tracking
- Reference counting verification
- Callback survival testing
- Allocation stress testing
"""

import gc
import os
import weakref
from typing import Any

import pytest

from tests.conftest import TEST_MODEL_URI_TEXT_RANDOM

# =============================================================================
# Model Discovery
# =============================================================================


@pytest.fixture(scope="module")
def test_model_path():
    """Return the model URI (TEST_MODEL_URI_TEXT env var or default)."""
    return TEST_MODEL_URI_TEXT_RANDOM


@pytest.fixture(scope="module")
def tokenizer(test_model_path):
    """Create tokenizer for memory tests."""
    from talu import Tokenizer

    return Tokenizer(test_model_path)


@pytest.fixture
def callback_ref_tracker():
    """Track callback reference survival."""

    class CallbackRefTracker:
        def __init__(self):
            self._weak_refs: list[weakref.ref] = []
            self._strong_refs: list[Any] = []

        def track_weak(self, obj) -> weakref.ref:
            """Track object weakly - should be GC'd."""
            ref = weakref.ref(obj)
            self._weak_refs.append(ref)
            return ref

        def track_strong(self, obj):
            """Track object strongly - must survive."""
            self._strong_refs.append(obj)
            return obj

        def assert_weak_collected(self):
            """Assert all weakly-tracked objects were GC'd."""
            gc.collect()
            gc.collect()
            gc.collect()
            alive = [r for r in self._weak_refs if r() is not None]
            assert not alive, f"{len(alive)} weak refs still alive"

        def assert_strong_alive(self):
            """Assert all strongly-tracked objects still exist."""
            for i, obj in enumerate(self._strong_refs):
                assert obj is not None, f"Strong ref {i} was collected"

        def clear(self):
            self._weak_refs.clear()
            self._strong_refs.clear()

    tracker = CallbackRefTracker()
    yield tracker
    tracker.clear()


@pytest.fixture
def allocation_counter():
    """Count allocations/frees for balance verification."""

    class AllocationCounter:
        def __init__(self):
            self.allocs = 0
            self.frees = 0

        @property
        def balance(self) -> int:
            """Positive = leaks, negative = double-frees."""
            return self.allocs - self.frees

        def assert_balanced(self):
            assert self.balance == 0, (
                f"Allocation imbalance: {self.allocs} allocs, {self.frees} frees"
            )

    return AllocationCounter()


@pytest.fixture
def memory_tracker():
    """Track process RSS memory for leak detection."""

    class MemoryTracker:
        def __init__(self):
            self._baseline: int | None = None

        def force_gc(self):
            """Force garbage collection."""
            gc.collect()
            gc.collect()
            gc.collect()

        def get_rss(self) -> int:
            """Get current RSS in bytes."""
            try:
                # Linux
                with open("/proc/self/statm") as f:
                    # statm: size resident shared text lib data dt
                    parts = f.read().split()
                    resident_pages = int(parts[1])
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    return resident_pages * page_size
            except (OSError, FileNotFoundError):
                # Fallback for non-Linux (less accurate)
                import resource

                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

        def capture_baseline(self):
            """Capture current memory as baseline."""
            self.force_gc()
            self._baseline = self.get_rss()

        def get_growth_mb(self) -> float:
            """Get memory growth since baseline in MB."""
            if self._baseline is None:
                raise RuntimeError("Call capture_baseline() first")
            self.force_gc()
            current = self.get_rss()
            return (current - self._baseline) / (1024 * 1024)

        def assert_no_leak(self, threshold_mb: float = 10.0, context: str = ""):
            """Assert memory growth is below threshold."""
            growth = self.get_growth_mb()
            ctx = f" in {context}" if context else ""
            assert growth < threshold_mb, (
                f"Memory leak detected{ctx}: {growth:.1f}MB growth "
                f"exceeds {threshold_mb}MB threshold"
            )

    return MemoryTracker()


@pytest.fixture
def memory_pressure_detector(memory_tracker):
    """Detect memory pressure during stress tests."""

    class MemoryPressureDetector:
        def __init__(self, tracker):
            self._tracker = tracker
            self._baseline: int | None = None
            self._threshold_mb: float = 50.0  # Default 50MB threshold

        def set_threshold_mb(self, mb: float):
            self._threshold_mb = mb

        def capture_baseline(self):
            self._tracker.force_gc()
            self._baseline = self._tracker.get_rss()

        def assert_no_pressure(self, context: str = ""):
            self._tracker.force_gc()
            current = self._tracker.get_rss()
            growth_mb = (current - self._baseline) / (1024 * 1024)
            assert growth_mb < self._threshold_mb, (
                f"Memory pressure detected{' in ' + context if context else ''}: "
                f"{growth_mb:.1f}MB growth exceeds {self._threshold_mb}MB threshold"
            )

    return MemoryPressureDetector(memory_tracker)

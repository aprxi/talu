"""
Fixtures for memory safety reference tests.

Provides memory tracking tools for leak detection during
real-model inference lifecycle tests.
"""

import gc
import os

import pytest


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
                with open("/proc/self/statm") as f:
                    parts = f.read().split()
                    resident_pages = int(parts[1])
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    return resident_pages * page_size
            except (OSError, FileNotFoundError):
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

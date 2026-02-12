"""
Simple profiling utilities for PyTorch model validation.

Provides lightweight timing instrumentation for debugging and performance analysis.
"""

import time
from typing import Optional


_PROFILER = None


class Profiler:
    """Accumulates timing statistics for named code blocks."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def record(self, name: str, dt: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + dt
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> str:
        lines = []
        for name in sorted(self.totals, key=self.totals.get, reverse=True):
            total = self.totals[name]
            count = self.counts.get(name, 0)
            avg = total / count if count else 0.0
            lines.append(f"{name}: total={total:.3f}s count={count} avg={avg:.6f}s")
        return "\n".join(lines)


class ProfileBlock:
    """Context manager for timing a code block.

    Usage:
        with ProfileBlock("attention"):
            # code to profile
    """

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        if _PROFILER is None:
            return self
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if _PROFILER is None:
            return False
        _PROFILER.record(self.name, time.perf_counter() - self.start)
        return False


def set_profiler(profiler: Optional[Profiler]) -> None:
    """Set the global profiler instance (or None to disable profiling)."""
    global _PROFILER
    _PROFILER = profiler


def get_profiler() -> Optional[Profiler]:
    """Get the current global profiler instance."""
    return _PROFILER

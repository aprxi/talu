"""
Unified Progress API for Python bindings.

Provides a ProgressRenderer class that renders progress updates from core.
Supports both terminal (TTY) and non-terminal (Jupyter, logging) environments.

Usage:
    from talu._progress import ProgressRenderer

    # Create renderer (auto-detects environment)
    renderer = ProgressRenderer()

    # Use with converter
    options = ConvertOptions()
    options.progress_callback = renderer.callback
    options.progress_user_data = renderer.user_data

    # ... call C API ...

    # Cleanup
    renderer.finish()
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._bindings import cast_to_void_p, null_void_p
from ._native import CProgressCallback
from .converter import ProgressAction
from .exceptions import TaluError

if TYPE_CHECKING:
    import ctypes


class ProgressLine:
    """State for a single progress line."""

    def __init__(self, line_id: int, label: str, total: int, message: str, unit: str):
        self.line_id = line_id
        self.label = label
        self.total = total
        self.current = 0
        self.message = message
        self.unit = unit


class ProgressRenderer:
    """
    Renders unified progress updates from core.

    This class provides a zero-dependency progress renderer that works in:
    - Terminal (TTY): Uses ANSI escape codes for in-place updates
    - Non-terminal (Jupyter, pipes): Prints each update on a new line

    The renderer maintains state for multiple progress lines and handles
    add/update/complete actions from core.

    Example:
        renderer = ProgressRenderer()
        # Set up callback for C API
        options.progress_callback = renderer.callback
        options.progress_user_data = renderer.user_data
        # ... call C API ...
        renderer.finish()
    """

    def __init__(self, output: Callable[[str], None] | None = None):
        """
        Create a progress renderer.

        Args:
            output: Optional callback for custom output. If None, uses print().
        """
        self._lines: dict[int, ProgressLine] = {}
        self._output = output or self._default_output
        self._is_tty = sys.stdout.isatty() and sys.stderr.isatty()
        self._last_line_count = 0

        # Create the ctypes callback wrapper
        # We need to keep a reference to prevent garbage collection
        self._c_callback = CProgressCallback(self._handle_update)

    @property
    def callback(self) -> ctypes.c_void_p:
        """Get the C callback function pointer."""
        return cast_to_void_p(self._c_callback)

    @property
    def user_data(self) -> ctypes.c_void_p:
        """Get the user data pointer (unused, but required by API)."""
        return null_void_p()

    def _default_output(self, text: str) -> None:
        """Print text to stdout without newline."""
        print(text, end="", flush=True)

    def _handle_update(self, update_ptr, user_data: ctypes.c_void_p) -> None:
        """Handle a progress update from core."""
        if not update_ptr:  # pragma: no cover - C core must not send null callbacks
            raise TaluError(
                "unexpected null update_ptr in progress callback", code="INTERNAL_ERROR"
            )

        # update_ptr is already POINTER(ProgressUpdate) from CProgressCallback
        update = update_ptr.contents

        line_id = update.line_id
        action = update.action

        if action == ProgressAction.ADD:
            # Add or reset a progress line
            label = update.label.decode("utf-8") if update.label else ""
            message = update.message.decode("utf-8") if update.message else ""
            unit = update.unit.decode("utf-8") if update.unit else ""
            self._lines[line_id] = ProgressLine(line_id, label, update.total, message, unit)
            self._render()

        elif action == ProgressAction.UPDATE:
            # Update an existing line
            if line_id in self._lines:
                line = self._lines[line_id]
                if update.current > 0:
                    line.current = update.current
                if update.message:
                    line.message = update.message.decode("utf-8")
                self._render()

        elif action == ProgressAction.COMPLETE:
            # Remove a completed line
            if line_id in self._lines:
                del self._lines[line_id]
            self._render()

    def _render(self) -> None:
        """Render all active progress lines."""
        if self._is_tty:
            self._render_tty()
        else:
            self._render_simple()

    def _render_tty(self) -> None:
        """Render with ANSI escape codes for in-place updates."""
        # Clear previous lines
        if self._last_line_count > 0:
            # Move cursor up and clear each line
            for _ in range(self._last_line_count):
                self._output("\033[A\033[K")

        # Render current lines
        lines = sorted(self._lines.values(), key=lambda x: x.line_id)
        for line in lines:
            self._output(self._format_line(line) + "\n")

        self._last_line_count = len(lines)

    def _render_simple(self) -> None:
        """Render for non-TTY environments (one line per update)."""
        # In non-TTY mode, just print the most recently updated line
        if self._lines:
            # Get the line with highest current value (most recently updated)
            line = max(self._lines.values(), key=lambda x: x.current)
            self._output(self._format_line(line) + "\n")

    def _format_line(self, line: ProgressLine) -> str:
        """Format a single progress line."""
        if line.total > 0:
            # Determinate progress
            bar_width = 30
            filled = int(bar_width * line.current / line.total)
            bar = "#" * filled + "-" * (bar_width - filled)
            return f"{line.label} [{bar}] {line.current}/{line.total} {line.message}"
        else:
            # Indeterminate progress (spinner)
            return f"{line.label} ... {line.message}"

    def finish(self) -> None:
        """
        Finish rendering and clean up.

        Call this after the operation completes to ensure all progress
        lines are cleared (in TTY mode).
        """
        if self._is_tty and self._last_line_count > 0:
            # Clear remaining lines
            for _ in range(self._last_line_count):
                self._output("\033[A\033[K")
        self._lines.clear()
        self._last_line_count = 0


# Convenience function for creating a progress callback
def create_progress_callback() -> tuple[ctypes.c_void_p, ctypes.c_void_p, ProgressRenderer]:
    """
    Create a progress callback and renderer.

    Returns
    -------
        Tuple of (callback_ptr, user_data_ptr, renderer)

    Example:
        callback, user_data, renderer = create_progress_callback()
        options.progress_callback = callback
        options.progress_user_data = user_data
        # ... call C API ...
        renderer.finish()
    """
    renderer = ProgressRenderer()
    return renderer.callback, renderer.user_data, renderer

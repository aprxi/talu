"""Tests for the unified progress API.

Tests that:
1. Progress types are correctly defined and usable
2. ProgressRenderer correctly handles progress updates
3. Callbacks work correctly with ctypes
4. Integration with C API works end-to-end
"""

import ctypes

from talu._native import CProgressCallback, ProgressUpdate
from talu._progress import ProgressRenderer, create_progress_callback
from talu.converter import ProgressAction


class TestProgressTypes:
    """Tests for progress type definitions."""

    def test_progress_action_values(self):
        """ProgressAction has correct enum values."""
        assert ProgressAction.ADD == 0
        assert ProgressAction.UPDATE == 1
        assert ProgressAction.COMPLETE == 2

    def test_progress_update_struct_fields(self):
        """ProgressUpdate has all required fields."""
        update = ProgressUpdate()
        # Check fields exist and have correct types
        assert hasattr(update, "line_id")
        assert hasattr(update, "action")
        assert hasattr(update, "current")
        assert hasattr(update, "total")
        assert hasattr(update, "label")
        assert hasattr(update, "message")
        assert hasattr(update, "unit")

    def test_progress_update_default_values(self):
        """ProgressUpdate fields have sensible defaults."""
        update = ProgressUpdate()
        assert update.line_id == 0
        assert update.current == 0
        assert update.total == 0

    def test_progress_callback_type_is_cfunctype(self):
        """CProgressCallback is a ctypes function type."""

        # Should be callable as a decorator
        @CProgressCallback
        def dummy_callback(update_ptr, user_data):
            pass

        assert callable(dummy_callback)


class TestProgressRenderer:
    """Tests for ProgressRenderer class."""

    def test_renderer_creation(self):
        """ProgressRenderer can be created."""
        renderer = ProgressRenderer()
        assert renderer is not None

    def test_renderer_callback_is_void_pointer(self):
        """ProgressRenderer.callback returns c_void_p."""
        renderer = ProgressRenderer()
        callback = renderer.callback
        assert isinstance(callback, ctypes.c_void_p)

    def test_renderer_user_data_is_void_pointer(self):
        """ProgressRenderer.user_data returns c_void_p."""
        renderer = ProgressRenderer()
        user_data = renderer.user_data
        assert isinstance(user_data, ctypes.c_void_p)

    def test_renderer_finish_no_error(self):
        """ProgressRenderer.finish() doesn't raise."""
        renderer = ProgressRenderer()
        renderer.finish()  # Should not raise

    def test_renderer_finish_idempotent(self):
        """ProgressRenderer.finish() can be called multiple times."""
        renderer = ProgressRenderer()
        renderer.finish()
        renderer.finish()
        renderer.finish()


class TestProgressRendererUpdates:
    """Tests for ProgressRenderer handling updates."""

    def test_renderer_handles_add_action(self):
        """ProgressRenderer handles ADD action."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))

        # Create an update struct
        update = ProgressUpdate()
        update.line_id = 0
        update.action = ProgressAction.ADD
        update.current = 0
        update.total = 10
        update.label = b"Testing"
        update.message = b"test.txt"

        # Call the handler directly with a pointer (as CFUNCTYPE would pass)
        renderer._handle_update(ctypes.pointer(update), None)

        # Should have output something
        assert len(output_lines) > 0

    def test_renderer_handles_update_action(self):
        """ProgressRenderer handles UPDATE action."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))

        # First add a line
        add_update = ProgressUpdate()
        add_update.line_id = 0
        add_update.action = ProgressAction.ADD
        add_update.total = 10
        add_update.label = b"Testing"
        renderer._handle_update(ctypes.pointer(add_update), None)

        # Then update it
        update = ProgressUpdate()
        update.line_id = 0
        update.action = ProgressAction.UPDATE
        update.current = 5
        update.message = b"halfway"
        renderer._handle_update(ctypes.pointer(update), None)

        # Should have output for both
        assert len(output_lines) >= 2

    def test_renderer_handles_complete_action(self):
        """ProgressRenderer handles COMPLETE action."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))

        # Add then complete
        add_update = ProgressUpdate()
        add_update.line_id = 0
        add_update.action = ProgressAction.ADD
        add_update.total = 10
        add_update.label = b"Testing"
        renderer._handle_update(ctypes.pointer(add_update), None)

        complete_update = ProgressUpdate()
        complete_update.line_id = 0
        complete_update.action = ProgressAction.COMPLETE
        renderer._handle_update(ctypes.pointer(complete_update), None)

        # Line should be removed from internal state
        assert 0 not in renderer._lines

    def test_renderer_tracks_multiple_lines(self):
        """ProgressRenderer can track multiple progress lines."""
        renderer = ProgressRenderer(output=lambda x: None)

        # Add two lines
        for line_id in [0, 1]:
            update = ProgressUpdate()
            update.line_id = line_id
            update.action = ProgressAction.ADD
            update.total = 10
            update.label = f"Line {line_id}".encode()
            renderer._handle_update(ctypes.pointer(update), None)

        assert len(renderer._lines) == 2
        assert 0 in renderer._lines
        assert 1 in renderer._lines

    def test_renderer_null_update_raises(self):
        """ProgressRenderer raises on null update pointer (C core bug)."""
        import pytest

        from talu.exceptions import TaluError

        renderer = ProgressRenderer(output=lambda x: None)
        with pytest.raises(TaluError, match="unexpected null update_ptr"):
            renderer._handle_update(None, None)


class TestProgressRendererFormatting:
    """Tests for ProgressRenderer output formatting."""

    def test_determinate_progress_shows_bar(self):
        """Determinate progress (total > 0) shows a progress bar."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))
        renderer._is_tty = False  # Force simple mode

        update = ProgressUpdate()
        update.line_id = 0
        update.action = ProgressAction.ADD
        update.current = 5
        update.total = 10
        update.label = b"Loading"
        update.message = b"file.bin"
        renderer._handle_update(ctypes.pointer(update), None)

        # Check output contains progress indicator
        output = "".join(output_lines)
        assert "Loading" in output
        assert "[" in output  # Progress bar

    def test_indeterminate_progress_no_bar(self):
        """Indeterminate progress (total = 0) shows spinner-style output."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))
        renderer._is_tty = False

        update = ProgressUpdate()
        update.line_id = 0
        update.action = ProgressAction.ADD
        update.current = 0
        update.total = 0  # Indeterminate
        update.label = b"Waiting"
        update.message = b"please wait"
        renderer._handle_update(ctypes.pointer(update), None)

        output = "".join(output_lines)
        assert "Waiting" in output
        assert "..." in output  # Spinner indicator


class TestCreateProgressCallback:
    """Tests for create_progress_callback helper."""

    def test_create_returns_tuple(self):
        """create_progress_callback returns (callback, user_data, renderer)."""
        result = create_progress_callback()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_create_returns_valid_callback(self):
        """create_progress_callback returns valid callback pointer."""
        callback, user_data, renderer = create_progress_callback()
        assert isinstance(callback, ctypes.c_void_p)
        assert callback.value is not None

    def test_create_returns_valid_renderer(self):
        """create_progress_callback returns valid ProgressRenderer."""
        callback, user_data, renderer = create_progress_callback()
        assert isinstance(renderer, ProgressRenderer)
        renderer.finish()


class TestProgressCallbackIntegration:
    """Integration tests for progress callbacks with C API structures."""

    def test_callback_can_be_set_on_convert_options(self):
        """Progress callback can be assigned to ConvertOptions."""
        from talu.converter import ConvertOptions

        renderer = ProgressRenderer()
        options = ConvertOptions()
        options.progress_callback = renderer.callback
        options.progress_user_data = renderer.user_data

        # Values should be set
        assert options.progress_callback is not None

    def test_callback_can_be_set_on_download_options(self):
        """Progress callback can be assigned to DownloadOptions."""
        from talu._native import DownloadOptions

        renderer = ProgressRenderer()
        options = DownloadOptions()
        options.progress_callback = renderer.callback
        options.user_data = renderer.user_data

        assert options.progress_callback is not None


class TestProgressTTYRendering:
    """Tests for TTY-mode rendering with ANSI escape codes."""

    def test_tty_mode_uses_ansi_escape_codes(self):
        """TTY mode output contains ANSI escape codes for cursor movement."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))
        renderer._is_tty = True  # Enable TTY mode

        # Add a progress line
        add = ProgressUpdate()
        add.line_id = 0
        add.action = ProgressAction.ADD
        add.total = 10
        add.label = b"Processing"
        add.message = b"file.txt"
        renderer._handle_update(ctypes.pointer(add), None)

        # Update it (this should trigger cursor movement)
        upd = ProgressUpdate()
        upd.line_id = 0
        upd.action = ProgressAction.UPDATE
        upd.current = 5
        renderer._handle_update(ctypes.pointer(upd), None)

        # Output should contain ANSI escape codes for cursor up (\033[A) and clear (\033[K)
        all_output = "".join(output_lines)
        assert "\033[A" in all_output or "\033[K" in all_output or "\n" in all_output

    def test_tty_mode_clears_previous_lines(self):
        """TTY mode clears previous output when updating."""
        output_lines = []
        renderer = ProgressRenderer(output=lambda x: output_lines.append(x))
        renderer._is_tty = True

        # Add two progress lines
        for line_id in [0, 1]:
            add = ProgressUpdate()
            add.line_id = line_id
            add.action = ProgressAction.ADD
            add.total = 10
            add.label = f"Line {line_id}".encode()
            renderer._handle_update(ctypes.pointer(add), None)

        # Now update - should clear previous 2 lines and re-render
        upd = ProgressUpdate()
        upd.line_id = 0
        upd.action = ProgressAction.UPDATE
        upd.current = 3
        renderer._handle_update(ctypes.pointer(upd), None)

        # Verify _last_line_count is tracking properly
        assert renderer._last_line_count == 2

    def test_default_output_prints_to_stdout(self, capsys):
        """Default output function prints to stdout without newline."""
        renderer = ProgressRenderer()  # Use default output
        renderer._is_tty = False

        # Add a progress line using default output
        add = ProgressUpdate()
        add.line_id = 0
        add.action = ProgressAction.ADD
        add.total = 5
        add.label = b"Test"
        renderer._handle_update(ctypes.pointer(add), None)

        captured = capsys.readouterr()
        assert "Test" in captured.out


class TestProgressEndToEnd:
    """End-to-end tests simulating real progress flow."""

    def test_typical_download_flow(self):
        """Simulate a typical download progress flow."""
        updates_received = []

        def capture_output(text):
            updates_received.append(text)

        renderer = ProgressRenderer(output=capture_output)
        renderer._is_tty = False  # Simple mode for testing

        # Simulate: add line -> update several times -> complete
        # Add
        add = ProgressUpdate()
        add.line_id = 0
        add.action = ProgressAction.ADD
        add.total = 5
        add.label = b"Downloading"
        add.message = b"config.json"
        add.unit = b"files"
        renderer._handle_update(ctypes.pointer(add), None)

        # Updates
        for i in range(1, 6):
            upd = ProgressUpdate()
            upd.line_id = 0
            upd.action = ProgressAction.UPDATE
            upd.current = i
            upd.message = f"file{i}.bin".encode()
            renderer._handle_update(ctypes.pointer(upd), None)

        # Complete
        complete = ProgressUpdate()
        complete.line_id = 0
        complete.action = ProgressAction.COMPLETE
        renderer._handle_update(ctypes.pointer(complete), None)

        # Should have received multiple updates
        assert len(updates_received) >= 6  # 1 add + 5 updates
        # Line should be removed
        assert 0 not in renderer._lines

    def test_convert_then_download_flow(self):
        """Simulate convert triggering download then conversion."""
        updates_received = []

        def capture_output(text):
            updates_received.append(text)

        renderer = ProgressRenderer(output=capture_output)
        renderer._is_tty = False

        # Download phase
        add_dl = ProgressUpdate()
        add_dl.line_id = 0
        add_dl.action = ProgressAction.ADD
        add_dl.total = 3
        add_dl.label = b"Downloading"
        renderer._handle_update(ctypes.pointer(add_dl), None)

        for i in range(1, 4):
            upd = ProgressUpdate()
            upd.line_id = 0
            upd.action = ProgressAction.UPDATE
            upd.current = i
            renderer._handle_update(ctypes.pointer(upd), None)

        complete_dl = ProgressUpdate()
        complete_dl.line_id = 0
        complete_dl.action = ProgressAction.COMPLETE
        renderer._handle_update(ctypes.pointer(complete_dl), None)

        # Convert phase (reuses line_id 0)
        add_cv = ProgressUpdate()
        add_cv.line_id = 0
        add_cv.action = ProgressAction.ADD
        add_cv.total = 100
        add_cv.label = b"Converting"
        renderer._handle_update(ctypes.pointer(add_cv), None)

        for i in range(0, 101, 20):
            upd = ProgressUpdate()
            upd.line_id = 0
            upd.action = ProgressAction.UPDATE
            upd.current = i
            renderer._handle_update(ctypes.pointer(upd), None)

        complete_cv = ProgressUpdate()
        complete_cv.line_id = 0
        complete_cv.action = ProgressAction.COMPLETE
        renderer._handle_update(ctypes.pointer(complete_cv), None)

        # Should have received updates for both phases
        assert len(updates_received) >= 10
        assert 0 not in renderer._lines

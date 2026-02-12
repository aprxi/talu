"""Tests for Hook and HookManager classes."""

from unittest.mock import MagicMock

import pytest

from talu.chat.hooks import Hook, HookManager


class TestHookBaseClass:
    """Tests for the Hook base class."""

    def test_hook_can_be_instantiated(self):
        """Hook base class can be instantiated."""
        hook = Hook()
        assert isinstance(hook, Hook)

    def test_on_generation_start_is_noop(self):
        """Default on_generation_start does nothing."""
        hook = Hook()
        mock_chat = MagicMock()
        # Should not raise
        hook.on_generation_start(mock_chat, "hello")
        hook.on_generation_start(mock_chat, "hello", config={"max_tokens": 100})

    def test_on_first_token_is_noop(self):
        """Default on_first_token does nothing."""
        hook = Hook()
        mock_chat = MagicMock()
        # Should not raise
        hook.on_first_token(mock_chat, 42.5)

    def test_on_generation_end_is_noop(self):
        """Default on_generation_end does nothing."""
        hook = Hook()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        # Should not raise
        hook.on_generation_end(mock_chat, mock_response)
        hook.on_generation_end(mock_chat, None, error=ValueError("test"))


class TestHookSubclass:
    """Tests for subclassing Hook."""

    def test_subclass_can_override_start(self):
        """Subclass can override on_generation_start."""
        calls = []

        class CustomHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append(("start", chat, input_text, config))

        hook = CustomHook()
        mock_chat = MagicMock()
        hook.on_generation_start(mock_chat, "hello", config={"key": "value"})

        assert len(calls) == 1
        assert calls[0] == ("start", mock_chat, "hello", {"key": "value"})

    def test_subclass_can_override_first_token(self):
        """Subclass can override on_first_token."""
        calls = []

        class CustomHook(Hook):
            def on_first_token(self, chat, time_ms):
                calls.append(("first_token", chat, time_ms))

        hook = CustomHook()
        mock_chat = MagicMock()
        hook.on_first_token(mock_chat, 123.45)

        assert len(calls) == 1
        assert calls[0] == ("first_token", mock_chat, 123.45)

    def test_subclass_can_override_end(self):
        """Subclass can override on_generation_end."""
        calls = []

        class CustomHook(Hook):
            def on_generation_end(self, chat, response, *, error=None):
                calls.append(("end", chat, response, error))

        hook = CustomHook()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        error = ValueError("test")

        hook.on_generation_end(mock_chat, mock_response)
        hook.on_generation_end(mock_chat, None, error=error)

        assert len(calls) == 2
        assert calls[0] == ("end", mock_chat, mock_response, None)
        assert calls[1] == ("end", mock_chat, None, error)

    def test_subclass_can_override_subset(self):
        """Subclass can override only the methods it cares about."""
        calls = []

        class PartialHook(Hook):
            def on_first_token(self, chat, time_ms):
                calls.append(time_ms)

        hook = PartialHook()
        mock_chat = MagicMock()

        # These should work (no-op from base)
        hook.on_generation_start(mock_chat, "hello")
        hook.on_generation_end(mock_chat, MagicMock())

        # This should be captured
        hook.on_first_token(mock_chat, 50.0)

        assert calls == [50.0]


class TestHookManager:
    """Tests for HookManager class."""

    def test_init_empty(self):
        """HookManager can be created without hooks."""
        manager = HookManager()
        assert manager.hooks == []

    def test_init_with_hooks(self):
        """HookManager can be created with initial hooks."""
        hook1 = Hook()
        hook2 = Hook()
        manager = HookManager([hook1, hook2])
        assert manager.hooks == [hook1, hook2]

    def test_add_hook(self):
        """Add a hook to the manager."""
        manager = HookManager()
        hook = Hook()

        manager.add(hook)

        assert hook in manager.hooks
        assert len(manager.hooks) == 1

    def test_add_multiple_hooks(self):
        """Add multiple hooks maintains order."""
        manager = HookManager()
        hook1 = Hook()
        hook2 = Hook()
        hook3 = Hook()

        manager.add(hook1)
        manager.add(hook2)
        manager.add(hook3)

        assert manager.hooks == [hook1, hook2, hook3]

    def test_remove_hook(self):
        """Remove a hook from the manager."""
        hook1 = Hook()
        hook2 = Hook()
        manager = HookManager([hook1, hook2])

        manager.remove(hook1)

        assert manager.hooks == [hook2]

    def test_remove_nonexistent_raises(self):
        """Removing a hook that doesn't exist raises ValueError."""
        manager = HookManager()
        hook = Hook()

        with pytest.raises(ValueError):
            manager.remove(hook)

    def test_hooks_returns_copy(self):
        """hooks property returns a copy, not the internal list."""
        hook = Hook()
        manager = HookManager([hook])

        hooks = manager.hooks
        hooks.append(Hook())  # Modify the returned list

        # Internal list should be unchanged
        assert len(manager.hooks) == 1


class TestHookManagerDispatch:
    """Tests for HookManager dispatch methods."""

    def test_dispatch_start_calls_all_hooks(self):
        """dispatch_start calls on_generation_start on all hooks."""
        mock_hook1 = MagicMock(spec=Hook)
        mock_hook2 = MagicMock(spec=Hook)
        manager = HookManager([mock_hook1, mock_hook2])
        mock_chat = MagicMock()

        manager.dispatch_start(mock_chat, "hello", config={"key": "value"})

        mock_hook1.on_generation_start.assert_called_once_with(
            mock_chat, "hello", config={"key": "value"}
        )
        mock_hook2.on_generation_start.assert_called_once_with(
            mock_chat, "hello", config={"key": "value"}
        )

    def test_dispatch_first_token_calls_all_hooks(self):
        """dispatch_first_token calls on_first_token on all hooks."""
        mock_hook1 = MagicMock(spec=Hook)
        mock_hook2 = MagicMock(spec=Hook)
        manager = HookManager([mock_hook1, mock_hook2])
        mock_chat = MagicMock()

        manager.dispatch_first_token(mock_chat, 123.45)

        mock_hook1.on_first_token.assert_called_once_with(mock_chat, 123.45)
        mock_hook2.on_first_token.assert_called_once_with(mock_chat, 123.45)

    def test_dispatch_end_calls_all_hooks(self):
        """dispatch_end calls on_generation_end on all hooks."""
        mock_hook1 = MagicMock(spec=Hook)
        mock_hook2 = MagicMock(spec=Hook)
        manager = HookManager([mock_hook1, mock_hook2])
        mock_chat = MagicMock()
        mock_response = MagicMock()

        manager.dispatch_end(mock_chat, mock_response)

        mock_hook1.on_generation_end.assert_called_once_with(mock_chat, mock_response, error=None)
        mock_hook2.on_generation_end.assert_called_once_with(mock_chat, mock_response, error=None)

    def test_dispatch_end_with_error(self):
        """dispatch_end passes error to hooks."""
        mock_hook = MagicMock(spec=Hook)
        manager = HookManager([mock_hook])
        mock_chat = MagicMock()
        error = ValueError("generation failed")

        manager.dispatch_end(mock_chat, None, error=error)

        mock_hook.on_generation_end.assert_called_once_with(mock_chat, None, error=error)


class TestHookManagerExceptionHandling:
    """Tests for exception handling in HookManager dispatch."""

    def test_dispatch_start_continues_on_hook_error(self):
        """dispatch_start continues to next hook if one raises."""
        calls = []

        class FailingHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                raise RuntimeError("hook error")

        class TrackingHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append("tracking")

        manager = HookManager([FailingHook(), TrackingHook()])
        mock_chat = MagicMock()

        # Should not raise
        manager.dispatch_start(mock_chat, "hello")

        # Second hook should still be called
        assert calls == ["tracking"]

    def test_dispatch_first_token_continues_on_hook_error(self):
        """dispatch_first_token continues to next hook if one raises."""
        calls = []

        class FailingHook(Hook):
            def on_first_token(self, chat, time_ms):
                raise RuntimeError("hook error")

        class TrackingHook(Hook):
            def on_first_token(self, chat, time_ms):
                calls.append(time_ms)

        manager = HookManager([FailingHook(), TrackingHook()])
        mock_chat = MagicMock()

        # Should not raise
        manager.dispatch_first_token(mock_chat, 42.0)

        # Second hook should still be called
        assert calls == [42.0]

    def test_dispatch_end_continues_on_hook_error(self):
        """dispatch_end continues to next hook if one raises."""
        calls = []

        class FailingHook(Hook):
            def on_generation_end(self, chat, response, *, error=None):
                raise RuntimeError("hook error")

        class TrackingHook(Hook):
            def on_generation_end(self, chat, response, *, error=None):
                calls.append("end")

        manager = HookManager([FailingHook(), TrackingHook()])
        mock_chat = MagicMock()

        # Should not raise
        manager.dispatch_end(mock_chat, MagicMock())

        # Second hook should still be called
        assert calls == ["end"]

    def test_all_hooks_called_even_with_multiple_failures(self):
        """All hooks are attempted even if multiple fail."""
        calls = []

        class FailingHook1(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append("fail1")
                raise RuntimeError("error 1")

        class FailingHook2(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append("fail2")
                raise RuntimeError("error 2")

        class SuccessHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append("success")

        manager = HookManager([FailingHook1(), FailingHook2(), SuccessHook()])
        mock_chat = MagicMock()

        manager.dispatch_start(mock_chat, "hello")

        # All hooks should have been attempted
        assert calls == ["fail1", "fail2", "success"]


class TestHookManagerEmptyHooks:
    """Tests for HookManager with no hooks."""

    def test_dispatch_start_with_no_hooks(self):
        """dispatch_start with no hooks does nothing."""
        manager = HookManager()
        mock_chat = MagicMock()

        # Should not raise
        manager.dispatch_start(mock_chat, "hello")

    def test_dispatch_first_token_with_no_hooks(self):
        """dispatch_first_token with no hooks does nothing."""
        manager = HookManager()
        mock_chat = MagicMock()

        # Should not raise
        manager.dispatch_first_token(mock_chat, 42.0)

    def test_dispatch_end_with_no_hooks(self):
        """dispatch_end with no hooks does nothing."""
        manager = HookManager()
        mock_chat = MagicMock()

        # Should not raise
        manager.dispatch_end(mock_chat, MagicMock())


class TestHookManagerSpecificExceptions:
    """Verify dispatch catches specific exception types, not broad Exception."""

    CAUGHT_TYPES = (RuntimeError, ValueError, TypeError, AttributeError, OSError)

    @pytest.mark.parametrize("exc_type", CAUGHT_TYPES)
    def test_dispatch_start_catches_specific_types(self, exc_type):
        """dispatch_start catches each listed exception type."""
        calls = []

        class FailHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                raise exc_type("test")

        class OkHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                calls.append("ok")

        manager = HookManager([FailHook(), OkHook()])
        manager.dispatch_start(MagicMock(), "hello")
        assert calls == ["ok"]

    @pytest.mark.parametrize("exc_type", CAUGHT_TYPES)
    def test_dispatch_first_token_catches_specific_types(self, exc_type):
        """dispatch_first_token catches each listed exception type."""
        calls = []

        class FailHook(Hook):
            def on_first_token(self, chat, time_ms):
                raise exc_type("test")

        class OkHook(Hook):
            def on_first_token(self, chat, time_ms):
                calls.append("ok")

        manager = HookManager([FailHook(), OkHook()])
        manager.dispatch_first_token(MagicMock(), 1.0)
        assert calls == ["ok"]

    @pytest.mark.parametrize("exc_type", CAUGHT_TYPES)
    def test_dispatch_end_catches_specific_types(self, exc_type):
        """dispatch_end catches each listed exception type."""
        calls = []

        class FailHook(Hook):
            def on_generation_end(self, chat, response, *, error=None):
                raise exc_type("test")

        class OkHook(Hook):
            def on_generation_end(self, chat, response, *, error=None):
                calls.append("ok")

        manager = HookManager([FailHook(), OkHook()])
        manager.dispatch_end(MagicMock(), MagicMock())
        assert calls == ["ok"]

    def test_dispatch_start_propagates_uncaught_exception(self):
        """dispatch_start lets non-listed exceptions propagate."""

        class BadHook(Hook):
            def on_generation_start(self, chat, input_text, *, config=None):
                raise KeyboardInterrupt("stop")

        manager = HookManager([BadHook()])
        with pytest.raises(KeyboardInterrupt):
            manager.dispatch_start(MagicMock(), "hello")

    def test_dispatch_first_token_propagates_uncaught_exception(self):
        """dispatch_first_token lets non-listed exceptions propagate."""

        class BadHook(Hook):
            def on_first_token(self, chat, time_ms):
                raise KeyboardInterrupt("stop")

        manager = HookManager([BadHook()])
        with pytest.raises(KeyboardInterrupt):
            manager.dispatch_first_token(MagicMock(), 1.0)

    def test_dispatch_end_propagates_uncaught_exception(self):
        """dispatch_end lets non-listed exceptions propagate."""

        class BadHook(Hook):
            def on_generation_end(self, chat, response, *, error=None):
                raise KeyboardInterrupt("stop")

        manager = HookManager([BadHook()])
        with pytest.raises(KeyboardInterrupt):
            manager.dispatch_end(MagicMock(), MagicMock())

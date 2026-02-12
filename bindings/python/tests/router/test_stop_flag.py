"""Tests for StopFlag and async cancellation.

These tests verify the StopFlag functionality for cooperative cancellation
of generation requests.
"""

from talu.router import StopFlag
from talu.router._bindings import RouterGenerateConfig


class TestStopFlagBasic:
    """Tests for StopFlag basic functionality."""

    def test_stop_flag_initial_state(self):
        """StopFlag starts as False."""
        flag = StopFlag()
        assert not flag.is_set()
        assert not bool(flag)

    def test_stop_flag_signal(self):
        """signal() sets the flag to True."""
        flag = StopFlag()
        flag.signal()
        assert flag.is_set()
        assert bool(flag)

    def test_stop_flag_reset(self):
        """reset() clears the flag back to False."""
        flag = StopFlag()
        flag.signal()
        assert flag.is_set()
        flag.reset()
        assert not flag.is_set()

    def test_stop_flag_multiple_signals(self):
        """Multiple signals are idempotent."""
        flag = StopFlag()
        flag.signal()
        flag.signal()
        flag.signal()
        assert flag.is_set()

    def test_stop_flag_ptr_stable(self):
        """ptr returns a stable memory address."""
        flag = StopFlag()
        ptr1 = flag.ptr
        ptr2 = flag.ptr
        assert ptr1 == ptr2
        assert ptr1 != 0  # Should be a valid address

    def test_stop_flag_reusable(self):
        """StopFlag can be reused after reset."""
        flag = StopFlag()

        # First use
        flag.signal()
        assert flag.is_set()

        # Reset
        flag.reset()
        assert not flag.is_set()

        # Second use
        flag.signal()
        assert flag.is_set()


class TestStopFlagWithConfig:
    """Tests for StopFlag integration with RouterGenerateConfig."""

    def test_config_without_stop_flag(self):
        """RouterGenerateConfig works without stop_flag."""
        config = RouterGenerateConfig(max_tokens=100)
        assert config.stop_flag is None

    def test_config_with_stop_flag(self):
        """RouterGenerateConfig accepts stop_flag parameter."""
        flag = StopFlag()
        config = RouterGenerateConfig(max_tokens=100, stop_flag=flag)
        # The pointer should match
        assert config.stop_flag == flag.ptr

    def test_config_stop_flag_reference_kept_alive(self):
        """Config keeps stop_flag reference alive."""
        flag = StopFlag()
        config = RouterGenerateConfig(max_tokens=100, stop_flag=flag)
        # Signal through original flag
        flag.signal()
        # The flag should still be accessible through config's internal reference
        assert config._stop_flag_ref is flag
        assert config._stop_flag_ref.is_set()


class TestFinishReasonCancelled:
    """Tests for FinishReason.CANCELLED constant."""

    def test_finish_reason_cancelled_exists(self):
        """FinishReason has CANCELLED constant."""
        from talu.types import FinishReason

        assert hasattr(FinishReason, "CANCELLED")
        assert FinishReason.CANCELLED == "cancelled"

    def test_finish_reason_all_values(self):
        """All FinishReason values are defined."""
        from talu.types import FinishReason

        assert FinishReason.EOS_TOKEN == "eos_token"
        assert FinishReason.LENGTH == "length"
        assert FinishReason.STOP_SEQUENCE == "stop_sequence"
        assert FinishReason.TOOL_CALLS == "tool_calls"
        assert FinishReason.CANCELLED == "cancelled"


# Note: Integration tests for actual cancellation during generation
# require a model and are marked with @pytest.mark.requires_model
# They would test:
# 1. Generation stops when stop_flag is signalled
# 2. finish_reason is CANCELLED when stopped via flag
# 3. stream_async properly handles asyncio.CancelledError

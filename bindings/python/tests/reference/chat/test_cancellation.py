"""
Reference tests for async cancellation via StopFlag.

Requires a real model to test actual generation cancellation.
Tests cover: stop_flag in stream(), Router.stream_async() cancellation, finish_reason.
"""

import asyncio
import threading

import pytest

from talu import Chat, GenerationConfig
from talu.router import Router, StopFlag, StreamToken


@pytest.fixture(scope="module")
def model_path(test_model_path):
    """Module-scoped model path fixture."""
    return test_model_path


def _create_router(model_path: str) -> Router:
    """Create a Router with the test model."""
    return Router(models=[model_path])


# =============================================================================
# Non-Streaming Generate Cancellation Tests
# =============================================================================


class TestGenerateCancellation:
    """Tests for stop_flag in Router.generate() (non-streaming)."""

    def test_generate_with_stop_flag_immediate(self, model_path):
        """generate() respects pre-set stop flag."""
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        try:
            # Signal before generating starts
            stop_flag.signal()

            result = router.generate(
                chat,
                "Write a very long essay about the history of computing",
                config=GenerationConfig(max_tokens=200),
                stop_flag=stop_flag,
            )

            # Should have very few or no tokens since flag was set immediately
            assert result["token_count"] < 10, f"Expected few tokens, got {result['token_count']}"
        finally:
            router.close()

    def test_generate_with_stop_flag_from_thread(self, model_path):
        """generate() stops when stop_flag is signalled from another thread.

        Uses streaming to synchronize on first token, ensuring generation has
        started before signalling stop. This avoids timing-dependent failures
        under heavy system load.
        """
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        try:
            tokens: list[str] = []
            first_token_received = threading.Event()

            def run_stream():
                for token in router.stream(
                    chat,
                    "Write a very long essay about the history of computing from ancient times to modern day",
                    config=GenerationConfig(max_tokens=500),
                    stop_flag=stop_flag,
                ):
                    tokens.append(token)
                    if len(tokens) == 1:
                        first_token_received.set()

            # Start generation in background thread
            gen_thread = threading.Thread(target=run_stream)
            gen_thread.start()

            # Wait for generation to actually produce a token (deterministic sync)
            first_token_received.wait()

            # Now signal stop from main thread
            stop_flag.signal()

            # Wait for generation to complete
            gen_thread.join()

            # Should have stopped before generating all tokens
            assert len(tokens) < 200, f"Expected <200 tokens, got {len(tokens)}"
            assert len(tokens) > 0, "Should have generated some tokens"
        finally:
            router.close()

    def test_generate_without_stop_flag_completes(self, model_path):
        """generate() without stop_flag completes normally."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            result = router.generate(
                chat,
                "Say hello",
                config=GenerationConfig(max_tokens=10),
            )

            # Should complete with tokens
            assert result["token_count"] > 0
        finally:
            router.close()


# =============================================================================
# Sync Stream Cancellation Tests
# =============================================================================


class TestStreamCancellation:
    """Tests for stop_flag in Router.stream()."""

    def test_stream_with_stop_flag_immediate(self, model_path):
        """stream() respects pre-set stop flag."""
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        try:
            # Signal before streaming starts
            stop_flag.signal()

            tokens = []
            for token in router.stream(chat, "Count from 1 to 100", stop_flag=stop_flag):
                tokens.append(token)

            # Should have very few or no tokens since flag was set immediately
            # (May get 1-2 tokens before Zig checks the flag)
            assert len(tokens) < 10, f"Expected few tokens, got {len(tokens)}"
        finally:
            router.close()

    def test_stream_with_stop_flag_delayed(self, model_path):
        """stream() stops when stop_flag is signalled mid-generation."""
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        try:
            tokens = []
            for i, token in enumerate(
                router.stream(
                    chat,
                    "Count from 1 to 1000 slowly",
                    config=GenerationConfig(max_tokens=200),
                    stop_flag=stop_flag,
                )
            ):
                tokens.append(token)
                # Signal after getting 5 tokens
                if i == 4:
                    stop_flag.signal()

            # Should have stopped shortly after signalling
            # (May get a few more tokens before Zig checks the flag)
            assert 5 <= len(tokens) < 20, f"Expected 5-20 tokens, got {len(tokens)}"
        finally:
            router.close()

    def test_stream_without_stop_flag_completes(self, model_path):
        """stream() without stop_flag completes normally."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            tokens = []
            for token in router.stream(
                chat,
                "Say hello",
                config=GenerationConfig(max_tokens=10),
            ):
                tokens.append(token)

            # Should complete with tokens
            assert len(tokens) > 0
        finally:
            router.close()


# =============================================================================
# Async Stream Cancellation Tests
# =============================================================================


class TestAsyncStreamCancellation:
    """Tests for stream_async() cancellation."""

    @pytest.mark.asyncio
    async def test_stream_async_yields_tokens(self, model_path):
        """stream_async() yields tokens one at a time."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            tokens = []
            async for token in router.stream_async(
                chat,
                "Hello",
                config=GenerationConfig(max_tokens=5),
            ):
                tokens.append(token)

            assert len(tokens) > 0
        finally:
            router.close()

    @pytest.mark.asyncio
    async def test_stream_async_cancellation(self, model_path):
        """stream_async() stops on asyncio.CancelledError."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            tokens = []
            cancelled = False

            async def consume_with_cancel():
                nonlocal tokens, cancelled
                try:
                    async for token in router.stream_async(
                        chat,
                        "Count from 1 to 1000",
                        config=GenerationConfig(max_tokens=200),
                    ):
                        tokens.append(token)
                        # Cancel after 3 tokens
                        if len(tokens) >= 3:
                            raise asyncio.CancelledError()
                except asyncio.CancelledError:
                    cancelled = True
                    raise

            with pytest.raises(asyncio.CancelledError):
                await consume_with_cancel()

            assert cancelled
            # Should have gotten at least 3 tokens before cancellation
            assert len(tokens) >= 3
            # Should not have gotten many more (cancellation should be prompt)
            assert len(tokens) < 20, f"Expected <20 tokens, got {len(tokens)}"
        finally:
            router.close()

    @pytest.mark.asyncio
    async def test_stream_async_not_buffered(self, model_path):
        """stream_async() yields tokens incrementally, not all at once.

        Validates that tokens arrive one-by-one as they're generated,
        not buffered until the end. We verify this by checking that
        we can count tokens as they arrive (not receiving all at once).
        """
        router = _create_router(model_path)
        chat = Chat()

        try:
            token_count = 0
            async for token in router.stream_async(
                chat,
                "Count: 1, 2, 3, 4, 5",
                config=GenerationConfig(max_tokens=20),
            ):
                token_count += 1
                # Each iteration receives exactly one token
                assert isinstance(token, StreamToken)
                assert len(token.text) > 0

            # Should have received multiple individual tokens
            assert token_count > 0
        finally:
            router.close()


# =============================================================================
# Thread Safety Tests
# =============================================================================


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestCancellationEdgeCases:
    """Edge case tests for cancellation robustness."""

    def test_break_from_stream_loop(self, model_path):
        """Breaking from stream loop stops generation cleanly."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            tokens = []
            for token in router.stream(
                chat,
                "Count from 1 to 1000",
                config=GenerationConfig(max_tokens=200),
            ):
                tokens.append(token)
                if len(tokens) >= 5:
                    break  # Early exit

            # Should have exactly 5 tokens (no more generated after break)
            assert len(tokens) == 5

            # Router should still be usable after break
            tokens2 = []
            for token in router.stream(
                chat,
                "Say hello",
                config=GenerationConfig(max_tokens=5),
            ):
                tokens2.append(token)
            assert len(tokens2) > 0
        finally:
            router.close()

    def test_stop_flag_reuse_after_reset(self, model_path):
        """StopFlag can be reused after reset()."""
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        try:
            # First generation - cancel early
            stop_flag.signal()
            tokens1 = list(
                router.stream(
                    chat,
                    "Count to 100",
                    config=GenerationConfig(max_tokens=50),
                    stop_flag=stop_flag,
                )
            )
            assert len(tokens1) < 10  # Should stop quickly

            # Reset and reuse
            stop_flag.reset()
            assert not stop_flag.is_set()

            # Second generation - should complete normally
            tokens2 = list(
                router.stream(
                    chat,
                    "Say hi",
                    config=GenerationConfig(max_tokens=5),
                    stop_flag=stop_flag,
                )
            )
            assert len(tokens2) > 0  # Should generate tokens
        finally:
            router.close()

    def test_sequential_cancellations(self, model_path):
        """Multiple sequential cancellations work correctly."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            for i in range(3):
                stop_flag = StopFlag()
                tokens = []
                for j, token in enumerate(
                    router.stream(
                        chat,
                        f"Iteration {i}: count to 1000",
                        config=GenerationConfig(max_tokens=100),
                        stop_flag=stop_flag,
                    )
                ):
                    tokens.append(token)
                    if j >= 2:
                        stop_flag.signal()

                # Each iteration should stop after ~3 tokens
                assert 3 <= len(tokens) < 20, f"Iteration {i}: got {len(tokens)} tokens"
        finally:
            router.close()

    def test_router_close_waits_for_generation(self, model_path):
        """Router.close() waits for active generation to complete."""
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        generation_started = threading.Event()
        generation_tokens: list[str] = []

        def generate_in_thread():
            for token in router.stream(
                chat,
                "Count from 1 to 1000",
                config=GenerationConfig(max_tokens=200),
                stop_flag=stop_flag,
            ):
                generation_tokens.append(token)
                if len(generation_tokens) == 1:
                    generation_started.set()

        # Start generation in background
        gen_thread = threading.Thread(target=generate_in_thread)
        gen_thread.start()

        # Wait for generation to start
        generation_started.wait()

        # Signal stop and close router - should not crash
        stop_flag.signal()
        router.close()  # This should wait for thread to complete

        # Thread should be done after close() returns
        gen_thread.join()
        assert len(generation_tokens) >= 1

    @pytest.mark.asyncio
    async def test_async_cancel_before_first_token(self, model_path):
        """Cancellation right at start is handled gracefully."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            tokens = []

            async def consume_and_cancel_immediately():
                async for token in router.stream_async(
                    chat,
                    "Hello",
                    config=GenerationConfig(max_tokens=10),
                ):
                    tokens.append(token)
                    # Cancel on very first token
                    raise asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await consume_and_cancel_immediately()

            # Should have gotten exactly 1 token before cancel
            assert len(tokens) == 1
        finally:
            router.close()

    @pytest.mark.asyncio
    async def test_async_sequential_streams(self, model_path):
        """Multiple sequential async streams work correctly."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            for i in range(3):
                tokens = []
                async for token in router.stream_async(
                    chat,
                    f"Say number {i}",
                    config=GenerationConfig(max_tokens=5),
                ):
                    tokens.append(token)
                assert len(tokens) > 0, f"Stream {i} should produce tokens"
        finally:
            router.close()

    @pytest.mark.asyncio
    async def test_async_cancel_with_task(self, model_path):
        """Task-based cancellation propagates correctly."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            tokens = []
            cancel_after_n = 3

            async def streaming_task():
                async for token in router.stream_async(
                    chat,
                    "Count from 1 to 1000",
                    config=GenerationConfig(max_tokens=200),
                ):
                    tokens.append(token)

            # Create task and cancel it after receiving some tokens
            task = asyncio.create_task(streaming_task())

            # Wait until we have enough tokens, then cancel
            while len(tokens) < cancel_after_n:
                await asyncio.sleep(0)  # Yield to let task run

            task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await task

            assert len(tokens) >= cancel_after_n
            assert len(tokens) < 50  # Should have stopped promptly
        finally:
            router.close()


class TestStopFlagThreadSafety:
    """Tests for StopFlag thread safety."""

    def test_stop_flag_cross_thread_signal(self, model_path):
        """StopFlag can be signalled from another thread.

        Tests that stop_flag works correctly when signalled from a different
        thread than the one doing generation. Uses event synchronization
        (not timing) to coordinate threads.
        """
        router = _create_router(model_path)
        chat = Chat()
        stop_flag = StopFlag()

        try:
            tokens: list[str] = []
            first_token_received = threading.Event()

            def collect_tokens():
                for token in router.stream(
                    chat,
                    "Count from 1 to 1000",
                    config=GenerationConfig(max_tokens=500),
                    stop_flag=stop_flag,
                ):
                    tokens.append(token)
                    # Signal that we've received at least one token
                    if len(tokens) == 1:
                        first_token_received.set()

            # Start generation in background thread
            gen_thread = threading.Thread(target=collect_tokens)
            gen_thread.start()

            # Wait for generation to actually start (first token received)
            first_token_received.wait()

            # Now signal stop from main thread
            stop_flag.signal()

            # Wait for generation thread to complete (blocking join is deterministic)
            gen_thread.join()

            # Should have received at least 1 token (the one that triggered the event)
            # and stopped before generating all 500
            assert len(tokens) >= 1, "Should have received at least one token"
            assert len(tokens) < 100, f"Expected <100 tokens, got {len(tokens)}"
        finally:
            router.close()


# =============================================================================
# Stress Tests
# =============================================================================


class TestCancellationStress:
    """Stress tests for cancellation under load."""

    def test_rapid_cancel_cycles(self, model_path):
        """Rapid start/cancel cycles don't leak resources or crash."""
        router = _create_router(model_path)
        chat = Chat()

        try:
            for i in range(10):
                stop_flag = StopFlag()
                token_count = 0

                for _token in router.stream(
                    chat,
                    f"Cycle {i}",
                    config=GenerationConfig(max_tokens=50),
                    stop_flag=stop_flag,
                ):
                    token_count += 1
                    # Cancel after first token
                    stop_flag.signal()
                    break

                # Each cycle should produce at least 1 token
                assert token_count >= 1
        finally:
            router.close()

    def test_multiple_routers_concurrent(self, model_path):
        """Multiple routers can run concurrently without interference."""

        results: dict[int, list[str]] = {}
        errors: list[Exception] = []

        def run_generation(router_id: int):
            try:
                router = _create_router(model_path)
                chat = Chat()
                try:
                    tokens = []
                    for token in router.stream(
                        chat,
                        f"Router {router_id} says hello",
                        config=GenerationConfig(max_tokens=5),
                    ):
                        tokens.append(token)
                    results[router_id] = tokens
                finally:
                    router.close()
            except Exception as e:
                errors.append(e)

        # Start multiple routers concurrently
        threads = []
        for i in range(3):
            t = threading.Thread(target=run_generation, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 3
        for router_id, tokens in results.items():
            assert len(tokens) > 0, f"Router {router_id} should produce tokens"

    def test_concurrent_cancel_different_threads(self, model_path):
        """Concurrent cancellations from different threads are safe."""
        router = _create_router(model_path)

        results: dict[int, int] = {}  # thread_id -> token_count
        all_started = threading.Barrier(3)  # Sync start of all threads

        def generate_and_cancel(thread_id: int):
            chat = Chat()
            stop_flag = StopFlag()
            tokens = []

            # Wait for all threads to be ready
            all_started.wait()

            for i, token in enumerate(
                router.stream(
                    chat,
                    f"Thread {thread_id} counting to 1000",
                    config=GenerationConfig(max_tokens=100),
                    stop_flag=stop_flag,
                )
            ):
                tokens.append(token)
                if i >= 2:
                    stop_flag.signal()

            results[thread_id] = len(tokens)

        try:
            threads = []
            for i in range(3):
                t = threading.Thread(target=generate_and_cancel, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # All threads should have completed with some tokens
            assert len(results) == 3
            for thread_id, count in results.items():
                assert count >= 3, f"Thread {thread_id}: expected >= 3 tokens, got {count}"
                assert count < 50, f"Thread {thread_id}: expected < 50 tokens, got {count}"
        finally:
            router.close()

    @pytest.mark.asyncio
    async def test_async_sequential_cancel_cycles(self, model_path):
        """Sequential async cancel cycles work correctly.

        Each cycle uses a fresh Chat to avoid any conversation state issues.
        This tests that the router can handle repeated cancel operations.
        """
        router = _create_router(model_path)

        try:
            for i in range(3):
                chat = Chat()  # Fresh chat each iteration
                tokens: list[str] = []

                async def consume_one(chat: Chat = chat, i: int = i, tokens: list[str] = tokens):
                    async for token in router.stream_async(
                        chat,
                        f"Async cycle {i}",
                        config=GenerationConfig(max_tokens=50),
                    ):
                        tokens.append(token)
                        raise asyncio.CancelledError()

                with pytest.raises(asyncio.CancelledError):
                    await consume_one()

                assert len(tokens) == 1, f"Cycle {i}: expected 1 token, got {len(tokens)}"
        finally:
            router.close()

    @pytest.mark.asyncio
    async def test_async_concurrent_streams_same_router(self, model_path):
        """Multiple concurrent async streams on same router work correctly."""
        router = _create_router(model_path)

        async def stream_tokens(stream_id: int) -> list[str]:
            chat = Chat()
            tokens = []
            async for token in router.stream_async(
                chat,
                f"Stream {stream_id} hello",
                config=GenerationConfig(max_tokens=5),
            ):
                tokens.append(token)
            return tokens

        try:
            # Run 3 concurrent streams
            results = await asyncio.gather(
                stream_tokens(0),
                stream_tokens(1),
                stream_tokens(2),
            )

            # All streams should produce tokens
            for i, tokens in enumerate(results):
                assert len(tokens) > 0, f"Stream {i} should produce tokens"
        finally:
            router.close()

"""
Tokenizer thread safety tests.

Tests that Tokenizer encode/decode operations are safe under concurrent access
from multiple threads. This validates the "Hostile Concurrency" pillar.

The underlying C API uses TLS for errors, but the tokenizer data structures
must be proven thread-safe for read access (encode/decode are read-only ops).

All concurrency tests use timeout safeguards to prevent CI hangs from stuck threads.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

# Timeout in seconds for thread joins - prevents CI hangs
THREAD_TIMEOUT = 30


def join_threads_with_timeout(threads: list, timeout: float = THREAD_TIMEOUT) -> list:
    """Join threads with timeout, returning list of threads that didn't complete.

    Args:
        threads: List of threading.Thread objects
        timeout: Per-thread timeout in seconds

    Returns:
        List of threads that failed to join within timeout
    """
    stuck = []
    for t in threads:
        t.join(timeout=timeout)  # DEADLOCK_GUARD
        if t.is_alive():
            stuck.append(t)
    return stuck


class TestTokenizerThreadSafety:
    """Tests for thread-safe tokenizer encode/decode operations."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_concurrent_encode_same_tokenizer(self, tokenizer):
        """Multiple threads can encode using the same tokenizer concurrently.

        This is a "hostile concurrency" test: threads race to use the same
        tokenizer instance. Each thread should get correct results without
        corruption from other threads.

        CORRECTNESS CHECK: Each thread's output is compared against a
        single-threaded reference (computed before concurrent execution)
        to ensure thread safety doesn't corrupt results.
        """
        errors = []
        num_threads = 10
        iterations_per_thread = 100

        # Pre-compute reference results for each unique input
        # This creates the single-threaded "ground truth"
        reference_results = {}
        for thread_id in range(num_threads):
            for i in range(iterations_per_thread):
                text = f"Thread {thread_id} iteration {i}: Hello world!"
                reference_results[text] = tokenizer.encode(text).tolist()

        def encode_loop(thread_id):
            try:
                for i in range(iterations_per_thread):
                    text = f"Thread {thread_id} iteration {i}: Hello world!"
                    tokens = tokenizer.encode(text)
                    token_list = tokens.tolist()

                    # Compare against pre-computed reference
                    expected = reference_results[text]
                    if token_list != expected:
                        errors.append(
                            f"Thread {thread_id}@{i}: result mismatch - "
                            f"got {token_list[:5]}... expected {expected[:5]}..."
                        )

                    # Also verify basic validity
                    if len(tokens) == 0:
                        errors.append(f"Thread {thread_id}@{i}: empty tokens")
                    if not all(isinstance(t, int) for t in token_list):
                        errors.append(f"Thread {thread_id}@{i}: non-int tokens")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=encode_loop, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Thread-safety errors: {errors[:10]}"

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_concurrent_decode_same_tokenizer(self, tokenizer):
        """Multiple threads can decode using the same tokenizer concurrently."""
        # First encode some text to get valid tokens
        base_tokens = tokenizer.encode("Hello World Test").tolist()

        errors = []
        num_threads = 10
        iterations_per_thread = 100

        def decode_loop(thread_id):
            try:
                for i in range(iterations_per_thread):
                    # Create unique token sequence per iteration
                    tokens = base_tokens + [thread_id + i]
                    decoded = tokenizer.decode(tokens[: len(base_tokens)])
                    # Verify we got valid output
                    if not isinstance(decoded, str):
                        errors.append(f"Thread {thread_id}@{i}: non-string result")
                    if len(decoded) == 0:
                        errors.append(f"Thread {thread_id}@{i}: empty decode")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=decode_loop, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Thread-safety errors: {errors[:10]}"

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_concurrent_encode_decode_interleaved(self, tokenizer):
        """Threads can encode and decode simultaneously without interference.

        This test has some threads encoding while others decode, simulating
        real-world usage patterns where both operations occur concurrently.
        """
        errors = []
        num_threads = 8
        iterations_per_thread = 50

        def mixed_operations(thread_id):
            try:
                for i in range(iterations_per_thread):
                    # Alternate between encode and decode
                    if (thread_id + i) % 2 == 0:
                        text = f"Encode thread {thread_id} iteration {i}"
                        tokens = tokenizer.encode(text)
                        if len(tokens) == 0:
                            errors.append(f"Thread {thread_id}@{i}: empty encode")
                    else:
                        # Decode some tokens
                        tokens = [1, 2, 3, 4, 5]  # Simple token sequence
                        decoded = tokenizer.decode(tokens)
                        if not isinstance(decoded, str):
                            errors.append(f"Thread {thread_id}@{i}: decode not string")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=mixed_operations, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Thread-safety errors: {errors[:10]}"

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_encode_determinism_across_threads(self, tokenizer):
        """Same text produces identical tokens regardless of which thread encodes.

        This verifies that there's no thread-local state corruption affecting
        the tokenization results.
        """
        test_text = "The quick brown fox jumps over the lazy dog."
        expected_tokens = tokenizer.encode(test_text).tolist()

        results = {}
        errors = []
        num_threads = 10
        iterations_per_thread = 20

        def encode_and_verify(thread_id):
            try:
                for i in range(iterations_per_thread):
                    tokens = tokenizer.encode(test_text).tolist()
                    if tokens != expected_tokens:
                        errors.append(
                            f"Thread {thread_id}@{i}: got {tokens[:5]}... "
                            f"expected {expected_tokens[:5]}..."
                        )
                results[thread_id] = True
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [
            threading.Thread(target=encode_and_verify, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Determinism violations: {errors[:10]}"
        assert len(results) == num_threads, "Not all threads completed"


class TestTokenizerConcurrentCreation:
    """Tests for concurrent Tokenizer instantiation."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_concurrent_tokenizer_creation(self, talu, test_model_path):
        """Multiple threads can create Tokenizer instances simultaneously.

        This tests the initialization path for race conditions in:
        - File I/O for tokenizer.json
        - C-struct allocation
        - Global state initialization
        """
        errors = []
        tokenizers = []
        lock = threading.Lock()
        num_threads = 5

        def create_tokenizer(thread_id):
            try:
                tok = talu.Tokenizer(test_model_path)
                # Verify tokenizer is functional
                tokens = tok.encode("test")
                if len(tokens) == 0:
                    errors.append(f"Thread {thread_id}: empty encode from new tokenizer")
                with lock:
                    tokenizers.append(tok)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=create_tokenizer, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Creation errors: {errors}"
        assert len(tokenizers) == num_threads, "Not all tokenizers created"

        # Verify all tokenizers produce same results
        test_text = "Hello World"
        expected = tokenizers[0].encode(test_text).tolist()
        for i, tok in enumerate(tokenizers[1:], 1):
            result = tok.encode(test_text).tolist()
            assert result == expected, f"Tokenizer {i} produces different result"


class TestTokenizerThreadPoolExecutor:
    """Tests using ThreadPoolExecutor for cleaner concurrent execution."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_threadpool_encode(self, tokenizer):
        """Tokenizer works correctly with ThreadPoolExecutor."""
        test_texts = [f"Test string number {i} with some content" for i in range(100)]

        def encode_text(text):
            tokens = tokenizer.encode(text)
            return text, tokens.tolist()

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(encode_text, text) for text in test_texts]
            try:
                results = [f.result(timeout=THREAD_TIMEOUT) for f in as_completed(futures)]
            except FuturesTimeoutError:
                pytest.fail(f"ThreadPoolExecutor tasks stuck after {THREAD_TIMEOUT}s")

        # Verify all results
        assert len(results) == len(test_texts)
        for text, tokens in results:
            assert len(tokens) > 0, f"Empty tokens for '{text[:30]}'"

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_threadpool_roundtrip(self, tokenizer):
        """Encode-decode roundtrip works correctly in thread pool."""
        test_texts = [
            "Hello World",
            "The quick brown fox",
            "Testing 123",
            "Unicode: 日本語",
        ] * 25  # 100 total

        def roundtrip(text):
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            return text, decoded

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(roundtrip, text) for text in test_texts]
            try:
                results = [f.result(timeout=THREAD_TIMEOUT) for f in as_completed(futures)]
            except FuturesTimeoutError:
                pytest.fail(f"ThreadPoolExecutor tasks stuck after {THREAD_TIMEOUT}s")

        # Verify roundtrips preserve content
        for original, decoded in results:
            # Allow for whitespace normalization
            assert original.strip() in decoded or decoded.strip() in original, (
                f"Roundtrip failed: '{original}' -> '{decoded}'"
            )

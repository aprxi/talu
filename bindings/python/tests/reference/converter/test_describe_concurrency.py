"""
Concurrent describe() tests.

Tests that talu.converter.describe() is safe under concurrent access.
This validates the "Hostile Concurrency" pillar.

Focus areas:
- Concurrent describe() calls on the same model path
- Concurrent describe() calls on different model paths
- File I/O and C-struct allocation race conditions

All concurrency tests use timeout safeguards to prevent CI hangs.
"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

from talu.converter import describe
from tests.reference.helpers import create_minimal_model

# Timeout in seconds for thread joins - prevents CI hangs
THREAD_TIMEOUT = 30


def join_threads_with_timeout(threads: list, timeout: float = THREAD_TIMEOUT) -> list:
    """Join threads with timeout, returning list of threads that didn't complete."""
    stuck = []
    for t in threads:
        t.join(timeout=timeout)  # DEADLOCK_GUARD: fail fast if thread hangs
        if t.is_alive():
            stuck.append(t)
    return stuck


class TestConcurrentDescribeSamePath:
    """Tests for concurrent describe() on the same model path."""

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_concurrent_describe_same_model(self, test_model_path):
        """Multiple threads can describe() the same model simultaneously.

        This is a "hostile concurrency" test: threads race to describe
        the same model. Each thread should get identical, correct results.
        """
        results = {}
        errors = []
        lock = threading.Lock()
        num_threads = 10
        iterations_per_thread = 20

        def describe_loop(thread_id):
            try:
                for i in range(iterations_per_thread):
                    info = describe(test_model_path)
                    # Store first result for comparison
                    if thread_id == 0 and i == 0:
                        with lock:
                            results["reference"] = {
                                "vocab_size": info.vocab_size,
                                "hidden_size": info.hidden_size,
                                "num_layers": info.num_layers,
                                "num_heads": info.num_heads,
                            }
                    else:
                        # Compare against reference
                        with lock:
                            if "reference" in results:
                                ref = results["reference"]
                                if info.vocab_size != ref["vocab_size"]:
                                    errors.append(f"Thread {thread_id}@{i}: vocab_size mismatch")
                                if info.hidden_size != ref["hidden_size"]:
                                    errors.append(f"Thread {thread_id}@{i}: hidden_size mismatch")
                with lock:
                    results[thread_id] = True
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=describe_loop, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Concurrency errors: {errors[:10]}"
        # All threads plus reference should have recorded results
        assert len(results) >= num_threads, "Not all threads completed"


class TestConcurrentDescribeDeterministic:
    """Deterministic concurrent tests using synthetic models."""

    @pytest.mark.slow
    def test_concurrent_describe_synthetic_model(self):
        """Concurrent describe() on synthetic model returns consistent results."""
        # Create a model with known configuration
        model = create_minimal_model(
            vocab_size=5000,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
        )
        model_path = str(model.path)

        results = []
        errors = []
        lock = threading.Lock()
        num_threads = 8
        iterations_per_thread = 10

        def describe_loop(thread_id):
            try:
                for _ in range(iterations_per_thread):
                    info = describe(model_path)
                    with lock:
                        results.append(
                            {
                                "thread": thread_id,
                                "vocab_size": info.vocab_size,
                                "hidden_size": info.hidden_size,
                                "num_layers": info.num_layers,
                                "num_heads": info.num_heads,
                            }
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=describe_loop, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors: {errors}"

        # Verify all results match expected values
        expected_total = num_threads * iterations_per_thread
        assert len(results) == expected_total, (
            f"Expected {expected_total} results, got {len(results)}"
        )

        for r in results:
            assert r["vocab_size"] == 5000, f"Thread {r['thread']}: vocab_size wrong"
            assert r["hidden_size"] == 256, f"Thread {r['thread']}: hidden_size wrong"
            assert r["num_layers"] == 4, f"Thread {r['thread']}: num_layers wrong"
            assert r["num_heads"] == 8, f"Thread {r['thread']}: num_heads wrong"

    @pytest.mark.slow
    def test_concurrent_describe_different_models(self):
        """Concurrent describe() on different synthetic models."""
        # Create multiple models with different configurations
        models = [
            create_minimal_model(
                vocab_size=1000 + i * 500,
                hidden_size=64 + i * 32,
                num_layers=i + 1,
                num_heads=2 ** (i + 1),
            )
            for i in range(4)
        ]

        results = {}
        errors = []
        lock = threading.Lock()

        def describe_model(model_id, model):
            try:
                info = describe(str(model.path))
                with lock:
                    results[model_id] = {
                        "vocab_size": info.vocab_size,
                        "hidden_size": info.hidden_size,
                        "num_layers": info.num_layers,
                        "num_heads": info.num_heads,
                    }
            except Exception as e:
                errors.append(f"Model {model_id}: {e}")

        threads = [
            threading.Thread(target=describe_model, args=(i, model))
            for i, model in enumerate(models)
        ]

        # Start all threads at once to maximize concurrency
        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == len(models), "Not all models described"

        # Verify each result matches its expected configuration
        for i, _model in enumerate(models):
            r = results[i]
            expected_vocab = 1000 + i * 500
            expected_hidden = 64 + i * 32
            expected_layers = i + 1
            expected_heads = 2 ** (i + 1)

            assert r["vocab_size"] == expected_vocab, f"Model {i}: vocab_size wrong"
            assert r["hidden_size"] == expected_hidden, f"Model {i}: hidden_size wrong"
            assert r["num_layers"] == expected_layers, f"Model {i}: num_layers wrong"
            assert r["num_heads"] == expected_heads, f"Model {i}: num_heads wrong"


class TestThreadPoolDescribe:
    """Tests using ThreadPoolExecutor for describe() operations."""

    @pytest.mark.slow
    def test_threadpool_describe_synthetic(self):
        """describe() works correctly with ThreadPoolExecutor."""
        model = create_minimal_model(
            vocab_size=3000,
            hidden_size=128,
            num_layers=3,
            num_heads=4,
        )
        model_path = str(model.path)

        def describe_task(iteration):
            info = describe(model_path)
            return iteration, info.vocab_size, info.hidden_size

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(describe_task, i) for i in range(50)]
            try:
                # DEADLOCK_GUARD: fail fast if tasks hang
                results = [f.result(timeout=THREAD_TIMEOUT) for f in as_completed(futures)]
            except FuturesTimeoutError:
                pytest.fail(f"ThreadPoolExecutor tasks stuck after {THREAD_TIMEOUT}s")

        assert len(results) == 50
        for iteration, vocab_size, hidden_size in results:
            assert vocab_size == 3000, f"Iteration {iteration}: wrong vocab_size"
            assert hidden_size == 128, f"Iteration {iteration}: wrong hidden_size"

    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_threadpool_describe_real_model(self, test_model_path):
        """describe() on real model works correctly with ThreadPoolExecutor."""
        # Get reference values
        ref_info = describe(test_model_path)

        def describe_task(iteration):
            info = describe(test_model_path)
            return (
                iteration,
                info.vocab_size,
                info.hidden_size,
                info.num_layers,
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(describe_task, i) for i in range(30)]
            try:
                # DEADLOCK_GUARD: fail fast if tasks hang
                results = [f.result(timeout=THREAD_TIMEOUT) for f in as_completed(futures)]
            except FuturesTimeoutError:
                pytest.fail(f"ThreadPoolExecutor tasks stuck after {THREAD_TIMEOUT}s")

        assert len(results) == 30
        for iteration, vocab_size, hidden_size, num_layers in results:
            assert vocab_size == ref_info.vocab_size, f"Iteration {iteration}: vocab_size mismatch"
            assert hidden_size == ref_info.hidden_size, (
                f"Iteration {iteration}: hidden_size mismatch"
            )
            assert num_layers == ref_info.num_layers, f"Iteration {iteration}: num_layers mismatch"


class TestDescribeFileIOConcurrency:
    """Tests for file I/O race conditions in describe()."""

    @pytest.mark.slow
    def test_rapid_describe_same_file(self):
        """Rapid repeated describe() calls don't cause file handle races.

        CORRECTNESS CHECK: Each thread compares result against known
        configuration from SyntheticModel to detect data corruption.
        """
        # Create model with explicit known configuration
        expected_vocab_size = 1000
        expected_hidden_size = 64
        model = create_minimal_model(
            vocab_size=expected_vocab_size,
            hidden_size=expected_hidden_size,
        )
        model_path = str(model.path)

        errors = []
        num_threads = 4
        iterations = 100

        def rapid_describe(thread_id):
            try:
                for i in range(iterations):
                    info = describe(model_path)
                    # Compare against known SyntheticModel configuration
                    # This catches data corruption, not just "is positive"
                    if info.vocab_size != expected_vocab_size:
                        errors.append(
                            f"Thread {thread_id}@{i}: vocab_size {info.vocab_size} "
                            f"!= expected {expected_vocab_size}"
                        )
                    if info.hidden_size != expected_hidden_size:
                        errors.append(
                            f"Thread {thread_id}@{i}: hidden_size {info.hidden_size} "
                            f"!= expected {expected_hidden_size}"
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=rapid_describe, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"File I/O race errors: {errors[:10]}"

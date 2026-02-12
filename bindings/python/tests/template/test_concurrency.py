"""
Concurrency and thread safety tests for Template.

Tests that Template rendering is safe under concurrent access from multiple threads.
This validates the "Hostile Concurrency" pillar from the testing guidelines.

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


class TestThreadSafety:
    """Tests for thread-safe template rendering."""

    @pytest.mark.slow
    def test_concurrent_render_same_template(self, Template):
        """Multiple threads can render the same template concurrently."""
        template = Template("Hello {{ name }}!")
        results = []
        errors = []
        num_threads = 10
        iterations_per_thread = 100

        def render_loop(thread_id):
            thread_results = []
            try:
                for i in range(iterations_per_thread):
                    name = f"Thread{thread_id}_{i}"
                    result = template(name=name)
                    expected = f"Hello {name}!"
                    if result != expected:
                        errors.append(f"Thread {thread_id}: expected '{expected}', got '{result}'")
                    thread_results.append(result)
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")
            return thread_results

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=lambda tid=i: results.extend(render_loop(tid)))
            threads.append(t)

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors during concurrent rendering: {errors}"
        assert len(results) == num_threads * iterations_per_thread

    @pytest.mark.slow
    def test_concurrent_render_different_templates(self, Template):
        """Multiple threads can create and render different templates concurrently."""
        errors = []
        num_threads = 10

        def create_and_render(thread_id):
            try:
                template = Template(f"Template {{{{ id }}}} from thread {thread_id}")
                for i in range(50):
                    result = template(id=i)
                    expected = f"Template {i} from thread {thread_id}"
                    if result != expected:
                        errors.append(f"Thread {thread_id}: expected '{expected}', got '{result}'")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=create_and_render, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors during concurrent operations: {errors}"

    @pytest.mark.slow
    def test_concurrent_complex_templates(self, Template):
        """Complex templates with control flow work correctly under concurrency."""
        template = Template("""
{% for item in items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
Total: {{ items | length }}
""")
        errors = []
        num_threads = 8

        def render_complex(thread_id):
            try:
                for _ in range(20):
                    items = [
                        {"name": f"item_{thread_id}_{j}", "value": j * thread_id} for j in range(5)
                    ]
                    result = template(items=items)
                    # Verify result contains expected data
                    if f"item_{thread_id}_0" not in result:
                        errors.append(f"Thread {thread_id}: missing expected item in result")
                    if "Total: 5" not in result:
                        errors.append(f"Thread {thread_id}: missing total count")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=render_complex, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors during concurrent complex rendering: {errors}"

    @pytest.mark.slow
    def test_thread_pool_executor(self, Template):
        """Template works correctly with ThreadPoolExecutor."""
        template = Template("{{ x }} * {{ y }} = {{ x * y }}")

        def render(args):
            x, y = args
            result = template(x=x, y=y)
            expected = f"{x} * {y} = {x * y}"
            return result == expected, result, expected

        inputs = [(i, j) for i in range(10) for j in range(10)]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(render, inp) for inp in inputs]
            try:
                # Use timeout for future results to prevent hangs
                results = [f.result(timeout=THREAD_TIMEOUT) for f in as_completed(futures)]
            except FuturesTimeoutError:
                pytest.fail(f"ThreadPoolExecutor tasks stuck after {THREAD_TIMEOUT}s timeout")

        failures = [(r, e) for success, r, e in results if not success]
        assert len(failures) == 0, f"Failures: {failures[:5]}"  # Show first 5


class TestReentrantRendering:
    """Tests for re-entrant template rendering."""

    def test_nested_template_calls(self, Template):
        """A template can be rendered while another is being rendered."""
        outer = Template("Outer: {{ inner_result }}")
        inner = Template("Inner: {{ value }}")

        # Simulate nested rendering
        inner_result = inner(value="test")
        result = outer(inner_result=inner_result)

        assert result == "Outer: Inner: test"

    def test_template_reuse_sequential(self, Template):
        """Template can be reused many times sequentially."""
        template = Template("{{ n }}")

        for i in range(1000):
            result = template(n=i)
            assert result == str(i), f"Iteration {i}: expected '{i}', got '{result}'"


class TestStressConditions:
    """Stress tests for template rendering."""

    @pytest.mark.slow
    def test_rapid_create_destroy(self, Template):
        """Rapid template creation and destruction doesn't cause issues."""
        errors = []

        def create_destroy_loop(iterations):
            try:
                for i in range(iterations):
                    t = Template(f"Test {{{{ x }}}} #{i}")
                    result = t(x=i)
                    if f"Test {i} #{i}" != result:
                        errors.append(f"Mismatch at {i}")
            except Exception as e:
                errors.append(str(e))

        threads = []
        for _ in range(4):
            t = threading.Thread(target=create_destroy_loop, args=(100,))
            threads.append(t)

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors: {errors}"

    @pytest.mark.slow
    def test_large_variable_count(self, Template):
        """Template handles many variables correctly under concurrent access."""
        # Create template with many variables
        var_names = [f"var_{i}" for i in range(50)]
        template_str = " ".join(f"{{{{ {name} }}}}" for name in var_names)
        template = Template(template_str)

        errors = []

        def render_many_vars(thread_id):
            try:
                for iteration in range(20):
                    values = {
                        name: f"{thread_id}_{iteration}_{i}" for i, name in enumerate(var_names)
                    }
                    result = template(**values)
                    # Verify first and last values are present
                    if values["var_0"] not in result or values["var_49"] not in result:
                        errors.append(f"Thread {thread_id}: missing expected values")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = []
        for i in range(4):
            t = threading.Thread(target=render_many_vars, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors: {errors}"


class TestTemplateReuseAcrossThreads:
    """Tests for same template object reused across threads."""

    @pytest.mark.slow
    def test_single_template_many_threads(self, Template):
        """Single template instance is safe to use from many threads."""
        # Create ONE template instance
        shared_template = Template("Thread {{ tid }} iteration {{ i }}: {{ tid * i }}")

        results = {}
        errors = []
        num_threads = 10
        iterations = 100

        def worker(thread_id):
            local_results = []
            try:
                for i in range(iterations):
                    result = shared_template(tid=thread_id, i=i)
                    expected = f"Thread {thread_id} iteration {i}: {thread_id * i}"
                    if result != expected:
                        errors.append(f"T{thread_id}@{i}: '{result}' != '{expected}'")
                    local_results.append(result)
                results[thread_id] = local_results
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=worker, args=(tid,))
            threads.append(t)

        # Start all threads at roughly the same time
        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        # Verify no errors
        assert len(errors) == 0, f"Thread-safety errors: {errors[:10]}"
        # Verify all results collected
        assert len(results) == num_threads
        for tid in range(num_threads):
            assert len(results[tid]) == iterations

    @pytest.mark.slow
    def test_template_with_loop_reused(self, Template):
        """Template with loop.index is thread-safe."""
        shared_template = Template(
            "{% for item in items %}{{ loop.index }}:{{ item }} {% endfor %}"
        )

        errors = []
        num_threads = 8

        def worker(thread_id):
            try:
                for _ in range(50):
                    items = [f"t{thread_id}_a", f"t{thread_id}_b", f"t{thread_id}_c"]
                    result = shared_template(items=items)
                    # Each thread's items should have correct loop indices
                    if "1:" not in result or "2:" not in result or "3:" not in result:
                        errors.append(f"T{thread_id}: loop.index incorrect")
                    if f"t{thread_id}_a" not in result:
                        errors.append(f"T{thread_id}: item corruption")
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Errors: {errors[:10]}"

    @pytest.mark.slow
    def test_template_with_set_reused(self, Template):
        """Template with {% set %} is thread-safe (no state leakage)."""
        shared_template = Template("{% set x = value * 2 %}{% set y = x + 1 %}{{ y }}")

        errors = []
        num_threads = 8

        def worker(thread_id):
            try:
                for i in range(50):
                    value = thread_id * 100 + i
                    expected = str((value * 2) + 1)
                    result = shared_template(value=value)
                    if result != expected:
                        errors.append(f"T{thread_id}@{i}: {result} != {expected}")
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"State leakage errors: {errors[:10]}"

    def test_concurrent_namespace_modification(self, Template):
        """Multiple threads modifying namespace() objects have isolated state.

        This is a "hostile concurrency" test: threads try to corrupt each other's
        namespace state by simultaneously modifying the same attribute name.
        Each thread should see only its own modifications.
        """
        # Template that uses namespace with accumulator pattern
        shared_template = Template("""
{%- set ns = namespace(counter=0, items=[]) -%}
{%- for i in range(iterations) -%}
{%- set ns.counter = ns.counter + increment -%}
{%- endfor -%}
{{ ns.counter }}
""")

        errors = []
        num_threads = 8
        iterations = 20

        def worker(thread_id):
            try:
                increment = thread_id + 1  # Each thread has unique increment
                expected = str(iterations * increment)
                result = shared_template(iterations=iterations, increment=increment).strip()
                if result != expected:
                    errors.append(
                        f"T{thread_id}: namespace corruption - expected {expected}, got {result}"
                    )
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, (
            f"Namespace isolation failures (state corruption between threads): {errors[:5]}"
        )

    def test_concurrent_set_variable_isolation(self, Template):
        """Concurrent {% set %} on same variable name are isolated per-render.

        Each thread sets a variable with the same name but different value.
        No thread should see another thread's value.
        """
        shared_template = Template("{% set result = input_value * 3 %}{{ result }}")

        errors = []
        results_seen = []
        num_threads = 10

        def worker(thread_id):
            try:
                input_val = thread_id * 7 + 13  # Unique per thread
                expected = str(input_val * 3)
                for _ in range(30):
                    result = shared_template(input_value=input_val)
                    results_seen.append((thread_id, result))
                    if result != expected:
                        errors.append(
                            f"T{thread_id}: set variable leaked - expected {expected}, got {result}"
                        )
            except Exception as e:
                errors.append(f"T{thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()

        stuck = join_threads_with_timeout(threads)
        if stuck:
            pytest.fail(f"{len(stuck)} threads stuck after {THREAD_TIMEOUT}s timeout")

        assert len(errors) == 0, f"Set variable isolation failures: {errors[:5]}"

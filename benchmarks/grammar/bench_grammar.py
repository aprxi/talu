"""
Python grammar benchmarks - tests FFI overhead + core performance.

Run with: uv run python benchmarks/grammar/bench_grammar.py

For hot-path benchmark (requires model):
  uv run python benchmarks/grammar/bench_grammar.py --hot-path <model_path>
"""

import argparse
import time
from pathlib import Path

from pydantic import BaseModel

from talu.grammar import Grammar


class SimpleSchema(BaseModel):
    name: str
    age: int


def bench_grammar_creation():
    """Benchmark grammar compilation from schema."""
    iterations = 100

    start = time.perf_counter()
    for _ in range(iterations):
        g = Grammar(schema=SimpleSchema)
        del g
    elapsed = time.perf_counter() - start

    print(f"Grammar creation: {elapsed/iterations*1000:.2f} ms/iter")


def bench_validation():
    """Benchmark complete validation."""
    g = Grammar(schema=SimpleSchema)
    valid_json = '{"name":"Alice","age":30}'
    invalid_json = '{"name":123}'
    iterations = 100

    # Valid JSON
    start = time.perf_counter()
    for _ in range(iterations):
        result = g.validate(valid_json)
        assert result is True
    elapsed = time.perf_counter() - start

    print(f"validate valid JSON ({len(valid_json)} bytes): {elapsed/iterations*1000:.3f} ms/iter")

    # Invalid JSON
    start = time.perf_counter()
    for _ in range(iterations):
        result = g.validate(invalid_json)
        assert result is False
    elapsed = time.perf_counter() - start

    print(f"validate invalid JSON ({len(invalid_json)} bytes): {elapsed/iterations*1000:.3f} ms/iter")


def bench_streaming():
    """Benchmark streaming validation."""
    g = Grammar(schema=SimpleSchema)
    iterations = 100

    start = time.perf_counter()
    for _ in range(iterations):
        g.reset()
        g.accept('{"name":"')
        g.accept('test')
        g.accept('","age":25}')
        assert g.is_complete
    elapsed = time.perf_counter() - start

    print(f"Streaming validation (3 chunks): {elapsed/iterations*1000:.3f} ms/iter")


def bench_get_valid_tokens(model_path: str):
    """
    Benchmark get_valid_tokens - THE ACTUAL HOT PATH.

    This is called once per generated token during structured output.
    It's the primary source of grammar overhead.
    """
    from talu import Tokenizer

    print(f"\nLoading tokenizer from {model_path}...")
    tokenizer = Tokenizer(model_path)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Run realistic scenarios
    bench_realistic_schemas(tokenizer)
    bench_cache_effects(tokenizer)
    bench_token_by_token_generation(tokenizer)


def bench_can_accept_scaling():
    """
    Benchmark can_accept with different token lengths.

    This is called O(vocab_size) times per token inside get_valid_tokens.
    """
    g = Grammar(schema=SimpleSchema)
    g.accept('{"name":"')  # Get into "wide open" string state

    iterations = 10000

    print("\n--- can_accept scaling by token length ---")
    print("Called ~150k times per get_valid_tokens call.\n")

    test_tokens = [
        ("1 byte", "a"),
        ("4 bytes", "test"),
        ("8 bytes", "testtest"),
        ("16 bytes", "testtesttesttest"),
    ]

    for name, token in test_tokens:
        start = time.perf_counter()
        for _ in range(iterations):
            g.can_accept(token)
        elapsed = time.perf_counter() - start

        ns_per_iter = elapsed / iterations * 1_000_000_000
        print(f"  {name:10}: {ns_per_iter:8.0f} ns/iter")

    # Extrapolate to vocab size
    print(f"\n  Extrapolated cost for 150k tokens (4 bytes each):")
    g.reset()
    g.accept('{"name":"')
    start = time.perf_counter()
    for _ in range(iterations):
        g.can_accept("test")
    elapsed = time.perf_counter() - start
    ns_per_call = elapsed / iterations * 1_000_000_000
    estimated_ms = ns_per_call * 150_000 / 1_000_000
    print(f"    {ns_per_call:.0f} ns/call × 150k = {estimated_ms:.1f} ms")


# =============================================================================
# Realistic Production Scenarios
# =============================================================================

class UserProfile(BaseModel):
    """Simple schema - common for basic extractions."""
    name: str
    age: int
    email: str


class ProductReview(BaseModel):
    """Medium complexity - nested object, enum-like field."""
    product_name: str
    rating: int  # 1-5
    pros: list[str]
    cons: list[str]
    summary: str


class APIResponse(BaseModel):
    """Complex schema - nested objects, arrays, optional fields."""
    status: str
    data: dict[str, str | int | list[str]]
    errors: list[str] | None = None
    metadata: dict[str, str] | None = None


def bench_realistic_schemas(tokenizer):
    """
    Benchmark with production-like schemas.

    Tests different schema complexities to understand scaling.
    """
    print("\n" + "=" * 60)
    print("REALISTIC SCHEMA BENCHMARKS")
    print("=" * 60)

    vocab_size = tokenizer.vocab_size

    # Schema complexity progression
    schemas = [
        ("SimpleSchema (2 fields)", SimpleSchema),
        ("UserProfile (3 fields)", UserProfile),
        ("ProductReview (5 fields, arrays)", ProductReview),
    ]

    for name, schema_class in schemas:
        print(f"\n--- {name} ---")

        try:
            g = Grammar(schema=schema_class)
        except Exception as e:
            print(f"  SKIP: Failed to compile schema: {e}")
            continue

        # Cold start (no cache)
        g.reset()
        start = time.perf_counter()
        mask = g.get_valid_tokens(tokenizer)
        cold_ms = (time.perf_counter() - start) * 1000
        valid_count = mask.count_valid()

        # Warm (cached state)
        g.reset()
        start = time.perf_counter()
        mask = g.get_valid_tokens(tokenizer)
        warm_ms = (time.perf_counter() - start) * 1000

        print(f"  Initial state: {valid_count} valid tokens")
        print(f"  Cold: {cold_ms:6.2f} ms  |  Warm: {warm_ms:6.2f} ms")


def bench_cache_effects(tokenizer):
    """
    Benchmark L1/L2/L3 cache effects on grammar operations.

    Grammar overhead depends heavily on:
    - State hash computation hitting L1/L2
    - Token vocabulary lookup hitting L2/L3
    - Mask bit operations
    """
    print("\n" + "=" * 60)
    print("CACHE EFFECTS BENCHMARK")
    print("=" * 60)

    g = Grammar(schema=SimpleSchema)
    vocab_size = tokenizer.vocab_size

    # Test 1: Repeated same-state queries (should be L1/L2 hot)
    print("\n--- Same-state repeated queries (cache-hot) ---")
    g.reset()
    g.accept('{"name":"')  # Get into string state

    # Warm up L1/L2
    for _ in range(5):
        mask = g.get_valid_tokens(tokenizer)

    iterations = 50
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        mask = g.get_valid_tokens(tokenizer)
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"  {iterations} iterations at same state:")
    print(f"  Avg: {avg:.3f} ms  |  Min: {min_t:.3f} ms  |  Max: {max_t:.3f} ms")

    # Test 2: Different states (cache misses on grammar state)
    print("\n--- Different states (grammar cache misses) ---")

    # Generate sequence of unique states
    prefixes = [
        '{"name":"',
        '{"name":"A',
        '{"name":"Al',
        '{"name":"Ali',
        '{"name":"Alic',
        '{"name":"Alice',
        '{"name":"Alice"',
        '{"name":"Alice",',
        '{"name":"Alice","',
        '{"name":"Alice","a',
    ]

    times = []
    for prefix in prefixes:
        g.reset()
        g.accept(prefix)
        start = time.perf_counter()
        mask = g.get_valid_tokens(tokenizer)
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"  {len(prefixes)} unique states:")
    print(f"  Avg: {avg:.2f} ms  |  Min: {min(times):.2f} ms  |  Max: {max(times):.2f} ms")

    # Test 3: Interleaved access (simulates real generation with context switches)
    print("\n--- Interleaved access (simulates real workload) ---")

    # Create multiple grammar instances (like multiple concurrent requests)
    grammars = [Grammar(schema=SimpleSchema) for _ in range(3)]
    states = ['{"name":"', '{"age":', '{"name":"test"']

    # Prime each grammar
    for g, prefix in zip(grammars, states):
        g.accept(prefix)
        g.get_valid_tokens(tokenizer)  # Warm cache

    # Interleaved access pattern
    iterations = 30
    times = []
    for i in range(iterations):
        g = grammars[i % len(grammars)]
        g.reset()
        g.accept(states[i % len(states)])
        start = time.perf_counter()
        mask = g.get_valid_tokens(tokenizer)
        times.append((time.perf_counter() - start) * 1000)

    avg = sum(times) / len(times)
    print(f"  {iterations} interleaved queries across {len(grammars)} grammars:")
    print(f"  Avg: {avg:.2f} ms  |  Min: {min(times):.2f} ms  |  Max: {max(times):.2f} ms")


def bench_token_by_token_generation(tokenizer):
    """
    Simulate realistic token-by-token generation.

    This matches exactly what happens during structured output generation:
    1. Get valid tokens for current state
    2. (Model inference happens - not measured here)
    3. Advance grammar with selected token
    4. Repeat
    """
    print("\n" + "=" * 60)
    print("TOKEN-BY-TOKEN GENERATION SIMULATION")
    print("=" * 60)

    # Realistic JSON outputs that a model might generate
    scenarios = [
        (
            "Simple user profile",
            SimpleSchema,
            '{"name":"John Doe","age":32}'
        ),
        (
            "User with longer string",
            SimpleSchema,
            '{"name":"Alexandra Elizabeth Johnson","age":28}'
        ),
    ]

    for scenario_name, schema_class, target_json in scenarios:
        print(f"\n--- {scenario_name} ---")
        print(f"  Target: {target_json[:50]}{'...' if len(target_json) > 50 else ''}")

        try:
            g = Grammar(schema=schema_class)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        # Tokenize the target (simulates what model would produce)
        # For simplicity, we'll go character by character (worst case)
        # Real tokens would be longer, reducing overhead

        g.reset()
        times = []
        positions = []

        # Simulate byte-by-byte generation (conservative estimate)
        for i, char in enumerate(target_json):
            start = time.perf_counter()
            mask = g.get_valid_tokens(tokenizer)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            positions.append(i)

            valid_count = mask.count_valid()
            g.accept(char)

        total_ms = sum(times)
        avg_ms = total_ms / len(times)

        # Categorize times
        fast = [t for t in times if t < 0.1]
        medium = [t for t in times if 0.1 <= t < 1.0]
        slow = [t for t in times if t >= 1.0]

        print(f"  Tokens: {len(times)}")
        print(f"  Total grammar time: {total_ms:.1f} ms")
        print(f"  Avg per token: {avg_ms:.2f} ms")
        print(f"  Distribution: {len(fast)} fast (<0.1ms), {len(medium)} medium, {len(slow)} slow (>1ms)")

        if slow:
            # Find which positions were slow
            slow_positions = [(i, t) for i, t in zip(positions, times) if t >= 1.0]
            print(f"  Slow positions: {[p for p, _ in slow_positions[:5]]}...")

    # Estimate overhead vs model inference
    print("\n--- Overhead Estimation ---")
    print("  Assuming model inference times:")

    model_times = [
        ("Small model (Qwen 0.6B)", 15),
        ("Medium model (7B)", 50),
        ("Large model (70B)", 200),
    ]

    # Use average from simple profile as baseline
    grammar_per_token = 3.0  # ms, typical cache-miss cost
    tokens_per_response = 50  # typical structured output

    for name, model_ms_per_token in model_times:
        model_total = model_ms_per_token * tokens_per_response
        grammar_total = grammar_per_token * tokens_per_response
        overhead_pct = (grammar_total / model_total) * 100
        print(f"  {name}: {overhead_pct:.1f}% overhead ({grammar_total:.0f}ms / {model_total:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(description="Grammar benchmarks")
    parser.add_argument("--hot-path", type=str, metavar="MODEL_URI",
                       help="Run hot-path benchmark with tokenizer (requires model)")
    args = parser.parse_args()

    print("=== Grammar Benchmarks ===\n")

    if args.hot_path:
        print("--- Hot Path Mode (with tokenizer) ---\n")
        bench_get_valid_tokens(args.hot_path)
    else:
        print("(No model loading - isolated grammar operations)\n")
        bench_grammar_creation()
        bench_validation()
        bench_streaming()
        bench_can_accept_scaling()

    print("\n=== Benchmark Complete ===")


if __name__ == "__main__":
    main()

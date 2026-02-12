//! Integration tests for ThreadPool
//!
//! ThreadPool provides futex-based parallel execution optimized for LLM inference.
//! It uses aggressive spinning before falling back to kernel waits.

const std = @import("std");
const main = @import("main");
const compute = main.core.compute;
const ThreadPool = compute.ThreadPool;

// =============================================================================
// Creation and Lifecycle Tests
// =============================================================================

test "create returns valid pool" {
    const pool = try ThreadPool.create(std.testing.allocator, 4);
    defer pool.deinit();

    try std.testing.expect(pool.n_threads >= 1);
    try std.testing.expect(pool.n_threads <= 64);
}

test "create with zero threads uses CPU count" {
    const pool = try ThreadPool.create(std.testing.allocator, 0);
    defer pool.deinit();

    // Should use available CPU cores
    try std.testing.expect(pool.n_threads >= 1);
}

test "create with one thread works" {
    const pool = try ThreadPool.create(std.testing.allocator, 1);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 1), pool.n_threads);
}

test "deinit cleans up properly" {
    // Create and destroy multiple pools to check for leaks
    for (0..3) |_| {
        const pool = try ThreadPool.create(std.testing.allocator, 2);
        pool.deinit();
    }
}

// =============================================================================
// Parallel Execution Tests
// =============================================================================

test "parallelFor executes all items" {
    const pool = try ThreadPool.create(std.testing.allocator, 4);
    defer pool.deinit();

    const n_items: usize = 100;
    var results: [n_items]u32 = [_]u32{0} ** n_items;

    const Context = struct {
        results: *[n_items]u32,
    };
    var ctx = Context{ .results = &results };

    pool.parallelFor(n_items, struct {
        fn run(start: usize, end: usize, c: *Context) void {
            for (start..end) |i| {
                c.results[i] = 1;
            }
        }
    }.run, &ctx);

    // All items should be processed
    var sum: u32 = 0;
    for (results) |r| {
        sum += r;
    }
    try std.testing.expectEqual(@as(u32, n_items), sum);
}

test "parallelFor handles empty range" {
    const pool = try ThreadPool.create(std.testing.allocator, 4);
    defer pool.deinit();

    var executed = false;
    const Context = struct {
        executed: *bool,
    };
    var ctx = Context{ .executed = &executed };

    pool.parallelFor(0, struct {
        fn run(start: usize, end: usize, c: *Context) void {
            _ = start;
            _ = end;
            c.executed.* = true;
        }
    }.run, &ctx);

    // With 0 items, should still call the function with (0, 0)
    // depending on implementation
}

test "parallelFor with single thread runs sequentially" {
    const pool = try ThreadPool.create(std.testing.allocator, 1);
    defer pool.deinit();

    const n_items: usize = 50;
    var results: [n_items]u32 = [_]u32{0} ** n_items;

    const Context = struct {
        results: *[n_items]u32,
    };
    var ctx = Context{ .results = &results };

    pool.parallelFor(n_items, struct {
        fn run(start: usize, end: usize, c: *Context) void {
            for (start..end) |i| {
                c.results[i] = @intCast(i + 1);
            }
        }
    }.run, &ctx);

    // Verify sequential execution
    for (0..n_items) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), results[i]);
    }
}

test "parallelFor accumulates values correctly" {
    const pool = try ThreadPool.create(std.testing.allocator, 4);
    defer pool.deinit();

    // Sum of 1 to 1000
    const n_items: usize = 1000;
    var partial_sums: [64]u64 align(64) = [_]u64{0} ** 64;

    const Context = struct {
        sums: *[64]u64,
    };
    var ctx = Context{ .sums = &partial_sums };

    pool.parallelFor(n_items, struct {
        fn run(start: usize, end: usize, c: *Context) void {
            var local_sum: u64 = 0;
            for (start..end) |i| {
                local_sum += i + 1; // 1-indexed sum
            }
            // Use atomics to accumulate (slot 0)
            _ = @atomicRmw(u64, &c.sums[0], .Add, local_sum, .seq_cst);
        }
    }.run, &ctx);

    // Sum of 1 to n = n*(n+1)/2
    const expected: u64 = n_items * (n_items + 1) / 2;
    try std.testing.expectEqual(expected, partial_sums[0]);
}

// =============================================================================
// Global Pool Tests
// =============================================================================

test "global returns singleton pool" {
    const pool1 = compute.parallel.global();
    const pool2 = compute.parallel.global();

    // Should return the same instance
    try std.testing.expect(pool1 == pool2);
}

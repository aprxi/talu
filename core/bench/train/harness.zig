//! Timing aggregation for training benchmarks.
//!
//! Collects nanosecond samples and computes percentile summaries.
//! Same pattern as tokenizer/harness.zig.

const std = @import("std");

pub const Sample = struct {
    elapsed_ns: u64,
};

pub const Summary = struct {
    min_ns: u64,
    p50_ns: u64,
    p90_ns: u64,
    max_ns: u64,
    mean_ns: u64,
};

/// Compute percentile summary from a slice of nanosecond values.
/// Sorts the input in-place.
pub fn summarizeValues(values: []u64) Summary {
    if (values.len == 0) return .{ .min_ns = 0, .p50_ns = 0, .p90_ns = 0, .max_ns = 0, .mean_ns = 0 };

    std.mem.sort(u64, values, {}, std.sort.asc(u64));

    var total: u64 = 0;
    for (values) |v| total += v;

    return .{
        .min_ns = values[0],
        .p50_ns = values[percentileIndex(values.len, 50)],
        .p90_ns = values[percentileIndex(values.len, 90)],
        .max_ns = values[values.len - 1],
        .mean_ns = total / values.len,
    };
}

/// Compute percentile summary from Sample array.
pub fn summarizeSamples(samples: []Sample) Summary {
    if (samples.len == 0) return .{ .min_ns = 0, .p50_ns = 0, .p90_ns = 0, .max_ns = 0, .mean_ns = 0 };

    // Extract elapsed_ns values
    const allocator = std.heap.page_allocator;
    const values = allocator.alloc(u64, samples.len) catch return .{ .min_ns = 0, .p50_ns = 0, .p90_ns = 0, .max_ns = 0, .mean_ns = 0 };
    defer allocator.free(values);

    for (samples, 0..) |s, i| values[i] = s.elapsed_ns;
    return summarizeValues(values);
}

fn percentileIndex(len: usize, pct: u64) usize {
    if (len <= 1) return 0;
    return @min((len - 1) * pct / 100, len - 1);
}

test "summarizeValues basic" {
    var values = [_]u64{ 100, 50, 200, 150, 75 };
    const s = summarizeValues(&values);
    try std.testing.expectEqual(@as(u64, 50), s.min_ns);
    try std.testing.expectEqual(@as(u64, 200), s.max_ns);
    try std.testing.expectEqual(@as(u64, 100), s.p50_ns);
}

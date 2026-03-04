const std = @import("std");

pub const Sample = struct {
    encode_ns: u64,
};

pub const Summary = struct {
    min_ns: u64,
    p50_ns: u64,
    p90_ns: u64,
    max_ns: u64,
    mean_ns: f64,
};

fn percentileIndex(len: usize, pct: u8) usize {
    if (len == 0) return 0;
    return ((len - 1) * pct) / 100;
}

pub fn summarizeValues(allocator: std.mem.Allocator, values: []const u64) !Summary {
    if (values.len == 0) return error.InvalidInput;

    const sorted = try allocator.dupe(u64, values);
    defer allocator.free(sorted);
    std.sort.block(u64, sorted, {}, std.sort.asc(u64));

    var sum: u128 = 0;
    for (sorted) |value| sum += value;

    const p50_idx = percentileIndex(sorted.len, 50);
    const p90_idx = percentileIndex(sorted.len, 90);

    return .{
        .min_ns = sorted[0],
        .p50_ns = sorted[p50_idx],
        .p90_ns = sorted[p90_idx],
        .max_ns = sorted[sorted.len - 1],
        .mean_ns = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(sorted.len)),
    };
}

pub fn summarizeSamples(allocator: std.mem.Allocator, samples: []const Sample) !Summary {
    if (samples.len == 0) return error.InvalidInput;

    const values = try allocator.alloc(u64, samples.len);
    defer allocator.free(values);

    for (samples, 0..) |sample, idx| {
        values[idx] = sample.encode_ns;
    }

    return summarizeValues(allocator, values);
}

test "summarizeValues computes deterministic percentiles" {
    const allocator = std.testing.allocator;
    const values = [_]u64{ 90, 50, 10, 70, 30 };
    const summary = try summarizeValues(allocator, &values);
    try std.testing.expectEqual(@as(u64, 10), summary.min_ns);
    try std.testing.expectEqual(@as(u64, 50), summary.p50_ns);
    try std.testing.expectEqual(@as(u64, 70), summary.p90_ns);
    try std.testing.expectEqual(@as(u64, 90), summary.max_ns);
    try std.testing.expect(summary.mean_ns > 49.0 and summary.mean_ns < 51.0);
}

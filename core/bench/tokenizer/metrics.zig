/// Input throughput in MB/s (1 MB = 1_000_000 bytes).
pub fn mbps(bytes: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    // bytes/ns == GB/s; multiply by 1000 for MB/s
    return (@as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(elapsed_ns))) * 1000.0;
}

/// Output throughput in millions of tokens per second.
pub fn mtok_per_sec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    // tokens/ns == Gtok/s; multiply by 1000 for Mtok/s
    return (@as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(elapsed_ns))) * 1000.0;
}

test "mbps handles zero and positive durations" {
    const std = @import("std");
    try std.testing.expectEqual(@as(f64, 0.0), mbps(1, 0));
    // 1_000_000 bytes in 1_000_000_000 ns = 1 MB/s
    const value = mbps(1_000_000, 1_000_000_000);
    try std.testing.expect(value > 0.99 and value < 1.01);
}

test "mtok_per_sec handles zero and positive durations" {
    const std = @import("std");
    try std.testing.expectEqual(@as(f64, 0.0), mtok_per_sec(1, 0));
    // 1_000_000 tokens in 1_000_000_000 ns = 1 Mtok/s
    const value = mtok_per_sec(1_000_000, 1_000_000_000);
    try std.testing.expect(value > 0.99 and value < 1.01);
}

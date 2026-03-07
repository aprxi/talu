pub fn gbps(bytes: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(elapsed_ns));
}

pub fn tflops(flops: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    return (@as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(elapsed_ns))) / 1000.0;
}

test "metrics handle zero and positive durations" {
    const std = @import("std");
    try std.testing.expectEqual(@as(f64, 0.0), gbps(1, 0));
    try std.testing.expectEqual(@as(f64, 0.0), tflops(1, 0));
    try std.testing.expect(gbps(1_000_000_000, 1_000_000_000) > 0.99);
    try std.testing.expect(tflops(1_000_000_000_000, 1_000_000_000) > 0.99);
}

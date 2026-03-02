pub fn gbps(bytes: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    // bytes/ns == GB/s
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(elapsed_ns));
}

pub fn tflops(flops: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    // flops/ns == GFLOP/s, divide by 1000 for TFLOP/s
    return (@as(f64, @floatFromInt(flops)) / @as(f64, @floatFromInt(elapsed_ns))) / 1000.0;
}

test "gbps handles zero and positive durations" {
    try @import("std").testing.expectEqual(@as(f64, 0.0), gbps(1, 0));
    const value = gbps(1_000_000_000, 1_000_000_000);
    try @import("std").testing.expect(value > 0.99 and value < 1.01);
}

test "tflops handles zero and positive durations" {
    try @import("std").testing.expectEqual(@as(f64, 0.0), tflops(1, 0));
    const value = tflops(1_000_000_000_000, 1_000_000_000);
    try @import("std").testing.expect(value > 0.99 and value < 1.01);
}

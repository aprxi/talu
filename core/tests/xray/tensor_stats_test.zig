//! Integration tests for xray.TensorStats
//!
//! TensorStats holds computed statistics over a tensor: count, min, max,
//! sum, sum_sq, nan_count, inf_count. Provides derived stats like mean,
//! variance, stddev, rms.

const std = @import("std");
const main = @import("main");
const xray = main.xray;

const TensorStats = xray.TensorStats;

test "TensorStats: mean calculation" {
    const stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0, // 1+2+3+4 = 10
        .sum_sq = 30.0, // 1+4+9+16 = 30
        .nan_count = 0,
        .inf_count = 0,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), stats.mean(), 0.001);
}

test "TensorStats: variance calculation" {
    // Data: [1, 2, 3, 4], mean=2.5
    // Variance = E[X^2] - E[X]^2 = 30/4 - (10/4)^2 = 7.5 - 6.25 = 1.25
    const stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 1.25), stats.variance(), 0.001);
}

test "TensorStats: stddev calculation" {
    const stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    // stddev = sqrt(variance) = sqrt(1.25) ~ 1.118
    try std.testing.expectApproxEqAbs(@as(f32, 1.118), stats.stddev(), 0.01);
}

test "TensorStats: rms calculation" {
    const stats = TensorStats{
        .count = 4,
        .min = 1.0,
        .max = 4.0,
        .sum = 10.0,
        .sum_sq = 30.0,
        .nan_count = 0,
        .inf_count = 0,
    };

    // rms = sqrt(sum_sq / count) = sqrt(30/4) = sqrt(7.5) ~ 2.739
    try std.testing.expectApproxEqAbs(@as(f32, 2.739), stats.rms(), 0.01);
}

test "TensorStats: detects NaN anomalies" {
    const clean_stats = TensorStats{
        .count = 10,
        .min = 0.0,
        .max = 1.0,
        .sum = 5.0,
        .sum_sq = 3.0,
        .nan_count = 0,
        .inf_count = 0,
    };
    try std.testing.expect(!clean_stats.hasAnomalies());

    const nan_stats = TensorStats{
        .count = 10,
        .min = 0.0,
        .max = 1.0,
        .sum = 5.0,
        .sum_sq = 3.0,
        .nan_count = 1,
        .inf_count = 0,
    };
    try std.testing.expect(nan_stats.hasAnomalies());
}

test "TensorStats: detects Inf anomalies" {
    const inf_stats = TensorStats{
        .count = 10,
        .min = 0.0,
        .max = 1.0,
        .sum = 5.0,
        .sum_sq = 3.0,
        .nan_count = 0,
        .inf_count = 1,
    };
    try std.testing.expect(inf_stats.hasAnomalies());
}

test "TensorStats: EMPTY constant" {
    const empty = TensorStats.EMPTY;
    try std.testing.expectEqual(@as(u64, 0), empty.count);
    try std.testing.expectEqual(@as(f64, 0), empty.sum);
    try std.testing.expectEqual(@as(u64, 0), empty.nan_count);
    try std.testing.expectEqual(@as(u64, 0), empty.inf_count);
}

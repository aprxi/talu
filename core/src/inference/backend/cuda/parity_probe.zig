//! Shared CUDA parity probe utilities.
//!
//! This module keeps parity math and thresholds testable in isolation from
//! backend runtime state.

const std = @import("std");

pub const Summary = struct {
    cpu_top: u32,
    cuda_top: u32,
    mean_abs_diff: f32,
    max_abs_diff: f32,
    finite_count: usize,
    cpu_nan_count: usize,
    cuda_nan_count: usize,
};

pub fn summarize(cpu_logits: []const f32, cuda_logits: []const f32) !Summary {
    if (cpu_logits.len != cuda_logits.len) return error.InvalidArgument;

    var summary: Summary = .{
        .cpu_top = argmaxHost(cpu_logits),
        .cuda_top = argmaxHost(cuda_logits),
        .mean_abs_diff = 0.0,
        .max_abs_diff = 0.0,
        .finite_count = 0,
        .cpu_nan_count = 0,
        .cuda_nan_count = 0,
    };
    var sum_abs_diff: f64 = 0.0;
    for (cpu_logits, cuda_logits) |cpu_v, cuda_v| {
        if (std.math.isNan(cpu_v)) summary.cpu_nan_count += 1;
        if (std.math.isNan(cuda_v)) summary.cuda_nan_count += 1;
        if (std.math.isFinite(cpu_v) and std.math.isFinite(cuda_v)) {
            const diff = @abs(cpu_v - cuda_v);
            if (diff > summary.max_abs_diff) summary.max_abs_diff = diff;
            sum_abs_diff += diff;
            summary.finite_count += 1;
        }
    }
    summary.mean_abs_diff = if (summary.finite_count > 0)
        @floatCast(sum_abs_diff / @as(f64, @floatFromInt(summary.finite_count)))
    else
        0.0;
    return summary;
}

pub fn enforce(summary: Summary, max_mean_abs_diff: f32, max_abs_diff: f32) !void {
    if (summary.cpu_nan_count != 0 or
        summary.cuda_nan_count != 0 or
        summary.cpu_top != summary.cuda_top or
        summary.mean_abs_diff > max_mean_abs_diff or
        summary.max_abs_diff > max_abs_diff)
    {
        return error.CudaParityMismatch;
    }
}

fn argmaxHost(values: []const f32) u32 {
    var best_idx: usize = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (values, 0..) |v, idx| {
        if (v > best_val) {
            best_val = v;
            best_idx = idx;
        }
    }
    return @intCast(best_idx);
}

test "summarize computes deterministic parity stats" {
    const cpu = [_]f32{ 0.1, 0.9, -0.3, 0.0 };
    const cuda = [_]f32{ 0.1, 0.9, -0.2, 0.0 };
    const summary = try summarize(cpu[0..], cuda[0..]);
    try std.testing.expectEqual(@as(u32, 1), summary.cpu_top);
    try std.testing.expectEqual(@as(u32, 1), summary.cuda_top);
    try std.testing.expectEqual(@as(usize, 4), summary.finite_count);
    try std.testing.expect(summary.max_abs_diff > 0.09 and summary.max_abs_diff < 0.11);
    try std.testing.expect(summary.mean_abs_diff > 0.02 and summary.mean_abs_diff < 0.03);
    try std.testing.expectEqual(@as(usize, 0), summary.cpu_nan_count);
    try std.testing.expectEqual(@as(usize, 0), summary.cuda_nan_count);
}

test "summarize rejects mismatched lengths" {
    const cpu = [_]f32{ 1.0, 2.0 };
    const cuda = [_]f32{1.0};
    try std.testing.expectError(error.InvalidArgument, summarize(cpu[0..], cuda[0..]));
}

test "enforce accepts matching summary within thresholds" {
    const summary: Summary = .{
        .cpu_top = 3,
        .cuda_top = 3,
        .mean_abs_diff = 0.02,
        .max_abs_diff = 0.08,
        .finite_count = 16,
        .cpu_nan_count = 0,
        .cuda_nan_count = 0,
    };
    try enforce(summary, 0.05, 0.1);
}

test "enforce rejects top mismatch" {
    const summary: Summary = .{
        .cpu_top = 1,
        .cuda_top = 2,
        .mean_abs_diff = 0.0,
        .max_abs_diff = 0.0,
        .finite_count = 8,
        .cpu_nan_count = 0,
        .cuda_nan_count = 0,
    };
    try std.testing.expectError(error.CudaParityMismatch, enforce(summary, 1.0, 1.0));
}

test "enforce rejects NaN counts" {
    const summary: Summary = .{
        .cpu_top = 1,
        .cuda_top = 1,
        .mean_abs_diff = 0.0,
        .max_abs_diff = 0.0,
        .finite_count = 8,
        .cpu_nan_count = 0,
        .cuda_nan_count = 1,
    };
    try std.testing.expectError(error.CudaParityMismatch, enforce(summary, 1.0, 1.0));
}

test "enforce rejects threshold violations" {
    const mean_bad: Summary = .{
        .cpu_top = 1,
        .cuda_top = 1,
        .mean_abs_diff = 0.2,
        .max_abs_diff = 0.2,
        .finite_count = 8,
        .cpu_nan_count = 0,
        .cuda_nan_count = 0,
    };
    try std.testing.expectError(error.CudaParityMismatch, enforce(mean_bad, 0.1, 1.0));

    const max_bad: Summary = .{
        .cpu_top = 1,
        .cuda_top = 1,
        .mean_abs_diff = 0.01,
        .max_abs_diff = 2.0,
        .finite_count = 8,
        .cpu_nan_count = 0,
        .cuda_nan_count = 0,
    };
    try std.testing.expectError(error.CudaParityMismatch, enforce(max_bad, 1.0, 1.0));
}

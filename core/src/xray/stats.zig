//! Tensor Statistics
//!
//! Computes statistics over tensor data for inspection.
//! Own SIMD implementation - does not depend on inference compute code.

const std = @import("std");
const trace = @import("trace.zig");

/// Statistics computed over a tensor.
pub const TensorStats = struct {
    /// Number of elements
    count: u64,
    /// Minimum value
    min: f32,
    /// Maximum value
    max: f32,
    /// Sum of values
    sum: f64,
    /// Sum of squared values (for variance/rms)
    sum_sq: f64,
    /// Count of NaN values
    nan_count: u32,
    /// Count of Inf values
    inf_count: u32,

    pub fn mean(self: TensorStats) f32 {
        if (self.count == 0) return 0;
        return @floatCast(self.sum / @as(f64, @floatFromInt(self.count)));
    }

    pub fn variance(self: TensorStats) f32 {
        if (self.count == 0) return 0;
        const n: f64 = @floatFromInt(self.count);
        const mean_sq = (self.sum * self.sum) / n;
        return @floatCast((self.sum_sq - mean_sq) / n);
    }

    pub fn stddev(self: TensorStats) f32 {
        return @sqrt(self.variance());
    }

    pub fn rms(self: TensorStats) f32 {
        if (self.count == 0) return 0;
        return @floatCast(@sqrt(self.sum_sq / @as(f64, @floatFromInt(self.count))));
    }

    pub fn l2Norm(self: TensorStats) f32 {
        return @floatCast(@sqrt(self.sum_sq));
    }

    pub fn hasAnomalies(self: TensorStats) bool {
        return self.nan_count > 0 or self.inf_count > 0;
    }

    pub const EMPTY: TensorStats = .{
        .count = 0,
        .min = std.math.inf(f32),
        .max = -std.math.inf(f32),
        .sum = 0,
        .sum_sq = 0,
        .nan_count = 0,
        .inf_count = 0,
    };
};

/// Compute statistics from a TracedTensor.
pub fn compute(ref: trace.TracedTensor) TensorStats {
    return switch (ref.dtype) {
        .f32 => computeF32(@ptrCast(@alignCast(ref.ptr)), ref.elementCount()),
        .f16 => computeF16(@ptrCast(@alignCast(ref.ptr)), ref.elementCount()),
        .bf16 => computeBF16(@ptrCast(@alignCast(ref.ptr)), ref.elementCount()),
        else => TensorStats.EMPTY, // Unsupported dtype
    };
}

/// Compute statistics from f32 data.
pub fn computeF32(data: [*]const f32, count: usize) TensorStats {
    if (count == 0) return TensorStats.EMPTY;

    // Use SIMD for larger arrays on x86
    const is_x86 = comptime (std.Target.Cpu.Arch.x86_64 == @import("builtin").cpu.arch or
        std.Target.Cpu.Arch.x86 == @import("builtin").cpu.arch);
    if (count >= 8 and is_x86) {
        return computeF32Simd(data, count);
    }

    return computeF32Scalar(data, count);
}

fn computeF32Scalar(data: [*]const f32, count: usize) TensorStats {
    var stats = TensorStats.EMPTY;
    stats.count = count;

    for (0..count) |i| {
        const v = data[i];
        if (std.math.isNan(v)) {
            stats.nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            stats.inf_count += 1;
            continue;
        }
        const vf64: f64 = v;
        stats.min = @min(stats.min, v);
        stats.max = @max(stats.max, v);
        stats.sum += vf64;
        stats.sum_sq += vf64 * vf64;
    }

    return stats;
}

fn computeF32Simd(data: [*]const f32, count: usize) TensorStats {
    const Vec8 = @Vector(8, f32);

    var min_vec: Vec8 = @splat(std.math.inf(f32));
    var max_vec: Vec8 = @splat(-std.math.inf(f32));
    var sum_vec: @Vector(8, f64) = @splat(0.0);
    var sum_sq_vec: @Vector(8, f64) = @splat(0.0);
    var nan_count: u32 = 0;
    var inf_count: u32 = 0;

    const vec_count = count / 8;
    const remainder = count % 8;

    // Process 8 elements at a time (unaligned load)
    for (0..vec_count) |i| {
        const base = data + i * 8;
        const v = Vec8{ base[0], base[1], base[2], base[3], base[4], base[5], base[6], base[7] };

        // Check for NaN/Inf and count (must check element by element)
        const nan_mask = v != v; // NaN != NaN
        const abs_v = @abs(v);
        const inf_vec: Vec8 = @splat(std.math.inf(f32));
        const inf_mask = abs_v == inf_vec;

        // Count anomalies element by element
        inline for (0..8) |j| {
            if (nan_mask[j]) nan_count += 1;
            if (inf_mask[j]) inf_count += 1;
        }

        // Replace NaN/Inf with 0 for stats calculation
        const zero_vec: Vec8 = @splat(0);
        const clean = @select(f32, nan_mask, zero_vec, v);
        const clean2 = @select(f32, inf_mask, zero_vec, clean);

        // Update min/max (only for valid values - use inf as sentinel)
        const pos_inf: Vec8 = @splat(std.math.inf(f32));
        const neg_inf: Vec8 = @splat(-std.math.inf(f32));
        const valid_for_min = @select(f32, nan_mask, pos_inf, v);
        const valid_for_min2 = @select(f32, inf_mask, pos_inf, valid_for_min);
        min_vec = @min(min_vec, valid_for_min2);

        const valid_for_max = @select(f32, nan_mask, neg_inf, v);
        const valid_for_max2 = @select(f32, inf_mask, neg_inf, valid_for_max);
        max_vec = @max(max_vec, valid_for_max2);

        // Accumulate sums (convert to f64 for precision)
        const v_f64: @Vector(8, f64) = clean2;
        sum_vec += v_f64;
        sum_sq_vec += v_f64 * v_f64;
    }

    // Reduce vectors to scalars
    var stats = TensorStats{
        .count = count,
        .min = @reduce(.Min, min_vec),
        .max = @reduce(.Max, max_vec),
        .sum = @reduce(.Add, sum_vec),
        .sum_sq = @reduce(.Add, sum_sq_vec),
        .nan_count = nan_count,
        .inf_count = inf_count,
    };

    // Handle remainder
    if (remainder > 0) {
        const tail = computeF32Scalar(data + vec_count * 8, remainder);
        stats.min = @min(stats.min, tail.min);
        stats.max = @max(stats.max, tail.max);
        stats.sum += tail.sum;
        stats.sum_sq += tail.sum_sq;
        stats.nan_count += tail.nan_count;
        stats.inf_count += tail.inf_count;
    }

    return stats;
}

/// Compute statistics from f16 data.
pub fn computeF16(data: [*]const f16, count: usize) TensorStats {
    if (count == 0) return TensorStats.EMPTY;

    var stats = TensorStats.EMPTY;
    stats.count = count;

    for (0..count) |i| {
        const v: f32 = @floatCast(data[i]);
        if (std.math.isNan(v)) {
            stats.nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            stats.inf_count += 1;
            continue;
        }
        const vf64: f64 = v;
        stats.min = @min(stats.min, v);
        stats.max = @max(stats.max, v);
        stats.sum += vf64;
        stats.sum_sq += vf64 * vf64;
    }

    return stats;
}

/// Compute statistics from bf16 data.
pub fn computeBF16(data: [*]const u16, count: usize) TensorStats {
    if (count == 0) return TensorStats.EMPTY;

    var stats = TensorStats.EMPTY;
    stats.count = count;

    for (0..count) |i| {
        // BF16: sign(1) + exp(8) + mantissa(7), same exponent range as f32
        // Convert by shifting to upper 16 bits of f32
        const bits: u32 = @as(u32, data[i]) << 16;
        const v: f32 = @bitCast(bits);

        if (std.math.isNan(v)) {
            stats.nan_count += 1;
            continue;
        }
        if (std.math.isInf(v)) {
            stats.inf_count += 1;
            continue;
        }
        const vf64: f64 = v;
        stats.min = @min(stats.min, v);
        stats.max = @max(stats.max, v);
        stats.sum += vf64;
        stats.sum_sq += vf64 * vf64;
    }

    return stats;
}

// ============================================================================
// Tests
// ============================================================================

test "stats basic computation" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const stats = computeF32(&data, data.len);

    try std.testing.expectEqual(@as(u64, 5), stats.count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), stats.min, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), stats.max, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), stats.mean(), 0.001);
    try std.testing.expectEqual(@as(u32, 0), stats.nan_count);
    try std.testing.expectEqual(@as(u32, 0), stats.inf_count);
}

test "stats with NaN and Inf" {
    const data = [_]f32{ 1.0, std.math.nan(f32), 3.0, std.math.inf(f32), 5.0 };
    const stats = computeF32(&data, data.len);

    try std.testing.expectEqual(@as(u64, 5), stats.count);
    try std.testing.expectEqual(@as(u32, 1), stats.nan_count);
    try std.testing.expectEqual(@as(u32, 1), stats.inf_count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), stats.min, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), stats.max, 0.001);
    try std.testing.expect(stats.hasAnomalies());
}

test "stats empty array" {
    const stats = computeF32(@ptrFromInt(0x1000), 0);
    try std.testing.expectEqual(@as(u64, 0), stats.count);
    try std.testing.expectApproxEqAbs(@as(f32, 0), stats.mean(), 0.001);
}

test "stats rms" {
    // RMS of [1, 1, 1, 1] = 1
    const data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const stats = computeF32(&data, data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), stats.rms(), 0.001);
}

test "stats bf16" {
    // BF16 representation of 1.0 is 0x3F80
    const data = [_]u16{ 0x3F80, 0x4000, 0x4040 }; // 1.0, 2.0, 3.0
    const stats = computeBF16(&data, data.len);

    try std.testing.expectEqual(@as(u64, 3), stats.count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), stats.min, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), stats.max, 0.01);
}

test "stats large array uses SIMD" {
    var data: [1024]f32 = undefined;
    for (0..1024) |i| {
        data[i] = @floatFromInt(i);
    }
    const stats = computeF32(&data, data.len);

    try std.testing.expectEqual(@as(u64, 1024), stats.count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), stats.min, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 1023.0), stats.max, 0.001);
    // Mean of 0..1023 = 511.5
    try std.testing.expectApproxEqAbs(@as(f32, 511.5), stats.mean(), 0.001);
}

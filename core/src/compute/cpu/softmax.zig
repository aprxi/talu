//! Softmax primitives used by CPU kernels.

const std = @import("std");
const fast_math = @import("math_primitives/fast_math.zig");
const simd = @import("../simd/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

const fastExp = fast_math.fastExp;
const fastExpScalar = fast_math.fastExpScalar;

/// Numerically-stable softmax in-place over one contiguous row.
pub fn stableInPlace(values: []f32) void {
    @setFloatMode(.optimized);
    if (values.len == 0) return;

    var max_vec: F32Vec = @splat(-std.math.inf(f32));
    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < values.len) : (vec_idx += VEC_LEN) {
        const value_vec: F32Vec = values[vec_idx..][0..VEC_LEN].*;
        max_vec = @max(max_vec, value_vec);
    }
    var max_v = @reduce(.Max, max_vec);
    while (vec_idx < values.len) : (vec_idx += 1) {
        if (values[vec_idx] > max_v) max_v = values[vec_idx];
    }

    const max_vec_value: F32Vec = @splat(max_v);
    var sum_vec: F32Vec = @splat(0);
    vec_idx = 0;
    while (vec_idx + VEC_LEN - 1 < values.len) : (vec_idx += VEC_LEN) {
        const value_vec: F32Vec = values[vec_idx..][0..VEC_LEN].*;
        const exp_vec = fastExp(value_vec - max_vec_value);
        values[vec_idx..][0..VEC_LEN].* = exp_vec;
        sum_vec += exp_vec;
    }
    var sum_exp = @reduce(.Add, sum_vec);
    while (vec_idx < values.len) : (vec_idx += 1) {
        const exp_scalar = fastExpScalar(values[vec_idx] - max_v);
        values[vec_idx] = exp_scalar;
        sum_exp += exp_scalar;
    }

    if (sum_exp <= 0.0 or !std.math.isFinite(sum_exp)) {
        const uniform = 1.0 / @as(f32, @floatFromInt(values.len));
        for (values) |*v| v.* = uniform;
        return;
    }

    const inv_sum = 1.0 / sum_exp;
    const inv_sum_vec: F32Vec = @splat(inv_sum);
    vec_idx = 0;
    while (vec_idx + VEC_LEN - 1 < values.len) : (vec_idx += VEC_LEN) {
        const value_vec: F32Vec = values[vec_idx..][0..VEC_LEN].*;
        values[vec_idx..][0..VEC_LEN].* = value_vec * inv_sum_vec;
    }
    while (vec_idx < values.len) : (vec_idx += 1) {
        values[vec_idx] *= inv_sum;
    }
}

/// In-place masked softmax over `scores` with explicit active interval.
///
/// Values outside `[active_start, active_end)` are set to zero.
/// `sink_logit` contributes probability mass but is not written into `scores`.
pub fn maskedInPlaceWithMax(
    scores: []f32,
    active_start: usize,
    active_end: usize,
    sink_logit: ?f32,
    exact: bool,
    maxv: f32,
    mask_cutoff: ?f32,
) void {
    std.debug.assert(active_start <= active_end and active_end <= scores.len);
    std.debug.assert(active_start < active_end);

    const cutoff = mask_cutoff;
    var sum: f32 = 0.0;

    if (exact) {
        if (sink_logit) |sl| sum += @exp(sl - maxv);
        for (0..active_start) |idx| scores[idx] = 0.0;
        for (active_start..active_end) |idx| {
            const s = scores[idx];
            if (cutoff != null and s <= cutoff.?) {
                scores[idx] = 0.0;
                continue;
            }
            const e = @exp(s - maxv);
            scores[idx] = e;
            sum += e;
        }
        for (active_end..scores.len) |idx| scores[idx] = 0.0;
    } else {
        const maxv_vec: F32Vec = @splat(maxv);
        var sum_vec: F32Vec = @splat(0);
        const sink_exp: f32 = if (sink_logit) |sl| fastExpScalar(sl - maxv) else 0.0;
        for (0..active_start) |idx| scores[idx] = 0.0;

        var vec_idx: usize = active_start;
        if (cutoff == null) {
            while (vec_idx + VEC_LEN - 1 < active_end) : (vec_idx += VEC_LEN) {
                const sv: F32Vec = scores[vec_idx..][0..VEC_LEN].*;
                const ev = fastExp(sv - maxv_vec);
                scores[vec_idx..][0..VEC_LEN].* = ev;
                sum_vec += ev;
            }
        }

        sum = sink_exp + @reduce(.Add, sum_vec);
        while (vec_idx < active_end) : (vec_idx += 1) {
            const s = scores[vec_idx];
            if (cutoff != null and s <= cutoff.?) {
                scores[vec_idx] = 0.0;
                continue;
            }
            const e = fastExpScalar(s - maxv);
            scores[vec_idx] = e;
            sum += e;
        }
        for (active_end..scores.len) |idx| scores[idx] = 0.0;
    }

    if (sum == 0.0) {
        @memset(scores, 0.0);
        return;
    }

    const inv_sum = 1.0 / sum;
    const inv_sum_vec: F32Vec = @splat(inv_sum);
    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < scores.len) : (vec_idx += VEC_LEN) {
        const sv: F32Vec = scores[vec_idx..][0..VEC_LEN].*;
        scores[vec_idx..][0..VEC_LEN].* = sv * inv_sum_vec;
    }
    while (vec_idx < scores.len) : (vec_idx += 1) {
        scores[vec_idx] *= inv_sum;
    }
}

test "stableInPlace normalizes row to 1" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    stableInPlace(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), v[0] + v[1] + v[2], 1e-6);
}

test "maskedInPlaceWithMax zeroes masked region and normalizes active window" {
    var scores = [_]f32{ 1, 2, 3, 4, 5, 6 };
    maskedInPlaceWithMax(&scores, 2, 5, null, true, 5.0, null);
    try std.testing.expectEqual(@as(f32, 0.0), scores[0]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[1]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[5]);
    const active_sum = scores[2] + scores[3] + scores[4];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), active_sum, 1e-6);
}

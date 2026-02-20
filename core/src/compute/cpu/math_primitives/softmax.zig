//! Softmax operations: contiguous and masked in-place variants.

const std = @import("std");
const fast_math = @import("fast_math.zig");
const simd = @import("../simd/arch/root.zig");

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

const fastExp = fast_math.fastExp;
const fastExpScalar = fast_math.fastExpScalar;

pub fn softmaxContiguous(out: []f32, input: []const f32, rows: usize, cols: usize) void {
    @setFloatMode(.optimized);
    std.debug.assert(input.len == rows * cols);
    std.debug.assert(out.len == input.len);

    var row_idx: usize = 0;
    while (row_idx < rows) : (row_idx += 1) {
        const in_row = input[row_idx * cols ..][0..cols];
        const out_row = out[row_idx * cols ..][0..cols];

        var max_vec: F32Vec = @splat(-std.math.inf(f32));
        var col_idx: usize = 0;
        while (col_idx + VEC_LEN - 1 < cols) : (col_idx += VEC_LEN) {
            const vec: F32Vec = in_row[col_idx..][0..VEC_LEN].*;
            max_vec = @max(max_vec, vec);
        }
        var max_value = @reduce(.Max, max_vec);
        while (col_idx < cols) : (col_idx += 1) {
            max_value = @max(max_value, in_row[col_idx]);
        }

        const max_vec_value: F32Vec = @splat(max_value);
        var sum_vec: F32Vec = @splat(0);
        col_idx = 0;
        while (col_idx + VEC_LEN - 1 < cols) : (col_idx += VEC_LEN) {
            const vec: F32Vec = in_row[col_idx..][0..VEC_LEN].*;
            const shifted = vec - max_vec_value;
            const exp_vec = fastExp(shifted);
            out_row[col_idx..][0..VEC_LEN].* = exp_vec;
            sum_vec += exp_vec;
        }
        var sum_value = @reduce(.Add, sum_vec);
        while (col_idx < cols) : (col_idx += 1) {
            const exp_scalar = fastExpScalar(in_row[col_idx] - max_value);
            out_row[col_idx] = exp_scalar;
            sum_value += exp_scalar;
        }

        const inv_sum_value = 1.0 / sum_value;
        const inv_sum_vec: F32Vec = @splat(inv_sum_value);
        col_idx = 0;
        while (col_idx + VEC_LEN - 1 < cols) : (col_idx += VEC_LEN) {
            const vec: F32Vec = out_row[col_idx..][0..VEC_LEN].*;
            out_row[col_idx..][0..VEC_LEN].* = vec * inv_sum_vec;
        }
        while (col_idx < cols) : (col_idx += 1) {
            out_row[col_idx] *= inv_sum_value;
        }
    }
}

pub fn softmaxMaskedInPlaceWithMax(
    scores: []f32,
    active_start: usize,
    active_end: usize,
    sink_logit: ?f32,
    exact: bool,
    maxv: f32,
    mask_cutoff: ?f32,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(active_start <= active_end and active_end <= scores.len);
    std.debug.assert(active_start < active_end);

    const cutoff = mask_cutoff;
    var sum: f32 = 0;
    if (exact) {
        if (sink_logit) |sl| {
            sum += @exp(sl - maxv);
        }
        for (0..active_start) |elem_idx| scores[elem_idx] = 0;
        for (active_start..active_end) |elem_idx| {
            const score = scores[elem_idx];
            if (cutoff != null and score <= cutoff.?) {
                scores[elem_idx] = 0;
                continue;
            }
            const exp_score = @exp(score - maxv);
            scores[elem_idx] = exp_score;
            sum += exp_score;
        }
        for (active_end..scores.len) |elem_idx| scores[elem_idx] = 0;
    } else {
        const maxv_vec: F32Vec = @splat(maxv);
        var sum_vec: F32Vec = @splat(0);
        const sink_exp: f32 = if (sink_logit) |sl| fastExpScalar(sl - maxv) else 0;
        for (0..active_start) |elem_idx| scores[elem_idx] = 0;

        var vec_idx: usize = active_start;
        if (cutoff == null) {
            while (vec_idx + VEC_LEN - 1 < active_end) : (vec_idx += VEC_LEN) {
                const scores_vec: F32Vec = scores[vec_idx..][0..VEC_LEN].*;
                const shifted = scores_vec - maxv_vec;
                const exp_vec = fastExp(shifted);
                scores[vec_idx..][0..VEC_LEN].* = exp_vec;
                sum_vec += exp_vec;
            }
        }
        sum = sink_exp + @reduce(.Add, sum_vec);
        while (vec_idx < active_end) : (vec_idx += 1) {
            const score = scores[vec_idx];
            if (cutoff != null and score <= cutoff.?) {
                scores[vec_idx] = 0;
                continue;
            }
            const exp_score = fastExpScalar(score - maxv);
            scores[vec_idx] = exp_score;
            sum += exp_score;
        }
        for (active_end..scores.len) |elem_idx| scores[elem_idx] = 0;
    }

    if (sum == 0) {
        @memset(scores, 0);
        return;
    }

    const inv_sum = 1.0 / sum;
    const inv_sum_vec: F32Vec = @splat(inv_sum);
    var vec_idx: usize = 0;
    while (vec_idx + VEC_LEN - 1 < scores.len) : (vec_idx += VEC_LEN) {
        const scores_vec: F32Vec = scores[vec_idx..][0..VEC_LEN].*;
        scores[vec_idx..][0..VEC_LEN].* = scores_vec * inv_sum_vec;
    }
    while (vec_idx < scores.len) : (vec_idx += 1) {
        scores[vec_idx] *= inv_sum;
    }
}

test "softmaxContiguous output sums to 1.0" {
    const allocator = std.testing.allocator;

    // Test single row softmax
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    softmaxContiguous(output, &input, 1, input.len);

    // Verify output sums to 1.0
    var sum: f32 = 0;
    for (output) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqRel(1.0, sum, 1e-5);

    // Verify output is monotonically increasing (input was increasing)
    for (0..output.len - 1) |i| {
        try std.testing.expect(output[i + 1] > output[i]);
    }

    // Verify all values are in valid probability range [0, 1]
    for (output) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
    }
}

test "softmaxContiguous numerical stability - large logits" {
    const allocator = std.testing.allocator;

    // Test with very large logits (would overflow without max subtraction)
    const input = [_]f32{ 1000.0, 1001.0, 1002.0, 1003.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    softmaxContiguous(output, &input, 1, input.len);

    // Verify all outputs are finite (no NaN or Inf)
    for (output) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }

    // Verify output sums to 1.0
    var sum: f32 = 0;
    for (output) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqRel(1.0, sum, 1e-5);

    // Verify output is still monotonically increasing
    for (0..output.len - 1) |i| {
        try std.testing.expect(output[i + 1] > output[i]);
    }
}

test "softmaxContiguous with negative logits" {
    const allocator = std.testing.allocator;

    // Test with all negative logits
    const input = [_]f32{ -5.0, -3.0, -1.0, -2.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    softmaxContiguous(output, &input, 1, input.len);

    // Verify output sums to 1.0
    var sum: f32 = 0;
    for (output) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqRel(1.0, sum, 1e-5);

    // Verify all values are positive and valid probabilities
    for (output) |val| {
        try std.testing.expect(val > 0.0);
        try std.testing.expect(val < 1.0);
    }

    // Verify the maximum input (-1.0 at index 2) has the highest output
    var max_idx: usize = 0;
    var max_val: f32 = output[0];
    for (output, 0..) |val, i| {
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    try std.testing.expectEqual(@as(usize, 2), max_idx);
}

test "softmaxContiguous multiple rows" {
    const allocator = std.testing.allocator;

    // Test multiple rows are normalized independently
    const rows: usize = 3;
    const cols: usize = 4;
    const input = [_]f32{
        1.0, 2.0, 3.0, 4.0, // Row 0
        5.0, 6.0, 7.0, 8.0, // Row 1
        -1.0, -2.0, -3.0, -4.0, // Row 2
    };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    softmaxContiguous(output, &input, rows, cols);

    // Verify each row sums to 1.0
    for (0..rows) |row| {
        var sum: f32 = 0;
        for (0..cols) |col| {
            sum += output[row * cols + col];
        }
        try std.testing.expectApproxEqRel(1.0, sum, 1e-5);
    }
}

test "softmaxContiguous SIMD vs scalar consistency" {
    const allocator = std.testing.allocator;

    // Test softmax with different sizes to exercise both SIMD and scalar paths
    const small_cols: usize = 3; // Will use scalar path for remainder
    const large_cols: usize = VEC_LEN * 4 + 3; // Mix of SIMD and scalar

    // Test small size
    {
        const input = try allocator.alloc(f32, small_cols);
        defer allocator.free(input);
        const output = try allocator.alloc(f32, small_cols);
        defer allocator.free(output);

        for (0..small_cols) |i| {
            input[i] = @floatFromInt(i);
        }

        softmaxContiguous(output, input, 1, small_cols);

        // Verify sum is 1.0
        var sum: f32 = 0;
        for (output) |val| {
            sum += val;
        }
        try std.testing.expectApproxEqRel(1.0, sum, 1e-5);
    }

    // Test large size (SIMD + scalar remainder)
    {
        const input = try allocator.alloc(f32, large_cols);
        defer allocator.free(input);
        const output = try allocator.alloc(f32, large_cols);
        defer allocator.free(output);

        for (0..large_cols) |i| {
            input[i] = @as(f32, @floatFromInt(i)) * 0.1;
        }

        softmaxContiguous(output, input, 1, large_cols);

        // Verify sum is 1.0
        var sum: f32 = 0;
        for (output) |val| {
            sum += val;
        }
        try std.testing.expectApproxEqRel(1.0, sum, 1e-5);

        // Verify monotonicity
        for (0..large_cols - 1) |i| {
            try std.testing.expect(output[i + 1] > output[i]);
        }
    }
}

test "softmaxMaskedInPlaceWithMax - basic functionality exact mode" {
    const scores = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var output: [scores.len]f32 = scores;

    const maxv: f32 = 5.0;
    softmaxMaskedInPlaceWithMax(&output, 0, scores.len, null, true, maxv, null);

    // Verify output sums to 1.0 (softmax normalization)
    var sum: f32 = 0;
    for (output) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum, 1e-5);

    // Verify all values are positive probabilities
    for (output) |val| {
        try std.testing.expect(val > 0.0);
        try std.testing.expect(val <= 1.0);
    }

    // Verify monotonicity (increasing input -> increasing output)
    for (0..output.len - 1) |i| {
        try std.testing.expect(output[i + 1] > output[i]);
    }
}

test "softmaxMaskedInPlaceWithMax - basic functionality fast mode" {
    const scores = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var output: [scores.len]f32 = scores;

    const maxv: f32 = 5.0;
    softmaxMaskedInPlaceWithMax(&output, 0, scores.len, null, false, maxv, null);

    // Verify output sums to 1.0
    var sum: f32 = 0;
    for (output) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum, 1e-3);

    // Verify all values are valid probabilities
    for (output) |val| {
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val <= 1.0);
    }
}

test "softmaxMaskedInPlaceWithMax - masked region" {
    var scores = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    const active_start: usize = 2;
    const active_end: usize = 6;
    const maxv: f32 = 8.0;

    softmaxMaskedInPlaceWithMax(&scores, active_start, active_end, null, true, maxv, null);

    // Elements before active_start should be zero
    try std.testing.expectEqual(@as(f32, 0.0), scores[0]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[1]);

    // Elements after active_end should be zero
    try std.testing.expectEqual(@as(f32, 0.0), scores[6]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[7]);

    // Active region should sum to 1.0
    var sum: f32 = 0;
    for (active_start..active_end) |i| {
        sum += scores[i];
        try std.testing.expect(scores[i] > 0.0);
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum, 1e-5);
}

test "softmaxMaskedInPlaceWithMax - with sink logit" {
    var scores = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const maxv: f32 = 5.0;
    const sink_logit: f32 = 3.0;

    softmaxMaskedInPlaceWithMax(&scores, 0, scores.len, sink_logit, true, maxv, null);

    // Output should sum to < 1.0 (sink absorbs some probability mass)
    var sum: f32 = 0;
    for (scores) |val| {
        sum += val;
    }
    // Sum of visible scores should be less than 1.0 since sink takes some probability
    try std.testing.expect(sum < 1.0);
    try std.testing.expect(sum > 0.0);
}

test "softmaxMaskedInPlaceWithMax - with cutoff mask" {
    var scores = [_]f32{ -10.0, -5.0, 0.0, 1.0, 2.0 };
    const maxv: f32 = 2.0;
    const mask_cutoff: f32 = -6.0;

    softmaxMaskedInPlaceWithMax(&scores, 0, scores.len, null, true, maxv, mask_cutoff);

    // Values <= cutoff should be masked to 0
    try std.testing.expectEqual(@as(f32, 0.0), scores[0]); // -10.0 <= -6.0

    // Other values should be > 0
    try std.testing.expect(scores[1] > 0.0); // -5.0 > -6.0
    try std.testing.expect(scores[2] > 0.0);
    try std.testing.expect(scores[3] > 0.0);
    try std.testing.expect(scores[4] > 0.0);

    // Sum should still be 1.0
    var sum: f32 = 0;
    for (scores) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum, 1e-5);
}

test "softmaxMaskedInPlaceWithMax - all masked returns zeros" {
    var scores = [_]f32{ 1.0, 2.0, 3.0 };
    const maxv: f32 = 10.0;
    const mask_cutoff: f32 = 5.0; // All values below cutoff

    softmaxMaskedInPlaceWithMax(&scores, 0, scores.len, null, true, maxv, mask_cutoff);

    // All should be zero when everything is masked
    for (scores) |val| {
        try std.testing.expectEqual(@as(f32, 0.0), val);
    }
}

test "softmaxMaskedInPlaceWithMax - exact vs fast mode consistency" {
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var exact_output: [input.len]f32 = input;
    var fast_output: [input.len]f32 = input;

    const maxv: f32 = 5.0;

    softmaxMaskedInPlaceWithMax(&exact_output, 0, input.len, null, true, maxv, null);
    softmaxMaskedInPlaceWithMax(&fast_output, 0, input.len, null, false, maxv, null);

    // Results should be close (fast exp may have slight differences)
    for (exact_output, fast_output) |exact, fast| {
        try std.testing.expectApproxEqRel(exact, fast, 1e-2);
    }
}

test "softmaxContiguous basic" {
    const allocator = std.testing.allocator;
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);

    softmaxContiguous(output, &input, 1, input.len);

    // Sum = 1
    var sum: f32 = 0;
    for (output) |v| sum += v;
    try std.testing.expectApproxEqRel(@as(f32, 1.0), sum, 1e-5);

    // Monotonically increasing
    for (0..output.len - 1) |i| {
        try std.testing.expect(output[i + 1] > output[i]);
    }
}

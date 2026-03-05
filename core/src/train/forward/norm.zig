//! RMSNorm forward pass for training, with saved statistics for backward.

const std = @import("std");
const compute = @import("../../compute/root.zig");

const simd = compute.cpu.simd.arch;
const VEC = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// RMSNorm forward that saves inv_rms for the backward pass.
///
/// output[i] = input[i] * inv_rms[row] * weight
/// inv_rms[row] = 1 / sqrt(mean(input[row]^2) + eps)
pub fn rmsnormForwardSave(
    output: []f32,
    inv_rms: []f32,
    input: []const f32,
    weight: []const f32,
    eps: f32,
    rows: usize,
    cols: usize,
) void {
    @setFloatMode(.optimized);
    std.debug.assert(output.len >= rows * cols);
    std.debug.assert(input.len >= rows * cols);
    std.debug.assert(inv_rms.len >= rows);
    std.debug.assert(weight.len == cols);

    const cols_f: f32 = @floatFromInt(cols);

    for (0..rows) |row| {
        const in_row = input[row * cols ..][0..cols];
        const out_row = output[row * cols ..][0..cols];

        // SIMD sum-of-squares
        var sum_vec: F32Vec = @splat(0.0);
        var i: usize = 0;
        while (i + VEC <= cols) : (i += VEC) {
            const v: F32Vec = in_row[i..][0..VEC].*;
            sum_vec = @mulAdd(F32Vec, v, v, sum_vec);
        }
        var sum_sq = @reduce(.Add, sum_vec);
        while (i < cols) : (i += 1) {
            sum_sq += in_row[i] * in_row[i];
        }

        const rms = @sqrt(sum_sq / cols_f + eps);
        const irms = 1.0 / rms;
        inv_rms[row] = irms;

        // SIMD normalization and scale: out = x * irms * weight
        const irms_v: F32Vec = @splat(irms);
        i = 0;
        while (i + VEC <= cols) : (i += VEC) {
            const x: F32Vec = in_row[i..][0..VEC].*;
            const w: F32Vec = weight[i..][0..VEC].*;
            out_row[i..][0..VEC].* = x * irms_v * w;
        }
        while (i < cols) : (i += 1) {
            out_row[i] = in_row[i] * irms * weight[i];
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "rmsnormForwardSave produces normalized output and saves inv_rms" {
    const input = [_]f32{ 3.0, 4.0 }; // rms = sqrt((9+16)/2) = sqrt(12.5)
    var output: [2]f32 = undefined;
    var inv_rms: [1]f32 = undefined;
    const weight = [_]f32{ 1.0, 1.0 };

    rmsnormForwardSave(&output, &inv_rms, &input, &weight, 1e-5, 1, 2);

    const expected_rms = @sqrt(12.5 + 1e-5);
    const expected_inv = 1.0 / expected_rms;
    try testing.expectApproxEqAbs(expected_inv, inv_rms[0], 1e-5);
    try testing.expectApproxEqAbs(3.0 * expected_inv, output[0], 1e-5);
    try testing.expectApproxEqAbs(4.0 * expected_inv, output[1], 1e-5);
}

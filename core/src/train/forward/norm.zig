//! RMSNorm forward pass for training, with saved statistics for backward.

const std = @import("std");

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
    std.debug.assert(output.len >= rows * cols);
    std.debug.assert(input.len >= rows * cols);
    std.debug.assert(inv_rms.len >= rows);
    std.debug.assert(weight.len == cols);

    const cols_f: f32 = @floatFromInt(cols);

    for (0..rows) |row| {
        const in_row = input[row * cols ..][0..cols];
        const out_row = output[row * cols ..][0..cols];

        // Compute mean squared value
        var sum_sq: f32 = 0.0;
        for (in_row) |v| {
            sum_sq += v * v;
        }
        const rms = @sqrt(sum_sq / cols_f + eps);
        const irms = 1.0 / rms;
        inv_rms[row] = irms;

        // Apply normalization and scale
        for (out_row, in_row, weight) |*o, x, w| {
            o.* = x * irms * w;
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

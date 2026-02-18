//! State-space recurrence primitives for CPU Mamba-style kernels.

const std = @import("std");
const kernel = @import("../kernel.zig");
const activation = @import("activation.zig");

/// Discretize delta-time values with optional bias and Softplus nonlinearity.
///
/// `out` and `dt_raw` must have equal length.
pub fn discretizeDtSoftplus(out: []f32, dt_raw: []const f32, dt_bias: ?[]const f32) !void {
    if (out.len != dt_raw.len) return error.InvalidShape;
    if (dt_bias) |bias| {
        if (bias.len < dt_raw.len) return error.InvalidShape;
    }

    for (0..out.len) |idx| {
        var dt_value = dt_raw[idx];
        if (dt_bias) |bias| {
            dt_value += bias[idx];
        }
        out[idx] = activation.softplus(dt_value);
    }
}

/// Apply SiLU activation in-place.
pub fn siluInPlace(values: []f32) void {
    for (values) |*v| {
        v.* = activation.silu(v.*);
    }
}

/// Apply SiLU gate: `values *= silu(gates)` in-place.
pub fn applySiluGateInPlace(values: []f32, gates: []const f32) !void {
    if (values.len != gates.len) return error.InvalidShape;
    for (values, gates) |*v, g| {
        v.* *= activation.silu(g);
    }
}

/// Run one SSM scan/update step using the selected backend kernel implementation.
pub fn scanStep(
    ssm_scan: kernel.SsmScanFn,
    ssm_state: []f32,
    ssm_out: []f32,
    x_conv_out: []const f32,
    b_raw: []const f32,
    c_raw: []const f32,
    a_log: []const f32,
    d_skip: []const f32,
    dt: []const f32,
    d_head: usize,
    d_state: usize,
    n_heads: usize,
    n_groups: usize,
) void {
    ssm_scan(
        ssm_state,
        ssm_out,
        x_conv_out,
        b_raw,
        c_raw,
        a_log,
        d_skip,
        dt,
        d_head,
        d_state,
        n_heads,
        n_groups,
    );
}

test "discretizeDtSoftplus applies bias and softplus" {
    const raw = [_]f32{ 0.0, 1.0, -2.0 };
    const bias = [_]f32{ 0.5, -0.5, 1.0 };
    var out = [_]f32{ 0, 0, 0 };

    try discretizeDtSoftplus(&out, &raw, &bias);

    try std.testing.expectApproxEqAbs(activation.softplus(0.5), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(activation.softplus(0.5), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(activation.softplus(-1.0), out[2], 1e-6);
}

test "applySiluGateInPlace multiplies values by silu(gate)" {
    var values = [_]f32{ 1.0, 2.0, -1.0 };
    const gates = [_]f32{ 0.0, 1.0, -1.0 };

    try applySiluGateInPlace(&values, &gates);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0) * activation.silu(1.0), values[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0) * activation.silu(-1.0), values[2], 1e-6);
}

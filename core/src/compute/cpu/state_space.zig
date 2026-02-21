//! State-space recurrence primitives for CPU kernels.

const std = @import("std");
const activation = @import("activation.zig");
const ssm_scan_mod = @import("simd/ssm_scan.zig");

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

/// Run one state scan/update step using the selected backend implementation.
pub fn scanStep(
    ssm_scan: ssm_scan_mod.StateScanFn,
    ssm_state: []f32,
    ssm_out: []f32,
    input_values: []const f32,
    state_in_weights: []const f32,
    state_out_weights: []const f32,
    state_decay_log: []const f32,
    skip_weights: []const f32,
    dt: []const f32,
    feature_block_width: usize,
    state_width: usize,
    group_count: usize,
    source_group_count: usize,
) void {
    ssm_scan(
        ssm_state,
        ssm_out,
        input_values,
        state_in_weights,
        state_out_weights,
        state_decay_log,
        skip_weights,
        dt,
        feature_block_width,
        state_width,
        group_count,
        source_group_count,
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

test "siluInPlace applies silu activation per element" {
    var values = [_]f32{ -1.0, 0.0, 1.0 };
    siluInPlace(&values);
    try std.testing.expectApproxEqAbs(activation.silu(-1.0), values[0], 1e-6);
    try std.testing.expectApproxEqAbs(activation.silu(0.0), values[1], 1e-6);
    try std.testing.expectApproxEqAbs(activation.silu(1.0), values[2], 1e-6);
}

test "scanStep delegates to provided scan function" {
    const Mock = struct {
        fn scan(
            ssm_state: []f32,
            ssm_out: []f32,
            input_values: []const f32,
            state_in_weights: []const f32,
            state_out_weights: []const f32,
            state_decay_log: []const f32,
            skip_weights: []const f32,
            dt: []const f32,
            feature_block_width: usize,
            state_width: usize,
            group_count: usize,
            source_group_count: usize,
        ) void {
            _ = input_values;
            _ = state_in_weights;
            _ = state_out_weights;
            _ = state_decay_log;
            _ = skip_weights;
            _ = dt;
            _ = feature_block_width;
            _ = state_width;
            _ = group_count;
            _ = source_group_count;
            for (ssm_state) |*v| v.* = 42.0;
            for (ssm_out) |*v| v.* = 7.0;
        }
    };

    var state = [_]f32{ 0, 0, 0, 0 };
    var out = [_]f32{ 0, 0 };
    const x = [_]f32{ 1, 1 };
    const b = [_]f32{ 1, 1 };
    const c = [_]f32{ 1, 1 };
    const a_log = [_]f32{0};
    const d_skip = [_]f32{0};
    const dt = [_]f32{1};

    scanStep(Mock.scan, &state, &out, &x, &b, &c, &a_log, &d_skip, &dt, 2, 2, 1, 1);
    try std.testing.expectEqual(@as(f32, 42.0), state[0]);
    try std.testing.expectEqual(@as(f32, 42.0), state[3]);
    try std.testing.expectEqual(@as(f32, 7.0), out[0]);
    try std.testing.expectEqual(@as(f32, 7.0), out[1]);
}

//! Gated DeltaNet math primitives.
//!
//! These helpers keep the recurrent state update and gating math in the compute
//! layer. Inference kernels remain responsible only for tensor orchestration.

const std = @import("std");
const activation = @import("activation.zig");

pub fn applySiluInPlace(values: []f32) void {
    for (values) |*value| value.* = activation.silu(value.*);
}

pub fn normalizeQueryKeyInPlace(
    query: []f32,
    key: []f32,
    n_heads: usize,
    d_head: usize,
) !void {
    if (query.len != n_heads * d_head) return error.InvalidShape;
    if (key.len != n_heads * d_head) return error.InvalidShape;

    const q_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));
    for (0..n_heads) |head_idx| {
        const q_head = query[head_idx * d_head ..][0..d_head];
        const k_head = key[head_idx * d_head ..][0..d_head];
        var q_norm_sq: f32 = 0.0;
        var k_norm_sq: f32 = 0.0;
        for (q_head) |v| q_norm_sq += v * v;
        for (k_head) |v| k_norm_sq += v * v;
        const q_inv = 1.0 / @sqrt(q_norm_sq + 1e-6);
        const k_inv = 1.0 / @sqrt(k_norm_sq + 1e-6);
        for (q_head) |*v| v.* *= q_inv * q_scale;
        for (k_head) |*v| v.* *= k_inv;
    }
}

pub fn runStateSpaceStep(
    kv_mem: []f32,
    ssm_out: []f32,
    ssm_state: []f32,
    query: []const f32,
    key: []const f32,
    value: []const f32,
    beta_raw: []const f32,
    a_raw: []const f32,
    a_log: []const f32,
    dt_bias: ?[]const f32,
    n_heads: usize,
    d_head: usize,
) !void {
    const d_inner = n_heads * d_head;
    if (kv_mem.len != d_inner) return error.InvalidShape;
    if (ssm_out.len != d_inner) return error.InvalidShape;
    if (query.len != d_inner or key.len != d_inner or value.len != d_inner) return error.InvalidShape;
    if (beta_raw.len != n_heads or a_raw.len != n_heads or a_log.len != n_heads) return error.InvalidShape;
    if (dt_bias) |bias| if (bias.len != n_heads) return error.InvalidShape;
    if (ssm_state.len != n_heads * d_head * d_head) return error.InvalidShape;

    @memset(ssm_out, 0.0);

    for (0..n_heads) |head_idx| {
        const query_head = query[head_idx * d_head ..][0..d_head];
        const key_head = key[head_idx * d_head ..][0..d_head];
        const value_head = value[head_idx * d_head ..][0..d_head];
        const kv_mem_head = kv_mem[head_idx * d_head ..][0..d_head];
        const out_head = ssm_out[head_idx * d_head ..][0..d_head];

        const beta = 1.0 / (1.0 + @exp(-beta_raw[head_idx]));
        const dt_term = if (dt_bias) |bias| bias[head_idx] else 0.0;
        const g = -@exp(a_log[head_idx]) * activation.softplus(a_raw[head_idx] + dt_term);
        const g_exp = @exp(g);
        const state_base = head_idx * d_head * d_head;

        for (0..d_head) |v_idx| {
            var mem: f32 = 0.0;
            for (0..d_head) |k_idx| {
                const idx = state_base + k_idx * d_head + v_idx;
                ssm_state[idx] *= g_exp;
                mem += ssm_state[idx] * key_head[k_idx];
            }
            kv_mem_head[v_idx] = mem;
            out_head[v_idx] = (value_head[v_idx] - mem) * beta;
        }

        for (0..d_head) |k_idx| {
            const k_val = key_head[k_idx];
            const row_base = state_base + k_idx * d_head;
            for (0..d_head) |v_idx| {
                ssm_state[row_base + v_idx] += k_val * out_head[v_idx];
            }
        }

        for (0..d_head) |v_idx| {
            var out_val: f32 = 0.0;
            for (0..d_head) |k_idx| {
                out_val += ssm_state[state_base + k_idx * d_head + v_idx] * query_head[k_idx];
            }
            out_head[v_idx] = out_val;
        }
    }
}

pub fn normWeightSlice(
    norm_weight: ?[]const f32,
    head_idx: usize,
    d_head: usize,
    d_inner: usize,
) !?[]const f32 {
    const weights = norm_weight orelse return null;
    if (weights.len == d_head) return weights;
    if (weights.len == d_inner) return weights[head_idx * d_head ..][0..d_head];
    return error.InvalidShape;
}

pub fn applyGatedRmsNormInPlace(
    values: []f32,
    gate: []const f32,
    norm_weight: ?[]const f32,
) !void {
    if (values.len != gate.len) return error.InvalidShape;
    if (norm_weight) |w| if (w.len != values.len) return error.InvalidShape;

    var mean_sq: f32 = 0.0;
    for (values) |v| mean_sq += v * v;
    mean_sq /= @as(f32, @floatFromInt(values.len));
    const inv_rms = 1.0 / @sqrt(mean_sq + 1e-6);

    for (0..values.len) |i| {
        const weight = if (norm_weight) |w| w[i] else 1.0;
        values[i] = values[i] * inv_rms * weight * activation.silu(gate[i]);
    }
}

test "normWeightSlice rejects invalid norm shape" {
    const invalid = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.InvalidShape, normWeightSlice(&invalid, 0, 4, 8));
}

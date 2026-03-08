//! Gated DeltaNet math primitives.
//!
//! These helpers keep the recurrent state update and gating math in the compute
//! layer. Inference kernels remain responsible only for tensor orchestration.

const std = @import("std");
const activation = @import("activation.zig");
const math = @import("math.zig");
const reduction = @import("reduction.zig");
const rowwise = @import("rowwise.zig");

inline fn fastSigmoidScalar(x: f32) f32 {
    return 1.0 / (1.0 + math.fastExpScalar(-x));
}

inline fn fastSoftplusScalar(x: f32) f32 {
    if (x > 20.0) return x;
    return @log(1.0 + math.fastExpScalar(x));
}

fn scaleStateAndAccumulateAndProjectRow(
    state_row: []f32,
    accum_row: []f32,
    out_row: []f32,
    decay: f32,
    accum_weight: f32,
    project_weight: f32,
) void {
    std.debug.assert(state_row.len == accum_row.len);
    std.debug.assert(state_row.len == out_row.len);

    const simd = math.simd;
    const VEC_LEN = simd.f32_vec_len;
    const F32Vec = simd.F32Vec;
    const decay_vec: F32Vec = @splat(decay);
    const accum_weight_vec: F32Vec = @splat(accum_weight);
    const project_weight_vec: F32Vec = @splat(project_weight);

    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < state_row.len) : (idx += VEC_LEN) {
        const row = state_row[idx..][0..VEC_LEN];
        const scaled = row.* * decay_vec;
        row.* = scaled;
        const acc = accum_row[idx..][0..VEC_LEN];
        acc.* = @mulAdd(F32Vec, scaled, accum_weight_vec, acc.*);
        const out = out_row[idx..][0..VEC_LEN];
        out.* = @mulAdd(F32Vec, scaled, project_weight_vec, out.*);
    }
    while (idx < state_row.len) : (idx += 1) {
        const scaled = state_row[idx] * decay;
        state_row[idx] = scaled;
        accum_row[idx] += scaled * accum_weight;
        out_row[idx] += scaled * project_weight;
    }
}

fn updateStateRow(
    state_row: []f32,
    update_values: []const f32,
    update_weight: f32,
) void {
    std.debug.assert(state_row.len == update_values.len);

    const simd = math.simd;
    const VEC_LEN = simd.f32_vec_len;
    const F32Vec = simd.F32Vec;
    const update_vec: F32Vec = @splat(update_weight);

    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < state_row.len) : (idx += VEC_LEN) {
        const delta: F32Vec = update_values[idx..][0..VEC_LEN].*;
        const row = state_row[idx..][0..VEC_LEN];
        row.* = @mulAdd(F32Vec, delta, update_vec, row.*);
    }
    while (idx < state_row.len) : (idx += 1) {
        state_row[idx] += update_values[idx] * update_weight;
    }
}

fn applyResidualUpdateInPlace(
    values: []f32,
    target: []const f32,
    scale: f32,
) void {
    std.debug.assert(values.len == target.len);

    const simd = math.simd;
    const VEC_LEN = simd.f32_vec_len;
    const F32Vec = simd.F32Vec;
    const scale_vec: F32Vec = @splat(scale);

    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < values.len) : (idx += VEC_LEN) {
        const dst = values[idx..][0..VEC_LEN];
        const target_vec: F32Vec = target[idx..][0..VEC_LEN].*;
        dst.* = (target_vec - dst.*) * scale_vec;
    }
    while (idx < values.len) : (idx += 1) {
        values[idx] = (target[idx] - values[idx]) * scale;
    }
}

pub fn applySiluInPlace(values: []f32) void {
    math.siluContiguous(values, values);
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
        const q_norm_sq = reduction.dotRow(q_head, q_head);
        const k_norm_sq = reduction.dotRow(k_head, k_head);
        const q_inv = 1.0 / @sqrt(q_norm_sq + 1e-6);
        const k_inv = 1.0 / @sqrt(k_norm_sq + 1e-6);
        rowwise.scaleInPlace(q_head, q_inv * q_scale);
        rowwise.scaleInPlace(k_head, k_inv);
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
    n_qk_heads: usize,
    n_v_heads: usize,
    d_head: usize,
) !void {
    if (n_qk_heads == 0 or n_v_heads == 0 or d_head == 0) return error.InvalidShape;
    if ((n_v_heads % n_qk_heads) != 0) return error.InvalidShape;
    const qk_inner = n_qk_heads * d_head;
    const d_inner = n_v_heads * d_head;
    if (kv_mem.len != d_inner) return error.InvalidShape;
    if (ssm_out.len != d_inner) return error.InvalidShape;
    if (query.len != qk_inner or key.len != qk_inner or value.len != d_inner) return error.InvalidShape;
    if (beta_raw.len != n_v_heads or a_raw.len != n_v_heads or a_log.len != n_v_heads) return error.InvalidShape;
    if (dt_bias) |bias| if (bias.len != n_v_heads) return error.InvalidShape;
    if (ssm_state.len != n_v_heads * d_head * d_head) return error.InvalidShape;

    @memset(ssm_out, 0.0);
    const qk_repeat = n_v_heads / n_qk_heads;

    for (0..n_v_heads) |head_idx| {
        const qk_head_idx = head_idx / qk_repeat;
        const query_head = query[qk_head_idx * d_head ..][0..d_head];
        const key_head = key[qk_head_idx * d_head ..][0..d_head];
        const value_head = value[head_idx * d_head ..][0..d_head];
        const kv_mem_head = kv_mem[head_idx * d_head ..][0..d_head];
        const out_head = ssm_out[head_idx * d_head ..][0..d_head];

        const beta = fastSigmoidScalar(beta_raw[head_idx]);
        const dt_term = if (dt_bias) |bias| bias[head_idx] else 0.0;
        const g = -math.fastExpScalar(a_log[head_idx]) * fastSoftplusScalar(a_raw[head_idx] + dt_term);
        const g_exp = math.fastExpScalar(g);
        const state_base = head_idx * d_head * d_head;
        const qk_dot = reduction.dotRow(query_head, key_head);

        @memset(kv_mem_head, 0.0);
        @memset(out_head, 0.0);
        for (0..d_head) |k_idx| {
            const row_base = state_base + k_idx * d_head;
            const state_row = ssm_state[row_base .. row_base + d_head];
            scaleStateAndAccumulateAndProjectRow(
                state_row,
                kv_mem_head,
                out_head,
                g_exp,
                key_head[k_idx],
                query_head[k_idx],
            );
        }

        applyResidualUpdateInPlace(kv_mem_head, value_head, beta);
        rowwise.addScaledInPlace(out_head, kv_mem_head, qk_dot);
        for (0..d_head) |k_idx| {
            const row_base = state_base + k_idx * d_head;
            const state_row = ssm_state[row_base .. row_base + d_head];
            updateStateRow(state_row, kv_mem_head, key_head[k_idx]);
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

    const mean_sq = reduction.dotRow(values, values) / @as(f32, @floatFromInt(values.len));
    const inv_rms = 1.0 / @sqrt(mean_sq + 1e-6);
    const simd = math.simd;
    const VEC_LEN = simd.f32_vec_len;
    const F32Vec = simd.F32Vec;
    const one: F32Vec = @splat(1.0);

    for (0..values.len) |i| {
        const weight = if (norm_weight) |w| w[i] else 1.0;
        values[i] *= inv_rms * weight;
    }
    var idx: usize = 0;
    while (idx + VEC_LEN - 1 < values.len) : (idx += VEC_LEN) {
        const gate_vec: F32Vec = gate[idx..][0..VEC_LEN].*;
        const sig = one / (one + math.fastExp(-gate_vec));
        const silu = gate_vec * sig;
        const dst = values[idx..][0..VEC_LEN];
        dst.* = dst.* * silu;
    }
    while (idx < values.len) : (idx += 1) {
        values[idx] *= activation.silu(gate[idx]);
    }
}

test "normWeightSlice rejects invalid norm shape" {
    const invalid = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectError(error.InvalidShape, normWeightSlice(&invalid, 0, 4, 8));
}

test "fast softplus tracks reference softplus closely on gated-delta range" {
    const inputs = [_]f32{ -10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0 };
    for (inputs) |x| {
        try std.testing.expectApproxEqAbs(activation.softplus(x), fastSoftplusScalar(x), 5e-3);
    }
}

test "runStateSpaceStep matches naive reference update" {
    const n_heads = 1;
    const d_head = 4;
    var kv_mem = [_]f32{0} ** (n_heads * d_head);
    var ssm_out = [_]f32{0} ** (n_heads * d_head);
    var ssm_state = [_]f32{
        0.1, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0, 1.1, 1.2,
        1.3, 1.4, 1.5, 1.6,
    };
    const query = [_]f32{ 0.2, -0.1, 0.4, 0.3 };
    const key = [_]f32{ -0.3, 0.5, 0.1, 0.2 };
    const value = [_]f32{ 0.9, -0.4, 0.7, 0.2 };
    const beta_raw = [_]f32{0.35};
    const a_raw = [_]f32{-0.2};
    const a_log = [_]f32{-0.7};
    const dt_bias = [_]f32{0.05};

    var ref_kv_mem = kv_mem;
    var ref_ssm_out = ssm_out;
    var ref_ssm_state = ssm_state;

    try runStateSpaceStep(&kv_mem, &ssm_out, &ssm_state, &query, &key, &value, &beta_raw, &a_raw, &a_log, &dt_bias, n_heads, n_heads, d_head);

    const beta = 1.0 / (1.0 + @exp(-beta_raw[0]));
    const g = -@exp(a_log[0]) * activation.softplus(a_raw[0] + dt_bias[0]);
    const g_exp = @exp(g);

    @memset(&ref_kv_mem, 0.0);
    @memset(&ref_ssm_out, 0.0);
    for (0..d_head) |k_idx| {
        const row = ref_ssm_state[k_idx * d_head ..][0..d_head];
        for (0..d_head) |v_idx| {
            row[v_idx] *= g_exp;
            ref_kv_mem[v_idx] += row[v_idx] * key[k_idx];
            ref_ssm_out[v_idx] += row[v_idx] * query[k_idx];
        }
    }
    var qk_dot: f32 = 0.0;
    for (0..d_head) |idx| qk_dot += query[idx] * key[idx];
    for (0..d_head) |v_idx| {
        ref_kv_mem[v_idx] = (value[v_idx] - ref_kv_mem[v_idx]) * beta;
        ref_ssm_out[v_idx] += ref_kv_mem[v_idx] * qk_dot;
    }
    for (0..d_head) |k_idx| {
        const row = ref_ssm_state[k_idx * d_head ..][0..d_head];
        for (0..d_head) |v_idx| {
            row[v_idx] += ref_kv_mem[v_idx] * key[k_idx];
        }
    }

    for (ref_kv_mem, kv_mem) |expected, got| {
        try std.testing.expectApproxEqAbs(expected, got, 1e-5);
    }
    for (ref_ssm_out, ssm_out) |expected, got| {
        try std.testing.expectApproxEqAbs(expected, got, 1e-5);
    }
    for (ref_ssm_state, ssm_state) |expected, got| {
        try std.testing.expectApproxEqAbs(expected, got, 1e-5);
    }
}

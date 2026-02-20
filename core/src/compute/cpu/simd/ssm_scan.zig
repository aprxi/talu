//! SIMD SSM scan kernel for Mamba.
//!
//! Implements the Mamba recurrence:
//!   h = dA * h + dt * B * x
//!   y = C * h + D * x
//! using SIMD over the state dimension.

const std = @import("std");
const simd = @import("arch/root.zig");

pub const StateScanFn = *const fn (
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
) void;

pub fn stateScanF32(
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
    const vec_len: usize = simd.f32_vec_len;

    for (0..n_heads) |h| {
        const head_offset = h * d_head;
        const head_state_offset = h * d_head * d_state;
        const group_idx = h / (n_heads / n_groups);
        const b_offset = group_idx * d_state;
        const c_offset = group_idx * d_state;

        const a_val = -@exp(a_log[h]);
        const dA = @exp(a_val * dt[h]);
        const dt_h = dt[h];

        for (0..d_head) |d| {
            const x_val = x_conv_out[head_offset + d];
            var y_val: f32 = 0;
            const state_base = head_state_offset + d * d_state;

            var s: usize = 0;
            while (s + vec_len <= d_state) : (s += vec_len) {
                var h_vec: @Vector(simd.f32_vec_len, f32) = undefined;
                var b_vec: @Vector(simd.f32_vec_len, f32) = undefined;
                var c_vec: @Vector(simd.f32_vec_len, f32) = undefined;
                inline for (0..simd.f32_vec_len) |lane| {
                    h_vec[lane] = ssm_state[state_base + s + lane];
                    b_vec[lane] = b_raw[b_offset + s + lane];
                    c_vec[lane] = c_raw[c_offset + s + lane];
                }

                const h_new = @as(@Vector(simd.f32_vec_len, f32), @splat(dA)) * h_vec +
                    @as(@Vector(simd.f32_vec_len, f32), @splat(dt_h * x_val)) * b_vec;

                inline for (0..simd.f32_vec_len) |lane| {
                    ssm_state[state_base + s + lane] = h_new[lane];
                }

                y_val += @reduce(.Add, c_vec * h_new);
            }

            while (s < d_state) : (s += 1) {
                const si = state_base + s;
                const h_new = dA * ssm_state[si] + dt_h * b_raw[b_offset + s] * x_val;
                ssm_state[si] = h_new;
                y_val += c_raw[c_offset + s] * h_new;
            }

            y_val += d_skip[h] * x_val;
            ssm_out[head_offset + d] = y_val;
        }
    }
}

test "stateScanF32 preserves zero state with zero inputs" {
    var ssm_state: [8]f32 = .{0} ** 8;
    var ssm_out: [2]f32 = .{0} ** 2;
    const x: [2]f32 = .{0, 0};
    const b: [4]f32 = .{0} ** 4;
    const c: [4]f32 = .{0} ** 4;
    const a_log: [1]f32 = .{0};
    const d_skip: [1]f32 = .{0};
    const dt: [1]f32 = .{0};

    stateScanF32(&ssm_state, &ssm_out, &x, &b, &c, &a_log, &d_skip, &dt, 2, 4, 1, 1);
    try std.testing.expectEqual(@as(f32, 0), ssm_out[0]);
    try std.testing.expectEqual(@as(f32, 0), ssm_out[1]);
}

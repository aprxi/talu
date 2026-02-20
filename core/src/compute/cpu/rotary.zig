//! Rotary-position table primitives for CPU.

const std = @import("std");
const indexing = @import("indexing.zig");

/// Fill inverse-frequency table for RoPE.
///
/// `inv_freq[i] = 1 / rope_theta^(2*i/head_dim)`.
pub fn fillInverseFrequency(inv_freq: []f32, head_dim: usize, rope_theta: f32) !void {
    if ((head_dim % 2) != 0) return error.InvalidShape;
    if (inv_freq.len != head_dim / 2) return error.InvalidShape;
    if (rope_theta <= 0.0) return error.InvalidShape;

    for (0..inv_freq.len) |idx| {
        const exponent = @as(f32, @floatFromInt(2 * idx)) / @as(f32, @floatFromInt(head_dim));
        inv_freq[idx] = 1.0 / std.math.pow(f32, rope_theta, exponent);
    }
}

/// Build combined cos/sin tables from precomputed inverse frequencies and per-token
/// position components.
///
/// Writes `cos` and `sin` as `[seq_len, head_dim]` flattened contiguous buffers.
pub fn buildCosSinTablesFromPositions(
    cos: []f32,
    sin: []f32,
    pos_t: []const u32,
    pos_h: []const u32,
    pos_w: []const u32,
    inv_freq: []const f32,
    head_dim: usize,
    mrope_section: [3]usize,
) !void {
    const seq_len = pos_t.len;
    if (pos_h.len != seq_len or pos_w.len != seq_len) return error.InvalidShape;
    if ((head_dim % 2) != 0) return error.InvalidShape;
    const half_dim = head_dim / 2;
    if (inv_freq.len != half_dim) return error.InvalidShape;
    if (cos.len < seq_len * head_dim or sin.len < seq_len * head_dim) return error.InvalidShape;
    if (mrope_section[0] + mrope_section[1] + mrope_section[2] != half_dim) return error.InvalidShape;

    const h_limit = mrope_section[1] * 3;
    const w_limit = mrope_section[2] * 3;
    for (0..seq_len) |token_idx| {
        const base = token_idx * head_dim;
        for (0..half_dim) |freq_idx| {
            var pos_component = pos_t[token_idx];
            if (freq_idx < h_limit and (freq_idx % 3) == 1) pos_component = pos_h[token_idx];
            if (freq_idx < w_limit and (freq_idx % 3) == 2) pos_component = pos_w[token_idx];

            const angle = @as(f32, @floatFromInt(pos_component)) * inv_freq[freq_idx];
            const c = @cos(angle);
            const s = @sin(angle);
            cos[base + freq_idx] = c;
            sin[base + freq_idx] = s;
            cos[base + half_dim + freq_idx] = c;
            sin[base + half_dim + freq_idx] = s;
        }
    }
}

/// Apply precomputed runtime RoPE tables over Q/K buffers in-place.
pub fn applyRuntimeQK(
    query_values: []f32,
    key_values: []f32,
    sequence_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_dim: usize,
    kv_total_dim: usize,
    pos_offset: usize,
    cos: []const f32,
    sin: []const f32,
    rope_dim: usize,
) !void {
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim % 2) != 0) return error.InvalidShape;
    for (0..sequence_len) |token_idx| {
        const pos = pos_offset + token_idx;
        const base = pos * rope_dim;
        if (base + rope_dim > cos.len or base + rope_dim > sin.len) return error.InvalidShape;
        const cos_row = cos[base .. base + rope_dim];
        const sin_row = sin[base .. base + rope_dim];

        for (0..n_heads) |head_idx| {
            const off = token_idx * query_dim + head_idx * head_dim;
            applyFromCosSin(query_values[off .. off + rope_dim], cos_row, sin_row);
        }
        for (0..n_kv_heads) |head_idx| {
            const off = token_idx * kv_total_dim + head_idx * head_dim;
            applyFromCosSin(key_values[off .. off + rope_dim], cos_row, sin_row);
        }
    }
}

/// Apply static RoPE in-place for Q/K buffers.
///
/// `rope` must expose:
/// - `dim: usize`
/// - `applyInPlace(vec: []f32, pos: usize) void`
pub fn applyStaticQK(
    query_values: []f32,
    key_values: []f32,
    sequence_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_dim: usize,
    kv_total_dim: usize,
    pos_offset: usize,
    position_delta: isize,
    rope: anytype,
) !void {
    const rope_dim = rope.dim;
    if (rope_dim == 0 or rope_dim > head_dim) return error.InvalidShape;
    for (0..sequence_len) |token_idx| {
        const pos = try indexing.offsetSigned(pos_offset + token_idx, position_delta);
        for (0..n_heads) |head_idx| {
            const off = token_idx * query_dim + head_idx * head_dim;
            rope.applyInPlace(query_values[off .. off + rope_dim], pos);
        }
        for (0..n_kv_heads) |head_idx| {
            const off = token_idx * kv_total_dim + head_idx * head_dim;
            rope.applyInPlace(key_values[off .. off + rope_dim], pos);
        }
    }
}

/// Apply interleaved RoPE on one vector in-place.
///
/// `rope` must provide `applyInterleavedInPlace(vec: []f32, pos: usize)`.
pub fn applyInterleavedInPlace(values: []f32, rope: anytype, pos: usize) void {
    rope.applyInterleavedInPlace(values, pos);
}

/// Fill rotary cos/sin tables over 2D spatial positions.
pub fn fillSpatialRotaryTables(
    allocator: std.mem.Allocator,
    height_blocks: usize,
    width_blocks: usize,
    frame_blocks: usize,
    merge_factor: usize,
    feature_width: usize,
    head_count: usize,
    cos_out: []f32,
    sin_out: []f32,
) !void {
    if ((height_blocks % merge_factor) != 0 or (width_blocks % merge_factor) != 0) return error.InvalidShape;

    const merged_h = height_blocks / merge_factor;
    const merged_w = width_blocks / merge_factor;
    const token_count = frame_blocks * height_blocks * width_blocks;
    const head_dim = feature_width / head_count;
    if ((head_dim % 4) != 0) return error.InvalidShape;
    if (cos_out.len != token_count * head_dim or sin_out.len != token_count * head_dim) return error.InvalidShape;

    const half_dim = head_dim / 2;
    const freq_dim = half_dim / 2;

    const inv_freq = try allocator.alloc(f32, freq_dim);
    defer allocator.free(inv_freq);
    for (0..freq_dim) |idx| {
        const exponent = @as(f32, @floatFromInt(2 * idx)) / @as(f32, @floatFromInt(half_dim));
        inv_freq[idx] = 1.0 / std.math.pow(f32, 10000.0, exponent);
    }

    var token_idx: usize = 0;
    for (0..frame_blocks) |_| {
        for (0..merged_h) |bh| {
            for (0..merged_w) |bw| {
                for (0..merge_factor) |ih| {
                    for (0..merge_factor) |iw| {
                        const row = bh * merge_factor + ih;
                        const col = bw * merge_factor + iw;
                        const row_pos = @as(f32, @floatFromInt(row));
                        const col_pos = @as(f32, @floatFromInt(col));
                        const base = token_idx * head_dim;

                        for (0..freq_dim) |f| {
                            const row_angle = row_pos * inv_freq[f];
                            const col_angle = col_pos * inv_freq[f];
                            const row_cos = @cos(row_angle);
                            const row_sin = @sin(row_angle);
                            const col_cos = @cos(col_angle);
                            const col_sin = @sin(col_angle);

                            cos_out[base + f] = row_cos;
                            sin_out[base + f] = row_sin;
                            cos_out[base + freq_dim + f] = col_cos;
                            sin_out[base + freq_dim + f] = col_sin;

                            cos_out[base + half_dim + f] = row_cos;
                            sin_out[base + half_dim + f] = row_sin;
                            cos_out[base + half_dim + freq_dim + f] = col_cos;
                            sin_out[base + half_dim + freq_dim + f] = col_sin;
                        }
                        token_idx += 1;
                    }
                }
            }
        }
    }
    if (token_idx != token_count) return error.InvalidState;
}

/// Apply RoPE over tensor layouts used by graph primitive execution.
///
/// Supported shapes:
/// - 2D: `[seq, dim]`
/// - 3D: `[seq, n_heads, head_dim]`
/// - 4D: `[batch, n_heads, seq, head_dim]`
pub fn applyRopeTensorInPlace(
    input_data: []f32,
    n_dims: usize,
    shape: [8]i64,
    rope_dim: usize,
    pos_offset: usize,
    rope: anytype,
) !void {
    if (n_dims == 2) {
        const seq_len: usize = @intCast(shape[0]);
        const dim: usize = @intCast(shape[1]);
        const active_dim = @min(rope_dim, dim);
        for (0..seq_len) |t| {
            const pos = pos_offset + t;
            const base = t * dim;
            rope.applyInPlace(input_data[base .. base + active_dim], pos);
        }
        return;
    }

    if (n_dims == 3) {
        const seq_len: usize = @intCast(shape[0]);
        const n_heads: usize = @intCast(shape[1]);
        const head_dim: usize = @intCast(shape[2]);
        const active_dim = @min(rope_dim, head_dim);
        const total_dim = n_heads * head_dim;
        for (0..seq_len) |t| {
            const pos = pos_offset + t;
            for (0..n_heads) |h| {
                const base = t * total_dim + h * head_dim;
                rope.applyInPlace(input_data[base .. base + active_dim], pos);
            }
        }
        return;
    }

    if (n_dims == 4) {
        const batch: usize = @intCast(shape[0]);
        const n_heads: usize = @intCast(shape[1]);
        const seq_len: usize = @intCast(shape[2]);
        const head_dim: usize = @intCast(shape[3]);
        const active_dim = @min(rope_dim, head_dim);
        const head_stride = seq_len * head_dim;
        const batch_stride = n_heads * head_stride;
        for (0..batch) |b| {
            for (0..n_heads) |h| {
                for (0..seq_len) |t| {
                    const pos = pos_offset + t;
                    const base = b * batch_stride + h * head_stride + t * head_dim;
                    rope.applyInPlace(input_data[base .. base + active_dim], pos);
                }
            }
        }
        return;
    }

    return error.UnsupportedRopeShape;
}

fn applyFromCosSin(vec: []f32, cos: []const f32, sin: []const f32) void {
    const half = vec.len / 2;
    for (0..half) |idx| {
        const x1 = vec[idx];
        const x2 = vec[idx + half];
        vec[idx] = x1 * cos[idx] - x2 * sin[idx];
        vec[idx + half] = x2 * cos[idx + half] + x1 * sin[idx + half];
    }
}

test "buildCosSinTablesFromPositions fills duplicated halves" {
    const head_dim: usize = 4;
    const seq_len: usize = 2;
    const inv_freq = [_]f32{ 1.0, 0.5 };
    const pos_t = [_]u32{ 0, 1 };
    const pos_h = [_]u32{ 0, 1 };
    const pos_w = [_]u32{ 0, 1 };
    var cos = [_]f32{0} ** (seq_len * head_dim);
    var sin = [_]f32{0} ** (seq_len * head_dim);

    try buildCosSinTablesFromPositions(
        &cos,
        &sin,
        &pos_t,
        &pos_h,
        &pos_w,
        &inv_freq,
        head_dim,
        .{ 2, 0, 0 },
    );

    // token 0 => angle 0 for all freqs
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cos[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sin[0], 1e-6);
    // duplicated upper half
    try std.testing.expectApproxEqAbs(cos[0], cos[2], 1e-6);
    try std.testing.expectApproxEqAbs(sin[1], sin[3], 1e-6);
}

test "fillInverseFrequency computes expected first element" {
    var inv = [_]f32{0} ** 2;
    try fillInverseFrequency(&inv, 4, 10000.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), inv[0], 1e-6);
}

test "applyRuntimeQK rotates one token" {
    var q = [_]f32{ 1, 2, 3, 4 };
    var k = [_]f32{ 5, 6, 7, 8 };
    const cos = [_]f32{ 1, 1, 1, 1 };
    const sin = [_]f32{ 0, 0, 0, 0 };

    try applyRuntimeQK(&q, &k, 1, 1, 1, 4, 4, 4, 0, &cos, &sin, 4);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 4 }, &q);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 5, 6, 7, 8 }, &k);
}

test "applyStaticQK applies position offset to each head slice" {
    const MockRope = struct {
        dim: usize = 2,
        pub fn applyInPlace(_: *@This(), vec: []f32, pos: usize) void {
            const pos_f: f32 = @floatFromInt(pos);
            for (vec) |*v| v.* += pos_f;
        }
    };

    var rope = MockRope{};
    var q = [_]f32{ 1, 2, 3, 4 };
    var k = [_]f32{ 5, 6, 7, 8 };

    try applyStaticQK(&q, &k, 1, 2, 2, 2, 4, 4, 3, 0, &rope);

    try std.testing.expectEqualSlices(f32, &[_]f32{ 4, 5, 6, 7 }, &q);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 8, 9, 10, 11 }, &k);
}

test "applyInterleavedInPlace delegates to rope implementation" {
    const MockRope = struct {
        pub fn applyInterleavedInPlace(_: *@This(), vec: []f32, pos: usize) void {
            const p: f32 = @floatFromInt(pos);
            for (vec) |*v| v.* += p;
        }
    };

    var rope = MockRope{};
    var values = [_]f32{ 1.0, 2.0 };
    applyInterleavedInPlace(&values, &rope, 3);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4.0, 5.0 }, &values);
}

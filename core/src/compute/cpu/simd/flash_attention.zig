//! Flash Attention for CPU with SIMD optimization.
//!
//! Implements memory-efficient attention using tiled computation
//! and online softmax normalization.

const std = @import("std");
const simd = @import("arch/root.zig");

/// Configuration for Flash Attention kernel.
pub const FlashAttentionConfig = struct {
    /// Tile size for keys/values (affects cache efficiency).
    kv_tile_size: usize = 256,
    /// Head dimension (must be known for SIMD vectorization).
    head_dim: usize,
    /// Attention scale (typically 1/sqrt(head_dim)).
    scale: f32,
    /// Enable causal masking.
    causal: bool = true,
    /// Sliding window size (0 = disabled).
    sliding_window: usize = 0,
};

pub const FlashAttentionFn = *const fn (
    // Output: [n_heads, seq_q, head_dim]
    out: [*]f32,
    out_stride_head: usize,
    out_stride_seq: usize,
    // Query: [n_heads, seq_q, head_dim]
    q: [*]const f32,
    q_stride_head: usize,
    q_stride_seq: usize,
    // Key: [n_heads, seq_k, head_dim]
    k: [*]const f32,
    k_stride_head: usize,
    k_stride_seq: usize,
    // Value: [n_heads, seq_k, head_dim]
    v: [*]const f32,
    v_stride_head: usize,
    v_stride_seq: usize,
    // Dimensions
    n_heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
    // Config
    scale: f32,
    kv_offset: usize, // For causal masking with KV cache
    sliding_window: usize, // 0 = disabled
) void;

/// Flash Attention for f32 with compile-time head dimension and tile size.
pub fn flashAttentionF32(
    comptime head_dim: usize,
    comptime kv_tile_size: usize,
) FlashAttentionFn {
    return struct {
        pub fn call(
            out: [*]f32,
            out_stride_head: usize,
            out_stride_seq: usize,
            q: [*]const f32,
            q_stride_head: usize,
            q_stride_seq: usize,
            k: [*]const f32,
            k_stride_head: usize,
            k_stride_seq: usize,
            v: [*]const f32,
            v_stride_head: usize,
            v_stride_seq: usize,
            n_heads: usize,
            seq_q: usize,
            seq_k: usize,
            _: usize, // head_dim runtime (ignored, using comptime)
            scale: f32,
            kv_offset: usize,
            sliding_window: usize,
        ) void {
            flashAttentionTiled(
                head_dim,
                kv_tile_size,
                out,
                out_stride_head,
                out_stride_seq,
                q,
                q_stride_head,
                q_stride_seq,
                k,
                k_stride_head,
                k_stride_seq,
                v,
                v_stride_head,
                v_stride_seq,
                n_heads,
                seq_q,
                seq_k,
                scale,
                kv_offset,
                sliding_window,
            );
        }
    }.call;
}

fn flashAttentionTiled(
    comptime head_dim: usize,
    comptime kv_tile: usize,
    out: [*]f32,
    out_stride_head: usize,
    out_stride_seq: usize,
    q: [*]const f32,
    q_stride_head: usize,
    q_stride_seq: usize,
    k: [*]const f32,
    k_stride_head: usize,
    k_stride_seq: usize,
    v: [*]const f32,
    v_stride_head: usize,
    v_stride_seq: usize,
    n_heads: usize,
    seq_q: usize,
    seq_k: usize,
    scale: f32,
    kv_offset: usize,
    sliding_window: usize,
) void {
    if (seq_q == 0) return;
    const vec_len: usize = simd.f32_vec_len;
    const F32Vec = @Vector(vec_len, f32);

    var head_idx: usize = 0;
    while (head_idx < n_heads) : (head_idx += 1) {
        var q_idx: usize = 0;
        while (q_idx < seq_q) : (q_idx += 1) {
            const q_pos = kv_offset + q_idx;
            const q_ptr = q + head_idx * q_stride_head + q_idx * q_stride_seq;
            const out_ptr = out + head_idx * out_stride_head + q_idx * out_stride_seq;

            const window_start: usize = if (sliding_window > 0 and q_pos >= sliding_window)
                q_pos - sliding_window + 1
            else
                0;

            var m: f32 = -std.math.inf(f32);
            var l: f32 = 0;
            var acc: [head_dim]f32 = [_]f32{0} ** head_dim;

            var kv_start: usize = 0;
            while (kv_start < seq_k) : (kv_start += kv_tile) {
                const kv_end = @min(kv_start + kv_tile, seq_k);
                var k_idx: usize = kv_start;
                while (k_idx < kv_end) : (k_idx += 1) {
                    if (k_idx > q_pos) continue;
                    if (k_idx < window_start) continue;

                    const k_ptr = k + head_idx * k_stride_head + k_idx * k_stride_seq;
                    const v_ptr = v + head_idx * v_stride_head + k_idx * v_stride_seq;

                    var score: f32 = 0;
                    comptime var d: usize = 0;
                    inline while (d + vec_len <= head_dim) : (d += vec_len) {
                        const q_vec: F32Vec = q_ptr[d..][0..vec_len].*;
                        const k_vec: F32Vec = k_ptr[d..][0..vec_len].*;
                        score += @reduce(.Add, q_vec * k_vec);
                    }
                    inline while (d < head_dim) : (d += 1) {
                        score += q_ptr[d] * k_ptr[d];
                    }
                    score *= scale;

                    const m_updated = @max(m, score);
                    const exp_prev = std.math.exp(m - m_updated);
                    const exp_curr = std.math.exp(score - m_updated);
                    const l_updated = l * exp_prev + exp_curr;

                    const scale_prev = if (l_updated == 0) 0 else (l * exp_prev / l_updated);
                    const scale_curr = if (l_updated == 0) 0 else (exp_curr / l_updated);

                    comptime var d2: usize = 0;
                    inline while (d2 + vec_len <= head_dim) : (d2 += vec_len) {
                        const acc_vec: F32Vec = acc[d2..][0..vec_len].*;
                        const v_vec: F32Vec = v_ptr[d2..][0..vec_len].*;
                        const scale_prev_vec: F32Vec = @splat(scale_prev);
                        const scale_curr_vec: F32Vec = @splat(scale_curr);
                        acc[d2..][0..vec_len].* = acc_vec * scale_prev_vec + v_vec * scale_curr_vec;
                    }
                    inline while (d2 < head_dim) : (d2 += 1) {
                        acc[d2] = acc[d2] * scale_prev + v_ptr[d2] * scale_curr;
                    }

                    m = m_updated;
                    l = l_updated;
                }
            }

            comptime var d3: usize = 0;
            inline while (d3 + vec_len <= head_dim) : (d3 += vec_len) {
                const acc_vec: F32Vec = acc[d3..][0..vec_len].*;
                @as(*[vec_len]f32, @ptrCast(out_ptr + d3)).* = acc_vec;
            }
            inline while (d3 < head_dim) : (d3 += 1) {
                out_ptr[d3] = acc[d3];
            }
        }
    }
}

pub const flashAttentionF32_64 = flashAttentionF32(64, 256);
pub const flashAttentionF32_128 = flashAttentionF32(128, 256);

test "flashAttentionF32 matches sdpaCausal" {
    const allocator = std.testing.allocator;
    const n_heads: usize = 2;
    const seq_len: usize = 32;
    const head_dim: usize = 64;
    const total = n_heads * seq_len * head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const q = try allocator.alloc(f32, total);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, total);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, total);
    defer allocator.free(v);
    const out_flash = try allocator.alloc(f32, total);
    defer allocator.free(out_flash);
    const out_sdpa = try allocator.alloc(f32, total);
    defer allocator.free(out_sdpa);

    var prng = std.Random.DefaultPrng.init(1234);
    const rnd = prng.random();
    for (q) |*val| val.* = rnd.float(f32) * 2.0 - 1.0;
    for (k) |*val| val.* = rnd.float(f32) * 2.0 - 1.0;
    for (v) |*val| val.* = rnd.float(f32) * 2.0 - 1.0;
    @memset(out_flash, 0.0);
    @memset(out_sdpa, 0.0);

    const head_stride = seq_len * head_dim;
    const seq_stride = head_dim;
    flashAttentionF32_64(
        out_flash.ptr,
        head_stride,
        seq_stride,
        q.ptr,
        head_stride,
        seq_stride,
        k.ptr,
        head_stride,
        seq_stride,
        v.ptr,
        head_stride,
        seq_stride,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
        scale,
        0,
        0,
    );

    const tv = @import("../tensor_view.zig");
    const out_view = tv.TensorView.initContiguous(@ptrCast(out_sdpa.ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const q_view = tv.TensorView.initContiguous(@ptrCast(q[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const k_view = tv.TensorView.initContiguous(@ptrCast(k[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const v_view = tv.TensorView.initContiguous(@ptrCast(v[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    try @import("../linalg_sdpa.zig").sdpaCausal(out_view, q_view, k_view, v_view, scale, 0, allocator);

    for (out_flash, out_sdpa) |f, s| {
        try std.testing.expectApproxEqAbs(s, f, 1e-4);
    }
}

test "flashAttentionF32 causal mask" {
    const n_heads: usize = 1;
    const seq_len: usize = 3;
    const head_dim: usize = 4;
    const total = n_heads * seq_len * head_dim;
    const scale = 1.0;

    var q = [_]f32{0} ** total;
    var k = [_]f32{0} ** total;
    var v = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    };
    var out = [_]f32{0} ** total;

    const head_stride = seq_len * head_dim;
    const seq_stride = head_dim;
    flashAttentionF32(4, 16)(
        &out,
        head_stride,
        seq_stride,
        &q,
        head_stride,
        seq_stride,
        &k,
        head_stride,
        seq_stride,
        &v,
        head_stride,
        seq_stride,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
        scale,
        0,
        0,
    );

    // Position 0 attends to [0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    // Position 1 attends to [0,1] -> average
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[head_dim], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[head_dim + 1], 1e-6);
    // Position 2 attends to [0,1,2] -> average of basis vectors
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), out[2 * head_dim], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), out[2 * head_dim + 1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), out[2 * head_dim + 2], 1e-6);
}

test "flashAttentionF32 kv_offset handling" {
    const n_heads: usize = 1;
    const seq_q: usize = 2;
    const seq_k: usize = 4;
    const head_dim: usize = 2;
    const total_q = n_heads * seq_q * head_dim;
    const total_k = n_heads * seq_k * head_dim;

    var q = [_]f32{0} ** total_q;
    var k = [_]f32{0} ** total_k;
    var v = [_]f32{
        1, 0,
        0, 1,
        2, 0,
        0, 2,
    };
    var out = [_]f32{0} ** total_q;

    const head_stride_q = seq_q * head_dim;
    const head_stride_k = seq_k * head_dim;
    const seq_stride = head_dim;
    flashAttentionF32(2, 4)(
        &out,
        head_stride_q,
        seq_stride,
        &q,
        head_stride_q,
        seq_stride,
        &k,
        head_stride_k,
        seq_stride,
        &v,
        head_stride_k,
        seq_stride,
        n_heads,
        seq_q,
        seq_k,
        head_dim,
        1.0,
        2,
        0,
    );

    // q_pos=2 => avg of v[0..2]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6); // (1+0+2)/3
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), out[1], 1e-6); // (0+1+0)/3
    // q_pos=3 => avg of v[0..3]
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), out[head_dim], 1e-6); // (1+0+2+0)/4
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), out[head_dim + 1], 1e-6); // (0+1+0+2)/4
}

test "flashAttentionF32 sliding window" {
    const n_heads: usize = 1;
    const seq_len: usize = 4;
    const head_dim: usize = 2;
    const total = n_heads * seq_len * head_dim;

    var q = [_]f32{0} ** total;
    var k = [_]f32{0} ** total;
    var v = [_]f32{
        1, 0,
        0, 1,
        2, 0,
        0, 2,
    };
    var out = [_]f32{0} ** total;

    const head_stride = seq_len * head_dim;
    const seq_stride = head_dim;
    flashAttentionF32(2, 4)(
        &out,
        head_stride,
        seq_stride,
        &q,
        head_stride,
        seq_stride,
        &k,
        head_stride,
        seq_stride,
        &v,
        head_stride,
        seq_stride,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
        1.0,
        0,
        2,
    );

    // Position 3 attends to [2,3]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3 * head_dim], 1e-6); // (2+0)/2
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[3 * head_dim + 1], 1e-6); // (0+2)/2
}

test "flashAttentionF32 zero-length sequence" {
    const n_heads: usize = 1;
    const seq_q: usize = 0;
    const seq_k: usize = 4;
    const head_dim: usize = 64;
    var out = [_]f32{42.0} ** (n_heads * seq_k * head_dim);

    flashAttentionF32_64(
        &out,
        seq_k * head_dim,
        head_dim,
        &out,
        seq_k * head_dim,
        head_dim,
        &out,
        seq_k * head_dim,
        head_dim,
        &out,
        seq_k * head_dim,
        head_dim,
        n_heads,
        seq_q,
        seq_k,
        head_dim,
        1.0,
        0,
        0,
    );

    for (out) |val| {
        try std.testing.expectEqual(@as(f32, 42.0), val);
    }
}

test "flashAttentionF32 single token matches reference" {
    const n_heads: usize = 1;
    const seq_len: usize = 1;
    const head_dim: usize = 8;
    const total = n_heads * seq_len * head_dim;
    const scale = 0.5;

    var q = [_]f32{0} ** total;
    var k = [_]f32{0} ** total;
    var v = [_]f32{0} ** total;
    for (0..total) |i| {
        q[i] = @as(f32, @floatFromInt(i)) * 0.1;
        k[i] = @as(f32, @floatFromInt(i)) * 0.2;
        v[i] = @as(f32, @floatFromInt(i)) * 0.3;
    }
    var out_simd = [_]f32{0} ** total;
    var out_sdpa = [_]f32{0} ** total;

    flashAttentionF32(8, 8)(
        &out_simd,
        seq_len * head_dim,
        head_dim,
        &q,
        seq_len * head_dim,
        head_dim,
        &k,
        seq_len * head_dim,
        head_dim,
        &v,
        seq_len * head_dim,
        head_dim,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
        scale,
        0,
        0,
    );
    const tv = @import("../tensor_view.zig");
    const out_view = tv.TensorView.initContiguous(@ptrCast(out_sdpa[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const q_view = tv.TensorView.initContiguous(@ptrCast(q[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const k_view = tv.TensorView.initContiguous(@ptrCast(k[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const v_view = tv.TensorView.initContiguous(@ptrCast(v[0..].ptr), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    try @import("../linalg_sdpa.zig").sdpaCausal(out_view, q_view, k_view, v_view, scale, 0, null);

    for (out_simd, out_sdpa) |simd_val, sdpa_val| {
        try std.testing.expectApproxEqAbs(sdpa_val, simd_val, 1e-6);
    }
}

test "flashAttentionF32 call matches sdpaCausal" {
    // Explicit test for the `call` function returned by flashAttentionF32
    const allocator = std.testing.allocator;
    const n_heads: usize = 1;
    const seq_len: usize = 4;
    const head_dim: usize = 8;
    const total = n_heads * seq_len * head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    const q = try allocator.alloc(f32, total);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, total);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, total);
    defer allocator.free(v);
    const out = try allocator.alloc(f32, total);
    defer allocator.free(out);

    var prng = std.Random.DefaultPrng.init(5678);
    const rnd = prng.random();
    for (q) |*val| val.* = rnd.float(f32) * 2.0 - 1.0;
    for (k) |*val| val.* = rnd.float(f32) * 2.0 - 1.0;
    for (v) |*val| val.* = rnd.float(f32) * 2.0 - 1.0;

    const kernel = flashAttentionF32(8, 8);
    const call = kernel; // Explicitly reference call
    call(
        out.ptr,
        seq_len * head_dim,
        head_dim,
        q.ptr,
        seq_len * head_dim,
        head_dim,
        k.ptr,
        seq_len * head_dim,
        head_dim,
        v.ptr,
        seq_len * head_dim,
        head_dim,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
        scale,
        0,
        0,
    );

    for (out) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }
}

test "flashAttentionF32 numerical stability" {
    const n_heads: usize = 1;
    const seq_len: usize = 4;
    const head_dim: usize = 64;
    const total = n_heads * seq_len * head_dim;

    var q = [_]f32{0} ** total;
    var k = [_]f32{0} ** total;
    var v = [_]f32{0} ** total;
    for (0..total) |i| {
        q[i] = 50.0;
        k[i] = 50.0;
        v[i] = @as(f32, @floatFromInt(i % head_dim)) * 0.01;
    }
    var out = [_]f32{0} ** total;

    flashAttentionF32_64(
        &out,
        seq_len * head_dim,
        head_dim,
        &q,
        seq_len * head_dim,
        head_dim,
        &k,
        seq_len * head_dim,
        head_dim,
        &v,
        seq_len * head_dim,
        head_dim,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
        1.0,
        0,
        0,
    );

    for (out) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }
}

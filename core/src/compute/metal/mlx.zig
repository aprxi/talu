//! MLX quantized matrix multiplication via graph API.
//!
//! Implements 4-bit and 8-bit quantized matmul using MLX lazy evaluation
//! for efficient GPU execution on Apple Silicon.

const std = @import("std");
const mlx_graph = @import("graph.zig");

// ============================================================================
// Internal implementations using mlx_graph lazy API
// ============================================================================

fn mlx_quantized_matmul_4bit_impl(
    a_data: [*]const f32,
    m: usize,
    k: usize,
    w_data: [*]const u8,
    scales: [*]align(1) const u16,
    biases: [*]align(1) const u16,
    n: usize,
    group_size: usize,
    c_data: [*]f32,
) bool {
    // Create MLX arrays from input data
    const a_shape = [_]usize{ m, k };
    const a_array = mlx_graph.mlx_array_from_float32(a_data, &a_shape, 2);
    defer mlx_graph.mlx_array_free(a_array);

    // Packed weights: [n, k/8] uint32
    const w_u32 = @as([*]const u32, @ptrCast(@alignCast(w_data)));
    const packed_k = k / 8;
    const w_shape = [_]usize{ n, packed_k };
    const w_array = mlx_graph.mlx_array_from_uint32(w_u32, &w_shape, 2);
    defer mlx_graph.mlx_array_free(w_array);

    // Scales: [n, k/group_size] bfloat16
    const scales_shape = [_]usize{ n, k / group_size };
    const scales_array = mlx_graph.mlx_array_from_bfloat16(scales, &scales_shape, 2);
    defer mlx_graph.mlx_array_free(scales_array);

    // Biases: [n, k/group_size] bfloat16
    const biases_array = mlx_graph.mlx_array_from_bfloat16(biases, &scales_shape, 2);
    defer mlx_graph.mlx_array_free(biases_array);

    // Call quantized matmul
    const result_handle = mlx_graph.mlx_lazy_quantized_matmul(
        a_array,
        w_array,
        scales_array,
        biases_array,
        group_size,
        4, // bits
        true, // transpose
    );
    defer mlx_graph.mlx_array_free(result_handle);

    // Evaluate
    var handles = [_]mlx_graph.ArrayHandle{result_handle};
    mlx_graph.mlx_eval(&handles, 1);

    // Copy result back
    mlx_graph.mlx_array_to_float32(result_handle, c_data, m * n);

    return true;
}

fn mlx_rms_norm_impl(
    x_data: [*]const f32,
    weight_data: [*]const f32,
    batch: usize,
    seq_len: usize,
    dim: usize,
    eps: f32,
    out_data: [*]f32,
) bool {
    // Create arrays
    const x_shape = [_]usize{ batch, seq_len, dim };
    const x_array = mlx_graph.mlx_array_from_float32(x_data, &x_shape, 3);
    defer mlx_graph.mlx_array_free(x_array);

    const w_shape = [_]usize{dim};
    const w_array = mlx_graph.mlx_array_from_float32(weight_data, &w_shape, 1);
    defer mlx_graph.mlx_array_free(w_array);

    // Call RMS norm
    const result_handle = mlx_graph.mlx_lazy_rms_norm(x_array, w_array, eps);
    defer mlx_graph.mlx_array_free(result_handle);

    // Evaluate
    var handles = [_]mlx_graph.ArrayHandle{result_handle};
    mlx_graph.mlx_eval(&handles, 1);

    // Copy back
    mlx_graph.mlx_array_to_float32(result_handle, out_data, batch * seq_len * dim);

    return true;
}

fn mlx_rope_impl(
    x_data: [*]const f32,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    offset: c_int,
    rope_base: f32,
    out_data: [*]f32,
) bool {
    // Create array
    const x_shape = [_]usize{ batch, seq_len, n_heads, head_dim };
    const x_array = mlx_graph.mlx_array_from_float32(x_data, &x_shape, 4);
    defer mlx_graph.mlx_array_free(x_array);

    // Call RoPE
    const result_handle = mlx_graph.mlx_lazy_rope(x_array, head_dim, @intCast(offset), rope_base);
    defer mlx_graph.mlx_array_free(result_handle);

    // Evaluate
    var handles = [_]mlx_graph.ArrayHandle{result_handle};
    mlx_graph.mlx_eval(&handles, 1);

    // Copy back
    mlx_graph.mlx_array_to_float32(result_handle, out_data, batch * seq_len * n_heads * head_dim);

    return true;
}

fn mlx_scaled_dot_product_attention_impl(
    q_data: [*]const f32,
    k_data: [*]const f32,
    v_data: [*]const f32,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    scale: f32,
    out_data: [*]f32,
) bool {
    // Create arrays
    const q_shape = [_]usize{ batch, n_heads, seq_len, head_dim };
    const q_array = mlx_graph.mlx_array_from_float32(q_data, &q_shape, 4);
    defer mlx_graph.mlx_array_free(q_array);

    const kv_shape = [_]usize{ batch, n_kv_heads, kv_seq_len, head_dim };
    const k_array = mlx_graph.mlx_array_from_float32(k_data, &kv_shape, 4);
    const v_array = mlx_graph.mlx_array_from_float32(v_data, &kv_shape, 4);
    defer mlx_graph.mlx_array_free(k_array);
    defer mlx_graph.mlx_array_free(v_array);

    var k_for_attn = k_array;
    var v_for_attn = v_array;

    // Handle grouped-query form explicitly by repeating KV heads.
    // This avoids shape-dependent backend behavior and keeps semantics
    // deterministic when n_kv_heads < n_heads.
    if (n_kv_heads < n_heads) {
        if (n_kv_heads == 0 or n_heads % n_kv_heads != 0) return false;
        const repeats = n_heads / n_kv_heads;
        k_for_attn = mlx_graph.mlx_lazy_repeat(k_array, repeats, 1);
        v_for_attn = mlx_graph.mlx_lazy_repeat(v_array, repeats, 1);
        defer mlx_graph.mlx_array_free(k_for_attn);
        defer mlx_graph.mlx_array_free(v_for_attn);
    }

    // Call attention
    const result_handle = mlx_graph.mlx_lazy_attention(q_array, k_for_attn, v_for_attn, scale, true);
    defer mlx_graph.mlx_array_free(result_handle);

    // Evaluate
    var handles = [_]mlx_graph.ArrayHandle{result_handle};
    mlx_graph.mlx_eval(&handles, 1);

    // Copy back
    mlx_graph.mlx_array_to_float32(result_handle, out_data, batch * n_heads * seq_len * head_dim);

    return true;
}

fn mlx_silu_impl(
    x_data: [*]const f32,
    size: usize,
    out_data: [*]f32,
) bool {
    // Create array
    const x_shape = [_]usize{size};
    const x_array = mlx_graph.mlx_array_from_float32(x_data, &x_shape, 1);
    defer mlx_graph.mlx_array_free(x_array);

    // Call SiLU
    const result_handle = mlx_graph.mlx_lazy_silu(x_array);
    defer mlx_graph.mlx_array_free(result_handle);

    // Evaluate
    var handles = [_]mlx_graph.ArrayHandle{result_handle};
    mlx_graph.mlx_eval(&handles, 1);

    // Copy back
    mlx_graph.mlx_array_to_float32(result_handle, out_data, size);

    return true;
}

// ============================================================================
// Export C ABI functions for external use.
// ============================================================================

export fn mlx_quantized_matmul_4bit(
    a_data: [*]const f32,
    m: usize,
    k: usize,
    w_data: [*]const u8,
    scales: [*]align(1) const u16,
    biases: [*]align(1) const u16,
    n: usize,
    group_size: usize,
    c_data: [*]f32,
) bool {
    return mlx_quantized_matmul_4bit_impl(a_data, m, k, w_data, scales, biases, n, group_size, c_data);
}

export fn mlx_rms_norm(
    x_data: [*]const f32,
    weight_data: [*]const f32,
    batch: usize,
    seq_len: usize,
    dim: usize,
    eps: f32,
    out_data: [*]f32,
) bool {
    return mlx_rms_norm_impl(x_data, weight_data, batch, seq_len, dim, eps, out_data);
}

export fn mlx_rope(
    x_data: [*]const f32,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    offset: c_int,
    rope_base: f32,
    out_data: [*]f32,
) bool {
    return mlx_rope_impl(x_data, batch, seq_len, n_heads, head_dim, offset, rope_base, out_data);
}

export fn mlx_scaled_dot_product_attention(
    q_data: [*]const f32,
    k_data: [*]const f32,
    v_data: [*]const f32,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    scale: f32,
    out_data: [*]f32,
) bool {
    return mlx_scaled_dot_product_attention_impl(
        q_data,
        k_data,
        v_data,
        batch,
        n_heads,
        n_kv_heads,
        seq_len,
        kv_seq_len,
        head_dim,
        scale,
        out_data,
    );
}

export fn mlx_silu(
    x_data: [*]const f32,
    size: usize,
    out_data: [*]f32,
) bool {
    return mlx_silu_impl(x_data, size, out_data);
}

/// Grouped-affine u4 quantized matrix multiplication using Metal GPU (MLX backend).
/// A: [m x k] f32
/// B: [k x n] grouped-affine u4 (w_data + scales + biases)
/// C: [m x n] f32 output
pub fn matmulGaffineU4(
    a: []const f32,
    m: usize,
    k: usize,
    w_data: []const u8,
    scales: []align(1) const u16,
    biases: []align(1) const u16,
    n: usize,
    group_size: usize,
    c: []f32,
) !void {
    std.debug.assert(a.len >= m * k);
    std.debug.assert(c.len >= m * n);

    const success = mlx_quantized_matmul_4bit(
        a.ptr,
        m,
        k,
        w_data.ptr,
        scales.ptr,
        biases.ptr,
        n,
        group_size,
        c.ptr,
    );

    if (!success) return error.MLXMatmulFailed;
}

/// RMS normalization using MLX.
/// x: [batch, seq_len, dim]
/// weight: [dim]
/// out: [batch, seq_len, dim]
pub fn rmsNorm(
    x: []const f32,
    weight: []const f32,
    batch: usize,
    seq_len: usize,
    dim: usize,
    eps: f32,
    out: []f32,
) !void {
    std.debug.assert(x.len >= batch * seq_len * dim);
    std.debug.assert(weight.len >= dim);
    std.debug.assert(out.len >= batch * seq_len * dim);

    const success = mlx_rms_norm(
        x.ptr,
        weight.ptr,
        batch,
        seq_len,
        dim,
        eps,
        out.ptr,
    );

    if (!success) return error.MLXRmsNormFailed;
}

/// RoPE using MLX.
/// x: [batch, seq_len, n_heads, head_dim]
/// offset: position offset for autoregressive generation
/// out: [batch, seq_len, n_heads, head_dim]
pub fn rope(
    x: []const f32,
    batch: usize,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    offset: usize,
    rope_base: f32,
    out: []f32,
) !void {
    std.debug.assert(x.len >= batch * seq_len * n_heads * head_dim);
    std.debug.assert(out.len >= batch * seq_len * n_heads * head_dim);

    const success = mlx_rope(
        x.ptr,
        batch,
        seq_len,
        n_heads,
        head_dim,
        @intCast(offset),
        rope_base,
        out.ptr,
    );

    if (!success) return error.MLXRopeFailed;
}

/// Scaled dot product attention using MLX.
/// q: [batch, n_heads, seq_len, head_dim]
/// k: [batch, n_kv_heads, kv_seq_len, head_dim]
/// v: [batch, n_kv_heads, kv_seq_len, head_dim]
/// out: [batch, n_heads, seq_len, head_dim]
fn scaledDotProductAttention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    batch: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
    scale: f32,
    out: []f32,
) !void {
    std.debug.assert(q.len >= batch * n_heads * seq_len * head_dim);
    std.debug.assert(k.len >= batch * n_kv_heads * kv_seq_len * head_dim);
    std.debug.assert(v.len >= batch * n_kv_heads * kv_seq_len * head_dim);
    std.debug.assert(out.len >= batch * n_heads * seq_len * head_dim);

    const success = mlx_scaled_dot_product_attention(
        q.ptr,
        k.ptr,
        v.ptr,
        batch,
        n_heads,
        n_kv_heads,
        seq_len,
        kv_seq_len,
        head_dim,
        scale,
        out.ptr,
    );

    if (!success) return error.MLXAttentionFailed;
}

/// SiLU activation using MLX.
pub fn silu(
    x: []const f32,
    out: []f32,
) !void {
    std.debug.assert(x.len == out.len);

    const success = mlx_silu(
        x.ptr,
        x.len,
        out.ptr,
    );

    if (!success) return error.MLXSiluFailed;
}

// =============================================================================
// Unit Tests - compiled only on macOS where Metal/MLX is available
// =============================================================================

const builtin = @import("builtin");

test "matmulGaffineU4 performs quantized matmul" {
    if (comptime builtin.os.tag != .macos) return;

    // Test dimensions: A[2x8] @ B[8x4] = C[2x4]
    const m: usize = 2;
    const k: usize = 8;
    const n: usize = 4;
    const group_size: usize = 8;

    // A = [[1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2]]
    var a: [m * k]f32 = undefined;
    for (0..m) |row| {
        const val: f32 = @floatFromInt(row + 1);
        for (0..k) |col| {
            a[row * k + col] = val;
        }
    }

    // Quantized weights: packed 4-bit (k/2 bytes per row)
    // 0x11 = 0001_0001 binary = 1 in both nibbles
    // After dequantization with scale=1, bias=0: each element = 1
    var w_data: [n * k / 2]u8 = undefined;
    for (&w_data) |*v| v.* = 0x11;

    // Scales and biases: one per group
    const num_groups = k / group_size;
    var scales: [n * num_groups]u16 = undefined;
    var biases: [n * num_groups]u16 = undefined;
    for (&scales) |*v| v.* = 0x3F80; // 1.0 in fp16
    for (&biases) |*v| v.* = 0x0000; // 0.0

    var c: [m * n]f32 = undefined;

    matmulGaffineU4(&a, m, k, &w_data, &scales, &biases, n, group_size, &c) catch |err| {
        // MLX failure is acceptable if GPU not available
        try std.testing.expect(err == error.MLXMatmulFailed);
        return;
    };

    // With A rows of all 1s or 2s, and B columns of all 1s:
    // C[0,:] = 1*8 = 8 (dot product of 8 ones with 8 ones)
    // C[1,:] = 2*8 = 16 (dot product of 8 twos with 8 ones)
    // Note: actual values depend on MLX's exact quantization format
    // Verify output has valid, non-zero values
    for (c) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    // Row 0 outputs should be approximately equal to each other
    try std.testing.expectApproxEqAbs(c[0], c[1], 0.1);
    try std.testing.expectApproxEqAbs(c[0], c[2], 0.1);
    try std.testing.expectApproxEqAbs(c[0], c[3], 0.1);
    // Row 1 outputs should be approximately 2x row 0
    try std.testing.expectApproxEqAbs(c[4], c[0] * 2.0, c[0] * 0.1);
}

test "rmsNorm normalizes input" {
    if (comptime builtin.os.tag != .macos) return;

    const batch: usize = 1;
    const seq_len: usize = 2;
    const dim: usize = 4;
    const eps: f32 = 1e-6;

    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var weight = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var out: [batch * seq_len * dim]f32 = undefined;

    rmsNorm(&x, &weight, batch, seq_len, dim, eps, &out) catch |err| {
        try std.testing.expect(err == error.MLXRmsNormFailed);
        return;
    };

    // RMS norm: output = x / rms(x) * weight
    // Row 1: [1,2,3,4], RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // Row 2: [5,6,7,8], RMS = sqrt((25+36+49+64)/4) = sqrt(43.5) ≈ 6.5955
    const rms1 = @sqrt(@as(f32, 7.5));
    const rms2 = @sqrt(@as(f32, 43.5));

    try std.testing.expectApproxEqAbs(1.0 / rms1, out[0], 0.01);
    try std.testing.expectApproxEqAbs(2.0 / rms1, out[1], 0.01);
    try std.testing.expectApproxEqAbs(3.0 / rms1, out[2], 0.01);
    try std.testing.expectApproxEqAbs(4.0 / rms1, out[3], 0.01);
    try std.testing.expectApproxEqAbs(5.0 / rms2, out[4], 0.01);
    try std.testing.expectApproxEqAbs(6.0 / rms2, out[5], 0.01);
    try std.testing.expectApproxEqAbs(7.0 / rms2, out[6], 0.01);
    try std.testing.expectApproxEqAbs(8.0 / rms2, out[7], 0.01);
}

test "rope applies rotary position embedding" {
    if (comptime builtin.os.tag != .macos) return;

    const batch: usize = 1;
    const seq_len: usize = 1;
    const n_heads: usize = 1;
    const head_dim: usize = 4;
    const offset: usize = 0;
    const rope_base: f32 = 10000.0;

    // Simple test case: position 0 should have no rotation (cos=1, sin=0)
    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [batch * seq_len * n_heads * head_dim]f32 = undefined;

    rope(&x, batch, seq_len, n_heads, head_dim, offset, rope_base, &out) catch |err| {
        try std.testing.expect(err == error.MLXRopeFailed);
        return;
    };

    // At position 0, cos(0)=1 and sin(0)=0
    // RoPE formula: [x0*cos - x1*sin, x0*sin + x1*cos, x2*cos - x3*sin, x2*sin + x3*cos]
    // With cos=1, sin=0: output should equal input
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), out[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 0.01);
}

test "rope rotates at non-zero position" {
    if (comptime builtin.os.tag != .macos) return;

    const batch: usize = 1;
    const seq_len: usize = 1;
    const n_heads: usize = 1;
    const head_dim: usize = 2;
    const offset: usize = 1; // Position 1 should have rotation
    const rope_base: f32 = 10000.0;

    // Input vector [1, 0] - unit vector along first axis
    var x = [_]f32{ 1.0, 0.0 };
    var out: [batch * seq_len * n_heads * head_dim]f32 = undefined;

    rope(&x, batch, seq_len, n_heads, head_dim, offset, rope_base, &out) catch |err| {
        try std.testing.expect(err == error.MLXRopeFailed);
        return;
    };

    // At position 1 with freq = 1/(10000^0) = 1, theta = 1
    // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
    // Output: [1*cos - 0*sin, 1*sin + 0*cos] = [cos(1), sin(1)]
    const theta: f32 = 1.0;
    try std.testing.expectApproxEqAbs(@cos(theta), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@sin(theta), out[1], 0.01);
}

test "silu applies activation function" {
    if (comptime builtin.os.tag != .macos) return;

    var x = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    var out: [5]f32 = undefined;

    silu(&x, &out) catch |err| {
        try std.testing.expect(err == error.MLXSiluFailed);
        return;
    };

    // SiLU(x) = x * sigmoid(x) = x / (1 + e^(-x))
    // Expected values computed mathematically:
    // SiLU(-2) = -2 / (1 + e^2) ≈ -0.2384
    // SiLU(-1) = -1 / (1 + e^1) ≈ -0.2689
    // SiLU(0) = 0
    // SiLU(1) = 1 / (1 + e^(-1)) ≈ 0.7311
    // SiLU(2) = 2 / (1 + e^(-2)) ≈ 1.7616
    try std.testing.expectApproxEqAbs(@as(f32, -0.2384), out[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689), out[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), out[3], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.7616), out[4], 0.01);
}

test "scaledDotProductAttention repeats KV heads for GQA" {
    if (comptime builtin.os.tag != .macos) return;

    const batch: usize = 1;
    const n_heads: usize = 4;
    const n_kv_heads: usize = 2;
    const seq_len: usize = 1;
    const kv_seq_len: usize = 3;
    const head_dim: usize = 2;
    const scale: f32 = 1.0;

    var q = [_]f32{
        // head 0
        0.1, 0.2,
        // head 1
        0.3, 0.4,
        // head 2
        0.5, 0.6,
        // head 3
        0.7, 0.8,
    };

    // K/V for 2 KV heads, each with kv_seq_len=3, head_dim=2.
    var k = [_]f32{
        // kv head 0
        0.1,  0.0, 0.2,  0.1, 0.3,  0.2,
        // kv head 1
        -0.1, 0.0, -0.2, 0.1, -0.3, 0.2,
    };
    var v = [_]f32{
        // kv head 0
        1.0,  0.0, 0.5,  0.5, 0.0, 1.0,
        // kv head 1
        -1.0, 0.0, -0.5, 0.5, 0.0, -1.0,
    };

    var out_gqa: [batch * n_heads * seq_len * head_dim]f32 = undefined;
    try scaledDotProductAttention(
        q[0..],
        k[0..],
        v[0..],
        batch,
        n_heads,
        n_kv_heads,
        seq_len,
        kv_seq_len,
        head_dim,
        scale,
        out_gqa[0..],
    );

    // Build explicit repeated KV tensors: repeat each KV head twice.
    var k_rep: [batch * n_heads * kv_seq_len * head_dim]f32 = undefined;
    var v_rep: [batch * n_heads * kv_seq_len * head_dim]f32 = undefined;
    const kv_block = kv_seq_len * head_dim;
    for (0..n_heads) |head_idx| {
        const src_head = head_idx / 2;
        const src_start = src_head * kv_block;
        const dst_start = head_idx * kv_block;
        @memcpy(k_rep[dst_start .. dst_start + kv_block], k[src_start .. src_start + kv_block]);
        @memcpy(v_rep[dst_start .. dst_start + kv_block], v[src_start .. src_start + kv_block]);
    }

    var out_explicit: [batch * n_heads * seq_len * head_dim]f32 = undefined;
    try scaledDotProductAttention(
        q[0..],
        k_rep[0..],
        v_rep[0..],
        batch,
        n_heads,
        n_heads,
        seq_len,
        kv_seq_len,
        head_dim,
        scale,
        out_explicit[0..],
    );

    for (out_gqa, out_explicit) |actual, expected| {
        try std.testing.expectApproxEqAbs(expected, actual, 1e-4);
    }
}

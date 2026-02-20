//! Attention operations with stride-aware implementations.
//!
//! Includes RoPE (rotary position embeddings) and SDPA (scaled dot-product attention).

const std = @import("std");
const tv = @import("tensor_view.zig");
const math = @import("math.zig");

/// Maximum sequence length for stack-allocated score buffers.
/// Sequences longer than this will use heap allocation.
const MAX_STACK_SEQ: usize = 8192;

pub const AttentionError = error{
    SequenceTooLong,
    OutOfMemory,
};

const TensorView = tv.TensorView;
const DType = tv.DType;
const MAX_NDIM = tv.MAX_NDIM;

// SIMD and math infrastructure
const simd = math.simd;
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;
const fastExp = math.fastExp;
const fastExpScalar = math.fastExpScalar;

/// Dtype conversion helpers - wrapped to remove inline calling convention
const dtype_mod = @import("../../dtype.zig");

fn fp16ToF32(x: u16) f32 {
    return dtype_mod.fp16ToF32(x);
}

fn f32ToFp16(x: f32) u16 {
    return dtype_mod.f32ToFp16(x);
}

fn bf16ToF32(x: u16) f32 {
    return dtype_mod.bf16ToF32(x);
}

fn f32ToBf16(x: f32) u16 {
    return dtype_mod.f32ToBf16(x);
}

/// Compute RoPE frequencies: 1 / (theta^(2i/d))
pub fn ropeFreqs(out: TensorView, theta: f32, offset: usize) void {
    std.debug.assert(out.dtype == .f32);
    std.debug.assert(out.ndim == 2); // [seq_len, dim]

    const out_data = @as([*]f32, @ptrCast(@alignCast(out.data)));
    const seq_len = out.shape[0];
    const dim = out.shape[1];
    const stride0 = out.strides[0];
    const stride1 = out.strides[1];
    math.ropeFillCosSinCombinedStrided(out_data, stride0, stride1, seq_len, dim, theta, offset);
}

/// Apply RoPE to query and key tensors in-place (PyTorch-compatible).
/// q, k: [batch, heads, seq, head_dim]
/// cos, sin: [batch, seq, head_dim/2] or [1, seq, head_dim/2] (broadcasts over batch/heads)
///           Also supports [seq, head_dim] format (used by C API rope_freqs).
pub fn applyRope(q: TensorView, k: TensorView, cos: TensorView, sin: TensorView) void {
    switch (q.dtype) {
        .f32 => applyRopeTyped(f32, f32Identity, f32Identity, q, k, cos, sin),
        .f16 => applyRopeTyped(u16, fp16ToF32, f32ToFp16, q, k, cos, sin),
        .bf16 => applyRopeTyped(u16, bf16ToF32, f32ToBf16, q, k, cos, sin),
        else => unreachable,
    }
}

fn f32Identity(x: f32) f32 {
    return x;
}

fn applyRopeTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    q: TensorView,
    k: TensorView,
    cos: TensorView,
    sin: TensorView,
) void {
    std.debug.assert(q.ndim == 4); // [batch, heads, seq, head_dim]

    const q_data = @as([*]T, @ptrCast(@alignCast(q.data)));
    const k_data = @as([*]T, @ptrCast(@alignCast(k.data)));

    // cos/sin can be f32, f16, or bf16 - read with same type as q/k
    const cos_data = @as([*]const T, @ptrCast(@alignCast(cos.data)));
    const sin_data = @as([*]const T, @ptrCast(@alignCast(sin.data)));

    const batch = q.shape[0];
    const q_heads = q.shape[1];
    const k_heads = k.shape[1];
    const seq_len = q.shape[2];
    const head_dim = q.shape[3];
    const half_dim = head_dim / 2;

    // Detect cos/sin layout based on ndim:
    // - Runtime format: [batch, seq, half_dim] with ndim=3
    // - C API format: [seq, head_dim] with ndim=2
    const is_batched_format = cos.ndim == 3;

    // Apply RoPE: x_rotated = x * cos - x_rotated_half * sin
    for (0..batch) |b| {
        for (0..seq_len) |s| {
            var freq_offset: usize = undefined; // Safe: both branches assign before use
            if (is_batched_format) {
                // Runtime: [batch, seq, half_dim] - broadcast batch dim if size 1
                const cos_batch = if (cos.shape[0] == 1) 0 else b;
                freq_offset = cos_batch * @as(usize, @intCast(cos.strides[0])) +
                    s * @as(usize, @intCast(cos.strides[1]));
            } else {
                // C API: [seq, head_dim] - cos in first half, sin in second half
                freq_offset = s * head_dim;
            }

            // Process query heads
            for (0..q_heads) |h| {
                const q_offset = b * q.strides[0] + h * q.strides[1] + s * q.strides[2];
                const q_ptr = q_data + q_offset;
                const q_stride = q.strides[3];
                const cos_ptr = cos_data + freq_offset;
                const sin_ptr = sin_data + freq_offset;
                const cos_stride = if (is_batched_format) cos.strides[2] else 1;
                const sin_stride = if (is_batched_format) sin.strides[2] else 1;

                math.applyRopeRotationStrided(
                    T,
                    toF32,
                    fromF32,
                    q_ptr,
                    q_stride,
                    cos_ptr,
                    cos_stride,
                    sin_ptr,
                    sin_stride,
                    half_dim,
                );
            }

            // Process key heads
            for (0..k_heads) |h| {
                const k_offset = b * k.strides[0] + h * k.strides[1] + s * k.strides[2];
                const k_ptr = k_data + k_offset;
                const k_stride = k.strides[3];
                const cos_ptr = cos_data + freq_offset;
                const sin_ptr = sin_data + freq_offset;
                const cos_stride = if (is_batched_format) cos.strides[2] else 1;
                const sin_stride = if (is_batched_format) sin.strides[2] else 1;

                math.applyRopeRotationStrided(
                    T,
                    toF32,
                    fromF32,
                    k_ptr,
                    k_stride,
                    cos_ptr,
                    cos_stride,
                    sin_ptr,
                    sin_stride,
                    half_dim,
                );
            }
        }
    }
}

/// Scaled Dot-Product Attention over 4D tensors.
/// Q: [batch, groups, query_steps, feature_width]
/// K: [batch, groups, key_steps, feature_width]
/// V: [batch, groups, key_steps, feature_width]
/// mask: optional [1, 1, query_steps, key_steps] or compatible broadcast shape
/// out: [batch, groups, query_steps, feature_width]
/// allocator: optional allocator for large sequences (>8192). If null, large sequences will error.
pub fn sdpa(
    out: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: ?TensorView,
    scale: f32,
    allocator: ?std.mem.Allocator,
) AttentionError!void {
    switch (out.dtype) {
        .f32 => try sdpaTyped(f32, f32Identity, f32Identity, out, q, k, v, mask, scale, allocator),
        .f16 => try sdpaTyped(u16, fp16ToF32, f32ToFp16, out, q, k, v, mask, scale, allocator),
        .bf16 => try sdpaTyped(u16, bf16ToF32, f32ToBf16, out, q, k, v, mask, scale, allocator),
        else => unreachable,
    }
}

fn sdpaTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    mask: ?TensorView,
    scale: f32,
    allocator: ?std.mem.Allocator,
) AttentionError!void {
    std.debug.assert(q.ndim == 4);

    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const q_data = @as([*]const T, @ptrCast(@alignCast(q.data)));
    const k_data = @as([*]const T, @ptrCast(@alignCast(k.data)));
    const v_data = @as([*]const T, @ptrCast(@alignCast(v.data)));

    const batch = q.shape[0];
    const group_count = q.shape[1];
    const query_steps = q.shape[2];
    const feature_width = q.shape[3];
    const key_steps = k.shape[2];

    // Allocate scores buffer - use stack for small sequences, heap for large
    var stack_scores: [MAX_STACK_SEQ]f32 = undefined; // Safe: only scores[0..key_steps] used, written before read
    const heap_scores: ?[]f32 = if (key_steps > MAX_STACK_SEQ) blk: {
        const alloc = allocator orelse return error.SequenceTooLong;
        break :blk alloc.alloc(f32, key_steps) catch return error.OutOfMemory;
    } else null;
    defer if (heap_scores) |hs| allocator.?.free(hs);
    const scores = if (heap_scores) |hs| hs else stack_scores[0..key_steps];

    // For each batch and head
    for (0..batch) |b| {
        for (0..group_count) |h| {
            // For each query position
            for (0..query_steps) |query_idx| {

                // Q @ K^T
                for (0..key_steps) |key_idx| {
                    var dot: f32 = 0;
                    for (0..feature_width) |d| {
                        const q_idx = b * @as(usize, @intCast(q.strides[0])) +
                            h * @as(usize, @intCast(q.strides[1])) +
                            query_idx * @as(usize, @intCast(q.strides[2])) +
                            d * @as(usize, @intCast(q.strides[3]));
                        const k_idx = b * @as(usize, @intCast(k.strides[0])) +
                            h * @as(usize, @intCast(k.strides[1])) +
                            key_idx * @as(usize, @intCast(k.strides[2])) +
                            d * @as(usize, @intCast(k.strides[3]));
                        dot += toF32(q_data[q_idx]) * toF32(k_data[k_idx]);
                    }
                    scores[key_idx] = dot * scale;

                    // Apply mask if provided
                    if (mask) |m| {
                        const m_data = @as([*]const f32, @ptrCast(@alignCast(m.data)));
                        // Broadcast mask: handle different shapes
                        const m_idx = (query_idx % m.shape[m.ndim - 2]) * @as(usize, @intCast(m.strides[m.ndim - 2])) +
                            (key_idx % m.shape[m.ndim - 1]) * @as(usize, @intCast(m.strides[m.ndim - 1]));
                        scores[key_idx] += m_data[m_idx];
                    }
                }

                // Softmax
                var max_score: f32 = -std.math.inf(f32);
                for (scores[0..key_steps]) |s| max_score = @max(max_score, s);
                math.softmaxMaskedInPlaceWithMax(scores[0..key_steps], 0, key_steps, null, false, max_score, -std.math.inf(f32) + 1.0);

                // Weighted sum of values
                for (0..feature_width) |d| {
                    var acc: f32 = 0;
                    for (0..key_steps) |key_idx| {
                        const v_idx = b * @as(usize, @intCast(v.strides[0])) +
                            h * @as(usize, @intCast(v.strides[1])) +
                            key_idx * @as(usize, @intCast(v.strides[2])) +
                            d * @as(usize, @intCast(v.strides[3]));
                        acc += scores[key_idx] * toF32(v_data[v_idx]);
                    }
                    const out_idx = b * @as(usize, @intCast(out.strides[0])) +
                        h * @as(usize, @intCast(out.strides[1])) +
                        query_idx * @as(usize, @intCast(out.strides[2])) +
                        d * @as(usize, @intCast(out.strides[3]));
                    out_data[out_idx] = fromF32(acc);
                }
            }
        }
    }
}

/// Scaled Dot-Product Attention with causal mask (optimized path)
/// This version doesn't require an explicit mask tensor - causal masking is applied implicitly.
/// Q: [batch, groups, query_steps, feature_width]
/// K: [batch, groups, key_steps, feature_width]
/// V: [batch, groups, key_steps, feature_width]
/// out: [batch, groups, query_steps, feature_width]
/// causal_mask_shift: offset applied to query position in causal masking.
/// allocator: optional allocator for large sequences (>8192). If null, large sequences will error.
pub fn sdpaCausal(
    out: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    scale: f32,
    causal_mask_shift: usize,
    allocator: ?std.mem.Allocator,
) AttentionError!void {
    switch (out.dtype) {
        .f32 => try sdpaCausalTyped(f32, f32Identity, f32Identity, out, q, k, v, scale, causal_mask_shift, allocator),
        .f16 => try sdpaCausalTyped(u16, fp16ToF32, f32ToFp16, out, q, k, v, scale, causal_mask_shift, allocator),
        .bf16 => try sdpaCausalTyped(u16, bf16ToF32, f32ToBf16, out, q, k, v, scale, causal_mask_shift, allocator),
        else => unreachable,
    }
}

fn sdpaCausalTyped(
    comptime T: type,
    comptime toF32: fn (T) f32,
    comptime fromF32: fn (f32) T,
    out: TensorView,
    q: TensorView,
    k: TensorView,
    v: TensorView,
    scale: f32,
    causal_mask_shift: usize,
    allocator: ?std.mem.Allocator,
) AttentionError!void {
    std.debug.assert(q.ndim == 4);

    const out_data = @as([*]T, @ptrCast(@alignCast(out.data)));
    const q_data = @as([*]const T, @ptrCast(@alignCast(q.data)));
    const k_data = @as([*]const T, @ptrCast(@alignCast(k.data)));
    const v_data = @as([*]const T, @ptrCast(@alignCast(v.data)));

    const batch = q.shape[0];
    const num_heads = q.shape[1];
    const seq_q = q.shape[2];
    const head_dim = q.shape[3];
    const seq_k = k.shape[2];

    // Allocate scores buffer - use stack for small sequences, heap for large
    var stack_scores: [MAX_STACK_SEQ]f32 = undefined; // Safe: only scores[0..seq_k] used, written before read
    const heap_scores: ?[]f32 = if (seq_k > MAX_STACK_SEQ) blk: {
        const alloc = allocator orelse return error.SequenceTooLong;
        break :blk alloc.alloc(f32, seq_k) catch return error.OutOfMemory;
    } else null;
    defer if (heap_scores) |hs| allocator.?.free(hs);
    const scores = if (heap_scores) |hs| hs else stack_scores[0..seq_k];

    const neg_inf = -std.math.inf(f32);

    for (0..batch) |b| {
        for (0..num_heads) |h| {
            for (0..seq_q) |sq| {

                // The query position in the full sequence
                const q_pos = causal_mask_shift + sq;

                // Q @ K^T with causal masking
                for (0..seq_k) |sk| {
                    // Causal: can only attend to positions <= current position
                    if (sk > q_pos) {
                        scores[sk] = neg_inf;
                        continue;
                    }

                    var dot: f32 = 0;
                    for (0..head_dim) |d| {
                        const q_idx = b * @as(usize, @intCast(q.strides[0])) +
                            h * @as(usize, @intCast(q.strides[1])) +
                            sq * @as(usize, @intCast(q.strides[2])) +
                            d * @as(usize, @intCast(q.strides[3]));
                        const k_idx = b * @as(usize, @intCast(k.strides[0])) +
                            h * @as(usize, @intCast(k.strides[1])) +
                            sk * @as(usize, @intCast(k.strides[2])) +
                            d * @as(usize, @intCast(k.strides[3]));
                        dot += toF32(q_data[q_idx]) * toF32(k_data[k_idx]);
                    }
                    scores[sk] = dot * scale;
                }

                // Softmax
                var max_score: f32 = neg_inf;
                for (scores[0..seq_k]) |s| max_score = @max(max_score, s);
                math.softmaxMaskedInPlaceWithMax(scores[0..seq_k], 0, seq_k, null, false, max_score, neg_inf + 1.0);

                // Weighted sum of values
                for (0..head_dim) |d| {
                    var acc: f32 = 0;
                    for (0..seq_k) |sk| {
                        const v_idx = b * @as(usize, @intCast(v.strides[0])) +
                            h * @as(usize, @intCast(v.strides[1])) +
                            sk * @as(usize, @intCast(v.strides[2])) +
                            d * @as(usize, @intCast(v.strides[3]));
                        acc += scores[sk] * toF32(v_data[v_idx]);
                    }
                    const out_idx = b * @as(usize, @intCast(out.strides[0])) +
                        h * @as(usize, @intCast(out.strides[1])) +
                        sq * @as(usize, @intCast(out.strides[2])) +
                        d * @as(usize, @intCast(out.strides[3]));
                    out_data[out_idx] = fromF32(acc);
                }
            }
        }
    }
}

/// SDPA with cached K/V and optional features.
/// Computes attention where K/V come from a pre-filled cache.
///
/// Parameters:
/// - out_data: output buffer [n_heads * seq_q * head_dim]
/// - out_strides: strides for output [batch, heads, seq, dim]
/// - q_data: query data [batch * n_heads * seq_q * head_dim]
/// - q_strides: strides for query
/// - k_cache: cached keys [max_seq * n_kv_heads * head_dim] for this layer
/// - v_cache: cached values [max_seq * n_kv_heads * head_dim] for this layer
/// - n_heads, n_kv_heads: number of query and kv heads
/// - seq_q: number of query positions
/// - cached_seq: number of valid positions in cache
/// - head_dim: dimension per head
/// - kv_offset: offset for causal masking (current position in full sequence)
/// - scale: attention scale (typically 1/sqrt(head_dim))
/// - sinks: optional per-head sink logits (null if not used)
/// - sliding_window: 0 = disabled, >0 = only attend to last N positions
/// - allocator: optional allocator for large sequences (>8192). If null, large sequences will error.
pub fn sdpaCached(
    out_data: [*]f32,
    out_strides: [4]usize,
    q_data: [*]const f32,
    q_strides: [4]usize,
    k_cache: []const f32,
    v_cache: []const f32,
    n_heads: usize,
    n_kv_heads: usize,
    seq_q: usize,
    cached_seq: usize,
    head_dim: usize,
    kv_offset: usize,
    scale: f32,
    sinks: ?[]const f32,
    sliding_window: usize,
    allocator: ?std.mem.Allocator,
) AttentionError!void {
    const neg_inf = -std.math.inf(f32);
    const heads_per_kv = n_heads / n_kv_heads;
    const kv_size = n_kv_heads * head_dim;

    // Allocate scores buffer - use stack for small sequences, heap for large
    var stack_scores: [MAX_STACK_SEQ]f32 = undefined; // Safe: only scores[0..cached_seq] used, written before read
    const heap_scores: ?[]f32 = if (cached_seq > MAX_STACK_SEQ) blk: {
        const alloc = allocator orelse return error.SequenceTooLong;
        break :blk alloc.alloc(f32, cached_seq) catch return error.OutOfMemory;
    } else null;
    defer if (heap_scores) |hs| allocator.?.free(hs);
    const scores = if (heap_scores) |hs| hs else stack_scores[0..cached_seq];

    for (0..n_heads) |qh| {
        const kv_head = qh / heads_per_kv;
        const sink_logit: ?f32 = if (sinks) |s| s[qh] else null;

        for (0..seq_q) |sq| {
            const q_pos = kv_offset + sq;

            // Determine attention window
            const window_start: usize = if (sliding_window > 0 and q_pos >= sliding_window)
                q_pos - sliding_window + 1
            else
                0;

            var max_score: f32 = neg_inf;

            // Q @ K^T with causal + sliding window masking
            for (0..cached_seq) |sk| {
                if (sk < window_start or sk > q_pos) {
                    scores[sk] = neg_inf;
                    continue;
                }

                const k_cache_idx = sk * kv_size + kv_head * head_dim;
                var dot: f32 = 0;
                for (0..head_dim) |d| {
                    const q_idx = qh * q_strides[1] + sq * q_strides[2] + d * q_strides[3];
                    dot += q_data[q_idx] * k_cache[k_cache_idx + d];
                }
                scores[sk] = dot * scale;
                max_score = @max(max_score, scores[sk]);
            }

            // Include sink in max calculation
            if (sink_logit) |sl| {
                max_score = @max(max_score, sl);
            }

            // Softmax with optional sink
            math.softmaxMaskedInPlaceWithMax(scores[0..cached_seq], 0, cached_seq, sink_logit, false, max_score, neg_inf + 1.0);

            // Weighted sum of values
            for (0..head_dim) |d| {
                var acc: f32 = 0;
                for (0..cached_seq) |sk| {
                    const v_cache_idx = sk * kv_size + kv_head * head_dim;
                    acc += scores[sk] * v_cache[v_cache_idx + d];
                }
                const out_idx = qh * out_strides[1] + sq * out_strides[2] + d * out_strides[3];
                out_data[out_idx] = acc;
            }
        }
    }
}

/// Parameters for KV cache attention, pre-validated.
/// Use validateKVCacheParams() to construct.
pub const KVCacheAttentionParams = struct {
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    layer_offset: usize,
    kv_size: usize,
    scale: f32,
    q_strides: [4]usize,
    k_strides: [4]usize,
    out_strides: [4]usize,
};

/// Validation error for KV cache parameters.
pub const KVCacheValidationError = error{
    LayerIndexOutOfBounds,
    InvalidTensorDims,
    CacheShapeMismatch,
    BatchMismatch,
    SeqLenMismatch,
    KVShapeMismatch,
    UnsupportedDType,
};

/// Convert i64 strides to usize array (4D).
pub fn stridesToUsize4D(strides: *const [8]i64) [4]usize {
    return .{
        @intCast(strides[0]),
        @intCast(strides[1]),
        @intCast(strides[2]),
        @intCast(strides[3]),
    };
}

/// Update KV cache with new K/V values (stride-aware copy)
/// Used by attention_with_kv_cache to update the cache before computing attention.
pub fn updateKVCache(
    k_cache: []f32,
    v_cache: []f32,
    k_data: [*]const f32,
    v_data: [*]const f32,
    k_strides: [4]usize,
    layer_offset: usize,
    seq_pos: usize,
    max_seq_len: usize,
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
) void {
    const kv_size = n_kv_heads * head_dim;

    for (0..seq_len) |s| {
        const cache_pos = (seq_pos + s) % max_seq_len;
        const cache_idx = layer_offset + cache_pos * kv_size;

        for (0..n_kv_heads) |h| {
            for (0..head_dim) |d| {
                const in_idx = h * k_strides[1] + s * k_strides[2] + d * k_strides[3];
                k_cache[cache_idx + h * head_dim + d] = k_data[in_idx];
                v_cache[cache_idx + h * head_dim + d] = v_data[in_idx];
            }
        }
    }
}

test "ropeFreqs basic" {
    var data = [_]f32{0} ** 8;
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 4 }, .f32);

    ropeFreqs(out, 10000.0, 0);

    // First position should have cos(0)=1, sin(0)=0 for first freq
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5); // cos at pos 0, freq 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-5); // sin at pos 0, freq 0
}

// ============================================================================
// Additional Unit Tests
// ============================================================================

test "ropeFreqs - verify cos/sin values for known positions" {
    // Test that RoPE frequencies are computed correctly for known positions
    // Formula: freq = 1 / (theta^(2i/d)) where i is dim index
    // angle = pos * freq, then cos(angle) and sin(angle)

    var data = [_]f32{0} ** 12; // 3 positions, 4 dims (2 freqs)
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 3, 4 }, .f32);

    const theta: f32 = 10000.0;
    ropeFreqs(out, theta, 0);

    // Position 0: all angles should be 0
    // cos(0) = 1, sin(0) = 0 for all frequencies
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5); // cos, freq 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[1], 1e-5); // cos, freq 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-5); // sin, freq 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[3], 1e-5); // sin, freq 1

    // Position 1: should have non-zero angles
    const freq0 = 1.0 / std.math.pow(f32, theta, 0.0 / 4.0); // = 1.0
    const freq1 = 1.0 / std.math.pow(f32, theta, 2.0 / 4.0); // = 1/100
    const angle0_pos1 = 1.0 * freq0;
    const angle1_pos1 = 1.0 * freq1;

    try std.testing.expectApproxEqAbs(@cos(angle0_pos1), data[4], 1e-5);
    try std.testing.expectApproxEqAbs(@cos(angle1_pos1), data[5], 1e-5);
    try std.testing.expectApproxEqAbs(@sin(angle0_pos1), data[6], 1e-5);
    try std.testing.expectApproxEqAbs(@sin(angle1_pos1), data[7], 1e-5);
}

test "ropeFreqs - with offset" {
    // Test that offset correctly shifts the position indices
    var data = [_]f32{0} ** 8; // 2 positions, 4 dims
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 2, 4 }, .f32);

    const theta: f32 = 10000.0;
    const offset: usize = 5;
    ropeFreqs(out, theta, offset);

    // With offset=5, first position is actually position 5
    const freq0 = 1.0 / std.math.pow(f32, theta, 0.0 / 4.0);
    const angle0_pos5 = 5.0 * freq0;

    try std.testing.expectApproxEqAbs(@cos(angle0_pos5), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@sin(angle0_pos5), data[2], 1e-5);
}

test "applyRope - verify rotation is applied correctly to Q/K vectors" {
    // Test that RoPE rotation correctly transforms Q and K tensors
    // Rotation formula: x_rotated = x * cos - x_half_rotated * sin
    //                   x_half_rotated = x_half * cos + x * sin

    const allocator = std.testing.allocator;

    // Create Q and K: [batch=1, heads=1, seq=1, head_dim=4]
    var q_data = [_]f32{ 1.0, 0.0, 2.0, 0.0 }; // Simple pattern for verification
    var k_data = [_]f32{ 0.0, 1.0, 0.0, 3.0 };
    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, 4 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 1, 4 }, .f32);

    // Create cos/sin: [batch=1, seq=1, half_dim=2]
    var cos_data = [_]f32{ 0.8, 0.6 }; // cos values for 2 frequencies
    var sin_data = [_]f32{ 0.6, 0.8 }; // sin values for 2 frequencies
    const cos = TensorView.initContiguous(@ptrCast(&cos_data), &.{ 1, 1, 2 }, .f32);
    const sin = TensorView.initContiguous(@ptrCast(&sin_data), &.{ 1, 1, 2 }, .f32);

    // Save original values
    const q0_orig = q_data[0];
    const q2_orig = q_data[2]; // half dim away

    applyRope(q, k, cos, sin);

    // Verify rotation was applied to Q
    // For dim 0: q[0] * cos[0] - q[2] * sin[0]
    const expected_q0 = q0_orig * cos_data[0] - q2_orig * sin_data[0];
    try std.testing.expectApproxEqAbs(expected_q0, q_data[0], 1e-5);

    // For dim 2: q[2] * cos[0] + q[0] * sin[0]
    const expected_q2 = q2_orig * cos_data[0] + q0_orig * sin_data[0];
    try std.testing.expectApproxEqAbs(expected_q2, q_data[2], 1e-5);

    _ = allocator;
}

test "applyRope - identity rotation with cos=1, sin=0" {
    // Test RoPE with identity rotation (cos=1, sin=0 should preserve values)
    const allocator = std.testing.allocator;

    var q_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var k_data = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, 4 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 1, 4 }, .f32);

    // Batched format: [batch=1, seq=1, half_dim=2]
    // cos=1, sin=0 means no rotation
    var cos_data = [_]f32{ 1.0, 1.0 };
    var sin_data = [_]f32{ 0.0, 0.0 };
    const cos = TensorView.initContiguous(@ptrCast(&cos_data), &.{ 1, 1, 2 }, .f32);
    const sin = TensorView.initContiguous(@ptrCast(&sin_data), &.{ 1, 1, 2 }, .f32);

    const q0_orig = q_data[0];
    const q1_orig = q_data[1];

    applyRope(q, k, cos, sin);

    // With cos=1, sin=0: rotation should preserve first half values
    // Formula: q_rotated[i] = q[i]*cos - q[i+half]*sin = q[i]*1 - q[i+half]*0 = q[i]
    try std.testing.expectApproxEqAbs(q0_orig, q_data[0], 1e-5);
    try std.testing.expectApproxEqAbs(q1_orig, q_data[1], 1e-5);

    _ = allocator;
}

test "sdpaCausal - causal mask" {
    // Test that causal masking prevents attending to future tokens
    const allocator = std.testing.allocator;

    // Q, K, V: [batch=1, heads=1, seq=3, head_dim=2]
    var q_data = [_]f32{ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 };
    var k_data = [_]f32{ 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 };
    var v_data = [_]f32{ 1.0, 0.0, 2.0, 0.0, 3.0, 0.0 }; // Distinct values
    var out_data = [_]f32{0} ** 6;

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 3, 2 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 3, 2 }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, 1, 3, 2 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 3, 2 }, .f32);

    const scale = 1.0 / @sqrt(2.0);
    try sdpaCausal(out, q, k, v, scale, 0, allocator);

    // Position 0 can only attend to position 0, should output v[0]
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-4);

    // Position 1 can attend to [0,1], should be weighted average
    // Since Q and K are identical, scores should be equal after softmax
    // Expected: (v[0] + v[1]) / 2 = (1.0 + 2.0) / 2 = 1.5
    const out_pos1 = out_data[2]; // [0, 0, 1, 0]
    try std.testing.expect(out_pos1 > 1.0 and out_pos1 < 2.0);
}

test "sdpa - explicit padding mask" {
    // Test that explicit mask values are applied correctly
    const allocator = std.testing.allocator;

    // Q, K, V: [batch=1, heads=1, seq=2, head_dim=2]
    var q_data = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    var k_data = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    var v_data = [_]f32{ 1.0, 0.0, 2.0, 0.0 };
    var out_data = [_]f32{0} ** 4;

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 2, 2 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 2, 2 }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, 1, 2, 2 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 2, 2 }, .f32);

    // Mask: [1, 1, 2, 2] - mask out position 1 with large negative value
    var mask_data = [_]f32{ 0.0, -1e9, 0.0, 0.0 };
    const mask = TensorView.initContiguous(@ptrCast(&mask_data), &.{ 1, 1, 2, 2 }, .f32);

    const scale = 1.0 / @sqrt(2.0);
    try sdpa(out, q, k, v, mask, scale, allocator);

    // Position 0 should only attend to position 0 (position 1 is masked)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 1e-4);

    // Position 1 can attend to both (no mask)
    const out_pos1 = out_data[2];
    try std.testing.expect(out_pos1 > 1.0 and out_pos1 < 2.0);
}

test "sdpa - single token sequence" {
    // Test attention with single token (decode step)
    const allocator = std.testing.allocator;

    // Q: [batch=1, heads=1, seq=1, head_dim=4]
    var q_data = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    // K, V: [batch=1, heads=1, seq=3, head_dim=4] (cache)
    var k_data = [_]f32{ 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };
    var v_data = [_]f32{ 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0 };
    var out_data = [_]f32{0} ** 4;

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, 4 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 3, 4 }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, 1, 3, 4 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 1, 4 }, .f32);

    const scale = 1.0 / @sqrt(4.0);
    try sdpa(out, q, k, v, null, scale, allocator);

    // Should produce a weighted average of v values
    try std.testing.expect(out_data[0] >= 1.0 and out_data[0] <= 3.0);
}

test "sdpa - long sequence" {
    // Test attention with longer sequences to verify correctness at scale
    const allocator = std.testing.allocator;

    const seq_len: usize = 16;
    const head_dim: usize = 8;

    var q_data = [_]f32{0} ** (seq_len * head_dim);
    var k_data = [_]f32{0} ** (seq_len * head_dim);
    var v_data = [_]f32{0} ** (seq_len * head_dim);
    var out_data = [_]f32{0} ** (seq_len * head_dim);

    // Initialize with simple patterns
    for (0..seq_len) |i| {
        for (0..head_dim) |d| {
            const idx = i * head_dim + d;
            q_data[idx] = @as(f32, @floatFromInt(i + 1));
            k_data[idx] = @as(f32, @floatFromInt(i + 1));
            v_data[idx] = @as(f32, @floatFromInt(i + 1));
        }
    }

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, seq_len, head_dim }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, seq_len, head_dim }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, 1, seq_len, head_dim }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, seq_len, head_dim }, .f32);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    try sdpaCausal(out, q, k, v, scale, 0, allocator);

    // Verify output is non-zero and reasonable
    try std.testing.expect(out_data[0] > 0.0);
    try std.testing.expect(out_data[seq_len * head_dim - 1] > 0.0);
}

test "sdpa - verify head dimension handling" {
    // Test that multi-head attention correctly isolates heads
    const allocator = std.testing.allocator;

    const n_heads: usize = 2;
    const seq_len: usize = 2;
    const head_dim: usize = 4;

    var q_data = [_]f32{0} ** (n_heads * seq_len * head_dim);
    var k_data = [_]f32{0} ** (n_heads * seq_len * head_dim);
    var v_data = [_]f32{0} ** (n_heads * seq_len * head_dim);
    var out_data = [_]f32{0} ** (n_heads * seq_len * head_dim);

    // Initialize head 0 with ones, head 1 with twos
    for (0..seq_len) |s| {
        for (0..head_dim) |d| {
            // Head 0
            const idx0 = 0 * seq_len * head_dim + s * head_dim + d;
            q_data[idx0] = 1.0;
            k_data[idx0] = 1.0;
            v_data[idx0] = 1.0;

            // Head 1
            const idx1 = 1 * seq_len * head_dim + s * head_dim + d;
            q_data[idx1] = 2.0;
            k_data[idx1] = 2.0;
            v_data[idx1] = 2.0;
        }
    }

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, n_heads, seq_len, head_dim }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, n_heads, seq_len, head_dim }, .f32);

    const scale = 1.0 / @sqrt(4.0);
    try sdpa(out, q, k, v, null, scale, allocator);

    // Head 0 output should be close to 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 0.1);

    // Head 1 output should be close to 2.0
    const head1_offset = seq_len * head_dim;
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out_data[head1_offset], 0.2);
}

test "sdpa - grouped query attention (GQA)" {
    // Test attention with different number of Q and KV heads (GQA)
    const allocator = std.testing.allocator;

    const n_q_heads: usize = 4;
    const n_kv_heads: usize = 2;
    const seq_len: usize = 2;
    const head_dim: usize = 4;

    var q_data = [_]f32{0} ** (n_q_heads * seq_len * head_dim);
    var k_data = [_]f32{0} ** (n_kv_heads * seq_len * head_dim);
    var v_data = [_]f32{0} ** (n_kv_heads * seq_len * head_dim);
    var out_data = [_]f32{0} ** (n_q_heads * seq_len * head_dim);

    // Initialize with ones
    for (0..n_q_heads * seq_len * head_dim) |i| q_data[i] = 1.0;
    for (0..n_kv_heads * seq_len * head_dim) |i| {
        k_data[i] = 1.0;
        v_data[i] = 1.0;
    }

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, n_q_heads, seq_len, head_dim }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, n_kv_heads, seq_len, head_dim }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, n_kv_heads, seq_len, head_dim }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, n_q_heads, seq_len, head_dim }, .f32);

    const scale = 1.0 / @sqrt(4.0);
    try sdpa(out, q, k, v, null, scale, allocator);

    // All outputs should be approximately 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[0], 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_data[seq_len * head_dim], 0.1);
}

test "sdpa - softmax with large values" {
    // Test that softmax remains stable with large attention scores
    const allocator = std.testing.allocator;

    // Q, K designed to produce large dot products
    var q_data = [_]f32{ 10.0, 10.0, 10.0, 10.0 };
    var k_data = [_]f32{ 10.0, 10.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1 };
    var v_data = [_]f32{ 1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0 };
    var out_data = [_]f32{0} ** 4;

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, 4 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 2, 4 }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, 1, 2, 4 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 1, 4 }, .f32);

    const scale = 1.0 / @sqrt(4.0);
    try sdpa(out, q, k, v, null, scale, allocator);

    // Output should be valid (not NaN or Inf)
    try std.testing.expect(std.math.isFinite(out_data[0]));
    try std.testing.expect(std.math.isFinite(out_data[1]));

    // Should heavily weight the first value (much larger dot product)
    try std.testing.expect(out_data[0] < 1.5); // Should be close to v[0]=1.0
}

test "sdpa - very small attention scores" {
    // Test handling of very small (near-zero) attention scores
    const allocator = std.testing.allocator;

    // Q, K designed to produce very small dot products
    var q_data = [_]f32{ 0.001, 0.001, 0.001, 0.001 };
    var k_data = [_]f32{ 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
    var v_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    var out_data = [_]f32{0} ** 4;

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, 4 }, .f32);
    const k = TensorView.initContiguous(@ptrCast(&k_data), &.{ 1, 1, 2, 4 }, .f32);
    const v = TensorView.initContiguous(@ptrCast(&v_data), &.{ 1, 1, 2, 4 }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 1, 4 }, .f32);

    const scale = 1.0 / @sqrt(4.0);
    try sdpa(out, q, k, v, null, scale, allocator);

    // Output should be valid and a reasonable average of values
    try std.testing.expect(std.math.isFinite(out_data[0]));
    try std.testing.expect(out_data[0] >= 1.0 and out_data[0] <= 5.0);
    try std.testing.expect(out_data[1] >= 2.0 and out_data[1] <= 6.0);
}

test "applyRope - mixed precision" {
    // Test RoPE with fp16 and bf16 to ensure conversion stability
    const allocator = std.testing.allocator;

    // Test with f16
    var q_f16_data = [_]u16{0} ** 4;
    var k_f16_data = [_]u16{0} ** 4;

    // Initialize with fp16 values
    for (0..4) |i| {
        q_f16_data[i] = f32ToFp16(@as(f32, @floatFromInt(i + 1)));
        k_f16_data[i] = f32ToFp16(@as(f32, @floatFromInt(i + 1)));
    }

    const q_f16 = TensorView.initContiguous(@ptrCast(&q_f16_data), &.{ 1, 1, 1, 4 }, .f16);
    const k_f16 = TensorView.initContiguous(@ptrCast(&k_f16_data), &.{ 1, 1, 1, 4 }, .f16);

    var cos_data = [_]u16{ f32ToFp16(1.0), f32ToFp16(1.0) };
    var sin_data = [_]u16{ f32ToFp16(0.0), f32ToFp16(0.0) };
    const cos_f16 = TensorView.initContiguous(@ptrCast(&cos_data), &.{ 1, 1, 2 }, .f16);
    const sin_f16 = TensorView.initContiguous(@ptrCast(&sin_data), &.{ 1, 1, 2 }, .f16);

    // Apply RoPE - should not crash or produce invalid values
    applyRope(q_f16, k_f16, cos_f16, sin_f16);

    // Verify output is still valid
    try std.testing.expect(fp16ToF32(q_f16_data[0]) > 0.0);
    try std.testing.expect(std.math.isFinite(fp16ToF32(q_f16_data[0])));

    _ = allocator;
}

test "updateKVCache - verify cache storage" {
    // Test that KV cache is correctly updated with new values
    const n_kv_heads: usize = 2;
    const head_dim: usize = 4;
    const max_seq: usize = 10;
    const seq_len: usize = 3;

    // Allocate cache
    var k_cache = [_]f32{0} ** (max_seq * n_kv_heads * head_dim);
    var v_cache = [_]f32{0} ** (max_seq * n_kv_heads * head_dim);

    // New K, V data: [1, n_kv_heads, seq_len, head_dim]
    var k_data = [_]f32{0} ** (n_kv_heads * seq_len * head_dim);
    var v_data = [_]f32{0} ** (n_kv_heads * seq_len * head_dim);

    // Fill with test pattern
    for (0..n_kv_heads * seq_len * head_dim) |i| {
        k_data[i] = @as(f32, @floatFromInt(i + 1));
        v_data[i] = @as(f32, @floatFromInt(i + 100));
    }

    const k_strides = [4]usize{ n_kv_heads * seq_len * head_dim, seq_len * head_dim, head_dim, 1 };

    updateKVCache(
        &k_cache,
        &v_cache,
        @ptrCast(&k_data),
        @ptrCast(&v_data),
        k_strides,
        0, // layer_offset
        0, // seq_pos
        max_seq,
        seq_len,
        n_kv_heads,
        head_dim,
    );

    // Verify first position was written correctly
    const kv_size = n_kv_heads * head_dim;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), k_cache[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), v_cache[0], 1e-6);

    // Verify second position
    try std.testing.expectApproxEqAbs(k_data[head_dim], k_cache[kv_size], 1e-6);
    try std.testing.expectApproxEqAbs(v_data[head_dim], v_cache[kv_size], 1e-6);
}

test "sdpaCached - basic cache attention" {
    // Test attention using cached K/V values
    const allocator = std.testing.allocator;

    const n_heads: usize = 2;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 4;
    const cached_seq: usize = 3;
    const seq_q: usize = 1;

    // Query: [n_heads, seq_q, head_dim]
    var q_data = [_]f32{0} ** (n_heads * seq_q * head_dim);
    for (0..n_heads * seq_q * head_dim) |i| q_data[i] = 1.0;

    // Cache: [cached_seq * n_kv_heads * head_dim]
    var k_cache = [_]f32{0} ** (cached_seq * n_kv_heads * head_dim);
    var v_cache = [_]f32{0} ** (cached_seq * n_kv_heads * head_dim);
    for (0..cached_seq * n_kv_heads * head_dim) |i| {
        k_cache[i] = 1.0;
        v_cache[i] = @as(f32, @floatFromInt(i + 1));
    }

    var out_data = [_]f32{0} ** (n_heads * seq_q * head_dim);

    const out_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };
    const q_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };

    try sdpaCached(
        @ptrCast(&out_data),
        out_strides,
        @ptrCast(&q_data),
        q_strides,
        &k_cache,
        &v_cache,
        n_heads,
        n_kv_heads,
        seq_q,
        cached_seq,
        head_dim,
        cached_seq - 1, // kv_offset (current position)
        1.0 / @sqrt(4.0),
        null, // no sinks
        0, // no sliding window
        allocator,
    );

    // Output should be valid and a weighted average of cache values
    try std.testing.expect(std.math.isFinite(out_data[0]));
    try std.testing.expect(out_data[0] > 0.0);
}

test "sdpaCached - with sliding window" {
    // Test attention with sliding window (only attend to recent tokens)
    const allocator = std.testing.allocator;

    const n_heads: usize = 1;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const cached_seq: usize = 10;
    const seq_q: usize = 1;
    const sliding_window: usize = 3; // Only attend to last 3 tokens

    var q_data = [_]f32{0} ** (n_heads * seq_q * head_dim);
    for (0..n_heads * seq_q * head_dim) |i| q_data[i] = 1.0;

    var k_cache = [_]f32{0} ** (cached_seq * n_kv_heads * head_dim);
    var v_cache = [_]f32{0} ** (cached_seq * n_kv_heads * head_dim);

    // Make older values large, recent values small
    for (0..cached_seq) |pos| {
        const base = pos * n_kv_heads * head_dim;
        for (0..head_dim) |d| {
            k_cache[base + d] = if (pos < 7) 100.0 else 1.0;
            v_cache[base + d] = if (pos < 7) 999.0 else @as(f32, @floatFromInt(pos));
        }
    }

    var out_data = [_]f32{0} ** (n_heads * seq_q * head_dim);

    const out_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };
    const q_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };

    try sdpaCached(
        @ptrCast(&out_data),
        out_strides,
        @ptrCast(&q_data),
        q_strides,
        &k_cache,
        &v_cache,
        n_heads,
        n_kv_heads,
        seq_q,
        cached_seq,
        head_dim,
        cached_seq - 1,
        1.0 / @sqrt(4.0),
        null,
        sliding_window,
        allocator,
    );

    // Output should be based on recent tokens (7,8,9), not old ones
    // Should be less than 999.0 (the old value) and closer to 7-9
    try std.testing.expect(out_data[0] < 50.0);
}

test "sdpa - error sequence too long without allocator" {
    // Test that SDPA returns SequenceTooLong error when seq > MAX_STACK_SEQ and no allocator
    const seq_len = MAX_STACK_SEQ + 1;
    const head_dim: usize = 4;

    var q_data = [_]f32{1.0} ** head_dim;
    const k_data = std.testing.allocator.alloc(f32, seq_len * head_dim) catch unreachable;
    defer std.testing.allocator.free(k_data);
    const v_data = std.testing.allocator.alloc(f32, seq_len * head_dim) catch unreachable;
    defer std.testing.allocator.free(v_data);
    var out_data = [_]f32{0} ** head_dim;

    @memset(k_data, 1.0);
    @memset(v_data, 1.0);

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, head_dim }, .f32);
    const k = TensorView.initContiguous(@ptrCast(k_data.ptr), &.{ 1, 1, seq_len, head_dim }, .f32);
    const v = TensorView.initContiguous(@ptrCast(v_data.ptr), &.{ 1, 1, seq_len, head_dim }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 1, head_dim }, .f32);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const result = sdpa(out, q, k, v, null, scale, null);

    try std.testing.expectError(error.SequenceTooLong, result);
}

test "sdpa - sequence too long with allocator succeeds" {
    // Test that SDPA succeeds with allocator even when seq > MAX_STACK_SEQ
    const allocator = std.testing.allocator;
    const seq_len = MAX_STACK_SEQ + 100;
    const head_dim: usize = 4;

    var q_data = [_]f32{0} ** (1 * 1 * 1 * head_dim);
    const k_data = try allocator.alloc(f32, seq_len * head_dim);
    defer allocator.free(k_data);
    const v_data = try allocator.alloc(f32, seq_len * head_dim);
    defer allocator.free(v_data);
    var out_data = [_]f32{0} ** (1 * 1 * 1 * head_dim);

    @memset(&q_data, 1.0);
    @memset(k_data, 1.0);
    @memset(v_data, 1.0);

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, head_dim }, .f32);
    const k = TensorView.initContiguous(@ptrCast(k_data.ptr), &.{ 1, 1, seq_len, head_dim }, .f32);
    const v = TensorView.initContiguous(@ptrCast(v_data.ptr), &.{ 1, 1, seq_len, head_dim }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 1, head_dim }, .f32);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    try sdpa(out, q, k, v, null, scale, allocator);

    // Should succeed and produce valid output
    try std.testing.expect(std.math.isFinite(out_data[0]));
}

test "sdpaCausal - error sequence too long without allocator" {
    // Test that sdpaCausal returns SequenceTooLong error when seq > MAX_STACK_SEQ and no allocator
    const seq_len = MAX_STACK_SEQ + 1;
    const head_dim: usize = 4;

    var q_data = [_]f32{0} ** (1 * 1 * 1 * head_dim);
    const k_data = std.testing.allocator.alloc(f32, seq_len * head_dim) catch unreachable;
    defer std.testing.allocator.free(k_data);
    const v_data = std.testing.allocator.alloc(f32, seq_len * head_dim) catch unreachable;
    defer std.testing.allocator.free(v_data);
    var out_data = [_]f32{0} ** (1 * 1 * 1 * head_dim);

    @memset(&q_data, 1.0);
    @memset(k_data, 1.0);
    @memset(v_data, 1.0);

    const q = TensorView.initContiguous(@ptrCast(&q_data), &.{ 1, 1, 1, head_dim }, .f32);
    const k = TensorView.initContiguous(@ptrCast(k_data.ptr), &.{ 1, 1, seq_len, head_dim }, .f32);
    const v = TensorView.initContiguous(@ptrCast(v_data.ptr), &.{ 1, 1, seq_len, head_dim }, .f32);
    const out = TensorView.initContiguous(@ptrCast(&out_data), &.{ 1, 1, 1, head_dim }, .f32);

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    const result = sdpaCausal(out, q, k, v, scale, 0, null);

    try std.testing.expectError(error.SequenceTooLong, result);
}

test "sdpaCached - error sequence too long without allocator" {
    // Test that sdpaCached returns SequenceTooLong error when seq > MAX_STACK_SEQ and no allocator
    const cached_seq = MAX_STACK_SEQ + 1;
    const n_heads: usize = 1;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const seq_q: usize = 1;

    var q_data = [_]f32{0} ** (n_heads * seq_q * head_dim);
    const k_cache = std.testing.allocator.alloc(f32, cached_seq * n_kv_heads * head_dim) catch unreachable;
    defer std.testing.allocator.free(k_cache);
    const v_cache = std.testing.allocator.alloc(f32, cached_seq * n_kv_heads * head_dim) catch unreachable;
    defer std.testing.allocator.free(v_cache);
    var out_data = [_]f32{0} ** (n_heads * seq_q * head_dim);

    @memset(&q_data, 1.0);
    @memset(k_cache, 1.0);
    @memset(v_cache, 1.0);

    const out_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };
    const q_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };

    const result = sdpaCached(
        @ptrCast(&out_data),
        out_strides,
        @ptrCast(&q_data),
        q_strides,
        k_cache,
        v_cache,
        n_heads,
        n_kv_heads,
        seq_q,
        cached_seq,
        head_dim,
        0,
        1.0 / @sqrt(4.0),
        null,
        0,
        null,
    );

    try std.testing.expectError(error.SequenceTooLong, result);
}

test "applyRope - edge case zero dimensions" {
    // Test RoPE with minimal dimensions
    var data = [_]f32{0} ** 2; // 1 position, 2 dims (1 freq)
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 1, 2 }, .f32);

    ropeFreqs(out, 10000.0, 0);

    // Position 0 should have cos=1, sin=0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 1e-5);
}

test "ropeFreqs - edge case large offset" {
    // Test RoPE with large position offset
    var data = [_]f32{0} ** 4; // 1 position, 4 dims
    const out = TensorView.initContiguous(@ptrCast(&data), &.{ 1, 4 }, .f32);

    const large_offset: usize = 1000;
    ropeFreqs(out, 10000.0, large_offset);

    // Should produce valid (finite) values
    for (data) |val| {
        try std.testing.expect(std.math.isFinite(val));
    }
}

test "updateKVCache edge case - wrapping around max_seq_len" {
    // Test KV cache update with position wrapping
    const n_kv_heads: usize = 1;
    const head_dim: usize = 2;
    const max_seq: usize = 4;
    const seq_len: usize = 2;

    var k_cache = [_]f32{0} ** (max_seq * n_kv_heads * head_dim);
    var v_cache = [_]f32{0} ** (max_seq * n_kv_heads * head_dim);

    var k_data = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    var v_data = [_]f32{ 100.0, 200.0, 300.0, 400.0 };

    const k_strides = [4]usize{ n_kv_heads * seq_len * head_dim, seq_len * head_dim, head_dim, 1 };

    // Write at position that will wrap (seq_pos=3, will write to 3 and 0)
    updateKVCache(
        &k_cache,
        &v_cache,
        @ptrCast(&k_data),
        @ptrCast(&v_data),
        k_strides,
        0, // layer_offset
        3, // seq_pos (near end)
        max_seq,
        seq_len,
        n_kv_heads,
        head_dim,
    );

    // Check position 3 (first token)
    const pos3_idx = (3 % max_seq) * n_kv_heads * head_dim;
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), k_cache[pos3_idx], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), v_cache[pos3_idx], 1e-6);

    // Check position 0 (second token wraps around)
    const pos0_idx = (4 % max_seq) * n_kv_heads * head_dim;
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), k_cache[pos0_idx], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), v_cache[pos0_idx], 1e-6);
}

test "updateKVCache edge case - layer offset" {
    // Test KV cache update with non-zero layer offset
    const n_kv_heads: usize = 1;
    const head_dim: usize = 2;
    const max_seq: usize = 4;
    const seq_len: usize = 1;
    const layer_offset: usize = max_seq * n_kv_heads * head_dim; // Offset for second layer

    var k_cache = [_]f32{0} ** (2 * max_seq * n_kv_heads * head_dim); // 2 layers
    var v_cache = [_]f32{0} ** (2 * max_seq * n_kv_heads * head_dim);

    var k_data = [_]f32{ 5.0, 6.0 };
    var v_data = [_]f32{ 50.0, 60.0 };

    const k_strides = [4]usize{ n_kv_heads * seq_len * head_dim, seq_len * head_dim, head_dim, 1 };

    updateKVCache(
        &k_cache,
        &v_cache,
        @ptrCast(&k_data),
        @ptrCast(&v_data),
        k_strides,
        layer_offset,
        0, // seq_pos
        max_seq,
        seq_len,
        n_kv_heads,
        head_dim,
    );

    // Verify data was written to second layer, not first
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), k_cache[0], 1e-6); // First layer unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), k_cache[layer_offset], 1e-6); // Second layer
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), v_cache[layer_offset], 1e-6);
}

test "sdpaCached - with sink logits" {
    // Test attention with sink tokens (streaming LLM feature)
    const allocator = std.testing.allocator;

    const n_heads: usize = 2;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 4;
    const cached_seq: usize = 4;
    const seq_q: usize = 1;

    var q_data = [_]f32{0} ** (n_heads * seq_q * head_dim);
    for (0..n_heads * seq_q * head_dim) |i| q_data[i] = 1.0;

    var k_cache = [_]f32{0} ** (cached_seq * n_kv_heads * head_dim);
    var v_cache = [_]f32{0} ** (cached_seq * n_kv_heads * head_dim);
    for (0..cached_seq * n_kv_heads * head_dim) |i| {
        k_cache[i] = 1.0;
        v_cache[i] = @as(f32, @floatFromInt(i));
    }

    // Sink logits: large values that should influence attention
    var sink_logits = [_]f32{ 5.0, 3.0 }; // One per head

    var out_data = [_]f32{0} ** (n_heads * seq_q * head_dim);

    const out_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };
    const q_strides = [4]usize{ n_heads * seq_q * head_dim, seq_q * head_dim, head_dim, 1 };

    try sdpaCached(
        @ptrCast(&out_data),
        out_strides,
        @ptrCast(&q_data),
        q_strides,
        &k_cache,
        &v_cache,
        n_heads,
        n_kv_heads,
        seq_q,
        cached_seq,
        head_dim,
        cached_seq - 1,
        1.0 / @sqrt(4.0),
        &sink_logits,
        0,
        allocator,
    );

    // Output should be valid and influenced by sink logits
    try std.testing.expect(std.math.isFinite(out_data[0]));
    try std.testing.expect(out_data[0] >= 0.0);
}

test "stridesToUsize4D basic conversion" {
    // Typical 4D strides for [batch, heads, seq, head_dim]
    const strides = [8]i64{ 1024, 256, 64, 1, 0, 0, 0, 0 };
    const result = stridesToUsize4D(&strides);

    try std.testing.expectEqual(@as(usize, 1024), result[0]);
    try std.testing.expectEqual(@as(usize, 256), result[1]);
    try std.testing.expectEqual(@as(usize, 64), result[2]);
    try std.testing.expectEqual(@as(usize, 1), result[3]);
}

test "stridesToUsize4D contiguous tensor" {
    // Shape [2, 4, 8, 16] contiguous has strides [512, 128, 16, 1]
    const strides = [8]i64{ 512, 128, 16, 1, 0, 0, 0, 0 };
    const result = stridesToUsize4D(&strides);

    try std.testing.expectEqual(@as(usize, 512), result[0]);
    try std.testing.expectEqual(@as(usize, 128), result[1]);
    try std.testing.expectEqual(@as(usize, 16), result[2]);
    try std.testing.expectEqual(@as(usize, 1), result[3]);
}

test "stridesToUsize4D unit strides" {
    // All unit strides (batch=1 case)
    const strides = [8]i64{ 1, 1, 1, 1, 0, 0, 0, 0 };
    const result = stridesToUsize4D(&strides);

    try std.testing.expectEqual(@as(usize, 1), result[0]);
    try std.testing.expectEqual(@as(usize, 1), result[1]);
    try std.testing.expectEqual(@as(usize, 1), result[2]);
    try std.testing.expectEqual(@as(usize, 1), result[3]);
}

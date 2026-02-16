//! CPU Attention Kernel
//! Multi-head attention with grouped query attention (GQA) support
//!
//! This module provides the core attention computation for CPU inference.
//! It supports both prefill (multiple tokens) and decode (single token with KV cache) modes.

const std = @import("std");
const build_options = @import("build_options");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.ops.matmul;
const ops = compute.ops.math;
const simd = compute.simd;
const flash_attention = compute.ops.simd.flash_attention;
const rope_kernel = @import("rope.zig");
const fused_attn = @import("fused_attention.zig");
const fmt = @import("describe_fmt.zig");
const inspect = @import("../../../../xray/root.zig");
const trace = inspect.trace;
const dump = if (build_options.dump_tensors) @import("../../../../xray/dump/capture.zig") else struct {
    pub fn recordGlobal(_: anytype, _: anytype, _: anytype, _: anytype, _: anytype, _: anytype) void {}
};

const Tensor = tensor.Tensor;
const MatmulFn = matmul.MatmulFn;
const RoPE = rope_kernel.RoPE;
const FlashAttentionFn = flash_attention.FlashAttentionFn;

// Use comptime-detected SIMD width for all vector operations
const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// Threshold for using Flash Attention (prefill only).
const FLASH_ATTENTION_THRESHOLD: usize = 512;

/// Temporary scratch buffers for attention computation.
/// These are safe to share across layers because they do not persist state.
pub const AttnTemp = struct {
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []f32 = &.{},
    qkv: []f32 = &.{},
    scores: []f32 = &.{},
    context_values: []f32 = &.{},

    pub fn deinit(self: *AttnTemp, allocator: std.mem.Allocator) void {
        if (self.q.len > 0) allocator.free(self.q);
        if (self.k.len > 0) allocator.free(self.k);
        if (self.v.len > 0) allocator.free(self.v);
        if (self.qkv.len > 0) allocator.free(self.qkv);
        if (self.scores.len > 0) allocator.free(self.scores);
        if (self.context_values.len > 0) allocator.free(self.context_values);
        self.* = .{};
    }
};

/// Per-layer KV cache (must persist across calls).
pub const AttnCache = struct {
    key_cache: []f32 = &.{},
    value_cache: []f32 = &.{},
    kv_capacity: usize = 0,
    cache_position: usize = 0,

    pub fn deinit(self: *AttnCache, allocator: std.mem.Allocator) void {
        if (self.key_cache.len > 0) allocator.free(self.key_cache);
        if (self.value_cache.len > 0) allocator.free(self.value_cache);
        self.* = .{};
    }

    pub fn resetCache(self: *AttnCache) void {
        self.cache_position = 0;
    }
};

/// Multi-head attention layer with grouped query attention support.
pub const MultiHeadAttention = struct {
    pub const RuntimeRoPE = struct {
        /// Flat table [seq_len * dim]
        cos: []const f32,
        /// Flat table [seq_len * dim]
        sin: []const f32,
        /// Rotary width applied to each head.
        dim: usize,
    };

    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    /// Attention softmax temperature (typically 1/sqrt(head_dim), but some models override)
    scale: f32,
    /// Offset added to QK norm weights (e.g., 1.0 for (1+w) formulation)
    qk_norm_weight_offset: f32 = 0.0,
    /// Sliding window attention (0 = disabled). When enabled, each query only attends
    /// to the most recent `sliding_window` keys (still causal).
    sliding_window: usize = 0,
    /// Whether to apply causal masking (default true for autoregressive models).
    /// Set to false for bidirectional/encoder models (e.g., BERT).
    is_causal: bool = true,
    /// Layer index for trace emissions.
    layer_idx: u16 = trace.TraceEmission.NO_LAYER,
    // Q/K/V projections - optional when using native fused QKV
    q_proj: ?*const Tensor = null,
    k_proj: ?*const Tensor = null,
    v_proj: ?*const Tensor = null,
    o_proj: *const Tensor,
    fused_qkv: ?Tensor = null,
    rope: ?*RoPE = null,
    /// Optional per-token precomputed RoPE table (used by multimodal vision prefill).
    /// When set, this overrides `rope` for prefill-style calls.
    runtime_rope: ?RuntimeRoPE = null,
    /// Optional signed offset applied to static RoPE positions.
    /// Used for multimodal decode where generated token positions continue from
    /// M-RoPE text positions rather than raw prompt length.
    position_delta: isize = 0,
    // QKNorm (optional) - applied after Q/K projection, before RoPE
    q_norm: ?*const Tensor = null,
    k_norm: ?*const Tensor = null,
    norm_eps: f32 = 1e-6,
    allocator: std.mem.Allocator,
    // Baked matmul kernels - resolved at load time, no runtime dispatch
    matmul_qkv: MatmulFn, // Default for Q, also used for K/V if they match Q's dtype
    matmul_k: ?MatmulFn = null, // Override for K if different dtype
    matmul_v: ?MatmulFn = null, // Override for V if different dtype
    matmul_qkv_fused: ?MatmulFn = null,
    matmul_o: MatmulFn,
    kernel_name_qkv: ?[]const u8 = null,
    kernel_name_k: ?[]const u8 = null,
    kernel_name_v: ?[]const u8 = null,
    kernel_name_qkv_fused: ?[]const u8 = null,
    kernel_name_o: ?[]const u8 = null,
    // Attention biases (optional)
    q_bias: ?[]const f32 = null,
    k_bias: ?[]const f32 = null,
    v_bias: ?[]const f32 = null,
    o_bias: ?[]const f32 = null,
    // Attention sinks - per-head extra logit prepended to the score vector before softmax.
    sinks: ?[]const f32 = null,
    /// Optional Flash Attention kernel (set at model load time if compatible).
    flash_attention_fn: ?FlashAttentionFn = null,

    fn shouldUseFlash(
        self: *const MultiHeadAttention,
        use_cache: bool,
        exact_softmax: bool,
        sequence_len: usize,
    ) bool {
        return !use_cache and
            !exact_softmax and
            self.flash_attention_fn != null and
            self.sinks == null and
            self.is_causal and
            self.n_heads == self.n_kv_heads and
            sequence_len > FLASH_ATTENTION_THRESHOLD;
    }

    pub fn forward(
        self: *const MultiHeadAttention,
        input_tensor: *const Tensor, // [1, sequence_len, d_model]
        output_tensor: *Tensor, // [1, sequence_len, d_model]
        cache: *AttnCache,
        scratch: *AttnTemp,
        matmul_scratch: *matmul.MatmulScratch,
        use_cache: bool,
    ) !void {
        const exact_softmax = std.process.hasEnvVar(self.allocator, "TALU_CPU_EXACT_SOFTMAX") catch false;
        // Internal invariants: model config must be valid after loading
        std.debug.assert(self.n_heads > 0 and self.n_kv_heads > 0);
        std.debug.assert(self.n_heads % self.n_kv_heads == 0);
        const query_dim = self.n_heads * self.head_dim;
        const kv_total_dim = self.n_kv_heads * self.head_dim;
        std.debug.assert(input_tensor.n_dims == 3 and output_tensor.n_dims == 3);
        std.debug.assert(input_tensor.shape[0] == 1 and output_tensor.shape[0] == 1); // Only batch=1 supported
        const sequence_len: usize = @intCast(input_tensor.shape[1]);
        std.debug.assert(input_tensor.shape[2] == self.d_model and output_tensor.shape[2] == self.d_model);

        // Use fused QKV when available (required for native fused, optional optimization for others)
        // Weight layout is [out_features, in_features] where out = query_dim + 2*kv_total_dim
        // For quantized weights, shape[1] is packed so we only check output dimension
        const use_fused_projection = if (self.fused_qkv) |fq| blk: {
            if (fq.dtype == .f32 and fq.shape[1] == query_dim + 2 * kv_total_dim) break :blk true;
            break :blk fq.shape[0] == query_dim + 2 * kv_total_dim;
        } else false;
        try self.ensureTemp(scratch, sequence_len, use_cache, query_dim, kv_total_dim, use_fused_projection);

        // Flatten batch dimension for matmul
        const input_view = Tensor.view2D(input_tensor.data(), sequence_len, self.d_model);
        var query_view: Tensor = undefined; // Safe: both branches assign before use
        var key_view: Tensor = undefined; // Safe: both branches assign before use
        var value_view: Tensor = undefined; // Safe: both branches assign before use
        if (use_fused_projection) {
            const fused_weights = self.fused_qkv.?;
            const fused_kernel = self.matmul_qkv_fused orelse self.matmul_qkv;
            const views = fused_attn.projectQkv(&input_view, &fused_weights, scratch.qkv, sequence_len, query_dim, kv_total_dim, fused_kernel, matmul_scratch);
            query_view = views.q;
            key_view = views.k;
            value_view = views.v;
        } else {
            // Separate Q/K/V projections - must have all three
            const query_weights = self.q_proj orelse return error.MissingAttentionWeights;
            const key_weights = self.k_proj orelse return error.MissingAttentionWeights;
            const value_weights = self.v_proj orelse return error.MissingAttentionWeights;

            var query_workspace = Tensor.view2DSlice(scratch.q[0 .. sequence_len * query_dim], sequence_len, query_dim);
            var key_workspace = Tensor.view2DSlice(scratch.k[0 .. sequence_len * kv_total_dim], sequence_len, kv_total_dim);
            var value_workspace = Tensor.view2DSlice(scratch.v[0 .. sequence_len * kv_total_dim], sequence_len, kv_total_dim);
            self.matmul_qkv(&input_view, query_weights, &query_workspace, matmul_scratch);
            // Use separate matmul for K/V if they have different dtype than Q
            const matmul_kernel_k = self.matmul_k orelse self.matmul_qkv;
            const matmul_kernel_v = self.matmul_v orelse self.matmul_qkv;
            matmul_kernel_k(&input_view, key_weights, &key_workspace, matmul_scratch);
            matmul_kernel_v(&input_view, value_weights, &value_workspace, matmul_scratch);
            query_view = query_workspace;
            key_view = key_workspace;
            value_view = value_workspace;
        }

        // Apply attention biases if present
        if (self.q_bias) |bias| {
            addBias(query_view.asSlice(f32), bias, sequence_len, query_dim);
        }
        if (self.k_bias) |bias| {
            addBias(key_view.asSlice(f32), bias, sequence_len, kv_total_dim);
        }
        if (self.v_bias) |bias| {
            addBias(value_view.asSlice(f32), bias, sequence_len, kv_total_dim);
        }

        // Apply QKNorm if present (optional per-head normalization)
        // Normalize each head's Q/K vectors independently
        // Note: weight tensors may be stored as BF16/F16 and may not be aligned
        if (self.q_norm) |qn| {
            const q_slice = query_view.asSlice(f32);
            var token_index: usize = 0;
            while (token_index < sequence_len) : (token_index += 1) {
                var head_index: usize = 0;
                while (head_index < self.n_heads) : (head_index += 1) {
                    const offset = token_index * query_dim + head_index * self.head_dim;
                    applyQKNormInPlace(q_slice[offset .. offset + self.head_dim], qn, self.norm_eps, self.qk_norm_weight_offset);
                }
            }
        }
        if (self.k_norm) |kn| {
            const k_slice = key_view.asSlice(f32);
            var token_index: usize = 0;
            while (token_index < sequence_len) : (token_index += 1) {
                var head_index: usize = 0;
                while (head_index < self.n_kv_heads) : (head_index += 1) {
                    const offset = token_index * kv_total_dim + head_index * self.head_dim;
                    applyQKNormInPlace(k_slice[offset .. offset + self.head_dim], kn, self.norm_eps, self.qk_norm_weight_offset);
                }
            }
        }

        // Apply RoPE to Q/K per position/head
        // pos_offset is the position in the sequence (accounting for cached tokens)
        // Note: For partial rotary, rope.dim < head_dim. We only apply RoPE
        // to the first rope.dim dimensions, leaving the rest unchanged.
        const pos_offset = if (use_cache) cache.cache_position else 0;
        if (self.runtime_rope) |runtime_rope| {
            try applyRuntimeRoPE(
                query_view.asSlice(f32),
                key_view.asSlice(f32),
                sequence_len,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                query_dim,
                kv_total_dim,
                pos_offset,
                runtime_rope,
            );
        } else if (self.rope) |rope| {
            const rope_dim = rope.dim;
            var token_index: usize = 0;
            while (token_index < sequence_len) : (token_index += 1) {
                const pos = try applyPositionDelta(pos_offset + token_index, self.position_delta);
                var head_index: usize = 0;
                while (head_index < self.n_heads) : (head_index += 1) {
                    const offset = token_index * query_dim + head_index * self.head_dim;
                    rope.applyInPlace(query_view.asSlice(f32)[offset .. offset + rope_dim], pos);
                }
                head_index = 0;
                while (head_index < self.n_kv_heads) : (head_index += 1) {
                    const offset = token_index * kv_total_dim + head_index * self.head_dim;
                    rope.applyInPlace(key_view.asSlice(f32)[offset .. offset + rope_dim], pos);
                }
            }
        }

        if (trace.isEnabled()) {
            const trace_position: u32 = if (use_cache) @intCast(cache.cache_position) else @intCast(sequence_len);
            // Get kernel names for Q/K/V projections
            const q_kernel = self.kernel_name_qkv;
            const k_kernel = self.kernel_name_k orelse q_kernel;
            const v_kernel = self.kernel_name_v orelse q_kernel;
            trace.emit(
                .attn_q,
                self.layer_idx,
                0,
                trace_position,
                query_view.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(query_dim), 0 },
                3,
                q_kernel,
            );
            trace.emit(
                .attn_k,
                self.layer_idx,
                0,
                trace_position,
                key_view.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(kv_total_dim), 0 },
                3,
                k_kernel,
            );
            trace.emit(
                .attn_v,
                self.layer_idx,
                0,
                trace_position,
                value_view.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(kv_total_dim), 0 },
                3,
                v_kernel,
            );
            // Emit embed_pos after Q/K/V (RoPE was applied in-place before projection trace)
            if (self.rope != null) {
                trace.emit(
                    .embed_pos,
                    self.layer_idx,
                    0,
                    trace_position,
                    query_view.data().ptr,
                    .f32,
                    .{ 1, @intCast(sequence_len), @intCast(query_dim), 0 },
                    3,
                    "rope",
                );
            }
        }

        // Populate KV cache during prefill so subsequent decode steps can attend to the prompt.
        // Cache layout matches the decode path: [kv_head, seq_pos, head_dim].
        if (!use_cache) {
            const k_data_prefill = key_view.asSlice(f32);
            const v_data_prefill = value_view.asSlice(f32);
            try self.ensureKvCapacity(cache, sequence_len, kv_total_dim);
            const head_dim = self.head_dim;
            const n_kv_heads = self.n_kv_heads;
            const kv_stride = cache.kv_capacity;

            for (0..n_kv_heads) |kv_h| {
                var pos: usize = 0;
                while (pos < sequence_len) : (pos += 1) {
                    const src_k = k_data_prefill[pos * kv_total_dim + kv_h * head_dim ..][0..head_dim];
                    const src_v = v_data_prefill[pos * kv_total_dim + kv_h * head_dim ..][0..head_dim];
                    const dst_k = cache.key_cache[kv_h * kv_stride * head_dim + pos * head_dim ..][0..head_dim];
                    const dst_v = cache.value_cache[kv_h * kv_stride * head_dim + pos * head_dim ..][0..head_dim];
                    @memcpy(dst_k, src_k);
                    @memcpy(dst_v, src_v);
                }
            }
            cache.cache_position = sequence_len;
        }

        // Attention scores and context
        const score_values = scratch.scores;
        const context_values = scratch.context_values;
        const query_values = query_view.asSlice(f32);
        const key_values = key_view.asSlice(f32);
        const value_values = value_view.asSlice(f32);
        const scale = self.scale;
        const heads_per_kv_group = self.n_heads / self.n_kv_heads;

        if (use_cache) {
            std.debug.assert(sequence_len == 1); // Cache mode only processes one token at a time
            if (cache.cache_position >= self.max_seq_len) return error.CacheOverflow;

            const kv_sequence_len = cache.cache_position + sequence_len;
            if (kv_sequence_len > self.max_seq_len) return error.CacheOverflow;
            const start_kv_index: usize = if (self.sliding_window > 0 and kv_sequence_len > self.sliding_window)
                kv_sequence_len - self.sliding_window
            else
                0;
            // Grow cache buffers as needed (layout: [kv_head, seq_pos, head_dim])
            try self.ensureKvCapacity(cache, kv_sequence_len, kv_total_dim);

            // Append current K/V to cache with transposed layout [kv_head, seq_pos, head_dim]
            const head_dim = self.head_dim;
            const n_kv_heads = self.n_kv_heads;
            const kv_stride = cache.kv_capacity;
            const cache_position = cache.cache_position;
            const score_stride = self.max_seq_len;

            // Copy K/V for each kv_head to its contiguous region
            for (0..n_kv_heads) |kv_head_idx| {
                const src_k = key_values[kv_head_idx * head_dim ..][0..head_dim];
                const src_v = value_values[kv_head_idx * head_dim ..][0..head_dim];
                // Cache layout: kv_head * (kv_stride * head_dim) + seq_pos * head_dim
                const dst_k = cache.key_cache[kv_head_idx * kv_stride * head_dim + cache_position * head_dim ..][0..head_dim];
                const dst_v = cache.value_cache[kv_head_idx * kv_stride * head_dim + cache_position * head_dim ..][0..head_dim];
                @memcpy(dst_k, src_k);
                @memcpy(dst_v, src_v);
            }

            // Iterate by kv_head first to maximize K/V cache reuse
            // All Q heads sharing the same KV head process together
            var kv_head_idx: usize = 0;
            while (kv_head_idx < n_kv_heads) : (kv_head_idx += 1) {
                // K/V cache for this kv_head - read once, reuse for all Q heads
                const k_cache_base = cache.key_cache[kv_head_idx * kv_stride * head_dim ..];
                const v_cache_base = cache.value_cache[kv_head_idx * kv_stride * head_dim ..];

                // Process all Q heads that share this KV head
                const q_head_start = kv_head_idx * heads_per_kv_group;
                const q_head_end = q_head_start + heads_per_kv_group;

                var head_index: usize = q_head_start;
                while (head_index < q_head_end) : (head_index += 1) {
                    const query_head = query_values[head_index * head_dim ..][0..head_dim];
                    const scores_for_head = score_values[head_index * score_stride ..][0..kv_sequence_len];

                    // Q·K dot products with inline SIMD
                    var max_score: f32 = -std.math.inf(f32);
                    var key_index: usize = 0;
                    while (key_index < start_kv_index) : (key_index += 1) {
                        scores_for_head[key_index] = -std.math.inf(f32);
                    }
                    while (key_index < kv_sequence_len) : (key_index += 1) {
                        const k_row = k_cache_base[key_index * head_dim ..][0..head_dim];

                        // Inline SIMD dot product
                        var sum0: F32Vec = @splat(0);
                        var sum1: F32Vec = @splat(0);
                        var dimension_index: usize = 0;
                        while (dimension_index + 2 * VEC_LEN - 1 < head_dim) : (dimension_index += 2 * VEC_LEN) {
                            const q_vec0: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                            const k_vec0: F32Vec = k_row[dimension_index..][0..VEC_LEN].*;
                            const q_vec1: F32Vec = query_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                            const k_vec1: F32Vec = k_row[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                            sum0 = @mulAdd(F32Vec, q_vec0, k_vec0, sum0);
                            sum1 = @mulAdd(F32Vec, q_vec1, k_vec1, sum1);
                        }
                        while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                            const q_vec: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                            const k_vec: F32Vec = k_row[dimension_index..][0..VEC_LEN].*;
                            sum0 = @mulAdd(F32Vec, q_vec, k_vec, sum0);
                        }
                        var dot = @reduce(.Add, sum0 + sum1);
                        while (dimension_index < head_dim) : (dimension_index += 1) {
                            dot += query_head[dimension_index] * k_row[dimension_index];
                        }
                        dot *= scale;
                        scores_for_head[key_index] = dot;
                        if (dot > max_score) max_score = dot;
                    }

                    // Attention sinks (MLX semantics): add an extra "sink" logit before softmax,
                    // then discard its probability mass (do not renormalize).
                    // This effectively dampens attention outputs by (1 - p_sink).
                    const sink_logit: ?f32 = if (self.sinks) |s| s[head_index] else null;
                    if (sink_logit) |sl| {
                        if (sl > max_score) max_score = sl;
                    }

                    ops.softmaxMaskedInPlaceWithMax(
                        scores_for_head,
                        start_kv_index,
                        kv_sequence_len,
                        sink_logit,
                        exact_softmax,
                        max_score,
                        null,
                    );

                    // Initialize context output for this head to zero
                    const context_for_head = context_values[head_index * head_dim ..][0..head_dim];
                    @memset(context_for_head, 0);

                    // Context accumulation
                    var kv_index: usize = 0;
                    while (kv_index < kv_sequence_len) : (kv_index += 1) {
                        const attn_weight = scores_for_head[kv_index];
                        const v_row = v_cache_base[kv_index * head_dim ..][0..head_dim];

                        // SIMD accumulation with FMA
                        const weight_vec: F32Vec = @splat(attn_weight);
                        var dimension_index: usize = 0;
                        while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                            const v_vec: F32Vec = v_row[dimension_index..][0..VEC_LEN].*;
                            const out_slice = context_for_head[dimension_index..][0..VEC_LEN];
                            out_slice.* = @mulAdd(F32Vec, weight_vec, v_vec, out_slice.*);
                        }
                        while (dimension_index < head_dim) : (dimension_index += 1) {
                            context_for_head[dimension_index] += attn_weight * v_row[dimension_index];
                        }
                    }
                }
            }

            // Emit attention scores/weights (decode: scores are [n_heads, kv_sequence_len] post-softmax)
            if (trace.isEnabled()) {
                const trace_position: u32 = @intCast(cache.cache_position);
                trace.emit(
                    .attn_qk,
                    self.layer_idx,
                    0,
                    trace_position,
                    @ptrCast(score_values.ptr),
                    .f32,
                    .{ @intCast(self.n_heads), @intCast(kv_sequence_len), 0, 0 },
                    2,
                    null,
                );
                trace.emit(
                    .attn_weights,
                    self.layer_idx,
                    0,
                    trace_position,
                    @ptrCast(score_values.ptr),
                    .f32,
                    .{ @intCast(self.n_heads), @intCast(kv_sequence_len), 0, 0 },
                    2,
                    null,
                );
            }

            // Apply output projection directly from context buffer (already laid out as [sequence_len, query_dim])
            const attn_view = Tensor.view2DSlice(context_values[0 .. sequence_len * query_dim], sequence_len, query_dim);
            var out_view = Tensor.view2DSlice(output_tensor.asSlice(f32), sequence_len, self.d_model);
            self.matmul_o(&attn_view, self.o_proj, &out_view, matmul_scratch);
            // Apply output bias if present
            if (self.o_bias) |bias| {
                addBias(output_tensor.asSlice(f32), bias, sequence_len, self.d_model);
            }
            if (trace.isEnabled()) {
                const trace_position_out: u32 = @intCast(cache.cache_position);
                trace.emit(
                    .attn_out,
                    self.layer_idx,
                    0,
                    trace_position_out,
                    output_tensor.data().ptr,
                    .f32,
                    .{ 1, @intCast(sequence_len), @intCast(self.d_model), 0 },
                    3,
                    self.kernel_name_o,
                );
            }
            // Dump capture (compiled in only for dump binary)
            if (build_options.dump_tensors) {
                const shape = [4]usize{ 1, sequence_len, self.d_model, 0 };
                dump.recordGlobal(.attn_out, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
            }
            cache.cache_position = kv_sequence_len;
            return;
        }

        const can_use_flash = self.shouldUseFlash(use_cache, exact_softmax, sequence_len);

        if (can_use_flash) {
            const head_dim = self.head_dim;
            const q_stride_head = head_dim;
            const q_stride_seq = query_dim;
            const k_stride_head = head_dim;
            const k_stride_seq = query_dim;
            const v_stride_head = head_dim;
            const v_stride_seq = query_dim;
            const out_stride_head = head_dim;
            const out_stride_seq = query_dim;

            self.flash_attention_fn.?(
                context_values.ptr,
                out_stride_head,
                out_stride_seq,
                query_values.ptr,
                q_stride_head,
                q_stride_seq,
                key_values.ptr,
                k_stride_head,
                k_stride_seq,
                value_values.ptr,
                v_stride_head,
                v_stride_seq,
                self.n_heads,
                sequence_len,
                sequence_len,
                head_dim,
                scale,
                0,
                self.sliding_window,
            );

            // Flash attention fuses QK and softmax — emit trace points for completeness
            if (trace.isEnabled()) {
                trace.emit(
                    .attn_qk,
                    self.layer_idx,
                    0,
                    @intCast(sequence_len),
                    @ptrCast(context_values.ptr),
                    .f32,
                    .{ @intCast(sequence_len), @intCast(self.n_heads), @intCast(self.head_dim), 0 },
                    3,
                    "flash_attention",
                );
                trace.emit(
                    .attn_weights,
                    self.layer_idx,
                    0,
                    @intCast(sequence_len),
                    @ptrCast(context_values.ptr),
                    .f32,
                    .{ @intCast(sequence_len), @intCast(self.n_heads), @intCast(self.head_dim), 0 },
                    3,
                    "flash_attention",
                );
            }

            const attn_view = Tensor.view2DSlice(context_values[0 .. sequence_len * query_dim], sequence_len, query_dim);
            var out_view = Tensor.view2DSlice(output_tensor.asSlice(f32), sequence_len, self.d_model);
            self.matmul_o(&attn_view, self.o_proj, &out_view, matmul_scratch);
            if (self.o_bias) |bias| {
                addBias(output_tensor.asSlice(f32), bias, sequence_len, self.d_model);
            }
            if (trace.isEnabled()) {
                trace.emit(
                    .attn_out,
                    self.layer_idx,
                    0,
                    @intCast(sequence_len),
                    output_tensor.data().ptr,
                    .f32,
                    .{ 1, @intCast(sequence_len), @intCast(self.d_model), 0 },
                    3,
                    self.kernel_name_o,
                );
            }
            // Dump capture (compiled in only for dump binary)
            if (build_options.dump_tensors) {
                const shape = [4]usize{ 1, sequence_len, self.d_model, 0 };
                dump.recordGlobal(.attn_out, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
            }
            cache.cache_position = sequence_len;
            return;
        }

        // SIMD-optimized prefill attention
        const head_dim = self.head_dim;
        var head_index: usize = 0;
        while (head_index < self.n_heads) : (head_index += 1) {
            const kv_head_idx = head_index / heads_per_kv_group;
            var query_index: usize = 0;
            while (query_index < sequence_len) : (query_index += 1) {
                const end_kv_index: usize = if (self.is_causal) query_index + 1 else sequence_len;
                const start_kv_index: usize = if (self.sliding_window > 0 and end_kv_index > self.sliding_window)
                    end_kv_index - self.sliding_window
                else
                    0;
                var max_score: f32 = -std.math.inf(f32);
                const query_head = query_values[query_index * query_dim + head_index * head_dim ..][0..head_dim];
                const scores_for_query = score_values[head_index * sequence_len ..][0..sequence_len];
                const context_for_head = context_values[(query_index * self.n_heads + head_index) * head_dim ..][0..head_dim];

                var key_index: usize = 0;
                while (key_index < start_kv_index) : (key_index += 1) {
                    scores_for_query[key_index] = -std.math.inf(f32);
                }
                while (key_index < end_kv_index) : (key_index += 1) {
                    const key_head = key_values[key_index * kv_total_dim + kv_head_idx * head_dim ..][0..head_dim];

                    // SIMD dot product
                    var sum0: F32Vec = @splat(0);
                    var sum1: F32Vec = @splat(0);
                    var dimension_index: usize = 0;
                    while (dimension_index + 2 * VEC_LEN - 1 < head_dim) : (dimension_index += 2 * VEC_LEN) {
                        const q_vec0: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                        const k_vec0: F32Vec = key_head[dimension_index..][0..VEC_LEN].*;
                        const q_vec1: F32Vec = query_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                        const k_vec1: F32Vec = key_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                        sum0 = @mulAdd(F32Vec, q_vec0, k_vec0, sum0);
                        sum1 = @mulAdd(F32Vec, q_vec1, k_vec1, sum1);
                    }
                    while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                        const q_vec: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                        const k_vec: F32Vec = key_head[dimension_index..][0..VEC_LEN].*;
                        sum0 = @mulAdd(F32Vec, q_vec, k_vec, sum0);
                    }
                    var dot = @reduce(.Add, sum0 + sum1);
                    while (dimension_index < head_dim) : (dimension_index += 1) {
                        dot += query_head[dimension_index] * key_head[dimension_index];
                    }
                    dot *= scale;
                    scores_for_query[key_index] = dot;
                    if (dot > max_score) max_score = dot;
                }
                // Fill causal mask (only needed for causal models; bidirectional attends to all keys)
                if (self.is_causal) {
                    while (key_index < sequence_len) : (key_index += 1) scores_for_query[key_index] = -std.math.inf(f32);
                }

                // Attention sinks (MLX semantics): add an extra "sink" logit before softmax,
                // then discard its probability mass (do not renormalize).
                const sink_logit: ?f32 = if (self.sinks) |s| s[head_index] else null;
                if (sink_logit) |sl| {
                    if (sl > max_score) max_score = sl;
                }

                ops.softmaxMaskedInPlaceWithMax(
                    scores_for_query,
                    start_kv_index,
                    end_kv_index,
                    sink_logit,
                    exact_softmax,
                    max_score,
                    null,
                );

                // Context accumulation directly into output buffer (avoid storing full scores matrix)
                @memset(context_for_head, 0);
                var value_index: usize = start_kv_index;
                while (value_index < end_kv_index) : (value_index += 1) {
                    const attn_weight = scores_for_query[value_index];
                    if (attn_weight == 0) continue;
                    const value_head = value_values[value_index * kv_total_dim + kv_head_idx * head_dim ..][0..head_dim];
                    const weight_vec: F32Vec = @splat(attn_weight);

                    var dimension_index: usize = 0;
                    while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                        const v_vec: F32Vec = value_head[dimension_index..][0..VEC_LEN].*;
                        const out_slice = context_for_head[dimension_index..][0..VEC_LEN];
                        out_slice.* = @mulAdd(F32Vec, weight_vec, v_vec, out_slice.*);
                    }
                    while (dimension_index < head_dim) : (dimension_index += 1) {
                        context_for_head[dimension_index] += attn_weight * value_head[dimension_index];
                    }
                }
            }
        }

        // Emit attention scores/weights (prefill: scores are [n_heads, sequence_len] post-softmax)
        if (trace.isEnabled()) {
            trace.emit(
                .attn_qk,
                self.layer_idx,
                0,
                @intCast(sequence_len),
                @ptrCast(score_values.ptr),
                .f32,
                .{ @intCast(self.n_heads), @intCast(sequence_len), 0, 0 },
                2,
                null,
            );
            trace.emit(
                .attn_weights,
                self.layer_idx,
                0,
                @intCast(sequence_len),
                @ptrCast(score_values.ptr),
                .f32,
                .{ @intCast(self.n_heads), @intCast(sequence_len), 0, 0 },
                2,
                null,
            );
        }

        // Apply output projection directly from context buffer (layout: [sequence_len, heads, head_dim])
        var attn_view = Tensor.view2DSlice(context_values[0 .. sequence_len * query_dim], sequence_len, query_dim);
        var out_view = Tensor.view2DSlice(output_tensor.asSlice(f32), sequence_len, self.d_model);
        self.matmul_o(&attn_view, self.o_proj, &out_view, matmul_scratch);
        // Apply output bias if present
        if (self.o_bias) |bias| {
            addBias(output_tensor.asSlice(f32), bias, sequence_len, self.d_model);
        }
        if (trace.isEnabled()) {
            trace.emit(
                .attn_out,
                self.layer_idx,
                0,
                @intCast(sequence_len),
                output_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(self.d_model), 0 },
                3,
                self.kernel_name_o,
            );
        }
        // Dump capture (compiled in only for dump binary)
        if (build_options.dump_tensors) {
            const shape = [4]usize{ 1, sequence_len, self.d_model, 0 };
            dump.recordGlobal(.attn_out, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
        }
    }

    pub fn ensureTemp(self: *const MultiHeadAttention, scratch: *AttnTemp, sequence_len: usize, use_cache: bool, query_dim: usize, kv_total_dim: usize, use_fused_projection: bool) !void {
        // Always allocate separate Q, K, V buffers
        try ensureSlice(self.allocator, &scratch.q, sequence_len * query_dim);
        try ensureSlice(self.allocator, &scratch.k, sequence_len * kv_total_dim);
        try ensureSlice(self.allocator, &scratch.v, sequence_len * kv_total_dim);
        if (use_fused_projection) {
            // Need buffer for fused matmul output + rearranged result (2x size)
            // First half: final rearranged Q/K/V
            // Second half: temporary matmul output before rearrangement
            try ensureSlice(self.allocator, &scratch.qkv, 2 * sequence_len * (query_dim + 2 * kv_total_dim));
        }
        // Decode uses scores[head, max_seq_len] for current token; prefill reuses scores[head, sequence_len] per query.
        const scores_needed = if (use_cache) self.n_heads * self.max_seq_len else self.n_heads * sequence_len;
        try ensureSlice(self.allocator, &scratch.scores, scores_needed);
        try ensureSlice(self.allocator, &scratch.context_values, sequence_len * self.n_heads * self.head_dim);
    }

    pub fn ensureKvCapacity(self: *const MultiHeadAttention, cache: *AttnCache, needed_seq: usize, kv_total_dim: usize) !void {
        if (needed_seq <= cache.kv_capacity and cache.key_cache.len > 0 and cache.value_cache.len > 0) return;

        const current = cache.kv_capacity;
        const grow_to = if (current == 0) needed_seq else @max(needed_seq, current * 2);
        const target_seq = @min(self.max_seq_len, grow_to);
        const total = target_seq * kv_total_dim;
        const new_k = try self.allocator.alloc(f32, total);
        errdefer self.allocator.free(new_k);
        const new_v = try self.allocator.alloc(f32, total);
        errdefer self.allocator.free(new_v);

        if (cache.kv_capacity > 0) {
            const old_stride = cache.kv_capacity;
            const head_dim = self.head_dim;
            for (0..self.n_kv_heads) |kv_h| {
                for (0..cache.cache_position) |pos| {
                    const src_k = cache.key_cache[kv_h * old_stride * head_dim + pos * head_dim ..][0..head_dim];
                    const src_v = cache.value_cache[kv_h * old_stride * head_dim + pos * head_dim ..][0..head_dim];
                    const dst_k = new_k[kv_h * target_seq * head_dim + pos * head_dim ..][0..head_dim];
                    const dst_v = new_v[kv_h * target_seq * head_dim + pos * head_dim ..][0..head_dim];
                    @memcpy(dst_k, src_k);
                    @memcpy(dst_v, src_v);
                }
            }
            self.allocator.free(cache.key_cache);
            self.allocator.free(cache.value_cache);
        }

        cache.key_cache = new_k;
        cache.value_cache = new_v;
        cache.kv_capacity = target_seq;
    }

    /// Forward pass using BatchedKVCache instead of AttnCache.
    /// This enables graph-based execution with batched caching for continuous batching.
    ///
    /// Parameters:
    /// - input_tensor: input tensor [1, sequence_len, d_model]
    /// - output_tensor: output tensor [1, sequence_len, d_model]
    /// - cache: batched KV cache
    /// - slot_index: slot in the batched cache for this sequence
    /// - scratch: temporary buffers for Q/K/V/scores/context
    /// - matmul_scratch: scratch for matmul operations
    /// - use_cache: false for prefill (populates cache), true for decode (uses cache)
    pub fn forwardWithBatchedCache(
        self: *const MultiHeadAttention,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        cache: *BatchedKVCache,
        slot_index: usize,
        scratch: *AttnTemp,
        matmul_scratch: *matmul.MatmulScratch,
        use_cache: bool,
    ) !void {
        const exact_softmax = std.process.hasEnvVar(self.allocator, "TALU_CPU_EXACT_SOFTMAX") catch false;
        std.debug.assert(self.n_heads > 0 and self.n_kv_heads > 0);
        std.debug.assert(self.n_heads % self.n_kv_heads == 0);

        const query_dim = self.n_heads * self.head_dim;
        const kv_total_dim = self.n_kv_heads * self.head_dim;
        const head_dim = self.head_dim;
        const n_heads = self.n_heads;
        const n_kv_heads = self.n_kv_heads;
        const heads_per_kv_group = n_heads / n_kv_heads;
        const scale = self.scale;

        std.debug.assert(input_tensor.n_dims == 3 and output_tensor.n_dims == 3);
        std.debug.assert(input_tensor.shape[0] == 1 and output_tensor.shape[0] == 1);
        const sequence_len: usize = @intCast(input_tensor.shape[1]);
        std.debug.assert(input_tensor.shape[2] == self.d_model and output_tensor.shape[2] == self.d_model);

        // Check for fused QKV
        const use_fused_projection = if (self.fused_qkv) |fq| blk: {
            if (fq.dtype == .f32 and fq.shape[1] == query_dim + 2 * kv_total_dim) break :blk true;
            break :blk fq.shape[0] == query_dim + 2 * kv_total_dim;
        } else false;

        try self.ensureTemp(scratch, sequence_len, use_cache, query_dim, kv_total_dim, use_fused_projection);

        // Project Q/K/V
        const input_view = Tensor.view2D(input_tensor.data(), sequence_len, self.d_model);
        var query_view: Tensor = undefined; // Safe: both branches assign before use
        var key_view: Tensor = undefined; // Safe: both branches assign before use
        var value_view: Tensor = undefined; // Safe: both branches assign before use

        if (use_fused_projection) {
            const fused_weights = self.fused_qkv.?;
            const fused_kernel = self.matmul_qkv_fused orelse self.matmul_qkv;
            const views = fused_attn.projectQkv(&input_view, &fused_weights, scratch.qkv, sequence_len, query_dim, kv_total_dim, fused_kernel, matmul_scratch);
            query_view = views.q;
            key_view = views.k;
            value_view = views.v;
        } else {
            const query_weights = self.q_proj orelse return error.MissingAttentionWeights;
            const key_weights = self.k_proj orelse return error.MissingAttentionWeights;
            const value_weights = self.v_proj orelse return error.MissingAttentionWeights;

            var query_workspace = Tensor.view2DSlice(scratch.q[0 .. sequence_len * query_dim], sequence_len, query_dim);
            var key_workspace = Tensor.view2DSlice(scratch.k[0 .. sequence_len * kv_total_dim], sequence_len, kv_total_dim);
            var value_workspace = Tensor.view2DSlice(scratch.v[0 .. sequence_len * kv_total_dim], sequence_len, kv_total_dim);

            self.matmul_qkv(&input_view, query_weights, &query_workspace, matmul_scratch);
            const matmul_kernel_k = self.matmul_k orelse self.matmul_qkv;
            const matmul_kernel_v = self.matmul_v orelse self.matmul_qkv;
            matmul_kernel_k(&input_view, key_weights, &key_workspace, matmul_scratch);
            matmul_kernel_v(&input_view, value_weights, &value_workspace, matmul_scratch);

            query_view = query_workspace;
            key_view = key_workspace;
            value_view = value_workspace;
        }

        // Apply biases if present
        if (self.q_bias) |bias| addBias(query_view.asSlice(f32), bias, sequence_len, query_dim);
        if (self.k_bias) |bias| addBias(key_view.asSlice(f32), bias, sequence_len, kv_total_dim);
        if (self.v_bias) |bias| addBias(value_view.asSlice(f32), bias, sequence_len, kv_total_dim);

        // Apply QKNorm if present (before RoPE)
        if (self.q_norm) |qn| {
            const q_slice = query_view.asSlice(f32);
            for (0..sequence_len) |token_index| {
                for (0..n_heads) |head_index| {
                    const offset = token_index * query_dim + head_index * head_dim;
                    applyQKNormInPlace(q_slice[offset .. offset + head_dim], qn, self.norm_eps, self.qk_norm_weight_offset);
                }
            }
        }
        if (self.k_norm) |kn| {
            const k_slice = key_view.asSlice(f32);
            for (0..sequence_len) |token_index| {
                for (0..n_kv_heads) |head_index| {
                    const offset = token_index * kv_total_dim + head_index * head_dim;
                    applyQKNormInPlace(k_slice[offset .. offset + head_dim], kn, self.norm_eps, self.qk_norm_weight_offset);
                }
            }
        }

        // Get current cache position for RoPE
        const cache_position = cache.getPosition(slot_index);
        const pos_offset = if (use_cache) cache_position else 0;

        // Apply RoPE (after QKNorm)
        if (self.runtime_rope) |runtime_rope| {
            try applyRuntimeRoPE(
                query_view.asSlice(f32),
                key_view.asSlice(f32),
                sequence_len,
                n_heads,
                n_kv_heads,
                head_dim,
                query_dim,
                kv_total_dim,
                pos_offset,
                runtime_rope,
            );
        } else if (self.rope) |rope| {
            const rope_dim = rope.dim;
            for (0..sequence_len) |token_index| {
                const pos = try applyPositionDelta(pos_offset + token_index, self.position_delta);
                for (0..n_heads) |head_index| {
                    const offset = token_index * query_dim + head_index * head_dim;
                    rope.applyInPlace(query_view.asSlice(f32)[offset .. offset + rope_dim], pos);
                }
                for (0..n_kv_heads) |head_index| {
                    const offset = token_index * kv_total_dim + head_index * head_dim;
                    rope.applyInPlace(key_view.asSlice(f32)[offset .. offset + rope_dim], pos);
                }
            }
        }

        // Trace Q/K/V after projections and RoPE
        if (trace.isEnabled()) {
            const trace_pos: u32 = @intCast(if (use_cache) cache_position else sequence_len);
            const q_kernel = self.kernel_name_qkv;
            const k_kernel = self.kernel_name_k orelse q_kernel;
            const v_kernel = self.kernel_name_v orelse q_kernel;
            trace.emit(.attn_q, self.layer_idx, 0, trace_pos, query_view.data().ptr, .f32, .{ 1, @intCast(sequence_len), @intCast(query_dim), 0 }, 3, q_kernel);
            trace.emit(.attn_k, self.layer_idx, 0, trace_pos, key_view.data().ptr, .f32, .{ 1, @intCast(sequence_len), @intCast(kv_total_dim), 0 }, 3, k_kernel);
            trace.emit(.attn_v, self.layer_idx, 0, trace_pos, value_view.data().ptr, .f32, .{ 1, @intCast(sequence_len), @intCast(kv_total_dim), 0 }, 3, v_kernel);
            // Emit embed_pos after Q/K/V (RoPE was applied in-place before projection trace)
            if (self.rope != null or self.runtime_rope != null) {
                trace.emit(.embed_pos, self.layer_idx, 0, trace_pos, query_view.data().ptr, .f32, .{ 1, @intCast(sequence_len), @intCast(query_dim), 0 }, 3, "rope");
            }
        }

        const score_values = scratch.scores;
        const context_values = scratch.context_values;
        const query_values = query_view.asSlice(f32);
        const key_values = key_view.asSlice(f32);
        const value_values = value_view.asSlice(f32);

        if (use_cache) {
            // === DECODE MODE ===
            std.debug.assert(sequence_len == 1);
            if (cache_position >= self.max_seq_len) return error.CacheOverflow;

            // Append current K/V to batched cache
            try cache.appendKV(slot_index, key_values, value_values);
            const kv_sequence_len = cache.getPosition(slot_index);

            const start_kv_index: usize = if (self.sliding_window > 0 and kv_sequence_len > self.sliding_window)
                kv_sequence_len - self.sliding_window
            else
                0;

            const score_stride = self.max_seq_len;

            // Compute attention for each KV head group
            for (0..n_kv_heads) |kv_head_idx| {
                const k_cache_head = cache.getKHead(slot_index, kv_head_idx);
                const v_cache_head = cache.getVHead(slot_index, kv_head_idx);

                const q_head_start = kv_head_idx * heads_per_kv_group;
                const q_head_end = q_head_start + heads_per_kv_group;

                for (q_head_start..q_head_end) |head_index| {
                    const query_head = query_values[head_index * head_dim ..][0..head_dim];
                    const scores_for_head = score_values[head_index * score_stride ..][0..kv_sequence_len];

                    // Q·K dot products
                    var max_score: f32 = -std.math.inf(f32);
                    for (0..start_kv_index) |key_index| {
                        scores_for_head[key_index] = -std.math.inf(f32);
                    }
                    for (start_kv_index..kv_sequence_len) |key_index| {
                        const k_row = k_cache_head[key_index * head_dim ..][0..head_dim];
                        var sum0: F32Vec = @splat(0);
                        var sum1: F32Vec = @splat(0);
                        var dimension_index: usize = 0;
                        while (dimension_index + 2 * VEC_LEN - 1 < head_dim) : (dimension_index += 2 * VEC_LEN) {
                            const q_vec0: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                            const k_vec0: F32Vec = k_row[dimension_index..][0..VEC_LEN].*;
                            const q_vec1: F32Vec = query_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                            const k_vec1: F32Vec = k_row[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                            sum0 = @mulAdd(F32Vec, q_vec0, k_vec0, sum0);
                            sum1 = @mulAdd(F32Vec, q_vec1, k_vec1, sum1);
                        }
                        while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                            const q_vec: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                            const k_vec: F32Vec = k_row[dimension_index..][0..VEC_LEN].*;
                            sum0 = @mulAdd(F32Vec, q_vec, k_vec, sum0);
                        }
                        var dot = @reduce(.Add, sum0 + sum1);
                        while (dimension_index < head_dim) : (dimension_index += 1) {
                            dot += query_head[dimension_index] * k_row[dimension_index];
                        }
                        dot *= scale;
                        scores_for_head[key_index] = dot;
                        if (dot > max_score) max_score = dot;
                    }

                    // Softmax with sink support
                    const sink_logit: ?f32 = if (self.sinks) |s| s[head_index] else null;
                    if (sink_logit) |sl| {
                        if (sl > max_score) max_score = sl;
                    }
                    ops.softmaxMaskedInPlaceWithMax(scores_for_head, start_kv_index, kv_sequence_len, sink_logit, exact_softmax, max_score, null);

                    // Context accumulation
                    const context_for_head = context_values[head_index * head_dim ..][0..head_dim];
                    @memset(context_for_head, 0);
                    for (0..kv_sequence_len) |kv_index| {
                        const attn_weight = scores_for_head[kv_index];
                        const v_row = v_cache_head[kv_index * head_dim ..][0..head_dim];
                        const weight_vec: F32Vec = @splat(attn_weight);
                        var dimension_index: usize = 0;
                        while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                            const v_vec: F32Vec = v_row[dimension_index..][0..VEC_LEN].*;
                            const out_slice = context_for_head[dimension_index..][0..VEC_LEN];
                            out_slice.* = @mulAdd(F32Vec, weight_vec, v_vec, out_slice.*);
                        }
                        while (dimension_index < head_dim) : (dimension_index += 1) {
                            context_for_head[dimension_index] += attn_weight * v_row[dimension_index];
                        }
                    }
                }
            }
        } else {
            // === PREFILL MODE ===
            // Store K/V in batched cache and compute attention (causal or bidirectional)

            // First, store all K/V in the cache
            for (0..sequence_len) |token_index| {
                const k_token = key_values[token_index * kv_total_dim ..][0..kv_total_dim];
                const v_token = value_values[token_index * kv_total_dim ..][0..kv_total_dim];
                try cache.appendKV(slot_index, k_token, v_token);
            }

            // Compute self-attention (respects is_causal for encoder vs decoder)
            for (0..n_heads) |head_index| {
                const kv_head_idx = head_index / heads_per_kv_group;
                for (0..sequence_len) |query_index| {
                    const end_kv_index: usize = if (self.is_causal) query_index + 1 else sequence_len;
                    const start_kv_index: usize = if (self.sliding_window > 0 and end_kv_index > self.sliding_window)
                        end_kv_index - self.sliding_window
                    else
                        0;

                    var max_score: f32 = -std.math.inf(f32);
                    const query_head = query_values[query_index * query_dim + head_index * head_dim ..][0..head_dim];
                    const scores_for_query = score_values[head_index * sequence_len ..][0..sequence_len];
                    const context_for_head = context_values[(query_index * n_heads + head_index) * head_dim ..][0..head_dim];

                    // Compute scores for attended positions
                    for (0..start_kv_index) |key_index| {
                        scores_for_query[key_index] = -std.math.inf(f32);
                    }
                    for (start_kv_index..end_kv_index) |key_index| {
                        const key_head = key_values[key_index * kv_total_dim + kv_head_idx * head_dim ..][0..head_dim];
                        var sum0: F32Vec = @splat(0);
                        var sum1: F32Vec = @splat(0);
                        var dimension_index: usize = 0;
                        while (dimension_index + 2 * VEC_LEN - 1 < head_dim) : (dimension_index += 2 * VEC_LEN) {
                            const q_vec0: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                            const k_vec0: F32Vec = key_head[dimension_index..][0..VEC_LEN].*;
                            const q_vec1: F32Vec = query_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                            const k_vec1: F32Vec = key_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                            sum0 = @mulAdd(F32Vec, q_vec0, k_vec0, sum0);
                            sum1 = @mulAdd(F32Vec, q_vec1, k_vec1, sum1);
                        }
                        while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                            const q_vec: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                            const k_vec: F32Vec = key_head[dimension_index..][0..VEC_LEN].*;
                            sum0 = @mulAdd(F32Vec, q_vec, k_vec, sum0);
                        }
                        var dot = @reduce(.Add, sum0 + sum1);
                        while (dimension_index < head_dim) : (dimension_index += 1) {
                            dot += query_head[dimension_index] * key_head[dimension_index];
                        }
                        dot *= scale;
                        scores_for_query[key_index] = dot;
                        if (dot > max_score) max_score = dot;
                    }

                    // Mask future positions (only for causal/decoder models)
                    if (self.is_causal) {
                        for (end_kv_index..sequence_len) |key_index| {
                            scores_for_query[key_index] = -std.math.inf(f32);
                        }
                    }

                    // Softmax
                    const sink_logit: ?f32 = if (self.sinks) |s| s[head_index] else null;
                    if (sink_logit) |sl| {
                        if (sl > max_score) max_score = sl;
                    }
                    ops.softmaxMaskedInPlaceWithMax(scores_for_query, start_kv_index, end_kv_index, sink_logit, exact_softmax, max_score, null);

                    // Context
                    @memset(context_for_head, 0);
                    for (0..end_kv_index) |key_index| {
                        const attn_weight = scores_for_query[key_index];
                        const value_head = value_values[key_index * kv_total_dim + kv_head_idx * head_dim ..][0..head_dim];
                        const weight_vec: F32Vec = @splat(attn_weight);
                        var dimension_index: usize = 0;
                        while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                            const v_vec: F32Vec = value_head[dimension_index..][0..VEC_LEN].*;
                            const out_slice = context_for_head[dimension_index..][0..VEC_LEN];
                            out_slice.* = @mulAdd(F32Vec, weight_vec, v_vec, out_slice.*);
                        }
                        while (dimension_index < head_dim) : (dimension_index += 1) {
                            context_for_head[dimension_index] += attn_weight * value_head[dimension_index];
                        }
                    }
                }
            }
        }

        // Emit attention scores/weights trace points
        if (trace.isEnabled()) {
            const trace_pos_scores: u32 = @intCast(if (use_cache) cache.getPosition(slot_index) else sequence_len);
            const scores_dim1: u32 = if (use_cache) @intCast(cache.getPosition(slot_index)) else @intCast(sequence_len);
            trace.emit(
                .attn_qk,
                self.layer_idx,
                0,
                trace_pos_scores,
                @ptrCast(score_values.ptr),
                .f32,
                .{ @intCast(n_heads), scores_dim1, 0, 0 },
                2,
                null,
            );
            trace.emit(
                .attn_weights,
                self.layer_idx,
                0,
                trace_pos_scores,
                @ptrCast(score_values.ptr),
                .f32,
                .{ @intCast(n_heads), scores_dim1, 0, 0 },
                2,
                null,
            );
        }

        // Output projection
        const attn_view = Tensor.view2DSlice(context_values[0 .. sequence_len * query_dim], sequence_len, query_dim);
        var out_view = Tensor.view2DSlice(output_tensor.asSlice(f32), sequence_len, self.d_model);
        self.matmul_o(&attn_view, self.o_proj, &out_view, matmul_scratch);
        if (self.o_bias) |bias| addBias(output_tensor.asSlice(f32), bias, sequence_len, self.d_model);

        // Trace: emit attention output
        if (trace.isEnabled()) {
            trace.emit(
                .attn_out,
                self.layer_idx,
                0,
                @intCast(if (use_cache) cache.getPosition(slot_index) else sequence_len),
                output_tensor.data().ptr,
                .f32,
                .{ 1, @intCast(sequence_len), @intCast(self.d_model), 0 },
                3,
                self.kernel_name_o,
            );
        }
        // Dump capture (compiled in only for dump binary)
        if (build_options.dump_tensors) {
            const shape = [4]usize{ 1, sequence_len, self.d_model, 0 };
            dump.recordGlobal(.attn_out, self.layer_idx, output_tensor.data().ptr, .f32, shape, 3);
        }
    }

    /// Describe this attention module for introspection/debugging.
    pub fn describe(self: *const MultiHeadAttention, writer: anytype, indent: usize, show_kernels: bool) !void {
        const query_dim = self.n_heads * self.head_dim;
        const kv_dim = self.n_kv_heads * self.head_dim;

        try fmt.writeIndent(writer, indent);
        try writer.print("Attention(n_heads={}, n_kv_heads={}, head_dim={})\n", .{
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
        });

        // Sub-modules (may be fused)
        if (self.fused_qkv) |fq| {
            try fmt.writeIndent(writer, indent + 2);
            try writer.print("(qkv_proj): Linear(in={}, out={}, dtype={s})\n", .{ self.d_model, query_dim + 2 * kv_dim, @tagName(fq.dtype) });
        } else {
            if (self.q_proj) |qp| try fmt.describeLinearLine(writer, indent + 2, "q_proj", qp, self.q_bias, self.d_model, query_dim);
            if (self.k_proj) |kp| try fmt.describeLinearLine(writer, indent + 2, "k_proj", kp, self.k_bias, self.d_model, kv_dim);
            if (self.v_proj) |vp| try fmt.describeLinearLine(writer, indent + 2, "v_proj", vp, self.v_bias, self.d_model, kv_dim);
        }
        try fmt.describeLinearLine(writer, indent + 2, "o_proj", self.o_proj, self.o_bias, query_dim, self.d_model);

        if (self.q_norm != null) {
            try fmt.describeRmsNormLine(writer, indent + 2, "q_norm", self.head_dim, self.norm_eps, self.qk_norm_weight_offset);
        }
        if (self.k_norm != null) {
            try fmt.describeRmsNormLine(writer, indent + 2, "k_norm", self.head_dim, self.norm_eps, self.qk_norm_weight_offset);
        }

        if (show_kernels) {
            try fmt.writeIndent(writer, indent + 2);
            try writer.writeAll("Kernels:\n");
            try self.formatKernels(writer, indent + 4);
        }
    }

    fn formatKernels(self: *const MultiHeadAttention, writer: anytype, indent: usize) !void {
        const query_dim = self.n_heads * self.head_dim;
        const kv_dim = self.n_kv_heads * self.head_dim;

        // Q/K/V projections (may be fused)
        if (self.fused_qkv) |fq| {
            try fmt.formatSeqMatmulOp(writer, indent, self.d_model, query_dim + 2 * kv_dim, fq.dtype);
        } else if (self.q_proj != null and self.k_proj != null and self.v_proj != null) {
            try fmt.formatSeqMatmulOp(writer, indent, self.d_model, query_dim, self.q_proj.?.dtype);
            try fmt.formatSeqMatmulOp(writer, indent, self.d_model, kv_dim, self.k_proj.?.dtype);
            try fmt.formatSeqMatmulOp(writer, indent, self.d_model, kv_dim, self.v_proj.?.dtype);
        }

        // QK norm if present
        if (self.q_norm != null) {
            const qk_norm_op = fmt.KernelOp{ .rmsnorm = .{ .dim = self.head_dim, .eps = self.norm_eps } };
            try qk_norm_op.format(writer, indent);
        }

        // RoPE
        if (self.rope) |r| {
            const rope_op = fmt.KernelOp{ .rope = .{ .dim = self.head_dim, .theta = r.theta } };
            try rope_op.format(writer, indent);
        }

        // SDPA
        const sdpa_op = fmt.KernelOp{ .sdpa = .{
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .scale = self.scale,
            .causal = true,
        } };
        try sdpa_op.format(writer, indent);

        // O projection
        try fmt.formatSeqMatmulOp(writer, indent, query_dim, self.d_model, self.o_proj.dtype);
    }

    /// Forward pass with performance tracing.
    pub fn forwardTraced(
        self: *const MultiHeadAttention,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        cache: *AttnCache,
        scratch: *AttnTemp,
        matmul_scratch: *matmul.MatmulScratch,
        use_cache: bool,
    ) !void {
        try self.forward(input_tensor, output_tensor, cache, scratch, matmul_scratch, use_cache);
    }
};

/// Apply QKNorm (RMS normalization) in-place with support for BF16/F16/F32 weight tensors.
/// This handles the weight tensor dtype conversion for QKNorm weights which may be stored
/// in various formats in safetensors files.
fn applyQKNormInPlace(vec: []f32, weight_tensor: *const Tensor, eps: f32, weight_offset: f32) void {
    ops.rmsnormInPlaceWeightTensor(vec, weight_tensor, eps, weight_offset);
}

fn ensureSlice(allocator: std.mem.Allocator, storage: *[]f32, needed: usize) !void {
    if (storage.*.len >= needed) return;
    if (storage.*.len > 0) allocator.free(storage.*);
    storage.* = try allocator.alloc(f32, needed);
}

/// Add bias to output tensor (for attention with bias)
/// data: [sequence_len, dim], bias: [dim]
fn addBias(data: []f32, bias: []const f32, sequence_len: usize, dim: usize) void {
    for (0..sequence_len) |token_index| {
        const row = data[token_index * dim ..][0..dim];
        var vec_idx: usize = 0;
        while (vec_idx + VEC_LEN - 1 < dim) : (vec_idx += VEC_LEN) {
            const row_vec: F32Vec = row[vec_idx..][0..VEC_LEN].*;
            const bias_vec: F32Vec = bias[vec_idx..][0..VEC_LEN].*;
            row[vec_idx..][0..VEC_LEN].* = row_vec + bias_vec;
        }
        while (vec_idx < dim) : (vec_idx += 1) {
            row[vec_idx] += bias[vec_idx];
        }
    }
}

fn applyRuntimeRoPE(
    query_values: []f32,
    key_values: []f32,
    sequence_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    query_dim: usize,
    kv_total_dim: usize,
    pos_offset: usize,
    runtime_rope: MultiHeadAttention.RuntimeRoPE,
) !void {
    const rope_dim = runtime_rope.dim;
    if (rope_dim == 0 or rope_dim > head_dim or (rope_dim % 2) != 0) return error.InvalidShape;

    for (0..sequence_len) |token_index| {
        const pos = pos_offset + token_index;
        const base = pos * rope_dim;
        if (base + rope_dim > runtime_rope.cos.len or base + rope_dim > runtime_rope.sin.len) return error.InvalidShape;
        const cos = runtime_rope.cos[base .. base + rope_dim];
        const sin = runtime_rope.sin[base .. base + rope_dim];

        for (0..n_heads) |head_index| {
            const offset = token_index * query_dim + head_index * head_dim;
            applyRoPEFromCosSin(query_values[offset .. offset + rope_dim], cos, sin);
        }
        for (0..n_kv_heads) |head_index| {
            const offset = token_index * kv_total_dim + head_index * head_dim;
            applyRoPEFromCosSin(key_values[offset .. offset + rope_dim], cos, sin);
        }
    }
}

fn applyRoPEFromCosSin(vec: []f32, cos: []const f32, sin: []const f32) void {
    const half = vec.len / 2;
    for (0..half) |idx| {
        const x1 = vec[idx];
        const x2 = vec[idx + half];
        vec[idx] = x1 * cos[idx] - x2 * sin[idx];
        vec[idx + half] = x2 * cos[idx + half] + x1 * sin[idx + half];
    }
}

fn applyPositionDelta(base_pos: usize, delta: isize) !usize {
    if (delta == 0) return base_pos;
    const shifted: i64 = @as(i64, @intCast(base_pos)) + @as(i64, delta);
    if (shifted < 0) return error.InvalidShape;
    return @as(usize, @intCast(shifted));
}

// =============================================================================
// Batched Decode Support (for continuous batching)
// =============================================================================

const kv_cache = @import("kv_cache.zig");
const BatchedKVCache = kv_cache.BatchedKVCache;

/// Request for batched decode - one token per active sequence.
pub const BatchedDecodeRequest = struct {
    /// Slot index in the batched KV cache
    slot_index: usize,
    /// Input hidden state for this sequence [d_model]
    input_values: []const f32,
    /// Output buffer for this sequence [d_model]
    output_values: []f32,
};

/// Temporary scratch for batched attention.
/// Sized for max_batch_size sequences.
pub const BatchedAttnTemp = struct {
    allocator: std.mem.Allocator,
    max_batch_size: usize,

    // Per-sequence buffers: [max_batch * dim]
    q: []f32 = &.{},
    k: []f32 = &.{},
    v: []f32 = &.{},
    qkv: []f32 = &.{},
    scores: []f32 = &.{}, // [max_batch * n_heads * max_seq_len]
    context_values: []f32 = &.{}, // [max_batch * n_heads * head_dim]

    pub fn init(
        allocator: std.mem.Allocator,
        max_batch_size: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) !BatchedAttnTemp {
        const query_dim = n_heads * head_dim;
        const kv_total_dim = n_kv_heads * head_dim;

        return .{
            .allocator = allocator,
            .max_batch_size = max_batch_size,
            .q = try allocator.alloc(f32, max_batch_size * query_dim),
            .k = try allocator.alloc(f32, max_batch_size * kv_total_dim),
            .v = try allocator.alloc(f32, max_batch_size * kv_total_dim),
            .qkv = try allocator.alloc(f32, 2 * max_batch_size * (query_dim + 2 * kv_total_dim)),
            .scores = try allocator.alloc(f32, max_batch_size * n_heads * max_seq_len),
            .context_values = try allocator.alloc(f32, max_batch_size * query_dim),
        };
    }

    pub fn deinit(self: *BatchedAttnTemp) void {
        if (self.context_values.len > 0) self.allocator.free(self.context_values);
        if (self.scores.len > 0) self.allocator.free(self.scores);
        if (self.qkv.len > 0) self.allocator.free(self.qkv);
        if (self.v.len > 0) self.allocator.free(self.v);
        if (self.k.len > 0) self.allocator.free(self.k);
        if (self.q.len > 0) self.allocator.free(self.q);
        self.* = undefined;
    }

    /// Get Q buffer for batch index
    pub fn getQ(self: *BatchedAttnTemp, batch_index: usize, query_dim: usize) []f32 {
        return self.q[batch_index * query_dim ..][0..query_dim];
    }

    /// Get K buffer for batch index
    pub fn getK(self: *BatchedAttnTemp, batch_index: usize, kv_total_dim: usize) []f32 {
        return self.k[batch_index * kv_total_dim ..][0..kv_total_dim];
    }

    /// Get V buffer for batch index
    pub fn getV(self: *BatchedAttnTemp, batch_index: usize, kv_total_dim: usize) []f32 {
        return self.v[batch_index * kv_total_dim ..][0..kv_total_dim];
    }

    /// Get scores buffer for batch index
    pub fn getScores(self: *BatchedAttnTemp, batch_index: usize, n_heads: usize, max_seq_len: usize) []f32 {
        const stride = n_heads * max_seq_len;
        return self.scores[batch_index * stride ..][0..stride];
    }

    /// Get context buffer for batch index
    pub fn getContext(self: *BatchedAttnTemp, batch_index: usize, query_dim: usize) []f32 {
        return self.context_values[batch_index * query_dim ..][0..query_dim];
    }
};

/// Batched decode: process multiple sequences in parallel.
///
/// Each request represents one sequence with its own:
/// - slot_index: position in the BatchedKVCache
/// - input: hidden state for the current token
/// - output: buffer for the attention output
///
/// This function:
/// 1. Projects Q/K/V for all sequences
/// 2. Appends K/V to each sequence's cache slot
/// 3. Computes attention for each sequence against its cached K/V
/// 4. Projects output for all sequences
fn forwardBatchedDecode(
    self: *const MultiHeadAttention,
    requests: []const BatchedDecodeRequest,
    cache: *BatchedKVCache,
    scratch: *BatchedAttnTemp,
    matmul_scratch: *matmul.MatmulScratch,
) !void {
    const batch_size = requests.len;
    if (batch_size == 0) return;

    const query_dim = self.n_heads * self.head_dim;
    const kv_total_dim = self.n_kv_heads * self.head_dim;
    const head_dim = self.head_dim;
    const n_heads = self.n_heads;
    const n_kv_heads = self.n_kv_heads;
    const heads_per_kv_group = n_heads / n_kv_heads;
    const scale = self.scale;
    const max_seq_len = self.max_seq_len;

    // Process each sequence (could be parallelized in future)
    for (requests, 0..) |request, batch_index| {
        const slot_index = request.slot_index;
        const cache_position = cache.getPosition(slot_index);

        // 1. Project Q/K/V for this sequence
        const input_view = Tensor.view2DSlice(@constCast(request.input_values), 1, self.d_model);
        const query_buffer = scratch.getQ(batch_index, query_dim);
        const key_buffer = scratch.getK(batch_index, kv_total_dim);
        const value_buffer = scratch.getV(batch_index, kv_total_dim);

        // Q projection
        var query_view = Tensor.view2DSlice(query_buffer, 1, query_dim);
        self.matmul_qkv(&input_view, self.q_proj.?, &query_view, matmul_scratch);
        if (self.q_bias) |bias| addBias(query_buffer, bias, 1, query_dim);

        // K projection
        var key_view = Tensor.view2DSlice(key_buffer, 1, kv_total_dim);
        const matmul_k = self.matmul_k orelse self.matmul_qkv;
        matmul_k(&input_view, self.k_proj.?, &key_view, matmul_scratch);
        if (self.k_bias) |bias| addBias(key_buffer, bias, 1, kv_total_dim);

        // V projection
        var value_view = Tensor.view2DSlice(value_buffer, 1, kv_total_dim);
        const matmul_v = self.matmul_v orelse self.matmul_qkv;
        matmul_v(&input_view, self.v_proj.?, &value_view, matmul_scratch);
        if (self.v_bias) |bias| addBias(value_buffer, bias, 1, kv_total_dim);

        // Apply QKNorm if configured (must be before RoPE, same as in forward())
        if (self.q_norm) |q_norm_weights| {
            for (0..n_heads) |h| {
                applyQKNormInPlace(
                    query_buffer[h * head_dim ..][0..head_dim],
                    q_norm_weights,
                    self.norm_eps,
                    self.qk_norm_weight_offset,
                );
            }
        }
        if (self.k_norm) |k_norm_weights| {
            for (0..n_kv_heads) |h| {
                applyQKNormInPlace(
                    key_buffer[h * head_dim ..][0..head_dim],
                    k_norm_weights,
                    self.norm_eps,
                    self.qk_norm_weight_offset,
                );
            }
        }

        // Apply RoPE if configured (after QKNorm)
        if (self.rope) |rope_ptr| {
            const rope_dim = rope_ptr.dim;
            for (0..n_heads) |h| {
                rope_ptr.applyInPlace(query_buffer[h * head_dim ..][0..rope_dim], cache_position);
            }
            for (0..n_kv_heads) |h| {
                rope_ptr.applyInPlace(key_buffer[h * head_dim ..][0..rope_dim], cache_position);
            }
        }

        // 2. Append K/V to cache
        try cache.appendKV(slot_index, key_buffer, value_buffer);
        const kv_sequence_len = cache.getPosition(slot_index);

        // Sliding window
        const start_kv_index: usize = if (self.sliding_window > 0 and kv_sequence_len > self.sliding_window)
            kv_sequence_len - self.sliding_window
        else
            0;

        // 3. Compute attention scores and context
        const scores_base = scratch.getScores(batch_index, n_heads, max_seq_len);
        const context_base = scratch.getContext(batch_index, query_dim);

        for (0..n_kv_heads) |kv_head_idx| {
            const k_cache_head = cache.getKHead(slot_index, kv_head_idx);
            const v_cache_head = cache.getVHead(slot_index, kv_head_idx);

            const q_head_start = kv_head_idx * heads_per_kv_group;
            const q_head_end = q_head_start + heads_per_kv_group;

            for (q_head_start..q_head_end) |head_index| {
                const query_head = query_buffer[head_index * head_dim ..][0..head_dim];
                const scores_for_head = scores_base[head_index * max_seq_len ..][0..kv_sequence_len];

                // Q·K dot products
                var max_score: f32 = -std.math.inf(f32);
                for (0..start_kv_index) |key_index| {
                    scores_for_head[key_index] = -std.math.inf(f32);
                }
                for (start_kv_index..kv_sequence_len) |key_index| {
                    const k_row = k_cache_head[key_index * head_dim ..][0..head_dim];

                    // SIMD dot product
                    var sum0: F32Vec = @splat(0);
                    var sum1: F32Vec = @splat(0);
                    var dimension_index: usize = 0;
                    while (dimension_index + 2 * VEC_LEN - 1 < head_dim) : (dimension_index += 2 * VEC_LEN) {
                        const q_vec0: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                        const k_vec0: F32Vec = k_row[dimension_index..][0..VEC_LEN].*;
                        const q_vec1: F32Vec = query_head[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                        const k_vec1: F32Vec = k_row[dimension_index + VEC_LEN ..][0..VEC_LEN].*;
                        sum0 = @mulAdd(F32Vec, q_vec0, k_vec0, sum0);
                        sum1 = @mulAdd(F32Vec, q_vec1, k_vec1, sum1);
                    }
                    while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                        const q_vec: F32Vec = query_head[dimension_index..][0..VEC_LEN].*;
                        const k_vec: F32Vec = k_row[dimension_index..][0..VEC_LEN].*;
                        sum0 = @mulAdd(F32Vec, q_vec, k_vec, sum0);
                    }
                    var dot = @reduce(.Add, sum0 + sum1);
                    while (dimension_index < head_dim) : (dimension_index += 1) {
                        dot += query_head[dimension_index] * k_row[dimension_index];
                    }
                    dot *= scale;
                    scores_for_head[key_index] = dot;
                    if (dot > max_score) max_score = dot;
                }

                // Softmax
                const sink_logit: ?f32 = if (self.sinks) |s| s[head_index] else null;
                if (sink_logit) |sl| {
                    if (sl > max_score) max_score = sl;
                }
                ops.softmaxMaskedInPlaceWithMax(scores_for_head, start_kv_index, kv_sequence_len, sink_logit, false, max_score, null);

                // Context accumulation
                const context_for_head = context_base[head_index * head_dim ..][0..head_dim];
                @memset(context_for_head, 0);

                for (0..kv_sequence_len) |kv_index| {
                    const attn_weight = scores_for_head[kv_index];
                    const v_row = v_cache_head[kv_index * head_dim ..][0..head_dim];

                    const weight_vec: F32Vec = @splat(attn_weight);
                    var dimension_index: usize = 0;
                    while (dimension_index + VEC_LEN - 1 < head_dim) : (dimension_index += VEC_LEN) {
                        const v_vec: F32Vec = v_row[dimension_index..][0..VEC_LEN].*;
                        const out_slice = context_for_head[dimension_index..][0..VEC_LEN];
                        out_slice.* = @mulAdd(F32Vec, weight_vec, v_vec, out_slice.*);
                    }
                    while (dimension_index < head_dim) : (dimension_index += 1) {
                        context_for_head[dimension_index] += attn_weight * v_row[dimension_index];
                    }
                }
            }
        }

        // 4. Output projection
        const context_view = Tensor.view2DSlice(context_base, 1, query_dim);
        var out_view = Tensor.view2DSlice(request.output_values, 1, self.d_model);
        self.matmul_o(&context_view, self.o_proj, &out_view, matmul_scratch);
        if (self.o_bias) |bias| addBias(request.output_values, bias, 1, self.d_model);
    }
}

// =============================================================================
// Unit Tests
// =============================================================================

fn noopFlashAttention(
    _: [*]f32,
    _: usize,
    _: usize,
    _: [*]const f32,
    _: usize,
    _: usize,
    _: [*]const f32,
    _: usize,
    _: usize,
    _: [*]const f32,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: usize,
    _: f32,
    _: usize,
    _: usize,
) void {}

test "MultiHeadAttention.shouldUseFlash rejects non-causal prefill" {
    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);
    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 1024,
        .scale = 0.5,
        .is_causal = false,
        .o_proj = &o_proj,
        .allocator = std.testing.allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
        .flash_attention_fn = noopFlashAttention,
    };

    try std.testing.expect(!mha.shouldUseFlash(false, false, FLASH_ATTENTION_THRESHOLD + 1));
}

test "MultiHeadAttention.forward RoPE position 0 identity" {
    const allocator = std.testing.allocator;

    const dim = 4;
    const max_seq = 100;
    const theta: f32 = 10000.0;

    var rope = try RoPE.init(allocator, dim, max_seq, theta, 1.0);
    defer rope.deinit(allocator);

    // Create a test vector with distinct values
    var vec = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original = vec;

    // Apply RoPE at position 0
    rope.applyInPlace(&vec, 0);

    // At position 0, rotation angles are 0, so we expect:
    // cos(0) = 1, sin(0) = 0
    // rotated = [x1*1 - x2*0, x2*1 + x1*0] = [x1, x2]
    // Should be close to identity
    const epsilon = 1e-5;
    try std.testing.expectApproxEqAbs(original[0], vec[0], epsilon);
    try std.testing.expectApproxEqAbs(original[1], vec[1], epsilon);
    try std.testing.expectApproxEqAbs(original[2], vec[2], epsilon);
    try std.testing.expectApproxEqAbs(original[3], vec[3], epsilon);
}

test "MultiHeadAttention.forward RoPE sin/cos application" {
    const allocator = std.testing.allocator;

    const dim = 4;
    const max_seq = 100;
    const theta: f32 = 10000.0;

    var rope = try RoPE.init(allocator, dim, max_seq, theta, 1.0);
    defer rope.deinit(allocator);

    // Test at position 1 where rotation is small but non-zero
    var vec = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    rope.applyInPlace(&vec, 1);

    // The rotation should preserve vector magnitude in each pair
    const mag1 = @sqrt(vec[0] * vec[0] + vec[2] * vec[2]);
    const mag2 = @sqrt(vec[1] * vec[1] + vec[3] * vec[3]);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mag1, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mag2, 1e-5);
}

test "MultiHeadAttention.forward RoPE position differences" {
    const allocator = std.testing.allocator;

    const dim = 4;
    const max_seq = 100;
    const theta: f32 = 10000.0;

    var rope = try RoPE.init(allocator, dim, max_seq, theta, 1.0);
    defer rope.deinit(allocator);

    var vec1 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var vec2 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    rope.applyInPlace(&vec1, 0);
    rope.applyInPlace(&vec2, 10);

    // Vectors at different positions should differ
    const diff = @abs(vec1[0] - vec2[0]) + @abs(vec1[1] - vec2[1]) +
        @abs(vec1[2] - vec2[2]) + @abs(vec1[3] - vec2[3]);

    try std.testing.expect(diff > 0.01);
}

test "forward simple 1-head scores" {
    const head_dim = 4;

    // Simple Q and K vectors
    const q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const k = [_]f32{ 1.0, 0.0, 0.0, 0.0 };

    // Compute dot product
    var dot: f32 = 0;
    for (0..head_dim) |i| {
        dot += q[i] * k[i];
    }

    try std.testing.expectEqual(@as(f32, 1.0), dot);
}

test "forward orthogonal vectors" {
    const head_dim = 4;

    const q = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const k = [_]f32{ 0.0, 1.0, 0.0, 0.0 };

    var dot: f32 = 0;
    for (0..head_dim) |i| {
        dot += q[i] * k[i];
    }

    try std.testing.expectEqual(@as(f32, 0.0), dot);
}

test "forward scaling sqrt_dk" {
    const head_dim: f32 = 4.0;
    const scale = 1.0 / @sqrt(head_dim);

    const q = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const k = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    var dot: f32 = 0;
    for (0..@as(usize, @intFromFloat(head_dim))) |i| {
        dot += q[i] * k[i];
    }
    dot *= scale;

    // Expected: 4.0 / sqrt(4.0) = 4.0 / 2.0 = 2.0
    try std.testing.expectEqual(@as(f32, 2.0), dot);
}

test "forward softmax sums to 1" {
    var scores = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    // Find max for numerical stability
    var max_score: f32 = -std.math.inf(f32);
    for (scores) |s| {
        if (s > max_score) max_score = s;
    }

    // Compute softmax manually
    var sum: f32 = 0;
    for (&scores) |*s| {
        const exp_val = @exp(s.* - max_score);
        s.* = exp_val;
        sum += exp_val;
    }
    for (&scores) |*s| {
        s.* /= sum;
    }

    // Verify sum is 1.0
    var total: f32 = 0;
    for (scores) |s| {
        total += s;
    }

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-6);
}

test "forward softmax numerical stability" {
    var scores = [_]f32{ 1000.0, 1001.0, 1002.0 };

    var max_score: f32 = -std.math.inf(f32);
    for (scores) |s| {
        if (s > max_score) max_score = s;
    }

    // Subtract max before exp to avoid overflow
    var sum: f32 = 0;
    for (&scores) |*s| {
        const exp_val = @exp(s.* - max_score);
        s.* = exp_val;
        sum += exp_val;
    }
    for (&scores) |*s| {
        s.* /= sum;
    }

    // Result should be valid (not NaN or Inf)
    for (scores) |s| {
        try std.testing.expect(!std.math.isNan(s));
        try std.testing.expect(!std.math.isInf(s));
    }

    // Should still sum to 1.0
    var total: f32 = 0;
    for (scores) |s| {
        total += s;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);
}

test "forward softmax causal mask" {
    const query_index = 1; // Current query position
    const sequence_len = 4;

    var scores = [_]f32{ 1.0, 2.0, -std.math.inf(f32), -std.math.inf(f32) };

    // Positions > query_index should be -inf and become 0 after softmax
    var max_score: f32 = -std.math.inf(f32);
    for (scores[0 .. query_index + 1]) |s| {
        if (s > max_score) max_score = s;
    }

    var sum: f32 = 0;
    for (0..sequence_len) |i| {
        if (i <= query_index) {
            const exp_val = @exp(scores[i] - max_score);
            scores[i] = exp_val;
            sum += exp_val;
        } else {
            scores[i] = 0;
        }
    }
    for (&scores) |*s| {
        s.* /= sum;
    }

    // Future positions should be 0
    try std.testing.expectEqual(@as(f32, 0.0), scores[2]);
    try std.testing.expectEqual(@as(f32, 0.0), scores[3]);

    // Valid positions should sum to 1.0
    var total: f32 = 0;
    for (scores) |s| {
        total += s;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-6);
}

test "forward output scores V" {
    const head_dim = 4;

    // Simple case: uniform attention weights
    const scores = [_]f32{ 0.5, 0.5 };
    const v1 = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const v2 = [_]f32{ 0.0, 1.0, 0.0, 0.0 };

    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    // Compute weighted sum
    for (0..head_dim) |i| {
        output[i] += scores[0] * v1[i];
        output[i] += scores[1] * v2[i];
    }

    // Expected: [0.5, 0.5, 0.0, 0.0]
    try std.testing.expectEqual(@as(f32, 0.5), output[0]);
    try std.testing.expectEqual(@as(f32, 0.5), output[1]);
    try std.testing.expectEqual(@as(f32, 0.0), output[2]);
    try std.testing.expectEqual(@as(f32, 0.0), output[3]);
}

test "forward output single weight" {
    const head_dim = 4;

    // All attention on first value
    const scores = [_]f32{ 1.0, 0.0 };
    const v1 = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const v2 = [_]f32{ 6.0, 7.0, 8.0, 9.0 };

    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    for (0..head_dim) |i| {
        output[i] += scores[0] * v1[i];
        output[i] += scores[1] * v2[i];
    }

    // Should match v1 exactly
    try std.testing.expectEqual(@as(f32, 2.0), output[0]);
    try std.testing.expectEqual(@as(f32, 3.0), output[1]);
    try std.testing.expectEqual(@as(f32, 4.0), output[2]);
    try std.testing.expectEqual(@as(f32, 5.0), output[3]);
}

test "forward cache write read" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    const n_kv_heads = 2;
    const head_dim = 4;
    const kv_total_dim = n_kv_heads * head_dim;

    // Allocate cache for 10 positions
    const capacity = 10;
    cache.kv_capacity = capacity;
    cache.key_cache = try allocator.alloc(f32, capacity * kv_total_dim);
    cache.value_cache = try allocator.alloc(f32, capacity * kv_total_dim);
    @memset(cache.key_cache, 0);
    @memset(cache.value_cache, 0);

    // Write K/V at position 0
    const pos = 0;
    for (0..n_kv_heads) |kv_h| {
        for (0..head_dim) |d| {
            const src_idx = kv_h * head_dim + d;
            const dst_idx = kv_h * capacity * head_dim + pos * head_dim + d;
            cache.key_cache[dst_idx] = @floatFromInt(src_idx);
            cache.value_cache[dst_idx] = @floatFromInt(src_idx + 100);
        }
    }

    // Read back and verify
    for (0..n_kv_heads) |kv_h| {
        for (0..head_dim) |d| {
            const src_idx = kv_h * head_dim + d;
            const dst_idx = kv_h * capacity * head_dim + pos * head_dim + d;
            try std.testing.expectEqual(@as(f32, @floatFromInt(src_idx)), cache.key_cache[dst_idx]);
            try std.testing.expectEqual(@as(f32, @floatFromInt(src_idx + 100)), cache.value_cache[dst_idx]);
        }
    }
}

test "forward cache sequential" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    const n_kv_heads = 1;
    const head_dim = 4;
    const kv_total_dim = n_kv_heads * head_dim;

    const capacity = 5;
    cache.kv_capacity = capacity;
    cache.key_cache = try allocator.alloc(f32, capacity * kv_total_dim);
    cache.value_cache = try allocator.alloc(f32, capacity * kv_total_dim);
    @memset(cache.key_cache, 0);
    @memset(cache.value_cache, 0);

    // Write multiple positions
    for (0..3) |pos| {
        for (0..head_dim) |d| {
            const dst_idx = pos * head_dim + d;
            cache.key_cache[dst_idx] = @floatFromInt(pos * 10 + d);
            cache.value_cache[dst_idx] = @floatFromInt(pos * 10 + d + 1000);
        }
    }

    // Verify each position
    for (0..3) |pos| {
        for (0..head_dim) |d| {
            const dst_idx = pos * head_dim + d;
            try std.testing.expectEqual(@as(f32, @floatFromInt(pos * 10 + d)), cache.key_cache[dst_idx]);
            try std.testing.expectEqual(@as(f32, @floatFromInt(pos * 10 + d + 1000)), cache.value_cache[dst_idx]);
        }
    }
}

test "resetCache functionality" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    cache.cache_position = 10;
    cache.resetCache();

    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);
}

test "forward GQA head indexing" {
    // Test that query heads map to correct KV head
    const n_heads = 4;
    const n_kv_heads = 2;
    const heads_per_kv_group = n_heads / n_kv_heads;

    try std.testing.expectEqual(@as(usize, 2), heads_per_kv_group);

    // Query heads 0,1 -> KV head 0
    // Query heads 2,3 -> KV head 1
    for (0..n_heads) |query_head| {
        const kv_head = query_head / heads_per_kv_group;
        if (query_head < 2) {
            try std.testing.expectEqual(@as(usize, 0), kv_head);
        } else {
            try std.testing.expectEqual(@as(usize, 1), kv_head);
        }
    }
}

test "forward GQA kv sharing" {
    // Verify that multiple Q heads can share the same K/V
    const n_heads = 4;
    const n_kv_heads = 2;
    const heads_per_kv_group = n_heads / n_kv_heads;

    // Q heads in same group share KV head
    const q_head_0 = 0;
    const q_head_1 = 1;
    const kv_head_0 = q_head_0 / heads_per_kv_group;
    const kv_head_1 = q_head_1 / heads_per_kv_group;

    try std.testing.expectEqual(kv_head_0, kv_head_1);
}

test "forward sliding window mask" {
    const sliding_window = 3;
    const kv_sequence_len = 10;

    // Calculate window start
    const start_kv_index: usize = if (kv_sequence_len > sliding_window)
        kv_sequence_len - sliding_window
    else
        0;

    try std.testing.expectEqual(@as(usize, 7), start_kv_index);

    // Positions 0-6 should be masked (outside window)
    // Positions 7-9 should be visible
    for (0..kv_sequence_len) |pos| {
        if (pos < start_kv_index) {
            // Should be masked
            try std.testing.expect(pos < 7);
        } else {
            // Should be visible
            try std.testing.expect(pos >= 7);
        }
    }
}

test "forward sliding window short seq" {
    const sliding_window = 10;
    const kv_sequence_len = 5;

    const start_kv_index: usize = if (kv_sequence_len > sliding_window)
        kv_sequence_len - sliding_window
    else
        0;

    try std.testing.expectEqual(@as(usize, 0), start_kv_index);
}

test "forward sliding window causal" {
    const sliding_window = 3;
    const query_index = 5;

    // Window start for this query position
    const start_kv_index: usize = if ((query_index + 1) > sliding_window)
        (query_index + 1) - sliding_window
    else
        0;

    // Should only see positions 3, 4, 5
    try std.testing.expectEqual(@as(usize, 3), start_kv_index);

    // Valid range: [start_kv_index, query_index]
    const valid_positions = query_index - start_kv_index + 1;
    try std.testing.expectEqual(@as(usize, 3), valid_positions);
}

test "forward multi-head slicing" {
    const n_heads = 2;
    const head_dim = 4;
    const query_dim = n_heads * head_dim;

    // Verify head offsets
    for (0..n_heads) |head_index| {
        const offset = head_index * head_dim;
        const expected_offset = head_index * 4;
        try std.testing.expectEqual(expected_offset, offset);
    }

    try std.testing.expectEqual(@as(usize, 8), query_dim);
}

test "forward multi-head independent" {
    const n_heads = 2;
    const head_dim = 4;

    // Each head should have independent Q, K, V slices
    // Head 0: indices 0-3
    // Head 1: indices 4-7
    var data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };

    for (0..n_heads) |head_index| {
        const start = head_index * head_dim;
        const end = start + head_dim;
        const head_slice = data[start..end];

        // Verify slice boundaries
        if (head_index == 0) {
            try std.testing.expectEqual(@as(f32, 1), head_slice[0]);
            try std.testing.expectEqual(@as(f32, 4), head_slice[3]);
        } else {
            try std.testing.expectEqual(@as(f32, 5), head_slice[0]);
            try std.testing.expectEqual(@as(f32, 8), head_slice[3]);
        }
    }
}

test "MultiHeadAttention.forward addBias single token" {
    const sequence_len = 1;
    const dim = 4;

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const bias = [_]f32{ 0.1, 0.2, 0.3, 0.4 };

    addBias(&data, &bias, sequence_len, dim);

    try std.testing.expectApproxEqAbs(@as(f32, 1.1), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.2), data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.3), data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.4), data[3], 1e-6);
}

test "MultiHeadAttention.forward addBias multiple tokens" {
    const sequence_len = 2;
    const dim = 3;

    var data = [_]f32{
        1.0, 2.0, 3.0, // Token 0
        4.0, 5.0, 6.0, // Token 1
    };
    const bias = [_]f32{ 0.5, 1.0, 1.5 };

    addBias(&data, &bias, sequence_len, dim);

    // Token 0
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), data[2], 1e-6);

    // Token 1
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), data[5], 1e-6);
}

test "init BatchedKVCache slot management" {
    const allocator = std.testing.allocator;

    var cache = try BatchedKVCache.init(allocator, 4, 2, 8, 64);
    defer cache.deinit();

    try std.testing.expectEqual(@as(usize, 4), cache.max_batch_size);
    try std.testing.expectEqual(@as(usize, 2), cache.n_kv_heads);
    try std.testing.expectEqual(@as(usize, 8), cache.head_dim);
    try std.testing.expectEqual(@as(usize, 64), cache.max_seq_len);

    // All slots should start inactive
    for (cache.slots) |slot| {
        try std.testing.expectEqual(false, slot.active);
        try std.testing.expectEqual(@as(usize, 0), slot.position);
    }
}

test "init BatchedKVCache position tracking" {
    const allocator = std.testing.allocator;

    var cache = try BatchedKVCache.init(allocator, 2, 1, 4, 32);
    defer cache.deinit();

    // Manually set a position
    cache.slots[0].position = 5;
    cache.slots[0].active = true;

    const pos = cache.getPosition(0);
    try std.testing.expectEqual(@as(usize, 5), pos);
}

// =============================================================================
// AttnTemp Tests
// =============================================================================

test "AttnTemp: initialization and deinit" {
    const allocator = std.testing.allocator;

    var scratch = AttnTemp{};
    defer scratch.deinit(allocator);

    // Initially all slices should be empty
    try std.testing.expectEqual(@as(usize, 0), scratch.q.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.k.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.v.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.qkv.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.scores.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.context_values.len);
}

test "AttnTemp: deinit with allocated memory" {
    const allocator = std.testing.allocator;

    var scratch = AttnTemp{
        .q = try allocator.alloc(f32, 10),
        .k = try allocator.alloc(f32, 10),
        .v = try allocator.alloc(f32, 10),
        .qkv = try allocator.alloc(f32, 30),
        .scores = try allocator.alloc(f32, 20),
        .context_values = try allocator.alloc(f32, 15),
    };

    // Verify allocations succeeded
    try std.testing.expectEqual(@as(usize, 10), scratch.q.len);
    try std.testing.expectEqual(@as(usize, 10), scratch.k.len);
    try std.testing.expectEqual(@as(usize, 10), scratch.v.len);
    try std.testing.expectEqual(@as(usize, 30), scratch.qkv.len);
    try std.testing.expectEqual(@as(usize, 20), scratch.scores.len);
    try std.testing.expectEqual(@as(usize, 15), scratch.context_values.len);

    // Deinit should free all memory
    scratch.deinit(allocator);

    // After deinit, all should be empty
    try std.testing.expectEqual(@as(usize, 0), scratch.q.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.k.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.v.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.qkv.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.scores.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.context_values.len);
}

test "AttnTemp: multiple deinit is safe" {
    const allocator = std.testing.allocator;

    var scratch = AttnTemp{
        .q = try allocator.alloc(f32, 5),
    };

    // First deinit
    scratch.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), scratch.q.len);

    // Second deinit should be safe (no-op)
    scratch.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), scratch.q.len);
}

// =============================================================================
// AttnCache Tests
// =============================================================================

test "AttnCache: initialization and deinit" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    // Initially all fields should be zero/empty
    try std.testing.expectEqual(@as(usize, 0), cache.key_cache.len);
    try std.testing.expectEqual(@as(usize, 0), cache.value_cache.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_capacity);
    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);
}

test "AttnCache: deinit with allocated memory" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{
        .key_cache = try allocator.alloc(f32, 100),
        .value_cache = try allocator.alloc(f32, 100),
        .kv_capacity = 10,
        .cache_position = 5,
    };

    // Verify allocations
    try std.testing.expectEqual(@as(usize, 100), cache.key_cache.len);
    try std.testing.expectEqual(@as(usize, 100), cache.value_cache.len);
    try std.testing.expectEqual(@as(usize, 10), cache.kv_capacity);
    try std.testing.expectEqual(@as(usize, 5), cache.cache_position);

    // Deinit should free memory and reset fields
    cache.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), cache.key_cache.len);
    try std.testing.expectEqual(@as(usize, 0), cache.value_cache.len);
    try std.testing.expectEqual(@as(usize, 0), cache.kv_capacity);
    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);
}

test "AttnCache: resetCache resets position only" {
    const allocator = std.testing.allocator;

    var cache = AttnCache{
        .key_cache = try allocator.alloc(f32, 50),
        .value_cache = try allocator.alloc(f32, 50),
        .kv_capacity = 10,
        .cache_position = 7,
    };
    defer cache.deinit(allocator);

    // Reset cache
    cache.resetCache();

    // Position should be reset to 0
    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);

    // But memory should still be allocated
    try std.testing.expectEqual(@as(usize, 50), cache.key_cache.len);
    try std.testing.expectEqual(@as(usize, 50), cache.value_cache.len);
    try std.testing.expectEqual(@as(usize, 10), cache.kv_capacity);
}

test "resetCache multiple calls" {
    var cache = AttnCache{ .cache_position = 15 };

    cache.resetCache();
    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);

    cache.cache_position = 20;
    cache.resetCache();
    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);

    // Multiple resets should be safe
    cache.resetCache();
    try std.testing.expectEqual(@as(usize, 0), cache.cache_position);
}

// =============================================================================
// BatchedAttnTemp Tests
// =============================================================================

test "BatchedAttnTemp: init allocates correct sizes" {
    const allocator = std.testing.allocator;

    const max_batch = 4;
    const n_heads = 8;
    const n_kv_heads = 4;
    const head_dim = 64;
    const max_seq_len = 512;

    var scratch = try BatchedAttnTemp.init(allocator, max_batch, n_heads, n_kv_heads, head_dim, max_seq_len);
    defer scratch.deinit();

    const query_dim = n_heads * head_dim;
    const kv_total_dim = n_kv_heads * head_dim;

    try std.testing.expectEqual(max_batch, scratch.max_batch_size);
    try std.testing.expectEqual(max_batch * query_dim, scratch.q.len);
    try std.testing.expectEqual(max_batch * kv_total_dim, scratch.k.len);
    try std.testing.expectEqual(max_batch * kv_total_dim, scratch.v.len);
    try std.testing.expectEqual(2 * max_batch * (query_dim + 2 * kv_total_dim), scratch.qkv.len);
    try std.testing.expectEqual(max_batch * n_heads * max_seq_len, scratch.scores.len);
    try std.testing.expectEqual(max_batch * query_dim, scratch.context_values.len);
}

test "BatchedAttnTemp: init with minimal parameters" {
    const allocator = std.testing.allocator;

    var scratch = try BatchedAttnTemp.init(allocator, 1, 1, 1, 1, 1);
    defer scratch.deinit();

    try std.testing.expectEqual(@as(usize, 1), scratch.max_batch_size);
    try std.testing.expectEqual(@as(usize, 1), scratch.q.len);
    try std.testing.expectEqual(@as(usize, 1), scratch.k.len);
    try std.testing.expectEqual(@as(usize, 1), scratch.v.len);
}

test "BatchedAttnTemp: deinit frees all memory" {
    const allocator = std.testing.allocator;

    var scratch = try BatchedAttnTemp.init(allocator, 2, 4, 2, 32, 128);

    // Verify memory is allocated
    try std.testing.expect(scratch.q.len > 0);
    try std.testing.expect(scratch.k.len > 0);
    try std.testing.expect(scratch.v.len > 0);
    try std.testing.expect(scratch.qkv.len > 0);
    try std.testing.expect(scratch.scores.len > 0);
    try std.testing.expect(scratch.context_values.len > 0);

    // deinit frees memory and sets struct to undefined
    // std.testing.allocator will detect leaks if deinit fails to free
    scratch.deinit();
}

test "BatchedAttnTemp: getQ returns correct slice" {
    const allocator = std.testing.allocator;

    const max_batch = 3;
    const n_heads = 4;
    const head_dim = 8;
    const query_dim = n_heads * head_dim;

    var scratch = try BatchedAttnTemp.init(allocator, max_batch, n_heads, 2, head_dim, 64);
    defer scratch.deinit();

    // Initialize data
    for (scratch.q, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Get Q buffer for batch index 0
    const q0 = scratch.getQ(0, query_dim);
    try std.testing.expectEqual(query_dim, q0.len);
    try std.testing.expectEqual(@as(f32, 0), q0[0]);
    try std.testing.expectEqual(@as(f32, @floatFromInt(query_dim - 1)), q0[query_dim - 1]);

    // Get Q buffer for batch index 1
    const q1 = scratch.getQ(1, query_dim);
    try std.testing.expectEqual(query_dim, q1.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(query_dim)), q1[0]);

    // Get Q buffer for batch index 2
    const q2 = scratch.getQ(2, query_dim);
    try std.testing.expectEqual(query_dim, q2.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(2 * query_dim)), q2[0]);
}

test "BatchedAttnTemp: getK returns correct slice" {
    const allocator = std.testing.allocator;

    const max_batch = 2;
    const n_kv_heads = 2;
    const head_dim = 4;
    const kv_total_dim = n_kv_heads * head_dim;

    var scratch = try BatchedAttnTemp.init(allocator, max_batch, 4, n_kv_heads, head_dim, 32);
    defer scratch.deinit();

    // Initialize K data
    for (scratch.k, 0..) |*val, i| {
        val.* = @floatFromInt(i + 100);
    }

    // Get K buffer for batch index 0
    const k0 = scratch.getK(0, kv_total_dim);
    try std.testing.expectEqual(kv_total_dim, k0.len);
    try std.testing.expectEqual(@as(f32, 100), k0[0]);

    // Get K buffer for batch index 1
    const k1 = scratch.getK(1, kv_total_dim);
    try std.testing.expectEqual(kv_total_dim, k1.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(100 + kv_total_dim)), k1[0]);
}

test "BatchedAttnTemp: getV returns correct slice" {
    const allocator = std.testing.allocator;

    const max_batch = 2;
    const n_kv_heads = 2;
    const head_dim = 4;
    const kv_total_dim = n_kv_heads * head_dim;

    var scratch = try BatchedAttnTemp.init(allocator, max_batch, 4, n_kv_heads, head_dim, 32);
    defer scratch.deinit();

    // Initialize V data
    for (scratch.v, 0..) |*val, i| {
        val.* = @floatFromInt(i + 200);
    }

    // Get V buffer for batch index 0
    const v0 = scratch.getV(0, kv_total_dim);
    try std.testing.expectEqual(kv_total_dim, v0.len);
    try std.testing.expectEqual(@as(f32, 200), v0[0]);

    // Get V buffer for batch index 1
    const v1 = scratch.getV(1, kv_total_dim);
    try std.testing.expectEqual(kv_total_dim, v1.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(200 + kv_total_dim)), v1[0]);
}

test "BatchedAttnTemp: getScores returns correct slice" {
    const allocator = std.testing.allocator;

    const max_batch = 2;
    const n_heads = 4;
    const max_seq_len = 128;
    const stride = n_heads * max_seq_len;

    var scratch = try BatchedAttnTemp.init(allocator, max_batch, n_heads, 2, 8, max_seq_len);
    defer scratch.deinit();

    // Initialize scores data
    for (scratch.scores, 0..) |*val, i| {
        val.* = @floatFromInt(i + 300);
    }

    // Get scores for batch index 0
    const scores0 = scratch.getScores(0, n_heads, max_seq_len);
    try std.testing.expectEqual(stride, scores0.len);
    try std.testing.expectEqual(@as(f32, 300), scores0[0]);

    // Get scores for batch index 1
    const scores1 = scratch.getScores(1, n_heads, max_seq_len);
    try std.testing.expectEqual(stride, scores1.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(300 + stride)), scores1[0]);
}

test "BatchedAttnTemp: getContext returns correct slice" {
    const allocator = std.testing.allocator;

    const max_batch = 3;
    const n_heads = 4;
    const head_dim = 16;
    const query_dim = n_heads * head_dim;

    var scratch = try BatchedAttnTemp.init(allocator, max_batch, n_heads, 2, head_dim, 64);
    defer scratch.deinit();

    // Initialize context data
    for (scratch.context_values, 0..) |*val, i| {
        val.* = @floatFromInt(i + 400);
    }

    // Get context for batch index 0
    const ctx0 = scratch.getContext(0, query_dim);
    try std.testing.expectEqual(query_dim, ctx0.len);
    try std.testing.expectEqual(@as(f32, 400), ctx0[0]);

    // Get context for batch index 1
    const ctx1 = scratch.getContext(1, query_dim);
    try std.testing.expectEqual(query_dim, ctx1.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(400 + query_dim)), ctx1[0]);

    // Get context for batch index 2
    const ctx2 = scratch.getContext(2, query_dim);
    try std.testing.expectEqual(query_dim, ctx2.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(400 + 2 * query_dim)), ctx2[0]);
}

test "getQ getK getV accessor methods" {
    const allocator = std.testing.allocator;

    var scratch = try BatchedAttnTemp.init(allocator, 2, 2, 1, 4, 16);
    defer scratch.deinit();

    const query_dim = 2 * 4;
    const kv_dim = 1 * 4;

    // Get and modify Q
    const q0 = scratch.getQ(0, query_dim);
    q0[0] = 1.5;
    q0[7] = 2.5;

    // Verify modification persists
    const q0_again = scratch.getQ(0, query_dim);
    try std.testing.expectEqual(@as(f32, 1.5), q0_again[0]);
    try std.testing.expectEqual(@as(f32, 2.5), q0_again[7]);

    // Get and modify K
    const k1 = scratch.getK(1, kv_dim);
    k1[0] = 3.5;

    // Verify modification persists
    const k1_again = scratch.getK(1, kv_dim);
    try std.testing.expectEqual(@as(f32, 3.5), k1_again[0]);
}

// =============================================================================
// ensureTemp Tests
// =============================================================================

test "MultiHeadAttention.ensureTemp: allocates buffers on first call" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 64,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var scratch = AttnTemp{};
    defer scratch.deinit(allocator);

    const sequence_len = 2;
    const query_dim = 8;
    const kv_total_dim = 8;

    try mha.ensureTemp(&scratch, sequence_len, false, query_dim, kv_total_dim, false);

    // Verify buffers are allocated
    try std.testing.expectEqual(sequence_len * query_dim, scratch.q.len);
    try std.testing.expectEqual(sequence_len * kv_total_dim, scratch.k.len);
    try std.testing.expectEqual(sequence_len * kv_total_dim, scratch.v.len);
    try std.testing.expectEqual(@as(usize, 0), scratch.qkv.len); // Not using fused
    try std.testing.expect(scratch.scores.len > 0);
    try std.testing.expect(scratch.context_values.len > 0);
}

test "MultiHeadAttention.ensureTemp: reuses buffers when large enough" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 64,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var scratch = AttnTemp{};
    defer scratch.deinit(allocator);

    // First call allocates
    try mha.ensureTemp(&scratch, 2, false, 8, 8, false);
    const q_ptr = scratch.q.ptr;

    // Second call with same or smaller size should reuse
    try mha.ensureTemp(&scratch, 2, false, 8, 8, false);
    try std.testing.expectEqual(q_ptr, scratch.q.ptr);
}

test "MultiHeadAttention.ensureTemp: reallocates when size increases" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 64,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var scratch = AttnTemp{};
    defer scratch.deinit(allocator);

    // Allocate for small size
    try mha.ensureTemp(&scratch, 2, false, 8, 8, false);
    const initial_len = scratch.q.len;

    // Allocate for larger size
    try mha.ensureTemp(&scratch, 10, false, 8, 8, false);
    const new_len = scratch.q.len;

    try std.testing.expect(new_len > initial_len);
}

test "MultiHeadAttention.ensureTemp: allocates fused qkv buffer when needed" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 64,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var scratch = AttnTemp{};
    defer scratch.deinit(allocator);

    const sequence_len = 2;
    const query_dim = 8;
    const kv_total_dim = 8;

    // Use fused QKV
    try mha.ensureTemp(&scratch, sequence_len, false, query_dim, kv_total_dim, true);

    // QKV buffer should be allocated (2x size for temporary + rearranged)
    const expected_qkv_size = 2 * sequence_len * (query_dim + 2 * kv_total_dim);
    try std.testing.expectEqual(expected_qkv_size, scratch.qkv.len);
}

test "MultiHeadAttention.ensureTemp: decode mode allocates correct scores size" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const max_seq = 128;
    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = max_seq,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var scratch = AttnTemp{};
    defer scratch.deinit(allocator);

    // Decode mode (use_cache = true)
    try mha.ensureTemp(&scratch, 1, true, 16, 8, false);

    // Scores should be sized for n_heads * max_seq_len
    const expected_scores = 4 * max_seq;
    try std.testing.expectEqual(expected_scores, scratch.scores.len);
}

// =============================================================================
// ensureKvCapacity Tests
// =============================================================================

test "MultiHeadAttention.ensureKvCapacity: initial allocation" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 128,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    const needed_seq = 10;
    const kv_total_dim = 8;

    try mha.ensureKvCapacity(&cache, needed_seq, kv_total_dim);

    // Verify allocation
    try std.testing.expect(cache.kv_capacity >= needed_seq);
    try std.testing.expectEqual(cache.kv_capacity * kv_total_dim, cache.key_cache.len);
    try std.testing.expectEqual(cache.kv_capacity * kv_total_dim, cache.value_cache.len);
}

test "MultiHeadAttention.ensureKvCapacity: no reallocation when sufficient" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 128,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    // Initial allocation
    try mha.ensureKvCapacity(&cache, 20, 8);
    const initial_ptr = cache.key_cache.ptr;
    const initial_capacity = cache.kv_capacity;

    // Request smaller size - should not reallocate
    try mha.ensureKvCapacity(&cache, 10, 8);
    try std.testing.expectEqual(initial_ptr, cache.key_cache.ptr);
    try std.testing.expectEqual(initial_capacity, cache.kv_capacity);
}

test "MultiHeadAttention.ensureKvCapacity: grows capacity exponentially" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 128,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    // Initial allocation
    try mha.ensureKvCapacity(&cache, 10, 8);
    const first_capacity = cache.kv_capacity;

    // Request larger size - should grow
    try mha.ensureKvCapacity(&cache, 25, 8);
    const second_capacity = cache.kv_capacity;

    // Capacity should have grown (doubled from first)
    try std.testing.expect(second_capacity > first_capacity);
    try std.testing.expect(second_capacity >= 25);
}

test "MultiHeadAttention.ensureKvCapacity: respects max_seq_len limit" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const max_seq = 32;
    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = max_seq,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    // Request size larger than max - should cap at max
    try mha.ensureKvCapacity(&cache, 1000, 8);

    // Capacity should not exceed max_seq_len
    try std.testing.expect(cache.kv_capacity <= max_seq);
}

test "MultiHeadAttention.ensureKvCapacity: preserves cached data during growth" {
    const allocator = std.testing.allocator;

    var o_proj_data = [_]f32{0} ** 16;
    const o_proj = Tensor.view2DSlice(&o_proj_data, 4, 4);

    const mha = MultiHeadAttention{
        .d_model = 8,
        .n_heads = 2,
        .n_kv_heads = 2,
        .head_dim = 4,
        .max_seq_len = 128,
        .scale = 0.5,
        .o_proj = &o_proj,
        .allocator = allocator,
        .matmul_qkv = undefined,
        .matmul_o = undefined,
    };

    var cache = AttnCache{};
    defer cache.deinit(allocator);

    // Initial allocation and populate some data
    try mha.ensureKvCapacity(&cache, 5, 8);
    cache.cache_position = 2;

    // Write test data at position 0
    const head_dim = 4;
    for (0..2) |kv_h| {
        for (0..head_dim) |d| {
            const idx = kv_h * cache.kv_capacity * head_dim + 0 * head_dim + d;
            cache.key_cache[idx] = @floatFromInt(kv_h * 10 + d);
            cache.value_cache[idx] = @floatFromInt(kv_h * 10 + d + 100);
        }
    }

    // Force reallocation by requesting larger size
    try mha.ensureKvCapacity(&cache, 50, 8);

    // Verify data was preserved
    for (0..2) |kv_h| {
        for (0..head_dim) |d| {
            const idx = kv_h * cache.kv_capacity * head_dim + 0 * head_dim + d;
            try std.testing.expectEqual(@as(f32, @floatFromInt(kv_h * 10 + d)), cache.key_cache[idx]);
            try std.testing.expectEqual(@as(f32, @floatFromInt(kv_h * 10 + d + 100)), cache.value_cache[idx]);
        }
    }

    // Cache position should be preserved
    try std.testing.expectEqual(@as(usize, 2), cache.cache_position);
}

test "applyPositionDelta applies negative multimodal offset" {
    try std.testing.expectEqual(@as(usize, 42), try applyPositionDelta(198, -156));
    try std.testing.expectEqual(@as(usize, 43), try applyPositionDelta(199, -156));
}

test "applyPositionDelta rejects negative resulting positions" {
    try std.testing.expectError(error.InvalidShape, applyPositionDelta(3, -4));
}

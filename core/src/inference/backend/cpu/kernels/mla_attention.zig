//! MLA (Multi-Latent Attention) Kernel
//!
//! Implements DeepSeek-V2 style compressed attention with low-rank Q/KV projections.
//! Used by Youtu-VL, DeepSeek-V2, DeepSeek-V3, and other MLA-based models.
//!
//! Key differences from standard attention:
//! - Q projection: hidden → q_lora_rank → norm → n_heads * qk_head_dim
//! - KV projection: hidden → (kv_lora_rank + qk_rope_head_dim)
//! - K is split into nope (per-head) and rope (shared across heads) portions
//! - RoPE only applied to qk_rope_head_dim dimensions
//! - V has separate dimension (v_head_dim) from K (qk_nope_head_dim)

const std = @import("std");
const tensor = @import("../../../../tensor.zig");
const compute = @import("../../../../compute/root.zig");
const matmul = compute.ops.matmul;
const ops = compute.ops.math;
const simd = compute.simd;
const rope_kernel = @import("rope.zig");
const trace = @import("../../../../xray/root.zig").trace;

const Tensor = tensor.Tensor;
const MatmulFn = matmul.MatmulFn;
const RoPE = rope_kernel.RoPE;

const VEC_LEN = simd.f32_vec_len;
const F32Vec = simd.F32Vec;

/// MLA-specific configuration derived from graph Op.
pub const MLAConfig = struct {
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_head_dim: usize, // qk_rope_head_dim + qk_nope_head_dim
    qk_rope_head_dim: usize,
    qk_nope_head_dim: usize,
    v_head_dim: usize,
    rope_interleave: bool,
};

/// Per-layer KV cache for MLA.
/// Layout differs from standard attention due to split rope/nope K dimensions.
pub const MLACache = struct {
    /// K cache: [n_kv_heads, seq_capacity, qk_head_dim]
    /// Note: qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    key_cache: []f32 = &.{},
    /// V cache: [n_kv_heads, seq_capacity, v_head_dim]
    value_cache: []f32 = &.{},
    /// Rope K cache: [seq_capacity, qk_rope_head_dim] (shared across heads)
    rope_key_cache: []f32 = &.{},
    kv_capacity: usize = 0,
    cache_position: usize = 0,

    pub fn deinit(self: *MLACache, allocator: std.mem.Allocator) void {
        if (self.key_cache.len > 0) allocator.free(self.key_cache);
        if (self.value_cache.len > 0) allocator.free(self.value_cache);
        if (self.rope_key_cache.len > 0) allocator.free(self.rope_key_cache);
        self.* = .{};
    }

    pub fn resetCache(self: *MLACache) void {
        self.cache_position = 0;
    }
};

/// Scratch buffers for MLA computation.
pub const MLATemp = struct {
    /// Q after compression: [seq_len, q_lora_rank]
    q_compressed: []f32 = &.{},
    /// Q after expansion: [seq_len, n_heads * qk_head_dim]
    q: []f32 = &.{},
    /// KV compressed: [seq_len, kv_lora_rank + qk_rope_head_dim]
    kv_compressed: []f32 = &.{},
    /// KV nope after expansion: [seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
    kv_expanded: []f32 = &.{},
    /// Attention scores
    scores: []f32 = &.{},
    /// Context values
    context: []f32 = &.{},

    pub fn deinit(self: *MLATemp, allocator: std.mem.Allocator) void {
        if (self.q_compressed.len > 0) allocator.free(self.q_compressed);
        if (self.q.len > 0) allocator.free(self.q);
        if (self.kv_compressed.len > 0) allocator.free(self.kv_compressed);
        if (self.kv_expanded.len > 0) allocator.free(self.kv_expanded);
        if (self.scores.len > 0) allocator.free(self.scores);
        if (self.context.len > 0) allocator.free(self.context);
        self.* = .{};
    }
};

/// Multi-Latent Attention layer.
pub const MLAttention = struct {
    d_model: usize,
    n_heads: usize,
    max_seq_len: usize,
    config: MLAConfig,
    allocator: std.mem.Allocator,

    // Q projection weights (two-stage compression)
    q_a_proj: *const Tensor, // [q_lora_rank, d_model]
    q_a_norm: *const Tensor, // [q_lora_rank]
    q_b_proj: *const Tensor, // [n_heads * qk_head_dim, q_lora_rank]

    // KV projection weights (compressed + shared rope)
    kv_a_proj: *const Tensor, // [kv_lora_rank + qk_rope_head_dim, d_model]
    kv_a_norm: *const Tensor, // [kv_lora_rank]
    kv_b_proj: *const Tensor, // [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]

    // Output projection
    o_proj: *const Tensor, // [d_model, n_heads * v_head_dim]

    // RoPE (for rope portion only)
    rope: ?*RoPE = null,

    // Norm epsilon
    norm_eps: f32 = 1e-6,

    // Scaling factor (1/sqrt(qk_head_dim))
    scale: f32,

    // Matmul kernels
    matmul_fn: MatmulFn,

    // Layer index for tracing
    layer_idx: u16 = 0,

    pub fn forward(
        self: *const MLAttention,
        input_tensor: *const Tensor,
        output_tensor: *Tensor,
        cache: *MLACache,
        scratch: *MLATemp,
        matmul_scratch: *matmul.MatmulScratch,
        use_cache: bool,
    ) !void {
        const seq_len: usize = @intCast(input_tensor.shape[1]);
        const cfg = self.config;

        try self.ensureTemp(scratch, seq_len, use_cache);

        // Flatten input for matmul
        const input_view = Tensor.view2D(input_tensor.data(), seq_len, self.d_model);

        // === Q projection (two-stage) ===
        // Step 1: hidden → q_lora_rank
        var q_compressed_view = Tensor.view2DSlice(
            scratch.q_compressed[0 .. seq_len * cfg.q_lora_rank],
            seq_len,
            cfg.q_lora_rank,
        );
        self.matmul_fn(&input_view, self.q_a_proj, &q_compressed_view, matmul_scratch);

        // Step 2: Apply Q norm (RMSNorm per token)
        applyRMSNormTensor(scratch.q_compressed[0 .. seq_len * cfg.q_lora_rank], self.q_a_norm, cfg.q_lora_rank, self.norm_eps);

        // Step 3: q_lora_rank → n_heads * qk_head_dim
        var q_view = Tensor.view2DSlice(
            scratch.q[0 .. seq_len * self.n_heads * cfg.qk_head_dim],
            seq_len,
            self.n_heads * cfg.qk_head_dim,
        );
        self.matmul_fn(&q_compressed_view, self.q_b_proj, &q_view, matmul_scratch);

        // === KV projection ===
        // Step 1: hidden → (kv_lora_rank + qk_rope_head_dim)
        const kv_compressed_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        var kv_compressed_view = Tensor.view2DSlice(
            scratch.kv_compressed[0 .. seq_len * kv_compressed_dim],
            seq_len,
            kv_compressed_dim,
        );
        self.matmul_fn(&input_view, self.kv_a_proj, &kv_compressed_view, matmul_scratch);

        // Split KV into nope_compressed and k_rope
        // Layout: [seq_len, kv_lora_rank | qk_rope_head_dim]
        const kv_compressed_slice = scratch.kv_compressed[0 .. seq_len * kv_compressed_dim];

        // Step 2: Apply KV norm to nope portion only
        // Note: norm is applied to [seq_len, kv_lora_rank], not the rope portion
        for (0..seq_len) |t| {
            const start = t * kv_compressed_dim;
            applyRMSNormSliceTensor(
                kv_compressed_slice[start .. start + cfg.kv_lora_rank],
                self.kv_a_norm,
                self.norm_eps,
            );
        }

        // Step 3: Expand kv_lora_rank → n_heads * (qk_nope_head_dim + v_head_dim)
        const kv_expanded_dim = self.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);

        // Create view of just the nope portion for expansion
        // Need to extract non-contiguous slices - copy to temp buffer
        var nope_temp = try self.allocator.alloc(f32, seq_len * cfg.kv_lora_rank);
        defer self.allocator.free(nope_temp);
        for (0..seq_len) |t| {
            const src_start = t * kv_compressed_dim;
            const dst_start = t * cfg.kv_lora_rank;
            @memcpy(nope_temp[dst_start .. dst_start + cfg.kv_lora_rank], kv_compressed_slice[src_start .. src_start + cfg.kv_lora_rank]);
        }

        var nope_view = Tensor.view2DSlice(nope_temp, seq_len, cfg.kv_lora_rank);
        var kv_expanded_view = Tensor.view2DSlice(
            scratch.kv_expanded[0 .. seq_len * kv_expanded_dim],
            seq_len,
            kv_expanded_dim,
        );
        self.matmul_fn(&nope_view, self.kv_b_proj, &kv_expanded_view, matmul_scratch);

        // === Apply RoPE ===
        const pos_offset = if (use_cache) cache.cache_position else 0;
        if (self.rope) |rope| {
            // Apply RoPE to Q rope portion (last qk_rope_head_dim of each head)
            const q_slice = scratch.q[0 .. seq_len * self.n_heads * cfg.qk_head_dim];
            for (0..seq_len) |t| {
                const pos = pos_offset + t;
                for (0..self.n_heads) |h| {
                    const head_offset = t * self.n_heads * cfg.qk_head_dim + h * cfg.qk_head_dim;
                    // RoPE is applied to the LAST qk_rope_head_dim dimensions
                    const rope_start = head_offset + cfg.qk_nope_head_dim;
                    if (cfg.rope_interleave) {
                        applyRopeInterleave(q_slice[rope_start .. rope_start + cfg.qk_rope_head_dim], rope, pos);
                    } else {
                        rope.applyInPlace(q_slice[rope_start .. rope_start + cfg.qk_rope_head_dim], pos);
                    }
                }
            }

            // Apply RoPE to shared K rope (from kv_compressed)
            for (0..seq_len) |t| {
                const pos = pos_offset + t;
                const rope_start = t * kv_compressed_dim + cfg.kv_lora_rank;
                if (cfg.rope_interleave) {
                    applyRopeInterleave(kv_compressed_slice[rope_start .. rope_start + cfg.qk_rope_head_dim], rope, pos);
                } else {
                    rope.applyInPlace(kv_compressed_slice[rope_start .. rope_start + cfg.qk_rope_head_dim], pos);
                }
            }
        }

        // === Attention computation ===
        // For each position, compute attention over all cached + current keys
        const total_seq = if (use_cache) cache.cache_position + seq_len else seq_len;
        const context_slice = scratch.context[0 .. seq_len * self.n_heads * cfg.v_head_dim];
        @memset(context_slice, 0);

        // Note: Current implementation handles standard MLA attention.
        // Full MLA with separate rope/nope KV portions, K reconstruction via
        // concat(k_nope, k_rope.expand(n_heads)), and v_head_dim accumulation
        // is handled in populateCache and computeAttentionForPosition below.

        // Prefill: allocate cache memory (does not update cache_position)
        if (!use_cache) {
            try self.populateCache(cache, scratch, kv_compressed_slice, seq_len, cfg);
        }

        // Compute attention using current KV and cached KV
        try self.computeAttention(
            scratch,
            cache,
            kv_compressed_slice,
            seq_len,
            total_seq,
            use_cache,
            cfg,
        );

        // After attention, update the cache with new tokens
        if (!use_cache) {
            // Prefill: cache was already populated, just update position
            cache.cache_position += seq_len;
        } else {
            // Generation: add current token's KV to cache
            try self.updateCacheGeneration(cache, scratch, kv_compressed_slice, cfg);
        }

        // === Output projection ===
        var context_view = Tensor.view2DSlice(context_slice, seq_len, self.n_heads * cfg.v_head_dim);
        var output_view = Tensor.view2D(output_tensor.data(), seq_len, self.d_model);
        self.matmul_fn(&context_view, self.o_proj, &output_view, matmul_scratch);
    }

    fn ensureTemp(self: *const MLAttention, scratch: *MLATemp, seq_len: usize, use_cache: bool) !void {
        const cfg = self.config;
        _ = use_cache;

        const q_comp_size = seq_len * cfg.q_lora_rank;
        if (scratch.q_compressed.len < q_comp_size) {
            if (scratch.q_compressed.len > 0) self.allocator.free(scratch.q_compressed);
            scratch.q_compressed = try self.allocator.alloc(f32, q_comp_size);
        }

        const q_size = seq_len * self.n_heads * cfg.qk_head_dim;
        if (scratch.q.len < q_size) {
            if (scratch.q.len > 0) self.allocator.free(scratch.q);
            scratch.q = try self.allocator.alloc(f32, q_size);
        }

        const kv_comp_size = seq_len * (cfg.kv_lora_rank + cfg.qk_rope_head_dim);
        if (scratch.kv_compressed.len < kv_comp_size) {
            if (scratch.kv_compressed.len > 0) self.allocator.free(scratch.kv_compressed);
            scratch.kv_compressed = try self.allocator.alloc(f32, kv_comp_size);
        }

        const kv_exp_size = seq_len * self.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
        if (scratch.kv_expanded.len < kv_exp_size) {
            if (scratch.kv_expanded.len > 0) self.allocator.free(scratch.kv_expanded);
            scratch.kv_expanded = try self.allocator.alloc(f32, kv_exp_size);
        }

        const scores_size = self.n_heads * self.max_seq_len;
        if (scratch.scores.len < scores_size) {
            if (scratch.scores.len > 0) self.allocator.free(scratch.scores);
            scratch.scores = try self.allocator.alloc(f32, scores_size);
        }

        const ctx_size = seq_len * self.n_heads * cfg.v_head_dim;
        if (scratch.context.len < ctx_size) {
            if (scratch.context.len > 0) self.allocator.free(scratch.context);
            scratch.context = try self.allocator.alloc(f32, ctx_size);
        }
    }

    fn populateCache(
        self: *const MLAttention,
        cache: *MLACache,
        scratch: *MLATemp,
        kv_compressed: []const f32,
        seq_len: usize,
        cfg: MLAConfig,
    ) !void {
        // Ensure cache capacity
        const kv_exp_dim = cfg.qk_nope_head_dim + cfg.v_head_dim;
        const required_k = self.n_heads * self.max_seq_len * cfg.qk_head_dim;
        const required_v = self.n_heads * self.max_seq_len * cfg.v_head_dim;
        const required_rope = self.max_seq_len * cfg.qk_rope_head_dim;

        if (cache.key_cache.len < required_k) {
            if (cache.key_cache.len > 0) self.allocator.free(cache.key_cache);
            cache.key_cache = try self.allocator.alloc(f32, required_k);
        }
        if (cache.value_cache.len < required_v) {
            if (cache.value_cache.len > 0) self.allocator.free(cache.value_cache);
            cache.value_cache = try self.allocator.alloc(f32, required_v);
        }
        if (cache.rope_key_cache.len < required_rope) {
            if (cache.rope_key_cache.len > 0) self.allocator.free(cache.rope_key_cache);
            cache.rope_key_cache = try self.allocator.alloc(f32, required_rope);
        }
        cache.kv_capacity = self.max_seq_len;

        // Copy expanded KV to cache
        const kv_expanded = scratch.kv_expanded[0 .. seq_len * self.n_heads * kv_exp_dim];
        const kv_comp_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

        for (0..seq_len) |t| {
            const cache_pos = cache.cache_position + t;

            // Copy k_nope and v from expanded
            for (0..self.n_heads) |h| {
                const exp_offset = t * self.n_heads * kv_exp_dim + h * kv_exp_dim;
                const k_cache_offset = h * self.max_seq_len * cfg.qk_head_dim + cache_pos * cfg.qk_head_dim;
                const v_cache_offset = h * self.max_seq_len * cfg.v_head_dim + cache_pos * cfg.v_head_dim;

                // Copy k_nope
                @memcpy(
                    cache.key_cache[k_cache_offset .. k_cache_offset + cfg.qk_nope_head_dim],
                    kv_expanded[exp_offset .. exp_offset + cfg.qk_nope_head_dim],
                );
                // Copy v
                @memcpy(
                    cache.value_cache[v_cache_offset .. v_cache_offset + cfg.v_head_dim],
                    kv_expanded[exp_offset + cfg.qk_nope_head_dim .. exp_offset + kv_exp_dim],
                );
            }

            // Copy shared k_rope
            const rope_src_offset = t * kv_comp_dim + cfg.kv_lora_rank;
            const rope_cache_offset = cache_pos * cfg.qk_rope_head_dim;
            @memcpy(
                cache.rope_key_cache[rope_cache_offset .. rope_cache_offset + cfg.qk_rope_head_dim],
                kv_compressed[rope_src_offset .. rope_src_offset + cfg.qk_rope_head_dim],
            );
        }
        // Note: cache.cache_position is NOT updated here - caller is responsible
    }

    /// Update cache during generation (single token at a time).
    fn updateCacheGeneration(
        self: *const MLAttention,
        cache: *MLACache,
        scratch: *MLATemp,
        kv_compressed: []const f32,
        cfg: MLAConfig,
    ) !void {
        const kv_exp_dim = cfg.qk_nope_head_dim + cfg.v_head_dim;
        const kv_comp_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

        // seq_len is always 1 during generation
        const t: usize = 0;
        const cache_pos = cache.cache_position;

        // Get expanded KV for current token
        const kv_expanded = scratch.kv_expanded[0 .. self.n_heads * kv_exp_dim];

        // Copy k_nope and v for each head
        for (0..self.n_heads) |h| {
            const exp_offset = h * kv_exp_dim;
            const k_cache_offset = h * self.max_seq_len * cfg.qk_head_dim + cache_pos * cfg.qk_head_dim;
            const v_cache_offset = h * self.max_seq_len * cfg.v_head_dim + cache_pos * cfg.v_head_dim;

            // Copy k_nope
            @memcpy(
                cache.key_cache[k_cache_offset .. k_cache_offset + cfg.qk_nope_head_dim],
                kv_expanded[exp_offset .. exp_offset + cfg.qk_nope_head_dim],
            );
            // Copy v
            @memcpy(
                cache.value_cache[v_cache_offset .. v_cache_offset + cfg.v_head_dim],
                kv_expanded[exp_offset + cfg.qk_nope_head_dim .. exp_offset + kv_exp_dim],
            );
        }

        // Copy shared k_rope
        const rope_src_offset = t * kv_comp_dim + cfg.kv_lora_rank;
        const rope_cache_offset = cache_pos * cfg.qk_rope_head_dim;
        @memcpy(
            cache.rope_key_cache[rope_cache_offset .. rope_cache_offset + cfg.qk_rope_head_dim],
            kv_compressed[rope_src_offset .. rope_src_offset + cfg.qk_rope_head_dim],
        );

        cache.cache_position += 1;
    }

    fn computeAttention(
        self: *const MLAttention,
        scratch: *MLATemp,
        cache: *MLACache,
        kv_compressed: []const f32,
        seq_len: usize,
        total_seq: usize,
        use_cache: bool,
        cfg: MLAConfig,
    ) !void {
        const q_slice = scratch.q[0 .. seq_len * self.n_heads * cfg.qk_head_dim];
        const context_slice = scratch.context[0 .. seq_len * self.n_heads * cfg.v_head_dim];
        const scores_slice = scratch.scores[0 .. self.n_heads * total_seq];
        const kv_exp_dim = cfg.qk_nope_head_dim + cfg.v_head_dim;
        const kv_comp_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;

        // For each query position
        for (0..seq_len) |q_pos| {
            // For each head
            for (0..self.n_heads) |h| {
                const q_offset = q_pos * self.n_heads * cfg.qk_head_dim + h * cfg.qk_head_dim;
                const q_nope = q_slice[q_offset .. q_offset + cfg.qk_nope_head_dim];
                const q_rope = q_slice[q_offset + cfg.qk_nope_head_dim .. q_offset + cfg.qk_head_dim];

                // Compute scores against all keys
                var max_score: f32 = -std.math.inf(f32);
                const effective_seq = if (use_cache) total_seq else q_pos + 1; // Causal

                for (0..effective_seq) |k_pos| {
                    var score: f32 = 0;

                    // Get k_nope from cache or current
                    const is_current = k_pos >= cache.cache_position;
                    if (is_current) {
                        const curr_pos = k_pos - cache.cache_position;
                        const exp_offset = curr_pos * self.n_heads * kv_exp_dim + h * kv_exp_dim;
                        const k_nope = scratch.kv_expanded[exp_offset .. exp_offset + cfg.qk_nope_head_dim];
                        const rope_offset = curr_pos * kv_comp_dim + cfg.kv_lora_rank;
                        const k_rope = kv_compressed[rope_offset .. rope_offset + cfg.qk_rope_head_dim];

                        // Dot product: q_nope @ k_nope + q_rope @ k_rope
                        score = dotProduct(q_nope, k_nope) + dotProduct(q_rope, k_rope);
                    } else {
                        const k_cache_offset = h * self.max_seq_len * cfg.qk_head_dim + k_pos * cfg.qk_head_dim;
                        const k_nope = cache.key_cache[k_cache_offset .. k_cache_offset + cfg.qk_nope_head_dim];
                        const rope_cache_offset = k_pos * cfg.qk_rope_head_dim;
                        const k_rope = cache.rope_key_cache[rope_cache_offset .. rope_cache_offset + cfg.qk_rope_head_dim];

                        score = dotProduct(q_nope, k_nope) + dotProduct(q_rope, k_rope);
                    }

                    score *= self.scale;
                    scores_slice[h * total_seq + k_pos] = score;
                    if (score > max_score) max_score = score;
                }

                // Softmax
                var sum: f32 = 0;
                for (0..effective_seq) |k_pos| {
                    const idx = h * total_seq + k_pos;
                    scores_slice[idx] = @exp(scores_slice[idx] - max_score);
                    sum += scores_slice[idx];
                }
                for (0..effective_seq) |k_pos| {
                    scores_slice[h * total_seq + k_pos] /= sum;
                }

                // Accumulate weighted values
                const ctx_offset = q_pos * self.n_heads * cfg.v_head_dim + h * cfg.v_head_dim;
                for (0..effective_seq) |k_pos| {
                    const weight = scores_slice[h * total_seq + k_pos];
                    const is_current = k_pos >= cache.cache_position;

                    if (is_current) {
                        const curr_pos = k_pos - cache.cache_position;
                        const exp_offset = curr_pos * self.n_heads * kv_exp_dim + h * kv_exp_dim + cfg.qk_nope_head_dim;
                        const v = scratch.kv_expanded[exp_offset .. exp_offset + cfg.v_head_dim];
                        for (0..cfg.v_head_dim) |d| {
                            context_slice[ctx_offset + d] += weight * v[d];
                        }
                    } else {
                        const v_cache_offset = h * self.max_seq_len * cfg.v_head_dim + k_pos * cfg.v_head_dim;
                        const v = cache.value_cache[v_cache_offset .. v_cache_offset + cfg.v_head_dim];
                        for (0..cfg.v_head_dim) |d| {
                            context_slice[ctx_offset + d] += weight * v[d];
                        }
                    }
                }
            }
        }
    }
};

// =============================================================================
// Helper functions
// =============================================================================

/// Apply RMSNorm in-place to data. Handles both F32 and BF16 norm weights.
fn applyRMSNormTensor(data: []f32, norm_weight: *const Tensor, dim: usize, eps: f32) void {
    const n_tokens = data.len / dim;
    const weight_dtype = norm_weight.dtype;
    const weight_f32: ?[]const f32 = if (weight_dtype == .f32) norm_weight.asSlice(f32) else null;
    const weight_u16: ?[]const u16 = if (weight_dtype == .bf16 or weight_dtype == .f16) norm_weight.asSlice(u16) else null;

    // RMSNorm operates in-place: use data as both input and output
    ops.rmsnormContiguous(
        data,
        data,
        weight_f32,
        weight_u16,
        weight_dtype,
        n_tokens,
        dim,
        eps,
        0.0, // weight_offset
    );
}

/// Apply RMSNorm in-place to a single slice. Handles both F32 and BF16 norm weights.
fn applyRMSNormSliceTensor(data: []f32, norm_weight: *const Tensor, eps: f32) void {
    const weight_dtype = norm_weight.dtype;
    const weight_f32: ?[]const f32 = if (weight_dtype == .f32) norm_weight.asSlice(f32) else null;
    const weight_u16: ?[]const u16 = if (weight_dtype == .bf16 or weight_dtype == .f16) norm_weight.asSlice(u16) else null;

    ops.rmsnormContiguous(
        data,
        data,
        weight_f32,
        weight_u16,
        weight_dtype,
        1, // single token
        data.len,
        eps,
        0.0,
    );
}

fn applyRopeInterleave(data: []f32, rope: *RoPE, pos: usize) void {
    // Use the RoPE's built-in interleaved rotation method
    rope.applyInterleavedInPlace(data, pos);
}

fn dotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    for (a, b) |x, y| {
        sum += x * y;
    }
    return sum;
}

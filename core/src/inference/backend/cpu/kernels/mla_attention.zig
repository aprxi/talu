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
const cpu_layout = compute.cpu.layout_transform;
const cpu_norm = compute.cpu.normalization;
const cpu_reduction = compute.cpu.reduction;
const cpu_rotary = compute.cpu.rotary;
const cpu_softmax = compute.cpu.softmax;
const cpu_cache_store = compute.cpu.cache_store;
const rope_kernel = @import("rope.zig");
const trace = @import("../../../../xray/root.zig").trace;

const Tensor = tensor.Tensor;
const MatmulFn = matmul.MatmulFn;
const RoPE = rope_kernel.RoPE;

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
    /// KV nope-only compact rows: [seq_len, kv_lora_rank]
    kv_nope: []f32 = &.{},
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
        if (self.kv_nope.len > 0) allocator.free(self.kv_nope);
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

        // Decode path appends one token per step into cache.
        if (use_cache and seq_len != 1) {
            return error.InvalidShape;
        }

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
        try cpu_norm.rmsnormContiguousWeightTensor(
            scratch.q_compressed[0 .. seq_len * cfg.q_lora_rank],
            seq_len,
            cfg.q_lora_rank,
            self.q_a_norm,
            self.norm_eps,
            0.0,
        );

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
            cpu_norm.rmsnormInPlaceWeightTensor(
                kv_compressed_slice[start .. start + cfg.kv_lora_rank],
                self.kv_a_norm,
                self.norm_eps,
                0.0,
            );
        }

        // Step 3: Expand kv_lora_rank → n_heads * (qk_nope_head_dim + v_head_dim)
        const kv_expanded_dim = self.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);

        // Extract contiguous nope rows for expansion.
        const nope_temp = scratch.kv_nope[0 .. seq_len * cfg.kv_lora_rank];
        try cpu_layout.extractRowPrefixes(
            kv_compressed_slice,
            seq_len,
            kv_compressed_dim,
            cfg.kv_lora_rank,
            nope_temp,
        );

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
                        cpu_rotary.applyInterleavedInPlace(
                            q_slice[rope_start .. rope_start + cfg.qk_rope_head_dim],
                            rope,
                            pos,
                        );
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
                    cpu_rotary.applyInterleavedInPlace(
                        kv_compressed_slice[rope_start .. rope_start + cfg.qk_rope_head_dim],
                        rope,
                        pos,
                    );
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

        // Cache layout and per-position MLA composition are handled by
        // populateCache/updateCacheGeneration and computeAttention.

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

        const kv_nope_size = seq_len * cfg.kv_lora_rank;
        if (scratch.kv_nope.len < kv_nope_size) {
            if (scratch.kv_nope.len > 0) self.allocator.free(scratch.kv_nope);
            scratch.kv_nope = try self.allocator.alloc(f32, kv_nope_size);
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

        const kv_expanded = scratch.kv_expanded[0 .. seq_len * self.n_heads * kv_exp_dim];
        cpu_cache_store.populateMLACache(
            cache.key_cache,
            cache.value_cache,
            cache.rope_key_cache,
            self.max_seq_len,
            self.n_heads,
            cfg.qk_nope_head_dim,
            cfg.qk_head_dim,
            cfg.v_head_dim,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
            cache.cache_position,
            kv_expanded,
            kv_compressed,
            seq_len,
        );
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
        const cache_pos = cache.cache_position;
        const kv_exp_dim = cfg.qk_nope_head_dim + cfg.v_head_dim;
        const kv_comp_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        const kv_expanded = scratch.kv_expanded[0 .. self.n_heads * kv_exp_dim];
        const kv_comp_token = kv_compressed[0..kv_comp_dim];
        cpu_cache_store.appendMLAToken(
            cache.key_cache,
            cache.value_cache,
            cache.rope_key_cache,
            self.max_seq_len,
            self.n_heads,
            cfg.qk_nope_head_dim,
            cfg.qk_head_dim,
            cfg.v_head_dim,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
            cache_pos,
            kv_expanded,
            kv_comp_token,
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
                const effective_seq = if (use_cache) total_seq else q_pos + 1; // Causal

                for (0..effective_seq) |k_pos| {
                    var score: f32 = 0.0;

                    // Get k_nope from cache or current
                    const is_current = k_pos >= cache.cache_position;
                    if (is_current) {
                        const curr_pos = k_pos - cache.cache_position;
                        const exp_offset = curr_pos * self.n_heads * kv_exp_dim + h * kv_exp_dim;
                        const k_nope = scratch.kv_expanded[exp_offset .. exp_offset + cfg.qk_nope_head_dim];
                        const rope_offset = curr_pos * kv_comp_dim + cfg.kv_lora_rank;
                        const k_rope = kv_compressed[rope_offset .. rope_offset + cfg.qk_rope_head_dim];

                        score = cpu_reduction.dotPairScaled(
                            q_nope,
                            k_nope,
                            q_rope,
                            k_rope,
                            self.scale,
                        );
                    } else {
                        const k_cache_offset = h * self.max_seq_len * cfg.qk_head_dim + k_pos * cfg.qk_head_dim;
                        const k_nope = cache.key_cache[k_cache_offset .. k_cache_offset + cfg.qk_nope_head_dim];
                        const rope_cache_offset = k_pos * cfg.qk_rope_head_dim;
                        const k_rope = cache.rope_key_cache[rope_cache_offset .. rope_cache_offset + cfg.qk_rope_head_dim];

                        score = cpu_reduction.dotPairScaled(
                            q_nope,
                            k_nope,
                            q_rope,
                            k_rope,
                            self.scale,
                        );
                    }
                    scores_slice[h * total_seq + k_pos] = score;
                }

                const head_scores = scores_slice[h * total_seq ..][0..effective_seq];
                cpu_softmax.stableInPlace(head_scores);

                // Accumulate weighted values
                const ctx_offset = q_pos * self.n_heads * cfg.v_head_dim + h * cfg.v_head_dim;
                for (0..effective_seq) |k_pos| {
                    const weight = scores_slice[h * total_seq + k_pos];
                    const is_current = k_pos >= cache.cache_position;

                    if (is_current) {
                        const curr_pos = k_pos - cache.cache_position;
                        const exp_offset = curr_pos * self.n_heads * kv_exp_dim + h * kv_exp_dim + cfg.qk_nope_head_dim;
                        const v = scratch.kv_expanded[exp_offset .. exp_offset + cfg.v_head_dim];
                        cpu_reduction.weightedAccumulateRow(
                            context_slice[ctx_offset .. ctx_offset + cfg.v_head_dim],
                            v,
                            weight,
                        );
                    } else {
                        const v_cache_offset = h * self.max_seq_len * cfg.v_head_dim + k_pos * cfg.v_head_dim;
                        const v = cache.value_cache[v_cache_offset .. v_cache_offset + cfg.v_head_dim];
                        cpu_reduction.weightedAccumulateRow(
                            context_slice[ctx_offset .. ctx_offset + cfg.v_head_dim],
                            v,
                            weight,
                        );
                    }
                }
            }
        }
    }
};

test "MLAttention.forward rejects decode cache updates with seq_len > 1" {
    const allocator = std.testing.allocator;

    const config = MLAConfig{
        .q_lora_rank = 1,
        .kv_lora_rank = 1,
        .qk_head_dim = 2,
        .qk_rope_head_dim = 1,
        .qk_nope_head_dim = 1,
        .v_head_dim = 1,
        .rope_interleave = false,
    };

    var dummy_weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 1 });
    defer dummy_weight_owned.deinit();
    var dummy_norm_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{1});
    defer dummy_norm_owned.deinit();
    var dummy_weight = dummy_weight_owned.view();
    var dummy_norm = dummy_norm_owned.view();

    const test_matmul = struct {
        fn noop(a: *const Tensor, b: *const Tensor, out: *Tensor, scratch: *matmul.MatmulScratch) void {
            _ = a;
            _ = b;
            _ = out;
            _ = scratch;
        }
    }.noop;

    const layer = MLAttention{
        .d_model = 4,
        .n_heads = 1,
        .max_seq_len = 8,
        .config = config,
        .allocator = allocator,
        .q_a_proj = &dummy_weight,
        .q_a_norm = &dummy_norm,
        .q_b_proj = &dummy_weight,
        .kv_a_proj = &dummy_weight,
        .kv_a_norm = &dummy_norm,
        .kv_b_proj = &dummy_weight,
        .o_proj = &dummy_weight,
        .rope = null,
        .norm_eps = 1e-6,
        .scale = 1.0,
        .matmul_fn = test_matmul,
        .layer_idx = 0,
    };

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 2, 4 });
    defer input_owned.deinit();
    var output_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 2, 4 });
    defer output_owned.deinit();
    var input = input_owned.view();
    var output = output_owned.view();

    var cache = MLACache{};
    defer cache.deinit(allocator);
    var scratch = MLATemp{};
    defer scratch.deinit(allocator);
    var matmul_scratch = try matmul.MatmulScratch.init(allocator);
    defer matmul_scratch.deinit();

    const result = layer.forward(&input, &output, &cache, &scratch, &matmul_scratch, true);
    try std.testing.expectError(error.InvalidShape, result);
}

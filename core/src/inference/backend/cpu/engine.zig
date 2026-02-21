//! Fused CPU Backend for Graph-based Inference
//!
//! This module provides the production CPU backend for fused/graph-based ops
//! (norm, attn, ffn, add). It supports continuous batching with multiple
//! concurrent sequences, each in its own slot in the batched KV cache.
//!
//! ## Current Implementation Status
//!
//! The API and data structures fully support batched inference:
//! - LayeredBatchedKVCache with slot management (alloc/free/reset)
//! - Per-slot buffers for hidden states, attention, FFN, logits
//! - decodeBatch() accepts multiple sequences per call
//! - Batched embedding lookup + batched LM-head logits matmul in decodeBatch()
//!
//! Transformer layer decode is batched for slot-compatible block programs, with
//! explicit slot-wise fallback for unsupported kernel topologies.
//!
//! This backend is the sole CPU inference path.
//!
//! ## Layer forward flow
//! 1. Pre-attention norm (ln1)
//! 2. Attention (Q/K/V proj, RoPE, attention scores, output proj)
//! 3. Residual add (hidden += attn_out * residual_multiplier)
//! 4. Post-attention norm (ln2)
//! 5. FFN (gate/up proj, activation, down proj)
//! 6. Residual add (hidden += ffn_out * residual_multiplier)

const std = @import("std");
const tensor = @import("../../../tensor.zig");
const Tensor = tensor.Tensor;
const OwnedTensor = tensor.OwnedTensor;
const compute = @import("../../../compute/root.zig");
const cpu_linalg = compute.cpu.linalg;
const cpu_rowwise = compute.cpu.rowwise;
const cpu_reduction = compute.cpu.reduction;
const cpu_memory = compute.cpu.memory;
const cpu_rotary = compute.cpu.rotary;
const models = @import("../../../models/root.zig");
const rope_scaling = models.rope_scaling;
const contract = @import("../contract.zig");
const log = @import("../../../log.zig");
const progress_mod = @import("../../../capi/progress.zig");

const cpu_executor = @import("executor/root.zig");
const Transformer = cpu_executor.Model;
const cpu_blocks = cpu_executor.weights;
const common_mrope = @import("vision/mrope.zig");
const trace = @import("../../../xray/trace.zig");
const PoolingStrategy = contract.PoolingStrategy;
const LoadedModel = models.LoadedModel;
const kernels = @import("kernels/root.zig");
const BatchedKVCache = kernels.BatchedKVCache;
const LayeredBatchedKVCache = kernels.LayeredBatchedKVCache;
const BatchedAttnTemp = kernels.BatchedAttnTemp;
const attn_mod = @import("kernels/attention.zig");
const vision_runtime_mod = @import("vision/root.zig");

/// Request for a single decode step.
pub const DecodeRequest = contract.DecodeRequest;

/// Result of a batch decode.
pub const DecodeResult = contract.DecodeResult;

/// Batched CPU Backend for continuous batching.
///
/// Supports multiple concurrent sequences, each with its own KV cache slot.
/// Sequences can join/leave at any decode step.
pub const FusedCpuBackend = struct {
    pub const capabilities: contract.Capabilities = .{
        .vision_prefill = true,
        .decode_batch = true,
        .decode_streaming = true,
        .embedding = true,
        .warmup = true,
    };

    allocator: std.mem.Allocator,
    loaded: *LoadedModel,

    /// Unified transformer for forward pass
    model: Transformer,

    /// CPU kernel blocks
    blocks: []cpu_blocks.TransformerBlock,
    /// Runtime RoPE objects owned by CPU backend (model loader stays data-only).
    rope_global: ?*cpu_blocks.RoPE = null,
    rope_local: ?*cpu_blocks.RoPE = null,

    /// Batched KV cache (replaces per-layer AttnCache)
    kv_cache: LayeredBatchedKVCache,

    /// Batched attention scratch
    batched_attn_scratch: BatchedAttnTemp,

    /// Standard scratch (for FFN, etc.)
    scratch: cpu_blocks.ScratchBuffer,
    /// Optional multimodal vision runtime (loaded for models with vision config).
    vision_runtime: ?vision_runtime_mod.VisionRuntime = null,

    /// Per-sequence hidden buffers [max_batch_size][d_model]
    hidden_buffers: []f32,

    /// Per-sequence norm output buffers [max_batch_size][d_model]
    norm_buffers: []f32,

    /// Per-sequence attention output buffers [max_batch_size][d_model]
    attn_out_buffers: []f32,

    /// Per-sequence FFN output buffers [max_batch_size][d_model]
    ffn_out_buffers: []f32,

    /// Per-sequence logits buffers [max_batch_size][vocab_size]
    logits_buffers: []f32,
    /// Compact token workspace [max_batch_size] used by decodeBatch embedding gather.
    decode_batch_tokens: []u32,
    /// Compact slot index workspace [max_batch_size] used by decodeBatch.
    decode_batch_slot_indices: []usize,
    /// Compact batched logits workspace [max_batch_size][vocab_size].
    decode_batch_logits: []f32,
    /// Per-slot RoPE position delta used during decode for multimodal prompts.
    slot_rope_position_deltas: []isize,

    // Model dimensions
    d_model: usize,
    vocab_size: usize,
    max_batch_size: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,

    // Optional prefill progress callback (set by caller before generation)
    prefill_progress_fn: ?PrefillProgressFn = null,
    prefill_progress_ctx: ?*anyopaque = null,

    pub const PrefillProgressFn = *const fn (usize, usize, ?*anyopaque) callconv(.c) void;
    pub const PrefillVisionInput = vision_runtime_mod.PrefillVisionInput;

    pub fn maxBatchSize(self: *const FusedCpuBackend) usize {
        return self.max_batch_size;
    }

    pub fn vocabSize(self: *const FusedCpuBackend) usize {
        return self.vocab_size;
    }

    const RuntimeRopeHandles = struct {
        global: ?*cpu_blocks.RoPE = null,
        local: ?*cpu_blocks.RoPE = null,
    };

    fn deinitRuntimeRopeHandles(
        allocator: std.mem.Allocator,
        handles: *RuntimeRopeHandles,
    ) void {
        if (handles.local) |local_rope| {
            local_rope.deinit(allocator);
            allocator.destroy(local_rope);
        }
        if (handles.global) |global_rope| {
            global_rope.deinit(allocator);
            allocator.destroy(global_rope);
        }
        handles.* = .{};
    }

    fn initRuntimeRopeHandles(
        allocator: std.mem.Allocator,
        loaded: *LoadedModel,
    ) !RuntimeRopeHandles {
        if (loaded.position_embeddings != null) return .{};

        const rope_dim: usize = if (loaded.config.rope_dim > 0)
            @intCast(loaded.config.rope_dim)
        else
            @intCast(loaded.config.head_dim);
        if (rope_dim == 0) return .{};

        var handles: RuntimeRopeHandles = .{};
        errdefer deinitRuntimeRopeHandles(allocator, &handles);

        var global_freqs = try rope_scaling.materializeInverseFrequencies(
            allocator,
            rope_dim,
            loaded.config.rope_theta,
            loaded.config.rope_scaling,
        );
        defer global_freqs.deinit(allocator);

        const global_rope = try allocator.create(cpu_blocks.RoPE);
        errdefer allocator.destroy(global_rope);
        global_rope.* = try cpu_blocks.RoPE.initFromInvFreq(
            allocator,
            rope_dim,
            @intCast(loaded.config.max_seq_len),
            global_freqs.inv_freq,
            global_freqs.attention_scaling,
        );
        handles.global = global_rope;

        if (loaded.config.rope_local_theta > 0 and loaded.config.sliding_window > 0) {
            var local_freqs = try rope_scaling.materializeInverseFrequencies(
                allocator,
                rope_dim,
                loaded.config.rope_local_theta,
                loaded.config.rope_scaling,
            );
            defer local_freqs.deinit(allocator);

            const local_rope = try allocator.create(cpu_blocks.RoPE);
            errdefer allocator.destroy(local_rope);
            local_rope.* = try cpu_blocks.RoPE.initFromInvFreq(
                allocator,
                rope_dim,
                @intCast(loaded.config.max_seq_len),
                local_freqs.inv_freq,
                local_freqs.attention_scaling,
            );
            handles.local = local_rope;
        }

        return handles;
    }

    fn assignRuntimeRoPEToExecutorBlocks(
        blocks: []cpu_blocks.TransformerBlock,
        handles: RuntimeRopeHandles,
    ) void {
        for (blocks) |*block| {
            if (block.getAttentionMut()) |attn| {
                const selected = if (attn.sliding_window > 0 and handles.local != null)
                    handles.local
                else
                    handles.global;
                attn.rope = selected;
            }
            if (block.getMLAAttentionMut()) |mla_attn| {
                mla_attn.rope = handles.global;
            }
        }
    }

    fn clearRuntimeRoPEOnExecutorBlocks(blocks: []cpu_blocks.TransformerBlock) void {
        for (blocks) |*block| {
            if (block.getAttentionMut()) |attn| {
                attn.rope = null;
            }
            if (block.getMLAAttentionMut()) |mla_attn| {
                mla_attn.rope = null;
            }
        }
    }

    pub fn init(
        allocator: std.mem.Allocator,
        loaded: *LoadedModel,
        max_batch_size: usize,
        progress: progress_mod.ProgressContext,
    ) !FusedCpuBackend {
        const model_width: usize = @intCast(loaded.config.d_model);
        const vocab_size: usize = @intCast(loaded.config.vocab_size);
        const layer_total: usize = @intCast(loaded.config.n_layers);
        const head_total: usize = @intCast(loaded.config.n_heads);
        const kv_head_total: usize = @intCast(loaded.config.n_kv_groups); // n_kv_groups = n_kv_heads
        const head_dim: usize = @intCast(loaded.config.head_dim);
        const max_sequence_len: usize = @intCast(loaded.config.max_seq_len);

        // Progress: n_layers (buildBlocks) + 3 (KV cache, scratch, model build)
        const progress_total: u64 = @intCast(layer_total + 3);
        progress.addLine(1, "Preparing", progress_total, null, null);

        var runtime_rope = try initRuntimeRopeHandles(allocator, loaded);
        errdefer deinitRuntimeRopeHandles(allocator, &runtime_rope);

        const cpu_block_set = try cpu_blocks.buildBlocks(
            allocator,
            loaded.config,
            loaded.runtime,
            loaded.blocks,
            progress,
        );
        if (loaded.runtime.explicit_qk_norm_ops) {
            for (cpu_block_set) |*block| {
                if (block.getAttentionMut()) |attn_ptr| {
                    attn_ptr.q_norm = null;
                    attn_ptr.k_norm = null;
                }
            }
        }
        assignRuntimeRoPEToExecutorBlocks(cpu_block_set, runtime_rope);

        // Build batched KV cache
        var kv_cache = try LayeredBatchedKVCache.init(
            allocator,
            layer_total,
            max_batch_size,
            kv_head_total,
            head_dim,
            max_sequence_len,
        );
        errdefer kv_cache.deinit();
        progress.updateLine(1, @intCast(layer_total + 1), null);

        // Build batched attention scratch
        var batched_attn_scratch = try BatchedAttnTemp.init(
            allocator,
            max_batch_size,
            head_total,
            kv_head_total,
            head_dim,
            max_sequence_len,
        );
        errdefer batched_attn_scratch.deinit();

        // Standard scratch for FFN, etc.
        var scratch = try cpu_blocks.ScratchBuffer.init(
            allocator,
            model_width,
            @intCast(loaded.config.d_ff),
            layer_total,
        );
        errdefer scratch.deinit();

        var vision_runtime = try vision_runtime_mod.VisionRuntime.init(allocator, loaded);
        errdefer if (vision_runtime) |*rt| rt.deinit();

        // Initialize Mamba state for heterogeneous models
        var mamba_layer_count: usize = 0;
        var mamba_config: ?cpu_blocks.MambaConfig = null;
        for (cpu_block_set) |*block| {
            if (block.isMamba()) {
                mamba_layer_count += 1;
                if (mamba_config == null) {
                    if (block.getMambaKernel()) |kernel| {
                        mamba_config = kernel.config;
                    }
                }
            }
        }
        if (mamba_layer_count > 0 and mamba_config != null) {
            try scratch.initMamba(mamba_layer_count, mamba_config.?);
            log.info("inference", "Heterogeneous model detected", .{
                .mamba_layers = mamba_layer_count,
                .attention_layers = @as(usize, layer_total) - mamba_layer_count,
            });
        }

        // Initialize ShortConv state for heterogeneous models with conv layers
        var shortconv_layer_count: usize = 0;
        var shortconv_config: ?cpu_blocks.ShortConvConfig = null;
        for (cpu_block_set) |*block| {
            if (block.isShortConv()) {
                shortconv_layer_count += 1;
                if (shortconv_config == null) {
                    if (block.getShortConvKernel()) |kernel| {
                        shortconv_config = kernel.config;
                    }
                }
            }
        }
        if (shortconv_layer_count > 0 and shortconv_config != null) {
            try scratch.initShortConv(shortconv_layer_count, shortconv_config.?);
            log.info("inference", "ShortConv heterogeneous model detected", .{
                .shortconv_layers = shortconv_layer_count,
                .attention_layers = @as(usize, layer_total) - shortconv_layer_count,
            });
        }

        // Initialize MLA (Multi-Latent Attention) cache for MLA models
        var mla_layer_count: usize = 0;
        for (cpu_block_set) |*block| {
            if (block.isMLA()) {
                mla_layer_count += 1;
            }
        }
        if (mla_layer_count > 0) {
            try scratch.initMLA(layer_total);
            log.info("inference", "MLA model detected", .{
                .mla_layers = mla_layer_count,
            });
        }

        // Ensure scratch has space for at least 1 token
        try scratch.ensure(1);
        progress.updateLine(1, @intCast(layer_total + 2), null);

        // Build executor model
        const model = try Transformer.build(allocator, loaded, cpu_block_set);
        progress.updateLine(1, progress_total, null);
        // Note: completeLine(1) is called by the caller after warmup, so the
        // progress bar stays active during the warmup forward pass.

        // Log model structure at debug level
        log.debug("inference", "Model loaded", .{
            .n_layers = layer_total,
            .d_model = model_width,
            .n_heads = head_total,
            .n_kv_heads = kv_head_total,
            .head_dim = head_dim,
            .vocab_size = vocab_size,
        }, @src());

        // Per-sequence buffers
        const hidden_buffers = try allocator.alloc(f32, max_batch_size * model_width);
        errdefer allocator.free(hidden_buffers);

        const norm_buffers = try allocator.alloc(f32, max_batch_size * model_width);
        errdefer allocator.free(norm_buffers);

        const attn_out_buffers = try allocator.alloc(f32, max_batch_size * model_width);
        errdefer allocator.free(attn_out_buffers);

        const ffn_out_buffers = try allocator.alloc(f32, max_batch_size * model_width);
        errdefer allocator.free(ffn_out_buffers);

        const logits_buffers = try allocator.alloc(f32, max_batch_size * vocab_size);
        errdefer allocator.free(logits_buffers);
        const decode_batch_tokens = try allocator.alloc(u32, max_batch_size);
        errdefer allocator.free(decode_batch_tokens);
        const decode_batch_slot_indices = try allocator.alloc(usize, max_batch_size);
        errdefer allocator.free(decode_batch_slot_indices);
        const decode_batch_logits = try allocator.alloc(f32, max_batch_size * vocab_size);
        errdefer allocator.free(decode_batch_logits);
        const slot_rope_position_deltas = try allocator.alloc(isize, max_batch_size);
        errdefer allocator.free(slot_rope_position_deltas);
        @memset(slot_rope_position_deltas, 0);

        return FusedCpuBackend{
            .allocator = allocator,
            .loaded = loaded,
            .model = model,
            .blocks = cpu_block_set,
            .rope_global = runtime_rope.global,
            .rope_local = runtime_rope.local,
            .kv_cache = kv_cache,
            .batched_attn_scratch = batched_attn_scratch,
            .scratch = scratch,
            .vision_runtime = vision_runtime,
            .hidden_buffers = hidden_buffers,
            .norm_buffers = norm_buffers,
            .attn_out_buffers = attn_out_buffers,
            .ffn_out_buffers = ffn_out_buffers,
            .logits_buffers = logits_buffers,
            .decode_batch_tokens = decode_batch_tokens,
            .decode_batch_slot_indices = decode_batch_slot_indices,
            .decode_batch_logits = decode_batch_logits,
            .slot_rope_position_deltas = slot_rope_position_deltas,
            .d_model = model_width,
            .vocab_size = vocab_size,
            .max_batch_size = max_batch_size,
            .n_layers = layer_total,
            .n_heads = head_total,
            .n_kv_heads = kv_head_total,
            .head_dim = head_dim,
        };
    }

    pub fn deinit(self: *FusedCpuBackend) void {
        clearRuntimeRoPEOnExecutorBlocks(self.blocks);
        var runtime_rope = RuntimeRopeHandles{
            .global = self.rope_global,
            .local = self.rope_local,
        };
        deinitRuntimeRopeHandles(self.allocator, &runtime_rope);

        if (self.vision_runtime) |*vision| {
            vision.deinit();
        }
        self.allocator.free(self.decode_batch_logits);
        self.allocator.free(self.decode_batch_slot_indices);
        self.allocator.free(self.decode_batch_tokens);
        self.allocator.free(self.logits_buffers);
        self.allocator.free(self.ffn_out_buffers);
        self.allocator.free(self.attn_out_buffers);
        self.allocator.free(self.norm_buffers);
        self.allocator.free(self.hidden_buffers);
        self.allocator.free(self.slot_rope_position_deltas);
        self.allocator.free(self.model.layers);
        for (self.blocks) |*block| block.deinit(self.allocator);
        self.allocator.free(self.blocks);
        self.scratch.deinit();
        self.batched_attn_scratch.deinit();
        self.kv_cache.deinit();
        self.* = undefined;
    }

    // =========================================================================
    // Slot Management (delegated to KV cache)
    // =========================================================================

    /// Allocate a slot for a new sequence.
    pub fn allocSlot(self: *FusedCpuBackend) ?usize {
        const slot = self.kv_cache.allocSlot() orelse return null;
        self.slot_rope_position_deltas[slot] = 0;
        return slot;
    }

    /// Free a slot when sequence completes.
    pub fn freeSlot(self: *FusedCpuBackend, slot_index: usize) void {
        self.kv_cache.freeSlot(slot_index);
        self.slot_rope_position_deltas[slot_index] = 0;
    }

    /// Reset a slot for reuse (new conversation in same slot).
    pub fn resetSlot(self: *FusedCpuBackend, slot_index: usize) void {
        self.kv_cache.resetSlot(slot_index);
        self.slot_rope_position_deltas[slot_index] = 0;
    }

    /// Get current position for a slot.
    pub fn getPosition(self: *const FusedCpuBackend, slot_index: usize) usize {
        return self.kv_cache.getPosition(slot_index);
    }

    // =========================================================================
    // Single-Sequence API (uses slot 0, compatible with Backend interface)
    // =========================================================================

    /// Prefill: process all prompt tokens, return logits for last position.
    /// This resets the KV cache and processes the full prompt.
    /// Uses slot 0 for single-sequence compatibility.
    pub fn prefill(self: *FusedCpuBackend, tokens: []const u32, logits_out: []f32) !void {
        // Always use slot 0 for single-sequence mode
        self.kv_cache.resetSlot(0);
        try self.prefillSlot(0, tokens, logits_out);
    }

    /// Decode: generate logits for a single token using KV cache.
    /// Returns logits for the next token prediction.
    /// Uses the graph-based model.forwardWithBatchedCache for architecture compatibility.
    pub fn decode(self: *FusedCpuBackend, token: u32, position: usize, logits_out: []f32) !void {
        const model_dim = self.d_model;
        const token_ids = &[_]u32{token};
        self.setPositionDeltaForTextLayers(self.slot_rope_position_deltas[0]);
        defer self.setPositionDeltaForTextLayers(0);

        // 1. Get embedding for the token
        const hidden_buffer = self.getHiddenBuffer(0);
        var hidden_view_3d = Tensor.view3D(
            std.mem.sliceAsBytes(hidden_buffer),
            1,
            model_dim,
        );
        try self.model.embed_tokens.forward(token_ids, &hidden_view_3d);
        cpu_rowwise.scaleInPlace(hidden_buffer, self.loaded.config.embedding_multiplier);
        if (trace.isEnabled()) {
            trace.emit(
                .embed,
                trace.TraceEmission.NO_LAYER,
                0,
                @intCast(position),
                hidden_view_3d.data().ptr,
                .f32,
                .{ 1, 1, @intCast(model_dim), 0 },
                3,
                "gatherEmbeddings",
            );
        }

        // 2. Forward through transformer layers using the graph with batched cache
        try self.model.forwardWithBatchedCache(&hidden_view_3d, &hidden_view_3d, &self.scratch, &self.kv_cache, 0, true);

        // 3. Apply final layer norm (if present — embed-only models skip this)
        if (self.model.norm) |*n| n.forward(&hidden_view_3d, &hidden_view_3d);

        // 4. Compute logits
        try self.computeLogitsFromHidden(hidden_buffer, logits_out);
    }

    /// Streaming token generation with callback support.
    /// Uses the graph-based model.forwardWithBatchedCache for architecture compatibility.
    pub fn decodeStreaming(
        self: *FusedCpuBackend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        tokens_out: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        _ = start_position; // Position tracked internally

        var current_token = first_token;
        var generated_count: usize = 0;
        self.setPositionDeltaForTextLayers(self.slot_rope_position_deltas[0]);
        defer self.setPositionDeltaForTextLayers(0);

        const model_dim = self.d_model;
        const hidden_buffer = self.getHiddenBuffer(0);
        const logits_buffer = self.getLogitsBuffer(0);

        while (generated_count < max_tokens) {
            const token_ids = &[_]u32{current_token};

            // 1. Get embedding
            var hidden_view_3d = Tensor.view3D(
                std.mem.sliceAsBytes(hidden_buffer),
                1,
                model_dim,
            );
            try self.model.embed_tokens.forward(token_ids, &hidden_view_3d);
            cpu_rowwise.scaleInPlace(hidden_buffer, self.loaded.config.embedding_multiplier);

            // 2. Forward through transformer layers with batched cache
            try self.model.forwardWithBatchedCache(&hidden_view_3d, &hidden_view_3d, &self.scratch, &self.kv_cache, 0, true);

            // 3. Final layer norm
            if (self.model.norm) |*n| n.forward(&hidden_view_3d, &hidden_view_3d);

            // 4. Compute logits
            try self.computeLogitsFromHidden(hidden_buffer, logits_buffer);

            // 5. Greedy sample next token
            const max_logit_index = cpu_reduction.argmaxIndex(logits_buffer);
            const next_token: u32 = @intCast(max_logit_index);

            // Store token
            tokens_out[generated_count] = next_token;
            generated_count += 1;

            // Invoke callback
            if (callback) |cb| {
                cb(next_token, callback_data);
            }

            // Check for EOS
            for (eos_token_ids) |eos_id| {
                if (next_token == eos_id) {
                    return generated_count;
                }
            }

            current_token = next_token;
        }

        return generated_count;
    }

    /// Single-step decode for one slot, returning logits for sampling.
    /// This is the minimal per-token operation used by the scheduler's fast path.
    pub fn decodeStep(self: *FusedCpuBackend, slot_index: usize, token: u32) []f32 {
        const model_dim = self.d_model;
        const hidden_buffer = self.getHiddenBuffer(slot_index);
        const logits_buffer = self.getLogitsBuffer(slot_index);
        const token_ids = &[_]u32{token};
        self.setPositionDeltaForTextLayers(self.slot_rope_position_deltas[slot_index]);
        defer self.setPositionDeltaForTextLayers(0);

        // 1. Get embedding
        var hidden_view_3d = Tensor.view3D(
            std.mem.sliceAsBytes(hidden_buffer),
            1,
            model_dim,
        );
        self.model.embed_tokens.forward(token_ids, &hidden_view_3d) catch return logits_buffer;
        cpu_rowwise.scaleInPlace(hidden_buffer, self.loaded.config.embedding_multiplier);

        // 2. Forward through transformer layers with batched cache
        self.model.forwardWithBatchedCache(&hidden_view_3d, &hidden_view_3d, &self.scratch, &self.kv_cache, slot_index, true) catch return logits_buffer;

        // 3. Final layer norm (if present)
        if (self.model.norm) |*n| n.forward(&hidden_view_3d, &hidden_view_3d);

        // 4. Compute logits
        self.computeLogitsFromHidden(hidden_buffer, logits_buffer) catch return logits_buffer;

        return logits_buffer;
    }

    /// Warmup: do a dummy forward pass to pull weights into CPU cache.
    pub fn warmup(self: *FusedCpuBackend) !void {
        // Suppress trace during warmup so xray doesn't capture warmup records
        const saved_handler = trace.getHandler();
        trace.setHandler(null);
        defer trace.setHandler(saved_handler);

        // Full single-token forward pass to warm up all layer weights.
        // This forces mmap pages to load, so the user doesn't wait during
        // the first real inference with no progress feedback.
        self.kv_cache.resetSlot(0);

        const hidden_buffer = self.getHiddenBuffer(0);
        const token_ids = &[_]u32{0}; // BOS or padding token
        var hidden_view_3d = Tensor.view3D(
            std.mem.sliceAsBytes(hidden_buffer),
            1,
            self.d_model,
        );
        try self.model.embed_tokens.forward(token_ids, &hidden_view_3d);

        // Forward through all transformer layers (triggers mmap page loads)
        try self.model.forwardWithBatchedCache(&hidden_view_3d, &hidden_view_3d, &self.scratch, &self.kv_cache, 0, false);

        // Reset after warmup
        self.kv_cache.resetSlot(0);
    }

    // =========================================================================
    // Prefill (per-slot)
    // =========================================================================

    /// Prefill a single slot with prompt tokens.
    /// Returns logits for the last token position.
    ///
    /// This processes the entire prompt through all layers, populating
    /// the KV cache for subsequent decode steps.
    pub fn prefillSlot(
        self: *FusedCpuBackend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        return self.prefillSlotWithVision(slot_index, tokens, null, logits_out);
    }

    /// Prefill with optional preprocessed vision input.
    pub fn prefillSlotWithVision(
        self: *FusedCpuBackend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        const prompt_len = tokens.len;
        if (prompt_len == 0) return;

        const model_dim = self.d_model;

        // Reset the slot in batched cache
        self.kv_cache.resetSlot(slot_index);
        self.slot_rope_position_deltas[slot_index] = 0;

        // Allocate prefill buffer
        const prefill_buffer = try self.allocator.alloc(f32, prompt_len * model_dim);
        defer self.allocator.free(prefill_buffer);

        // Ensure scratch has space for this sequence
        try self.scratch.ensure(prompt_len);

        // 1. Gather embeddings
        var prefill_view_3d = Tensor.view3D(
            std.mem.sliceAsBytes(prefill_buffer),
            prompt_len,
            model_dim,
        );
        log.debug("inference", "Prefill: embedding", .{ .prompt_len = prompt_len }, @src());
        try self.model.embed_tokens.forward(tokens, &prefill_view_3d);
        cpu_rowwise.scaleInPlace(prefill_buffer, self.loaded.config.embedding_multiplier);

        // Optional multimodal scatter for image placeholders.
        var deepstack_ctx: ?Transformer.DeepstackAdditions = null;
        var image_token_positions: []usize = &.{};
        defer if (image_token_positions.len > 0) self.allocator.free(image_token_positions);
        var encoded_vision_output: ?vision_runtime_mod.EncodedVisionOutput = null;
        defer if (encoded_vision_output) |*encoded| encoded.deinit(self.allocator);
        var text_mrope_cos: []f32 = &.{};
        var text_mrope_sin: []f32 = &.{};
        var text_mrope_enabled = false;
        defer if (text_mrope_cos.len > 0) self.allocator.free(text_mrope_cos);
        defer if (text_mrope_sin.len > 0) self.allocator.free(text_mrope_sin);
        defer if (text_mrope_enabled) self.clearRuntimeRoPEForTextLayers();

        if (vision_input) |vi| {
            const vision = if (self.vision_runtime) |*rt|
                rt
            else
                return error.UnsupportedContentType;
            log.debug("scheduler", "Vision encode start", .{
                .images = vi.images.len,
                .image_token_id = vi.image_token_id,
            }, @src());
            encoded_vision_output = try vision.encodeImages(vi.images);
            const encoded = &encoded_vision_output.?;
            log.debug("scheduler", "Vision encode done", .{
                .merged_len = encoded.merged_embeddings.len,
                .deepstack_layers = encoded.deepstack_layer_embeddings.len,
                .prefill_buf_len = prefill_buffer.len,
            }, @src());

            try vision_runtime_mod.scatterVisionEmbeddings(
                prefill_buffer,
                prompt_len,
                model_dim,
                tokens,
                vi.image_token_id,
                encoded.merged_embeddings,
            );

            if (encoded.deepstack_layer_embeddings.len > 0) {
                image_token_positions = try cpu_memory.collectPositionsU32(self.allocator, tokens, vi.image_token_id);
                if (image_token_positions.len == 0) return error.InvalidPromptImageTokens;
                const expected_values = image_token_positions.len * model_dim;
                const layer0_values = if (encoded.deepstack_layer_embeddings.len > 0)
                    encoded.deepstack_layer_embeddings[0].len
                else
                    0;
                log.debug("inference", "Vision deepstack prepared", .{
                    .image_positions = image_token_positions.len,
                    .expected_values = expected_values,
                    .layer_count = encoded.deepstack_layer_embeddings.len,
                    .layer0_values = layer0_values,
                }, @src());
                deepstack_ctx = .{
                    .positions = image_token_positions,
                    .layer_features = encoded.deepstack_layer_embeddings,
                };
            }

            log.debug("scheduler", "Vision scatter done", .{}, @src());

            const mrope_section = resolveMropeSection(&self.loaded.config, self.head_dim);
            const mrope_total = mrope_section[0] + mrope_section[1] + mrope_section[2];
            if (mrope_total > 0) {
                const spatial_merge = std.math.cast(usize, self.loaded.config.vision_spatial_merge_size) orelse return error.InvalidShape;
                const tables = try buildMultimodalMropeTables(
                    self.allocator,
                    tokens,
                    vi.images,
                    vi.image_token_id,
                    spatial_merge,
                    self.head_dim,
                    self.loaded.config.rope_theta,
                    mrope_section,
                );
                text_mrope_cos = tables.cos;
                text_mrope_sin = tables.sin;
                self.slot_rope_position_deltas[slot_index] = tables.position_delta;
                log.debug("scheduler", "MRoPE tables built", .{
                    .cos_len = text_mrope_cos.len,
                    .sin_len = text_mrope_sin.len,
                    .head_dim = self.head_dim,
                    .prompt_len = prompt_len,
                    .expected_len = prompt_len * self.head_dim,
                }, @src());
                try self.setRuntimeRoPEForTextLayers(text_mrope_cos, text_mrope_sin, self.head_dim);
                text_mrope_enabled = true;
                log.debug("inference", "Text MRoPE prepared", .{
                    .seq_len = prompt_len,
                    .head_dim = self.head_dim,
                    .section_t = mrope_section[0],
                    .section_h = mrope_section[1],
                    .section_w = mrope_section[2],
                    .position_delta = self.slot_rope_position_deltas[slot_index],
                }, @src());
            }
        }

        if (trace.isEnabled()) {
            trace.emit(
                .embed,
                trace.TraceEmission.NO_LAYER,
                0,
                @intCast(prompt_len),
                prefill_view_3d.data().ptr,
                .f32,
                .{ 1, @intCast(prompt_len), @intCast(model_dim), 0 },
                3,
                "gatherEmbeddings",
            );
        }

        // 2. Forward through transformer layers using the graph with batched cache
        // use_cache=false triggers prefill mode which populates the cache
        log.debug("inference", "Prefill: forward start", .{ .prompt_len = prompt_len }, @src());

        // Install prefill progress callback (if set), cleared after forward pass
        self.model.prefill_progress_fn = self.prefill_progress_fn;
        self.model.prefill_progress_ctx = self.prefill_progress_ctx;
        defer {
            self.model.prefill_progress_fn = null;
            self.model.prefill_progress_ctx = null;
        }

        log.debug("scheduler", "Transformer forward start", .{
            .prompt_len = prompt_len,
            .has_deepstack = @as(u8, @intFromBool(deepstack_ctx != null)),
            .has_mrope = @as(u8, @intFromBool(text_mrope_enabled)),
        }, @src());
        if (deepstack_ctx) |*ctx| {
            try self.model.forwardWithBatchedCacheWithDeepstack(
                &prefill_view_3d,
                &prefill_view_3d,
                &self.scratch,
                &self.kv_cache,
                slot_index,
                false,
                ctx,
            );
        } else {
            try self.model.forwardWithBatchedCache(
                &prefill_view_3d,
                &prefill_view_3d,
                &self.scratch,
                &self.kv_cache,
                slot_index,
                false,
            );
        }
        log.debug("inference", "Prefill: forward done", .{ .prompt_len = prompt_len }, @src());

        // 3. Final layer norm on last position
        const last_pos_offset = (prompt_len - 1) * model_dim;
        const last_hidden_slice = prefill_buffer[last_pos_offset..][0..model_dim];
        var last_hidden_view = Tensor.view3D(
            std.mem.sliceAsBytes(last_hidden_slice),
            1,
            model_dim,
        );
        if (self.model.norm) |*n| n.forward(&last_hidden_view, &last_hidden_view);

        // 4. Compute logits
        try self.computeLogitsFromHidden(last_hidden_slice, logits_out);
    }

    /// Copy K/V data from a single-sequence AttnCache to the batched cache slot.
    fn copyKVToBatchedCache(
        self: *FusedCpuBackend,
        layer_cache: *BatchedKVCache,
        slot_index: usize,
        source_cache: *const kernels.AttnCache,
        seq_len: usize,
    ) !void {
        cpu_memory.copy3DToSlotted4D(
            layer_cache.key_cache,
            layer_cache.value_cache,
            layer_cache.slot_stride,
            layer_cache.head_stride,
            slot_index,
            source_cache.key_cache,
            source_cache.value_cache,
            source_cache.kv_capacity,
            self.n_kv_heads, // axis1_count
            self.head_dim, // axis3_width
            seq_len, // axis2_len
        );
    }

    fn setRuntimeRoPEForTextLayers(
        self: *FusedCpuBackend,
        cos: []const f32,
        sin: []const f32,
        dim: usize,
    ) !void {
        if (cos.len != sin.len) return error.InvalidShape;
        for (self.model.layers) |layer| {
            var block_mut: *cpu_blocks.TransformerBlock = @constCast(layer.block);
            if (block_mut.getAttentionMut()) |attn| {
                attn.runtime_rope = .{
                    .cos = cos,
                    .sin = sin,
                    .dim = dim,
                };
            }
        }
    }

    fn clearRuntimeRoPEForTextLayers(self: *FusedCpuBackend) void {
        for (self.model.layers) |layer| {
            var block_mut: *cpu_blocks.TransformerBlock = @constCast(layer.block);
            if (block_mut.getAttentionMut()) |attn| {
                attn.runtime_rope = null;
            }
        }
    }

    fn setPositionDeltaForTextLayers(self: *FusedCpuBackend, delta: isize) void {
        for (self.model.layers) |layer| {
            var block_mut: *cpu_blocks.TransformerBlock = @constCast(layer.block);
            if (block_mut.getAttentionMut()) |attn| {
                attn.position_delta = delta;
            }
        }
    }

    const MropeTables = struct {
        cos: []f32,
        sin: []f32,
        position_delta: isize,
    };

    fn resolveMropeSection(config: *const tensor.ModelConfig, head_dim: usize) [3]usize {
        return common_mrope.resolveMropeSection(config, head_dim);
    }

    fn buildMultimodalMropeTables(
        allocator: std.mem.Allocator,
        tokens: []const u32,
        images: []const vision_runtime_mod.PrefillVisionImage,
        image_token_id: u32,
        spatial_merge_size: usize,
        head_dim: usize,
        rope_theta: f32,
        mrope_section: [3]usize,
    ) !MropeTables {
        if (tokens.len == 0) return .{ .cos = &.{}, .sin = &.{}, .position_delta = 0 };
        if ((head_dim % 2) != 0 or mrope_section[0] + mrope_section[1] + mrope_section[2] != head_dim / 2) {
            return error.InvalidShape;
        }

        const seq_len = tokens.len;
        const pos_t = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_t);
        const pos_h = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_h);
        const pos_w = try allocator.alloc(u32, seq_len);
        defer allocator.free(pos_w);

        try buildMultimodalMropePositions(
            tokens,
            images,
            image_token_id,
            spatial_merge_size,
            pos_t,
            pos_h,
            pos_w,
        );

        const half_dim = head_dim / 2;
        const inv_freq = try allocator.alloc(f32, half_dim);
        defer allocator.free(inv_freq);
        try cpu_rotary.fillInverseFrequency(inv_freq, head_dim, rope_theta);

        const cos = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(cos);
        const sin = try allocator.alloc(f32, seq_len * head_dim);
        errdefer allocator.free(sin);

        try cpu_rotary.buildCosSinTablesFromAxisTriples(
            cos,
            sin,
            pos_t,
            pos_h,
            pos_w,
            inv_freq,
            head_dim,
            mrope_section,
        );

        const position_delta = try common_mrope.computePositionDelta(pos_t, pos_h, pos_w);

        return .{
            .cos = cos,
            .sin = sin,
            .position_delta = position_delta,
        };
    }

    fn buildMultimodalMropePositions(
        tokens: []const u32,
        images: []const vision_runtime_mod.PrefillVisionImage,
        image_token_id: u32,
        spatial_merge_size: usize,
        pos_t: []u32,
        pos_h: []u32,
        pos_w: []u32,
    ) !void {
        try common_mrope.buildMultimodalMropePositions(
            tokens,
            images,
            image_token_id,
            spatial_merge_size,
            pos_t,
            pos_h,
            pos_w,
        );
    }

    // =========================================================================
    // Batched Decode
    // =========================================================================

    /// Decode one token for multiple sequences.
    ///
    /// Each request specifies a slot and token. Returns logits for each sequence.
    /// This is the core function for continuous batching.
    ///
    /// ## Implementation Note
    ///
    /// Embedding gather, transformer execution, and LM-head logits are batched
    /// when the loaded block topology supports slot-batched decode.
    /// Unsupported topologies use a slot-wise fallback for correctness.
    ///
    /// Uses graph-based execution via model.forwardWithBatchedCache() for
    /// architecture compatibility across all model types.
    pub fn decodeBatch(
        self: *FusedCpuBackend,
        requests: []const DecodeRequest,
        results: []DecodeResult,
    ) !void {
        const request_total = requests.len;
        if (request_total == 0) return;
        std.debug.assert(results.len >= request_total);
        std.debug.assert(request_total <= self.max_batch_size);
        defer self.setPositionDeltaForTextLayers(0);

        const model_width = self.d_model;
        const vocab_size = self.vocab_size;
        const embedding_multiplier = self.loaded.config.embedding_multiplier;
        const batch_tokens = self.decode_batch_tokens[0..request_total];
        const slot_indices = self.decode_batch_slot_indices[0..request_total];
        const compact_hidden = self.norm_buffers[0 .. request_total * model_width];
        const compact_logits = self.decode_batch_logits[0 .. request_total * vocab_size];

        // 1) Batched embedding gather for all requests in one kernel call.
        for (requests, 0..) |request, request_index| {
            batch_tokens[request_index] = request.token;
            slot_indices[request_index] = request.slot_index;
        }
        var compact_hidden_view = Tensor.view3D(
            std.mem.sliceAsBytes(compact_hidden),
            request_total,
            model_width,
        );
        self.model.embed_tokens.forward(batch_tokens, &compact_hidden_view) catch return;
        cpu_rowwise.scaleInPlace(compact_hidden, embedding_multiplier);

        // 2) Transformer forward through all layers.
        var slot_delta_equal = true;
        const shared_delta: isize = self.slot_rope_position_deltas[slot_indices[0]];
        for (slot_indices[1..]) |slot_index| {
            if (self.slot_rope_position_deltas[slot_index] != shared_delta) {
                slot_delta_equal = false;
                break;
            }
        }

        var used_batched_transformer = false;
        if (slot_delta_equal and self.model.supportsBatchedDecodeSlots()) {
            self.setPositionDeltaForTextLayers(shared_delta);
            self.model.forwardWithBatchedCacheSlots(
                &compact_hidden_view,
                &compact_hidden_view,
                &self.scratch,
                &self.kv_cache,
                slot_indices,
                true,
            ) catch |err| switch (err) {
                error.UnsupportedBatchedDecodeKernel, error.UnsupportedOpInBatchedMode => {},
                else => return err,
            };
            if (self.model.norm) |*n| n.forward(&compact_hidden_view, &compact_hidden_view);
            used_batched_transformer = true;
        }

        if (!used_batched_transformer) {
            for (requests, 0..) |request, request_index| {
                self.setPositionDeltaForTextLayers(self.slot_rope_position_deltas[request.slot_index]);
                const compact_hidden_row = compact_hidden[request_index * model_width ..][0..model_width];
                var hidden_tensor_view = Tensor.view3D(
                    std.mem.sliceAsBytes(compact_hidden_row),
                    1,
                    model_width,
                );
                self.model.forwardWithBatchedCache(
                    &hidden_tensor_view,
                    &hidden_tensor_view,
                    &self.scratch,
                    &self.kv_cache,
                    request.slot_index,
                    true,
                ) catch return;
                if (self.model.norm) |*n| n.forward(&hidden_tensor_view, &hidden_tensor_view);
            }
        }

        // 3) Batched LM-head logits in one matmul over compact hidden rows.
        self.computeLogitsFromHiddenBatch(compact_hidden, request_total, compact_logits) catch return;

        // 4) Scatter compact logits rows back to slot-local buffers and fill results.
        for (requests, 0..) |request, request_index| {
            const logits_buffer = self.getLogitsBuffer(request.slot_index);
            const compact_logits_row = compact_logits[request_index * vocab_size ..][0..vocab_size];
            @memcpy(logits_buffer, compact_logits_row);

            results[request_index] = .{
                .slot_index = request.slot_index,
                .logits = logits_buffer,
            };
        }
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    fn getHiddenBuffer(self: *FusedCpuBackend, slot_index: usize) []f32 {
        const buffer_offset = slot_index * self.d_model;
        return self.hidden_buffers[buffer_offset..][0..self.d_model];
    }

    fn getNormBuffer(self: *FusedCpuBackend, slot_index: usize) []f32 {
        const buffer_offset = slot_index * self.d_model;
        return self.norm_buffers[buffer_offset..][0..self.d_model];
    }

    fn getAttnOutBuffer(self: *FusedCpuBackend, slot_index: usize) []f32 {
        const buffer_offset = slot_index * self.d_model;
        return self.attn_out_buffers[buffer_offset..][0..self.d_model];
    }

    fn getFfnOutBuffer(self: *FusedCpuBackend, slot_index: usize) []f32 {
        const buffer_offset = slot_index * self.d_model;
        return self.ffn_out_buffers[buffer_offset..][0..self.d_model];
    }

    fn getLogitsBuffer(self: *FusedCpuBackend, slot_index: usize) []f32 {
        const offset = slot_index * self.vocab_size;
        return self.logits_buffers[offset..][0..self.vocab_size];
    }

    fn computeLogitsFromHidden(self: *FusedCpuBackend, hidden_slice: []const f32, logits_buffer: []f32) !void {
        const lm_head_ptr = &(self.loaded.lm_head orelse return error.MissingLmHead);
        var hidden_view = Tensor.view2DSlice(@constCast(hidden_slice), 1, self.d_model);
        var logits_view = Tensor.view2DSlice(logits_buffer, 1, self.vocab_size);
        try cpu_linalg.matmulAuto(&hidden_view, lm_head_ptr, &logits_view, &self.scratch.matmul_scratch);

        // Emit trace point for logits before scaling (if handler installed)
        trace.emitFinal(
            .lm_head,
            0, // token
            0, // position (could track this if needed)
            @ptrCast(logits_buffer.ptr),
            .f32,
            .{ @intCast(self.vocab_size), 0, 0, 0 },
            1,
            "matmul_lm_head",
        );

        cpu_rowwise.scaleInPlaceReciprocal(logits_buffer, self.loaded.config.logits_scaling);

        // Emit trace point for scaled logits (after temperature/scaling applied)
        if (self.loaded.config.logits_scaling != 1.0) {
            trace.emitFinal(
                .logits_scaled,
                0,
                0,
                @ptrCast(logits_buffer.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                null,
            );
        }
    }

    fn computeLogitsFromHiddenBatch(
        self: *FusedCpuBackend,
        hidden_rows: []const f32,
        row_count: usize,
        logits_out: []f32,
    ) !void {
        const lm_head_ptr = &(self.loaded.lm_head orelse return error.MissingLmHead);
        std.debug.assert(hidden_rows.len >= row_count * self.d_model);
        std.debug.assert(logits_out.len >= row_count * self.vocab_size);

        var hidden_view = Tensor.view2DSlice(@constCast(hidden_rows), row_count, self.d_model);
        var logits_view = Tensor.view2DSlice(logits_out, row_count, self.vocab_size);
        try cpu_linalg.matmulAuto(&hidden_view, lm_head_ptr, &logits_view, &self.scratch.matmul_scratch);
        cpu_rowwise.scaleInPlaceReciprocal(logits_out[0 .. row_count * self.vocab_size], self.loaded.config.logits_scaling);

        if (trace.isEnabled()) {
            for (0..row_count) |row_index| {
                const row_logits = logits_out[row_index * self.vocab_size ..][0..self.vocab_size];
                trace.emitFinal(
                    .lm_head,
                    @intCast(row_index),
                    0,
                    @ptrCast(row_logits.ptr),
                    .f32,
                    .{ @intCast(self.vocab_size), 0, 0, 0 },
                    1,
                    "matmul_lm_head",
                );
                if (self.loaded.config.logits_scaling != 1.0) {
                    trace.emitFinal(
                        .logits_scaled,
                        @intCast(row_index),
                        0,
                        @ptrCast(row_logits.ptr),
                        .f32,
                        .{ @intCast(self.vocab_size), 0, 0, 0 },
                        1,
                        null,
                    );
                }
            }
        }
    }

    // =========================================================================
    // Embedding Extraction
    // =========================================================================

    /// Extract embeddings from tokens.
    ///
    /// Runs the full transformer forward pass and returns pooled hidden states
    /// as dense vector embeddings. Uses slot 0 for the computation.
    ///
    /// Args:
    ///   tokens: Input token IDs
    ///   pooling: Strategy for reducing sequence to single vector
    ///   normalize: Whether to L2-normalize the output embedding
    ///   embedding_out: Caller-allocated buffer of size d_model
    pub fn embed(
        self: *FusedCpuBackend,
        tokens: []const u32,
        pooling: PoolingStrategy,
        normalize: bool,
        embedding_out: []f32,
    ) !void {
        const seq_len = tokens.len;
        if (seq_len == 0) return error.EmptyInput;
        if (embedding_out.len < self.d_model) return error.BufferTooSmall;

        const model_dim = self.d_model;

        // Use slot 0 for embedding extraction
        const slot_index: usize = 0;
        self.kv_cache.resetSlot(slot_index);

        // Allocate temporary buffer for full sequence hidden states
        const hidden_data = try self.allocator.alloc(f32, seq_len * model_dim);
        defer self.allocator.free(hidden_data);

        // Ensure scratch has space for this sequence
        try self.scratch.ensure(seq_len);

        // 1. Gather embeddings for all tokens
        var hidden_view_3d = Tensor.view3D(
            std.mem.sliceAsBytes(hidden_data),
            seq_len,
            model_dim,
        );
        try self.model.embed_tokens.forward(tokens, &hidden_view_3d);
        cpu_rowwise.scaleInPlace(hidden_data, self.loaded.config.embedding_multiplier);
        // Add position embeddings if present (BERT-family models)
        if (self.loaded.position_embeddings) |pos_emb| {
            const pos_data = pos_emb.asSlice(f32);
            const emb_dim: usize = @intCast(pos_emb.shape[1]);
            try cpu_rowwise.addRowsInPlace(hidden_data, pos_data, seq_len, model_dim, emb_dim);
        }
        // Add token type embeddings if present (all zeros = type 0)
        if (self.loaded.token_type_embeddings) |tt_emb| {
            const tt_data = tt_emb.asSlice(f32);
            const emb_dim: usize = @intCast(tt_emb.shape[1]);
            // Row 0 = type 0 (single-segment)
            const type0_row = tt_data[0..emb_dim];
            try cpu_rowwise.addBroadcastRowInPlace(hidden_data, seq_len, model_dim, type0_row);
        }
        // Apply embedding LayerNorm if present (BERT-family models)
        if (self.loaded.embedding_norm_weight) |emb_norm_w| {
            const norm_ops = compute.cpu.normalization;
            const tv = compute.cpu.tensor_view;
            const emb_norm_bias = self.loaded.embedding_norm_bias;
            for (0..seq_len) |pos| {
                const offset = pos * model_dim;
                var pos_view = Tensor.view3D(
                    std.mem.sliceAsBytes(hidden_data[offset..][0..model_dim]),
                    1,
                    model_dim,
                );
                const input_tv = tv.fromSimpleTensor(&pos_view) orelse unreachable;
                const weight_tv = tv.fromSimpleTensor(&emb_norm_w) orelse unreachable;
                const bias_tv = if (emb_norm_bias) |*b| tv.fromSimpleTensor(b) orelse unreachable else null;
                norm_ops.layerNorm(input_tv, input_tv, weight_tv, bias_tv, self.loaded.config.norm_eps);
            }
        }

        // 2. Forward through transformer layers using the graph with batched cache
        // use_cache=false triggers prefill mode which processes all positions
        try self.model.forwardWithBatchedCache(&hidden_view_3d, &hidden_view_3d, &self.scratch, &self.kv_cache, slot_index, false);
        // 3. Apply final layer norm to ALL positions (if present — embed-only models skip this)
        if (self.model.norm) |*n| {
            for (0..seq_len) |pos| {
                const offset = pos * model_dim;
                var pos_view = Tensor.view3D(
                    std.mem.sliceAsBytes(hidden_data[offset..][0..model_dim]),
                    1,
                    model_dim,
                );
                n.forward(&pos_view, &pos_view);
            }
        }

        // 4. Pool according to strategy
        switch (pooling) {
            .last => {
                const last_offset = (seq_len - 1) * model_dim;
                @memcpy(embedding_out[0..model_dim], hidden_data[last_offset..][0..model_dim]);
            },
            .first => {
                @memcpy(embedding_out[0..model_dim], hidden_data[0..model_dim]);
            },
            .mean => {
                try cpu_reduction.meanPoolRows(hidden_data, seq_len, model_dim, embedding_out[0..model_dim]);
            },
        }

        // 5. Optionally L2 normalize
        if (normalize) {
            cpu_reduction.l2NormalizeInPlace(embedding_out[0..model_dim]);
        }
    }

    /// Returns the embedding dimension (d_model) for this model.
    pub fn embeddingDim(self: *const FusedCpuBackend) usize {
        return self.d_model;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "init FusedCpuBackend types" {
    // Basic compile-time verification that types are defined correctly
    const testing = std.testing;
    _ = testing;

    // Verify DecodeRequest structure
    const req = DecodeRequest{ .slot_index = 0, .token = 42 };
    try std.testing.expectEqual(@as(usize, 0), req.slot_index);
    try std.testing.expectEqual(@as(u32, 42), req.token);

    // Verify DecodeResult structure
    var logits_buf: [10]f32 = undefined;
    const result = DecodeResult{ .slot_index = 1, .logits = &logits_buf };
    try std.testing.expectEqual(@as(usize, 1), result.slot_index);
    try std.testing.expectEqual(@as(usize, 10), result.logits.len);
}

test "FusedCpuBackend.decodeBatch addResidual basic" {
    var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const src = [_]f32{ 0.5, 0.5, 0.5, 0.5 };

    // Test with scale = 1.0
    cpu_rowwise.addScaledInPlace(dst[0..4], src[0..4], 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), dst[1], 0.001);

    // Test with scale = 2.0
    var dst2 = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    cpu_rowwise.addScaledInPlace(dst2[0..4], src[0..4], 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst2[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dst2[1], 0.001);
}

// =============================================================================
// Comprehensive Unit Tests
// =============================================================================

test "decode DecodeRequest instances" {
    const requests = [_]DecodeRequest{
        .{ .slot_index = 0, .token = 100 },
        .{ .slot_index = 1, .token = 200 },
        .{ .slot_index = 2, .token = 300 },
    };

    try std.testing.expectEqual(@as(usize, 3), requests.len);
    try std.testing.expectEqual(@as(u32, 100), requests[0].token);
    try std.testing.expectEqual(@as(u32, 200), requests[1].token);
    try std.testing.expectEqual(@as(u32, 300), requests[2].token);
    try std.testing.expectEqual(@as(usize, 0), requests[0].slot_index);
    try std.testing.expectEqual(@as(usize, 1), requests[1].slot_index);
    try std.testing.expectEqual(@as(usize, 2), requests[2].slot_index);
}

test "decode DecodeResult logits" {
    var buffer1 = [_]f32{ 1.0, 2.0, 3.0 };
    var buffer2 = [_]f32{ 4.0, 5.0, 6.0 };

    const result1 = DecodeResult{ .slot_index = 0, .logits = &buffer1 };
    const result2 = DecodeResult{ .slot_index = 1, .logits = &buffer2 };

    // Verify results point to different buffers
    try std.testing.expect(result1.logits.ptr != result2.logits.ptr);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result1.logits[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result2.logits[0], 0.0001);

    // Modify one buffer doesn't affect the other
    buffer1[0] = 99.0;
    try std.testing.expectApproxEqAbs(@as(f32, 99.0), result1.logits[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result2.logits[0], 0.0001);
}

test "FusedCpuBackend.decodeBatch addResidual negative scale" {
    var dst = [_]f32{ 10.0, 20.0, 30.0 };
    const src = [_]f32{ 1.0, 2.0, 3.0 };

    cpu_rowwise.addScaledInPlace(dst[0..3], src[0..3], -1.0);

    try std.testing.expectApproxEqAbs(@as(f32, 9.0), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), dst[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 27.0), dst[2], 0.001);
}

test "FusedCpuBackend.decodeBatch addResidual zero scale" {
    var dst = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const src = [_]f32{ 100.0, 200.0, 300.0, 400.0 };

    cpu_rowwise.addScaledInPlace(dst[0..4], src[0..4], 0.0);

    // dst should remain unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dst[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), dst[3], 0.001);
}

test "FusedCpuBackend.decodeBatch addResidual fractional scale" {
    var dst = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    const src = [_]f32{ 2.0, 4.0, 6.0, 8.0 };

    cpu_rowwise.addScaledInPlace(dst[0..4], src[0..4], 0.5);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), dst[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), dst[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), dst[3], 0.001);
}

test "FusedCpuBackend.decodeBatch addResidual empty arrays" {
    var dst = [_]f32{};
    const src = [_]f32{};

    cpu_rowwise.addScaledInPlace(dst[0..0], src[0..0], 1.0);
    // Should not crash
}

test "FusedCpuBackend.decodeBatch addResidual single element" {
    var dst = [_]f32{5.0};
    const src = [_]f32{3.0};

    cpu_rowwise.addScaledInPlace(dst[0..1], src[0..1], 2.0);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), dst[0], 0.001);
}

test "FusedCpuBackend.decodeBatch addResidual performance" {
    const n = 1000;
    var dst = [_]f32{1.0} ** n;
    const src = [_]f32{0.5} ** n;

    cpu_rowwise.addScaledInPlace(dst[0..n], src[0..n], 2.0);

    // Check first, middle, and last elements
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[n / 2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), dst[n - 1], 0.001);
}

test "embed applyEmbeddingScaling no scaling" {
    const config = tensor.ModelConfig{
        .model_arch = .custom,
        .d_model = 4,
        .vocab_size = 100,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .head_dim = 4,
        .d_ff = 8,
        .max_seq_len = 128,
        .embedding_multiplier = 1.0,
        .logits_scaling = 1.0,
        .rope_theta = 10000.0,
        .rope_scaling = .{ .rope_type = .none },
        .quant_method = .none,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    cpu_rowwise.scaleInPlace(&values, config.embedding_multiplier);

    // Should remain unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), values[3], 0.001);
}

test "embed applyEmbeddingScaling multiplier" {
    const config = tensor.ModelConfig{
        .model_arch = .custom,
        .d_model = 4,
        .vocab_size = 100,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .head_dim = 4,
        .d_ff = 8,
        .max_seq_len = 128,
        .embedding_multiplier = 2.0,
        .logits_scaling = 1.0,
        .rope_theta = 10000.0,
        .rope_scaling = .{ .rope_type = .none },
        .quant_method = .none,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    cpu_rowwise.scaleInPlace(&values, config.embedding_multiplier);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), values[3], 0.001);
}

test "embed applyEmbeddingScaling fractional" {
    const config = tensor.ModelConfig{
        .model_arch = .custom,
        .d_model = 4,
        .vocab_size = 100,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .head_dim = 4,
        .d_ff = 8,
        .max_seq_len = 128,
        .embedding_multiplier = 0.5,
        .logits_scaling = 1.0,
        .rope_theta = 10000.0,
        .rope_scaling = .{ .rope_type = .none },
        .quant_method = .none,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    var values = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    cpu_rowwise.scaleInPlace(&values, config.embedding_multiplier);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), values[3], 0.001);
}

test "decode applyLogitsScaling no scaling" {
    const config = tensor.ModelConfig{
        .model_arch = .custom,
        .d_model = 4,
        .vocab_size = 100,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .head_dim = 4,
        .d_ff = 8,
        .max_seq_len = 128,
        .embedding_multiplier = 1.0,
        .logits_scaling = 1.0,
        .rope_theta = 10000.0,
        .rope_scaling = .{ .rope_type = .none },
        .quant_method = .none,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    cpu_rowwise.scaleInPlaceReciprocal(&values, config.logits_scaling);

    // Should remain unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), values[3], 0.001);
}

test "decode applyLogitsScaling division" {
    const config = tensor.ModelConfig{
        .model_arch = .custom,
        .d_model = 4,
        .vocab_size = 100,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .head_dim = 4,
        .d_ff = 8,
        .max_seq_len = 128,
        .embedding_multiplier = 1.0,
        .logits_scaling = 2.0,
        .rope_theta = 10000.0,
        .rope_scaling = .{ .rope_type = .none },
        .quant_method = .none,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    var values = [_]f32{ 4.0, 8.0, 12.0, 16.0 };
    cpu_rowwise.scaleInPlaceReciprocal(&values, config.logits_scaling);

    // Logits scaling divides by the factor
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), values[3], 0.001);
}

test "decode applyLogitsScaling inverse" {
    const config = tensor.ModelConfig{
        .model_arch = .custom,
        .d_model = 4,
        .vocab_size = 100,
        .n_layers = 1,
        .n_heads = 1,
        .n_kv_groups = 1,
        .head_dim = 4,
        .d_ff = 8,
        .max_seq_len = 128,
        .embedding_multiplier = 1.0,
        .logits_scaling = 0.5,
        .rope_theta = 10000.0,
        .rope_scaling = .{ .rope_type = .none },
        .quant_method = .none,
        .norm_eps = 1e-5,
        .gaffine_group_size = 128,
    };

    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    cpu_rowwise.scaleInPlaceReciprocal(&values, config.logits_scaling);

    // Division by 0.5 is equivalent to multiplication by 2.0
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), values[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), values[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), values[3], 0.001);
}

test "buildMultimodalMropeTables computes multimodal decode position_delta" {
    const allocator = std.testing.allocator;
    const image_token_id: u32 = 151655;
    const vision_start_token_id: u32 = 151652;
    const vision_end_token_id: u32 = 151653;
    const seq_len: usize = 198;
    const image_span: usize = 169;
    const image_start: usize = 15;

    const tokens = try allocator.alloc(u32, seq_len);
    defer allocator.free(tokens);
    @memset(tokens, 42);
    tokens[image_start - 1] = vision_start_token_id;
    for (0..image_span) |idx| tokens[image_start + idx] = image_token_id;
    tokens[image_start + image_span] = vision_end_token_id;

    const images = [_]vision_runtime_mod.PrefillVisionImage{
        .{
            .pixels = &.{},
            .width = 416,
            .height = 416,
            .grid = .{ .temporal = 1, .height = 26, .width = 26 },
            .token_count = image_span,
        },
    };

    const tables = try FusedCpuBackend.buildMultimodalMropeTables(
        allocator,
        tokens,
        images[0..],
        image_token_id,
        2,
        128,
        5_000_000.0,
        .{ 24, 20, 20 },
    );
    defer allocator.free(tables.cos);
    defer allocator.free(tables.sin);

    try std.testing.expectEqual(seq_len * 128, tables.cos.len);
    try std.testing.expectEqual(seq_len * 128, tables.sin.len);
    // Matches HF rope_deltas behavior for this prompt shape:
    // max multimodal pos id is 41, so next decode pos starts at 42.
    // delta = 42 - 198 = -156.
    try std.testing.expectEqual(@as(isize, -156), tables.position_delta);
}

test "allocSlot LayeredBatchedKVCache" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 2;
    const max_batch: usize = 4;
    const n_kv_heads: usize = 2;
    const head_dim: usize = 8;
    const max_seq_len: usize = 128;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    // Allocate first slot
    const slot0 = cache.allocSlot();
    try std.testing.expect(slot0 != null);
    try std.testing.expectEqual(@as(usize, 0), slot0.?);

    // Check initial position is 0
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot0.?));
}

test "allocSlot multiple LayeredBatchedKVCache" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 1;
    const max_batch: usize = 3;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const max_seq_len: usize = 64;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    const slot0 = cache.allocSlot();
    const slot1 = cache.allocSlot();
    const slot2 = cache.allocSlot();

    try std.testing.expect(slot0 != null);
    try std.testing.expect(slot1 != null);
    try std.testing.expect(slot2 != null);
    try std.testing.expectEqual(@as(usize, 0), slot0.?);
    try std.testing.expectEqual(@as(usize, 1), slot1.?);
    try std.testing.expectEqual(@as(usize, 2), slot2.?);
}

test "allocSlot exhaustion" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 1;
    const max_batch: usize = 2;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const max_seq_len: usize = 64;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    const slot0 = cache.allocSlot();
    const slot1 = cache.allocSlot();
    const slot2 = cache.allocSlot(); // Should fail

    try std.testing.expect(slot0 != null);
    try std.testing.expect(slot1 != null);
    try std.testing.expect(slot2 == null); // No more slots available
}

test "freeSlot and allocSlot realloc" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 1;
    const max_batch: usize = 2;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const max_seq_len: usize = 64;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    const slot0 = cache.allocSlot();
    const slot1 = cache.allocSlot();
    try std.testing.expect(slot0 != null);
    try std.testing.expect(slot1 != null);

    // Free slot 0
    cache.freeSlot(slot0.?);

    // Should be able to allocate again
    const slot2 = cache.allocSlot();
    try std.testing.expect(slot2 != null);
    try std.testing.expectEqual(@as(usize, 0), slot2.?); // Should reuse slot 0
}

test "resetSlot position" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 1;
    const max_batch: usize = 2;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const max_seq_len: usize = 64;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    const slot0 = cache.allocSlot();
    try std.testing.expect(slot0 != null);

    // Simulate position advancement
    cache.layers[0].setPosition(slot0.?, 5);
    try std.testing.expectEqual(@as(usize, 5), cache.getPosition(slot0.?));

    // Reset the slot
    cache.resetSlot(slot0.?);
    try std.testing.expectEqual(@as(usize, 0), cache.getPosition(slot0.?));
}

test "getPosition multiple slots" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 1;
    const max_batch: usize = 3;
    const n_kv_heads: usize = 1;
    const head_dim: usize = 4;
    const max_seq_len: usize = 128;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );
    defer cache.deinit();

    const slot0 = cache.allocSlot();
    const slot1 = cache.allocSlot();
    const slot2 = cache.allocSlot();

    try std.testing.expect(slot0 != null);
    try std.testing.expect(slot1 != null);
    try std.testing.expect(slot2 != null);

    // Set different positions
    cache.layers[0].setPosition(slot0.?, 5);
    cache.layers[0].setPosition(slot1.?, 10);
    cache.layers[0].setPosition(slot2.?, 15);

    // Verify independence
    try std.testing.expectEqual(@as(usize, 5), cache.getPosition(slot0.?));
    try std.testing.expectEqual(@as(usize, 10), cache.getPosition(slot1.?));
    try std.testing.expectEqual(@as(usize, 15), cache.getPosition(slot2.?));
}

test "deinit memory cleanup" {
    const allocator = std.testing.allocator;
    const n_layers: usize = 4;
    const max_batch: usize = 8;
    const n_kv_heads: usize = 4;
    const head_dim: usize = 16;
    const max_seq_len: usize = 256;

    var cache = try LayeredBatchedKVCache.init(
        allocator,
        n_layers,
        max_batch,
        n_kv_heads,
        head_dim,
        max_seq_len,
    );

    // Allocate some slots
    _ = cache.allocSlot();
    _ = cache.allocSlot();
    _ = cache.allocSlot();

    // Deinit should clean up all memory without leaks
    cache.deinit();
}

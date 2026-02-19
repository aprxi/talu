//! Backend abstraction for inference execution.
//!
//! Supports multiple backends: CPU (x86/ARM), Metal (macOS GPU).
//! Provides a unified interface for running transformer inference
//! across different hardware backends with automatic selection.
//!
//! ## Auto-Selection Logic
//!
//! Backend is automatically selected based on:
//! 1. Platform (macOS prefers Metal if available)
//! 2. Model dtype (Metal requires Q4/U8/BF16)
//! 3. Model type (MoE/Mamba not supported on Metal)
//! 4. Build configuration (`enable_metal` flag)
//!
//! ## Override
//!
//! Set `BACKEND` environment variable to override:
//! - `BACKEND=cpu` - Force FusedCpuBackend
//! - `BACKEND=metal` - Force MetalBackend (macOS only)
//!
//! The `BACKEND` override is deprecated and will be removed in a future phase.
//!
//! ## Supported Backends
//!
//! | Backend | Type | Description |
//! |---------|------|------------|
//! | `cpu`   | Batched (FusedCpuBackend) | Production graph-based inference |
//! | `metal` | Lazy graph (MetalBackend) | Production GPU inference (macOS) |
//!
//! ## Legacy Path (Removed)
//!
//! The `CpuBackend` primitive-ops backend was removed because:
//! - No production code uses it (all models are graph-based)
//! - No meaningful test coverage beyond trivial module access
//! - It added a redundant code path and maintenance burden
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
pub const contract = @import("contract.zig");

const graph_mod = @import("../../graph/root.zig");
const capi = @import("../../capi/error.zig");
const log = @import("../../log.zig");
const progress_mod = @import("../../capi/progress.zig");
const tensor = @import("../../tensor.zig");
const ModelConfig = tensor.ModelConfig;
const dtype_mod = @import("../../dtype.zig");
const DType = dtype_mod.DType;
const LoadedModel = graph_mod.LoadedModel;
const LoadOptions = graph_mod.LoadOptions;

pub const cpu = @import("cpu/root.zig");
pub const topology = @import("topology.zig");

/// Re-export types used by the scheduler interface
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;
pub const PrefillProgressFn = cpu.BackendType.PrefillProgressFn;

/// Re-export pooling strategy for embedding extraction
pub const PoolingStrategy = contract.PoolingStrategy;
const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
pub const metal = if (has_metal) @import("metal/root.zig") else struct {
    pub const BackendType = void;
};

comptime {
    contract.assertBackendModuleLayout(cpu, "cpu");
    contract.assertExecutorModuleLayout(cpu.executor, "cpu");
    contract.assertExecutorSymbolLayout(cpu.executor, "cpu");
    contract.assertKernelModuleLayout(cpu.kernels, "cpu");
    contract.assertKernelSupportMap(cpu.kernels, "cpu");
    contract.assertKernelSymbolLayout(cpu.kernels, "cpu");
    contract.assertUnsupportedKernelPolicy(cpu.kernels, "cpu");
    contract.assertSchedulerModuleLayout(cpu.scheduler, "cpu");
    contract.assertSamplingModuleLayout(cpu.sampling, "cpu");
    contract.assertBackendType(cpu.BackendType);
    if (has_metal) {
        contract.assertBackendModuleLayout(metal, "metal");
        contract.assertExecutorModuleLayout(metal.executor, "metal");
        contract.assertExecutorSymbolLayout(metal.executor, "metal");
        contract.assertKernelModuleLayout(metal.kernels, "metal");
        contract.assertKernelSupportMap(metal.kernels, "metal");
        contract.assertKernelSymbolLayout(metal.kernels, "metal");
        contract.assertUnsupportedKernelPolicy(metal.kernels, "metal");
        contract.assertSchedulerModuleLayout(metal.scheduler, "metal");
        contract.assertSamplingModuleLayout(metal.sampling, "metal");
        contract.assertBackendType(metal.BackendType);
    }
}

/// Default batch size for FusedCpuBackend (supports up to N concurrent sequences)
const DEFAULT_MAX_BATCH_SIZE: usize = 8;

/// Compute model-load options before backend initialization.
/// This keeps backend/platform policy out of io/ while preserving fast paths.
pub fn defaultModelLoadOptions() LoadOptions {
    return .{
        .preserve_native_norm_dtype = shouldPreserveNativeNormDType(),
    };
}

fn shouldPreserveNativeNormDType() bool {
    if (std.posix.getenv("BACKEND")) |backend_override| {
        if (std.mem.eql(u8, backend_override, "cpu")) return false;
        if (std.mem.eql(u8, backend_override, "metal")) return has_metal;
    }
    return has_metal;
}

/// Backend type - tagged union of available backends
pub const Backend = union(enum) {
    /// Fused CPU backend for graph ops (production inference)
    cpu: cpu.BackendType,
    /// Metal GPU backend (macOS only)
    metal: if (has_metal) metal.BackendType else void,

    /// Vision input type for prefillSlotWithVision (shared across backends)
    pub const PrefillVisionInput = cpu.BackendType.PrefillVisionInput;

    /// Which generation strategy to use for a given request.
    pub const GenerationPath = enum {
        /// Continuous batching via GenericScheduler
        scheduler,
    };

    /// Select the generation path for this backend and request.
    /// Session routing was removed; all backends now use scheduler.
    pub fn generationPath(self: Backend, has_input_images: bool) GenerationPath {
        _ = self;
        _ = has_input_images;
        return .scheduler;
    }

    /// Whether scheduler should use backend decodeStreaming fast path.
    ///
    /// This is currently enabled only for Metal to preserve the legacy
    /// text-only performance path while removing session routing.
    pub fn supportsSchedulerStreamingFastPath(self: *const Backend) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => true,
        };
    }

    /// Initialize the appropriate backend based on platform and model format.
    /// Automatically selects FusedCpuBackend for CPU, Metal when available.
    pub fn init(allocator: std.mem.Allocator, loaded: *LoadedModel, progress: progress_mod.ProgressContext) !Backend {
        // Check for BACKEND override
        if (std.posix.getenv("BACKEND")) |backend_override| {
            return initFromOverride(allocator, loaded, backend_override, progress);
        }

        // Check if we should use Metal backend (macOS + quantized/bf16 model)
        const has_unsupported_blocks = modelHasMetalUnsupportedBlocks(loaded);
        if (has_metal and isMetalSupported(&loaded.config, loaded.original_weight_dtype, has_unsupported_blocks)) {
            const metal_backend_state = metal.BackendType.init(allocator, loaded) catch |err| {
                if (err == error.MoENotSupported or err == error.MLXNotAvailable or err == error.UnsupportedDType or err == error.ShortConvNotSupportedOnMetal or err == error.MambaNotSupportedOnMetal or err == error.MLANotSupportedOnMetal or err == error.InvalidTensorType) {
                    log.info("inference", "Metal backend unavailable, using CPU", .{
                        .reason = @errorName(err),
                        .detail = getMetalUnsupportedReason(&loaded.config, loaded.original_weight_dtype, has_unsupported_blocks),
                    });
                    const cpu_backend_state = try cpu.BackendType.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
                    return .{ .cpu = cpu_backend_state };
                }
                return err;
            };
            log.debug("inference", "Backend selected", .{ .backend = "metal", .reason = "auto" }, @src());
            return .{ .metal = metal_backend_state };
        }

        // Default to CPU backend
        const cpu_backend_state = try cpu.BackendType.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
        log.debug("inference", "Backend selected", .{ .backend = "cpu", .reason = "default" }, @src());
        return .{ .cpu = cpu_backend_state };
    }

    /// Clean up backend resources
    pub fn deinit(self: *Backend) void {
        switch (self.*) {
            .cpu => |*b| b.deinit(),
            .metal => |*b| if (has_metal) b.deinit() else unreachable,
        }
    }

    /// Prefill: process all prompt tokens, return logits for last position
    /// This resets the KV cache and processes the full prompt
    pub fn prefill(self: *Backend, tokens: []const u32, logits_out: []f32) !void {
        switch (self.*) {
            .cpu => |*b| try b.prefill(tokens, logits_out),
            .metal => |*b| if (has_metal) try b.prefill(tokens, logits_out) else unreachable,
        }
    }

    /// Decode: generate logits for a single token using KV cache
    /// Returns logits for the next token prediction
    pub fn decode(self: *Backend, token: u32, position: usize, logits_out: []f32) !void {
        switch (self.*) {
            .cpu => |*b| try b.decode(token, position, logits_out),
            .metal => |*b| if (has_metal) try b.decode(token, position, logits_out) else unreachable,
        }
    }

    /// Streaming token generation with callback support.
    ///
    /// Generates tokens autoregressively, invoking `callback` after each token.
    /// Some backends (Metal) can pipeline execution for better throughput.
    ///
    /// **Note:** This path uses greedy (argmax) sampling for streamed tokens.
    /// The configured sampling strategy is applied to the first token by the caller.
    pub fn decodeStreaming(
        self: *Backend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
    ) !usize {
        switch (self.*) {
            .cpu => |*b| return b.decodeStreaming(
                first_token,
                start_position,
                max_tokens,
                eos_token_ids,
                output_tokens,
                callback,
                callback_data,
            ),
            .metal => |*b| if (has_metal) {
                return b.decodeStreaming(
                    first_token,
                    start_position,
                    max_tokens,
                    eos_token_ids,
                    output_tokens,
                    callback,
                    callback_data,
                );
            } else unreachable,
        }
    }

    /// Get vocab size for this model
    pub fn vocabSize(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.vocab_size,
            .metal => |*b| if (has_metal) return b.vocab_size else unreachable,
        }
    }

    /// Warmup: do a dummy forward pass to pull weights into CPU cache
    /// This eliminates cold-cache latency on first real inference
    pub fn warmup(self: *Backend) !void {
        switch (self.*) {
            .cpu => |*b| try b.warmup(),
            .metal => {}, // Metal doesn't need warmup (GPU has own memory)
        }
    }

    /// Extract embeddings from tokens.
    ///
    /// Runs the full transformer forward pass and returns pooled hidden states
    /// as dense vector embeddings. Unlike prefill/decode which compute logits,
    /// this returns the normalized hidden states directly.
    ///
    /// Args:
    ///   tokens: Input token IDs
    ///   pooling: Strategy for reducing sequence to single vector
    ///   normalize: Whether to L2-normalize the output embedding
    ///   embedding_out: Caller-allocated buffer of size embeddingDim()
    pub fn embed(
        self: *Backend,
        tokens: []const u32,
        pooling: PoolingStrategy,
        normalize: bool,
        embedding_buffer: []f32,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.embed(tokens, pooling, normalize, embedding_buffer),
            .metal => return error.EmbeddingNotSupported, // Metal backend doesn't support embedding yet
        }
    }

    /// Returns the embedding dimension (d_model) for this model.
    pub fn embeddingDim(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.embeddingDim(),
            .metal => |*b| if (has_metal) return b.d_model else unreachable,
        }
    }

    // ---- Scheduler interface ----
    // These methods allow GenericScheduler to work with Backend directly,
    // keeping all architecture dispatch inside this module.

    pub fn maxBatchSize(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.max_batch_size,
            .metal => |*b| if (has_metal) return b.max_batch_size else unreachable,
        }
    }

    pub fn allocSlot(self: *Backend) ?usize {
        switch (self.*) {
            .cpu => |*b| return b.allocSlot(),
            .metal => |*b| if (has_metal) return b.allocSlot() else unreachable,
        }
    }

    pub fn freeSlot(self: *Backend, slot_index: usize) void {
        switch (self.*) {
            .cpu => |*b| b.freeSlot(slot_index),
            .metal => |*b| if (has_metal) b.freeSlot(slot_index) else unreachable,
        }
    }

    pub fn prefillSlot(
        self: *Backend,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.prefillSlot(slot_index, tokens, logits_out),
            .metal => |*b| if (has_metal) try b.prefillSlot(slot_index, tokens, logits_out) else unreachable,
        }
    }

    pub fn prefillSlotWithVision(
        self: *Backend,
        slot_index: usize,
        tokens: []const u32,
        vision_input: ?*const PrefillVisionInput,
        logits_out: []f32,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.prefillSlotWithVision(slot_index, tokens, vision_input, logits_out),
            .metal => |*b| if (has_metal)
                try b.prefillSlotWithVision(slot_index, tokens, vision_input, logits_out)
            else
                unreachable,
        }
    }

    pub fn decodeBatch(
        self: *Backend,
        requests: []const DecodeRequest,
        results: []DecodeResult,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.decodeBatch(requests, results),
            .metal => |*b| if (has_metal) try b.decodeBatch(requests, results) else unreachable,
        }
    }

    /// Set prefill progress callback. Backends that don't support it ignore silently.
    pub fn setPrefillProgress(
        self: *Backend,
        progress_fn: ?PrefillProgressFn,
        progress_ctx: ?*anyopaque,
    ) void {
        switch (self.*) {
            .cpu => |*b| {
                b.prefill_progress_fn = progress_fn;
                b.prefill_progress_ctx = progress_ctx;
            },
            .metal => {},
        }
    }
};

fn isMetalSupported(config: *const ModelConfig, weight_dtype: DType, has_unsupported_blocks: bool) bool {
    const dtype_supported = switch (weight_dtype) {
        .grouped_affine_u4, .grouped_affine_u8, .bf16 => true,
        else => false,
    };
    if (!dtype_supported) return false;
    if (config.num_experts > 0) return false;
    if (has_unsupported_blocks) return false;
    return true;
}

fn getMetalUnsupportedReason(config: *const ModelConfig, weight_dtype: DType, has_unsupported_blocks: bool) []const u8 {
    const dtype_supported = switch (weight_dtype) {
        .grouped_affine_u4, .grouped_affine_u8, .bf16 => true,
        else => false,
    };
    if (!dtype_supported) {
        return "Weight dtype not supported by Metal (requires Q4/U8/BF16)";
    }
    if (config.num_experts > 0) {
        return "MoE models not supported by Metal backend";
    }
    if (has_unsupported_blocks) {
        return "Mamba/MLA models not supported by Metal backend";
    }
    return "Unknown Metal incompatibility";
}

fn modelHasMetalUnsupportedBlocks(loaded: *const LoadedModel) bool {
    for (loaded.blocks) |block| {
        switch (block) {
            .mamba => return true,
            .attention_mlp => |attn| if (attn.isMLA()) return true,
            .shortconv => {},
        }
    }
    return false;
}

fn initFromOverride(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    backend_override: []const u8,
    progress: progress_mod.ProgressContext,
) !Backend {
    if (std.mem.eql(u8, backend_override, "cpu")) {
        log.info("inference", "BACKEND=cpu: forcing CPU backend", .{});
        const cpu_backend_state = try cpu.BackendType.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
        log.debug("inference", "Backend selected", .{ .backend = "cpu", .reason = "forced" }, @src());
        return .{ .cpu = cpu_backend_state };
    }
    if (std.mem.eql(u8, backend_override, "metal")) {
        if (!has_metal) {
            capi.setContext("BACKEND=metal requested but Metal backend is not enabled for this build/platform", .{});
            return error.MetalNotEnabled;
        }
        log.info("inference", "BACKEND=metal: forcing Metal backend", .{});
        const metal_backend_state = try metal.BackendType.init(allocator, loaded);
        log.debug("inference", "Backend selected", .{ .backend = "metal", .reason = "forced" }, @src());
        return .{ .metal = metal_backend_state };
    }
    capi.setContext("Unknown BACKEND value: {s}", .{backend_override});
    return error.InvalidBackendOverride;
}

// ============================================================================
// Tests
// ============================================================================

test "isMetalSupported supports quantized and bf16" {
    var config = std.mem.zeroes(ModelConfig);
    config.num_experts = 0;

    try std.testing.expect(isMetalSupported(&config, .grouped_affine_u4, false));
    try std.testing.expect(isMetalSupported(&config, .grouped_affine_u8, false));
    try std.testing.expect(isMetalSupported(&config, .bf16, false));
}

test "isMetalSupported rejects unsupported dtypes" {
    var config = std.mem.zeroes(ModelConfig);
    config.num_experts = 0;

    try std.testing.expect(!isMetalSupported(&config, .f32, false));
    try std.testing.expect(!isMetalSupported(&config, .f16, false));
}

test "isMetalSupported rejects moe models" {
    var config = std.mem.zeroes(ModelConfig);
    config.num_experts = 4;

    try std.testing.expect(!isMetalSupported(&config, .grouped_affine_u4, false));
}

test "getMetalUnsupportedReason mentions dtype" {
    var config = std.mem.zeroes(ModelConfig);
    config.num_experts = 0;

    const reason = getMetalUnsupportedReason(&config, .f32, false);
    try std.testing.expect(std.mem.indexOf(u8, reason, "dtype") != null);
}

test "getMetalUnsupportedReason mentions moe" {
    var config = std.mem.zeroes(ModelConfig);
    config.num_experts = 8;

    const reason = getMetalUnsupportedReason(&config, .grouped_affine_u4, false);
    try std.testing.expect(std.mem.indexOf(u8, reason, "MoE") != null);
}

test "isMetalSupported rejects mamba models" {
    var config = std.mem.zeroes(ModelConfig);
    config.num_experts = 0;

    try std.testing.expect(!isMetalSupported(&config, .grouped_affine_u4, true));
}

test "defaultModelLoadOptions follows platform capability" {
    const opts = defaultModelLoadOptions();
    try std.testing.expectEqual(has_metal, opts.preserve_native_norm_dtype);
}

test "backend selection" {
    // This test just verifies the module compiles correctly
    // Actual backend tests require model files
    const testing = std.testing;
    _ = testing;
}

test "generationPath: cpu always selects scheduler" {
    const cpu_backend: Backend = .{ .cpu = undefined };
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, cpu_backend.generationPath(false));
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, cpu_backend.generationPath(true));
}

test "generationPath: metal always selects scheduler" {
    if (!has_metal) return; // Metal variant is void on non-Metal platforms
    const metal_backend: Backend = .{ .metal = undefined };
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, metal_backend.generationPath(true));
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, metal_backend.generationPath(false));
}

test "supportsSchedulerStreamingFastPath: cpu disabled" {
    const cpu_backend: Backend = .{ .cpu = undefined };
    try std.testing.expectEqual(false, cpu_backend.supportsSchedulerStreamingFastPath());
}

test "supportsSchedulerStreamingFastPath: metal enabled" {
    if (!has_metal) return;
    const metal_backend: Backend = .{ .metal = undefined };
    try std.testing.expectEqual(true, metal_backend.supportsSchedulerStreamingFastPath());
}

test "kernel parity: rope cpu vs metal" {
    if (!has_metal) return;
    if (!metal.isAvailable()) return error.SkipZigTest;

    const graph = metal.kernels.graph;

    const allocator = std.testing.allocator;
    var cpu_rope = try cpu.kernels.rope.RoPE.init(allocator, 4, 16, 10_000.0, 1.0);
    defer cpu_rope.deinit(allocator);
    var cpu_kernel = cpu.kernels.rope.RotaryEmbedding{ .rope = &cpu_rope };

    const input = [_]f32{ 0.5, -1.0, 2.0, 3.5 };
    var cpu_output: [4]f32 = undefined;
    cpu_kernel.forward(&input, &cpu_output, 3);

    const input_shape = [_]i64{ 1, 1, 1, 4 };
    const input_handle = graph.createArrayF32(&input, &input_shape);
    defer graph.freeArray(input_handle);

    const metal_kernel = metal.kernels.rope.RotaryEmbedding{
        .head_dim = 4,
        .rope_theta = 10_000.0,
    };
    var output_handle: graph.ArrayHandle = null;
    metal_kernel.forward(input_handle, &output_handle, 3);
    defer graph.freeArray(output_handle);

    graph.eval(&[_]graph.ArrayHandle{output_handle});

    var metal_output: [4]f32 = undefined;
    graph.copyToHost(output_handle, &metal_output);

    for (cpu_output, metal_output) |cpu_v, metal_v| {
        try std.testing.expectApproxEqAbs(cpu_v, metal_v, 0.001);
    }
}

test "kernel parity: kv cache cpu vs metal" {
    if (!has_metal) return;
    if (!metal.isAvailable()) return error.SkipZigTest;

    const graph = metal.kernels.graph;

    const allocator = std.testing.allocator;
    var cpu_cache = try cpu.kernels.kv_cache.BatchedKVCache.init(allocator, 1, 1, 4, 8);
    defer cpu_cache.deinit();
    const slot_index = cpu_cache.allocSlot() orelse return error.TestUnexpectedResult;

    var cpu_kernel = cpu.kernels.kv_cache.KVCache{ .cache = &cpu_cache };
    const key_values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const value_values = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    try cpu_kernel.forward(slot_index, &key_values, &value_values);

    var metal_cache = graph.Cache.init(1, true);
    defer metal_cache.deinit();
    var metal_kernel = metal.kernels.kv_cache.KVCache{ .cache = &metal_cache };

    const shape = [_]i64{ 1, 1, 1, 4 };
    const key_handle = graph.createArrayF32(&key_values, &shape);
    const value_handle = graph.createArrayF32(&value_values, &shape);
    defer graph.freeArray(key_handle);
    defer graph.freeArray(value_handle);

    metal_kernel.forward(0, key_handle, value_handle);
    const cached = metal_cache.get(0);

    graph.eval(&[_]graph.ArrayHandle{ cached.k, cached.v });
    var metal_k: [4]f32 = undefined;
    var metal_v: [4]f32 = undefined;
    graph.copyToHost(cached.k, &metal_k);
    graph.copyToHost(cached.v, &metal_v);

    const cpu_k = cpu_cache.getK(slot_index, 0, 0);
    const cpu_v = cpu_cache.getV(slot_index, 0, 0);
    for (cpu_k, metal_k) |cpu_val, metal_val| {
        try std.testing.expectApproxEqAbs(cpu_val, metal_val, 0.001);
    }
    for (cpu_v, metal_v) |cpu_val, metal_val| {
        try std.testing.expectApproxEqAbs(cpu_val, metal_val, 0.001);
    }
}

test "kernel parity: embedding lookup cpu vs metal" {
    if (!has_metal) return;
    if (!metal.isAvailable()) return error.SkipZigTest;

    const graph = metal.kernels.graph;
    const allocator = std.testing.allocator;

    var cpu_embed_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 3, 2 });
    defer cpu_embed_owned.deinit();
    const embed_data = cpu_embed_owned.asSlice(f32);
    embed_data[0] = 1.0;
    embed_data[1] = 2.0;
    embed_data[2] = 3.0;
    embed_data[3] = 4.0;
    embed_data[4] = 5.0;
    embed_data[5] = 6.0;

    var cpu_out_owned = try tensor.OwnedTensor.init(allocator, .f32, &.{ 1, 2, 2 });
    defer cpu_out_owned.deinit();
    var cpu_embed_view = cpu_embed_owned.view();
    var cpu_out_view = cpu_out_owned.view();
    const cpu_lookup = cpu.kernels.embedding.EmbeddingLookup{
        .embedding_weights = &cpu_embed_view,
    };
    const token_ids = [_]u32{ 2, 0 };
    try cpu_lookup.forward(&token_ids, &cpu_out_view);

    const embed_shape = [_]i64{ 3, 2 };
    const metal_embed = graph.createArrayF32(embed_data, &embed_shape);
    defer graph.freeArray(metal_embed);
    const dummy_ln_data = [_]f32{1.0};
    const ln_shape = [_]i64{1};
    const dummy_ln = graph.createArrayF32(&dummy_ln_data, &ln_shape);
    defer graph.freeArray(dummy_ln);

    const empty_layers = [_]metal.executor.weights.WeightHandles.LayerWeights{};
    var handles = metal.executor.weights.WeightHandles{
        .embed_tokens = metal_embed,
        .embed_tokens_quantized = null,
        .layers = empty_layers[0..],
        .compiled_layers = null,
        .fused_model = null,
        .dense_model = null,
        .ln_final = dummy_ln,
        .lm_head = null,
        .lm_head_quantized = null,
    };

    const metal_lookup = metal.kernels.embedding.EmbeddingLookup{
        .weight_handles = &handles,
    };
    var output_handle: graph.ArrayHandle = null;
    try metal_lookup.forward(&token_ids, &output_handle);
    defer graph.freeArray(output_handle);

    graph.eval(&[_]graph.ArrayHandle{output_handle});

    var metal_output: [4]f32 = undefined;
    graph.copyToHost(output_handle, &metal_output);

    const cpu_output = cpu_out_owned.asSlice(f32);
    for (cpu_output, metal_output) |cpu_val, metal_val| {
        try std.testing.expectApproxEqAbs(cpu_val, metal_val, 0.001);
    }
}

test "kernel parity: weight access error semantics cpu vs metal" {
    if (!has_metal) return;
    if (!metal.isAvailable()) return error.SkipZigTest;

    const graph = metal.kernels.graph;
    const empty_cpu_blocks: [0]cpu.kernels.weights.TransformerBlock = .{};
    const cpu_access = cpu.kernels.weights.WeightAccess{
        .blocks = empty_cpu_blocks[0..],
    };
    var cpu_out: *const cpu.kernels.weights.TransformerBlock = undefined;
    try std.testing.expectError(error.InvalidArgument, cpu_access.forward(0, &cpu_out));

    const dummy_ln_data = [_]f32{1.0};
    const ln_shape = [_]i64{1};
    const dummy_ln = graph.createArrayF32(&dummy_ln_data, &ln_shape);
    defer graph.freeArray(dummy_ln);

    const empty_layers = [_]metal.executor.weights.WeightHandles.LayerWeights{};
    var handles = metal.executor.weights.WeightHandles{
        .embed_tokens = null,
        .embed_tokens_quantized = null,
        .layers = empty_layers[0..],
        .compiled_layers = null,
        .fused_model = null,
        .dense_model = null,
        .ln_final = dummy_ln,
        .lm_head = null,
        .lm_head_quantized = null,
    };
    const metal_access = metal.kernels.weights.WeightAccess{
        .weight_handles = &handles,
    };
    var metal_out: *const metal.executor.weights.WeightHandles.LayerWeights = undefined;
    try std.testing.expectError(error.InvalidArgument, metal_access.forward(0, &metal_out));
}

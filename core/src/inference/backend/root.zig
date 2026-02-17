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

const io = @import("../../io/root.zig");
const capi = @import("../../capi/error.zig");
const log = @import("../../log.zig");
const progress_mod = @import("../../capi/progress.zig");
const loader = io.weights;
const tensor = @import("../../tensor.zig");
const ModelConfig = tensor.ModelConfig;
const dtype_mod = @import("../../dtype.zig");
const DType = dtype_mod.DType;
const pooling_mod = @import("pooling.zig");

pub const fused_cpu = @import("cpu/fused.zig");

/// Re-export types used by the scheduler interface
pub const DecodeRequest = fused_cpu.DecodeRequest;
pub const DecodeResult = fused_cpu.DecodeResult;
pub const PrefillProgressFn = fused_cpu.FusedCpuBackend.PrefillProgressFn;

/// Re-export pooling strategy for embedding extraction
pub const PoolingStrategy = pooling_mod.PoolingStrategy;
const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
pub const metal = if (has_metal) @import("metal/root.zig") else struct {
    pub const MetalBackend = void;
};

/// Default batch size for FusedCpuBackend (supports up to N concurrent sequences)
const DEFAULT_MAX_BATCH_SIZE: usize = 8;

/// Compute model-load options before backend initialization.
/// This keeps backend/platform policy out of io/ while preserving fast paths.
pub fn defaultModelLoadOptions() loader.LoadOptions {
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
    cpu: fused_cpu.FusedCpuBackend,
    /// Metal GPU backend (macOS only)
    metal: if (has_metal) metal.MetalBackend else void,

    /// Vision input type for prefillSlotWithVision (shared across backends)
    pub const PrefillVisionInput = fused_cpu.FusedCpuBackend.PrefillVisionInput;

    /// Which generation strategy to use for a given request.
    pub const GenerationPath = enum {
        /// Continuous batching via GenericScheduler
        scheduler,
        /// Legacy single-sequence session
        session,
    };

    /// Select the generation path for this backend and request.
    /// Encapsulates per-backend routing so callers stay architecture-agnostic.
    pub fn generationPath(self: Backend, has_input_images: bool) GenerationPath {
        return switch (self) {
            .cpu => .scheduler,
            .metal => if (has_input_images) .scheduler else .session,
        };
    }

    /// Initialize the appropriate backend based on platform and model format.
    /// Automatically selects FusedCpuBackend for CPU, Metal when available.
    pub fn init(allocator: std.mem.Allocator, loaded: *loader.LoadedModel, progress: progress_mod.ProgressContext) !Backend {
        // Check for BACKEND override
        if (std.posix.getenv("BACKEND")) |backend_override| {
            return initFromOverride(allocator, loaded, backend_override, progress);
        }

        // Check if we should use Metal backend (macOS + quantized/bf16 model)
        const has_unsupported_blocks = modelHasMetalUnsupportedBlocks(loaded);
        if (has_metal and isMetalSupported(&loaded.config, loaded.original_weight_dtype, has_unsupported_blocks)) {
            const metal_backend_state = metal.MetalBackend.init(allocator, loaded) catch |err| {
                if (err == error.MoENotSupported or err == error.MLXNotAvailable or err == error.UnsupportedDType or err == error.ShortConvNotSupportedOnMetal or err == error.MambaNotSupportedOnMetal or err == error.InvalidTensorType) {
                    log.info("inference", "Metal backend unavailable, using CPU", .{
                        .reason = @errorName(err),
                        .detail = getMetalUnsupportedReason(&loaded.config, loaded.original_weight_dtype, has_unsupported_blocks),
                    });
                    const cpu_backend_state = try fused_cpu.FusedCpuBackend.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
                    return .{ .cpu = cpu_backend_state };
                }
                return err;
            };
            log.debug("inference", "Backend selected", .{ .backend = "metal", .reason = "auto" }, @src());
            return .{ .metal = metal_backend_state };
        }

        // Default to CPU backend
        const cpu_backend_state = try fused_cpu.FusedCpuBackend.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
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
    /// **Note:** Currently uses greedy (argmax) sampling. The configured sampling
    /// strategy is only applied to the first token (by session.generate()).
    /// See backend implementations for details.
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
        return "Mamba models not supported by Metal backend";
    }
    return "Unknown Metal incompatibility";
}

fn modelHasMetalUnsupportedBlocks(loaded: *const loader.LoadedModel) bool {
    for (loaded.blocks) |block| {
        switch (block) {
            .mamba => return true,
            .attention_mlp, .shortconv => {},
        }
    }
    return false;
}

fn initFromOverride(
    allocator: std.mem.Allocator,
    loaded: *loader.LoadedModel,
    backend_override: []const u8,
    progress: progress_mod.ProgressContext,
) !Backend {
    if (std.mem.eql(u8, backend_override, "cpu")) {
        log.info("inference", "BACKEND=cpu: forcing CPU backend", .{});
        const cpu_backend_state = try fused_cpu.FusedCpuBackend.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
        log.debug("inference", "Backend selected", .{ .backend = "cpu", .reason = "forced" }, @src());
        return .{ .cpu = cpu_backend_state };
    }
    if (std.mem.eql(u8, backend_override, "metal")) {
        if (!has_metal) {
            capi.setContext("BACKEND=metal requested but Metal backend is not enabled for this build/platform", .{});
            return error.MetalNotEnabled;
        }
        log.info("inference", "BACKEND=metal: forcing Metal backend", .{});
        const metal_backend_state = try metal.MetalBackend.init(allocator, loaded);
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

test "generationPath: metal selects scheduler for vision, session otherwise" {
    if (!has_metal) return; // Metal variant is void on non-Metal platforms
    const metal_backend: Backend = .{ .metal = undefined };
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, metal_backend.generationPath(true));
    try std.testing.expectEqual(Backend.GenerationPath.session, metal_backend.generationPath(false));
}

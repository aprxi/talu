//! Backend abstraction for inference execution.
//!
//! Supports multiple backends: CPU, Metal, CUDA.
//! Provides a unified interface for running transformer inference
//! across different hardware backends with automatic selection.
//!
//! ## Auto-Selection Logic
//!
//! Backend is automatically selected based on:
//! 1. Environment override (`BACKEND=cpu|metal|cuda|auto`) when selection is `.auto`
//! 2. Build flags (`enable_cuda`, `enable_metal`)
//! 3. Platform (CUDA on Linux/Windows, Metal on macOS)
//! 4. Model compatibility checks (for Metal)
//!
//! Note: CUDA is opt-in only for now. Auto-selection does not choose CUDA.
//!
//! ## Supported Backends
//!
//! | Backend | Type | Description |
//! |---------|------|------------|
//! | `cpu`   | Batched (FusedCpuBackend) | Production graph-based inference |
//! | `metal` | Lazy graph (MetalBackend) | Production GPU inference (macOS) |
//! | `cuda`  | Stub (CudaBackend) | Experimental backend scaffold |
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

const models = @import("../../models/root.zig");
const log = @import("../../log.zig");
const progress_mod = @import("../../progress.zig");
const tensor = @import("../../tensor.zig");
const compute = @import("../../compute/root.zig");
const runtime_contract = @import("../runtime_contract/root.zig");
const ModelConfig = tensor.ModelConfig;
const dtype_mod = @import("../../dtype.zig");
const DType = dtype_mod.DType;
const LoadedModel = models.LoadedModel;
const LoadOptions = models.LoadOptions;

pub const cpu = @import("cpu/root.zig");

/// Re-export types used by the scheduler interface
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;
pub const PrefillProgressFn = cpu.BackendType.PrefillProgressFn;

/// Re-export pooling strategy for embedding extraction
pub const PoolingStrategy = contract.PoolingStrategy;
const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);
pub const metal = if (has_metal) @import("metal/root.zig") else struct {
    pub const BackendType = void;
};
pub const cuda = if (has_cuda) @import("cuda/root.zig") else struct {
    pub const BackendType = void;
};

comptime {
    contract.assertBackendModuleLayout(cpu, "cpu");
    contract.assertVisionModuleLayout(cpu.vision, "cpu");
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
        contract.assertVisionModuleLayout(metal.vision, "metal");
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
    if (has_cuda) {
        contract.assertBackendModuleLayout(cuda, "cuda");
        contract.assertVisionModuleLayout(cuda.vision, "cuda");
        contract.assertExecutorModuleLayout(cuda.executor, "cuda");
        contract.assertExecutorSymbolLayout(cuda.executor, "cuda");
        contract.assertKernelModuleLayout(cuda.kernels, "cuda");
        contract.assertKernelSupportMap(cuda.kernels, "cuda");
        contract.assertKernelSymbolLayout(cuda.kernels, "cuda");
        contract.assertUnsupportedKernelPolicy(cuda.kernels, "cuda");
        contract.assertSchedulerModuleLayout(cuda.scheduler, "cuda");
        contract.assertSamplingModuleLayout(cuda.sampling, "cuda");
        contract.assertBackendType(cuda.BackendType);
    }
}

/// Default batch size for FusedCpuBackend (supports up to N concurrent sequences)
const DEFAULT_MAX_BATCH_SIZE: usize = 8;

/// Compute model-load options before backend initialization.
/// This keeps backend/platform policy out of io/ while preserving optimized execution routes.
pub fn defaultModelLoadOptions(init_options: InitOptions) LoadOptions {
    return .{
        .preserve_native_norm_dtype = shouldPreserveNativeNormDType(init_options.selection),
    };
}

pub const Selection = enum {
    auto,
    cpu,
    metal,
    cuda,
};

/// Backend initialization options selected at startup/config layer.
pub const InitOptions = struct {
    selection: Selection = .auto,
};

fn shouldPreserveNativeNormDType(selection: Selection) bool {
    return switch (selection) {
        .auto => has_metal,
        .cpu => false,
        .metal => has_metal,
        .cuda => false,
    };
}

fn parseSelectionToken(raw: []const u8) ?Selection {
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (token.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(token, "auto")) return .auto;
    if (std.ascii.eqlIgnoreCase(token, "cpu")) return .cpu;
    if (std.ascii.eqlIgnoreCase(token, "metal")) return .metal;
    if (std.ascii.eqlIgnoreCase(token, "cuda")) return .cuda;
    return null;
}

fn selectionOverrideFromEnv(allocator: std.mem.Allocator) ?Selection {
    const raw = std.process.getEnvVarOwned(allocator, "BACKEND") catch return null;
    defer allocator.free(raw);
    return parseSelectionToken(raw);
}

fn selectionName(selection: Selection) []const u8 {
    return @tagName(selection);
}

fn optionalSelectionName(selection: ?Selection) []const u8 {
    if (selection) |value| return selectionName(value);
    return "unset";
}

const CudaProbe = compute.cuda.Probe;

fn cudaProbeName(probe: CudaProbe) []const u8 {
    return @tagName(probe);
}

fn probeCudaRuntime() CudaProbe {
    if (!has_cuda) return .disabled;
    return compute.cuda.probeRuntime();
}

/// Backend type - tagged union of available backends
pub const Backend = union(enum) {
    /// Fused CPU backend for graph ops (production inference)
    cpu: cpu.BackendType,
    /// Metal GPU backend (macOS only)
    metal: if (has_metal) metal.BackendType else void,
    /// CUDA backend (Linux/Windows, experimental scaffold)
    cuda: if (has_cuda) cuda.BackendType else void,

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

    /// Whether scheduler should route decode-tail token generation through
    /// backend `decodeStreaming`.
    pub fn supportsSchedulerBackendDecodeStreamingRoute(self: *const Backend) bool {
        switch (self.*) {
            .cpu => return false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "supportsSchedulerBackendDecodeStreamingRoute"))
                return b.supportsSchedulerBackendDecodeStreamingRoute(),
            .cuda => return false,
        }
        return false;
    }

    /// Initialize the appropriate backend based on platform and model format.
    /// Auto order: Metal (if supported) -> CPU.
    /// CUDA is selected only when explicitly configured.
    pub fn init(
        allocator: std.mem.Allocator,
        loaded: *LoadedModel,
        init_options: InitOptions,
        progress: progress_mod.Context,
    ) !Backend {
        const env_override = if (init_options.selection == .auto)
            selectionOverrideFromEnv(allocator)
        else
            null;
        const selected = if (init_options.selection == .auto)
            (env_override orelse .auto)
        else
            init_options.selection;
        const cuda_probe = probeCudaRuntime();

        log.info("inference", "Backend init policy", .{
            .requested = selectionName(init_options.selection),
            .env_override = optionalSelectionName(env_override),
            .effective = selectionName(selected),
            .cuda_runtime = cudaProbeName(cuda_probe),
            .build_cuda = @as(u8, @intFromBool(build_options.enable_cuda)),
            .build_metal = @as(u8, @intFromBool(build_options.enable_metal)),
        });

        switch (selected) {
            .cpu => return initCpu(allocator, loaded, "configured", progress),
            .metal => return initMetal(allocator, loaded, "configured"),
            .cuda => return initCuda(allocator, loaded, "configured", cuda_probe),
            .auto => {},
        }

        // Check if we should use Metal backend (macOS + quantized/bf16 model)
        const has_unsupported_runtime_features = runtimeHasMetalUnsupportedFeatures(&loaded.runtime);
        if (has_metal and isMetalSupported(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features)) {
            const metal_backend_state = metal.BackendType.init(allocator, loaded) catch |err| {
                if (err == error.MoENotSupported or
                    err == error.MLXNotAvailable or
                    err == error.UnsupportedDType or
                    err == error.ShortConvNotSupportedOnMetal or
                    err == error.MLANotSupportedOnMetal or
                    err == error.InvalidTensorType or
                    err == error.UnsupportedModel or
                    err == error.NotImplemented or
                    err == error.DecodeModelUnavailable)
                {
                    log.info("inference", "Metal backend unavailable, using CPU", .{
                        .reason = @errorName(err),
                        .detail = getMetalUnsupportedReason(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features),
                    });
                    return initCpu(allocator, loaded, "auto_fallback", progress);
                }
                return err;
            };
            log.info("inference", "Backend selected: metal", .{ .reason = "auto" });
            return .{ .metal = metal_backend_state };
        }

        // Default to CPU backend
        return initCpu(allocator, loaded, "default", progress);
    }

    /// Clean up backend resources
    pub fn deinit(self: *Backend) void {
        switch (self.*) {
            .cpu => |*b| b.deinit(),
            .metal => |*b| if (has_metal) b.deinit() else unreachable,
            .cuda => |*b| if (has_cuda) b.deinit() else unreachable,
        }
    }

    /// Prefill: process all prompt tokens, return logits for last position
    /// This resets the KV cache and processes the full prompt
    pub fn prefill(self: *Backend, tokens: []const u32, logits_out: []f32) !void {
        switch (self.*) {
            .cpu => |*b| try b.prefill(tokens, logits_out),
            .metal => |*b| if (has_metal) try b.prefill(tokens, logits_out) else unreachable,
            .cuda => |*b| if (has_cuda) try b.prefill(tokens, logits_out) else unreachable,
        }
    }

    /// Decode: generate logits for a single token using KV cache
    /// Returns logits for the next token prediction
    pub fn decode(self: *Backend, token: u32, position: usize, logits_out: []f32) !void {
        switch (self.*) {
            .cpu => |*b| try b.decode(token, position, logits_out),
            .metal => |*b| if (has_metal) try b.decode(token, position, logits_out) else unreachable,
            .cuda => |*b| if (has_cuda) try b.decode(token, position, logits_out) else unreachable,
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
            .cuda => |*b| if (has_cuda) {
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
            .cuda => |*b| if (has_cuda) return b.vocab_size else unreachable,
        }
    }

    /// Warmup: do a dummy forward pass to pull weights into CPU cache
    /// This eliminates cold-cache latency on first real inference
    pub fn warmup(self: *Backend) !void {
        switch (self.*) {
            .cpu => |*b| try b.warmup(),
            .metal => {}, // Metal doesn't need warmup (GPU has own memory)
            .cuda => {},
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
            .metal => |*b| if (has_metal) try b.embed(tokens, pooling, normalize, embedding_buffer) else unreachable,
            .cuda => return error.EmbeddingNotSupported,
        }
    }

    /// Returns the embedding dimension (d_model) for this model.
    pub fn embeddingDim(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.embeddingDim(),
            .metal => |*b| if (has_metal) return b.d_model else unreachable,
            .cuda => |*b| if (has_cuda) return b.d_model else unreachable,
        }
    }

    // ---- Scheduler interface ----
    // These methods allow GenericScheduler to work with Backend directly,
    // keeping all architecture dispatch inside this module.

    pub fn maxBatchSize(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.max_batch_size,
            .metal => |*b| if (has_metal) return b.max_batch_size else unreachable,
            .cuda => |*b| if (has_cuda) return b.max_batch_size else unreachable,
        }
    }

    pub fn allocSlot(self: *Backend) ?usize {
        switch (self.*) {
            .cpu => |*b| return b.allocSlot(),
            .metal => |*b| if (has_metal) return b.allocSlot() else unreachable,
            .cuda => |*b| if (has_cuda) return b.allocSlot() else unreachable,
        }
    }

    pub fn freeSlot(self: *Backend, slot_index: usize) void {
        switch (self.*) {
            .cpu => |*b| b.freeSlot(slot_index),
            .metal => |*b| if (has_metal) b.freeSlot(slot_index) else unreachable,
            .cuda => |*b| if (has_cuda) b.freeSlot(slot_index) else unreachable,
        }
    }

    pub fn stateDescriptors(self: *const Backend) []const runtime_contract.StateDescriptor {
        switch (self.*) {
            .cpu => |*b| return b.stateDescriptors(),
            .metal => |*b| if (has_metal) return b.stateDescriptors() else unreachable,
            .cuda => |*b| if (has_cuda) return b.stateDescriptors() else unreachable,
        }
    }

    pub fn bindSlotStateBlocks(
        self: *Backend,
        slot_index: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.bindSlotStateBlocks(slot_index, state_blocks),
            .metal => |*b| if (has_metal)
                try b.bindSlotStateBlocks(slot_index, state_blocks)
            else
                unreachable,
            .cuda => |*b| if (has_cuda)
                try b.bindSlotStateBlocks(slot_index, state_blocks)
            else
                unreachable,
        }
    }

    pub fn unbindSlotStateBlocks(self: *Backend, slot_index: usize) void {
        switch (self.*) {
            .cpu => |*b| b.unbindSlotStateBlocks(slot_index),
            .metal => |*b| if (has_metal) b.unbindSlotStateBlocks(slot_index) else unreachable,
            .cuda => |*b| if (has_cuda) b.unbindSlotStateBlocks(slot_index) else unreachable,
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
            .cuda => |*b| if (has_cuda) try b.prefillSlot(slot_index, tokens, logits_out) else unreachable,
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
            .cuda => |*b| if (has_cuda)
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
            .cuda => |*b| if (has_cuda) try b.decodeBatch(requests, results) else unreachable,
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
            .cuda => {},
        }
    }

    /// Maximum pixel count the vision encoder can handle efficiently.
    pub fn visionMaxPixels(self: *const Backend) u64 {
        return switch (self.*) {
            .cpu => cpu.vision.maxPixels(),
            .metal => if (has_metal) metal.vision.maxPixels() else unreachable,
            .cuda => if (has_cuda) cuda.vision.maxPixels() else unreachable,
        };
    }
};

fn isMetalSupported(
    config: *const ModelConfig,
    runtime: *const tensor.ModelRuntime,
    weight_dtype: DType,
    has_unsupported_runtime_features: bool,
) bool {
    const dtype_supported = switch (weight_dtype) {
        .grouped_affine_u4, .grouped_affine_u8, .bf16 => true,
        else => false,
    };
    if (!dtype_supported) return false;
    _ = config;
    _ = runtime;
    if (has_unsupported_runtime_features) return false;
    return true;
}

fn getMetalUnsupportedReason(
    config: *const ModelConfig,
    runtime: *const tensor.ModelRuntime,
    weight_dtype: DType,
    has_unsupported_runtime_features: bool,
) []const u8 {
    const dtype_supported = switch (weight_dtype) {
        .grouped_affine_u4, .grouped_affine_u8, .bf16 => true,
        else => false,
    };
    if (!dtype_supported) {
        return "Weight dtype not supported by Metal (requires Q4/U8/BF16)";
    }
    _ = config;
    if (runtime.has_mla) {
        return "Metal decode-model path requires a supported MLA tensor layout";
    }
    if (has_unsupported_runtime_features) {
        return "Model runtime topology is not yet supported by Metal decode-model path";
    }
    return "Unknown Metal incompatibility";
}

fn runtimeHasMetalUnsupportedFeatures(runtime: *const tensor.ModelRuntime) bool {
    // Metal decode-model path currently does not support mamba layer topology.
    return runtime.has_mamba;
}

fn initCpu(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    progress: progress_mod.Context,
) !Backend {
    const cpu_backend_state = try cpu.BackendType.init(allocator, loaded, DEFAULT_MAX_BATCH_SIZE, progress);
    log.info("inference", "Backend selected: cpu", .{ .reason = reason });
    return .{ .cpu = cpu_backend_state };
}

fn initMetal(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
) !Backend {
    if (!has_metal) {
        return error.MetalNotEnabled;
    }
    const has_unsupported_runtime_features = runtimeHasMetalUnsupportedFeatures(&loaded.runtime);
    if (!isMetalSupported(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features)) {
        log.info("inference", "Metal backend rejected model", .{
            .reason = getMetalUnsupportedReason(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features),
        });
        return error.UnsupportedModel;
    }
    const metal_backend_state = try metal.BackendType.init(allocator, loaded);
    log.info("inference", "Backend selected: metal", .{ .reason = reason });
    return .{ .metal = metal_backend_state };
}

fn initCuda(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    probe: CudaProbe,
) !Backend {
    if (!has_cuda) {
        return error.CudaNotEnabled;
    }
    if (probe != .available) {
        log.info("inference", "CUDA runtime unavailable", .{ .reason = cudaProbeName(probe) });
        return error.CudaUnavailable;
    }
    const cuda_backend_state = try cuda.BackendType.init(allocator, loaded);
    log.info("inference", "Backend selected: cuda", .{ .reason = reason });
    return .{ .cuda = cuda_backend_state };
}

// ============================================================================
// Tests
// ============================================================================

test "isMetalSupported supports quantized and bf16" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(tensor.ModelRuntime);
    config.num_experts = 0;

    try std.testing.expect(isMetalSupported(&config, &runtime, .grouped_affine_u4, false));
    try std.testing.expect(isMetalSupported(&config, &runtime, .grouped_affine_u8, false));
    try std.testing.expect(isMetalSupported(&config, &runtime, .bf16, false));
}

test "isMetalSupported rejects unsupported dtypes" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(tensor.ModelRuntime);
    config.num_experts = 0;

    try std.testing.expect(!isMetalSupported(&config, &runtime, .f32, false));
    try std.testing.expect(!isMetalSupported(&config, &runtime, .f16, false));
}

test "isMetalSupported allows moe models" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(tensor.ModelRuntime);
    config.num_experts = 4;
    runtime.has_moe = true;

    try std.testing.expect(isMetalSupported(&config, &runtime, .grouped_affine_u4, false));
}

test "getMetalUnsupportedReason mentions dtype" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(tensor.ModelRuntime);
    config.num_experts = 0;

    const reason = getMetalUnsupportedReason(&config, &runtime, .f32, false);
    try std.testing.expect(std.mem.indexOf(u8, reason, "dtype") != null);
}

test "isMetalSupported rejects models when runtime features are unsupported" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(tensor.ModelRuntime);
    config.num_experts = 0;
    runtime.has_mamba = true;

    try std.testing.expect(!isMetalSupported(&config, &runtime, .grouped_affine_u4, true));
}

test "runtimeHasMetalUnsupportedFeatures flags unsupported metal topology" {
    var runtime = std.mem.zeroes(tensor.ModelRuntime);
    try std.testing.expect(!runtimeHasMetalUnsupportedFeatures(&runtime));

    runtime.has_mla = true;
    try std.testing.expect(!runtimeHasMetalUnsupportedFeatures(&runtime));

    runtime.has_mla = false;
    runtime.has_mamba = true;
    try std.testing.expect(runtimeHasMetalUnsupportedFeatures(&runtime));
}

test "defaultModelLoadOptions follows platform capability" {
    const opts = defaultModelLoadOptions(.{});
    try std.testing.expectEqual(has_metal, opts.preserve_native_norm_dtype);
}

test "defaultModelLoadOptions honors explicit CPU selection" {
    const opts = defaultModelLoadOptions(.{ .selection = .cpu });
    try std.testing.expectEqual(false, opts.preserve_native_norm_dtype);
}

test "defaultModelLoadOptions honors explicit CUDA selection" {
    const opts = defaultModelLoadOptions(.{ .selection = .cuda });
    try std.testing.expectEqual(false, opts.preserve_native_norm_dtype);
}

test "parseSelectionToken accepts supported backend values" {
    try std.testing.expectEqual(Selection.auto, parseSelectionToken("auto").?);
    try std.testing.expectEqual(Selection.cpu, parseSelectionToken("CPU").?);
    try std.testing.expectEqual(Selection.metal, parseSelectionToken("metal").?);
    try std.testing.expectEqual(Selection.cuda, parseSelectionToken("cuda").?);
}

test "parseSelectionToken rejects unsupported values" {
    try std.testing.expectEqual(@as(?Selection, null), parseSelectionToken(""));
    try std.testing.expectEqual(@as(?Selection, null), parseSelectionToken("  "));
    try std.testing.expectEqual(@as(?Selection, null), parseSelectionToken("rocm"));
}

test "optionalSelectionName returns tag or unset" {
    try std.testing.expectEqualStrings("unset", optionalSelectionName(null));
    try std.testing.expectEqualStrings("cpu", optionalSelectionName(.cpu));
    try std.testing.expectEqualStrings("cuda", optionalSelectionName(.cuda));
}

test "cudaProbeName exposes stable tags" {
    try std.testing.expectEqualStrings("disabled", cudaProbeName(.disabled));
    try std.testing.expectEqualStrings("available", cudaProbeName(.available));
    try std.testing.expectEqualStrings("driver_not_found", cudaProbeName(.driver_not_found));
}

test "probeCudaRuntime returns disabled when CUDA backend is unsupported by target" {
    if (has_cuda) return;
    try std.testing.expectEqual(CudaProbe.disabled, probeCudaRuntime());
}

test "initCuda returns CudaNotEnabled when build target has no CUDA backend" {
    if (has_cuda) return;
    const undefined_loaded: *LoadedModel = undefined;
    try std.testing.expectError(
        error.CudaNotEnabled,
        initCuda(std.testing.allocator, undefined_loaded, "test", .disabled),
    );
}

test "initCuda returns CudaUnavailable when runtime probe is unavailable" {
    if (!has_cuda) return;
    const undefined_loaded: *LoadedModel = undefined;
    try std.testing.expectError(
        error.CudaUnavailable,
        initCuda(std.testing.allocator, undefined_loaded, "test", .driver_not_found),
    );
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

test "generationPath: cuda always selects scheduler" {
    if (!has_cuda) return;
    const cuda_backend: Backend = .{ .cuda = undefined };
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, cuda_backend.generationPath(true));
    try std.testing.expectEqual(Backend.GenerationPath.scheduler, cuda_backend.generationPath(false));
}

test "supportsSchedulerBackendDecodeStreamingRoute: cpu disabled" {
    const cpu_backend: Backend = .{ .cpu = undefined };
    try std.testing.expectEqual(false, cpu_backend.supportsSchedulerBackendDecodeStreamingRoute());
}

test "supportsSchedulerBackendDecodeStreamingRoute: metal delegated" {
    if (!has_metal) return;
    const metal_backend: Backend = .{ .metal = undefined };
    try std.testing.expectEqual(true, metal_backend.supportsSchedulerBackendDecodeStreamingRoute());
}

test "supportsSchedulerBackendDecodeStreamingRoute: cuda disabled" {
    if (!has_cuda) return;
    const cuda_backend: Backend = .{ .cuda = undefined };
    try std.testing.expectEqual(false, cuda_backend.supportsSchedulerBackendDecodeStreamingRoute());
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

    var metal_cache = metal.runtime_graph.Cache.init(1, true);
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

test "visionMaxPixels dispatches to backend vision module" {
    var cpu_backend = Backend{ .cpu = undefined };
    try std.testing.expectEqual(cpu.vision.maxPixels(), cpu_backend.visionMaxPixels());

    if (has_metal) {
        var metal_backend = Backend{ .metal = undefined };
        try std.testing.expectEqual(metal.vision.maxPixels(), metal_backend.visionMaxPixels());
    }
    if (has_cuda) {
        var cuda_backend = Backend{ .cuda = undefined };
        try std.testing.expectEqual(cuda.vision.maxPixels(), cuda_backend.visionMaxPixels());
    }
}

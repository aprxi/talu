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
//! | `cuda`  | CUDA backend | GPU inference with single-device and staged topologies |
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
pub const contract = @import("contract.zig");

const models = @import("models_pkg");
const log = @import("log_pkg");
const progress_mod = @import("progress_pkg");
const compute = @import("compute_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const ModelConfig = models.config.ModelConfig;
const dtype_mod = @import("compute_pkg").dtype;
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
pub const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);
pub const metal = if (has_metal) @import("metal/root.zig") else struct {
    pub const BackendType = void;
};
pub const cuda = if (has_cuda) @import("cuda/root.zig") else struct {
    pub const BackendType = void;
};
const shared_scheduler = @import("../scheduler/contracts.zig");

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

/// Default max concurrent decode slots on non-Windows backends.
/// Override at runtime via TALU_MAX_BATCH_SIZE.
const default_max_batch_size: usize = 8;

/// Conservative Windows CPU default to avoid oversizing prompt/decode state on
/// machines where CPU is commonly the configured backend.
const windows_cpu_max_batch_size: usize = 1;

/// Practical Windows CPU runtime seq_len cap.
/// The fused CPU backend currently pre-allocates KV cache for max_seq_len at
/// startup, so using the model's theoretical maximum can blow up memory on
/// long-context models. Users can override this with TALU_CPU_MAX_SEQ_LEN.
const windows_cpu_runtime_kv_seq_len_cap: usize = 8192;

/// Compute model-load options before backend initialization.
/// This keeps backend/platform policy out of io/ while preserving optimized execution routes.
pub fn defaultModelLoadOptions(init_options: InitOptions) LoadOptions {
    const effective_selection = effectiveLoadSelection(init_options.selection);
    return .{
        .preserve_native_norm_dtype = shouldPreserveNativeNormDType(effective_selection),
        .dequantize_mxfp8_to_bf16 = switch (effective_selection) {
            .cpu => true,
            else => false,
        },
        .dequantize_nvfp4_to_bf16 = switch (effective_selection) {
            .cuda => false,
            else => true,
        },
    };
}

pub fn effectiveLoadSelection(requested: Selection) Selection {
    if (requested != .auto) return requested;
    if (@import("env_pkg").getenv("BACKEND")) |raw_ptr| {
        if (parseSelectionToken(raw_ptr)) |parsed| return parsed;
    }
    return .auto;
}

pub const Selection = enum {
    auto,
    cpu,
    metal,
    cuda,
};

const topology_mod = @import("topology.zig");
pub const CudaTopologyMode = topology_mod.CudaTopologyMode;
pub const CudaTopologyConfig = topology_mod.CudaTopologyConfig;
const CudaTopology = topology_mod.CudaTopology;
const cuda_topology_capabilities = topology_mod.cuda_topology_capabilities;
const validateCudaTopologyConfig = topology_mod.validateCudaTopologyConfig;
const resolveCudaTopology = topology_mod.resolveCudaTopology;
const autoDetectTopologyForModel = topology_mod.autoDetectTopologyForModel;
const resolveCpuLayersTopology = topology_mod.resolveCpuLayersTopology;

fn defaultSchedulerSingleDecodeRoute(
    plan: *const shared_scheduler.SchedulerSingleDecodeRoutePlan,
) shared_scheduler.SchedulerSingleDecodeRoute {
    if (plan.greedy_streaming_semantic_eligible and plan.greedy_streaming_backend_supported) return .greedy_streaming;
    if (plan.top_k_streaming_semantic_eligible and plan.top_k_streaming_backend_supported) return .top_k_streaming;
    if (plan.top_k_candidate_semantic_eligible and plan.top_k_candidate_backend_supported) return .top_k_candidate;
    return .queued;
}

/// Backend initialization options selected at startup/config layer.
pub const InitOptions = struct {
    pub const MetalConfig = struct {
        /// Resolved model directory path (snapshot dir containing config/weights files).
        model_path: ?[]const u8 = null,
        /// User model reference (for example a registry/model identifier) for metadata/logging.
        model_id: ?[]const u8 = null,
        /// Absolute safetensors path used by Metal vision runtime when payload
        /// data is required beyond metadata-only load.
        weights_path: ?[]const u8 = null,
    };

    selection: Selection = .auto,
    /// Multi-stage topology for CUDA backends. When null, single-GPU execution is used.
    cuda_topology: ?CudaTopologyConfig = null,
    /// Metal backend startup metadata.
    metal: ?MetalConfig = null,
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

fn parseSelectionOverrideToken(raw: []const u8) !Selection {
    return parseSelectionToken(raw) orelse error.InvalidArgument;
}

fn selectionOverrideFromEnv(allocator: std.mem.Allocator) !?Selection {
    const raw = std.process.getEnvVarOwned(allocator, "BACKEND") catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return null,
        else => return err,
    };
    defer allocator.free(raw);
    const parsed = parseSelectionOverrideToken(raw) catch |err| {
        log.err("inference", "Invalid BACKEND override", .{
            .value = std.mem.trim(u8, raw, " \t\r\n"),
            .supported = "auto|cpu|metal|cuda",
        }, @src());
        return err;
    };
    return parsed;
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

fn defaultMaxBatchSize(selection: Selection) usize {
    return switch (selection) {
        .cpu => if (builtin.os.tag == .windows) windows_cpu_max_batch_size else default_max_batch_size,
        .auto, .metal, .cuda => default_max_batch_size,
    };
}

fn resolveMaxBatchSize(selection: Selection) usize {
    const default_value = defaultMaxBatchSize(selection);
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_MAX_BATCH_SIZE") catch {
        return default_value;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_MAX_BATCH_SIZE; using default", .{
            .value = trimmed,
            .default = default_value,
        });
        return default_value;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_MAX_BATCH_SIZE must be >= 1; clamping", .{
            .value = parsed,
            .clamped = 1,
        });
        return 1;
    }
    return parsed;
}

pub fn resolveCpuMaxSeqLenForRuntime(allocator: std.mem.Allocator, model_max: usize) usize {
    const default_value = if (builtin.os.tag == .windows)
        @min(model_max, windows_cpu_runtime_kv_seq_len_cap)
    else
        model_max;
    const raw = std.process.getEnvVarOwned(allocator, "TALU_CPU_MAX_SEQ_LEN") catch {
        return default_value;
    };
    defer allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CPU_MAX_SEQ_LEN; using default", .{
            .value = trimmed,
            .default = default_value,
        });
        return default_value;
    };
    if (parsed == 0) {
        log.warn("inference", "TALU_CPU_MAX_SEQ_LEN must be >= 1; clamping", .{
            .value = parsed,
            .clamped = 1,
        });
        return 1;
    }
    return @min(model_max, parsed);
}

/// Backend type - tagged union of available backends
pub const Backend = union(enum) {
    /// Fused CPU backend for graph ops (production inference)
    cpu: cpu.BackendType,
    /// Metal GPU backend (macOS only).
    metal: if (has_metal) metal.BackendType else void,
    /// CUDA backend (Linux/Windows when built with CUDA support).
    cuda: if (has_cuda) cuda.BackendType else void,

    /// Vision input type for prefillSlotWithVision (shared across backends)
    pub const PrefillVisionInput = cpu.BackendType.PrefillVisionInput;

    /// Which generation strategy to use for a given request.
    pub const GenerationPath = enum {
        /// Continuous batching via GenericScheduler
        scheduler,
    };

    /// Select the generation path for this backend and request.
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
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "supportsSchedulerBackendDecodeStreamingRoute"))
                return b.supportsSchedulerBackendDecodeStreamingRoute(),
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
            try selectionOverrideFromEnv(allocator)
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
            .metal => return initMetal(allocator, loaded, "configured", init_options.metal),
            .cuda => return initCuda(allocator, loaded, "configured", cuda_probe, init_options.cuda_topology, progress),
            .auto => {},
        }

        // Check if we should use Metal backend (macOS + quantized/bf16 model)
        const has_unsupported_runtime_features = runtimeHasMetalUnsupportedFeatures(&loaded.runtime);
        if (has_metal and isMetalSupported(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features)) {
            return initMetal(allocator, loaded, "auto", init_options.metal);
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

    /// Explicit device/barrier synchronization for correctness-sensitive
    /// observability flows such as xray capture finalization.
    ///
    /// This must never change backend math or route selection. It exists so
    /// boundary code can guarantee that backend work has finished producing any
    /// host-visible trace/capture outputs before those outputs are serialized
    /// or destroyed during teardown.
    pub fn synchronize(self: *Backend) !void {
        switch (self.*) {
            .cpu => {},
            .metal => |*b| if (has_metal) b.synchronize() else unreachable,
            .cuda => |*b| if (has_cuda) try b.synchronize() else unreachable,
        }
    }

    /// Explicit end-of-run cleanup for backend state that is thread-local to
    /// the execution thread. This must not change math or route selection; it
    /// only clears transient per-run resources after a generation completes.
    pub fn cleanupExecutionThreadState(self: *Backend) void {
        switch (self.*) {
            .cpu => {},
            .metal => |*b| if (has_metal) b.cleanupExecutionThreadState() else unreachable,
            .cuda => {},
        }
    }

    /// Explicit execution-thread teardown barrier.
    ///
    /// Call this only on the thread that actually executed backend work and
    /// only when that thread is about to stop issuing more work. This is
    /// distinct from per-run cleanup: it is allowed to destroy thread-local
    /// caches whose contents must not outlive the worker thread lifecycle.
    pub fn teardownExecutionThreadState(self: *Backend) void {
        switch (self.*) {
            .cpu => {},
            .metal => |*b| if (has_metal) b.teardownExecutionThreadState() else unreachable,
            .cuda => {},
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

    pub fn supportsSchedulerBackendTopKDecodeRoute(
        self: *const Backend,
        sampling_config: *const cpu.sampling.SamplingConfig,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal) b.supportsSchedulerBackendTopKDecodeRoute(sampling_config) else unreachable,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "supportsSchedulerBackendTopKDecodeRoute"))
                b.supportsSchedulerBackendTopKDecodeRoute(sampling_config)
            else
                false,
        };
    }

    pub fn planSchedulerSingleDecodeRoute(
        self: *const Backend,
        plan: *const shared_scheduler.SchedulerSingleDecodeRoutePlan,
    ) shared_scheduler.SchedulerSingleDecodeRoute {
        return switch (self.*) {
            // CPU intentionally stays on the queued scheduler path until it
            // advertises explicit fast-path support through the backend
            // support methods. The backend union itself exposes a planner, so
            // using the default helper here would incorrectly trust planner
            // probe flags that are only meaningful for GPU backends.
            .cpu => .queued,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "planSchedulerSingleDecodeRoute"))
                b.planSchedulerSingleDecodeRoute(plan)
            else
                defaultSchedulerSingleDecodeRoute(plan),
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "planSchedulerSingleDecodeRoute"))
                b.planSchedulerSingleDecodeRoute(plan)
            else
                defaultSchedulerSingleDecodeRoute(plan),
        };
    }

    pub fn supportsSchedulerBackendTopKCandidateSamplingRoute(
        self: *const Backend,
        sampling_config: *const cpu.sampling.SamplingConfig,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal)
                b.supportsSchedulerBackendTopKCandidateSamplingRoute(sampling_config)
            else
                unreachable,
            .cuda => false,
        };
    }

    pub fn supportsSchedulerBackendTopKStreamingRoute(
        self: *const Backend,
        sampling_config: *const cpu.sampling.SamplingConfig,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "supportsSchedulerBackendTopKStreamingRoute"))
                b.supportsSchedulerBackendTopKStreamingRoute(sampling_config)
            else
                false,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "supportsSchedulerBackendTopKStreamingRoute"))
                b.supportsSchedulerBackendTopKStreamingRoute(sampling_config)
            else
                false,
        };
    }

    pub fn lastDecodeComputeNs(self: *const Backend) ?u64 {
        return switch (self.*) {
            .cpu => null,
            .metal => null,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "lastDecodeComputeNs"))
                b.lastDecodeComputeNs()
            else
                null,
        };
    }

    pub fn decodeTopKCandidates(
        self: *Backend,
        slot_index: usize,
        token: u32,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        return switch (self.*) {
            .cpu => error.InvalidArgument,
            .metal => |*b| if (has_metal)
                b.decodeTopKCandidates(slot_index, token, top_k, candidate_logits_out, candidate_ids_out)
            else
                unreachable,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "decodeTopKCandidates"))
                b.decodeTopKCandidates(slot_index, token, top_k, candidate_logits_out, candidate_ids_out)
            else
                error.InvalidArgument,
        };
    }

    pub fn decodeTopKStreaming(
        self: *Backend,
        first_token: u32,
        start_position: usize,
        max_tokens: usize,
        eos_token_ids: []const u32,
        sampling_config: *const cpu.sampling.SamplingConfig,
        output_tokens: []u32,
        callback: ?*const fn (u32, ?*anyopaque) void,
        callback_data: ?*anyopaque,
        decode_ns_out: ?*u64,
    ) !usize {
        return switch (self.*) {
            .cpu => error.InvalidArgument,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "decodeTopKStreaming"))
                b.decodeTopKStreaming(
                    first_token,
                    start_position,
                    max_tokens,
                    eos_token_ids,
                    sampling_config,
                    output_tokens,
                    callback,
                    callback_data,
                    decode_ns_out,
                )
            else
                error.InvalidArgument,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "decodeTopKStreaming"))
                b.decodeTopKStreaming(
                    first_token,
                    start_position,
                    max_tokens,
                    eos_token_ids,
                    sampling_config,
                    output_tokens,
                    callback,
                    callback_data,
                    decode_ns_out,
                )
            else
                error.InvalidArgument,
        };
    }

    pub fn decodeTopKCandidatesWithSampling(
        self: *Backend,
        slot_index: usize,
        token: u32,
        sampling_config: *const cpu.sampling.SamplingConfig,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
    ) !usize {
        return switch (self.*) {
            .cpu => error.InvalidArgument,
            .metal => |*b| if (has_metal)
                b.decodeTopKCandidatesWithSampling(
                    slot_index,
                    token,
                    sampling_config,
                    candidate_logits_out,
                    candidate_ids_out,
                )
            else
                unreachable,
            .cuda => error.InvalidArgument,
        };
    }

    pub fn supportsSchedulerBackendBatchedTopKDecodeRoute(
        self: *const Backend,
        sampling_config: *const cpu.sampling.SamplingConfig,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "supportsSchedulerBackendBatchedTopKDecodeRoute"))
                b.supportsSchedulerBackendBatchedTopKDecodeRoute(sampling_config)
            else
                false,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "supportsSchedulerBackendBatchedTopKDecodeRoute"))
                b.supportsSchedulerBackendBatchedTopKDecodeRoute(sampling_config)
            else
                false,
        };
    }

    pub fn shouldUseSchedulerBatchedTopKDecodeRoute(
        self: *const Backend,
        plan: *const shared_scheduler.SchedulerBatchedTopKRoutePlan,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "shouldUseSchedulerBatchedTopKDecodeRoute"))
                b.shouldUseSchedulerBatchedTopKDecodeRoute(plan)
            else if (has_metal)
                plan.decode_batch_size >= 2 and b.supportsSchedulerBackendBatchedTopKDecodeRoute(plan.sampling_config)
            else
                false,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "shouldUseSchedulerBatchedTopKDecodeRoute"))
                b.shouldUseSchedulerBatchedTopKDecodeRoute(plan)
            else if (has_cuda and @hasDecl(cuda.BackendType, "supportsSchedulerBackendBatchedTopKDecodeRoute"))
                plan.decode_batch_size >= 2 and b.supportsSchedulerBackendBatchedTopKDecodeRoute(plan.sampling_config)
            else
                false,
        };
    }

    pub fn decodeBatchTopKCandidates(
        self: *Backend,
        requests: []const contract.DecodeRequest,
        top_k: usize,
        candidate_logits_out: []f32,
        candidate_ids_out: []u32,
        candidate_counts_out: []usize,
    ) !void {
        switch (self.*) {
            .cpu => return error.InvalidArgument,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "decodeBatchTopKCandidates"))
                return b.decodeBatchTopKCandidates(requests, top_k, candidate_logits_out, candidate_ids_out, candidate_counts_out)
            else
                return error.InvalidArgument,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "decodeBatchTopKCandidates"))
                return b.decodeBatchTopKCandidates(requests, top_k, candidate_logits_out, candidate_ids_out, candidate_counts_out)
            else
                return error.InvalidArgument,
        }
    }
    /// Get vocab size for this model
    pub fn vocabSize(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.vocab_size,
            .metal => |*b| if (has_metal) return b.vocabSize() else unreachable,
            .cuda => |*b| if (has_cuda) return b.vocab_size else unreachable,
        }
    }

    /// Warmup: do a dummy forward pass to pull weights into CPU cache
    /// This eliminates cold-cache latency on first real inference
    pub fn warmup(self: *Backend) !void {
        switch (self.*) {
            .cpu => |*b| try b.warmup(),
            .metal => |*b| if (has_metal) try b.warmup() else unreachable,
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
            .metal => |*b| if (has_metal) return b.embeddingDim() else unreachable,
            .cuda => |*b| if (has_cuda) return b.d_model else unreachable,
        }
    }

    // ---- Scheduler interface ----
    // These methods allow GenericScheduler to work with Backend directly,
    // keeping all architecture dispatch inside this module.

    pub fn maxBatchSize(self: *const Backend) usize {
        switch (self.*) {
            .cpu => |*b| return b.max_batch_size,
            .metal => |*b| if (has_metal) return b.maxBatchSize() else unreachable,
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

    pub fn prefillBatch(
        self: *Backend,
        requests: []const contract.PrefillBatchRequest,
    ) !void {
        switch (self.*) {
            .cpu => |*b| {
                for (requests) |request| {
                    try b.prefillSlot(request.slot_index, request.prompt_tokens, request.logits_out);
                }
            },
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "prefillBatch")) {
                try b.prefillBatch(requests);
            } else if (has_metal) {
                for (requests) |request| {
                    try b.prefillSlot(request.slot_index, request.prompt_tokens, request.logits_out);
                }
            } else unreachable,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "prefillBatch")) {
                try b.prefillBatch(requests);
            } else if (has_cuda) {
                for (requests) |request| {
                    try b.prefillSlot(request.slot_index, request.prompt_tokens, request.logits_out);
                }
            } else unreachable,
        }
    }

    pub fn prefillGreedySeedToken(
        self: *Backend,
        slot_index: usize,
        tokens: []const u32,
    ) !u32 {
        return switch (self.*) {
            .cpu => error.InvalidArgument,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "prefillGreedySeedToken"))
                b.prefillGreedySeedToken(slot_index, tokens)
            else
                error.InvalidArgument,
            .cuda => error.InvalidArgument,
        };
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

    pub fn takeLastVisionEncodeNs(self: *Backend) u64 {
        return switch (self.*) {
            .cpu => 0,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "takeLastVisionEncodeNs"))
                b.takeLastVisionEncodeNs()
            else
                0,
            .cuda => 0,
        };
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

    /// Set prefill progress callback when the backend exposes prefill progress.
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

    /// Set stop flag for cancellation during prefill (checked per-layer).
    pub fn setStopFlag(self: *Backend, flag: ?*const std.atomic.Value(bool)) void {
        switch (self.*) {
            .cpu => |*b| b.stop_flag = flag,
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
    runtime: *const models.config.ModelRuntime,
    weight_dtype: DType,
    has_unsupported_runtime_features: bool,
) bool {
    const dtype_supported = switch (weight_dtype) {
        .grouped_affine_u4, .grouped_affine_u8, .bf16, .f8_e4m3 => true,
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
    runtime: *const models.config.ModelRuntime,
    weight_dtype: DType,
    has_unsupported_runtime_features: bool,
) []const u8 {
    const dtype_supported = switch (weight_dtype) {
        .grouped_affine_u4, .grouped_affine_u8, .bf16, .f8_e4m3 => true,
        else => false,
    };
    if (!dtype_supported) {
        return "Weight dtype not supported by Metal (requires Q4/U8/BF16/FP8)";
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

fn runtimeHasMetalUnsupportedFeatures(runtime: *const models.config.ModelRuntime) bool {
    // Do not hard-reject runtime topologies during backend selection.
    // Auto mode should let Metal initialization make the concrete capability
    // decision so backend choice stays aligned with backend capabilities.
    _ = runtime;
    return false;
}

fn initCpu(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    progress: progress_mod.Context,
) !Backend {
    const model_max_seq: usize = @intCast(@max(@as(i32, 1), loaded.config.max_seq_len));
    const cpu_max_batch_size = resolveMaxBatchSize(.cpu);
    const cpu_max_seq_len = resolveCpuMaxSeqLenForRuntime(allocator, model_max_seq);
    log.info("inference", "CPU backend init config", .{
        .max_batch = cpu_max_batch_size,
        .runtime_max_seq_len = cpu_max_seq_len,
        .model_max_seq_len = model_max_seq,
    });

    const cpu_backend_state = cpu.BackendType.init(
        allocator,
        loaded,
        cpu_max_batch_size,
        cpu_max_seq_len,
        progress,
    ) catch |err| {
        log.warn("inference", "CPU backend init failed", .{
            .reason = @errorName(err),
            .arch = @tagName(loaded.config.model_arch),
            .has_gated_delta = loaded.runtime.has_gated_delta,
            .has_shortconv = loaded.runtime.has_shortconv,
            .has_mamba = loaded.runtime.has_mamba,
        });
        return err;
    };
    log.info("inference", "Backend selected: cpu", .{ .reason = reason });
    return .{ .cpu = cpu_backend_state };
}

fn initMetal(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    config: ?InitOptions.MetalConfig,
) !Backend {
    if (!has_metal) {
        return error.MetalNotEnabled;
    }
    if (!metal.isAvailable()) {
        log.info("inference", "Metal backend unavailable", .{
            .reason = "mlx bridge reported unavailable",
        });
        return error.MLXNotAvailable;
    }
    const has_unsupported_runtime_features = runtimeHasMetalUnsupportedFeatures(&loaded.runtime);
    if (!isMetalSupported(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features)) {
        log.info("inference", "Metal backend rejected model", .{
            .reason = getMetalUnsupportedReason(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features),
            .has_mamba = @as(u8, @intFromBool(loaded.runtime.has_mamba)),
            .has_gated_delta = @as(u8, @intFromBool(loaded.runtime.has_gated_delta)),
            .has_shortconv = @as(u8, @intFromBool(loaded.runtime.has_shortconv)),
            .has_mla = @as(u8, @intFromBool(loaded.runtime.has_mla)),
        });
        return error.UnsupportedModel;
    }
    const metal_backend_state = try metal.BackendType.init(allocator, loaded, .{
        .model_path = if (config) |c| c.model_path else null,
        .model_id = if (config) |c| c.model_id else null,
        .weights_path = if (config) |c| c.weights_path else null,
        .memory_fit_is_error = std.mem.eql(u8, reason, "configured"),
    });
    log.info("inference", "Backend selected: metal", .{ .reason = reason });
    return .{ .metal = metal_backend_state };
}

/// Renders a two-color bar showing CPU vs GPU layer split.
/// Yellow '#' = CPU layers, cyan '#' = GPU layers. 40-char width matches the Loading bar.
/// Example output: `[####################████████████████████] 32/32  cpu 20 · gpu 12`
fn renderColoredBar(
    buf: *[512]u8,
    mode: CudaTopologyMode,
    n_layers: usize,
    split_layer: ?usize,
) ?[*:0]const u8 {
    const bar_width: usize = 40;
    if (mode == .single or n_layers == 0) return null;

    const cpu_layers: usize = switch (mode) {
        .cpu_gpu, .cpu_gpu_gpu => split_layer orelse 0,
        .pipeline2 => 0,
        .single => unreachable,
    };
    const gpu_layers: usize = n_layers - cpu_layers;

    const w_cpu: usize = if (cpu_layers > 0) (cpu_layers * bar_width + n_layers / 2) / n_layers else 0;
    const w_gpu: usize = bar_width - w_cpu;

    const yellow = "\x1b[33m";
    const cyan = "\x1b[36m";
    const reset = "\x1b[0m";

    var stream = std.io.fixedBufferStream(buf);
    const w = stream.writer();

    w.writeAll("[") catch return null;
    if (w_cpu > 0) {
        w.writeAll(yellow) catch return null;
        for (0..w_cpu) |_| w.writeByte('#') catch return null;
        w.writeAll(reset) catch return null;
    }
    w.writeAll(cyan) catch return null;
    for (0..w_gpu) |_| w.writeByte('#') catch return null;
    w.writeAll(reset) catch return null;

    w.print("] {d}/{d}", .{ n_layers, n_layers }) catch return null;
    if (cpu_layers > 0) {
        w.print("  {s}cpu {d}{s} \xc2\xb7 {s}gpu {d}{s}", .{ yellow, cpu_layers, reset, cyan, gpu_layers, reset }) catch return null;
    }

    const pos = stream.pos;
    if (pos >= buf.len) return null;
    buf[pos] = 0;
    return @ptrCast(buf[0..pos :0]);
}

fn initCuda(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    probe: CudaProbe,
    topology_override: ?CudaTopologyConfig,
    progress: progress_mod.Context,
) !Backend {
    if (!has_cuda) {
        return error.CudaNotEnabled;
    }
    if (probe != .available) {
        log.info("inference", "CUDA runtime unavailable", .{ .reason = cudaProbeName(probe) });
        return error.CudaUnavailable;
    }
    var topology = try resolveCudaTopology(allocator, topology_override);
    const has_explicit_topology_env = std.process.hasEnvVar(allocator, "TALU_CUDA_TOPOLOGY") catch |err| {
        log.err("inference", "Failed to query TALU_CUDA_TOPOLOGY", .{ .err = @errorName(err) }, @src());
        return err;
    };
    const has_explicit_cpu_layers_env = std.process.hasEnvVar(allocator, "TALU_CPU_LAYERS") catch |err| {
        log.err("inference", "Failed to query TALU_CPU_LAYERS", .{ .err = @errorName(err) }, @src());
        return err;
    };

    // Topology resolution priority:
    //   1. topology_override (programmatic)
    //   2. TALU_CUDA_TOPOLOGY env var (explicit mode)
    //   3. TALU_CPU_LAYERS env var (user preference → auto-determines mode)
    //   4. Auto-detection (probe GPU memory, estimate model size)
    if (topology_override == null and topology.mode == .single) {
        if (!has_explicit_topology_env) {
            if (try resolveCpuLayersTopology(allocator, loaded)) |from_layers| {
                topology = from_layers;
            } else if (autoDetectTopologyForModel(allocator, loaded)) |detected| {
                topology = detected;
            } else |err| {
                log.err("inference", "Auto topology detection failed", .{
                    .err = @errorName(err),
                }, @src());
                return err;
            }
        }
    }

    if (topology_override == null and
        !has_explicit_topology_env and
        !has_explicit_cpu_layers_env and
        (topology.mode == .cpu_gpu or topology.mode == .cpu_gpu_gpu) and
        topology_mod.loadedModelHasPackedNvfp4Weights(loaded))
    {
        const primary_device = topology.primaryDeviceOrdinal();
        log.warn("inference", "CUDA auto-topology disabled CPU stages for packed NVFP4 model", .{
            .requested_topology = @tagName(topology.mode),
            .selected_topology = @tagName(CudaTopologyMode.single),
            .device = primary_device,
        });
        topology = .{
            .mode = .single,
            .stage_device_ordinals = .{ primary_device, primary_device },
        };
    }

    if (topology.mode != .single) {
        const device_count = if (has_cuda)
            compute.cuda.Device.deviceCount() catch |err| {
                log.err("inference", "Failed to query CUDA device count for topology validation", .{
                    .err = @errorName(err),
                }, @src());
                return err;
            }
        else
            0;
        validateCudaTopologyConfig(topology, loaded.blocks.len, device_count) catch |err| switch (err) {
            error.Pipeline2Unsupported => {
                log.err("inference", "Pipeline2 topology is not supported by current capability envelope", .{
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.Pipeline2InsufficientLayers => {
                log.err("inference", "Pipeline2 requires at least two decoder layers", .{
                    .total_layers = loaded.blocks.len,
                    .required = cuda_topology_capabilities.min_pipeline2_layers,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.Pipeline2InvalidSplitLayer => {
                log.err("inference", "Pipeline2 split layer is out of range", .{
                    .split_layer = if (topology.split_layer) |split| split else std.math.maxInt(usize),
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.Pipeline2RequiresDistinctDevices => {
                log.err("inference", "Pipeline2 requires two distinct CUDA device ordinals", .{
                    .ordinal0 = topology.stage_device_ordinals[0],
                    .ordinal1 = topology.stage_device_ordinals[1],
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.Pipeline2DeviceOrdinalOutOfRange => {
                log.err("inference", "Pipeline2 device ordinal out of range", .{
                    .ordinal0 = topology.stage_device_ordinals[0],
                    .ordinal1 = topology.stage_device_ordinals[1],
                    .device_count = device_count,
                }, @src());
                return error.CudaInvalidDevice;
            },
            error.CpuGpuUnsupported => {
                log.err("inference", "CPU+GPU topology is not supported by current capability envelope", .{
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuInsufficientLayers => {
                log.err("inference", "CPU+GPU topology requires at least two decoder layers", .{
                    .total_layers = loaded.blocks.len,
                    .required = cuda_topology_capabilities.min_cpu_gpu_layers,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuInvalidSplitLayer => {
                log.err("inference", "CPU+GPU split layer is out of range", .{
                    .split_layer = if (topology.split_layer) |split| split else std.math.maxInt(usize),
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuDeviceOrdinalOutOfRange => {
                log.err("inference", "CPU+GPU topology device ordinal out of range", .{
                    .ordinal0 = topology.stage_device_ordinals[0],
                    .device_count = device_count,
                }, @src());
                return error.CudaInvalidDevice;
            },
            error.CpuGpuGpuUnsupported => {
                log.err("inference", "CPU+GPU+GPU topology is not supported by current capability envelope", .{
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuGpuInsufficientLayers => {
                log.err("inference", "CPU+GPU+GPU topology requires at least three decoder layers", .{
                    .total_layers = loaded.blocks.len,
                    .required = cuda_topology_capabilities.min_cpu_gpu_gpu_layers,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuGpuInvalidSplitLayer => {
                log.err("inference", "CPU+GPU+GPU first split layer is out of range", .{
                    .split_layer = if (topology.split_layer) |split| split else std.math.maxInt(usize),
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuGpuInvalidSplitLayerStage2 => {
                log.err("inference", "CPU+GPU+GPU second split layer is out of range", .{
                    .split_layer_stage2 = if (topology.split_layer_stage2) |split| split else std.math.maxInt(usize),
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuGpuSplitOrderInvalid => {
                log.err("inference", "CPU+GPU+GPU split layers must satisfy split0 < split1 < total", .{
                    .split_layer = if (topology.split_layer) |split| split else std.math.maxInt(usize),
                    .split_layer_stage2 = if (topology.split_layer_stage2) |split| split else std.math.maxInt(usize),
                    .total_layers = loaded.blocks.len,
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuGpuRequiresDistinctDevices => {
                log.err("inference", "CPU+GPU+GPU requires two distinct CUDA device ordinals for gpu stages", .{
                    .ordinal0 = topology.stage_device_ordinals[0],
                    .ordinal1 = topology.stage_device_ordinals[1],
                }, @src());
                return error.InvalidTopologyConfig;
            },
            error.CpuGpuGpuDeviceOrdinalOutOfRange => {
                log.err("inference", "CPU+GPU+GPU topology device ordinal out of range", .{
                    .ordinal0 = topology.stage_device_ordinals[0],
                    .ordinal1 = topology.stage_device_ordinals[1],
                    .device_count = device_count,
                }, @src());
                return error.CudaInvalidDevice;
            },
        };
    }
    const cuda_max_batch_size = resolveMaxBatchSize(.cuda);
    const n_layers = loaded.blocks.len;
    const cpu_layer_count: usize = switch (topology.mode) {
        .cpu_gpu, .cpu_gpu_gpu => topology.split_layer orelse 0,
        .pipeline2, .single => 0,
    };
    const gpu_layer_count: usize = n_layers - cpu_layer_count;
    log.info("inference", "CUDA backend init config", .{
        .max_batch = cuda_max_batch_size,
        .topology = @tagName(topology.mode),
        .cpu_layers = cpu_layer_count,
        .gpu_layers = gpu_layer_count,
        .total_layers = n_layers,
        .device = topology.primaryDeviceOrdinal(),
    });
    const total_layers: u64 = @intCast(n_layers);

    // For multi-device topologies, render a colored bar in the message field.
    // Use spinner mode (total=0) so we control the entire visual via {msg}.
    // Yellow(\x1b[33m) = CPU, Cyan(\x1b[36m) = GPU0/gpu, Green(\x1b[32m) = GPU1.
    const use_colored_bar = topology.mode != .single;
    const bar_total: u64 = if (use_colored_bar) 0 else total_layers;

    progress.addLine(1, "Devices", bar_total, null, null);
    const cuda_backend_state = if (has_cuda)
        try cuda.BackendType.init(allocator, loaded, cuda_max_batch_size, .{
            .device_ordinal = topology.primaryDeviceOrdinal(),
            .topology_mode = topology.mode,
            .stage_device_ordinals = topology.stage_device_ordinals,
            .split_layer = topology.split_layer,
            .split_layer_stage2 = topology.split_layer_stage2,
            .progress = progress,
        })
    else
        unreachable;

    if (use_colored_bar) {
        var bar_buf: [512]u8 = undefined;
        if (renderColoredBar(&bar_buf, topology.mode, n_layers, topology.split_layer)) |msg| {
            progress.updateLine(1, 0, msg);
        }
    }
    progress.completeLine(1);
    log.info("inference", "Backend selected: cuda", .{ .reason = reason });
    return .{ .cuda = cuda_backend_state };
}

// ============================================================================
// Tests
// ============================================================================

test "isMetalSupported supports quantized and bf16" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(models.config.ModelRuntime);
    config.num_experts = 0;

    try std.testing.expect(isMetalSupported(&config, &runtime, .grouped_affine_u4, false));
    try std.testing.expect(isMetalSupported(&config, &runtime, .grouped_affine_u8, false));
    try std.testing.expect(isMetalSupported(&config, &runtime, .bf16, false));
}

test "isMetalSupported rejects unsupported dtypes" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(models.config.ModelRuntime);
    config.num_experts = 0;

    try std.testing.expect(!isMetalSupported(&config, &runtime, .f32, false));
    try std.testing.expect(!isMetalSupported(&config, &runtime, .f16, false));
}

test "isMetalSupported allows moe models" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(models.config.ModelRuntime);
    config.num_experts = 4;
    runtime.has_moe = true;

    try std.testing.expect(isMetalSupported(&config, &runtime, .grouped_affine_u4, false));
}

test "getMetalUnsupportedReason mentions dtype" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(models.config.ModelRuntime);
    config.num_experts = 0;

    const reason = getMetalUnsupportedReason(&config, &runtime, .f32, false);
    try std.testing.expect(std.mem.indexOf(u8, reason, "dtype") != null);
}

test "isMetalSupported rejects models when runtime features are unsupported" {
    var config = std.mem.zeroes(ModelConfig);
    var runtime = std.mem.zeroes(models.config.ModelRuntime);
    config.num_experts = 0;
    runtime.has_mamba = true;

    try std.testing.expect(!isMetalSupported(&config, &runtime, .grouped_affine_u4, true));
}

test "runtimeHasMetalUnsupportedFeatures does not pre-reject known topologies" {
    var runtime = std.mem.zeroes(models.config.ModelRuntime);
    try std.testing.expect(!runtimeHasMetalUnsupportedFeatures(&runtime));

    runtime.has_mla = true;
    try std.testing.expect(!runtimeHasMetalUnsupportedFeatures(&runtime));

    runtime.has_mla = false;
    runtime.has_mamba = true;
    try std.testing.expect(!runtimeHasMetalUnsupportedFeatures(&runtime));

    runtime.has_mamba = false;
    runtime.has_gated_delta = true;
    try std.testing.expect(!runtimeHasMetalUnsupportedFeatures(&runtime));
}

test "defaultModelLoadOptions follows platform capability" {
    const opts = defaultModelLoadOptions(.{});
    try std.testing.expectEqual(has_metal, opts.preserve_native_norm_dtype);
}

test "defaultModelLoadOptions honors explicit CPU selection" {
    const opts = defaultModelLoadOptions(.{ .selection = .cpu });
    try std.testing.expectEqual(false, opts.preserve_native_norm_dtype);
    try std.testing.expectEqual(true, opts.dequantize_mxfp8_to_bf16);
    try std.testing.expectEqual(true, opts.dequantize_nvfp4_to_bf16);
}

test "defaultModelLoadOptions honors explicit CUDA selection" {
    const opts = defaultModelLoadOptions(.{ .selection = .cuda });
    try std.testing.expectEqual(false, opts.preserve_native_norm_dtype);
    try std.testing.expectEqual(false, opts.dequantize_mxfp8_to_bf16);
    try std.testing.expectEqual(false, opts.dequantize_nvfp4_to_bf16);
}

test "defaultModelLoadOptions honors explicit metal selection" {
    const opts = defaultModelLoadOptions(.{ .selection = .metal });
    try std.testing.expectEqual(has_metal, opts.preserve_native_norm_dtype);
    try std.testing.expectEqual(false, opts.dequantize_mxfp8_to_bf16);
    try std.testing.expectEqual(true, opts.dequantize_nvfp4_to_bf16);
}

test "defaultMaxBatchSize keeps CPU default platform-scoped" {
    const expected_cpu_default: usize = if (builtin.os.tag == .windows) 1 else 8;
    try std.testing.expectEqual(expected_cpu_default, defaultMaxBatchSize(.cpu));
    try std.testing.expectEqual(@as(usize, 8), defaultMaxBatchSize(.cuda));
    try std.testing.expectEqual(@as(usize, 8), defaultMaxBatchSize(.metal));
    try std.testing.expectEqual(@as(usize, 8), defaultMaxBatchSize(.auto));
}

test "resolveCpuMaxSeqLenForRuntime keeps default cap platform-scoped" {
    const existing = std.process.getEnvVarOwned(std.testing.allocator, "TALU_CPU_MAX_SEQ_LEN") catch null;
    defer if (existing) |value| std.testing.allocator.free(value);
    if (existing != null) return error.SkipZigTest;

    const model_max: usize = 262144;
    const expected_default: usize = if (builtin.os.tag == .windows) 8192 else model_max;
    try std.testing.expectEqual(expected_default, resolveCpuMaxSeqLenForRuntime(std.testing.allocator, model_max));
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

test "parseSelectionOverrideToken rejects invalid BACKEND override" {
    try std.testing.expectError(error.InvalidArgument, parseSelectionOverrideToken(""));
    try std.testing.expectError(error.InvalidArgument, parseSelectionOverrideToken("rocm"));
}

test "optionalSelectionName returns tag or unset" {
    try std.testing.expectEqualStrings("unset", optionalSelectionName(null));
    try std.testing.expectEqualStrings("cpu", optionalSelectionName(.cpu));
    try std.testing.expectEqualStrings("metal", optionalSelectionName(.metal));
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
        initCuda(std.testing.allocator, undefined_loaded, "test", .disabled, null, progress_mod.Context.NONE),
    );
}

test "initCuda returns CudaUnavailable when runtime probe is unavailable" {
    if (!has_cuda) return;
    const undefined_loaded: *LoadedModel = undefined;
    try std.testing.expectError(
        error.CudaUnavailable,
        initCuda(std.testing.allocator, undefined_loaded, "test", .driver_not_found, null, progress_mod.Context.NONE),
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

test "supportsSchedulerBackendTopKDecodeRoute: cpu disabled" {
    const cpu_backend: Backend = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 40,
        .temperature = 0.7,
        .min_p = 0.0,
    };
    try std.testing.expectEqual(false, cpu_backend.supportsSchedulerBackendTopKDecodeRoute(&sampling_config));
}

test "planSchedulerSingleDecodeRoute: cpu keeps unsupported fast paths queued" {
    const cpu_backend: Backend = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 20,
        .temperature = 1.0,
        .top_p = 0.95,
    };

    const route = cpu_backend.planSchedulerSingleDecodeRoute(&.{
        .sampling_config = &sampling_config,
        .decode_batch_size = 1,
        .has_callback = true,
        .capture_final_logits = false,
        .has_grammar_sampler = false,
        .prompt_token_count = 8,
        .greedy_streaming_semantic_eligible = true,
        .top_k_streaming_semantic_eligible = true,
        .top_k_candidate_semantic_eligible = true,
        .greedy_streaming_backend_supported = true,
        .top_k_streaming_backend_supported = true,
        .top_k_candidate_backend_supported = true,
    });

    try std.testing.expectEqual(shared_scheduler.SchedulerSingleDecodeRoute.queued, route);
}

test "supportsSchedulerBackendTopKDecodeRoute: cuda disabled" {
    if (!has_cuda) return;
    const cuda_backend: Backend = .{ .cuda = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 40,
        .temperature = 0.7,
        .min_p = 0.0,
    };
    try std.testing.expectEqual(false, cuda_backend.supportsSchedulerBackendTopKDecodeRoute(&sampling_config));
}

test "supportsSchedulerBackendBatchedTopKDecodeRoute: cpu disabled" {
    const cpu_backend: Backend = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 40,
    };
    try std.testing.expectEqual(false, cpu_backend.supportsSchedulerBackendBatchedTopKDecodeRoute(&sampling_config));
}

test "supportsSchedulerBackendBatchedTopKDecodeRoute: metal delegated" {
    if (!has_metal) return;
    const metal_backend: Backend = .{ .metal = undefined };
    const valid_sampling = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 64,
    };
    const invalid_sampling = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 0,
    };
    try std.testing.expect(metal_backend.supportsSchedulerBackendBatchedTopKDecodeRoute(&valid_sampling));
    try std.testing.expect(!metal_backend.supportsSchedulerBackendBatchedTopKDecodeRoute(&invalid_sampling));
}

test "metal module surface keeps runtime graph internals private" {
    if (!has_metal) return;
    try std.testing.expect(!@hasDecl(metal, "runtime_graph"));
}

test "metal module surface maps to shared helper modules without vision type aliasing" {
    if (!has_metal) return;
    try std.testing.expect(metal.executor.Model == cpu.executor.Model);
    try std.testing.expect(metal.kernels.RMSNorm == cpu.kernels.RMSNorm);
    try std.testing.expect(metal.vision.VisionRuntime != cpu.vision.VisionRuntime);
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

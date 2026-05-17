//! Scheduler execution-target selection and ownership.
//!
//! The scheduler drives this interface. The concrete target can be a single
//! backend or a local pipeline that owns multiple backend stages.
//!
//! ## Target Selection
//!
//! Execution target initialization selects based on:
//! 1. Environment override (`BACKEND=cpu|metal|cuda|auto`) when selection is `.auto`
//! 2. Build flags (`enable_cuda`, `enable_metal`)
//! 3. Platform (CUDA on Linux/Windows, Metal on macOS)
//! 4. Model compatibility checks (for Metal)
//!
//! Note: CUDA is opt-in only for now. Auto-selection does not choose CUDA.
//!
//! ## Supported Targets
//!
//! | Target | Type | Description |
//! |---------|------|------------|
//! | `cpu`   | Batched (FusedCpuBackend) | Production graph-based inference |
//! | `metal` | Lazy graph (MetalBackend) | Production GPU inference (macOS) |
//! | `cuda`  | CUDA backend | GPU inference with single-device and staged topologies |
const std = @import("std");
const builtin = @import("builtin");

const backend = @import("../backend/root.zig");
const models = @import("models_pkg");
const log = @import("log_pkg");
const progress_mod = @import("progress_pkg");
const compute = @import("compute_pkg");
const runtime_contract = @import("runtime_contract_pkg");
const local_pipeline_runtime = @import("../pipeline/local_pipeline_runtime.zig");
const local_stage_topology = @import("../pipeline/local_pipeline_topology.zig");
const device_summary = @import("device_summary.zig");
const local_stage_config = @import("local_stage_config.zig");
const local_pipeline_factory = @import("local_pipeline_factory.zig");
const ModelConfig = models.config.ModelConfig;
const dtype_mod = @import("compute_pkg").dtype;
const DType = dtype_mod.DType;
const LoadedModel = models.LoadedModel;
const LoadOptions = models.LoadOptions;

pub const contract = backend.contract;
const cpu = backend.cpu;
const metal = backend.metal;
const cuda = backend.cuda;

/// Re-export types used by the scheduler interface
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;
pub const PrefillProgressFn = cpu.BackendType.PrefillProgressFn;

/// Re-export pooling strategy for embedding extraction
pub const PoolingStrategy = contract.PoolingStrategy;
pub const has_metal = backend.has_metal;
const has_cuda = backend.has_cuda;
const shared_scheduler = @import("contracts.zig");

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

/// ExecutionTarget initialization options selected at startup/config layer.
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
    /// Ordered local stage plan for CUDA-backed local execution. When null,
    /// CUDA uses a single local stage or auto-detects a local stage plan.
    local_stage_specs: ?[]const local_stage_topology.LocalStageSpec = null,
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

/// ExecutionTarget type - tagged union of available backends
pub const ExecutionTarget = union(enum) {
    /// Fused CPU backend for graph ops (production inference)
    cpu: cpu.BackendType,
    /// Metal GPU backend (macOS only).
    metal: if (has_metal) metal.BackendType else void,
    /// CUDA backend (Linux/Windows when built with CUDA support).
    cuda: if (has_cuda) cuda.BackendType else void,
    /// Pipeline-owned local CPU/CUDA pipeline runtime.
    local_pipeline: local_pipeline_runtime.LocalPipelineRuntime,

    /// Vision input type for prefillSlotWithVision (shared across backends)
    pub const PrefillVisionInput = cpu.BackendType.PrefillVisionInput;

    /// Which generation strategy to use for a given request.
    pub const GenerationPath = enum {
        /// Continuous batching via GenericScheduler
        scheduler,
    };

    /// Select the generation path for this backend and request.
    pub fn generationPath(self: ExecutionTarget, has_input_images: bool) GenerationPath {
        _ = self;
        _ = has_input_images;
        return .scheduler;
    }

    /// Whether scheduler should route a single-request decode tail through
    /// backend-owned streaming for this sampling configuration.
    pub fn supportsSchedulerBackendStreamingRoute(
        self: *const ExecutionTarget,
        sampling_config: *const cpu.sampling.SamplingConfig,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "supportsSchedulerBackendStreamingRoute"))
                b.supportsSchedulerBackendStreamingRoute(sampling_config)
            else
                false,
            .cuda => false,
            .local_pipeline => false,
        };
    }

    /// Initialize the appropriate backend based on platform and model format.
    /// Auto order: Metal (if supported) -> CPU.
    /// CUDA is selected only when explicitly configured.
    pub fn init(
        allocator: std.mem.Allocator,
        loaded: *LoadedModel,
        init_options: InitOptions,
        progress: progress_mod.Context,
    ) !ExecutionTarget {
        const env_override = if (init_options.selection == .auto)
            try selectionOverrideFromEnv(allocator)
        else
            null;
        const selected = if (init_options.selection == .auto)
            (env_override orelse .auto)
        else
            init_options.selection;
        const cuda_probe = probeCudaRuntime();

        log.info("inference", "Execution target init policy", .{
            .requested = selectionName(init_options.selection),
            .env_override = optionalSelectionName(env_override),
            .effective = selectionName(selected),
            .cuda_runtime = cudaProbeName(cuda_probe),
            .cuda_enabled = @as(u8, @intFromBool(has_cuda)),
            .metal_enabled = @as(u8, @intFromBool(has_metal)),
        });

        switch (selected) {
            .cpu => return initCpu(allocator, loaded, "configured", progress),
            .metal => return initMetal(allocator, loaded, "configured", init_options.metal, progress),
            .cuda => return initCuda(allocator, loaded, "configured", cuda_probe, init_options.local_stage_specs, progress),
            .auto => {},
        }

        // Check if we should use Metal backend (macOS + quantized/bf16 model)
        const has_unsupported_runtime_features = runtimeHasMetalUnsupportedFeatures(&loaded.runtime);
        if (has_metal and isMetalSupported(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features)) {
            return initMetal(allocator, loaded, "auto", init_options.metal, progress);
        }

        // Default to CPU backend
        return initCpu(allocator, loaded, "default", progress);
    }

    /// Clean up backend resources
    pub fn deinit(self: *ExecutionTarget) void {
        switch (self.*) {
            .cpu => |*b| b.deinit(),
            .metal => |*b| if (has_metal) b.deinit() else unreachable,
            .cuda => |*b| if (has_cuda) b.deinit() else unreachable,
            .local_pipeline => |*b| b.deinit(),
        }
    }

    pub fn canReleaseLoadedModel(self: *ExecutionTarget) bool {
        return switch (self.*) {
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "canReleaseLoadedModel"))
                b.canReleaseLoadedModel()
            else
                false,
            else => false,
        };
    }

    /// Explicit device/barrier synchronization for correctness-sensitive
    /// observability flows such as xray capture finalization.
    ///
    /// This must never change backend math or route selection. It exists so
    /// boundary code can guarantee that backend work has finished producing any
    /// host-visible trace/capture outputs before those outputs are serialized
    /// or destroyed during teardown.
    pub fn synchronize(self: *ExecutionTarget) !void {
        switch (self.*) {
            .cpu => {},
            .metal => |*b| if (has_metal) b.synchronize() else unreachable,
            .cuda => |*b| if (has_cuda) try b.synchronize() else unreachable,
            .local_pipeline => |*b| try b.synchronize(),
        }
    }

    /// Explicit end-of-run cleanup for backend state that is thread-local to
    /// the execution thread. This must not change math or route selection; it
    /// only clears transient per-run resources after a generation completes.
    pub fn cleanupExecutionThreadState(self: *ExecutionTarget) void {
        switch (self.*) {
            .cpu => {},
            .metal => |*b| if (has_metal) b.cleanupExecutionThreadState() else unreachable,
            .cuda => {},
            .local_pipeline => {},
        }
    }

    /// Explicit execution-thread teardown barrier.
    ///
    /// Call this only on the thread that actually executed backend work and
    /// only when that thread is about to stop issuing more work. This is
    /// distinct from per-run cleanup: it is allowed to destroy thread-local
    /// caches whose contents must not outlive the worker thread lifecycle.
    pub fn teardownExecutionThreadState(self: *ExecutionTarget) void {
        switch (self.*) {
            .cpu => {},
            .metal => |*b| if (has_metal) b.teardownExecutionThreadState() else unreachable,
            .cuda => {},
            .local_pipeline => {},
        }
    }

    /// Prefill: process all prompt tokens, return logits for last position
    /// This resets the KV cache and processes the full prompt
    pub fn prefill(self: *ExecutionTarget, tokens: []const u32, logits_out: []f32) !void {
        switch (self.*) {
            .cpu => |*b| try b.prefill(tokens, logits_out),
            .metal => |*b| if (has_metal) try b.prefill(tokens, logits_out) else unreachable,
            .cuda => |*b| if (has_cuda) try b.prefill(tokens, logits_out) else unreachable,
            .local_pipeline => |*b| try b.prefill(tokens, logits_out),
        }
    }

    /// Decode: generate logits for a single token using KV cache
    /// Returns logits for the next token prediction
    pub fn decode(self: *ExecutionTarget, token: u32, position: usize, logits_out: []f32) !void {
        switch (self.*) {
            .cpu => |*b| try b.decode(token, position, logits_out),
            .metal => |*b| if (has_metal) try b.decode(token, position, logits_out) else unreachable,
            .cuda => |*b| if (has_cuda) try b.decode(token, position, logits_out) else unreachable,
            .local_pipeline => |*b| try b.decode(token, position, logits_out),
        }
    }

    /// ExecutionTarget-owned streaming token generation for a single-request decode tail.
    pub fn decodeSchedulerStreaming(
        self: *ExecutionTarget,
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
        switch (self.*) {
            .cpu => return error.UnsupportedModel,
            .metal => |*b| if (has_metal) {
                return b.decodeSchedulerStreaming(
                    first_token,
                    start_position,
                    max_tokens,
                    eos_token_ids,
                    sampling_config,
                    output_tokens,
                    callback,
                    callback_data,
                    decode_ns_out,
                );
            } else unreachable,
            .cuda => return error.UnsupportedModel,
            .local_pipeline => return error.UnsupportedModel,
        }
    }

    pub fn shouldUseSchedulerTopKCandidateRoute(
        self: *const ExecutionTarget,
        plan: *const shared_scheduler.SchedulerTopKCandidateRoutePlan,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "shouldUseSchedulerTopKCandidateRoute"))
                b.shouldUseSchedulerTopKCandidateRoute(plan)
            else
                false,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "shouldUseSchedulerTopKCandidateRoute"))
                b.shouldUseSchedulerTopKCandidateRoute(plan)
            else
                false,
            .local_pipeline => false,
        };
    }

    pub fn supportsSchedulerBackendTopKCandidateSamplingRoute(
        self: *const ExecutionTarget,
        sampling_config: *const cpu.sampling.SamplingConfig,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal)
                b.supportsSchedulerBackendTopKCandidateSamplingRoute(sampling_config)
            else
                unreachable,
            .cuda => false,
            .local_pipeline => false,
        };
    }

    pub fn lastDecodeComputeNs(self: *const ExecutionTarget) ?u64 {
        return switch (self.*) {
            .cpu => null,
            .metal => null,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "lastDecodeComputeNs"))
                b.lastDecodeComputeNs()
            else
                null,
            .local_pipeline => null,
        };
    }

    pub fn decodeTopKCandidates(
        self: *ExecutionTarget,
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
            .local_pipeline => |*b| b.decodeTopKCandidates(slot_index, token, top_k, candidate_logits_out, candidate_ids_out),
        };
    }

    pub fn decodeTopKCandidatesWithSampling(
        self: *ExecutionTarget,
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
            .local_pipeline => error.InvalidArgument,
        };
    }

    pub fn shouldUseSchedulerBatchedTopKDecodeRoute(
        self: *const ExecutionTarget,
        plan: *const shared_scheduler.SchedulerBatchedTopKRoutePlan,
    ) bool {
        return switch (self.*) {
            .cpu => false,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "shouldUseSchedulerBatchedTopKDecodeRoute"))
                b.shouldUseSchedulerBatchedTopKDecodeRoute(plan)
            else
                false,
            .cuda => |*b| if (has_cuda and @hasDecl(cuda.BackendType, "shouldUseSchedulerBatchedTopKDecodeRoute"))
                b.shouldUseSchedulerBatchedTopKDecodeRoute(plan)
            else
                false,
            .local_pipeline => false,
        };
    }

    pub fn decodeBatchTopKCandidates(
        self: *ExecutionTarget,
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
            .local_pipeline => |*b| return b.decodeBatchTopKCandidates(requests, top_k, candidate_logits_out, candidate_ids_out, candidate_counts_out),
        }
    }
    /// Get vocab size for this model
    pub fn vocabSize(self: *const ExecutionTarget) usize {
        switch (self.*) {
            .cpu => |*b| return b.vocab_size,
            .metal => |*b| if (has_metal) return b.vocabSize() else unreachable,
            .cuda => |*b| if (has_cuda) return b.vocab_size else unreachable,
            .local_pipeline => |*b| return b.vocab_size,
        }
    }

    /// Warmup: do a dummy forward pass to pull weights into CPU cache
    /// This eliminates cold-cache latency on first real inference
    pub fn warmup(self: *ExecutionTarget) !void {
        switch (self.*) {
            .cpu => |*b| try b.warmup(),
            .metal => |*b| if (has_metal) try b.warmup() else unreachable,
            .cuda => {},
            .local_pipeline => {},
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
        self: *ExecutionTarget,
        tokens: []const u32,
        pooling: PoolingStrategy,
        normalize: bool,
        embedding_buffer: []f32,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.embed(tokens, pooling, normalize, embedding_buffer),
            .metal => |*b| if (has_metal) try b.embed(tokens, pooling, normalize, embedding_buffer) else unreachable,
            .cuda => return error.EmbeddingNotSupported,
            .local_pipeline => return error.EmbeddingNotSupported,
        }
    }

    /// Returns the embedding dimension (d_model) for this model.
    pub fn embeddingDim(self: *const ExecutionTarget) usize {
        switch (self.*) {
            .cpu => |*b| return b.embeddingDim(),
            .metal => |*b| if (has_metal) return b.embeddingDim() else unreachable,
            .cuda => |*b| if (has_cuda) return b.d_model else unreachable,
            .local_pipeline => |*b| return b.d_model,
        }
    }

    // ---- Scheduler interface ----
    // These methods allow GenericScheduler to work with ExecutionTarget directly,
    // keeping all architecture dispatch inside this module.

    pub fn maxBatchSize(self: *const ExecutionTarget) usize {
        switch (self.*) {
            .cpu => |*b| return b.max_batch_size,
            .metal => |*b| if (has_metal) return b.maxBatchSize() else unreachable,
            .cuda => |*b| if (has_cuda) return b.max_batch_size else unreachable,
            .local_pipeline => |*b| return b.max_batch_size,
        }
    }

    pub fn allocSlot(self: *ExecutionTarget) ?usize {
        switch (self.*) {
            .cpu => |*b| return b.allocSlot(),
            .metal => |*b| if (has_metal) return b.allocSlot() else unreachable,
            .cuda => |*b| if (has_cuda) return b.allocSlot() else unreachable,
            .local_pipeline => |*b| return b.allocSlot(),
        }
    }

    pub fn freeSlot(self: *ExecutionTarget, slot_index: usize) void {
        switch (self.*) {
            .cpu => |*b| b.freeSlot(slot_index),
            .metal => |*b| if (has_metal) b.freeSlot(slot_index) else unreachable,
            .cuda => |*b| if (has_cuda) b.freeSlot(slot_index) else unreachable,
            .local_pipeline => |*b| b.freeSlot(slot_index),
        }
    }

    pub fn stateDescriptors(self: *const ExecutionTarget) []const runtime_contract.StateDescriptor {
        switch (self.*) {
            .cpu => |*b| return b.stateDescriptors(),
            .metal => |*b| if (has_metal) return b.stateDescriptors() else unreachable,
            .cuda => |*b| if (has_cuda) return b.stateDescriptors() else unreachable,
            .local_pipeline => |*b| return b.stateDescriptors(),
        }
    }

    pub fn bindSlotStateBlocks(
        self: *ExecutionTarget,
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
            .local_pipeline => |*b| try b.bindSlotStateBlocks(slot_index, state_blocks),
        }
    }

    pub fn unbindSlotStateBlocks(self: *ExecutionTarget, slot_index: usize) void {
        switch (self.*) {
            .cpu => |*b| b.unbindSlotStateBlocks(slot_index),
            .metal => |*b| if (has_metal) b.unbindSlotStateBlocks(slot_index) else unreachable,
            .cuda => |*b| if (has_cuda) b.unbindSlotStateBlocks(slot_index) else unreachable,
            .local_pipeline => |*b| b.unbindSlotStateBlocks(slot_index),
        }
    }

    pub fn prefillSlot(
        self: *ExecutionTarget,
        slot_index: usize,
        tokens: []const u32,
        logits_out: []f32,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.prefillSlot(slot_index, tokens, logits_out),
            .metal => |*b| if (has_metal) try b.prefillSlot(slot_index, tokens, logits_out) else unreachable,
            .cuda => |*b| if (has_cuda) try b.prefillSlot(slot_index, tokens, logits_out) else unreachable,
            .local_pipeline => |*b| try b.prefillSlot(slot_index, tokens, logits_out),
        }
    }

    pub fn prefillBatch(
        self: *ExecutionTarget,
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
            .local_pipeline => |*b| try b.prefillBatch(requests),
        }
    }

    pub fn prefillGreedySeedToken(
        self: *ExecutionTarget,
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
            .local_pipeline => error.InvalidArgument,
        };
    }

    pub fn prefillSlotWithVision(
        self: *ExecutionTarget,
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
            .local_pipeline => |*b| {
                const erased: ?*const anyopaque = if (vision_input) |input| @ptrCast(input) else null;
                try b.prefillSlotWithVision(slot_index, tokens, erased, logits_out);
            },
        }
    }

    pub fn takeLastVisionEncodeNs(self: *ExecutionTarget) u64 {
        return switch (self.*) {
            .cpu => 0,
            .metal => |*b| if (has_metal and @hasDecl(metal.BackendType, "takeLastVisionEncodeNs"))
                b.takeLastVisionEncodeNs()
            else
                0,
            .cuda => 0,
            .local_pipeline => 0,
        };
    }

    pub fn decodeBatch(
        self: *ExecutionTarget,
        requests: []const DecodeRequest,
        results: []DecodeResult,
    ) !void {
        switch (self.*) {
            .cpu => |*b| try b.decodeBatch(requests, results),
            .metal => |*b| if (has_metal) try b.decodeBatch(requests, results) else unreachable,
            .cuda => |*b| if (has_cuda) try b.decodeBatch(requests, results) else unreachable,
            .local_pipeline => |*b| try b.decodeBatch(requests, results),
        }
    }

    /// Set prefill progress callback when the backend exposes prefill progress.
    pub fn setPrefillProgress(
        self: *ExecutionTarget,
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
            .local_pipeline => {},
        }
    }

    /// Set stop flag for cancellation during prefill (checked per-layer).
    pub fn setStopFlag(self: *ExecutionTarget, flag: ?*const std.atomic.Value(bool)) void {
        switch (self.*) {
            .cpu => |*b| b.stop_flag = flag,
            .metal => {},
            .cuda => {},
            .local_pipeline => {},
        }
    }

    /// Maximum pixel count the vision encoder can handle efficiently.
    pub fn visionMaxPixels(self: *const ExecutionTarget) u64 {
        return switch (self.*) {
            .cpu => cpu.vision.maxPixels(),
            .metal => if (has_metal) metal.vision.maxPixels() else unreachable,
            .cuda => if (has_cuda) cuda.vision.maxPixels() else unreachable,
            .local_pipeline => cpu.vision.maxPixels(),
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
) !ExecutionTarget {
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
        .{
            .max_batch_size = cpu_max_batch_size,
            .max_sequence_len = cpu_max_seq_len,
            .layer_range = .{ .start = 0, .end = @intCast(loaded.config.n_layers) },
            .build_logits_head = true,
            .progress = progress,
        },
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
    progress.completeLine(1);
    device_summary.publish(progress, .{
        .cpu_layers = loaded.blocks.len,
    });
    log.info("inference", "Execution target selected: cpu", .{ .reason = reason });
    return .{ .cpu = cpu_backend_state };
}

fn initMetal(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    config: ?InitOptions.MetalConfig,
    progress: progress_mod.Context,
) !ExecutionTarget {
    if (!has_metal) {
        return error.MetalNotEnabled;
    }
    if (!metal.isAvailable()) {
        log.info("inference", "Metal backend unavailable", .{
            .reason = "mlx runtime reported unavailable",
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
    progress.completeLine(1);
    device_summary.publish(progress, .{
        .metal_layers = loaded.blocks.len,
    });
    log.info("inference", "Execution target selected: metal", .{ .reason = reason });
    return .{ .metal = metal_backend_state };
}

fn initCuda(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    probe: CudaProbe,
    local_stage_override: ?[]const local_stage_topology.LocalStageSpec,
    progress: progress_mod.Context,
) !ExecutionTarget {
    if (!has_cuda) {
        return error.CudaNotEnabled;
    }
    if (probe != .available) {
        log.info("inference", "CUDA runtime unavailable", .{ .reason = cudaProbeName(probe) });
        return error.CudaUnavailable;
    }
    const resolved = try local_stage_config.resolveCudaLocalStagePlan(allocator, loaded, local_stage_override);
    const plan = resolved.plan;
    const resolved_summary = resolved.summary;
    const cuda_max_batch_size = resolveMaxBatchSize(.cuda);
    const n_layers = loaded.blocks.len;
    const cpu_layer_count = resolved_summary.cpu_layers;
    const gpu_layer_count = resolved_summary.gpu0_layers + resolved_summary.gpu1_layers;
    const gpu0_layer_count = resolved_summary.gpu0_layers;
    const gpu1_layer_count = resolved_summary.gpu1_layers;
    log.info("inference", "CUDA backend init config", .{
        .max_batch = cuda_max_batch_size,
        .stage_count = plan.stage_count,
        .cpu_layers = cpu_layer_count,
        .gpu_layers = gpu_layer_count,
        .gpu0_layers = gpu0_layer_count,
        .gpu1_layers = gpu1_layer_count,
        .total_layers = n_layers,
        .device = plan.primaryDeviceOrdinal(),
        .runtime_pipeline = @as(u8, @intFromBool(!plan.isSingleCudaStage())),
    });
    const total_layers: u64 = @intCast(n_layers);

    // Multi-stage plans use spinner mode so the completed line can be the
    // colored layer ownership summary. Single-GPU keeps the load progress bar
    // during init, then publishes the same summary before completion.
    const bar_total: u64 = if (!plan.isSingleCudaStage()) 0 else total_layers;

    progress.addLine(1, "Loading", bar_total, null, null);
    if (!plan.isSingleCudaStage()) {
        return initLocalPipeline(allocator, loaded, reason, plan, cuda_max_batch_size, resolved_summary, progress);
    }

    const single_stage = plan.stages[0];
    const cuda_backend_state = if (has_cuda)
        try cuda.BackendType.init(allocator, loaded, cuda_max_batch_size, .{
            .device_ordinal = single_stage.device_ordinal orelse plan.primaryDeviceOrdinal(),
            .layer_range = .{ .start = single_stage.layer_start, .end = single_stage.layer_end },
            .owns_embedding = single_stage.owns_embedding,
            .owns_projection = single_stage.owns_projection,
            .progress = progress,
        })
    else
        unreachable;

    progress.completeLine(1);
    device_summary.publish(progress, resolved_summary);
    log.info("inference", "Execution target selected: cuda", .{ .reason = reason });
    return .{ .cuda = cuda_backend_state };
}

fn initLocalPipeline(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    plan: local_stage_topology.LocalStagePlan,
    max_batch_size: usize,
    summary: device_summary.DeviceLayerSummary,
    progress: progress_mod.Context,
) !ExecutionTarget {
    if (plan.stage_count < 2) return error.InvalidTopologyConfig;
    const model_max_seq: usize = @intCast(@max(@as(i32, 1), loaded.config.max_seq_len));
    const cpu_max_seq_len = resolveCpuMaxSeqLenForRuntime(allocator, model_max_seq);
    const load_semantics = models.stage_plan.LoadSemantics.fromLoadOptions(defaultModelLoadOptions(.{ .selection = .cuda }));
    const runtime = try local_pipeline_factory.init(allocator, loaded, plan, max_batch_size, cpu_max_seq_len, load_semantics);

    progress.completeLine(1);
    device_summary.publish(progress, summary);
    log.info("inference", "Execution target selected: local_pipeline", .{
        .reason = reason,
        .stage_count = plan.stage_count,
    });
    return .{ .local_pipeline = runtime };
}

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
    const cpu_backend: ExecutionTarget = .{ .cpu = undefined };
    try std.testing.expectEqual(ExecutionTarget.GenerationPath.scheduler, cpu_backend.generationPath(false));
    try std.testing.expectEqual(ExecutionTarget.GenerationPath.scheduler, cpu_backend.generationPath(true));
}

test "generationPath: metal always selects scheduler" {
    if (!has_metal) return; // Metal variant is void on non-Metal platforms
    const metal_backend: ExecutionTarget = .{ .metal = undefined };
    try std.testing.expectEqual(ExecutionTarget.GenerationPath.scheduler, metal_backend.generationPath(true));
    try std.testing.expectEqual(ExecutionTarget.GenerationPath.scheduler, metal_backend.generationPath(false));
}

test "generationPath: cuda always selects scheduler" {
    if (!has_cuda) return;
    const cuda_backend: ExecutionTarget = .{ .cuda = undefined };
    try std.testing.expectEqual(ExecutionTarget.GenerationPath.scheduler, cuda_backend.generationPath(true));
    try std.testing.expectEqual(ExecutionTarget.GenerationPath.scheduler, cuda_backend.generationPath(false));
}

test "supportsSchedulerBackendStreamingRoute: cpu disabled" {
    const cpu_backend: ExecutionTarget = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{};
    try std.testing.expectEqual(false, cpu_backend.supportsSchedulerBackendStreamingRoute(&sampling_config));
}

test "supportsSchedulerBackendStreamingRoute: metal delegated" {
    if (!has_metal) return;
    const metal_backend: ExecutionTarget = .{ .metal = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{};
    try std.testing.expectEqual(true, metal_backend.supportsSchedulerBackendStreamingRoute(&sampling_config));
}

test "supportsSchedulerBackendStreamingRoute: cuda disabled" {
    if (!has_cuda) return;
    const cuda_backend: ExecutionTarget = .{ .cuda = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{};
    try std.testing.expectEqual(false, cuda_backend.supportsSchedulerBackendStreamingRoute(&sampling_config));
}

test "decodeSchedulerStreaming: cpu returns unsupported route error" {
    var cpu_backend: ExecutionTarget = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{};
    var output_tokens: [2]u32 = undefined;
    var decode_ns: u64 = 99;
    try std.testing.expectError(
        error.UnsupportedModel,
        cpu_backend.decodeSchedulerStreaming(
            1,
            0,
            output_tokens.len,
            &.{},
            &sampling_config,
            output_tokens[0..],
            null,
            null,
            &decode_ns,
        ),
    );
    try std.testing.expectEqual(@as(u64, 99), decode_ns);
}

test "decodeSchedulerStreaming: cuda returns unsupported route error" {
    if (!has_cuda) return;
    var cuda_backend: ExecutionTarget = .{ .cuda = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{};
    var output_tokens: [2]u32 = undefined;
    var decode_ns: u64 = 99;
    try std.testing.expectError(
        error.UnsupportedModel,
        cuda_backend.decodeSchedulerStreaming(
            1,
            0,
            output_tokens.len,
            &.{},
            &sampling_config,
            output_tokens[0..],
            null,
            null,
            &decode_ns,
        ),
    );
    try std.testing.expectEqual(@as(u64, 99), decode_ns);
}

test "shouldUseSchedulerTopKCandidateRoute: cpu disabled" {
    const cpu_backend: ExecutionTarget = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 40,
        .temperature = 0.7,
        .min_p = 0.0,
    };
    try std.testing.expectEqual(false, cpu_backend.shouldUseSchedulerTopKCandidateRoute(&.{
        .sampling_config = &sampling_config,
        .has_callback = true,
    }));
}

test "shouldUseSchedulerTopKCandidateRoute: cuda disabled" {
    if (!has_cuda) return;
    const cuda_backend: ExecutionTarget = .{ .cuda = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 40,
        .temperature = 0.7,
        .min_p = 0.0,
    };
    try std.testing.expectEqual(false, cuda_backend.shouldUseSchedulerTopKCandidateRoute(&.{
        .sampling_config = &sampling_config,
        .has_callback = true,
    }));
}

test "shouldUseSchedulerBatchedTopKDecodeRoute: cpu disabled" {
    const cpu_backend: ExecutionTarget = .{ .cpu = undefined };
    const sampling_config = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 40,
    };
    try std.testing.expectEqual(false, cpu_backend.shouldUseSchedulerBatchedTopKDecodeRoute(&.{
        .decode_batch_size = 1,
        .route_top_k = 40,
        .sampling_config = &sampling_config,
    }));
}

test "shouldUseSchedulerBatchedTopKDecodeRoute: metal delegated" {
    if (!has_metal) return;
    const metal_backend: ExecutionTarget = .{ .metal = undefined };
    const valid_sampling = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 64,
    };
    const invalid_sampling = cpu.sampling.SamplingConfig{
        .strategy = .top_k,
        .top_k = 0,
    };
    try std.testing.expect(metal_backend.shouldUseSchedulerBatchedTopKDecodeRoute(&.{
        .decode_batch_size = 1,
        .route_top_k = 64,
        .sampling_config = &valid_sampling,
    }));
    try std.testing.expect(!metal_backend.shouldUseSchedulerBatchedTopKDecodeRoute(&.{
        .decode_batch_size = 1,
        .route_top_k = 0,
        .sampling_config = &invalid_sampling,
    }));
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
    var cpu_backend = ExecutionTarget{ .cpu = undefined };
    try std.testing.expectEqual(cpu.vision.maxPixels(), cpu_backend.visionMaxPixels());

    if (has_metal) {
        var metal_backend = ExecutionTarget{ .metal = undefined };
        try std.testing.expectEqual(metal.vision.maxPixels(), metal_backend.visionMaxPixels());
    }
    if (has_cuda) {
        var cuda_backend = ExecutionTarget{ .cuda = undefined };
        try std.testing.expectEqual(cuda.vision.maxPixels(), cuda_backend.visionMaxPixels());
    }
}

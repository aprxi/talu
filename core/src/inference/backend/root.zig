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
pub const pipeline = @import("pipeline.zig");
pub const staged_orchestrator = @import("staged_orchestrator.zig");

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

/// Default max concurrent decode slots for all backends.
/// Override at runtime via TALU_MAX_BATCH_SIZE.
const default_max_batch_size: usize = 8;

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

fn effectiveLoadSelection(requested: Selection) Selection {
    if (requested != .auto) return requested;
    if (std.posix.getenv("BACKEND")) |raw_ptr| {
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

/// Execution topology for CUDA-capable backends.
///
/// - `single`: All layers on one GPU (default).
/// - `pipeline2`: Two CUDA GPUs, layers split at `split_layer`.
/// - `cpu_gpu`: CPU executes layers [0, split_layer), GPU executes [split_layer, N).
/// - `cpu_gpu_gpu`: CPU [0, split_layer), GPU0 [split_layer, split_layer_stage2), GPU1 [split_layer_stage2, N).
pub const CudaTopologyMode = enum {
    single,
    pipeline2,
    cpu_gpu,
    cpu_gpu_gpu,
};

/// Configuration for multi-stage execution topologies.
///
/// `mode` selects the topology. `split_layer` sets the first stage boundary
/// (layer index where stage 0 ends and stage 1 begins). For 3-stage modes,
/// `split_layer_stage2` sets the second boundary. `stage_device_ordinals`
/// identifies which CUDA devices to use for GPU stages.
///
/// When `split_layer` is null, the runtime uses a default heuristic (n/2 for
/// 2-stage, n/3 for 3-stage). Explicit values are validated against the model's
/// layer count at init.
pub const CudaTopologyConfig = struct {
    mode: CudaTopologyMode = .single,
    stage_device_ordinals: [2]usize = .{ 0, 1 },
    split_layer: ?usize = null,
    split_layer_stage2: ?usize = null,
};

/// Backend initialization options selected at startup/config layer.
pub const InitOptions = struct {
    pub const MetalConfig = struct {
        /// Resolved model directory path (snapshot dir containing config/weights files).
        model_path: ?[]const u8 = null,
        /// User model reference (e.g. "Qwen/Qwen3.5-0.8B") for metadata/logging.
        model_id: ?[]const u8 = null,
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

fn selectionOverrideFromEnv(allocator: std.mem.Allocator) ?Selection {
    const raw = std.process.getEnvVarOwned(allocator, "BACKEND") catch return null;
    defer allocator.free(raw);
    const parsed = parseSelectionToken(raw);
    if (parsed == null) {
        log.warn("inference", "Ignoring invalid BACKEND override", .{
            .value = std.mem.trim(u8, raw, " \t\r\n"),
            .supported = "auto|cpu|metal|cuda",
        });
    }
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

const CudaTopology = struct {
    mode: CudaTopologyMode = .single,
    stage_device_ordinals: [2]usize = .{ 0, 1 },
    split_layer: ?usize = null,
    split_layer_stage2: ?usize = null,

    fn primaryDeviceOrdinal(self: *const CudaTopology) usize {
        return switch (self.mode) {
            .cpu_gpu_gpu => self.stage_device_ordinals[1],
            else => self.stage_device_ordinals[0],
        };
    }
};

const CudaTopologyCapabilities = struct {
    supports_pipeline2: bool = true,
    supports_cpu_gpu: bool = true,
    supports_cpu_gpu_gpu: bool = true,
    min_pipeline2_layers: usize = 2,
    min_cpu_gpu_layers: usize = 2,
    min_cpu_gpu_gpu_layers: usize = 3,
    requires_distinct_stage_devices: bool = true,
};

const cuda_topology_capabilities = CudaTopologyCapabilities{};

const CudaTopologyValidationError = error{
    Pipeline2Unsupported,
    Pipeline2InsufficientLayers,
    Pipeline2InvalidSplitLayer,
    Pipeline2RequiresDistinctDevices,
    Pipeline2DeviceOrdinalOutOfRange,
    CpuGpuUnsupported,
    CpuGpuInsufficientLayers,
    CpuGpuInvalidSplitLayer,
    CpuGpuDeviceOrdinalOutOfRange,
    CpuGpuGpuUnsupported,
    CpuGpuGpuInsufficientLayers,
    CpuGpuGpuInvalidSplitLayer,
    CpuGpuGpuInvalidSplitLayerStage2,
    CpuGpuGpuSplitOrderInvalid,
    CpuGpuGpuRequiresDistinctDevices,
    CpuGpuGpuDeviceOrdinalOutOfRange,
};

fn validateCudaTopologyConfig(
    topology: CudaTopology,
    total_layers: usize,
    device_count: usize,
) CudaTopologyValidationError!void {
    switch (topology.mode) {
        .single => return,
        .pipeline2 => {
            if (!cuda_topology_capabilities.supports_pipeline2) return error.Pipeline2Unsupported;
            if (total_layers < cuda_topology_capabilities.min_pipeline2_layers) return error.Pipeline2InsufficientLayers;
            if (topology.split_layer) |split| {
                if (split == 0 or split >= total_layers) return error.Pipeline2InvalidSplitLayer;
            }
            if (cuda_topology_capabilities.requires_distinct_stage_devices and
                topology.stage_device_ordinals[0] == topology.stage_device_ordinals[1])
            {
                return error.Pipeline2RequiresDistinctDevices;
            }
            if (topology.stage_device_ordinals[0] >= device_count or
                topology.stage_device_ordinals[1] >= device_count)
            {
                return error.Pipeline2DeviceOrdinalOutOfRange;
            }
        },
        .cpu_gpu => {
            if (!cuda_topology_capabilities.supports_cpu_gpu) return error.CpuGpuUnsupported;
            if (total_layers < cuda_topology_capabilities.min_cpu_gpu_layers) return error.CpuGpuInsufficientLayers;
            if (topology.split_layer) |split| {
                if (split == 0 or split >= total_layers) return error.CpuGpuInvalidSplitLayer;
            }
            if (topology.stage_device_ordinals[0] >= device_count) return error.CpuGpuDeviceOrdinalOutOfRange;
        },
        .cpu_gpu_gpu => {
            if (!cuda_topology_capabilities.supports_cpu_gpu_gpu) return error.CpuGpuGpuUnsupported;
            if (total_layers < cuda_topology_capabilities.min_cpu_gpu_gpu_layers) return error.CpuGpuGpuInsufficientLayers;
            const split_default = @max(@as(usize, 1), total_layers / 3);
            const split = topology.split_layer orelse split_default;
            if (split == 0 or split >= total_layers) return error.CpuGpuGpuInvalidSplitLayer;
            const split2_default = split + @max(@as(usize, 1), (total_layers - split) / 2);
            const split2 = topology.split_layer_stage2 orelse split2_default;
            if (split2 == 0 or split2 >= total_layers) return error.CpuGpuGpuInvalidSplitLayerStage2;
            if (split2 <= split) return error.CpuGpuGpuSplitOrderInvalid;
            if (cuda_topology_capabilities.requires_distinct_stage_devices and
                topology.stage_device_ordinals[0] == topology.stage_device_ordinals[1])
            {
                return error.CpuGpuGpuRequiresDistinctDevices;
            }
            if (topology.stage_device_ordinals[0] >= device_count or
                topology.stage_device_ordinals[1] >= device_count)
            {
                return error.CpuGpuGpuDeviceOrdinalOutOfRange;
            }
        },
    }
}

fn cudaProbeName(probe: CudaProbe) []const u8 {
    return @tagName(probe);
}

fn probeCudaRuntime() CudaProbe {
    if (!has_cuda) return .disabled;
    return compute.cuda.probeRuntime();
}

fn parseCudaTopologyMode(raw: []const u8) ?CudaTopologyMode {
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (token.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(token, "single")) return .single;
    if (std.ascii.eqlIgnoreCase(token, "pipeline2")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_2")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_2way")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "cpu_gpu")) return .cpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "cpu+gpu")) return .cpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_cpu_gpu")) return .cpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "cpu_gpu_gpu")) return .cpu_gpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "cpu+gpu+gpu")) return .cpu_gpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_cpu_gpu_gpu")) return .cpu_gpu_gpu;
    return null;
}

fn parseTwoDeviceOrdinals(raw: []const u8) ?[2]usize {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return null;
    var iter = std.mem.splitScalar(u8, trimmed, ',');
    const left_raw = iter.next() orelse return null;
    const right_raw = iter.next() orelse return null;
    if (iter.next() != null) return null;
    const left = std.fmt.parseUnsigned(usize, std.mem.trim(u8, left_raw, " \t\r\n"), 10) catch return null;
    const right = std.fmt.parseUnsigned(usize, std.mem.trim(u8, right_raw, " \t\r\n"), 10) catch return null;
    return .{ left, right };
}

fn resolveCudaTopology(
    allocator: std.mem.Allocator,
    topology_override: ?CudaTopologyConfig,
) CudaTopology {
    var topology = CudaTopology{};
    if (topology_override) |cfg| {
        topology.mode = cfg.mode;
        topology.stage_device_ordinals = cfg.stage_device_ordinals;
        topology.split_layer = cfg.split_layer;
        topology.split_layer_stage2 = cfg.split_layer_stage2;
        return topology;
    }

    const mode_raw = std.process.getEnvVarOwned(allocator, "TALU_CUDA_TOPOLOGY") catch null;
    if (mode_raw) |value| {
        defer allocator.free(value);
        if (parseCudaTopologyMode(value)) |mode| {
            topology.mode = mode;
        } else {
            log.warn("inference", "Invalid TALU_CUDA_TOPOLOGY; using single", .{ .value = value });
        }
    }

    const devices_raw = std.process.getEnvVarOwned(allocator, "TALU_CUDA_STAGE_DEVICES") catch null;
    if (devices_raw) |value| {
        defer allocator.free(value);
        if (parseTwoDeviceOrdinals(value)) |ordinals| {
            topology.stage_device_ordinals = ordinals;
        } else {
            log.warn("inference", "Invalid TALU_CUDA_STAGE_DEVICES; expected '<gpu0>,<gpu1>'", .{ .value = value });
        }
    }

    const single_device_raw = std.process.getEnvVarOwned(allocator, "TALU_CUDA_DEVICE") catch null;
    if (single_device_raw) |value| {
        defer allocator.free(value);
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
            log.warn("inference", "Invalid TALU_CUDA_DEVICE; keeping topology primary device", .{
                .value = trimmed,
                .current = topology.primaryDeviceOrdinal(),
            });
            return topology;
        };
        if (topology.mode == .cpu_gpu_gpu) {
            topology.stage_device_ordinals[1] = parsed;
        } else {
            topology.stage_device_ordinals[0] = parsed;
        }
    }

    const split_layer_raw = std.process.getEnvVarOwned(allocator, "TALU_CUDA_SPLIT_LAYER") catch null;
    if (split_layer_raw) |value| {
        defer allocator.free(value);
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
            log.warn("inference", "Invalid TALU_CUDA_SPLIT_LAYER; using default split", .{
                .value = trimmed,
            });
            return topology;
        };
        topology.split_layer = parsed;
    }

    const split_layer_stage2_raw = std.process.getEnvVarOwned(allocator, "TALU_CUDA_SPLIT_LAYER_STAGE2") catch null;
    if (split_layer_stage2_raw) |value| {
        defer allocator.free(value);
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
            log.warn("inference", "Invalid TALU_CUDA_SPLIT_LAYER_STAGE2; using default split", .{
                .value = trimmed,
            });
            return topology;
        };
        topology.split_layer_stage2 = parsed;
    }

    return topology;
}

// ---------------------------------------------------------------------------
// Auto topology selection
//
// When no explicit TALU_CUDA_TOPOLOGY is set, automatically choose the best
// topology based on available GPU memory and estimated model requirements.
//
// Scenarios (see also AGENTS.md):
//
//   S0  BACKEND=cpu           → CPU backend, no GPU. Handled before we get here.
//   S1  TALU_CUDA_TOPOLOGY    → User override. Skip auto-detection.
//   S2  0 GPUs visible        → Error (caller handles).
//   S3  1 GPU, model fits     → single.
//   S4  1 GPU, model !fits    → cpu_gpu. Maximize layers on GPU.
//   S5  2+ GPUs, fits on 1    → single on best GPU. Avoids pipeline overhead.
//   S6  2+ GPUs, fits on 2    → pipeline2. Split proportional to free memory.
//   S7  2+ GPUs, !fits on 2   → cpu_gpu_gpu. CPU takes overflow layers.
//   S8  Too large for all     → Error (caller handles).
//   S9  3+ GPUs               → Pick best 2 GPUs, apply S5-S7.
// ---------------------------------------------------------------------------

const GpuMemoryInfo = struct {
    ordinal: usize,
    free: usize,
    total: usize,
};

/// Fixed per-GPU overhead for CUDA context, allocator fragmentation, and
/// miscellaneous buffers not individually estimated (topk, decode pointer tables, etc.).
/// All other costs (weights, KV cache, activations, embed/proj) are computed exactly.
const AUTO_TOPO_OVERHEAD_BYTES: usize = 256 * 1024 * 1024; // 256 MiB per GPU
/// Practical seq_len cap for KV cache estimation. The engine allocates KV cache
/// dynamically (starting at 256 tokens, doubling up to max_seq_len). Using the
/// model's theoretical max (e.g. 262K for Qwen) would wildly overestimate.
/// 8192 is a practical generation context that covers most real-world usage.
const AUTO_TOPO_KV_SEQ_LEN_CAP: usize = 8192;

/// Probe total memory on each visible CUDA device without creating contexts.
/// Uses cuDeviceTotalMem which requires no CUDA context, avoiding interference
/// with later Device.initAt calls during backend initialization.
/// Caller owns the returned slice.
fn probeGpuTotalMemory(allocator: std.mem.Allocator, device_count: usize) ![]GpuMemoryInfo {
    if (device_count == 0) return allocator.alloc(GpuMemoryInfo, 0);
    const mems = if (has_cuda)
        try compute.cuda.Device.deviceTotalMemories(allocator)
    else
        return allocator.alloc(GpuMemoryInfo, 0);
    defer allocator.free(mems);

    const n = @min(mems.len, device_count);
    const infos = try allocator.alloc(GpuMemoryInfo, n);
    for (0..n) |ord| {
        infos[ord] = .{ .ordinal = ord, .free = mems[ord], .total = mems[ord] };
    }
    return infos;
}

/// Model size parameters for per-GPU memory estimation.
const ModelSizeParams = struct {
    file_size: usize,
    vocab_size: usize,
    d_model: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    n_layers: usize,
};

/// Compute GPU memory for a specific stage with a given number of layers.
///
/// Separates fixed costs (embedding, projection, activation buffers) from
/// per-layer costs (weights, KV cache). All components are derived from
/// known model parameters — no estimation fudge factors.
/// Intermediate stages skip embedding/projection uploads at init time.
fn estimatePerGpuBytes(
    p: ModelSizeParams,
    gpu_layers: usize,
    needs_embedding: bool,
    needs_projection: bool,
) usize {
    // Embedding and projection: vocab_size * d_model * 2 bytes (bf16/f16).
    const elem_bytes: usize = 2;
    const embed_bytes = p.vocab_size *| p.d_model *| elem_bytes;
    const proj_bytes = embed_bytes; // projection is typically same size as embedding

    // Per-layer weight: (file_size - embed - proj) / n_layers.
    const non_layer_bytes = embed_bytes +| proj_bytes;
    const layer_weight_bytes = if (p.file_size > non_layer_bytes and p.n_layers > 0)
        (p.file_size - non_layer_bytes) / p.n_layers
    else if (p.n_layers > 0)
        p.file_size / p.n_layers
    else
        p.file_size;

    // KV cache per layer: n_kv_heads * head_dim * max_seq_len * 2(K+V) * 2(f16).
    const kv_per_layer = p.n_kv_heads *| p.head_dim *| p.max_seq_len *| 4;

    // Activation buffers: ~30 d_model-sized f32 buffers + dequant scratch.
    const activation_bytes = p.d_model *| 30 *| 4;

    // Sum components.
    var total: usize = 0;
    total +|= gpu_layers *| (layer_weight_bytes +| kv_per_layer);
    total +|= activation_bytes;
    if (needs_embedding) total +|= embed_bytes;
    if (needs_projection) total +|= proj_bytes;

    return total;
}

/// Compute total GPU memory for full model (used for logging).
fn estimateModelGpuBytes(p: ModelSizeParams) usize {
    return estimatePerGpuBytes(p, p.n_layers, true, true);
}

/// Resolve max_seq_len for memory estimation.
/// Uses TALU_CUDA_MAX_SEQ_LEN if set, otherwise caps at AUTO_TOPO_KV_SEQ_LEN_CAP.
/// The engine allocates KV cache dynamically, so using the model's theoretical max
/// (which can be 128K-1M) would wildly overestimate actual memory needs.
fn resolveMaxSeqLenForEstimation(allocator: std.mem.Allocator, model_max: usize) usize {
    const raw = std.process.getEnvVarOwned(allocator, "TALU_CUDA_MAX_SEQ_LEN") catch {
        return @min(model_max, AUTO_TOPO_KV_SEQ_LEN_CAP);
    };
    defer allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch
        return @min(model_max, AUTO_TOPO_KV_SEQ_LEN_CAP);
    if (parsed == 0) return @min(model_max, AUTO_TOPO_KV_SEQ_LEN_CAP);
    return @min(model_max, parsed);
}

/// Find max gpu_layers that fit in budget for a single GPU stage.
fn maxLayersForBudget(
    p: ModelSizeParams,
    budget: usize,
    total_layers: usize,
    needs_embedding: bool,
    needs_projection: bool,
) usize {
    // Binary search for max layers that fit.
    var lo: usize = 1;
    var hi: usize = total_layers;
    var best: usize = 0;
    while (lo <= hi) {
        const mid = lo + (hi - lo) / 2;
        const est = estimatePerGpuBytes(p, mid, needs_embedding, needs_projection);
        if (est <= budget) {
            best = mid;
            lo = mid + 1;
        } else {
            if (mid == 0) break;
            hi = mid - 1;
        }
    }
    return best;
}

/// Select topology automatically based on estimated model size and GPU memory.
///
/// Uses per-GPU estimation that separates fixed costs (embedding, projection,
/// activation buffers) from per-layer costs (weights, KV cache).
/// Intermediate pipeline stages skip embedding/projection at init time.
///
/// Pure logic — no I/O, no side effects. Fully testable with synthetic values.
/// Returns .single when in doubt (safest default).
///
/// gpu_infos must be sorted by ordinal (as returned by probeGpuTotalMemory).
fn autoSelectTopology(
    p: ModelSizeParams,
    total_layers: usize,
    gpu_infos: []const GpuMemoryInfo,
) CudaTopology {
    if (gpu_infos.len == 0 or total_layers < 2) return .{};

    // Find the two GPUs with the most free memory.
    var best0: usize = 0; // index into gpu_infos
    var best1: usize = if (gpu_infos.len > 1) @as(usize, 1) else 0;
    for (gpu_infos, 0..) |info, i| {
        if (info.free > gpu_infos[best0].free) {
            best1 = best0;
            best0 = i;
        } else if (i != best0 and info.free > gpu_infos[best1].free) {
            best1 = i;
        }
    }

    const best0_budget = if (gpu_infos[best0].free > AUTO_TOPO_OVERHEAD_BYTES)
        gpu_infos[best0].free - AUTO_TOPO_OVERHEAD_BYTES
    else
        0;

    // S3/S5: Model fits on one GPU (with embedding + projection).
    const single_est = estimatePerGpuBytes(p, total_layers, true, true);
    if (single_est <= best0_budget) {
        return .{
            .mode = .single,
            .stage_device_ordinals = .{ gpu_infos[best0].ordinal, gpu_infos[best0].ordinal },
        };
    }

    // From here: model doesn't fit on single GPU.
    if (gpu_infos.len == 1) {
        // S4: cpu_gpu — offload lower layers to CPU.
        // GPU stage: no embedding (CPU does it), has projection (last stage).
        const gpu_layers = std.math.clamp(
            maxLayersForBudget(p, best0_budget, total_layers, false, true),
            1,
            total_layers - 1,
        );
        return .{
            .mode = .cpu_gpu,
            .stage_device_ordinals = .{ gpu_infos[0].ordinal, gpu_infos[0].ordinal },
            .split_layer = total_layers - gpu_layers,
        };
    }

    // 2+ GPUs. Ensure ordinals are ordered (lower first for pipeline2).
    const ord0 = @min(gpu_infos[best0].ordinal, gpu_infos[best1].ordinal);
    const ord1 = @max(gpu_infos[best0].ordinal, gpu_infos[best1].ordinal);
    const free0 = if (gpu_infos[best0].ordinal == ord0) gpu_infos[best0].free else gpu_infos[best1].free;
    const free1 = if (gpu_infos[best0].ordinal == ord0) gpu_infos[best1].free else gpu_infos[best0].free;
    const budget0 = if (free0 > AUTO_TOPO_OVERHEAD_BYTES) free0 - AUTO_TOPO_OVERHEAD_BYTES else 0;
    const budget1 = if (free1 > AUTO_TOPO_OVERHEAD_BYTES) free1 - AUTO_TOPO_OVERHEAD_BYTES else 0;

    // S6: Model fits across 2 GPUs (pipeline2).
    // GPU0: has embedding, no projection. GPU1: no embedding, has projection.
    // Find best split point where both GPUs fit.
    {
        const max_gpu0 = maxLayersForBudget(p, budget0, total_layers - 1, true, false);
        if (max_gpu0 >= 1) {
            const gpu1_layers = total_layers - max_gpu0;
            const gpu1_est = estimatePerGpuBytes(p, gpu1_layers, false, true);
            if (gpu1_est <= budget1) {
                // Try to balance: find split where GPU1 also fits, starting from
                // proportional split and adjusting.
                const total_free = free0 + free1;
                var split = if (total_free > 0)
                    std.math.clamp(total_layers * free0 / total_free, 1, total_layers - 1)
                else
                    total_layers / 2;
                // Adjust: decrease split if GPU0 overflows, increase if GPU1 overflows.
                // Track direction to detect oscillation (GPU0 fits at S but GPU1
                // doesn't, GPU0 doesn't fit at S+1 → no valid split exists).
                var last_dir: enum { none, up, down } = .none;
                while (split < total_layers - 1) {
                    const g0 = estimatePerGpuBytes(p, split, true, false);
                    const g1 = estimatePerGpuBytes(p, total_layers - split, false, true);
                    if (g0 <= budget0 and g1 <= budget1) break;
                    if (g0 > budget0) {
                        if (last_dir == .up) break; // oscillation
                        last_dir = .down;
                        split -= 1;
                        if (split == 0) break;
                    } else {
                        if (last_dir == .down) break; // oscillation
                        last_dir = .up;
                        split += 1;
                    }
                }
                // Validate final split.
                const g0_final = estimatePerGpuBytes(p, split, true, false);
                const g1_final = estimatePerGpuBytes(p, total_layers - split, false, true);
                if (g0_final <= budget0 and g1_final <= budget1 and split >= 1 and split < total_layers) {
                    return .{
                        .mode = .pipeline2,
                        .stage_device_ordinals = .{ ord0, ord1 },
                        .split_layer = split,
                    };
                }
            }
        }
    }

    // S7: Doesn't fit on 2 GPUs — cpu_gpu_gpu. CPU takes overflow.
    // GPU0: no embedding, no projection (middle stage).
    // GPU1: no embedding, has projection (last stage).
    const max_gpu1 = maxLayersForBudget(p, budget1, total_layers - 1, false, true);
    const max_gpu0_mid = maxLayersForBudget(p, budget0, total_layers - 1, false, false);
    const total_gpu_layers = @min(max_gpu0_mid + max_gpu1, total_layers - 1);
    if (total_gpu_layers < 2) {
        // S8: Not enough GPU memory even for cpu_gpu_gpu. Return single and let
        // the caller handle the OOM or error.
        return .{};
    }
    const gpu_layers = std.math.clamp(total_gpu_layers, 2, total_layers - 1);
    const cpu_layers = total_layers - gpu_layers;
    // Balance GPU layers to minimize max utilization across both GPUs.
    var best_split: usize = gpu_layers / 2;
    var best_max_util: usize = std.math.maxInt(usize);
    {
        var s: usize = 1;
        while (s < gpu_layers) : (s += 1) {
            const g0 = estimatePerGpuBytes(p, s, false, false);
            const g1 = estimatePerGpuBytes(p, gpu_layers - s, false, true);
            const util0 = g0 * 1000 / budget0;
            const util1 = g1 * 1000 / budget1;
            const max_util = @max(util0, util1);
            if (max_util < best_max_util) {
                best_max_util = max_util;
                best_split = s;
            }
        }
    }
    const gpu0_share = best_split;
    return .{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ ord0, ord1 },
        .split_layer = cpu_layers,
        .split_layer_stage2 = cpu_layers + gpu0_share,
    };
}

/// Run full auto-detection: probe GPU memory, estimate model requirements, select topology.
/// Falls back to the provided `fallback` topology on probe failure.
fn autoDetectTopologyForModel(
    allocator: std.mem.Allocator,
    loaded: *const LoadedModel,
    fallback: CudaTopology,
) !CudaTopology {
    const device_count = if (has_cuda)
        try compute.cuda.Device.deviceCount()
    else
        return fallback;

    if (device_count == 0) return fallback;

    const gpu_infos = try probeGpuTotalMemory(allocator, device_count);
    defer allocator.free(gpu_infos);

    const model_max_seq: usize = @intCast(@max(@as(i32, 1), loaded.config.max_seq_len));
    const max_seq_len = resolveMaxSeqLenForEstimation(allocator, model_max_seq);
    const n_kv_heads: usize = @intCast(@max(@as(i32, 0), loaded.config.n_kv_groups));
    const head_dim: usize = @intCast(@max(@as(i32, 0), loaded.config.head_dim));
    const n_layers: usize = @intCast(@max(@as(i32, 0), loaded.config.n_layers));
    const vocab_size: usize = @intCast(@max(@as(i32, 1), loaded.config.vocab_size));
    const d_model: usize = @intCast(@max(@as(i32, 1), loaded.config.d_model));

    const params = ModelSizeParams{
        .file_size = loaded.file_size,
        .vocab_size = vocab_size,
        .d_model = d_model,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .max_seq_len = max_seq_len,
        .n_layers = n_layers,
    };

    const estimated = estimateModelGpuBytes(params);

    const result = autoSelectTopology(params, loaded.blocks.len, gpu_infos);

    // Log the auto-detection result.
    const estimated_mib = estimated / (1024 * 1024);
    switch (result.mode) {
        .single => {
            const gpu_free_mib = if (gpu_infos.len > result.stage_device_ordinals[0])
                gpu_infos[result.stage_device_ordinals[0]].free / (1024 * 1024)
            else
                0;
            log.info("inference", "Auto topology: single", .{
                .estimated_mib = estimated_mib,
                .gpu = result.stage_device_ordinals[0],
                .gpu_free_mib = gpu_free_mib,
            });
        },
        .cpu_gpu => log.info("inference", "Auto topology: cpu_gpu", .{
            .estimated_mib = estimated_mib,
            .split_layer = result.split_layer orelse 0,
            .gpu = result.stage_device_ordinals[0],
            .gpu_free_mib = if (gpu_infos.len > 0) gpu_infos[result.stage_device_ordinals[0]].free / (1024 * 1024) else 0,
        }),
        .pipeline2 => log.info("inference", "Auto topology: pipeline2", .{
            .estimated_mib = estimated_mib,
            .split_layer = result.split_layer orelse 0,
            .gpu0 = result.stage_device_ordinals[0],
            .gpu1 = result.stage_device_ordinals[1],
            .gpu0_free_mib = if (result.stage_device_ordinals[0] < gpu_infos.len) gpu_infos[result.stage_device_ordinals[0]].free / (1024 * 1024) else 0,
            .gpu1_free_mib = if (result.stage_device_ordinals[1] < gpu_infos.len) gpu_infos[result.stage_device_ordinals[1]].free / (1024 * 1024) else 0,
        }),
        .cpu_gpu_gpu => log.info("inference", "Auto topology: cpu_gpu_gpu", .{
            .estimated_mib = estimated_mib,
            .split_layer = result.split_layer orelse 0,
            .split_layer_stage2 = result.split_layer_stage2 orelse 0,
            .gpu0 = result.stage_device_ordinals[0],
            .gpu1 = result.stage_device_ordinals[1],
        }),
    }

    return result;
}

/// Find the minimum KV shared source layer index across all shared layers.
///
/// For models with KV sharing (e.g., Gemma4), later layers reuse the KV cache
/// from earlier "source" layers. Returns the lowest such source layer index,
/// or null if the model has no KV sharing.
///
/// Used to detect when cpu_gpu_gpu topology is infeasible (source layers in
/// CPU range means no valid GPU1/GPU2 split exists).
fn minKvSharedSourceLayer(config: ModelConfig) ?usize {
    if (config.num_kv_shared_layers <= 0) return null;
    const n_layers: usize = @intCast(config.n_layers);
    const layer_types = config.layer_types orelse return null;
    if (layer_types.len != n_layers) return null;
    const shared_count: usize = @min(@as(usize, @intCast(config.num_kv_shared_layers)), n_layers);
    if (shared_count == 0 or shared_count >= n_layers) return null;
    const first_shared = n_layers - shared_count;
    if (first_shared == 0) return null;

    // Mirrors resolveSharedKvSourceLayer: search backward for matching layer_type.
    var min_src: usize = first_shared;
    var idx = first_shared;
    while (idx < n_layers) : (idx += 1) {
        const target_type = layer_types[idx];
        var src = first_shared;
        while (src > 0) {
            src -= 1;
            if (layer_types[src] == target_type) {
                min_src = @min(min_src, src);
                break;
            }
        }
    }
    return if (min_src < first_shared) min_src else null;
}

/// Pure-logic topology selection from a requested CPU layer count.
///
/// CPU runs layers [0, cpu_layers), GPU(s) run [cpu_layers, total_layers).
/// With 2+ GPUs the GPU portion is split using model-aware estimation that
/// accounts for projection weight on the last stage.
///
/// Returns null when cpu_layers is 0 (all on GPU → auto-detect) or invalid.
fn topologyFromCpuLayers(
    cpu_layers: usize,
    total_layers: usize,
    gpu_infos: []const GpuMemoryInfo,
    model_params: ?ModelSizeParams,
) ?CudaTopology {
    if (total_layers < 2 or cpu_layers == 0 or cpu_layers >= total_layers) return null;
    if (gpu_infos.len == 0) return null;

    const gpu_layers = total_layers - cpu_layers;

    if (gpu_infos.len == 1 or gpu_layers < 2) {
        // 1 GPU or only 1 GPU layer → cpu_gpu.
        return .{
            .mode = .cpu_gpu,
            .stage_device_ordinals = .{ gpu_infos[0].ordinal, gpu_infos[0].ordinal },
            .split_layer = cpu_layers,
        };
    }

    // 2+ GPUs: cpu_gpu_gpu. Pick best 2 GPUs and split.
    var best0: usize = 0;
    var best1: usize = 1;
    for (gpu_infos, 0..) |info, i| {
        if (info.free > gpu_infos[best0].free) {
            best1 = best0;
            best0 = i;
        } else if (i != best0 and info.free > gpu_infos[best1].free) {
            best1 = i;
        }
    }
    const ord0 = @min(gpu_infos[best0].ordinal, gpu_infos[best1].ordinal);
    const ord1 = @max(gpu_infos[best0].ordinal, gpu_infos[best1].ordinal);
    const free0 = if (gpu_infos[best0].ordinal == ord0) gpu_infos[best0].free else gpu_infos[best1].free;
    const free1 = if (gpu_infos[best0].ordinal == ord0) gpu_infos[best1].free else gpu_infos[best0].free;

    // Model-aware split: GPU0 (middle) has no embed/proj, GPU1 (last) has proj.
    // Find the split that balances utilization across both GPUs.
    const gpu0_share = if (model_params) |p| blk: {
        const budget0 = if (free0 > AUTO_TOPO_OVERHEAD_BYTES) free0 - AUTO_TOPO_OVERHEAD_BYTES else 1;
        const budget1 = if (free1 > AUTO_TOPO_OVERHEAD_BYTES) free1 - AUTO_TOPO_OVERHEAD_BYTES else 1;
        // Iterate all possible splits and pick the one that minimizes the
        // maximum utilization ratio. This accounts for projection asymmetry.
        var best_split: usize = gpu_layers / 2;
        var best_max_util: usize = std.math.maxInt(usize);
        var s: usize = 1;
        while (s < gpu_layers) : (s += 1) {
            const g0 = estimatePerGpuBytes(p, s, false, false);
            const g1 = estimatePerGpuBytes(p, gpu_layers - s, false, true);
            const util0 = g0 * 1000 / budget0; // per-mille for precision
            const util1 = g1 * 1000 / budget1;
            const max_util = @max(util0, util1);
            if (max_util < best_max_util) {
                best_max_util = max_util;
                best_split = s;
            }
        }
        break :blk best_split;
    } else blk: {
        // Fallback: proportional by memory (no model info available).
        const total_free = free0 + free1;
        break :blk if (total_free > 0)
            std.math.clamp(gpu_layers * free0 / total_free, 1, gpu_layers - 1)
        else
            gpu_layers / 2;
    };

    return .{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ ord0, ord1 },
        .split_layer = cpu_layers,
        .split_layer_stage2 = cpu_layers + gpu0_share,
    };
}

/// Resolve topology from TALU_CPU_LAYERS env var.
///
/// Returns null when the env var is not set or 0 (all on GPU → auto-detect).
///
/// When set to N (0 < N < total_layers):
///   CPU runs layers [0, N), GPU(s) run [N, total_layers).
///   With 2+ GPUs the GPU portion is auto-split proportional to memory.
fn resolveCpuLayersTopology(
    allocator: std.mem.Allocator,
    loaded: *const LoadedModel,
    current: CudaTopology,
) ?CudaTopology {
    const raw = std.process.getEnvVarOwned(allocator, "TALU_CPU_LAYERS") catch return null;
    defer allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const cpu_layers_i = std.fmt.parseInt(i32, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_CPU_LAYERS; ignoring", .{ .value = trimmed });
        return null;
    };

    const total_layers = loaded.blocks.len;
    if (total_layers < 2) return null;
    if (cpu_layers_i <= 0) return null;

    // CUDA backend requires at least 1 GPU layer. Clamp if needed.
    const cpu_layers: usize = if (cpu_layers_i >= @as(i32, @intCast(total_layers))) blk: {
        const clamped = total_layers - 1;
        log.warn("inference", "TALU_CPU_LAYERS clamped: CUDA backend requires at least 1 GPU layer", .{
            .requested = cpu_layers_i,
            .clamped = clamped,
            .total_layers = total_layers,
        });
        break :blk clamped;
    } else @intCast(cpu_layers_i);

    // Query GPU count and memory.
    const device_count = if (has_cuda)
        compute.cuda.Device.deviceCount() catch |err| {
            log.warn("inference", "TALU_CPU_LAYERS: failed to query device count", .{
                .err = @errorName(err),
            });
            return null;
        }
    else
        return null;

    if (device_count == 0) return null;

    // Build model params for model-aware GPU layer splitting.
    const model_max_seq: usize = @intCast(@max(@as(i32, 1), loaded.config.max_seq_len));
    const p = ModelSizeParams{
        .file_size = loaded.file_size,
        .vocab_size = @intCast(@max(@as(i32, 1), loaded.config.vocab_size)),
        .d_model = @intCast(@max(@as(i32, 1), loaded.config.d_model)),
        .n_kv_heads = @intCast(@max(@as(i32, 0), loaded.config.n_kv_groups)),
        .head_dim = @intCast(@max(@as(i32, 0), loaded.config.head_dim)),
        .max_seq_len = resolveMaxSeqLenForEstimation(allocator, model_max_seq),
        .n_layers = @intCast(@max(@as(i32, 0), loaded.config.n_layers)),
    };

    const gpu_infos = probeGpuTotalMemory(allocator, device_count) catch |err| {
        log.warn("inference", "TALU_CPU_LAYERS: failed to probe GPU memory", .{
            .err = @errorName(err),
        });
        // Fallback: assume 1 GPU with unknown memory.
        return topologyFromCpuLayers(cpu_layers, total_layers, &.{
            .{ .ordinal = current.stage_device_ordinals[0], .free = 0, .total = 0 },
        }, null);
    };
    defer allocator.free(gpu_infos);

    var result = topologyFromCpuLayers(cpu_layers, total_layers, gpu_infos, p) orelse return null;

    // cpu_gpu_gpu splits GPU layers across 2 devices which requires KV shared
    // source layers to be strictly above cpu_layers (room for GPU1 stage).
    // When sources fall in CPU range, fall back to cpu_gpu (single GPU) where
    // cross-device KV replication handles it.
    if (result.mode == .cpu_gpu_gpu) {
        if (minKvSharedSourceLayer(loaded.config)) |min_src| {
            if (cpu_layers >= min_src) {
                result = .{
                    .mode = .cpu_gpu,
                    .stage_device_ordinals = .{ result.stage_device_ordinals[0], result.stage_device_ordinals[0] },
                    .split_layer = cpu_layers,
                };
            }
        }
    }

    // Log the resolved topology.
    const gpu_layers = total_layers - cpu_layers;
    switch (result.mode) {
        .cpu_gpu => log.info("inference", "TALU_CPU_LAYERS: cpu_gpu", .{
            .cpu_layers = cpu_layers,
            .gpu_layers = gpu_layers,
            .total_layers = total_layers,
        }),
        .cpu_gpu_gpu => log.info("inference", "TALU_CPU_LAYERS: cpu_gpu_gpu", .{
            .cpu_layers = cpu_layers,
            .gpu_layers = gpu_layers,
            .split_layer_stage2 = result.split_layer_stage2 orelse 0,
            .gpu0 = result.stage_device_ordinals[0],
            .gpu1 = result.stage_device_ordinals[1],
            .total_layers = total_layers,
        }),
        else => {},
    }

    return result;
}

fn resolveMaxBatchSize() usize {
    const raw = std.process.getEnvVarOwned(std.heap.c_allocator, "TALU_MAX_BATCH_SIZE") catch {
        return default_max_batch_size;
    };
    defer std.heap.c_allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
        log.warn("inference", "Invalid TALU_MAX_BATCH_SIZE; using default", .{
            .value = trimmed,
            .default = default_max_batch_size,
        });
        return default_max_batch_size;
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

/// Backend type - tagged union of available backends
pub const Backend = union(enum) {
    /// Fused CPU backend for graph ops (production inference)
    cpu: cpu.BackendType,
    /// Metal GPU backend (macOS only).
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
            .metal => return initMetal(allocator, loaded, "configured", init_options.metal),
            .cuda => return initCuda(allocator, loaded, "configured", cuda_probe, init_options.cuda_topology, progress),
            .auto => {},
        }

        // Check if we should use Metal backend (macOS + quantized/bf16 model)
        const has_unsupported_runtime_features = runtimeHasMetalUnsupportedFeatures(&loaded.runtime);
        if (has_metal and isMetalSupported(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features)) {
            return initMetal(allocator, loaded, "auto", init_options.metal) catch |err| {
                if (err == error.MoENotSupported or
                    err == error.MLXNotAvailable or
                    err == error.UnsupportedDType or
                    err == error.ShortConvNotSupportedOnMetal or
                    err == error.MLANotSupportedOnMetal or
                    err == error.InvalidTensorType or
                    err == error.OutOfMemory or
                    err == error.UnsupportedModel or
                    err == error.NotImplemented or
                    err == error.DecodeModelUnavailable)
                {
                    log.warn("inference", "Metal backend unavailable; trying CPU fallback", .{
                        .reason = @errorName(err),
                        .detail = getMetalUnsupportedReason(&loaded.config, &loaded.runtime, loaded.original_weight_dtype, has_unsupported_runtime_features),
                    });
                    return initCpu(allocator, loaded, "auto_fallback", progress);
                }
                return err;
            };
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
            .cuda => false,
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
                )
            else
                error.InvalidArgument,
            .cuda => error.InvalidArgument,
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
    runtime: *const tensor.ModelRuntime,
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
    runtime: *const tensor.ModelRuntime,
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

fn runtimeHasMetalUnsupportedFeatures(runtime: *const tensor.ModelRuntime) bool {
    // Metal decode-model path currently does not support recurrent mamba
    // layer topologies.
    return runtime.has_mamba;
}

fn initCpu(
    allocator: std.mem.Allocator,
    loaded: *LoadedModel,
    reason: []const u8,
    progress: progress_mod.Context,
) !Backend {
    const cpu_backend_state = cpu.BackendType.init(allocator, loaded, resolveMaxBatchSize(), progress) catch |err| {
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
    var topology = resolveCudaTopology(allocator, topology_override);

    // Topology resolution priority:
    //   1. topology_override (programmatic)
    //   2. TALU_CUDA_TOPOLOGY env var (explicit mode)
    //   3. TALU_CPU_LAYERS env var (user preference → auto-determines mode)
    //   4. Auto-detection (probe GPU memory, estimate model size)
    if (topology_override == null and topology.mode == .single) {
        const has_explicit_env = blk: {
            const v = std.process.getEnvVarOwned(allocator, "TALU_CUDA_TOPOLOGY") catch break :blk false;
            allocator.free(v);
            break :blk true;
        };
        if (!has_explicit_env) {
            if (resolveCpuLayersTopology(allocator, loaded, topology)) |from_layers| {
                topology = from_layers;
            } else if (autoDetectTopologyForModel(allocator, loaded, topology)) |detected| {
                topology = detected;
            } else |err| {
                log.warn("inference", "Auto topology detection failed; using single GPU", .{
                    .err = @errorName(err),
                });
            }
        }
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
    const cuda_max_batch_size = resolveMaxBatchSize();
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

test "parseCudaTopologyMode parses supported modes" {
    try std.testing.expectEqual(CudaTopologyMode.single, parseCudaTopologyMode("single").?);
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, parseCudaTopologyMode("pipeline2").?);
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, parseCudaTopologyMode("pipeline_2way").?);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, parseCudaTopologyMode("cpu_gpu").?);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, parseCudaTopologyMode("cpu+gpu").?);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, parseCudaTopologyMode("cpu_gpu_gpu").?);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, parseCudaTopologyMode("cpu+gpu+gpu").?);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, parseCudaTopologyMode("pipeline_cpu_gpu_gpu").?);
}

test "parseCudaTopologyMode rejects unsupported modes" {
    try std.testing.expectEqual(@as(?CudaTopologyMode, null), parseCudaTopologyMode(""));
    try std.testing.expectEqual(@as(?CudaTopologyMode, null), parseCudaTopologyMode("mesh"));
}

test "parseTwoDeviceOrdinals parses gpu pairs" {
    try std.testing.expectEqualDeep(@as([2]usize, .{ 0, 1 }), parseTwoDeviceOrdinals("0,1").?);
    try std.testing.expectEqualDeep(@as([2]usize, .{ 3, 9 }), parseTwoDeviceOrdinals(" 3 , 9 ").?);
    try std.testing.expectEqual(@as(?[2]usize, null), parseTwoDeviceOrdinals("0"));
    try std.testing.expectEqual(@as(?[2]usize, null), parseTwoDeviceOrdinals("a,1"));
    try std.testing.expectEqual(@as(?[2]usize, null), parseTwoDeviceOrdinals("0,1,2"));
}

test "resolveCudaTopology honors explicit override" {
    const topology = resolveCudaTopology(std.testing.allocator, .{
        .mode = .pipeline2,
        .stage_device_ordinals = .{ 4, 5 },
        .split_layer = 7,
        .split_layer_stage2 = 11,
    });
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, topology.mode);
    try std.testing.expectEqualDeep(@as([2]usize, .{ 4, 5 }), topology.stage_device_ordinals);
    try std.testing.expectEqual(@as(?usize, 7), topology.split_layer);
    try std.testing.expectEqual(@as(?usize, 11), topology.split_layer_stage2);
}

test "validateCudaTopologyConfig accepts single topology" {
    const topology = CudaTopology{
        .mode = .single,
        .stage_device_ordinals = .{ 0, 0 },
    };
    try validateCudaTopologyConfig(topology, 1, 1);
}

test "validateCudaTopologyConfig rejects pipeline2 with insufficient layers" {
    const topology = CudaTopology{
        .mode = .pipeline2,
        .stage_device_ordinals = .{ 0, 1 },
    };
    try std.testing.expectError(
        error.Pipeline2InsufficientLayers,
        validateCudaTopologyConfig(topology, 1, 2),
    );
}

test "validateCudaTopologyConfig rejects pipeline2 with duplicate device ordinals" {
    const topology = CudaTopology{
        .mode = .pipeline2,
        .stage_device_ordinals = .{ 1, 1 },
    };
    try std.testing.expectError(
        error.Pipeline2RequiresDistinctDevices,
        validateCudaTopologyConfig(topology, 8, 4),
    );
}

test "validateCudaTopologyConfig rejects pipeline2 when ordinal is out of range" {
    const topology = CudaTopology{
        .mode = .pipeline2,
        .stage_device_ordinals = .{ 0, 3 },
    };
    try std.testing.expectError(
        error.Pipeline2DeviceOrdinalOutOfRange,
        validateCudaTopologyConfig(topology, 8, 2),
    );
}

test "validateCudaTopologyConfig rejects pipeline2 split layer out of range" {
    const topology = CudaTopology{
        .mode = .pipeline2,
        .stage_device_ordinals = .{ 0, 1 },
        .split_layer = 8,
    };
    try std.testing.expectError(
        error.Pipeline2InvalidSplitLayer,
        validateCudaTopologyConfig(topology, 8, 2),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu with insufficient layers" {
    const topology = CudaTopology{
        .mode = .cpu_gpu,
        .stage_device_ordinals = .{ 0, 0 },
    };
    try std.testing.expectError(
        error.CpuGpuInsufficientLayers,
        validateCudaTopologyConfig(topology, 1, 1),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu split layer out of range" {
    const topology = CudaTopology{
        .mode = .cpu_gpu,
        .stage_device_ordinals = .{ 0, 0 },
        .split_layer = 0,
    };
    try std.testing.expectError(
        error.CpuGpuInvalidSplitLayer,
        validateCudaTopologyConfig(topology, 8, 2),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu when gpu ordinal is out of range" {
    const topology = CudaTopology{
        .mode = .cpu_gpu,
        .stage_device_ordinals = .{ 3, 0 },
    };
    try std.testing.expectError(
        error.CpuGpuDeviceOrdinalOutOfRange,
        validateCudaTopologyConfig(topology, 8, 2),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu_gpu with insufficient layers" {
    const topology = CudaTopology{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ 0, 1 },
    };
    try std.testing.expectError(
        error.CpuGpuGpuInsufficientLayers,
        validateCudaTopologyConfig(topology, 2, 2),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu_gpu split layer ordering" {
    const topology = CudaTopology{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ 0, 1 },
        .split_layer = 5,
        .split_layer_stage2 = 4,
    };
    try std.testing.expectError(
        error.CpuGpuGpuSplitOrderInvalid,
        validateCudaTopologyConfig(topology, 12, 2),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu_gpu with duplicate gpu ordinals" {
    const topology = CudaTopology{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ 1, 1 },
        .split_layer = 3,
        .split_layer_stage2 = 6,
    };
    try std.testing.expectError(
        error.CpuGpuGpuRequiresDistinctDevices,
        validateCudaTopologyConfig(topology, 10, 4),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu_gpu second split out of range" {
    const topology = CudaTopology{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ 0, 1 },
        .split_layer = 2,
        .split_layer_stage2 = 10,
    };
    try std.testing.expectError(
        error.CpuGpuGpuInvalidSplitLayerStage2,
        validateCudaTopologyConfig(topology, 10, 2),
    );
}

test "validateCudaTopologyConfig rejects cpu_gpu_gpu when any gpu ordinal is out of range" {
    const topology = CudaTopology{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ 0, 3 },
        .split_layer = 2,
        .split_layer_stage2 = 7,
    };
    try std.testing.expectError(
        error.CpuGpuGpuDeviceOrdinalOutOfRange,
        validateCudaTopologyConfig(topology, 12, 2),
    );
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

test "metal module surface does not expose legacy runtime graph symbols" {
    if (!has_metal) return;
    try std.testing.expect(!@hasDecl(metal, "runtime_graph"));
}

test "metal module surface maps to current cpu-backed helper modules" {
    if (!has_metal) return;
    try std.testing.expect(metal.executor.Model == cpu.executor.Model);
    try std.testing.expect(metal.kernels.RMSNorm == cpu.kernels.RMSNorm);
    try std.testing.expect(metal.vision.VisionRuntime == cpu.vision.VisionRuntime);
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

// ---------------------------------------------------------------------------
// Auto topology selection tests
// ---------------------------------------------------------------------------

fn testGpuInfos(comptime N: usize, pairs: [N][2]usize) [N]GpuMemoryInfo {
    var infos: [N]GpuMemoryInfo = undefined;
    for (pairs, 0..) |p, i| {
        infos[i] = .{ .ordinal = i, .free = p[0], .total = p[1] };
    }
    return infos;
}

/// Test helper: minimal model params where file_size dominates.
/// embed/proj ≈ 0, kv ≈ 0, so computed bytes ≈ file_size.
fn testMinimalParams(file_size: usize, n_layers: usize) ModelSizeParams {
    return .{
        .file_size = file_size,
        .vocab_size = 1,
        .d_model = 1,
        .n_kv_heads = 0,
        .head_dim = 0,
        .max_seq_len = 0,
        .n_layers = n_layers,
    };
}

test "estimateModelGpuBytes includes weights and KV cache" {
    const gb = 1024 * 1024 * 1024;
    const p = ModelSizeParams{
        .file_size = gb,
        .vocab_size = 32000,
        .d_model = 4096,
        .n_kv_heads = 8,
        .head_dim = 128,
        .max_seq_len = 4096,
        .n_layers = 32,
    };
    const result = estimateModelGpuBytes(p);
    // Should be > file_size due to KV cache + activation buffers + embed/proj.
    try std.testing.expect(result > gb);
    // Should be reasonable (< 3x file_size).
    try std.testing.expect(result < 3 * gb);
}

test "estimatePerGpuBytes: intermediate stage is cheaper than full" {
    const p = ModelSizeParams{
        .file_size = 2 * 1024 * 1024 * 1024,
        .vocab_size = 32000,
        .d_model = 4096,
        .n_kv_heads = 8,
        .head_dim = 128,
        .max_seq_len = 4096,
        .n_layers = 32,
    };
    const full = estimatePerGpuBytes(p, 16, true, true);
    const mid = estimatePerGpuBytes(p, 16, false, false);
    // Intermediate stage (no embed/proj) should be significantly cheaper.
    try std.testing.expect(mid < full);
}

test "autoSelectTopology S3: 1 GPU, model fits → single" {
    const p = testMinimalParams(2 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    const result = autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
    try std.testing.expectEqual(@as(usize, 0), result.stage_device_ordinals[0]);
}

test "autoSelectTopology S4: 1 GPU, model too large → cpu_gpu" {
    // 4 GB free, ~10 GB model, 32 layers.
    const p = testMinimalParams(10 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(1, .{.{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 }});
    const result = autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    // split_layer should leave some layers on GPU.
    try std.testing.expect(result.split_layer != null);
    try std.testing.expect(result.split_layer.? > 0);
    try std.testing.expect(result.split_layer.? < 32);
}

test "autoSelectTopology S5: 2 GPUs, fits on best single → single on best" {
    // GPU0: 4 GB free, GPU1: 12 GB free. Model needs ~5 GB.
    const p = testMinimalParams(5 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 4 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
        .{ 12 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
    });
    const result = autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
    // Should pick GPU1 (more free memory).
    try std.testing.expectEqual(@as(usize, 1), result.stage_device_ordinals[0]);
}

test "autoSelectTopology S6: 2 GPUs, fits across both → pipeline2" {
    // GPU0: 4 GB free, GPU1: 4 GB free. Model needs ~6 GB.
    const p = testMinimalParams(6 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, result.mode);
    try std.testing.expect(result.split_layer != null);
    // Equal memory → split near middle.
    try std.testing.expect(result.split_layer.? >= 12 and result.split_layer.? <= 20);
    try std.testing.expectEqual(@as(usize, 0), result.stage_device_ordinals[0]);
    try std.testing.expectEqual(@as(usize, 1), result.stage_device_ordinals[1]);
}

test "autoSelectTopology S6: pipeline2 split proportional to unequal GPU memory" {
    // GPU0: 6 GB free, GPU1: 2 GB free. Model needs ~7 GB.
    const p = testMinimalParams(7 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, result.mode);
    // GPU0 has 3x more memory → split should put more layers on GPU0 (higher split_layer).
    try std.testing.expect(result.split_layer.? >= 18);
}

test "autoSelectTopology S7: 2 GPUs, doesn't fit → cpu_gpu_gpu" {
    // GPU0: 4 GB, GPU1: 4 GB. Model needs ~12 GB.
    const p = testMinimalParams(12 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, result.mode);
    // CPU takes some layers, rest split across GPUs.
    try std.testing.expect(result.split_layer != null);
    try std.testing.expect(result.split_layer.? > 0);
    try std.testing.expect(result.split_layer_stage2 != null);
    try std.testing.expect(result.split_layer_stage2.? > result.split_layer.?);
    try std.testing.expect(result.split_layer_stage2.? < 32);
}

test "autoSelectTopology S8: too large for everything → falls back to single" {
    // GPU0: 1 GB, GPU1: 1 GB. Model needs ~100 GB.
    const p = testMinimalParams(100 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024 },
        .{ 1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024 },
    });
    const result = autoSelectTopology(p, 32, &infos);
    // Falls back to single (caller handles OOM).
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
}

test "autoSelectTopology with 0 GPUs returns single" {
    const p = testMinimalParams(1024, 32);
    const infos: []const GpuMemoryInfo = &.{};
    const result = autoSelectTopology(p, 32, infos);
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
}

test "autoSelectTopology with 1 layer returns single" {
    const p = testMinimalParams(4096, 1);
    const infos = testGpuInfos(1, .{.{ 1024, 2048 }});
    const result = autoSelectTopology(p, 1, &infos);
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
}

test "autoSelectTopology S9: 3 GPUs, picks best 2" {
    // GPU0: 2 GB, GPU1: 6 GB, GPU2: 6 GB. Model needs ~8 GB.
    const p = testMinimalParams(8 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(3, .{
        .{ 2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = autoSelectTopology(p, 32, &infos);
    // Should use GPU1 and GPU2 (most free memory), as pipeline2.
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, result.mode);
    try std.testing.expectEqual(@as(usize, 1), result.stage_device_ordinals[0]);
    try std.testing.expectEqual(@as(usize, 2), result.stage_device_ordinals[1]);
}

// ---------------------------------------------------------------------------
// topologyFromCpuLayers tests
// ---------------------------------------------------------------------------

test "topologyFromCpuLayers: cpu_layers=0 returns null" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    try std.testing.expect(topologyFromCpuLayers(0, 32, &infos, null) == null);
}

test "topologyFromCpuLayers: cpu_layers >= total_layers returns null" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    try std.testing.expect(topologyFromCpuLayers(32, 32, &infos, null) == null);
    try std.testing.expect(topologyFromCpuLayers(33, 32, &infos, null) == null);
}

test "topologyFromCpuLayers: 1 GPU → cpu_gpu" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    const result = topologyFromCpuLayers(8, 32, &infos, null).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 8), result.split_layer);
}

test "topologyFromCpuLayers: 2 GPUs equal memory → cpu_gpu_gpu even split (no model)" {
    const infos = testGpuInfos(2, .{
        .{ 16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
        .{ 16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
    });
    const result = topologyFromCpuLayers(8, 32, &infos, null).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 8), result.split_layer);
    // 24 GPU layers, equal GPUs → gpu0 gets 12, gpu1 gets 12.
    try std.testing.expectEqual(@as(?usize, 20), result.split_layer_stage2); // 8 + 12 = 20
}

test "topologyFromCpuLayers: model-aware split gives GPU0 more layers (proj on GPU1)" {
    const gb = 1024 * 1024 * 1024;
    const infos = testGpuInfos(2, .{
        .{ 16 * gb, 16 * gb },
        .{ 16 * gb, 16 * gb },
    });
    // Large vocab means projection weight is significant.
    const p = ModelSizeParams{
        .file_size = 10 * gb,
        .vocab_size = 262144,
        .d_model = 5376,
        .n_kv_heads = 8,
        .head_dim = 256,
        .max_seq_len = 4096,
        .n_layers = 60,
    };
    const result = topologyFromCpuLayers(36, 60, &infos, p).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, result.mode);
    // GPU0 (no proj) should get more layers than GPU1 (has proj).
    const gpu0_layers = result.split_layer_stage2.? - 36;
    const gpu1_layers = 60 - result.split_layer_stage2.?;
    try std.testing.expect(gpu0_layers > gpu1_layers);
}

test "topologyFromCpuLayers: 2 GPUs unequal memory → proportional split" {
    const gb = 1024 * 1024 * 1024;
    const infos = testGpuInfos(2, .{
        .{ 12 * gb, 16 * gb }, // GPU0: 12 GB
        .{ 4 * gb, 8 * gb }, // GPU1: 4 GB
    });
    const result = topologyFromCpuLayers(12, 32, &infos, null).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 12), result.split_layer);
    // 20 GPU layers, GPU0 has 3x more memory → gets most layers.
    const stage2 = result.split_layer_stage2.?;
    try std.testing.expect(stage2 > 12 and stage2 < 32);
    try std.testing.expect(stage2 >= 24); // GPU0 share ≈ 15
}

test "topologyFromCpuLayers: only 1 GPU layer → cpu_gpu even with 2 GPUs" {
    const gb = 1024 * 1024 * 1024;
    const infos = testGpuInfos(2, .{
        .{ 8 * gb, 16 * gb },
        .{ 8 * gb, 16 * gb },
    });
    // cpu_layers=31 → only 1 GPU layer, can't split across 2 GPUs.
    const result = topologyFromCpuLayers(31, 32, &infos, null).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 31), result.split_layer);
}

test "topologyFromCpuLayers: no GPUs returns null" {
    const infos = [_]GpuMemoryInfo{};
    try std.testing.expect(topologyFromCpuLayers(8, 32, &infos, null) == null);
}

test "topologyFromCpuLayers: total_layers < 2 returns null" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    try std.testing.expect(topologyFromCpuLayers(1, 1, &infos, null) == null);
}

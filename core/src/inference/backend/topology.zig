//! CUDA topology selection and validation policy.
//!
//! This module owns CUDA stage topology parsing, validation, memory estimation,
//! and model-shape policy used before CUDA backend initialization.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const compute = @import("compute_pkg");
const models = @import("models_pkg");
const log = @import("log_pkg");
const tensor = @import("compute_pkg").tensor;

const ModelConfig = models.config.ModelConfig;
const LoadedModel = models.LoadedModel;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);

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

pub const CudaTopology = struct {
    mode: CudaTopologyMode = .single,
    stage_device_ordinals: [2]usize = .{ 0, 1 },
    split_layer: ?usize = null,
    split_layer_stage2: ?usize = null,

    pub fn primaryDeviceOrdinal(self: *const CudaTopology) usize {
        return switch (self.mode) {
            .cpu_gpu_gpu => self.stage_device_ordinals[1],
            else => self.stage_device_ordinals[0],
        };
    }
};

pub const CudaTopologyCapabilities = struct {
    supports_pipeline2: bool = true,
    supports_cpu_gpu: bool = true,
    supports_cpu_gpu_gpu: bool = true,
    min_pipeline2_layers: usize = 2,
    min_cpu_gpu_layers: usize = 2,
    min_cpu_gpu_gpu_layers: usize = 3,
    requires_distinct_stage_devices: bool = true,
};

pub const cuda_topology_capabilities = CudaTopologyCapabilities{};

pub const CudaTopologyValidationError = error{
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

pub fn validateCudaTopologyConfig(
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

fn parseCudaTopologyMode(raw: []const u8) !CudaTopologyMode {
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (token.len == 0) return error.InvalidTopologyConfig;
    if (std.ascii.eqlIgnoreCase(token, "single")) return .single;
    if (std.ascii.eqlIgnoreCase(token, "pipeline2")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_2")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_2way")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "gpu_gpu")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "gpu-gpu")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "gpu+gpu")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_gpu_gpu")) return .pipeline2;
    if (std.ascii.eqlIgnoreCase(token, "cpu_gpu")) return .cpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "cpu+gpu")) return .cpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_cpu_gpu")) return .cpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "cpu_gpu_gpu")) return .cpu_gpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "cpu+gpu+gpu")) return .cpu_gpu_gpu;
    if (std.ascii.eqlIgnoreCase(token, "pipeline_cpu_gpu_gpu")) return .cpu_gpu_gpu;
    return error.InvalidTopologyConfig;
}

fn parseTwoDeviceOrdinals(raw: []const u8) ![2]usize {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidTopologyConfig;
    var iter = std.mem.splitScalar(u8, trimmed, ',');
    const left_raw = iter.next() orelse return error.InvalidTopologyConfig;
    const right_raw = iter.next() orelse return error.InvalidTopologyConfig;
    if (iter.next() != null) return error.InvalidTopologyConfig;
    const left = std.fmt.parseUnsigned(usize, std.mem.trim(u8, left_raw, " \t\r\n"), 10) catch return error.InvalidTopologyConfig;
    const right = std.fmt.parseUnsigned(usize, std.mem.trim(u8, right_raw, " \t\r\n"), 10) catch return error.InvalidTopologyConfig;
    return .{ left, right };
}

fn getOptionalEnvVarOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => err,
    };
}

pub fn resolveCudaTopology(
    allocator: std.mem.Allocator,
    topology_override: ?CudaTopologyConfig,
) !CudaTopology {
    var topology = CudaTopology{};
    if (topology_override) |cfg| {
        topology.mode = cfg.mode;
        topology.stage_device_ordinals = cfg.stage_device_ordinals;
        topology.split_layer = cfg.split_layer;
        topology.split_layer_stage2 = cfg.split_layer_stage2;
        return topology;
    }

    const mode_raw = try getOptionalEnvVarOwned(allocator, "TALU_CUDA_TOPOLOGY");
    if (mode_raw) |value| {
        defer allocator.free(value);
        topology.mode = parseCudaTopologyMode(value) catch |err| {
            log.err("inference", "Invalid TALU_CUDA_TOPOLOGY", .{ .value = value }, @src());
            return err;
        };
    }

    const devices_raw = try getOptionalEnvVarOwned(allocator, "TALU_CUDA_STAGE_DEVICES");
    if (devices_raw) |value| {
        defer allocator.free(value);
        topology.stage_device_ordinals = parseTwoDeviceOrdinals(value) catch |err| {
            log.err("inference", "Invalid TALU_CUDA_STAGE_DEVICES; expected '<gpu0>,<gpu1>'", .{ .value = value }, @src());
            return err;
        };
    }

    const single_device_raw = try getOptionalEnvVarOwned(allocator, "TALU_CUDA_DEVICE");
    if (single_device_raw) |value| {
        defer allocator.free(value);
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
            log.err("inference", "Invalid TALU_CUDA_DEVICE", .{
                .value = trimmed,
                .current = topology.primaryDeviceOrdinal(),
            }, @src());
            return error.InvalidTopologyConfig;
        };
        if (topology.mode == .cpu_gpu_gpu) {
            topology.stage_device_ordinals[1] = parsed;
        } else {
            topology.stage_device_ordinals[0] = parsed;
        }
    }

    const split_layer_raw = try getOptionalEnvVarOwned(allocator, "TALU_CUDA_SPLIT_LAYER");
    if (split_layer_raw) |value| {
        defer allocator.free(value);
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
            log.err("inference", "Invalid TALU_CUDA_SPLIT_LAYER", .{
                .value = trimmed,
            }, @src());
            return error.InvalidTopologyConfig;
        };
        topology.split_layer = parsed;
    }

    const split_layer_stage2_raw = try getOptionalEnvVarOwned(allocator, "TALU_CUDA_SPLIT_LAYER_STAGE2");
    if (split_layer_stage2_raw) |value| {
        defer allocator.free(value);
        const trimmed = std.mem.trim(u8, value, " \t\r\n");
        const parsed = std.fmt.parseUnsigned(usize, trimmed, 10) catch {
            log.err("inference", "Invalid TALU_CUDA_SPLIT_LAYER_STAGE2", .{
                .value = trimmed,
            }, @src());
            return error.InvalidTopologyConfig;
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
    manifest: ?*const models.manifest.ModelManifest = null,
};

/// Compute GPU memory for a specific stage layer range.
///
/// Separates checkpoint weights from KV cache and activation buffers. When a
/// model manifest is present, stage weight bytes come from exact tensor
/// ownership; otherwise the estimator uses the historical coarse size model.
/// Intermediate stages skip embedding/projection uploads at init time.
fn estimatePerGpuBytes(
    p: ModelSizeParams,
    layer_start: usize,
    layer_end: usize,
    needs_embedding: bool,
    needs_projection: bool,
) !usize {
    const gpu_layers = layer_end - layer_start;

    const elem_bytes: usize = 2;
    const coarse_embed_bytes = p.vocab_size *| p.d_model *| elem_bytes;
    const coarse_proj_bytes = coarse_embed_bytes;
    const coarse_non_layer_bytes = coarse_embed_bytes +| coarse_proj_bytes;
    const coarse_layer_weight_bytes = if (p.file_size > coarse_non_layer_bytes and p.n_layers > 0)
        (p.file_size - coarse_non_layer_bytes) / p.n_layers
    else if (p.n_layers > 0)
        p.file_size / p.n_layers
    else
        p.file_size;

    const weight_bytes = if (p.manifest) |manifest| blk: {
        const include_global_side = needs_embedding or needs_projection;
        const report = manifest.stageResidencyReport(.{
            .layer_start = layer_start,
            .layer_end = layer_end,
            .include_token_embeddings = needs_embedding,
            .include_final_norm = needs_projection,
            .include_lm_head = needs_projection,
            .include_embedding_side = needs_embedding,
            .include_vision_side = include_global_side,
            .include_architecture_side = include_global_side,
            .include_unclassified_global = include_global_side,
        }) catch return error.InvalidTopologyConfig;
        break :blk report.total_checkpoint_bytes;
    } else gpu_layers *| coarse_layer_weight_bytes +
        (if (needs_embedding) coarse_embed_bytes else 0) +
        (if (needs_projection) coarse_proj_bytes else 0);

    // KV cache per layer: n_kv_heads * head_dim * max_seq_len * 2(K+V) * 2(f16).
    const kv_per_layer = p.n_kv_heads *| p.head_dim *| p.max_seq_len *| 4;

    // Activation buffers: ~30 d_model-sized f32 buffers + dequant scratch.
    const activation_bytes = p.d_model *| 30 *| 4;

    // Sum components.
    var total: usize = 0;
    total +|= weight_bytes;
    total +|= gpu_layers *| kv_per_layer;
    total +|= activation_bytes;

    return total;
}

/// Compute total GPU memory for full model (used for logging).
fn estimateModelGpuBytes(p: ModelSizeParams) !usize {
    return estimatePerGpuBytes(p, 0, p.n_layers, true, true);
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
    layer_start: usize,
    max_layers: usize,
    needs_embedding: bool,
    needs_projection: bool,
) !usize {
    // Binary search for max layers that fit.
    var lo: usize = 1;
    var hi: usize = max_layers;
    var best: usize = 0;
    while (lo <= hi) {
        const mid = lo + (hi - lo) / 2;
        const est = try estimatePerGpuBytes(p, layer_start, layer_start + mid, needs_embedding, needs_projection);
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

fn maxSuffixLayersForBudget(
    p: ModelSizeParams,
    budget: usize,
    total_layers: usize,
    needs_embedding: bool,
    needs_projection: bool,
) !usize {
    var lo: usize = 1;
    var hi: usize = total_layers;
    var best: usize = 0;
    while (lo <= hi) {
        const mid = lo + (hi - lo) / 2;
        const start = total_layers - mid;
        const est = try estimatePerGpuBytes(p, start, total_layers, needs_embedding, needs_projection);
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

fn utilPerMille(bytes: usize, budget: usize) usize {
    if (budget == 0) return std.math.maxInt(usize);
    return (bytes *| 1000) / budget;
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
) !CudaTopology {
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
    const single_est = try estimatePerGpuBytes(p, 0, total_layers, true, true);
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
            try maxSuffixLayersForBudget(p, best0_budget, total_layers, false, true),
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
        const max_gpu0 = try maxLayersForBudget(p, budget0, 0, total_layers - 1, true, false);
        if (max_gpu0 >= 1) {
            const gpu1_est = try estimatePerGpuBytes(p, max_gpu0, total_layers, false, true);
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
                    const g0 = try estimatePerGpuBytes(p, 0, split, true, false);
                    const g1 = try estimatePerGpuBytes(p, split, total_layers, false, true);
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
                const g0_final = try estimatePerGpuBytes(p, 0, split, true, false);
                const g1_final = try estimatePerGpuBytes(p, split, total_layers, false, true);
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
    var best_cpu_layers: usize = 0;
    var best_split_stage2: usize = 0;
    var best_gpu_layers: usize = 0;
    var best_max_util: usize = std.math.maxInt(usize);

    var cpu_layers: usize = 1;
    while (cpu_layers <= total_layers - 2) : (cpu_layers += 1) {
        var split_stage2 = cpu_layers + 1;
        while (split_stage2 < total_layers) : (split_stage2 += 1) {
            const g0 = try estimatePerGpuBytes(p, cpu_layers, split_stage2, false, false);
            const g1 = try estimatePerGpuBytes(p, split_stage2, total_layers, false, true);
            if (g0 > budget0 or g1 > budget1) continue;

            const gpu_layers = total_layers - cpu_layers;
            const util0 = utilPerMille(g0, budget0);
            const util1 = utilPerMille(g1, budget1);
            const max_util = @max(util0, util1);
            if (gpu_layers > best_gpu_layers or
                (gpu_layers == best_gpu_layers and max_util < best_max_util))
            {
                best_cpu_layers = cpu_layers;
                best_split_stage2 = split_stage2;
                best_gpu_layers = gpu_layers;
                best_max_util = max_util;
            }
        }
    }
    if (best_gpu_layers < 2) {
        // S8: Not enough GPU memory even for cpu_gpu_gpu. Return single and let
        // the caller handle the OOM or error.
        return .{};
    }
    return .{
        .mode = .cpu_gpu_gpu,
        .stage_device_ordinals = .{ ord0, ord1 },
        .split_layer = best_cpu_layers,
        .split_layer_stage2 = best_split_stage2,
    };
}

/// Run full auto-detection: probe GPU memory, estimate model requirements, select topology.
pub fn autoDetectTopologyForModel(
    allocator: std.mem.Allocator,
    loaded: *const LoadedModel,
) !CudaTopology {
    const device_count = if (has_cuda)
        compute.cuda.Device.deviceCount() catch |err| {
            log.err("inference", "CUDA topology auto-detection failed to query device count", .{
                .err = @errorName(err),
            }, @src());
            return error.CudaUnavailable;
        }
    else
        return error.CudaNotEnabled;

    if (device_count == 0) return error.CudaUnavailable;

    const gpu_infos = probeGpuTotalMemory(allocator, device_count) catch |err| {
        log.err("inference", "CUDA topology auto-detection failed to probe GPU memory", .{
            .err = @errorName(err),
        }, @src());
        return error.InvalidTopologyConfig;
    };
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
        .manifest = loaded.manifestPtr(),
    };

    const estimated = try estimateModelGpuBytes(params);

    const result = try autoSelectTopology(params, loaded.blocks.len, gpu_infos);

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
/// For models with KV sharing, later layers reuse the KV cache
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
/// With 2+ GPUs, prefer one GPU when the GPU-owned suffix fits there; split the
/// suffix only when model-aware estimation says one GPU is insufficient.
///
/// Returns null when cpu_layers is 0 (all on GPU → auto-detect) or invalid.
fn topologyFromCpuLayers(
    cpu_layers: usize,
    total_layers: usize,
    gpu_infos: []const GpuMemoryInfo,
    model_params: ?ModelSizeParams,
) !?CudaTopology {
    if (total_layers < 2 or cpu_layers == 0 or cpu_layers >= total_layers) return null;
    if (gpu_infos.len == 0) return null;

    const gpu_layers = total_layers - cpu_layers;

    var best0: usize = 0;
    var best1: usize = if (gpu_infos.len > 1) 1 else 0;
    for (gpu_infos, 0..) |info, i| {
        if (info.free > gpu_infos[best0].free) {
            best1 = best0;
            best0 = i;
        } else if (gpu_infos.len > 1 and i != best0 and info.free > gpu_infos[best1].free) {
            best1 = i;
        }
    }
    const best0_budget = if (gpu_infos[best0].free > AUTO_TOPO_OVERHEAD_BYTES)
        gpu_infos[best0].free - AUTO_TOPO_OVERHEAD_BYTES
    else
        0;

    if (model_params) |p| {
        const one_gpu_est = try estimatePerGpuBytes(p, cpu_layers, total_layers, false, true);
        if (one_gpu_est <= best0_budget) {
            return .{
                .mode = .cpu_gpu,
                .stage_device_ordinals = .{ gpu_infos[best0].ordinal, gpu_infos[best0].ordinal },
                .split_layer = cpu_layers,
            };
        }
    } else {
        // Without model parameters, avoid using a second GPU just because it is
        // visible. The caller passes model parameters on the normal load path.
        return .{
            .mode = .cpu_gpu,
            .stage_device_ordinals = .{ gpu_infos[best0].ordinal, gpu_infos[best0].ordinal },
            .split_layer = cpu_layers,
        };
    }

    if (gpu_infos.len == 1 or gpu_layers < 2) {
        // One visible GPU, or only one GPU layer, cannot use a second GPU stage.
        return .{
            .mode = .cpu_gpu,
            .stage_device_ordinals = .{ gpu_infos[best0].ordinal, gpu_infos[best0].ordinal },
            .split_layer = cpu_layers,
        };
    }

    // 2+ GPUs: cpu_gpu_gpu. Pick best 2 GPUs and split.
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
            const split_stage2 = cpu_layers + s;
            const g0 = try estimatePerGpuBytes(p, cpu_layers, split_stage2, false, false);
            const g1 = try estimatePerGpuBytes(p, split_stage2, total_layers, false, true);
            const util0 = utilPerMille(g0, budget0);
            const util1 = utilPerMille(g1, budget1);
            const max_util = @max(util0, util1);
            if (max_util < best_max_util) {
                best_max_util = max_util;
                best_split = s;
            }
        }
        break :blk best_split;
    } else blk: {
        // Default: proportional by memory (no model info available).
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
pub fn resolveCpuLayersTopology(
    allocator: std.mem.Allocator,
    loaded: *const LoadedModel,
) !?CudaTopology {
    const raw = (try getOptionalEnvVarOwned(allocator, "TALU_CPU_LAYERS")) orelse return null;
    defer allocator.free(raw);
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    const cpu_layers_i = std.fmt.parseInt(i32, trimmed, 10) catch {
        log.err("inference", "Invalid TALU_CPU_LAYERS", .{ .value = trimmed }, @src());
        return error.InvalidTopologyConfig;
    };

    const total_layers = loaded.blocks.len;
    if (total_layers < 2) {
        log.err("inference", "TALU_CPU_LAYERS requires at least two decoder layers", .{
            .total_layers = total_layers,
        }, @src());
        return error.InvalidTopologyConfig;
    }
    if (cpu_layers_i < 0) {
        log.err("inference", "Invalid TALU_CPU_LAYERS", .{ .value = trimmed }, @src());
        return error.InvalidTopologyConfig;
    }
    if (cpu_layers_i == 0) return null;

    if (cpu_layers_i >= @as(i32, @intCast(total_layers))) {
        log.err("inference", "TALU_CPU_LAYERS leaves no layer for CUDA", .{
            .requested = cpu_layers_i,
            .total_layers = total_layers,
        }, @src());
        return error.InvalidTopologyConfig;
    }
    const cpu_layers: usize = @intCast(cpu_layers_i);

    // Query GPU count and memory.
    const device_count = if (has_cuda)
        compute.cuda.Device.deviceCount() catch |err| {
            log.err("inference", "TALU_CPU_LAYERS: failed to query device count", .{
                .err = @errorName(err),
            }, @src());
            return error.CudaUnavailable;
        }
    else
        return error.CudaNotEnabled;

    if (device_count == 0) return error.CudaUnavailable;

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
        .manifest = loaded.manifestPtr(),
    };

    const gpu_infos = probeGpuTotalMemory(allocator, device_count) catch |err| {
        log.err("inference", "TALU_CPU_LAYERS: failed to probe GPU memory", .{
            .err = @errorName(err),
        }, @src());
        return error.InvalidTopologyConfig;
    };
    defer allocator.free(gpu_infos);

    var result = (try topologyFromCpuLayers(cpu_layers, total_layers, gpu_infos, p)) orelse {
        log.err("inference", "TALU_CPU_LAYERS produced an invalid topology", .{
            .cpu_layers = cpu_layers,
            .total_layers = total_layers,
            .device_count = device_count,
        }, @src());
        return error.InvalidTopologyConfig;
    };

    // cpu_gpu_gpu splits GPU layers across 2 devices which requires KV shared
    // source layers to be strictly above cpu_layers (room for GPU1 stage).
    // When sources are in the CPU range, use cpu_gpu so cross-device KV
    // replication handles them.
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

pub fn loadedModelHasPackedNvfp4Weights(loaded: *const LoadedModel) bool {
    if (loaded.token_embeddings.nvfp4 != null) return true;
    if (loaded.lm_head) |tensor_view| if (tensor_view.nvfp4 != null) return true;
    if (loaded.position_embeddings) |tensor_view| if (tensor_view.nvfp4 != null) return true;
    if (loaded.token_type_embeddings) |tensor_view| if (tensor_view.nvfp4 != null) return true;
    if (loaded.embedding_norm_weight) |tensor_view| if (tensor_view.nvfp4 != null) return true;
    if (loaded.embedding_norm_bias) |tensor_view| if (tensor_view.nvfp4 != null) return true;

    for (loaded.blocks) |layer| {
        var it = layer.weight_map.iterator();
        while (it.next()) |entry| {
            const tensor_view = entry.value_ptr.*.*;
            if (tensor_view.nvfp4 != null) return true;
        }
    }

    return false;
}

// ============================================================================
// Tests
// ============================================================================

test "loadedModelHasPackedNvfp4Weights detects packed tensors in layer maps" {
    var packed_tensor = std.mem.zeroes(tensor.Tensor);
    packed_tensor.dtype = .u8;
    packed_tensor.nvfp4 = .{
        .block_scales_data = &.{},
        .block_scales_len = 0,
        .rows = 1,
        .cols = 1,
        .packed_cols = 1,
        .scale_cols = 1,
        .group_size = 16,
        .weight_global_scale = 1.0,
    };

    var layer_map: models.runtime_blocks.WeightMap = .{};
    defer layer_map.deinit(std.testing.allocator);
    try layer_map.put(std.testing.allocator, "weight", &packed_tensor);

    const layers = [_]models.runtime_blocks.LayerWeights{.{
        .block_type = .attention_mlp,
        .weight_map = layer_map,
        .map_context = .{},
    }};

    var embeddings = std.mem.zeroes(tensor.Tensor);
    embeddings.dtype = .f32;

    var loaded = std.mem.zeroes(LoadedModel);
    loaded.token_embeddings = embeddings;
    loaded.blocks = layers[0..];

    try std.testing.expect(loadedModelHasPackedNvfp4Weights(&loaded));
}

test "parseCudaTopologyMode parses supported modes" {
    try std.testing.expectEqual(CudaTopologyMode.single, try parseCudaTopologyMode("single"));
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, try parseCudaTopologyMode("pipeline2"));
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, try parseCudaTopologyMode("pipeline_2way"));
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, try parseCudaTopologyMode("gpu_gpu"));
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, try parseCudaTopologyMode("gpu-gpu"));
    try std.testing.expectEqual(CudaTopologyMode.pipeline2, try parseCudaTopologyMode("gpu+gpu"));
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, try parseCudaTopologyMode("cpu_gpu"));
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, try parseCudaTopologyMode("cpu+gpu"));
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, try parseCudaTopologyMode("cpu_gpu_gpu"));
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, try parseCudaTopologyMode("cpu+gpu+gpu"));
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, try parseCudaTopologyMode("pipeline_cpu_gpu_gpu"));
}

test "parseCudaTopologyMode rejects unsupported modes" {
    try std.testing.expectError(error.InvalidTopologyConfig, parseCudaTopologyMode(""));
    try std.testing.expectError(error.InvalidTopologyConfig, parseCudaTopologyMode("mesh"));
}

test "parseTwoDeviceOrdinals parses gpu pairs" {
    try std.testing.expectEqualDeep(@as([2]usize, .{ 0, 1 }), try parseTwoDeviceOrdinals("0,1"));
    try std.testing.expectEqualDeep(@as([2]usize, .{ 3, 9 }), try parseTwoDeviceOrdinals(" 3 , 9 "));
    try std.testing.expectError(error.InvalidTopologyConfig, parseTwoDeviceOrdinals("0"));
    try std.testing.expectError(error.InvalidTopologyConfig, parseTwoDeviceOrdinals("a,1"));
    try std.testing.expectError(error.InvalidTopologyConfig, parseTwoDeviceOrdinals("0,1,2"));
}

test "resolveCudaTopology honors explicit override" {
    const topology = try resolveCudaTopology(std.testing.allocator, .{
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
    const result = try estimateModelGpuBytes(p);
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
    const full = try estimatePerGpuBytes(p, 0, 16, true, true);
    const mid = try estimatePerGpuBytes(p, 0, 16, false, false);
    // Intermediate stage (no embed/proj) should be significantly cheaper.
    try std.testing.expect(mid < full);
}

test "estimatePerGpuBytes rejects manifest range mismatch" {
    var manifest = models.manifest.ModelManifest{
        .arena = std.heap.ArenaAllocator.init(std.testing.allocator),
        .architecture_id = "test",
        .layer_count = 1,
        .entries = &.{},
        .total_checkpoint_bytes = 0,
        .role_bytes = [_]usize{0} ** models.manifest.role_count,
    };
    defer manifest.deinit();

    const p = ModelSizeParams{
        .file_size = 1024,
        .vocab_size = 1,
        .d_model = 1,
        .n_kv_heads = 0,
        .head_dim = 0,
        .max_seq_len = 0,
        .n_layers = 2,
        .manifest = &manifest,
    };

    try std.testing.expectError(
        error.InvalidTopologyConfig,
        estimatePerGpuBytes(p, 0, 2, false, false),
    );
}

test "autoSelectTopology S3: 1 GPU, model fits → single" {
    const p = testMinimalParams(2 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    const result = try autoSelectTopology(p, 32, &infos);
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
    try std.testing.expectEqual(@as(usize, 0), result.stage_device_ordinals[0]);
}

test "autoSelectTopology S4: 1 GPU, model too large → cpu_gpu" {
    // 4 GB free, ~10 GB model, 32 layers.
    const p = testMinimalParams(10 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(1, .{.{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 }});
    const result = try autoSelectTopology(p, 32, &infos);
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
    const result = try autoSelectTopology(p, 32, &infos);
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
    const result = try autoSelectTopology(p, 32, &infos);
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
    const result = try autoSelectTopology(p, 32, &infos);
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
    const result = try autoSelectTopology(p, 32, &infos);
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
    const result = try autoSelectTopology(p, 32, &infos);
    // Falls back to single (caller handles OOM).
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
}

test "autoSelectTopology with 0 GPUs returns single" {
    const p = testMinimalParams(1024, 32);
    const infos: []const GpuMemoryInfo = &.{};
    const result = try autoSelectTopology(p, 32, infos);
    try std.testing.expectEqual(CudaTopologyMode.single, result.mode);
}

test "autoDetectTopologyForModel returns CudaNotEnabled when CUDA is disabled" {
    if (has_cuda) return;
    const loaded: *const LoadedModel = undefined;
    try std.testing.expectError(
        error.CudaNotEnabled,
        autoDetectTopologyForModel(std.testing.allocator, loaded),
    );
}

test "autoSelectTopology with 1 layer returns single" {
    const p = testMinimalParams(4096, 1);
    const infos = testGpuInfos(1, .{.{ 1024, 2048 }});
    const result = try autoSelectTopology(p, 1, &infos);
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
    const result = try autoSelectTopology(p, 32, &infos);
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
    try std.testing.expect((try topologyFromCpuLayers(0, 32, &infos, null)) == null);
}

test "topologyFromCpuLayers: cpu_layers >= total_layers returns null" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    try std.testing.expect((try topologyFromCpuLayers(32, 32, &infos, null)) == null);
    try std.testing.expect((try topologyFromCpuLayers(33, 32, &infos, null)) == null);
}

test "topologyFromCpuLayers: 1 GPU → cpu_gpu" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    const result = (try topologyFromCpuLayers(8, 32, &infos, null)).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 8), result.split_layer);
}

test "topologyFromCpuLayers: 2 GPUs without model params prefer one GPU" {
    const infos = testGpuInfos(2, .{
        .{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
        .{ 16 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
    });
    const result = (try topologyFromCpuLayers(8, 32, &infos, null)).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 8), result.split_layer);
    try std.testing.expectEqual(@as(usize, 1), result.stage_device_ordinals[0]);
    try std.testing.expectEqual(@as(usize, 1), result.stage_device_ordinals[1]);
}

test "topologyFromCpuLayers: 2 GPUs keep GPU suffix on one GPU when it fits" {
    const gb = 1024 * 1024 * 1024;
    const infos = testGpuInfos(2, .{
        .{ 8 * gb, 16 * gb },
        .{ 8 * gb, 16 * gb },
    });
    const p = testMinimalParams(3 * gb, 32);
    const result = (try topologyFromCpuLayers(1, 32, &infos, p)).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 1), result.split_layer);
    try std.testing.expectEqual(@as([2]usize, .{ 0, 0 }), result.stage_device_ordinals);
}

test "topologyFromCpuLayers: model-aware split uses 2 GPUs when suffix exceeds one GPU" {
    const gb = 1024 * 1024 * 1024;
    const infos = testGpuInfos(2, .{
        .{ 4 * gb, 4 * gb },
        .{ 4 * gb, 4 * gb },
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
    const result = (try topologyFromCpuLayers(36, 60, &infos, p)).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu_gpu, result.mode);
    // GPU0 (no proj) should get more layers than GPU1 (has proj).
    const gpu0_layers = result.split_layer_stage2.? - 36;
    const gpu1_layers = 60 - result.split_layer_stage2.?;
    try std.testing.expect(gpu0_layers > gpu1_layers);
}

test "topologyFromCpuLayers: 2 GPUs split overflowing suffix by memory" {
    const gb = 1024 * 1024 * 1024;
    const infos = testGpuInfos(2, .{
        .{ 12 * gb, 16 * gb }, // GPU0: 12 GB
        .{ 4 * gb, 8 * gb }, // GPU1: 4 GB
    });
    const p = testMinimalParams(24 * gb, 32);
    const result = (try topologyFromCpuLayers(12, 32, &infos, p)).?;
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
    const result = (try topologyFromCpuLayers(31, 32, &infos, null)).?;
    try std.testing.expectEqual(CudaTopologyMode.cpu_gpu, result.mode);
    try std.testing.expectEqual(@as(?usize, 31), result.split_layer);
}

test "topologyFromCpuLayers: no GPUs returns null" {
    const infos = [_]GpuMemoryInfo{};
    try std.testing.expect((try topologyFromCpuLayers(8, 32, &infos, null)) == null);
}

test "topologyFromCpuLayers: total_layers < 2 returns null" {
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    try std.testing.expect((try topologyFromCpuLayers(1, 1, &infos, null)) == null);
}

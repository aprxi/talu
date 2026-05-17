//! Local pipeline stage plan selection and validation policy.
//!
//! This module owns local stage parsing, validation, memory estimation,
//! and model-shape policy used before local pipeline runtime construction.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const compute = @import("compute_pkg");
const models = @import("models_pkg");
const log = @import("log_pkg");
const tensor = @import("compute_pkg").tensor;

const LoadedModel = models.LoadedModel;
const has_cuda = build_options.enable_cuda and (builtin.os.tag == .linux or builtin.os.tag == .windows);

pub const LocalStageBackendKind = enum {
    cpu,
    cuda,
    metal,
};

pub const LocalStageSpec = struct {
    backend_kind: LocalStageBackendKind,
    device_ordinal: ?usize = null,
    layer_start: usize,
    layer_end: usize,
    owns_embedding: bool = false,
    owns_projection: bool = false,
};

pub const max_local_stage_count: usize = 16;

pub const LocalStageSpecList = struct {
    allocator: std.mem.Allocator,
    stages: []LocalStageSpec,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.stages);
        self.* = undefined;
    }
};

pub const LocalStagePlan = struct {
    stage_count: usize = 0,
    stages: [max_local_stage_count]LocalStageSpec = undefined,

    pub fn stagesSlice(self: *const @This()) []const LocalStageSpec {
        return self.stages[0..self.stage_count];
    }

    pub fn primaryDeviceOrdinal(self: *const @This()) usize {
        for (self.stagesSlice()) |stage| {
            if (stage.backend_kind == .cuda) return stage.device_ordinal orelse 0;
        }
        return 0;
    }

    pub fn cudaStageCount(self: *const @This()) usize {
        var count: usize = 0;
        for (self.stagesSlice()) |stage| {
            if (stage.backend_kind == .cuda) count += 1;
        }
        return count;
    }

    pub fn isSingleCudaStage(self: *const @This()) bool {
        return self.stage_count == 1 and self.stages[0].backend_kind == .cuda;
    }
};

pub const LocalStageValidationError = error{
    InvalidTopologyConfig,
    LocalStageTooManyStages,
    LocalStageInsufficientLayers,
    LocalStageUnsupportedBackend,
    LocalStageDeviceOrdinalOutOfRange,
};

pub fn hostBackendKind(kind: LocalStageBackendKind) @import("host_capability.zig").HostBackendKind {
    return switch (kind) {
        .cpu => .cpu,
        .cuda => .cuda,
        .metal => .metal,
    };
}

pub fn validateLocalStagePlan(
    plan: LocalStagePlan,
    total_layers: usize,
    device_count: usize,
) LocalStageValidationError!void {
    if (total_layers == 0 or plan.stage_count == 0) return error.InvalidTopologyConfig;
    if (plan.stage_count > max_local_stage_count) return error.LocalStageTooManyStages;
    if (plan.stage_count > 1 and total_layers < plan.stage_count) return error.LocalStageInsufficientLayers;
    var expected_start: usize = 0;
    for (plan.stagesSlice(), 0..) |stage, stage_index| {
        if (stage.layer_start != expected_start) return error.InvalidTopologyConfig;
        if (stage.layer_end <= stage.layer_start or stage.layer_end > total_layers) return error.InvalidTopologyConfig;
        if (stage.owns_embedding != (stage_index == 0)) return error.InvalidTopologyConfig;
        if (stage.owns_projection != (stage_index + 1 == plan.stage_count)) return error.InvalidTopologyConfig;
        switch (stage.backend_kind) {
            .cpu => {
                if (stage.device_ordinal != null) return error.InvalidTopologyConfig;
            },
            .cuda => {
                const ordinal = stage.device_ordinal orelse return error.InvalidTopologyConfig;
                if (ordinal >= device_count) return error.LocalStageDeviceOrdinalOutOfRange;
            },
            .metal => {
                return error.LocalStageUnsupportedBackend;
            },
        }
        expected_start = stage.layer_end;
    }
    if (expected_start != total_layers) return error.InvalidTopologyConfig;
}

fn getOptionalEnvVarOwned(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    return std.process.getEnvVarOwned(allocator, name) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => null,
        else => err,
    };
}

fn parseLocalStageBackendKind(raw: []const u8) !LocalStageBackendKind {
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (std.ascii.eqlIgnoreCase(token, "cpu")) return .cpu;
    if (std.ascii.eqlIgnoreCase(token, "cuda")) return .cuda;
    if (std.ascii.eqlIgnoreCase(token, "metal")) return .metal;
    return error.InvalidTopologyConfig;
}

fn parseLayerBound(raw: []const u8, total_layers: usize) !usize {
    const token = std.mem.trim(u8, raw, " \t\r\n");
    if (std.ascii.eqlIgnoreCase(token, "end")) return total_layers;
    return std.fmt.parseUnsigned(usize, token, 10) catch return error.InvalidTopologyConfig;
}

fn parseLocalStageEntry(raw: []const u8, total_layers: usize) !LocalStageSpec {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidTopologyConfig;
    const colon = std.mem.indexOfScalar(u8, trimmed, ':') orelse return error.InvalidTopologyConfig;
    const backend_part = trimmed[0..colon];
    const range_part = trimmed[colon + 1 ..];
    const range_sep = std.mem.indexOf(u8, range_part, "..") orelse return error.InvalidTopologyConfig;
    if (std.mem.indexOf(u8, range_part[range_sep + 2 ..], "..") != null) return error.InvalidTopologyConfig;

    const at = std.mem.indexOfScalar(u8, backend_part, '@');
    const kind_raw = if (at) |index| backend_part[0..index] else backend_part;
    const kind = try parseLocalStageBackendKind(kind_raw);
    const device_ordinal: ?usize = if (at) |index| blk: {
        const device_raw = std.mem.trim(u8, backend_part[index + 1 ..], " \t\r\n");
        if (device_raw.len == 0) return error.InvalidTopologyConfig;
        break :blk std.fmt.parseUnsigned(usize, device_raw, 10) catch return error.InvalidTopologyConfig;
    } else null;
    if (kind == .cpu and device_ordinal != null) return error.InvalidTopologyConfig;
    if ((kind == .cuda or kind == .metal) and device_ordinal == null) return error.InvalidTopologyConfig;

    const start = try parseLayerBound(range_part[0..range_sep], total_layers);
    const end = try parseLayerBound(range_part[range_sep + 2 ..], total_layers);
    return .{
        .backend_kind = kind,
        .device_ordinal = device_ordinal,
        .layer_start = start,
        .layer_end = end,
    };
}

fn localStageSpecWithOwnership(stage: LocalStageSpec, stage_index: usize, stage_count: usize) LocalStageSpec {
    var result = stage;
    result.owns_embedding = stage_index == 0;
    result.owns_projection = stage_index + 1 == stage_count;
    return result;
}

pub fn parseLocalStageSpecs(
    allocator: std.mem.Allocator,
    raw: []const u8,
    total_layers: usize,
) !LocalStageSpecList {
    const trimmed = std.mem.trim(u8, raw, " \t\r\n");
    if (trimmed.len == 0 or total_layers == 0) return error.InvalidTopologyConfig;
    var count: usize = 1;
    for (trimmed) |byte| {
        if (byte == ',') count += 1;
    }
    const stages = try allocator.alloc(LocalStageSpec, count);
    errdefer allocator.free(stages);

    var iter = std.mem.splitScalar(u8, trimmed, ',');
    var index: usize = 0;
    var expected_start: usize = 0;
    while (iter.next()) |entry| {
        if (index >= stages.len) return error.InvalidTopologyConfig;
        stages[index] = try parseLocalStageEntry(entry, total_layers);
        if (stages[index].layer_start != expected_start) return error.InvalidTopologyConfig;
        if (stages[index].layer_end <= stages[index].layer_start or stages[index].layer_end > total_layers) {
            return error.InvalidTopologyConfig;
        }
        expected_start = stages[index].layer_end;
        index += 1;
    }
    if (index != stages.len or expected_start != total_layers) return error.InvalidTopologyConfig;
    for (stages, 0..) |*stage, stage_index| {
        stage.* = localStageSpecWithOwnership(stage.*, stage_index, stages.len);
    }
    return .{ .allocator = allocator, .stages = stages };
}

pub fn localStagePlanFromSpecs(stages: []const LocalStageSpec) !LocalStagePlan {
    if (stages.len == 0 or stages.len > max_local_stage_count) return error.InvalidTopologyConfig;
    var plan = LocalStagePlan{ .stage_count = stages.len };
    for (stages, 0..) |stage, index| {
        plan.stages[index] = localStageSpecWithOwnership(stage, index, stages.len);
    }
    return plan;
}

pub fn defaultCudaLocalStagePlan(total_layers: usize, device_ordinal: usize) !LocalStagePlan {
    if (total_layers == 0) return error.InvalidTopologyConfig;
    return localStagePlanFromSpecs(&.{.{
        .backend_kind = .cuda,
        .device_ordinal = device_ordinal,
        .layer_start = 0,
        .layer_end = total_layers,
    }});
}

pub fn cpuPrefixCudaLocalStagePlan(total_layers: usize, cpu_layers: usize, device_ordinal: usize) !LocalStagePlan {
    if (total_layers == 0) return error.InvalidTopologyConfig;
    if (cpu_layers == 0) return defaultCudaLocalStagePlan(total_layers, device_ordinal);
    if (cpu_layers >= total_layers) return error.InvalidTopologyConfig;
    return localStagePlanFromSpecs(&.{
        .{
            .backend_kind = .cpu,
            .layer_start = 0,
            .layer_end = cpu_layers,
        },
        .{
            .backend_kind = .cuda,
            .device_ordinal = device_ordinal,
            .layer_start = cpu_layers,
            .layer_end = total_layers,
        },
    });
}

pub fn resolveLocalStagePlan(
    explicit_stages: ?[]const LocalStageSpec,
    total_layers: usize,
    default_device_ordinal: usize,
) !LocalStagePlan {
    if (explicit_stages) |stages| {
        return localStagePlanFromSpecs(stages);
    }
    return defaultCudaLocalStagePlan(total_layers, default_device_ordinal);
}

pub fn rootCudaStageIdForLocalPlan(stages: []const LocalStageSpec) !usize {
    if (stages.len == 0) return error.InvalidTopologyConfig;
    if (stages[0].backend_kind == .cuda) return 0;
    var index = stages.len;
    while (index > 0) {
        index -= 1;
        if (stages[index].backend_kind == .cuda) return index;
    }
    return error.InvalidTopologyConfig;
}

// ---------------------------------------------------------------------------
// Auto local stage-plan selection
//
// When no explicit TALU_LOCAL_STAGES list is set, automatically choose the best
// stage plan based on available GPU memory and estimated model requirements.
//
// Scenarios (see also AGENTS.md):
//
//   S0  BACKEND=cpu           -> CPU backend, no GPU. Handled before we get here.
//   S1  TALU_LOCAL_STAGES     -> User override. Skip auto-detection.
//   S2  0 GPUs visible        -> Error (caller handles).
//   S3  1 GPU, model fits     -> one CUDA stage.
//   S4  1 GPU, model !fits    -> CPU prefix plus CUDA suffix.
//   S5  2+ GPUs, fits on 1    -> one CUDA stage on best GPU.
//   S6  2+ GPUs, fits on 2    -> two CUDA stages split by free memory.
//   S7  2+ GPUs, !fits on 2   -> CPU prefix plus two CUDA stages.
//   S8  Too large for all     -> Error (caller handles).
//   S9  3+ GPUs               -> Pick best 2 GPUs, apply S5-S7.
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

/// Select a local stage plan automatically based on estimated model size and GPU memory.
///
/// Uses per-GPU estimation that separates fixed costs (embedding, projection,
/// activation buffers) from per-layer costs (weights, KV cache).
/// Intermediate local stages skip embedding/projection at init time.
///
/// Pure logic — no I/O, no side effects. Fully testable with synthetic values.
/// Returns one CUDA stage when in doubt.
///
/// gpu_infos must be sorted by ordinal (as returned by probeGpuTotalMemory).
fn autoSelectLocalStagePlan(
    p: ModelSizeParams,
    total_layers: usize,
    gpu_infos: []const GpuMemoryInfo,
) !LocalStagePlan {
    if (gpu_infos.len == 0) return error.InvalidTopologyConfig;
    if (total_layers < 2) return defaultCudaLocalStagePlan(total_layers, gpu_infos[0].ordinal);

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
        return defaultCudaLocalStagePlan(total_layers, gpu_infos[best0].ordinal);
    }

    // From here: model doesn't fit on single GPU.
    if (gpu_infos.len == 1) {
        // S4: offload lower layers to CPU.
        // GPU stage: no embedding (CPU does it), has projection (last stage).
        const gpu_layers = std.math.clamp(
            try maxSuffixLayersForBudget(p, best0_budget, total_layers, false, true),
            1,
            total_layers - 1,
        );
        const split = total_layers - gpu_layers;
        return localStagePlanFromSpecs(&.{
            .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = split },
            .{ .backend_kind = .cuda, .device_ordinal = gpu_infos[0].ordinal, .layer_start = split, .layer_end = total_layers },
        });
    }

    // 2+ GPUs. Ensure ordinals are ordered so stage ids remain stable.
    const ord0 = @min(gpu_infos[best0].ordinal, gpu_infos[best1].ordinal);
    const ord1 = @max(gpu_infos[best0].ordinal, gpu_infos[best1].ordinal);
    const free0 = if (gpu_infos[best0].ordinal == ord0) gpu_infos[best0].free else gpu_infos[best1].free;
    const free1 = if (gpu_infos[best0].ordinal == ord0) gpu_infos[best1].free else gpu_infos[best0].free;
    const budget0 = if (free0 > AUTO_TOPO_OVERHEAD_BYTES) free0 - AUTO_TOPO_OVERHEAD_BYTES else 0;
    const budget1 = if (free1 > AUTO_TOPO_OVERHEAD_BYTES) free1 - AUTO_TOPO_OVERHEAD_BYTES else 0;

    // S6: Model fits across 2 GPUs.
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
                    return localStagePlanFromSpecs(&.{
                        .{ .backend_kind = .cuda, .device_ordinal = ord0, .layer_start = 0, .layer_end = split },
                        .{ .backend_kind = .cuda, .device_ordinal = ord1, .layer_start = split, .layer_end = total_layers },
                    });
                }
            }
        }
    }

    // S7: Doesn't fit on 2 GPUs. CPU takes overflow.
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
        // S8: Not enough GPU memory even with a CPU prefix. Return single and let
        // the caller handle the OOM or error.
        return defaultCudaLocalStagePlan(total_layers, gpu_infos[best0].ordinal);
    }
    return localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = best_cpu_layers },
        .{ .backend_kind = .cuda, .device_ordinal = ord0, .layer_start = best_cpu_layers, .layer_end = best_split_stage2 },
        .{ .backend_kind = .cuda, .device_ordinal = ord1, .layer_start = best_split_stage2, .layer_end = total_layers },
    });
}

/// Run full auto-detection: probe GPU memory, estimate model requirements, select a stage plan.
pub fn autoDetectLocalStagePlanForModel(
    allocator: std.mem.Allocator,
    loaded: *const LoadedModel,
) !LocalStagePlan {
    const device_count = if (has_cuda)
        compute.cuda.Device.deviceCount() catch |err| {
            log.err("inference", "CUDA local stage auto-detection failed to query device count", .{
                .err = @errorName(err),
            }, @src());
            return error.CudaUnavailable;
        }
    else
        return error.CudaNotEnabled;

    if (device_count == 0) return error.CudaUnavailable;

    const gpu_infos = probeGpuTotalMemory(allocator, device_count) catch |err| {
        log.err("inference", "CUDA local stage auto-detection failed to probe GPU memory", .{
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

    const result = try autoSelectLocalStagePlan(params, loaded.blocks.len, gpu_infos);

    // Log the auto-detection result.
    const estimated_mib = estimated / (1024 * 1024);
    log.info("inference", "Auto local stage plan", .{
        .estimated_mib = estimated_mib,
        .stage_count = result.stage_count,
        .cuda_stage_count = result.cudaStageCount(),
        .primary_cuda_device = result.primaryDeviceOrdinal(),
    });

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

test "parseLocalStageSpecs parses ordered backend layer ranges" {
    var specs = try parseLocalStageSpecs(std.testing.allocator, "cpu:0..2,cuda@0:2..7,metal@0:7..end", 9);
    defer specs.deinit();

    try std.testing.expectEqual(@as(usize, 3), specs.stages.len);
    try std.testing.expectEqual(LocalStageBackendKind.cpu, specs.stages[0].backend_kind);
    try std.testing.expectEqual(@as(?usize, null), specs.stages[0].device_ordinal);
    try std.testing.expectEqual(@as(usize, 0), specs.stages[0].layer_start);
    try std.testing.expectEqual(@as(usize, 2), specs.stages[0].layer_end);
    try std.testing.expect(specs.stages[0].owns_embedding);
    try std.testing.expect(!specs.stages[0].owns_projection);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, specs.stages[1].backend_kind);
    try std.testing.expectEqual(@as(?usize, 0), specs.stages[1].device_ordinal);
    try std.testing.expectEqual(@as(usize, 2), specs.stages[1].layer_start);
    try std.testing.expectEqual(@as(usize, 7), specs.stages[1].layer_end);
    try std.testing.expect(!specs.stages[1].owns_embedding);
    try std.testing.expect(!specs.stages[1].owns_projection);
    try std.testing.expectEqual(LocalStageBackendKind.metal, specs.stages[2].backend_kind);
    try std.testing.expectEqual(@as(?usize, 0), specs.stages[2].device_ordinal);
    try std.testing.expectEqual(@as(usize, 7), specs.stages[2].layer_start);
    try std.testing.expectEqual(@as(usize, 9), specs.stages[2].layer_end);
    try std.testing.expect(!specs.stages[2].owns_embedding);
    try std.testing.expect(specs.stages[2].owns_projection);
}

test "parseLocalStageSpecs rejects non-contiguous or ambiguous specs" {
    try std.testing.expectError(error.InvalidTopologyConfig, parseLocalStageSpecs(std.testing.allocator, "", 4));
    try std.testing.expectError(error.InvalidTopologyConfig, parseLocalStageSpecs(std.testing.allocator, "cpu:0..2,cuda@0:3..end", 4));
    try std.testing.expectError(error.InvalidTopologyConfig, parseLocalStageSpecs(std.testing.allocator, "cpu@0:0..1,cuda@0:1..end", 4));
    try std.testing.expectError(error.InvalidTopologyConfig, parseLocalStageSpecs(std.testing.allocator, "cpu:0..1,cuda:1..end", 4));
    try std.testing.expectError(error.InvalidTopologyConfig, parseLocalStageSpecs(std.testing.allocator, "cpu:0..1,metal:1..end", 4));
}

test "localStagePlanFromSpecs preserves ordered generic stages" {
    const plan = try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 2 },
        .{ .backend_kind = .cuda, .device_ordinal = 4, .layer_start = 2, .layer_end = 7 },
        .{ .backend_kind = .cuda, .device_ordinal = 5, .layer_start = 7, .layer_end = 11 },
    });

    try std.testing.expectEqual(@as(usize, 3), plan.stage_count);
    try std.testing.expectEqual(LocalStageBackendKind.cpu, plan.stages[0].backend_kind);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, plan.stages[1].backend_kind);
    try std.testing.expectEqual(@as(?usize, 4), plan.stages[1].device_ordinal);
    try std.testing.expectEqual(@as(usize, 7), plan.stages[1].layer_end);
    try std.testing.expectEqual(@as(?usize, 5), plan.stages[2].device_ordinal);
    try std.testing.expect(plan.stages[0].owns_embedding);
    try std.testing.expect(!plan.stages[0].owns_projection);
    try std.testing.expect(!plan.stages[1].owns_embedding);
    try std.testing.expect(!plan.stages[1].owns_projection);
    try std.testing.expect(!plan.stages[2].owns_embedding);
    try std.testing.expect(plan.stages[2].owns_projection);
}

test "validateLocalStagePlan accepts single and mixed CPU/CUDA ordered plans" {
    try validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 0, .layer_end = 1 },
    }), 1, 1);

    try validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 2 },
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 2, .layer_end = 5 },
    }), 5, 2);
}

test "validateLocalStagePlan rejects invalid generic plans" {
    try std.testing.expectError(error.LocalStageInsufficientLayers, validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 1, .layer_end = 2 },
        .{ .backend_kind = .cuda, .device_ordinal = 1, .layer_start = 2, .layer_end = 3 },
    }), 2, 2));

    try std.testing.expectError(error.InvalidTopologyConfig, validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 0, .layer_end = 2 },
        .{ .backend_kind = .cuda, .device_ordinal = 1, .layer_start = 3, .layer_end = 4 },
    }), 4, 2));

    try std.testing.expectError(error.InvalidTopologyConfig, validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cpu, .device_ordinal = 0, .layer_start = 0, .layer_end = 1 },
        .{ .backend_kind = .cuda, .device_ordinal = 0, .layer_start = 1, .layer_end = 2 },
    }), 2, 1));

    try std.testing.expectError(error.LocalStageDeviceOrdinalOutOfRange, validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .cuda, .device_ordinal = 2, .layer_start = 0, .layer_end = 4 },
    }), 4, 2));

    try std.testing.expectError(error.LocalStageUnsupportedBackend, validateLocalStagePlan(try localStagePlanFromSpecs(&.{
        .{ .backend_kind = .metal, .device_ordinal = 0, .layer_start = 0, .layer_end = 2 },
    }), 2, 1));
}

test "resolveLocalStagePlan honors explicit generic override" {
    const explicit = [_]LocalStageSpec{
        .{ .backend_kind = .cpu, .layer_start = 0, .layer_end = 3 },
        .{ .backend_kind = .cuda, .device_ordinal = 2, .layer_start = 3, .layer_end = 8 },
    };
    const plan = try resolveLocalStagePlan(&explicit, 8, 0);

    try std.testing.expectEqual(@as(usize, 2), plan.stage_count);
    try std.testing.expectEqual(LocalStageBackendKind.cpu, plan.stages[0].backend_kind);
    try std.testing.expectEqual(@as(?usize, 2), plan.stages[1].device_ordinal);
}

test "cpuPrefixCudaLocalStagePlan builds explicit CPU prefix plan" {
    const plan = try cpuPrefixCudaLocalStagePlan(32, 5, 1);

    try std.testing.expectEqual(@as(usize, 2), plan.stage_count);
    try std.testing.expectEqual(LocalStageBackendKind.cpu, plan.stages[0].backend_kind);
    try std.testing.expectEqual(@as(usize, 0), plan.stages[0].layer_start);
    try std.testing.expectEqual(@as(usize, 5), plan.stages[0].layer_end);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, plan.stages[1].backend_kind);
    try std.testing.expectEqual(@as(?usize, 1), plan.stages[1].device_ordinal);
    try std.testing.expectEqual(@as(usize, 5), plan.stages[1].layer_start);
    try std.testing.expectEqual(@as(usize, 32), plan.stages[1].layer_end);
}

test "cpuPrefixCudaLocalStagePlan accepts zero and rejects all-CPU CUDA plan" {
    const single = try cpuPrefixCudaLocalStagePlan(32, 0, 0);
    try std.testing.expect(single.isSingleCudaStage());
    try std.testing.expectEqual(@as(?usize, 0), single.stages[0].device_ordinal);

    try std.testing.expectError(error.InvalidTopologyConfig, cpuPrefixCudaLocalStagePlan(32, 32, 0));
    try std.testing.expectError(error.InvalidTopologyConfig, cpuPrefixCudaLocalStagePlan(32, 33, 0));
}

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

test "autoSelectLocalStagePlan uses one CUDA stage when the model fits one GPU" {
    const p = testMinimalParams(2 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(1, .{.{ 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 }});
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expect(result.isSingleCudaStage());
    try std.testing.expectEqual(@as(?usize, 0), result.stages[0].device_ordinal);
}

test "autoSelectLocalStagePlan adds a CPU prefix when one GPU is too small" {
    // 4 GB free, ~10 GB model, 32 layers.
    const p = testMinimalParams(10 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(1, .{.{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 }});
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expectEqual(@as(usize, 2), result.stage_count);
    try std.testing.expectEqual(LocalStageBackendKind.cpu, result.stages[0].backend_kind);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, result.stages[1].backend_kind);
    try std.testing.expect(result.stages[0].layer_end > 0);
    try std.testing.expect(result.stages[0].layer_end < 32);
}

test "autoSelectLocalStagePlan chooses the best single GPU" {
    // GPU0: 4 GB free, GPU1: 12 GB free. Model needs ~5 GB.
    const p = testMinimalParams(5 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 4 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
        .{ 12 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024 },
    });
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expect(result.isSingleCudaStage());
    try std.testing.expectEqual(@as(?usize, 1), result.stages[0].device_ordinal);
}

test "autoSelectLocalStagePlan splits across two GPUs when needed" {
    // GPU0: 4 GB free, GPU1: 4 GB free. Model needs ~6 GB.
    const p = testMinimalParams(6 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expectEqual(@as(usize, 2), result.stage_count);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, result.stages[0].backend_kind);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, result.stages[1].backend_kind);
    try std.testing.expect(result.stages[0].layer_end >= 12 and result.stages[0].layer_end <= 20);
    try std.testing.expectEqual(@as(?usize, 0), result.stages[0].device_ordinal);
    try std.testing.expectEqual(@as(?usize, 1), result.stages[1].device_ordinal);
}

test "autoSelectLocalStagePlan split tracks unequal GPU memory" {
    // GPU0: 6 GB free, GPU1: 2 GB free. Model needs ~7 GB.
    const p = testMinimalParams(7 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expectEqual(@as(usize, 2), result.stage_count);
    try std.testing.expect(result.stages[0].layer_end >= 18);
}

test "autoSelectLocalStagePlan uses CPU plus two GPUs when two GPUs are not enough" {
    // GPU0: 4 GB, GPU1: 4 GB. Model needs ~12 GB.
    const p = testMinimalParams(12 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expectEqual(@as(usize, 3), result.stage_count);
    try std.testing.expectEqual(LocalStageBackendKind.cpu, result.stages[0].backend_kind);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, result.stages[1].backend_kind);
    try std.testing.expectEqual(LocalStageBackendKind.cuda, result.stages[2].backend_kind);
    try std.testing.expect(result.stages[0].layer_end > 0);
    try std.testing.expect(result.stages[1].layer_end > result.stages[0].layer_end);
    try std.testing.expect(result.stages[1].layer_end < 32);
}

test "autoSelectLocalStagePlan falls back to one CUDA stage when no split fits" {
    // GPU0: 1 GB, GPU1: 1 GB. Model needs ~100 GB.
    const p = testMinimalParams(100 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(2, .{
        .{ 1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024 },
        .{ 1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024 },
    });
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expect(result.isSingleCudaStage());
}

test "autoSelectLocalStagePlan with no GPUs rejects the request" {
    const p = testMinimalParams(1024, 32);
    const infos: []const GpuMemoryInfo = &.{};
    try std.testing.expectError(error.InvalidTopologyConfig, autoSelectLocalStagePlan(p, 32, infos));
}

test "autoDetectLocalStagePlanForModel returns CudaNotEnabled when CUDA is disabled" {
    if (has_cuda) return;
    const loaded: *const LoadedModel = undefined;
    try std.testing.expectError(
        error.CudaNotEnabled,
        autoDetectLocalStagePlanForModel(std.testing.allocator, loaded),
    );
}

test "autoSelectLocalStagePlan with one layer returns one CUDA stage" {
    const p = testMinimalParams(4096, 1);
    const infos = testGpuInfos(1, .{.{ 1024, 2048 }});
    const result = try autoSelectLocalStagePlan(p, 1, &infos);
    try std.testing.expect(result.isSingleCudaStage());
}

test "autoSelectLocalStagePlan picks the two best GPUs" {
    // GPU0: 2 GB, GPU1: 6 GB, GPU2: 6 GB. Model needs ~8 GB.
    const p = testMinimalParams(8 * 1024 * 1024 * 1024, 32);
    const infos = testGpuInfos(3, .{
        .{ 2 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
        .{ 6 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 },
    });
    const result = try autoSelectLocalStagePlan(p, 32, &infos);
    try std.testing.expectEqual(@as(usize, 2), result.stage_count);
    try std.testing.expectEqual(@as(?usize, 1), result.stages[0].device_ordinal);
    try std.testing.expectEqual(@as(?usize, 2), result.stages[1].device_ordinal);
}

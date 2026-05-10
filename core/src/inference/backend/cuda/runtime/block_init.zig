//! CUDA block runtime construction.
//!
//! This module owns model-weight inspection, device uploads, and layer-range
//! assembly for `BlockRuntime`. Runtime type declarations stay in `blocks.zig`.

const std = @import("std");
const models = @import("models_pkg");
const op_types = models.op_types;
const plan_compiler = models.plan.compiler;
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const log = @import("log_pkg");
const cpu_kernels = @import("../../cpu/kernels/root.zig");

const LoadedModel = models.LoadedModel;
const Tensor = tensor.Tensor;
const GateUpLayout = models.runtime_blocks.GateUpLayout;

const config = @import("config.zig");
const weights = @import("weights.zig");
const block_types = @import("blocks.zig");

const KvCacheDtype = config.KvCacheDtype;
const resolveSharedKvSourceLayer = config.resolveSharedKvSourceLayer;
const DeviceTensor = weights.DeviceTensor;
const LinearWeight = weights.LinearWeight;

const BlockRuntime = block_types.BlockRuntime;
const BlockRuntimeLayer = block_types.BlockRuntimeLayer;
const LayerAttentionRuntime = block_types.LayerAttentionRuntime;
const MoEWeightRefs = block_types.MoEWeightRefs;
const ReplicatedKvSource = block_types.ReplicatedKvSource;
const MirrorKvBuffers = block_types.MirrorKvBuffers;
const GatedDeltaSsmStateFormat = block_types.GatedDeltaSsmStateFormat;

const tensorProjectionOutputDim = block_types.tensorProjectionOutputDim;
const buildCudaLayerProgramRegisterSlotMap = block_types.buildCudaLayerProgramRegisterSlotMap;
const buildCudaLayerProgramSlotWidthHints = block_types.buildCudaLayerProgramSlotWidthHints;
const validateCompiledLayerPlanForCuda = block_types.validateCompiledLayerPlanForCuda;

const engine_weights = @import("../weights/root.zig");
const uploadLinearWeightWithContext = engine_weights.uploadLinearWeightWithContext;
const uploadTensor = engine_weights.uploadTensor;
const uploadFusedQkvWeights = engine_weights.uploadFusedQkvWeights;
const uploadFusedGateUpWeights = engine_weights.uploadFusedGateUpWeights;
const uploadMoEWeights = engine_weights.uploadMoEWeights;
const uploadVectorTensor = engine_weights.uploadVectorTensor;
const uploadShortConvWeightTimeMajor = engine_weights.uploadShortConvWeightTimeMajor;
const allocZeroedF32Buffer = engine_weights.allocZeroedF32Buffer;
const materializeTensorF32 = engine_weights.materializeTensorF32;
const bufferSlice = engine_weights.bufferSlice;
const kvCacheBytesForCapacityDtype = engine_weights.kvCacheBytesForCapacityDtype;
const allocDeviceKvPairWithScales = engine_weights.allocDeviceKvPairWithScales;

pub const GatedDeltaFfnUploadPlan = union(enum) {
    none,
    split: struct {
        w1: *const Tensor,
        w2: *const Tensor,
        w3: *const Tensor,
    },
    fused: struct {
        gate_up: Tensor,
        gate_up_layout: GateUpLayout,
        w2: *const Tensor,
    },
};

fn supportsFusedGateUpDenseUpload(dtype_tag: tensor.DType) bool {
    return switch (dtype_tag) {
        .f16, .bf16, .f32 => true,
        else => false,
    };
}

pub fn resolveGatedDeltaFfnUploadPlan(gated_delta: *const models.runtime_blocks.GatedDeltaBlockWeights) !GatedDeltaFfnUploadPlan {
    if (gated_delta.moe_weights != null) return .none;

    const split_w1 = gated_delta.w1;
    const split_w2 = gated_delta.w2 orelse gated_delta.down_proj;
    const split_w3 = gated_delta.w3;

    if (gated_delta.fused_gate_up) |fused| {
        const gate_up = fused.gate_up orelse return error.MissingWeight;
        const w2 = split_w2 orelse return error.MissingWeight;
        if (supportsFusedGateUpDenseUpload(gate_up.dtype)) {
            return .{
                .fused = .{
                    .gate_up = gate_up,
                    .gate_up_layout = fused.gate_up_layout,
                    .w2 = w2,
                },
            };
        }
        if (split_w1 != null or split_w2 != null or split_w3 != null) {
            return .{
                .split = .{
                    .w1 = split_w1 orelse return error.MissingWeight,
                    .w2 = split_w2 orelse return error.MissingWeight,
                    .w3 = split_w3 orelse return error.MissingWeight,
                },
            };
        }
        return error.UnsupportedModel;
    }

    if (split_w1 != null or split_w2 != null or split_w3 != null) {
        return .{
            .split = .{
                .w1 = split_w1 orelse return error.MissingWeight,
                .w2 = split_w2 orelse return error.MissingWeight,
                .w3 = split_w3 orelse return error.MissingWeight,
            },
        };
    }

    return .none;
}

const MoEByteTotals = struct {
    linear_weight_bytes: usize = 0,
    norm_weight_bytes: usize = 0,
};

fn computeMoEByteTotals(moe_refs: *const MoEWeightRefs) MoEByteTotals {
    var totals: MoEByteTotals = .{};
    for (moe_refs.expert_gate_up) |weight| totals.linear_weight_bytes += weight.byteSize();
    for (moe_refs.expert_down) |weight| totals.linear_weight_bytes += weight.byteSize();
    totals.linear_weight_bytes += moe_refs.router_proj.byteSize();

    if (moe_refs.pre_ffn_norm) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    if (moe_refs.post_shared_norm) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    if (moe_refs.pre_expert_norm) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    if (moe_refs.post_expert_norm) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    if (moe_refs.post_combine_norm) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    if (moe_refs.router_input_scale) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    if (moe_refs.router_per_expert_scale) |tensor_| totals.norm_weight_bytes += tensor_.byteSize();
    return totals;
}

fn compileLayerProgramMetadata(
    allocator: std.mem.Allocator,
    layer: *BlockRuntimeLayer,
    program: anytype,
    options: plan_compiler.CompileOptions,
    layer_idx: usize,
    kind: op_types.BlockKind,
    adapter_table: anytype,
) !void {
    layer.compiled_plan = try plan_compiler.compileLayerProgram(allocator, program, .decode, options);
    errdefer if (layer.compiled_plan) |*compiled_plan| {
        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
        layer.compiled_plan = null;
    };

    try validateCompiledLayerPlanForCuda(&layer.compiled_plan.?, layer_idx, kind, adapter_table);

    layer.register_to_slot_map = try buildCudaLayerProgramRegisterSlotMap(allocator, &layer.compiled_plan.?);
    errdefer if (layer.register_to_slot_map.len != 0) {
        allocator.free(layer.register_to_slot_map);
        layer.register_to_slot_map = &.{};
    };

    layer.slot_width_hints = try buildCudaLayerProgramSlotWidthHints(
        allocator,
        &layer.compiled_plan.?,
        layer.register_to_slot_map,
    );
}

fn deinitLayerProgramMetadata(layer: *BlockRuntimeLayer, allocator: std.mem.Allocator) void {
    if (layer.slot_width_hints.len != 0) {
        allocator.free(layer.slot_width_hints);
        layer.slot_width_hints = &.{};
    }
    if (layer.register_to_slot_map.len != 0) {
        allocator.free(layer.register_to_slot_map);
        layer.register_to_slot_map = &.{};
    }
    if (layer.compiled_plan) |*compiled_plan| {
        plan_compiler.deinitCompiledPlan(allocator, compiled_plan);
        layer.compiled_plan = null;
    }
}

pub fn init(
    allocator: std.mem.Allocator,
    device: *compute.cuda.Device,
    loaded: *const LoadedModel,
    max_seq_len: usize,
    kv_init_tokens: usize,
    gated_delta_ssm_i8_state: bool,
    adapter_table: anytype,
    kv_cache_dtype: KvCacheDtype,
) !BlockRuntime {
    return initRange(
        allocator,
        device,
        loaded,
        max_seq_len,
        kv_init_tokens,
        0,
        loaded.blocks.len,
        gated_delta_ssm_i8_state,
        adapter_table,
        kv_cache_dtype,
    );
}

/// Initialize a BlockRuntime for a contiguous range of decoder layers
/// [layer_start, layer_end). Used by pipeline parallel to split layers
/// across devices.
pub fn initRange(
    allocator: std.mem.Allocator,
    device: *compute.cuda.Device,
    loaded: *const LoadedModel,
    max_seq_len: usize,
    kv_init_tokens: usize,
    layer_start: usize,
    layer_end: usize,
    gated_delta_ssm_i8_state: bool,
    adapter_table: anytype,
    kv_cache_dtype: KvCacheDtype,
) !BlockRuntime {
    const d_model: usize = @intCast(loaded.config.d_model);
    const n_heads: usize = @intCast(loaded.config.n_heads);
    const n_kv_heads: usize = @intCast(loaded.config.n_kv_groups);
    const head_dim: usize = @intCast(loaded.config.head_dim);
    if (n_heads == 0 or n_kv_heads == 0 or head_dim == 0 or max_seq_len == 0) return error.InvalidArgument;
    if (n_heads % n_kv_heads != 0) return error.UnsupportedModel;
    if (layer_end > loaded.blocks.len or layer_start > layer_end) return error.InvalidArgument;
    const initial_kv_tokens = @min(max_seq_len, @max(@as(usize, 1), kv_init_tokens));
    const arena_allocator = @constCast(&loaded.arena).allocator();
    const static_entry = if (loaded.runtime.architecture_id) |arch_id|
        models.registry.detectByArchitectureId(arch_id)
    else
        null;
    const layer_count = layer_end - layer_start;
    var attention_block_count: usize = 0;
    var shortconv_block_count: usize = 0;
    var gated_delta_block_count: usize = 0;
    var q_norm_blocks: usize = 0;
    var k_norm_blocks: usize = 0;
    var linear_weight_bytes: usize = 0;
    var norm_weight_bytes: usize = 0;
    var kv_cache_bytes: usize = 0;
    var shortconv_state_bytes: usize = 0;
    var gated_delta_state_bytes: usize = 0;
    var max_shortconv_dim: usize = 0;

    // Track cross-device KV sharing references for mirror replication.
    const PendingMirror = struct { local_idx: usize, source_global: usize, kv_dim: usize };
    var pending_mirrors: std.ArrayListUnmanaged(PendingMirror) = .{};
    defer pending_mirrors.deinit(allocator);
    var max_gdelta_proj: usize = 0;
    var blocks = try allocator.alloc(BlockRuntimeLayer, layer_count);
    errdefer allocator.free(blocks);
    for (blocks) |*layer| layer.* = .{};

    var initialized: usize = 0;
    errdefer {
        while (initialized > 0) {
            initialized -= 1;
            blocks[initialized].deinit(allocator, device);
        }
    }
    const layer_blocks = loaded.blocks[layer_start..layer_end];
    for (layer_blocks, 0..) |*layer_weights, local_idx| {
        const layer_idx = layer_start + local_idx;
        const block_weights = try models.runtime_blocks.layerToBlockWeights(arena_allocator, layer_weights);
        switch (block_weights) {
            .attention_mlp => |attn| {
                const entry = static_entry orelse {
                    log.warn("inference", "CUDA block runtime missing architecture metadata", .{ .layer = layer_idx });
                    return error.UnsupportedModel;
                };
                const program = models.registry.blockProgramFor(entry, .attention_mlp) orelse {
                    log.warn("inference", "CUDA block runtime missing LayerOp program", .{
                        .layer = layer_idx,
                        .kind = @intFromEnum(op_types.BlockKind.attention_mlp),
                        .architecture = entry.id,
                    });
                    return error.UnsupportedModel;
                };
                try compileLayerProgramMetadata(
                    allocator,
                    &blocks[local_idx],
                    program,
                    .{
                        .size_floor = d_model,
                        .state_descriptor_entry = entry,
                    },
                    layer_idx,
                    .attention_mlp,
                    adapter_table,
                );
                errdefer deinitLayerProgramMetadata(&blocks[local_idx], allocator);
                if (attn.mla_config != null) {
                    log.warn("inference", "CUDA block runtime MLA not supported yet", .{ .layer = layer_idx });
                    return error.UnsupportedModel;
                }
                const has_moe = attn.moe_weights != null;
                const w2 = if (!has_moe) (attn.w2 orelse return error.MissingWeight) else null;
                const q_proj_src = attn.q_proj orelse return error.MissingWeight;
                const k_proj_src = attn.k_proj orelse return error.MissingWeight;
                const v_proj_src = attn.v_proj orelse return error.MissingWeight;
                const q_proj_out = try tensorProjectionOutputDim(q_proj_src, d_model);
                const q_out = if (attn.attention_config.query_gate) blk: {
                    if ((q_proj_out % 2) != 0) {
                        log.warn("inference", "CUDA block runtime q_proj dim unsupported for query_gate", .{
                            .layer = layer_idx,
                            .q_proj_out = q_proj_out,
                        });
                        return error.UnsupportedModel;
                    }
                    break :blk q_proj_out / 2;
                } else q_proj_out;
                const kv_out = try tensorProjectionOutputDim(k_proj_src, d_model);
                const v_out = try tensorProjectionOutputDim(v_proj_src, d_model);
                if (v_out != kv_out) {
                    log.warn("inference", "CUDA block runtime k/v dim mismatch", .{
                        .layer = layer_idx,
                        .k_cols = kv_out,
                        .v_cols = v_out,
                    });
                    return error.UnsupportedModel;
                }
                var layer_head_dim = head_dim;
                if (attn.q_norm) |q_norm_src| {
                    const q_norm_dim: usize = switch (q_norm_src.n_dims) {
                        1 => @intCast(q_norm_src.shape[0]),
                        2 => if (q_norm_src.shape[0] == 1 and q_norm_src.shape[1] > 0)
                            @intCast(q_norm_src.shape[1])
                        else if (q_norm_src.shape[1] == 1 and q_norm_src.shape[0] > 0)
                            @intCast(q_norm_src.shape[0])
                        else
                            return error.UnsupportedModel,
                        else => return error.UnsupportedModel,
                    };
                    if (q_norm_dim == 0) return error.UnsupportedModel;
                    layer_head_dim = q_norm_dim;
                } else if (attn.k_norm) |k_norm_src| {
                    const k_norm_dim: usize = switch (k_norm_src.n_dims) {
                        1 => @intCast(k_norm_src.shape[0]),
                        2 => if (k_norm_src.shape[0] == 1 and k_norm_src.shape[1] > 0)
                            @intCast(k_norm_src.shape[1])
                        else if (k_norm_src.shape[1] == 1 and k_norm_src.shape[0] > 0)
                            @intCast(k_norm_src.shape[0])
                        else
                            return error.UnsupportedModel,
                        else => return error.UnsupportedModel,
                    };
                    if (k_norm_dim == 0) return error.UnsupportedModel;
                    layer_head_dim = k_norm_dim;
                } else if ((q_out % n_heads) == 0 and n_heads > 0) {
                    layer_head_dim = q_out / n_heads;
                } else if ((kv_out % n_kv_heads) == 0 and n_kv_heads > 0) {
                    layer_head_dim = kv_out / n_kv_heads;
                }
                if (layer_head_dim == 0) return error.UnsupportedModel;
                const layer_n_heads = if ((q_out % layer_head_dim) == 0) q_out / layer_head_dim else n_heads;
                const layer_n_kv_heads = if ((kv_out % layer_head_dim) == 0) kv_out / layer_head_dim else n_kv_heads;
                if (layer_n_heads == 0 or layer_n_kv_heads == 0 or (layer_n_heads % layer_n_kv_heads) != 0) {
                    log.warn("inference", "CUDA block runtime inferred attention shape unsupported", .{
                        .layer = layer_idx,
                        .q_dim = q_out,
                        .kv_dim = kv_out,
                        .head_dim = layer_head_dim,
                        .n_heads = layer_n_heads,
                        .n_kv_heads = layer_n_kv_heads,
                    });
                    return error.UnsupportedModel;
                }
                if (attention_block_count == 0) {
                    if (has_moe) {
                        const moe = attn.moe_weights.?;
                        log.info("inference", "CUDA block0 MoE weight mode", .{
                            .num_experts = moe.num_experts,
                            .experts_per_token = moe.experts_per_token,
                            .q_out = q_out,
                            .q_proj_out = q_proj_out,
                            .kv_out = kv_out,
                            .head_dim = layer_head_dim,
                            .n_heads = layer_n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                        });
                    } else if (attn.fused.qkv_proj != null or attn.fused.gate_up != null) {
                        log.info("inference", "CUDA block0 fused weight mode", .{
                            .qkv_fused = @as(u8, @intFromBool(attn.fused.qkv_proj != null)),
                            .gate_up_fused = @as(u8, @intFromBool(attn.fused.gate_up != null)),
                            .gate_up_layout = @tagName(attn.fused.gate_up_layout),
                            .qkv_dtype = if (attn.fused.qkv_proj) |qkv| @tagName(qkv.dtype) else "none",
                            .gate_up_dtype = if (attn.fused.gate_up) |gate_up| @tagName(gate_up.dtype) else "none",
                            .w2_dtype = @tagName(w2.?.dtype),
                            .q_out = q_out,
                            .q_proj_out = q_proj_out,
                            .kv_out = kv_out,
                            .head_dim = layer_head_dim,
                            .n_heads = layer_n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                        });
                    } else {
                        const w1 = attn.w1 orelse return error.MissingWeight;
                        const w3 = attn.w3 orelse return error.MissingWeight;
                        log.info("inference", "CUDA block0 weight dtypes", .{
                            .q_proj = @tagName(q_proj_src.dtype),
                            .k_proj = @tagName(k_proj_src.dtype),
                            .v_proj = @tagName(v_proj_src.dtype),
                            .o_proj = @tagName(attn.o_proj.dtype),
                            .w1 = @tagName(w1.dtype),
                            .w2 = @tagName(w2.?.dtype),
                            .w3 = @tagName(w3.dtype),
                        });
                        log.info("inference", "CUDA block0 weight shapes", .{
                            .q0 = q_proj_src.shape[0],
                            .q1 = q_proj_src.shape[1],
                            .k0 = k_proj_src.shape[0],
                            .k1 = k_proj_src.shape[1],
                            .v0 = v_proj_src.shape[0],
                            .v1 = v_proj_src.shape[1],
                            .o0 = attn.o_proj.shape[0],
                            .o1 = attn.o_proj.shape[1],
                            .w10 = w1.shape[0],
                            .w11 = w1.shape[1],
                            .w20 = w2.?.shape[0],
                            .w21 = w2.?.shape[1],
                            .w30 = w3.shape[0],
                            .w31 = w3.shape[1],
                        });
                    }
                }

                var ln1_weight = try uploadTensor(device, allocator, attn.ln1_weight);
                errdefer ln1_weight.deinit(device);
                var ln2_weight = try uploadTensor(device, allocator, attn.ln2_weight);
                errdefer ln2_weight.deinit(device);
                if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                    log.warn("inference", "CUDA block runtime ln1 shape unsupported", .{
                        .layer = layer_idx,
                        .rows = ln1_weight.rows,
                        .cols = ln1_weight.cols,
                        .d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }
                if (!(ln2_weight.rows == d_model and ln2_weight.cols == 1)) {
                    log.warn("inference", "CUDA block runtime ln2 shape unsupported", .{
                        .layer = layer_idx,
                        .rows = ln2_weight.rows,
                        .cols = ln2_weight.cols,
                        .d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }

                var pre_ffn_norm_weight: ?DeviceTensor = null;
                if (attn.pre_ffn_norm) |pre_ffn_norm| {
                    var pre_ffn = try uploadTensor(device, allocator, pre_ffn_norm);
                    errdefer pre_ffn.deinit(device);
                    if (!(pre_ffn.rows == d_model and pre_ffn.cols == 1)) {
                        log.warn("inference", "CUDA block runtime pre_ffn_norm shape unsupported", .{
                            .layer = layer_idx,
                            .rows = pre_ffn.rows,
                            .cols = pre_ffn.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    pre_ffn_norm_weight = pre_ffn;
                }
                errdefer if (pre_ffn_norm_weight) |*w| w.deinit(device);

                var post_ffn_norm_weight: ?DeviceTensor = null;
                if (attn.post_ffn_norm) |post_ffn_norm| {
                    var post_ffn = try uploadTensor(device, allocator, post_ffn_norm);
                    errdefer post_ffn.deinit(device);
                    if (!(post_ffn.rows == d_model and post_ffn.cols == 1)) {
                        log.warn("inference", "CUDA block runtime post_ffn_norm shape unsupported", .{
                            .layer = layer_idx,
                            .rows = post_ffn.rows,
                            .cols = post_ffn.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    post_ffn_norm_weight = post_ffn;
                }
                errdefer if (post_ffn_norm_weight) |*w| w.deinit(device);

                var q_norm_weight: ?DeviceTensor = null;
                if (attn.q_norm) |q_norm| {
                    var qn = try uploadTensor(device, allocator, q_norm);
                    errdefer qn.deinit(device);
                    if (!(qn.rows == layer_head_dim and qn.cols == 1)) {
                        log.warn("inference", "CUDA block runtime q_norm shape unsupported", .{
                            .layer = layer_idx,
                            .rows = qn.rows,
                            .cols = qn.cols,
                            .head_dim = layer_head_dim,
                        });
                        return error.UnsupportedModel;
                    }
                    q_norm_weight = qn;
                    q_norm_blocks += 1;
                }
                errdefer if (q_norm_weight) |*w| w.deinit(device);

                var k_norm_weight: ?DeviceTensor = null;
                if (attn.k_norm) |k_norm| {
                    var kn = try uploadTensor(device, allocator, k_norm);
                    errdefer kn.deinit(device);
                    if (!(kn.rows == layer_head_dim and kn.cols == 1)) {
                        log.warn("inference", "CUDA block runtime k_norm shape unsupported", .{
                            .layer = layer_idx,
                            .rows = kn.rows,
                            .cols = kn.cols,
                            .head_dim = layer_head_dim,
                        });
                        return error.UnsupportedModel;
                    }
                    k_norm_weight = kn;
                    k_norm_blocks += 1;
                }
                errdefer if (k_norm_weight) |*w| w.deinit(device);

                var q_proj_dev: LinearWeight = undefined;
                var k_proj_dev: LinearWeight = undefined;
                var v_proj_dev: LinearWeight = undefined;
                if (attn.fused.qkv_proj) |qkv_proj| {
                    if (attn.attention_config.query_gate) {
                        log.warn("inference", "CUDA block runtime fused qkv with query_gate unsupported", .{
                            .layer = layer_idx,
                        });
                        return error.UnsupportedModel;
                    }
                    const fused_qkv = try uploadFusedQkvWeights(
                        device,
                        allocator,
                        &qkv_proj,
                        d_model,
                        q_out,
                        kv_out,
                    );
                    q_proj_dev = fused_qkv.q;
                    k_proj_dev = fused_qkv.k;
                    v_proj_dev = fused_qkv.v;
                } else {
                    q_proj_dev = try uploadLinearWeightWithContext(device, allocator, q_proj_src, d_model, layer_idx, "self_attn.q_proj.weight");
                    k_proj_dev = try uploadLinearWeightWithContext(device, allocator, k_proj_src, d_model, layer_idx, "self_attn.k_proj.weight");
                    v_proj_dev = try uploadLinearWeightWithContext(device, allocator, v_proj_src, d_model, layer_idx, "self_attn.v_proj.weight");
                }
                errdefer q_proj_dev.deinit(device);
                errdefer k_proj_dev.deinit(device);
                errdefer v_proj_dev.deinit(device);

                var o_proj_dev = try uploadLinearWeightWithContext(device, allocator, attn.o_proj, q_out, layer_idx, "self_attn.o_proj.weight");
                errdefer o_proj_dev.deinit(device);

                // Upload FFN weights: either standard SwiGLU (w1/w2/w3) or MoE
                var w1_dev: LinearWeight = undefined;
                var w3_dev: LinearWeight = undefined;
                var w2_dev: LinearWeight = undefined;
                var d_ff: usize = 0;
                var moe_weight_refs: ?MoEWeightRefs = null;
                if (has_moe) {
                    const moe = attn.moe_weights.?;
                    const moe_result = try uploadMoEWeights(device, allocator, moe, d_model, layer_idx, loaded.config.use_gelu);
                    moe_weight_refs = moe_result;
                    // Use dummy LinearWeights for w1/w2/w3 — not accessed for MoE layers
                    w1_dev = moe_result.shared_gate;
                    w3_dev = moe_result.shared_up;
                    w2_dev = moe_result.shared_down;
                    d_ff = @max(moe_result.shared_d_ff, 2 * @as(usize, moe_result.expert_d_ff));
                } else if (attn.fused.gate_up) |gate_up| {
                    if (supportsFusedGateUpDenseUpload(gate_up.dtype)) {
                        const fused_gate_up = try uploadFusedGateUpWeights(
                            device,
                            allocator,
                            &gate_up,
                            d_model,
                            attn.fused.gate_up_layout,
                        );
                        w1_dev = fused_gate_up.gate;
                        w3_dev = fused_gate_up.up;
                        d_ff = w1_dev.cols();
                        w2_dev = try uploadLinearWeightWithContext(device, allocator, w2.?, d_ff, layer_idx, "mlp.down_proj.weight");
                    } else {
                        const w1 = attn.w1 orelse return error.MissingWeight;
                        const w3 = attn.w3 orelse return error.MissingWeight;
                        w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                        w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                        if (w1_dev.cols() != w3_dev.cols()) {
                            log.warn("inference", "CUDA block runtime gate/up dim mismatch", .{
                                .layer = layer_idx,
                                .w1_cols = w1_dev.cols(),
                                .w3_cols = w3_dev.cols(),
                            });
                            return error.UnsupportedModel;
                        }
                        d_ff = w1_dev.cols();
                        w2_dev = try uploadLinearWeightWithContext(device, allocator, w2.?, d_ff, layer_idx, "mlp.down_proj.weight");
                    }
                } else {
                    const w1 = attn.w1 orelse return error.MissingWeight;
                    const w3 = attn.w3 orelse return error.MissingWeight;
                    w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                    w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                    if (w1_dev.cols() != w3_dev.cols()) {
                        log.warn("inference", "CUDA block runtime gate/up dim mismatch", .{
                            .layer = layer_idx,
                            .w1_cols = w1_dev.cols(),
                            .w3_cols = w3_dev.cols(),
                        });
                        return error.UnsupportedModel;
                    }
                    d_ff = w1_dev.cols();
                    w2_dev = try uploadLinearWeightWithContext(device, allocator, w2.?, d_ff, layer_idx, "mlp.down_proj.weight");
                }
                errdefer w1_dev.deinit(device);
                errdefer w3_dev.deinit(device);
                errdefer w2_dev.deinit(device);
                const cpu_attention_kernel: ?cpu_kernels.MultiHeadAttention = null;
                const cpu_attention_cache: ?cpu_kernels.AttnCache = null;
                const cpu_attention_scratch: ?cpu_kernels.AttnTemp = null;
                const cpu_attention_matmul_scratch: ?compute.cpu.linalg.MatmulScratch = null;
                if (k_proj_dev.cols() != v_proj_dev.cols()) {
                    log.warn("inference", "CUDA block runtime k/v dim mismatch", .{
                        .layer = layer_idx,
                        .k_cols = k_proj_dev.cols(),
                        .v_cols = v_proj_dev.cols(),
                    });
                    return error.UnsupportedModel;
                }
                if (o_proj_dev.cols() != d_model) {
                    log.warn("inference", "CUDA block runtime o_proj out dim unsupported", .{
                        .layer = layer_idx,
                        .o_proj_cols = o_proj_dev.cols(),
                        .d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }
                if (w2_dev.cols() != d_model) {
                    log.warn("inference", "CUDA block runtime down_proj out dim unsupported", .{
                        .layer = layer_idx,
                        .w2_cols = w2_dev.cols(),
                        .d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }
                if (q_proj_dev.cols() != q_proj_out) {
                    log.warn("inference", "CUDA block runtime q_proj dim unsupported", .{
                        .layer = layer_idx,
                        .q_cols = q_proj_dev.cols(),
                        .expected = q_proj_out,
                        .query_gate = @as(u8, @intFromBool(attn.attention_config.query_gate)),
                    });
                    return error.UnsupportedModel;
                }
                if (k_proj_dev.cols() != kv_out) {
                    log.warn("inference", "CUDA block runtime kv dim unsupported", .{
                        .layer = layer_idx,
                        .kv_cols = k_proj_dev.cols(),
                        .expected = kv_out,
                    });
                    return error.UnsupportedModel;
                }

                const kv_capacity = initial_kv_tokens;
                if (kv_capacity == 0) return error.InvalidArgument;
                const kv_cache_bytes_per_buffer = try kvCacheBytesForCapacityDtype(kv_capacity, k_proj_dev.cols(), kv_cache_dtype);
                var kv_pair = try allocDeviceKvPairWithScales(device, kv_capacity, k_proj_dev.cols(), layer_n_kv_heads, kv_cache_dtype);
                errdefer {
                    if (kv_pair.v_scale.pointer != 0) kv_pair.v_scale.deinit(device);
                    if (kv_pair.k_scale.pointer != 0) kv_pair.k_scale.deinit(device);
                    kv_pair.v.deinit(device);
                    kv_pair.k.deinit(device);
                }

                const layer_norm_bytes = ln1_weight.byteSize() +
                    ln2_weight.byteSize() +
                    (if (pre_ffn_norm_weight) |w| w.byteSize() else 0) +
                    (if (post_ffn_norm_weight) |w| w.byteSize() else 0) +
                    (if (q_norm_weight) |w| w.byteSize() else 0) +
                    (if (k_norm_weight) |w| w.byteSize() else 0);
                norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                const layer_linear_bytes = q_proj_dev.byteSize() +
                    k_proj_dev.byteSize() +
                    v_proj_dev.byteSize() +
                    o_proj_dev.byteSize() +
                    w1_dev.byteSize() +
                    w2_dev.byteSize() +
                    w3_dev.byteSize();
                linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;

                const layer_kv_bytes = std.math.mul(usize, kv_cache_bytes_per_buffer, 2) catch return error.InvalidArgument;
                kv_cache_bytes = std.math.add(usize, kv_cache_bytes, layer_kv_bytes) catch return error.InvalidArgument;

                const slot_kv_index = attention_block_count;
                const kv_shared_source_layer_global = resolveSharedKvSourceLayer(loaded.config, layer_idx);
                // Convert global source layer to local index. If the source
                // is on a different device, record it for mirror replication.
                const kv_shared_source_layer: ?usize = if (kv_shared_source_layer_global) |src_layer_idx| blk: {
                    if (src_layer_idx < layer_start or src_layer_idx >= layer_end) {
                        try pending_mirrors.append(allocator, .{
                            .local_idx = local_idx,
                            .source_global = src_layer_idx,
                            .kv_dim = k_proj_dev.cols(),
                        });
                        break :blk null;
                    }
                    break :blk src_layer_idx - layer_start;
                } else null;
                const kv_shared_source_slot_kv_index: ?usize = if (kv_shared_source_layer) |src_local_idx| blk: {
                    if (src_local_idx >= blocks.len) break :blk null;
                    const src_binding = blocks[src_local_idx].attention_binding orelse break :blk null;
                    break :blk src_binding.slot_kv_index;
                } else null;

                blocks[local_idx].attention_runtime = .{
                    .q_dim = q_out,
                    .q_projection_dim = q_proj_dev.cols(),
                    .kv_dim = k_proj_dev.cols(),
                    .d_ff = d_ff,
                    .sliding_window = attn.sliding_window,
                    .is_causal = attn.is_causal,
                    .query_gate = attn.attention_config.query_gate,
                    .ln1_weight = ln1_weight,
                    .ln2_weight = ln2_weight,
                    .pre_ffn_norm_weight = pre_ffn_norm_weight,
                    .post_ffn_norm_weight = post_ffn_norm_weight,
                    .q_norm_weight = q_norm_weight,
                    .k_norm_weight = k_norm_weight,
                    .q_proj = q_proj_dev,
                    .k_proj = k_proj_dev,
                    .v_proj = v_proj_dev,
                    .o_proj = o_proj_dev,
                    .w1 = w1_dev,
                    .w2 = w2_dev,
                    .w3 = w3_dev,
                    .k_cache = kv_pair.k,
                    .v_cache = kv_pair.v,
                    .k_scale = kv_pair.k_scale,
                    .v_scale = kv_pair.v_scale,
                    .kv_capacity = kv_capacity,
                    .slot_kv_index = slot_kv_index,
                    .kv_shared_source_layer = kv_shared_source_layer,
                    .kv_shared_source_slot_kv_index = kv_shared_source_slot_kv_index,
                    .use_v_norm = loaded.config.use_v_norm,
                    .cpu_kernel = cpu_attention_kernel,
                    .cpu_cache = cpu_attention_cache,
                    .cpu_scratch = cpu_attention_scratch,
                    .cpu_matmul_scratch = cpu_attention_matmul_scratch,
                };
                blocks[local_idx].attention_binding = &blocks[local_idx].attention_runtime.?;
                BlockRuntimeLayer.bindAttentionNormWeights(&blocks[local_idx], &blocks[local_idx].attention_runtime.?);
                if (moe_weight_refs) |moe_refs| {
                    blocks[local_idx].moe_runtime = moe_refs;
                    blocks[local_idx].moe_binding = &blocks[local_idx].moe_runtime.?;
                    const moe_bytes = computeMoEByteTotals(&moe_refs);
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, moe_bytes.linear_weight_bytes) catch return error.InvalidArgument;
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, moe_bytes.norm_weight_bytes) catch return error.InvalidArgument;
                }
                attention_block_count += 1;
            },
            .gated_delta => |gated_delta| {
                const in_proj_cols = try tensorProjectionOutputDim(gated_delta.weights.in_proj, d_model);
                max_gdelta_proj = @max(max_gdelta_proj, in_proj_cols);
                const entry = static_entry orelse {
                    log.warn("inference", "CUDA gated-delta runtime missing architecture metadata", .{ .layer = layer_idx });
                    return error.UnsupportedModel;
                };
                const program = models.registry.blockProgramFor(entry, .gated_delta) orelse {
                    log.warn("inference", "CUDA gated-delta runtime missing LayerOp program", .{
                        .layer = layer_idx,
                        .kind = @intFromEnum(op_types.BlockKind.gated_delta),
                        .architecture = entry.id,
                    });
                    return error.UnsupportedModel;
                };
                try compileLayerProgramMetadata(
                    allocator,
                    &blocks[local_idx],
                    program,
                    .{
                        .size_floor = d_model,
                        .state_descriptor_entry = entry,
                        .gated_delta_config_override = .{
                            .d_conv = @intCast(gated_delta.config.d_conv),
                            .n_heads = @intCast(gated_delta.config.n_heads),
                            .d_head = @intCast(gated_delta.config.d_head),
                            .d_inner = @intCast(@as(usize, gated_delta.config.n_heads) * @as(usize, gated_delta.config.d_head)),
                        },
                    },
                    layer_idx,
                    .gated_delta,
                    adapter_table,
                );
                errdefer deinitLayerProgramMetadata(&blocks[local_idx], allocator);
                if (gated_delta.config.d_model != d_model) {
                    log.warn("inference", "CUDA gated-delta d_model mismatch", .{
                        .layer = layer_idx,
                        .config_d_model = gated_delta.config.d_model,
                        .model_d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }

                var ln1_weight = try uploadTensor(device, allocator, gated_delta.ln1_weight);
                errdefer ln1_weight.deinit(device);
                if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                    log.warn("inference", "CUDA gated-delta ln1 shape unsupported", .{
                        .layer = layer_idx,
                        .rows = ln1_weight.rows,
                        .cols = ln1_weight.cols,
                        .d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }

                var ln2_weight: ?DeviceTensor = null;
                if (gated_delta.ln2_weight) |ln2| {
                    var ln2_dev = try uploadTensor(device, allocator, ln2);
                    errdefer ln2_dev.deinit(device);
                    if (!(ln2_dev.rows == d_model and ln2_dev.cols == 1)) {
                        log.warn("inference", "CUDA gated-delta ln2 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln2_dev.rows,
                            .cols = ln2_dev.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    ln2_weight = ln2_dev;
                }
                errdefer if (ln2_weight) |*w| w.deinit(device);

                // Upload FFN weights: either MoE or standard SwiGLU (w1/w2/w3)
                const has_gd_moe = gated_delta.moe_weights != null;
                var moe_weight_refs: ?MoEWeightRefs = null;
                if (has_gd_moe) {
                    const gd_moe = gated_delta.moe_weights.?;
                    moe_weight_refs = try uploadMoEWeights(device, allocator, gd_moe, d_model, layer_idx, loaded.config.use_gelu);
                }
                errdefer if (moe_weight_refs) |*mwr| mwr.deinit(allocator, device);

                var ffn_w1: ?LinearWeight = null;
                var ffn_w2: ?LinearWeight = null;
                var ffn_w3: ?LinearWeight = null;
                var d_ff: usize = 0;
                if (!has_gd_moe) {
                    const ffn_plan = try resolveGatedDeltaFfnUploadPlan(&gated_delta);
                    switch (ffn_plan) {
                        .none => {
                            log.warn("inference", "CUDA gated-delta missing FFN weights", .{
                                .layer = layer_idx,
                            });
                            return error.UnsupportedModel;
                        },
                        .split => |split| {
                            if (ln2_weight == null) {
                                log.warn("inference", "CUDA gated-delta ffn requires ln2", .{
                                    .layer = layer_idx,
                                });
                                return error.UnsupportedModel;
                            }
                            var w1_dev = try uploadLinearWeightWithContext(device, allocator, split.w1, d_model, layer_idx, "mlp.gate_proj.weight");
                            errdefer w1_dev.deinit(device);
                            var w3_dev = try uploadLinearWeightWithContext(device, allocator, split.w3, d_model, layer_idx, "mlp.up_proj.weight");
                            errdefer w3_dev.deinit(device);
                            if (w1_dev.cols() != w3_dev.cols()) {
                                log.warn("inference", "CUDA gated-delta gate/up dim mismatch", .{
                                    .layer = layer_idx,
                                    .w1_cols = w1_dev.cols(),
                                    .w3_cols = w3_dev.cols(),
                                });
                                return error.UnsupportedModel;
                            }
                            d_ff = w1_dev.cols();
                            var w2_dev = try uploadLinearWeightWithContext(device, allocator, split.w2, d_ff, layer_idx, "mlp.down_proj.weight");
                            errdefer w2_dev.deinit(device);
                            if (w2_dev.cols() != d_model) {
                                log.warn("inference", "CUDA gated-delta down_proj out dim unsupported", .{
                                    .layer = layer_idx,
                                    .w2_cols = w2_dev.cols(),
                                    .d_model = d_model,
                                });
                                return error.UnsupportedModel;
                            }
                            ffn_w1 = w1_dev;
                            ffn_w2 = w2_dev;
                            ffn_w3 = w3_dev;
                        },
                        .fused => |fused| {
                            if (ln2_weight == null) {
                                log.warn("inference", "CUDA gated-delta ffn requires ln2", .{
                                    .layer = layer_idx,
                                });
                                return error.UnsupportedModel;
                            }
                            const fused_gate_up = try uploadFusedGateUpWeights(
                                device,
                                allocator,
                                &fused.gate_up,
                                d_model,
                                fused.gate_up_layout,
                            );
                            var w1_dev = fused_gate_up.gate;
                            errdefer w1_dev.deinit(device);
                            var w3_dev = fused_gate_up.up;
                            errdefer w3_dev.deinit(device);
                            d_ff = w1_dev.cols();
                            var w2_dev = try uploadLinearWeightWithContext(device, allocator, fused.w2, d_ff, layer_idx, "mlp.down_proj.weight");
                            errdefer w2_dev.deinit(device);
                            if (w2_dev.cols() != d_model) {
                                log.warn("inference", "CUDA gated-delta down_proj out dim unsupported", .{
                                    .layer = layer_idx,
                                    .w2_cols = w2_dev.cols(),
                                    .d_model = d_model,
                                });
                                return error.UnsupportedModel;
                            }
                            ffn_w1 = w1_dev;
                            ffn_w2 = w2_dev;
                            ffn_w3 = w3_dev;
                        },
                    }
                }
                errdefer if (ffn_w1) |*w| w.deinit(device);
                errdefer if (ffn_w2) |*w| w.deinit(device);
                errdefer if (ffn_w3) |*w| w.deinit(device);

                const gated_delta_kernel_config = cpu_kernels.GatedDeltaConfig{
                    .d_model = gated_delta.config.d_model,
                    .d_conv = gated_delta.config.d_conv,
                    .n_heads = gated_delta.config.n_heads,
                    .d_head = gated_delta.config.d_head,
                    .n_key_heads = gated_delta.config.n_key_heads,
                };
                const conv_values = try materializeTensorF32(allocator, gated_delta.weights.conv1d_weight);
                defer allocator.free(conv_values);
                if (gated_delta_kernel_config.d_conv == 0 or (conv_values.len % gated_delta_kernel_config.d_conv) != 0) return error.UnsupportedModel;
                const gated_delta_conv_dim = conv_values.len / gated_delta_kernel_config.d_conv;

                var in_proj_dev = try uploadLinearWeightWithContext(device, allocator, gated_delta.weights.in_proj, d_model, layer_idx, "gated_delta.in_proj");
                errdefer in_proj_dev.deinit(device);
                var out_proj_dev = try uploadLinearWeightWithContext(device, allocator, gated_delta.weights.out_proj, @as(usize, gated_delta.config.n_heads) * @as(usize, gated_delta.config.d_head), layer_idx, "gated_delta.out_proj");
                errdefer out_proj_dev.deinit(device);
                var conv_weight_time_major = try uploadShortConvWeightTimeMajor(
                    device,
                    allocator,
                    gated_delta.weights.conv1d_weight,
                    gated_delta_conv_dim,
                    gated_delta_kernel_config.d_conv,
                );
                errdefer conv_weight_time_major.deinit(device);
                var conv_bias_dev: ?DeviceTensor = null;
                if (gated_delta.weights.conv1d_bias) |bias| {
                    var bias_dev = try uploadVectorTensor(device, allocator, bias, gated_delta_conv_dim);
                    errdefer bias_dev.deinit(device);
                    conv_bias_dev = bias_dev;
                }
                errdefer if (conv_bias_dev) |*w| w.deinit(device);
                var a_log_dev = try uploadTensor(device, allocator, gated_delta.weights.A_log);
                errdefer a_log_dev.deinit(device);
                var dt_bias_dev: ?DeviceTensor = null;
                if (gated_delta.weights.dt_bias) |bias| {
                    var bias_dev = try uploadTensor(device, allocator, bias);
                    errdefer bias_dev.deinit(device);
                    dt_bias_dev = bias_dev;
                }
                errdefer if (dt_bias_dev) |*w| w.deinit(device);
                const norm_weight_dev = blk: {
                    if (gated_delta.weights.norm_weight) |norm_weight| {
                        break :blk try uploadTensor(device, allocator, norm_weight);
                    }
                    const default_len: usize = gated_delta.config.d_head;
                    const ones = try allocator.alloc(f32, default_len);
                    defer allocator.free(ones);
                    @memset(ones, 1.0);
                    var buffer = try device.allocBuffer(default_len * @sizeOf(f32));
                    errdefer buffer.deinit(device);
                    try buffer.upload(device, std.mem.sliceAsBytes(ones));
                    break :blk DeviceTensor{ .rows = 1, .cols = default_len, .buffer = buffer };
                };
                errdefer {
                    var norm = norm_weight_dev;
                    norm.deinit(device);
                }

                // CPU matmul dispatchers for the GatedDeltaKernel struct. The CUDA engine
                // never calls these (GPU handles all matmul), but the struct requires them.
                // Fall back to matmulF32 for dtypes without a CPU kernel (e.g. FP8).
                const in_proj_fn = (compute.cpu.linalg.matmulKernel(gated_delta.weights.in_proj.dtype) catch
                    compute.cpu.linalg.DispatchedKernel{ .func = compute.cpu.linalg.matmulF32, .name = "matmulF32" }).func;
                const out_proj_fn = (compute.cpu.linalg.matmulKernel(gated_delta.weights.out_proj.dtype) catch
                    compute.cpu.linalg.DispatchedKernel{ .func = compute.cpu.linalg.matmulF32, .name = "matmulF32" }).func;
                var gated_delta_kernel = cpu_kernels.GatedDeltaKernel.init(
                    gated_delta_kernel_config,
                    .{
                        .in_proj = gated_delta.weights.in_proj,
                        .conv1d_weight = gated_delta.weights.conv1d_weight,
                        .conv1d_bias = gated_delta.weights.conv1d_bias,
                        .A_log = gated_delta.weights.A_log,
                        .dt_bias = gated_delta.weights.dt_bias,
                        .norm_weight = gated_delta.weights.norm_weight,
                        .out_proj = gated_delta.weights.out_proj,
                    },
                    in_proj_fn,
                    out_proj_fn,
                );
                gated_delta_kernel.layer_idx = @intCast(layer_idx);
                try gated_delta_kernel.initTransposedWeights(allocator);
                errdefer gated_delta_kernel.deinit();

                var gated_delta_state = try cpu_kernels.GatedDeltaState.init(
                    allocator,
                    1,
                    .{
                        .d_model = gated_delta.config.d_model,
                        .d_conv = gated_delta.config.d_conv,
                        .n_heads = gated_delta.config.n_heads,
                        .d_head = gated_delta.config.d_head,
                        .n_key_heads = gated_delta.config.n_key_heads,
                    },
                );
                errdefer gated_delta_state.deinit();
                const conv_state_bytes = gated_delta_state.conv_state.len * @sizeOf(f32);
                var conv_state_dev = try device.allocBuffer(conv_state_bytes);
                errdefer conv_state_dev.deinit(device);
                const zero_conv = try allocator.alloc(f32, gated_delta_state.conv_state.len);
                defer allocator.free(zero_conv);
                @memset(zero_conv, 0.0);
                try conv_state_dev.upload(device, std.mem.sliceAsBytes(zero_conv));
                const ssm_state_format: GatedDeltaSsmStateFormat = if (gated_delta_ssm_i8_state)
                    .i8_per_column_scale
                else
                    .f32;
                var ssm_state_scales_offset_bytes: usize = 0;
                var ssm_state_storage_bytes: usize = 0;
                var ssm_state_dev: compute.cuda.Buffer = undefined;
                switch (ssm_state_format) {
                    .f32 => {
                        const ssm_state_bytes = gated_delta_state.ssm_state.len * @sizeOf(f32);
                        ssm_state_storage_bytes = ssm_state_bytes;
                        ssm_state_dev = try device.allocBuffer(ssm_state_bytes);
                        errdefer ssm_state_dev.deinit(device);
                        const zero_ssm = try allocator.alloc(f32, gated_delta_state.ssm_state.len);
                        defer allocator.free(zero_ssm);
                        @memset(zero_ssm, 0.0);
                        try ssm_state_dev.upload(device, std.mem.sliceAsBytes(zero_ssm));
                    },
                    .i8_per_column_scale => {
                        const d_inner = @as(usize, gated_delta.config.n_heads) * @as(usize, gated_delta.config.d_head);
                        const ssm_state_i8_bytes = gated_delta_state.ssm_state.len;
                        const ssm_scales_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
                        ssm_state_scales_offset_bytes = std.mem.alignForward(usize, ssm_state_i8_bytes, @alignOf(f32));
                        ssm_state_storage_bytes = std.math.add(
                            usize,
                            ssm_state_scales_offset_bytes,
                            ssm_scales_bytes,
                        ) catch return error.InvalidArgument;
                        ssm_state_dev = try device.allocBuffer(ssm_state_storage_bytes);
                        errdefer ssm_state_dev.deinit(device);

                        const zero_ssm = try allocator.alloc(i8, gated_delta_state.ssm_state.len);
                        defer allocator.free(zero_ssm);
                        @memset(zero_ssm, 0);
                        var ssm_state_i8_dev = try bufferSlice(&ssm_state_dev, 0, ssm_state_i8_bytes);
                        try ssm_state_i8_dev.upload(device, std.mem.sliceAsBytes(zero_ssm));

                        const init_scales = try allocator.alloc(f32, d_inner);
                        defer allocator.free(init_scales);
                        @memset(init_scales, 1.0);
                        var ssm_state_scales_dev = try bufferSlice(&ssm_state_dev, ssm_state_scales_offset_bytes, ssm_scales_bytes);
                        try ssm_state_scales_dev.upload(device, std.mem.sliceAsBytes(init_scales));
                    },
                }

                var gated_delta_scratch = try cpu_kernels.GatedDeltaScratch.init(
                    allocator,
                    .{
                        .d_model = gated_delta.config.d_model,
                        .d_conv = gated_delta.config.d_conv,
                        .n_heads = gated_delta.config.n_heads,
                        .d_head = gated_delta.config.d_head,
                        .n_key_heads = gated_delta.config.n_key_heads,
                    },
                );
                errdefer gated_delta_scratch.deinit();

                var gated_delta_matmul_scratch = try compute.cpu.linalg.MatmulScratch.init(allocator);
                errdefer gated_delta_matmul_scratch.deinit();

                const layer_norm_bytes = ln1_weight.byteSize() + (if (ln2_weight) |w| w.byteSize() else 0);
                norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                var layer_linear_bytes: usize = in_proj_dev.byteSize() + out_proj_dev.byteSize();
                if (ffn_w1) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                if (ffn_w2) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                if (ffn_w3) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;

                const gated_delta_state_bytes_layer = std.math.add(
                    usize,
                    conv_state_bytes,
                    ssm_state_storage_bytes,
                ) catch return error.InvalidArgument;
                gated_delta_state_bytes = std.math.add(usize, gated_delta_state_bytes, gated_delta_state_bytes_layer) catch return error.InvalidArgument;

                blocks[local_idx].gated_delta_runtime = .{
                    .d_ff = d_ff,
                    .ln1_weight = ln1_weight,
                    .ln2_weight = ln2_weight,
                    .ffn_w1 = ffn_w1,
                    .ffn_w2 = ffn_w2,
                    .ffn_w3 = ffn_w3,
                    .in_proj = in_proj_dev,
                    .out_proj = out_proj_dev,
                    .conv_weight_time_major = conv_weight_time_major,
                    .conv_bias = conv_bias_dev,
                    .conv_state_dev = conv_state_dev,
                    .conv_ring_head = 0,
                    .a_log = a_log_dev,
                    .dt_bias = dt_bias_dev,
                    .norm_weight = norm_weight_dev,
                    .ssm_state_dev = ssm_state_dev,
                    .ssm_state_format = ssm_state_format,
                    .ssm_state_scales_offset = @intCast(ssm_state_scales_offset_bytes),
                    .kernel = gated_delta_kernel,
                    .state = gated_delta_state,
                    .scratch = gated_delta_scratch,
                    .matmul_scratch = gated_delta_matmul_scratch,
                };
                blocks[local_idx].gated_delta_binding = &blocks[local_idx].gated_delta_runtime.?;
                BlockRuntimeLayer.bindGatedDeltaNormWeights(&blocks[local_idx], &blocks[local_idx].gated_delta_runtime.?);
                if (moe_weight_refs) |moe_refs| {
                    blocks[local_idx].moe_runtime = moe_refs;
                    blocks[local_idx].moe_binding = &blocks[local_idx].moe_runtime.?;
                    const moe_bytes = computeMoEByteTotals(&moe_refs);
                    linear_weight_bytes = std.math.add(usize, linear_weight_bytes, moe_bytes.linear_weight_bytes) catch return error.InvalidArgument;
                    norm_weight_bytes = std.math.add(usize, norm_weight_bytes, moe_bytes.norm_weight_bytes) catch return error.InvalidArgument;
                }
                gated_delta_block_count += 1;
            },
            .shortconv => |shortconv| {
                const entry = static_entry orelse {
                    log.warn("inference", "CUDA shortconv runtime missing architecture metadata", .{ .layer = layer_idx });
                    return error.UnsupportedModel;
                };
                const program = models.registry.blockProgramFor(entry, .shortconv) orelse {
                    log.warn("inference", "CUDA shortconv runtime missing LayerOp program", .{
                        .layer = layer_idx,
                        .kind = @intFromEnum(op_types.BlockKind.shortconv),
                        .architecture = entry.id,
                    });
                    return error.UnsupportedModel;
                };
                try compileLayerProgramMetadata(
                    allocator,
                    &blocks[local_idx],
                    program,
                    .{
                        .size_floor = d_model,
                        .state_descriptor_entry = entry,
                    },
                    layer_idx,
                    .shortconv,
                    adapter_table,
                );
                errdefer deinitLayerProgramMetadata(&blocks[local_idx], allocator);
                if (shortconv.fused_gate_up != null) {
                    log.warn("inference", "CUDA block runtime fused shortconv gate_up not supported yet", .{
                        .layer = layer_idx,
                    });
                    return error.UnsupportedModel;
                }
                const conv_dim: usize = @intCast(shortconv.config.conv_dim);
                const d_conv: usize = @intCast(shortconv.config.d_conv);
                if (shortconv_block_count == 0) {
                    log.info("inference", "CUDA shortconv block0 config", .{
                        .layer = layer_idx,
                        .d_model = shortconv.config.d_model,
                        .conv_dim = shortconv.config.conv_dim,
                        .conv_dim_out = shortconv.config.conv_dim_out,
                        .d_conv = shortconv.config.d_conv,
                        .has_bias = @as(u8, @intFromBool(shortconv.config.has_bias)),
                        .in_proj_dtype = @tagName(shortconv.weights.in_proj.dtype),
                        .in_proj_0 = shortconv.weights.in_proj.shape[0],
                        .in_proj_1 = shortconv.weights.in_proj.shape[1],
                        .conv_weight_dtype = @tagName(shortconv.weights.conv1d_weight.dtype),
                        .conv_weight_n_dims = shortconv.weights.conv1d_weight.n_dims,
                        .conv_weight_0 = shortconv.weights.conv1d_weight.shape[0],
                        .conv_weight_1 = shortconv.weights.conv1d_weight.shape[1],
                        .conv_weight_2 = shortconv.weights.conv1d_weight.shape[2],
                        .out_proj_dtype = @tagName(shortconv.weights.out_proj.dtype),
                        .out_proj_0 = shortconv.weights.out_proj.shape[0],
                        .out_proj_1 = shortconv.weights.out_proj.shape[1],
                    });
                }
                if (conv_dim == 0 or d_conv == 0) return error.UnsupportedModel;
                if (shortconv.config.d_model != d_model) {
                    log.warn("inference", "CUDA shortconv d_model mismatch", .{
                        .layer = layer_idx,
                        .config_d_model = shortconv.config.d_model,
                        .model_d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }

                var ln1_weight = try uploadTensor(device, allocator, shortconv.ln1_weight);
                errdefer ln1_weight.deinit(device);
                if (!(ln1_weight.rows == d_model and ln1_weight.cols == 1)) {
                    log.warn("inference", "CUDA shortconv ln1 shape unsupported", .{
                        .layer = layer_idx,
                        .rows = ln1_weight.rows,
                        .cols = ln1_weight.cols,
                        .d_model = d_model,
                    });
                    return error.UnsupportedModel;
                }

                var ln2_weight: ?DeviceTensor = null;
                if (shortconv.ln2_weight) |ln2| {
                    var ln2_dev = try uploadTensor(device, allocator, ln2);
                    errdefer ln2_dev.deinit(device);
                    if (!(ln2_dev.rows == d_model and ln2_dev.cols == 1)) {
                        log.warn("inference", "CUDA shortconv ln2 shape unsupported", .{
                            .layer = layer_idx,
                            .rows = ln2_dev.rows,
                            .cols = ln2_dev.cols,
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    ln2_weight = ln2_dev;
                }
                errdefer if (ln2_weight) |*w| w.deinit(device);

                var in_proj_dev = try uploadLinearWeightWithContext(device, allocator, shortconv.weights.in_proj, d_model, layer_idx, "conv.in_proj.weight");
                errdefer in_proj_dev.deinit(device);
                if (in_proj_dev.cols() != 3 * conv_dim) {
                    log.warn("inference", "CUDA shortconv in_proj dim unsupported", .{
                        .layer = layer_idx,
                        .cols = in_proj_dev.cols(),
                        .expected = 3 * conv_dim,
                    });
                    return error.UnsupportedModel;
                }

                var out_proj_dev = try uploadLinearWeightWithContext(device, allocator, shortconv.weights.out_proj, conv_dim, layer_idx, "conv.out_proj.weight");
                errdefer out_proj_dev.deinit(device);
                if (out_proj_dev.cols() != d_model) {
                    log.warn("inference", "CUDA shortconv out_proj dim unsupported", .{
                        .layer = layer_idx,
                        .cols = out_proj_dev.cols(),
                        .expected = d_model,
                    });
                    return error.UnsupportedModel;
                }

                var conv_weight_time_major = try uploadShortConvWeightTimeMajor(
                    device,
                    allocator,
                    shortconv.weights.conv1d_weight,
                    conv_dim,
                    d_conv,
                );
                errdefer conv_weight_time_major.deinit(device);

                var conv_bias: ?DeviceTensor = null;
                if (shortconv.weights.conv1d_bias) |bias| {
                    var bias_dev = try uploadVectorTensor(device, allocator, bias, conv_dim);
                    errdefer bias_dev.deinit(device);
                    conv_bias = bias_dev;
                }
                errdefer if (conv_bias) |*w| w.deinit(device);

                const conv_state_count = std.math.mul(usize, conv_dim, d_conv) catch return error.InvalidArgument;
                var conv_state = try allocZeroedF32Buffer(device, allocator, conv_state_count);
                errdefer conv_state.deinit(device);

                var ffn_w1: ?LinearWeight = null;
                var ffn_w2: ?LinearWeight = null;
                var ffn_w3: ?LinearWeight = null;
                var d_ff: usize = 0;
                if (shortconv.w1 != null or shortconv.w2 != null or shortconv.w3 != null) {
                    const w1 = shortconv.w1 orelse return error.MissingWeight;
                    const w2 = shortconv.w2 orelse return error.MissingWeight;
                    const w3 = shortconv.w3 orelse return error.MissingWeight;
                    if (ln2_weight == null) {
                        log.warn("inference", "CUDA shortconv ffn requires ln2", .{
                            .layer = layer_idx,
                        });
                        return error.UnsupportedModel;
                    }

                    var w1_dev = try uploadLinearWeightWithContext(device, allocator, w1, d_model, layer_idx, "mlp.gate_proj.weight");
                    errdefer w1_dev.deinit(device);
                    var w3_dev = try uploadLinearWeightWithContext(device, allocator, w3, d_model, layer_idx, "mlp.up_proj.weight");
                    errdefer w3_dev.deinit(device);
                    if (w1_dev.cols() != w3_dev.cols()) {
                        log.warn("inference", "CUDA shortconv gate/up dim mismatch", .{
                            .layer = layer_idx,
                            .w1_cols = w1_dev.cols(),
                            .w3_cols = w3_dev.cols(),
                        });
                        return error.UnsupportedModel;
                    }
                    d_ff = w1_dev.cols();
                    var w2_dev = try uploadLinearWeightWithContext(device, allocator, w2, d_ff, layer_idx, "mlp.down_proj.weight");
                    errdefer w2_dev.deinit(device);
                    if (w2_dev.cols() != d_model) {
                        log.warn("inference", "CUDA shortconv down_proj out dim unsupported", .{
                            .layer = layer_idx,
                            .w2_cols = w2_dev.cols(),
                            .d_model = d_model,
                        });
                        return error.UnsupportedModel;
                    }
                    ffn_w1 = w1_dev;
                    ffn_w2 = w2_dev;
                    ffn_w3 = w3_dev;
                }
                errdefer if (ffn_w1) |*w| w.deinit(device);
                errdefer if (ffn_w2) |*w| w.deinit(device);
                errdefer if (ffn_w3) |*w| w.deinit(device);

                const layer_norm_bytes = ln1_weight.byteSize() + (if (ln2_weight) |w| w.byteSize() else 0);
                norm_weight_bytes = std.math.add(usize, norm_weight_bytes, layer_norm_bytes) catch return error.InvalidArgument;

                var layer_linear_bytes = in_proj_dev.byteSize() +
                    out_proj_dev.byteSize() +
                    conv_weight_time_major.byteSize() +
                    (if (conv_bias) |w| w.byteSize() else 0);
                if (ffn_w1) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                if (ffn_w2) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                if (ffn_w3) |w| layer_linear_bytes = std.math.add(usize, layer_linear_bytes, w.byteSize()) catch return error.InvalidArgument;
                linear_weight_bytes = std.math.add(usize, linear_weight_bytes, layer_linear_bytes) catch return error.InvalidArgument;
                shortconv_state_bytes = std.math.add(usize, shortconv_state_bytes, conv_state.size) catch return error.InvalidArgument;
                max_shortconv_dim = @max(max_shortconv_dim, conv_dim);

                blocks[local_idx].shortconv_runtime = .{
                    .conv_dim = conv_dim,
                    .d_conv = d_conv,
                    .d_ff = d_ff,
                    .ln1_weight = ln1_weight,
                    .ln2_weight = ln2_weight,
                    .in_proj = in_proj_dev,
                    .out_proj = out_proj_dev,
                    .conv_weight_time_major = conv_weight_time_major,
                    .conv_bias = conv_bias,
                    .conv_state = conv_state,
                    .ffn_w1 = ffn_w1,
                    .ffn_w2 = ffn_w2,
                    .ffn_w3 = ffn_w3,
                };
                blocks[local_idx].shortconv_binding = &blocks[local_idx].shortconv_runtime.?;
                BlockRuntimeLayer.bindShortConvNormWeights(&blocks[local_idx], &blocks[local_idx].shortconv_runtime.?);
                shortconv_block_count += 1;
            },
            else => {
                log.warn("inference", "CUDA block runtime unsupported block kind", .{
                    .layer = layer_idx,
                });
                return error.UnsupportedModel;
            },
        }
        try blocks[local_idx].rebuildInstructionMetadata(allocator);
        initialized += 1;
    }

    // Resolve cross-device KV sharing: deduplicate sources, assign mirror
    // indices, allocate mirror KV buffers, and fixup consumer blocks.
    var replicated_kv_sources: []ReplicatedKvSource = &.{};
    var mirror_kv: []MirrorKvBuffers = &.{};
    if (pending_mirrors.items.len > 0) {
        var unique_sources: std.ArrayListUnmanaged(ReplicatedKvSource) = .{};
        errdefer unique_sources.deinit(allocator);

        for (pending_mirrors.items) |pm| {
            var found = false;
            for (unique_sources.items) |src| {
                if (src.global_layer_idx == pm.source_global) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                try unique_sources.append(allocator, .{
                    .global_layer_idx = pm.source_global,
                    .kv_dim = pm.kv_dim,
                    .mirror_kv_index = attention_block_count + unique_sources.items.len,
                });
            }
        }

        // Allocate mirror KV buffers at max_seq_len capacity (avoids
        // ensureKvCapacity changes; bandwidth cost is negligible).
        const n_mirrors = unique_sources.items.len;
        var mirrors = try allocator.alloc(MirrorKvBuffers, n_mirrors);
        errdefer allocator.free(mirrors);
        var mirrors_allocated: usize = 0;
        errdefer for (mirrors[0..mirrors_allocated]) |*mk| {
            if (mk.v_scale.pointer != 0) mk.v_scale.deinit(device);
            if (mk.k_scale.pointer != 0) mk.k_scale.deinit(device);
            mk.v.deinit(device);
            mk.k.deinit(device);
        };

        for (unique_sources.items, 0..) |src, mi| {
            const n_mirror_kv_heads: usize = if (src.kv_dim > 0 and head_dim > 0) src.kv_dim / head_dim else n_kv_heads;
            const kv_pair = try allocDeviceKvPairWithScales(device, max_seq_len, src.kv_dim, n_mirror_kv_heads, kv_cache_dtype);
            mirrors[mi] = .{
                .k = kv_pair.k,
                .v = kv_pair.v,
                .k_scale = kv_pair.k_scale,
                .v_scale = kv_pair.v_scale,
                .capacity = max_seq_len,
            };
            mirrors_allocated += 1;
        }

        // Fixup consumer blocks: set kv_shared_source_slot_kv_index to mirror.
        for (pending_mirrors.items) |pm| {
            for (unique_sources.items) |src| {
                if (src.global_layer_idx == pm.source_global) {
                    blocks[pm.local_idx].attention_runtime.?.kv_shared_source_slot_kv_index = src.mirror_kv_index;
                    break;
                }
            }
        }

        replicated_kv_sources = try unique_sources.toOwnedSlice(allocator);
        mirror_kv = mirrors;

        log.info("inference", "KV sharing: allocated mirror entries for cross-device sources", .{
            .n_mirrors = n_mirrors,
            .layer_start = layer_start,
            .layer_end = layer_end,
        });
    }

    return .{
        .blocks = blocks,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .max_seq_len = max_seq_len,
        .attention_block_count = attention_block_count,
        .shortconv_block_count = shortconv_block_count,
        .gated_delta_block_count = gated_delta_block_count,
        .q_norm_blocks = q_norm_blocks,
        .k_norm_blocks = k_norm_blocks,
        .linear_weight_bytes = linear_weight_bytes,
        .norm_weight_bytes = norm_weight_bytes,
        .kv_cache_bytes = kv_cache_bytes,
        .shortconv_state_bytes = shortconv_state_bytes,
        .gated_delta_state_bytes = gated_delta_state_bytes,
        .max_shortconv_dim = max_shortconv_dim,
        .max_gdelta_proj = max_gdelta_proj,
        .replicated_kv_sources = replicated_kv_sources,
        .mirror_kv = mirror_kv,
    };
}

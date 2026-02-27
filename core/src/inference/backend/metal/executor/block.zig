//! Metal backend block executor.
//!
//! Centralizes single-layer lazy graph assembly so model-level orchestration
//! can delegate layer work through a stable `TransformerBlock.forward` surface.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const layer_ops = @import("../../../../models/layer_ops.zig");
const op_types = @import("../../../../models/op_types.zig");
const models = @import("../../../../models/root.zig");
const opcode_map = @import("../../../../models/plan/opcode_map.zig");
const log = @import("../../../../log.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const runtime_graph = @import("../runtime_graph.zig");
const weights_mod = @import("weights.zig");
const attention_kernel = @import("../kernels/attention.zig");
const ffn_kernel = @import("../kernels/ffn.zig");
const mamba_kernel = @import("../kernels/mamba.zig");
const mla_kernel = @import("../kernels/mla_attention.zig");
const moe_kernel = @import("../kernels/moe.zig");
const norm_kernel = @import("../kernels/norm.zig");
const shortconv_kernel = @import("../kernels/shortconv.zig");
const mlx_graph = compute.metal.graph;

pub const Cache = runtime_graph.Cache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
pub const MambaCache = runtime_graph.MambaCache;
const ModelConfig = models.ModelConfig;
const WeightHandles = weights_mod.WeightHandles;
const LayerWeights = WeightHandles.LayerWeights;

pub const TransformerBlock = struct {
    fn finalOutputBuffer(program: []const layer_ops.LayerOp) layer_ops.BufferId {
        return layer_ops.finalOutputBuffer(program);
    }

    const LayerProgramExecutionContext = struct {
        lw: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        norm_index: *usize,
        param_blocks: []const runtime_contract.ParamBlock,
    };

    const LayerProgramStateRef = struct {
        ptr: *anyopaque,
    };

    const LayerProgramInstructionStateBlocks = struct {
        refs: [1]LayerProgramStateRef align(64) = .{.{ .ptr = undefined }},
        handles: [1]runtime_contract.StateBlockHandle = undefined,
        len: usize = 0,

        fn slice(self: *LayerProgramInstructionStateBlocks) []runtime_contract.StateBlockHandle {
            return self.handles[0..self.len];
        }
    };

    const layer_program_required_opcodes = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .shortconv,
        .swiglu,
        .moe,
        .mamba_mixer,
        .residual_add,
    };

    const layer_program_adapter_table: runtime_contract.AdapterTable = blk: {
        var table: runtime_contract.AdapterTable = [_]?runtime_contract.KernelAdapterFn{null} ** 256;
        table[@intFromEnum(opcode_map.Opcode.rmsnorm)] = layerProgramNormRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.multihead_attention)] = layerProgramMixerRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.shortconv)] = layerProgramMixerRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.swiglu)] = layerProgramFfnRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.moe)] = layerProgramFfnRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.mamba_mixer)] = layerProgramMambaRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.residual_add)] = layerProgramResidualAddRuntimeAdapter;
        break :blk table;
    };

    comptime {
        runtime_contract.assertAdapterTableCoverage(
            layer_program_adapter_table,
            layer_program_required_opcodes,
            "metal.executor.block.layer_program_adapter_table",
        );
    }

    fn layerProgramAdapterForOpcode(opcode: opcode_map.Opcode) ?runtime_contract.KernelAdapterFn {
        return layer_program_adapter_table[@intFromEnum(opcode)];
    }

    fn layerProgramExecutionState(ctx: *runtime_contract.ExecutionContext) !*LayerProgramExecutionContext {
        const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
        return @ptrCast(@alignCast(raw_state));
    }

    fn decodeInstructionLayerOp(
        insn: *const runtime_contract.Instruction,
        params: []const runtime_contract.ParamBlock,
    ) !layer_ops.LayerOp {
        const param_block_id = insn.param_block_id orelse return error.MissingParamBlock;
        if (param_block_id >= params.len) return error.MissingParamBlock;
        const param_block = params[param_block_id];
        try runtime_contract.validateParamBlockAbi(&param_block);
        if (param_block.opcode != insn.opcode) return error.ParamBlockOpcodeMismatch;
        if (param_block.data.len != @sizeOf(layer_ops.LayerOp)) return error.InvalidParamBlockSize;
        const op_ptr: *const layer_ops.LayerOp = @ptrCast(@alignCast(param_block.data.ptr));
        return op_ptr.*;
    }

    fn layerProgramStateBlocksForInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !LayerProgramInstructionStateBlocks {
        var blocks = LayerProgramInstructionStateBlocks{};
        const state_id = insn.state_block_id orelse return blocks;
        const state_ptr: *anyopaque = switch (state_id) {
            @intFromEnum(runtime_contract.StateBlockId.kv_cache) => blk: {
                if (ctx.cache) |*cache| break :blk cache;
                return error.InvalidStateDescriptorBinding;
            },
            @intFromEnum(runtime_contract.StateBlockId.shortconv) => blk: {
                if (ctx.shortconv_cache) |*shortconv_cache| break :blk shortconv_cache;
                return error.InvalidStateDescriptorBinding;
            },
            @intFromEnum(runtime_contract.StateBlockId.mamba) => blk: {
                if (ctx.mamba_cache) |*mamba_cache| break :blk mamba_cache;
                return error.InvalidStateDescriptorBinding;
            },
            else => return error.InvalidStateDescriptorBinding,
        };

        blocks.refs[0] = .{ .ptr = state_ptr };
        blocks.handles[0] = .{
            .id = state_id,
            .ptr = @ptrCast(&blocks.refs[0]),
            .size = @sizeOf(LayerProgramStateRef),
            .align_bytes = 64,
        };
        blocks.len = 1;
        return blocks;
    }

    fn validateLayerProgram(program: []const layer_ops.LayerOp, layer_idx: usize, kind: op_types.BlockKind) !void {
        if (runtime_contract.firstUnsupportedLayerProgramOpcode(program, layer_program_adapter_table)) |unsupported| {
            log.warn("inference", "Metal LayerOp program contains unsupported opcode", .{
                .layer = layer_idx,
                .op_index = unsupported.op_index,
                .kind = @intFromEnum(kind),
                .op = @tagName(program[unsupported.op_index]),
                .opcode = @intFromEnum(unsupported.opcode),
            });
            return error.NotImplemented;
        }
        if (runtime_contract.firstLayerProgramStateMismatch(program, kind)) |mismatch| {
            log.warn("inference", "Metal LayerOp program state binding mismatches block kind", .{
                .layer = layer_idx,
                .op_index = mismatch.op_index,
                .kind = @intFromEnum(kind),
                .op = @tagName(program[mismatch.op_index]),
                .opcode = @intFromEnum(mismatch.opcode),
                .state_id = mismatch.state_id,
            });
            return error.NotImplemented;
        }
    }

    fn getBuffer(
        buffer_id: layer_ops.BufferId,
        residual: mlx_graph.ArrayHandle,
        slot_buffers: *const [2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
    ) !mlx_graph.ArrayHandle {
        if (buffer_id == .residual) return residual;
        const register_idx = @intFromEnum(buffer_id);
        if (register_idx >= register_to_slot_map.len) return error.NotImplemented;
        const slot_idx = register_to_slot_map[register_idx];
        if (slot_idx == std.math.maxInt(u8) or slot_idx >= slot_buffers.len) return error.NotImplemented;
        return slot_buffers[slot_idx];
    }

    fn setBuffer(
        buffer_id: layer_ops.BufferId,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        value: mlx_graph.ArrayHandle,
    ) !void {
        if (buffer_id == .residual) {
            residual.* = value;
            return;
        }
        const register_idx = @intFromEnum(buffer_id);
        if (register_idx >= register_to_slot_map.len) return error.NotImplemented;
        const slot_idx = register_to_slot_map[register_idx];
        if (slot_idx == std.math.maxInt(u8) or slot_idx >= slot_buffers.len) return error.NotImplemented;
        slot_buffers[slot_idx] = value;
    }

    fn nextNormWeight(lw: *const LayerWeights, norm_index: *usize) !mlx_graph.ArrayHandle {
        const idx = norm_index.*;
        norm_index.* = idx + 1;
        return switch (idx) {
            0 => lw.getLn1(),
            1 => lw.getLn2(),
            // Gemma-style 4-norm blocks use pre/post-FFN norms.
            2 => lw.getPreFfnNorm() orelse lw.getPostFfnNorm() orelse error.InvalidState,
            3 => lw.getPostFfnNorm() orelse error.InvalidState,
            else => error.InvalidState,
        };
    }

    fn residualScale(
        scale: layer_ops.ResidualScale,
        weight_handles: *const WeightHandles,
    ) f32 {
        return switch (scale) {
            .one => 1.0,
            .residual_multiplier => weight_handles.residual_multiplier,
            .literal => |value| value,
        };
    }

    fn runMixerKernel(
        input: mlx_graph.ArrayHandle,
        lw: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const norm_eps = config.norm_eps;
        const head_count: usize = @intCast(config.n_heads);
        const kv_head_count: usize = @intCast(config.n_kv_groups);
        const head_dim: usize = @intCast(config.head_dim);
        const attention_storage = lw.attentionStorageKind();
        const shortconv_storage = lw.shortconvStorageKind();

        return switch (lw.kind) {
            .attention_mlp => blk: {
                if (lw.isMLA()) {
                    const mla_cfg = lw.mla_config orelse return error.MissingField;
                    const mla_attention = mla_kernel.MLAttention{
                        .n_heads = head_count,
                        .rope_theta = config.rope_theta,
                        .norm_eps = norm_eps,
                        .q_lora_rank = mla_cfg.q_lora_rank,
                        .kv_lora_rank = mla_cfg.kv_lora_rank,
                        .qk_head_dim = mla_cfg.qk_head_dim,
                        .qk_rope_head_dim = mla_cfg.qk_rope_head_dim,
                        .qk_nope_head_dim = mla_cfg.qk_nope_head_dim,
                        .v_head_dim = mla_cfg.v_head_dim,
                        .q_a_proj = lw.mla_q_a_proj,
                        .q_b_proj = lw.mla_q_b_proj,
                        .kv_a_proj = lw.mla_kv_a_proj,
                        .kv_b_proj = lw.mla_kv_b_proj,
                        .q_a_proj_bf16 = lw.mla_q_a_proj_bf16,
                        .q_b_proj_bf16 = lw.mla_q_b_proj_bf16,
                        .kv_a_proj_bf16 = lw.mla_kv_a_proj_bf16,
                        .kv_b_proj_bf16 = lw.mla_kv_b_proj_bf16,
                        .q_a_norm = lw.mla_q_a_norm,
                        .kv_a_norm = lw.mla_kv_a_norm,
                        .o_proj = lw.o_proj,
                        .o_proj_bf16 = lw.o_proj_bf16,
                    };
                    var mla_cache = mla_kernel.AttnCache{
                        .cache = cache,
                        .layer_idx = layer_idx,
                        .pos_offset = pos_offset,
                    };
                    var mla_scratch = mla_kernel.AttnTemp{
                        .runtime_rope_cos_handle = runtime_rope_cos_handle,
                        .runtime_rope_sin_handle = runtime_rope_sin_handle,
                        .runtime_rope_dim = runtime_rope_dim,
                    };
                    var mla_matmul_scratch = mla_kernel.MatmulScratch{};
                    var mla_out: mlx_graph.ArrayHandle = undefined;
                    try mla_attention.forward(
                        input,
                        &mla_out,
                        &mla_cache,
                        &mla_scratch,
                        &mla_matmul_scratch,
                        cache != null,
                    );
                    break :blk mla_out;
                }

                if (attention_storage == .invalid) return error.InvalidTensorType;
                if (attention_storage == .missing) return error.MissingField;
                const attention = attention_kernel.MultiHeadAttention{
                    .n_heads = head_count,
                    .n_kv_heads = kv_head_count,
                    .head_dim = head_dim,
                    .rope_theta = config.rope_theta,
                    .norm_eps = norm_eps,
                    .query_pre_attn_scalar = config.query_pre_attn_scalar,
                    .attention_multiplier = weight_handles.attention_multiplier,
                    .q_proj = lw.q_proj,
                    .k_proj = lw.k_proj,
                    .v_proj = lw.v_proj,
                    .o_proj = lw.o_proj,
                    .q_proj_bf16 = lw.q_proj_bf16,
                    .k_proj_bf16 = lw.k_proj_bf16,
                    .v_proj_bf16 = lw.v_proj_bf16,
                    .o_proj_bf16 = lw.o_proj_bf16,
                    .q_norm = lw.q_norm,
                    .k_norm = lw.k_norm,
                    .q_bias = lw.q_bias,
                    .k_bias = lw.k_bias,
                    .v_bias = lw.v_bias,
                    .o_bias = lw.o_bias,
                    .attn_sinks = lw.attn_sinks,
                };
                var attn_cache = attention_kernel.AttnCache{
                    .cache = cache,
                    .layer_idx = layer_idx,
                    .pos_offset = pos_offset,
                };
                var attn_scratch = attention_kernel.AttnTemp{
                    .runtime_rope_cos_handle = runtime_rope_cos_handle,
                    .runtime_rope_sin_handle = runtime_rope_sin_handle,
                    .runtime_rope_dim = runtime_rope_dim,
                };
                var attn_matmul_scratch = attention_kernel.MatmulScratch{};
                var attn_out: mlx_graph.ArrayHandle = undefined;
                try attention.forward(
                    input,
                    &attn_out,
                    &attn_cache,
                    &attn_scratch,
                    &attn_matmul_scratch,
                    cache != null,
                );
                break :blk attn_out;
            },
            .shortconv => blk: {
                if (shortconv_storage == .invalid) return error.InvalidTensorType;
                if (shortconv_storage == .missing) return error.MissingField;
                const conv_weight = lw.shortconv_conv_weight orelse return error.MissingField;
                const shortconv = shortconv_kernel.ShortConvKernel{
                    .in_proj = if (lw.shortconv_in_proj) |w| w else null,
                    .out_proj = if (lw.shortconv_out_proj) |w| w else null,
                    .in_proj_bf16 = if (lw.shortconv_in_proj_bf16) |h| h else null,
                    .out_proj_bf16 = if (lw.shortconv_out_proj_bf16) |h| h else null,
                    .conv_weight = conv_weight,
                    .conv_bias = if (lw.shortconv_conv_bias) |b| b else null,
                    .d_conv = lw.shortconv_d_conv,
                    .conv_dim = lw.shortconv_conv_dim,
                };
                var shortconv_state = shortconv_kernel.ShortConvState{
                    .cache = shortconv_cache,
                    .layer_idx = layer_idx,
                };
                var shortconv_scratch = shortconv_kernel.ShortConvScratch{};
                var shortconv_matmul_scratch = shortconv_kernel.MatmulScratch{};
                var shortconv_out: mlx_graph.ArrayHandle = undefined;
                try shortconv.forward(
                    input,
                    &shortconv_out,
                    &shortconv_state,
                    &shortconv_scratch,
                    &shortconv_matmul_scratch,
                );
                break :blk shortconv_out;
            },
            .mamba => blk: {
                const conv_weight = lw.mamba_conv_weight orelse return error.MissingField;
                const a_log = lw.mamba_a_log orelse return error.MissingField;
                const d_skip = lw.mamba_d_skip orelse return error.MissingField;
                const mamba = mamba_kernel.MambaKernel{
                    .d_state = lw.mamba_d_state,
                    .d_conv = lw.mamba_d_conv,
                    .n_heads = lw.mamba_n_heads,
                    .d_head = lw.mamba_d_head,
                    .n_groups = lw.mamba_n_groups,
                    .use_gelu = weight_handles.use_gelu,
                    .residual_multiplier = weight_handles.residual_multiplier,
                    .norm_eps = norm_eps,
                    .gate_up_layout = @intFromEnum(lw.mamba_gate_up_layout),
                    .ln1_weight = lw.getLn1(),
                    .in_proj = lw.mamba_in_proj,
                    .in_proj_bf16 = lw.mamba_in_proj_bf16,
                    .conv_weight = conv_weight,
                    .conv_bias = lw.mamba_conv_bias,
                    .a_log = a_log,
                    .d_skip = d_skip,
                    .dt_bias = lw.mamba_dt_bias,
                    .norm_weight = lw.mamba_norm_weight,
                    .out_proj = lw.mamba_out_proj,
                    .out_proj_bf16 = lw.mamba_out_proj_bf16,
                    .ln2_weight = lw.getLn2(),
                    .gate_up = lw.mamba_gate_up,
                    .gate_up_bf16 = lw.mamba_gate_up_bf16,
                    .down_proj = lw.mamba_down_proj,
                    .down_proj_bf16 = lw.mamba_down_proj_bf16,
                };
                var m_state = mamba_kernel.MambaState{
                    .cache = mamba_cache,
                    .layer_idx = layer_idx,
                };
                var m_scratch = mamba_kernel.MambaScratch{};
                var m_matmul = mamba_kernel.MatmulScratch{};
                var m_out: mlx_graph.ArrayHandle = undefined;
                try mamba.forward(
                    input,
                    &m_out,
                    &m_state,
                    &m_scratch,
                    &m_matmul,
                );
                break :blk m_out;
            },
        };
    }

    fn runFfnKernel(
        input: mlx_graph.ArrayHandle,
        lw: *const LayerWeights,
        weight_handles: *const WeightHandles,
    ) !mlx_graph.ArrayHandle {
        const ffn_storage = lw.ffnStorageKind();
        return switch (ffn_storage) {
            .moe => blk: {
                const moe = lw.moe orelse return error.MissingField;
                const ffn_moe = moe_kernel.MoEFFN{ .weights = moe };
                var moe_scratch = moe_kernel.MoEScratch{};
                var moe_matmul_scratch = moe_kernel.MatmulScratch{};
                var moe_out: mlx_graph.ArrayHandle = undefined;
                try ffn_moe.forward(
                    input,
                    &moe_out,
                    &moe_scratch,
                    &moe_matmul_scratch,
                );
                break :blk moe_out;
            },
            .quantized, .dense => blk: {
                const swiglu = ffn_kernel.SwiGLU{
                    .use_gelu = weight_handles.use_gelu,
                    .w1 = lw.w1,
                    .w2 = lw.w2,
                    .w3 = lw.w3,
                    .w1_bf16 = lw.w1_bf16,
                    .w2_bf16 = lw.w2_bf16,
                    .w3_bf16 = lw.w3_bf16,
                };
                var ffn_scratch = ffn_kernel.FfnScratch{};
                var ffn_matmul_scratch = ffn_kernel.MatmulScratch{};
                var ffn_result: mlx_graph.ArrayHandle = undefined;
                try swiglu.forward(
                    input,
                    &ffn_result,
                    &ffn_scratch,
                    &ffn_matmul_scratch,
                );
                break :blk ffn_result;
            },
            .missing => error.MissingField,
            .invalid => error.InvalidTensorType,
        };
    }

    fn layerProgramNormAdapter(
        op: layer_ops.LayerOp,
        lw: *const LayerWeights,
        _: usize,
        config: ModelConfig,
        _: *const WeightHandles,
        _: ?Cache,
        _: ?ShortConvCache,
        _: ?MambaCache,
        _: usize,
        _: mlx_graph.ArrayHandle,
        _: mlx_graph.ArrayHandle,
        _: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        norm_index: *usize,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .norm) return error.NotImplemented;
        const input = try getBuffer(kernel_op.in, residual.*, slot_buffers, register_to_slot_map);
        var output: mlx_graph.ArrayHandle = undefined;
        const norm = norm_kernel.RMSNorm{
            .weight = try nextNormWeight(lw, norm_index),
            .eps = config.norm_eps,
        };
        norm.forward(input, &output);
        try setBuffer(kernel_op.out, residual, slot_buffers, register_to_slot_map, output);
    }

    fn layerProgramMixerAdapter(
        op: layer_ops.LayerOp,
        lw: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        _: *usize,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .multihead_attention and kernel_op.debug_type != .shortconv) {
            return error.NotImplemented;
        }
        const input = try getBuffer(kernel_op.in, residual.*, slot_buffers, register_to_slot_map);
        const output = try runMixerKernel(
            input,
            lw,
            layer_idx,
            config,
            weight_handles,
            cache,
            shortconv_cache,
            mamba_cache,
            pos_offset,
            runtime_rope_cos_handle,
            runtime_rope_sin_handle,
            runtime_rope_dim,
        );
        try setBuffer(kernel_op.out, residual, slot_buffers, register_to_slot_map, output);
    }

    fn layerProgramFfnAdapter(
        op: layer_ops.LayerOp,
        lw: *const LayerWeights,
        _: usize,
        _: ModelConfig,
        weight_handles: *const WeightHandles,
        _: ?Cache,
        _: ?ShortConvCache,
        _: ?MambaCache,
        _: usize,
        _: mlx_graph.ArrayHandle,
        _: mlx_graph.ArrayHandle,
        _: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        _: *usize,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .mlp and kernel_op.debug_type != .moe) return error.NotImplemented;
        const input = try getBuffer(kernel_op.in, residual.*, slot_buffers, register_to_slot_map);
        const output = try runFfnKernel(input, lw, weight_handles);
        try setBuffer(kernel_op.out, residual, slot_buffers, register_to_slot_map, output);
    }

    fn layerProgramMambaAdapter(
        op: layer_ops.LayerOp,
        lw: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        _: *usize,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .mamba_mixer) return error.NotImplemented;
        const input = try getBuffer(kernel_op.in, residual.*, slot_buffers, register_to_slot_map);
        const output = try runMixerKernel(
            input,
            lw,
            layer_idx,
            config,
            weight_handles,
            cache,
            shortconv_cache,
            mamba_cache,
            pos_offset,
            runtime_rope_cos_handle,
            runtime_rope_sin_handle,
            runtime_rope_dim,
        );
        try setBuffer(kernel_op.out, residual, slot_buffers, register_to_slot_map, output);
    }

    fn layerProgramResidualAddAdapter(
        op: layer_ops.LayerOp,
        _: *const LayerWeights,
        _: usize,
        _: ModelConfig,
        weight_handles: *const WeightHandles,
        _: ?Cache,
        _: ?ShortConvCache,
        _: ?MambaCache,
        _: usize,
        _: mlx_graph.ArrayHandle,
        _: mlx_graph.ArrayHandle,
        _: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
        _: *usize,
    ) !void {
        const add_op = switch (op) {
            .add => |add| add,
            else => return error.NotImplemented,
        };
        const branch = try getBuffer(add_op.branch, residual.*, slot_buffers, register_to_slot_map);
        const scale = residualScale(add_op.scale, weight_handles);
        const scaled_branch = if (scale == 1.0)
            branch
        else
            mlx_graph.mlx_lazy_multiply_scalar(branch, scale);
        residual.* = mlx_graph.mlx_lazy_add(residual.*, scaled_branch);
    }

    fn layerProgramNormRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        _ = try runtime_contract.requireInstructionStateBlock(insn, state_blocks);
        const state = try layerProgramExecutionState(ctx);
        const op = try decodeInstructionLayerOp(insn, params);
        try layerProgramNormAdapter(
            op,
            state.lw,
            state.layer_idx,
            state.config,
            state.weight_handles,
            state.cache,
            state.shortconv_cache,
            state.mamba_cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
            state.residual,
            state.slot_buffers,
            state.register_to_slot_map,
            state.norm_index,
        );
    }

    fn layerProgramMixerRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        _ = try runtime_contract.requireInstructionStateBlock(insn, state_blocks);
        const state = try layerProgramExecutionState(ctx);
        const op = try decodeInstructionLayerOp(insn, params);
        try layerProgramMixerAdapter(
            op,
            state.lw,
            state.layer_idx,
            state.config,
            state.weight_handles,
            state.cache,
            state.shortconv_cache,
            state.mamba_cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
            state.residual,
            state.slot_buffers,
            state.register_to_slot_map,
            state.norm_index,
        );
    }

    fn layerProgramFfnRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        _ = try runtime_contract.requireInstructionStateBlock(insn, state_blocks);
        const state = try layerProgramExecutionState(ctx);
        const op = try decodeInstructionLayerOp(insn, params);
        try layerProgramFfnAdapter(
            op,
            state.lw,
            state.layer_idx,
            state.config,
            state.weight_handles,
            state.cache,
            state.shortconv_cache,
            state.mamba_cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
            state.residual,
            state.slot_buffers,
            state.register_to_slot_map,
            state.norm_index,
        );
    }

    fn layerProgramMambaRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        _ = try runtime_contract.requireInstructionStateBlock(insn, state_blocks);
        const state = try layerProgramExecutionState(ctx);
        const op = try decodeInstructionLayerOp(insn, params);
        try layerProgramMambaAdapter(
            op,
            state.lw,
            state.layer_idx,
            state.config,
            state.weight_handles,
            state.cache,
            state.shortconv_cache,
            state.mamba_cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
            state.residual,
            state.slot_buffers,
            state.register_to_slot_map,
            state.norm_index,
        );
    }

    fn layerProgramResidualAddRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        _: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        _ = try runtime_contract.requireInstructionStateBlock(insn, state_blocks);
        const state = try layerProgramExecutionState(ctx);
        const op = try decodeInstructionLayerOp(insn, params);
        try layerProgramResidualAddAdapter(
            op,
            state.lw,
            state.layer_idx,
            state.config,
            state.weight_handles,
            state.cache,
            state.shortconv_cache,
            state.mamba_cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
            state.residual,
            state.slot_buffers,
            state.register_to_slot_map,
            state.norm_index,
        );
    }

    fn dispatchLayerProgramInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        if (insn.weights.len != 0) return error.InvalidWeightRefCount;
        const adapter = layerProgramAdapterForOpcode(insn.opcode) orelse return error.NotImplemented;
        const active_slots: [1]usize = .{0};
        const no_seq_lengths: [0]u32 = .{};
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .decode,
            .active_slots = active_slots[0..],
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = 1,
            .workspace = .{ .any = @ptrCast(ctx) },
        };
        const no_registers: [0]runtime_contract.TensorHandle = .{};
        const no_views: [0]runtime_contract.TensorViewDesc = .{};
        var state_blocks = try layerProgramStateBlocksForInstruction(insn, ctx);
        _ = try runtime_contract.requireInstructionStateBlock(insn, state_blocks.slice());
        try adapter(
            &rt_ctx,
            insn,
            no_registers[0..],
            no_views[0..],
            state_blocks.slice(),
            ctx.param_blocks,
        );
    }

    fn forwardWithProgram(
        hidden: mlx_graph.ArrayHandle,
        lw: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const compiled_plan = lw.compiled_plan orelse return error.NotImplemented;
        var residual = hidden;
        var slot_buffers: [2]mlx_graph.ArrayHandle = .{ hidden, hidden };
        var norm_index: usize = 0;
        var exec_ctx = LayerProgramExecutionContext{
            .lw = lw,
            .layer_idx = layer_idx,
            .config = config,
            .weight_handles = weight_handles,
            .cache = cache,
            .shortconv_cache = shortconv_cache,
            .mamba_cache = mamba_cache,
            .pos_offset = pos_offset,
            .runtime_rope_cos_handle = runtime_rope_cos_handle,
            .runtime_rope_sin_handle = runtime_rope_sin_handle,
            .runtime_rope_dim = runtime_rope_dim,
            .residual = &residual,
            .slot_buffers = &slot_buffers,
            .register_to_slot_map = &lw.register_to_slot_map,
            .norm_index = &norm_index,
            .param_blocks = compiled_plan.param_blocks,
        };

        for (compiled_plan.plan.instructions) |insn| {
            try dispatchLayerProgramInstruction(&insn, &exec_ctx);
        }

        const final_register = runtime_contract.planFinalOutputRegister(&compiled_plan.plan);
        const final_register_idx = runtime_contract.registerToIndex(final_register);
        if (final_register_idx > @intFromEnum(layer_ops.BufferId.tmp63)) return error.NotImplemented;
        const final_buffer_id: layer_ops.BufferId = @enumFromInt(final_register_idx);
        return getBuffer(final_buffer_id, residual, &slot_buffers, &lw.register_to_slot_map);
    }

    pub fn forward(
        hidden: mlx_graph.ArrayHandle,
        layer_weights: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        cache: ?Cache,
        shortconv_cache: ?ShortConvCache,
        mamba_cache: ?MambaCache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const lw = layer_weights;

        if (lw.program) |program| {
            try validateLayerProgram(program, layer_idx, lw.kind);
            return forwardWithProgram(
                hidden,
                lw,
                layer_idx,
                config,
                weight_handles,
                cache,
                shortconv_cache,
                mamba_cache,
                pos_offset,
                runtime_rope_cos_handle,
                runtime_rope_sin_handle,
                runtime_rope_dim,
            );
        }
        log.warn("inference", "Metal block missing LayerOp program", .{
            .layer = layer_idx,
            .kind = @intFromEnum(lw.kind),
        });
        return error.UnsupportedModel;
    }

    pub fn projectLogits(
        hidden: mlx_graph.ArrayHandle,
        weight_handles: anytype,
        norm_eps: f32,
    ) mlx_graph.ArrayHandle {
        const final_normed = projectHidden(hidden, weight_handles, norm_eps);
        const logits = if (weight_handles.lm_head_quantized) |quantized_lm_head| blk: {
            break :blk mlx_graph.mlx_lazy_quantized_matmul(
                final_normed,
                quantized_lm_head.weights,
                quantized_lm_head.scales,
                quantized_lm_head.biases,
                quantized_lm_head.group_size,
                quantized_lm_head.bits,
                true,
            );
        } else blk: {
            if (weight_handles.lm_head_needs_transpose) {
                const transpose_axes = [_]usize{ 1, 0 };
                const lm_head_t = mlx_graph.mlx_lazy_transpose(weight_handles.lm_head.?, &transpose_axes, 2);
                break :blk mlx_graph.mlx_lazy_matmul(final_normed, lm_head_t);
            } else {
                break :blk mlx_graph.mlx_lazy_matmul(final_normed, weight_handles.lm_head.?);
            }
        };
        return if (weight_handles.logits_scaling != 1.0)
            mlx_graph.mlx_lazy_multiply_scalar(logits, 1.0 / weight_handles.logits_scaling)
        else
            logits;
    }

    pub fn projectHidden(
        hidden: mlx_graph.ArrayHandle,
        weight_handles: anytype,
        norm_eps: f32,
    ) mlx_graph.ArrayHandle {
        const final_norm = norm_kernel.RMSNorm{
            .weight = weight_handles.ln_final,
            .eps = norm_eps,
        };
        var final_normed: mlx_graph.ArrayHandle = undefined;
        final_norm.forward(hidden, &final_normed);
        return final_normed;
    }
};

test "finalOutputBuffer returns residual when program ends with add" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .multihead_attention,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.residual, TransformerBlock.finalOutputBuffer(&program));
}

test "finalOutputBuffer returns kernel output buffer for post-norm endings" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
    };
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, TransformerBlock.finalOutputBuffer(&program));
}

test "validateLayerProgram accepts kernel-add programs" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .norm_out,
            .debug_type = .norm,
        } },
        .{ .kernel = .{
            .id = 1,
            .in = .norm_out,
            .out = .branch_out,
            .debug_type = .mlp,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try TransformerBlock.validateLayerProgram(&program, 0, .attention_mlp);
}

test "layerProgramAdapterForOpcode covers Metal LayerOp execution subset" {
    const supported = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .residual_add,
    };
    for (supported) |opcode| {
        try std.testing.expect(TransformerBlock.layerProgramAdapterForOpcode(opcode) != null);
    }

    try std.testing.expect(TransformerBlock.layerProgramAdapterForOpcode(.mul_scalar) == null);
    try std.testing.expect(TransformerBlock.layerProgramAdapterForOpcode(.vision_patch_embed) == null);
}

test "validateLayerProgram rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    try std.testing.expectError(error.NotImplemented, TransformerBlock.validateLayerProgram(&program, 0, .attention_mlp));
}

test "validateLayerProgram rejects stateful opcode bound to wrong block kind" {
    const program = [_]layer_ops.LayerOp{
        .{ .kernel = .{
            .id = 0,
            .in = .residual,
            .out = .branch_out,
            .debug_type = .shortconv,
        } },
        .{ .add = .{
            .branch = .branch_out,
            .scale = .one,
        } },
    };
    try std.testing.expectError(
        error.NotImplemented,
        TransformerBlock.validateLayerProgram(&program, 0, .attention_mlp),
    );
}

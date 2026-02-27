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
var layer_program_dispatch_counters = runtime_contract.DispatchCounters{};

pub const Cache = runtime_graph.Cache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
pub const MambaCache = runtime_graph.MambaCache;
const ModelConfig = models.ModelConfig;
const WeightHandles = weights_mod.WeightHandles;
const LayerWeights = WeightHandles.LayerWeights;

pub const TransformerBlock = struct {
    const MaxLayerProgramStateBindings = 256;

    fn finalOutputBuffer(program: []const layer_ops.LayerOp) layer_ops.BufferId {
        return layer_ops.finalOutputBuffer(program);
    }

    fn planUsesOpcode(plan: *const runtime_contract.ExecutionPlan, opcode: opcode_map.Opcode) bool {
        for (plan.instructions) |insn| {
            if (insn.opcode == opcode) return true;
        }
        return false;
    }

    const AttentionRuntimeBinding = union(enum) {
        mla: mla_kernel.MLAttention,
        multihead: attention_kernel.MultiHeadAttention,
    };

    const FfnRuntimeBinding = union(enum) {
        moe: moe_kernel.MoEFFN,
        dense: ffn_kernel.SwiGLU,
    };

    const LayerProgramRuntimeBindings = struct {
        norm_eps: f32 = 1.0e-6,
        residual_multiplier: f32 = 1.0,
        norm_weights: [4]?mlx_graph.ArrayHandle = [_]?mlx_graph.ArrayHandle{null} ** 4,
        norm_weight_count: u8 = 0,
        attention: ?AttentionRuntimeBinding = null,
        shortconv: ?shortconv_kernel.ShortConvKernel = null,
        ffn: ?FfnRuntimeBinding = null,
        mamba: ?mamba_kernel.MambaKernel = null,
    };

    const LayerProgramStateBinding = struct {
        id: u8,
        ptr: *anyopaque,
    };

    const LayerProgramExecutionContext = struct {
        compiled_plan: *const runtime_contract.CompiledPlan,
        instruction_ops: []const layer_ops.LayerOp,
        op_index: usize,
        layer_idx: usize,
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
        bindings: LayerProgramRuntimeBindings,
        state_bindings: [MaxLayerProgramStateBindings]?LayerProgramStateBinding = [_]?LayerProgramStateBinding{null} ** MaxLayerProgramStateBindings,
        state_binding_count: u8 = 0,
    };
    const MAX_LAYER_PROGRAM_HANDLES: usize = 8;
    const IMPLICIT_BINDING_REGISTER_BASE: u16 = 0xF000;

    const BuiltLayerProgramHandles = struct {
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
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

    fn bindLayerProgramState(
        ctx: *LayerProgramExecutionContext,
        state_id: u8,
        ptr: *anyopaque,
    ) !void {
        var idx: usize = 0;
        while (idx < ctx.state_binding_count) : (idx += 1) {
            const existing = ctx.state_bindings[idx] orelse continue;
            if (existing.id == state_id) {
                ctx.state_bindings[idx] = .{ .id = state_id, .ptr = ptr };
                return;
            }
        }
        if (ctx.state_binding_count >= ctx.state_bindings.len) return error.InvalidStateDescriptorBinding;
        ctx.state_bindings[ctx.state_binding_count] = .{ .id = state_id, .ptr = ptr };
        ctx.state_binding_count += 1;
    }

    fn layerProgramStateBinding(
        ctx: *const LayerProgramExecutionContext,
        state_id: u8,
    ) ?LayerProgramStateBinding {
        var idx: usize = 0;
        while (idx < ctx.state_binding_count) : (idx += 1) {
            const binding = ctx.state_bindings[idx] orelse continue;
            if (binding.id == state_id) return binding;
        }
        return null;
    }

    fn bindLayerProgramStateDescriptors(
        ctx: *LayerProgramExecutionContext,
        plan: *const runtime_contract.ExecutionPlan,
    ) !void {
        ctx.state_bindings = [_]?LayerProgramStateBinding{null} ** MaxLayerProgramStateBindings;
        ctx.state_binding_count = 0;
        for (plan.state_descs) |state_desc| {
            if (state_desc.lifecycle != .slot_persistent) return error.InvalidStateDescriptorBinding;
            switch (state_desc.id) {
                @intFromEnum(runtime_contract.StateBlockId.kv_cache) => {
                    if (ctx.cache) |*cache_state| {
                        try bindLayerProgramState(ctx, state_desc.id, cache_state);
                    } else return error.InvalidStateDescriptorBinding;
                },
                @intFromEnum(runtime_contract.StateBlockId.shortconv) => {
                    if (ctx.shortconv_cache) |*shortconv_state| {
                        try bindLayerProgramState(ctx, state_desc.id, shortconv_state);
                    } else return error.InvalidStateDescriptorBinding;
                },
                @intFromEnum(runtime_contract.StateBlockId.mamba) => {
                    if (ctx.mamba_cache) |*mamba_state| {
                        try bindLayerProgramState(ctx, state_desc.id, mamba_state);
                    } else return error.InvalidStateDescriptorBinding;
                },
                else => return error.InvalidStateDescriptorBinding,
            }
        }
    }

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
        table[@intFromEnum(opcode_map.Opcode.multihead_attention)] = layerProgramAttentionRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.shortconv)] = layerProgramShortConvRuntimeAdapter;
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

    fn layerProgramOpForInstruction(
        ctx: *const LayerProgramExecutionContext,
        _: *const runtime_contract.Instruction,
        instruction_index: usize,
    ) !layer_ops.LayerOp {
        if (instruction_index >= ctx.instruction_ops.len) return error.InvalidInstructionIndex;
        return ctx.instruction_ops[instruction_index];
    }

    fn layerProgramStateBlocksForInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !LayerProgramInstructionStateBlocks {
        var blocks = LayerProgramInstructionStateBlocks{};
        const state_id = insn.state_block_id orelse return blocks;
        const binding = layerProgramStateBinding(ctx, state_id) orelse return error.InvalidStateDescriptorBinding;
        const descriptor = runtime_contract.findStateDescriptor(&ctx.compiled_plan.plan, state_id) orelse {
            return error.UnknownStateDescriptorId;
        };

        blocks.refs[0] = .{ .ptr = binding.ptr };
        blocks.handles[0] = .{
            .id = state_id,
            .ptr = @ptrCast(&blocks.refs[0]),
            .size = if (descriptor.size_bytes > 0) descriptor.size_bytes else @sizeOf(LayerProgramStateRef),
            .align_bytes = descriptor.align_bytes,
        };
        blocks.len = 1;
        return blocks;
    }

    fn validateCompiledLayerProgram(lw: *const LayerWeights, layer_idx: usize) !void {
        const compiled_plan = lw.compiled_plan orelse return error.NotImplemented;
        if (lw.instruction_ops.len != compiled_plan.plan.instructions.len) {
            log.warn("inference", "Metal layer plan instruction cache mismatch", .{
                .layer = layer_idx,
                .kind = @intFromEnum(lw.kind),
                .plan_len = compiled_plan.plan.instructions.len,
                .cache_len = lw.instruction_ops.len,
            });
            return error.NotImplemented;
        }
        runtime_contract.validateExecutionPlanForBlockKind(&compiled_plan.plan, lw.kind) catch |err| {
            log.warn("inference", "Metal compiled layer plan fails block-kind validation", .{
                .layer = layer_idx,
                .kind = @intFromEnum(lw.kind),
                .reason = @errorName(err),
            });
            return error.NotImplemented;
        };
        if (runtime_contract.firstUnsupportedInstructionOpcode(&compiled_plan.plan, layer_program_adapter_table)) |unsupported| {
            log.warn("inference", "Metal compiled layer plan contains unsupported opcode", .{
                .layer = layer_idx,
                .op_index = unsupported.instruction_index,
                .kind = @intFromEnum(lw.kind),
                .opcode = @intFromEnum(unsupported.opcode),
            });
            return error.NotImplemented;
        }
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

    fn bufferSlotForRegister(
        reg: runtime_contract.RegisterRef,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: *[2]mlx_graph.ArrayHandle,
        register_to_slot_map: *const [64]u8,
    ) !*mlx_graph.ArrayHandle {
        const buffer_id: layer_ops.BufferId = @enumFromInt(runtime_contract.registerToIndex(reg));
        if (buffer_id == .residual) return residual;
        const register_idx = @intFromEnum(buffer_id);
        if (register_idx >= register_to_slot_map.len) return error.NotImplemented;
        const slot_idx = register_to_slot_map[register_idx];
        if (slot_idx == std.math.maxInt(u8) or slot_idx >= slot_buffers.len) return error.NotImplemented;
        return &slot_buffers[slot_idx];
    }

    fn implicitBindingRegister(slot: usize) !runtime_contract.RegisterRef {
        const slot_u16: u16 = std.math.cast(u16, slot) orelse return error.InvalidInstructionBinding;
        return runtime_contract.registerFromIndex(IMPLICIT_BINDING_REGISTER_BASE + slot_u16);
    }

    fn arraySlotFromHandle(handle: runtime_contract.TensorHandle) *mlx_graph.ArrayHandle {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn instructionIoSlices(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !struct { inputs: []const runtime_contract.TensorHandle, outputs: []const runtime_contract.TensorHandle } {
        const io_count = insn.inputs.len + insn.outputs.len;
        if (registers.len < io_count) return error.InvalidInstructionBinding;
        return .{
            .inputs = registers[0..insn.inputs.len],
            .outputs = registers[insn.inputs.len..io_count],
        };
    }

    fn instructionImplicitBindingHandle(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        slot: usize,
    ) !runtime_contract.TensorHandle {
        const idx = insn.inputs.len + insn.outputs.len + insn.weights.len + slot;
        if (idx >= registers.len) return error.InvalidInstructionBinding;
        return registers[idx];
    }

    fn tensorViewDescForMetalArray() runtime_contract.TensorViewDesc {
        return .{
            .dtype = .f32,
            .rank = 0,
            .shape = .{ 0, 0, 0, 0 },
            .stride_elems = .{ 0, 0, 0, 0 },
            .layout = .backend_native,
        };
    }

    fn buildLayerProgramInstructionHandles(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
        handle_storage: *[MAX_LAYER_PROGRAM_HANDLES]runtime_contract.TensorHandle,
        view_storage: *[MAX_LAYER_PROGRAM_HANDLES]runtime_contract.TensorViewDesc,
    ) !BuiltLayerProgramHandles {
        var handle_count: usize = 0;
        var view_count: usize = 0;
        for (insn.inputs) |reg| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const slot = try bufferSlotForRegister(reg, ctx.residual, ctx.slot_buffers, ctx.register_to_slot_map);
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(slot),
            };
            view_storage[view_count] = tensorViewDescForMetalArray();
            handle_count += 1;
            view_count += 1;
        }
        for (insn.outputs) |reg| {
            if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
            const slot = try bufferSlotForRegister(reg, ctx.residual, ctx.slot_buffers, ctx.register_to_slot_map);
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(slot),
            };
            view_storage[view_count] = tensorViewDescForMetalArray();
            handle_count += 1;
            view_count += 1;
        }

        switch (insn.opcode) {
            .multihead_attention => {
                if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
                const binding = ctx.bindings.attention orelse return error.MissingField;
                const binding_ptr = &ctx.bindings.attention.?;
                _ = binding;
                handle_storage[handle_count] = .{
                    .register = try implicitBindingRegister(0),
                    .ptr = @ptrCast(binding_ptr),
                };
                handle_count += 1;
            },
            .shortconv => {
                if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
                const binding = ctx.bindings.shortconv orelse return error.MissingField;
                const binding_ptr = &ctx.bindings.shortconv.?;
                _ = binding;
                handle_storage[handle_count] = .{
                    .register = try implicitBindingRegister(0),
                    .ptr = @ptrCast(binding_ptr),
                };
                handle_count += 1;
            },
            .swiglu, .moe => {
                if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
                const binding = ctx.bindings.ffn orelse return error.MissingField;
                const binding_ptr = &ctx.bindings.ffn.?;
                _ = binding;
                handle_storage[handle_count] = .{
                    .register = try implicitBindingRegister(0),
                    .ptr = @ptrCast(binding_ptr),
                };
                handle_count += 1;
            },
            .mamba_mixer => {
                if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
                const binding = ctx.bindings.mamba orelse return error.MissingField;
                const binding_ptr = &ctx.bindings.mamba.?;
                _ = binding;
                handle_storage[handle_count] = .{
                    .register = try implicitBindingRegister(0),
                    .ptr = @ptrCast(binding_ptr),
                };
                handle_count += 1;
            },
            else => {},
        }

        return .{
            .registers = handle_storage[0..handle_count],
            .views = view_storage[0..view_count],
        };
    }

    fn buildLayerProgramRuntimeBindings(
        plan: *const runtime_contract.ExecutionPlan,
        lw: *const LayerWeights,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
    ) !LayerProgramRuntimeBindings {
        var bindings = LayerProgramRuntimeBindings{};
        bindings.norm_eps = config.norm_eps;
        bindings.residual_multiplier = weight_handles.residual_multiplier;
        bindings.norm_weights[0] = lw.getLn1();
        bindings.norm_weights[1] = lw.getLn2();
        bindings.norm_weight_count = 2;
        if (lw.getPreFfnNorm() orelse lw.getPostFfnNorm()) |extra_norm| {
            bindings.norm_weights[2] = extra_norm;
            bindings.norm_weight_count = 3;
        }
        if (lw.getPostFfnNorm()) |post_ffn_norm| {
            bindings.norm_weights[3] = post_ffn_norm;
            bindings.norm_weight_count = 4;
        }

        if (planUsesOpcode(plan, .multihead_attention)) {
            if (lw.isMLA()) {
                const mla_cfg = lw.mla_config orelse return error.MissingField;
                bindings.attention = .{
                    .mla = .{
                        .n_heads = @intCast(config.n_heads),
                        .rope_theta = config.rope_theta,
                        .norm_eps = config.norm_eps,
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
                    },
                };
            } else {
                const attention_storage = lw.attentionStorageKind();
                if (attention_storage == .invalid) return error.InvalidTensorType;
                if (attention_storage == .missing) return error.MissingField;
                bindings.attention = .{
                    .multihead = .{
                        .n_heads = @intCast(config.n_heads),
                        .n_kv_heads = @intCast(config.n_kv_groups),
                        .head_dim = @intCast(config.head_dim),
                        .rope_theta = config.rope_theta,
                        .norm_eps = config.norm_eps,
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
                    },
                };
            }
        }

        if (planUsesOpcode(plan, .shortconv)) {
            const shortconv_storage = lw.shortconvStorageKind();
            if (shortconv_storage == .invalid) return error.InvalidTensorType;
            if (shortconv_storage == .missing) return error.MissingField;
            const conv_weight = lw.shortconv_conv_weight orelse return error.MissingField;
            bindings.shortconv = .{
                .in_proj = if (lw.shortconv_in_proj) |w| w else null,
                .out_proj = if (lw.shortconv_out_proj) |w| w else null,
                .in_proj_bf16 = if (lw.shortconv_in_proj_bf16) |h| h else null,
                .out_proj_bf16 = if (lw.shortconv_out_proj_bf16) |h| h else null,
                .conv_weight = conv_weight,
                .conv_bias = if (lw.shortconv_conv_bias) |b| b else null,
                .d_conv = lw.shortconv_d_conv,
                .conv_dim = lw.shortconv_conv_dim,
            };
        }

        if (planUsesOpcode(plan, .swiglu) or planUsesOpcode(plan, .moe)) {
            const ffn_storage = lw.ffnStorageKind();
            switch (ffn_storage) {
                .moe => {
                    const moe = lw.moe orelse return error.MissingField;
                    bindings.ffn = .{ .moe = .{ .weights = moe } };
                },
                .quantized, .dense => {
                    bindings.ffn = .{
                        .dense = .{
                            .use_gelu = weight_handles.use_gelu,
                            .w1 = lw.w1,
                            .w2 = lw.w2,
                            .w3 = lw.w3,
                            .w1_bf16 = lw.w1_bf16,
                            .w2_bf16 = lw.w2_bf16,
                            .w3_bf16 = lw.w3_bf16,
                        },
                    };
                },
                .missing => return error.MissingField,
                .invalid => return error.InvalidTensorType,
            }
        }

        if (planUsesOpcode(plan, .mamba_mixer)) {
            const conv_weight = lw.mamba_conv_weight orelse return error.MissingField;
            const a_log = lw.mamba_a_log orelse return error.MissingField;
            const d_skip = lw.mamba_d_skip orelse return error.MissingField;
            bindings.mamba = .{
                .d_state = lw.mamba_d_state,
                .d_conv = lw.mamba_d_conv,
                .n_heads = lw.mamba_n_heads,
                .d_head = lw.mamba_d_head,
                .n_groups = lw.mamba_n_groups,
                .use_gelu = weight_handles.use_gelu,
                .residual_multiplier = weight_handles.residual_multiplier,
                .norm_eps = config.norm_eps,
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
        }

        return bindings;
    }

    fn nextNormWeight(bindings: *const LayerProgramRuntimeBindings, norm_index: *usize) !mlx_graph.ArrayHandle {
        const idx = norm_index.*;
        norm_index.* = idx + 1;
        if (idx >= bindings.norm_weight_count) return error.InvalidState;
        return bindings.norm_weights[idx] orelse error.InvalidState;
    }

    fn residualScale(
        scale: layer_ops.ResidualScale,
        residual_multiplier: f32,
    ) f32 {
        return switch (scale) {
            .one => 1.0,
            .residual_multiplier => residual_multiplier,
            .literal => |value| value,
        };
    }

    fn runAttentionKernel(
        input: mlx_graph.ArrayHandle,
        binding: AttentionRuntimeBinding,
        layer_idx: usize,
        cache: ?Cache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        return switch (binding) {
            .mla => |mla_attention| blk: {
                var mla = mla_attention;
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
                try mla.forward(
                    input,
                    &mla_out,
                    &mla_cache,
                    &mla_scratch,
                    &mla_matmul_scratch,
                    cache != null,
                );
                break :blk mla_out;
            },
            .multihead => |attention| blk: {
                var mha = attention;
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
                try mha.forward(
                    input,
                    &attn_out,
                    &attn_cache,
                    &attn_scratch,
                    &attn_matmul_scratch,
                    cache != null,
                );
                break :blk attn_out;
            },
        };
    }

    fn runShortConvKernel(
        input: mlx_graph.ArrayHandle,
        shortconv: shortconv_kernel.ShortConvKernel,
        layer_idx: usize,
        shortconv_cache: ?ShortConvCache,
    ) !mlx_graph.ArrayHandle {
        var kernel = shortconv;
        var shortconv_state = shortconv_kernel.ShortConvState{
            .cache = shortconv_cache,
            .layer_idx = layer_idx,
        };
        var shortconv_scratch = shortconv_kernel.ShortConvScratch{};
        var shortconv_matmul_scratch = shortconv_kernel.MatmulScratch{};
        var shortconv_out: mlx_graph.ArrayHandle = undefined;
        try kernel.forward(
            input,
            &shortconv_out,
            &shortconv_state,
            &shortconv_scratch,
            &shortconv_matmul_scratch,
        );
        return shortconv_out;
    }

    fn runMambaKernel(
        input: mlx_graph.ArrayHandle,
        mamba: mamba_kernel.MambaKernel,
        layer_idx: usize,
        mamba_cache: ?MambaCache,
    ) !mlx_graph.ArrayHandle {
        var kernel = mamba;
        var m_state = mamba_kernel.MambaState{
            .cache = mamba_cache,
            .layer_idx = layer_idx,
        };
        var m_scratch = mamba_kernel.MambaScratch{};
        var m_matmul = mamba_kernel.MatmulScratch{};
        var m_out: mlx_graph.ArrayHandle = undefined;
        try kernel.forward(
            input,
            &m_out,
            &m_state,
            &m_scratch,
            &m_matmul,
        );
        return m_out;
    }

    fn runFfnKernel(
        input: mlx_graph.ArrayHandle,
        binding: FfnRuntimeBinding,
    ) !mlx_graph.ArrayHandle {
        return switch (binding) {
            .moe => |ffn_moe| blk: {
                var ffn = ffn_moe;
                var moe_scratch = moe_kernel.MoEScratch{};
                var moe_matmul_scratch = moe_kernel.MatmulScratch{};
                var moe_out: mlx_graph.ArrayHandle = undefined;
                try ffn.forward(
                    input,
                    &moe_out,
                    &moe_scratch,
                    &moe_matmul_scratch,
                );
                break :blk moe_out;
            },
            .dense => |swiglu| blk: {
                var ffn = swiglu;
                var ffn_scratch = ffn_kernel.FfnScratch{};
                var ffn_matmul_scratch = ffn_kernel.MatmulScratch{};
                var ffn_result: mlx_graph.ArrayHandle = undefined;
                try ffn.forward(
                    input,
                    &ffn_result,
                    &ffn_scratch,
                    &ffn_matmul_scratch,
                );
                break :blk ffn_result;
            },
        };
    }

    fn layerProgramNormAdapter(
        op: layer_ops.LayerOp,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        norm_index: *usize,
        bindings: *const LayerProgramRuntimeBindings,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .norm) return error.NotImplemented;
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        var output: mlx_graph.ArrayHandle = undefined;
        const norm = norm_kernel.RMSNorm{
            .weight = try nextNormWeight(bindings, norm_index),
            .eps = bindings.norm_eps,
        };
        norm.forward(input, &output);
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramAttentionAdapter(
        op: layer_ops.LayerOp,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        layer_idx: usize,
        cache: ?Cache,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .multihead_attention) return error.NotImplemented;
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const binding_handle = try instructionImplicitBindingHandle(insn, registers, 0);
        const attention_binding_ptr: *const ?AttentionRuntimeBinding = @ptrCast(@alignCast(binding_handle.ptr));
        const attention_binding = attention_binding_ptr.* orelse return error.MissingField;
        const output = try runAttentionKernel(
            input,
            attention_binding,
            layer_idx,
            cache,
            pos_offset,
            runtime_rope_cos_handle,
            runtime_rope_sin_handle,
            runtime_rope_dim,
        );
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramShortConvAdapter(
        op: layer_ops.LayerOp,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        layer_idx: usize,
        shortconv_cache: ?ShortConvCache,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .shortconv) return error.NotImplemented;
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const binding_handle = try instructionImplicitBindingHandle(insn, registers, 0);
        const shortconv_binding_ptr: *const ?shortconv_kernel.ShortConvKernel = @ptrCast(@alignCast(binding_handle.ptr));
        const shortconv_binding = shortconv_binding_ptr.* orelse return error.MissingField;
        const output = try runShortConvKernel(input, shortconv_binding, layer_idx, shortconv_cache);
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramFfnAdapter(
        op: layer_ops.LayerOp,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .mlp and kernel_op.debug_type != .moe) return error.NotImplemented;
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const binding_handle = try instructionImplicitBindingHandle(insn, registers, 0);
        const ffn_binding_ptr: *const ?FfnRuntimeBinding = @ptrCast(@alignCast(binding_handle.ptr));
        const ffn_binding = ffn_binding_ptr.* orelse return error.MissingField;
        const output = try runFfnKernel(input, ffn_binding);
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramMambaAdapter(
        op: layer_ops.LayerOp,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        layer_idx: usize,
        mamba_cache: ?MambaCache,
    ) !void {
        const kernel_op = switch (op) {
            .kernel => |kernel| kernel,
            else => return error.NotImplemented,
        };
        if (kernel_op.debug_type != .mamba_mixer) return error.NotImplemented;
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const binding_handle = try instructionImplicitBindingHandle(insn, registers, 0);
        const mamba_binding_ptr: *const ?mamba_kernel.MambaKernel = @ptrCast(@alignCast(binding_handle.ptr));
        const mamba_binding = mamba_binding_ptr.* orelse return error.MissingField;
        const output = try runMambaKernel(
            input,
            mamba_binding,
            layer_idx,
            mamba_cache,
        );
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramResidualAddAdapter(
        op: layer_ops.LayerOp,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        bindings: *const LayerProgramRuntimeBindings,
    ) !void {
        const add_op = switch (op) {
            .add => |add| add,
            else => return error.NotImplemented,
        };
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len < 2) return error.InvalidInstructionBinding;
        const residual_slot = arraySlotFromHandle(io.inputs[0]);
        const branch = arraySlotFromHandle(io.inputs[1]).*;
        const scale = residualScale(add_op.scale, bindings.residual_multiplier);
        const scaled_branch = if (scale == 1.0)
            branch
        else
            mlx_graph.mlx_lazy_multiply_scalar(branch, scale);
        residual_slot.* = mlx_graph.mlx_lazy_add(residual_slot.*, scaled_branch);
    }

    fn layerProgramNormRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &state.compiled_plan.plan, state_blocks);
        const op = try layerProgramOpForInstruction(state, insn, state.op_index);
        try layerProgramNormAdapter(
            op,
            insn,
            registers,
            state.norm_index,
            &state.bindings,
        );
    }

    fn layerProgramAttentionRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &state.compiled_plan.plan, state_blocks);
        const op = try layerProgramOpForInstruction(state, insn, state.op_index);
        try layerProgramAttentionAdapter(
            op,
            insn,
            registers,
            state.layer_idx,
            state.cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
        );
    }

    fn layerProgramShortConvRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &state.compiled_plan.plan, state_blocks);
        const op = try layerProgramOpForInstruction(state, insn, state.op_index);
        try layerProgramShortConvAdapter(
            op,
            insn,
            registers,
            state.layer_idx,
            state.shortconv_cache,
        );
    }

    fn layerProgramFfnRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &state.compiled_plan.plan, state_blocks);
        const op = try layerProgramOpForInstruction(state, insn, state.op_index);
        try layerProgramFfnAdapter(
            op,
            insn,
            registers,
        );
    }

    fn layerProgramMambaRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &state.compiled_plan.plan, state_blocks);
        const op = try layerProgramOpForInstruction(state, insn, state.op_index);
        try layerProgramMambaAdapter(
            op,
            insn,
            registers,
            state.layer_idx,
            state.mamba_cache,
        );
    }

    fn layerProgramResidualAddRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &state.compiled_plan.plan, state_blocks);
        const op = try layerProgramOpForInstruction(state, insn, state.op_index);
        try layerProgramResidualAddAdapter(
            op,
            insn,
            registers,
            &state.bindings,
        );
    }

    fn dispatchLayerProgramInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !void {
        const adapter = layerProgramAdapterForOpcode(insn.opcode) orelse return error.NotImplemented;
        const active_slots: [1]usize = .{0};
        const no_seq_lengths: [0]u32 = .{};
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .decode,
            .active_slots = active_slots[0..],
            .sequence_lengths = no_seq_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = &layer_program_dispatch_counters,
            .workspace = .{ .any = @ptrCast(ctx) },
        };
        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
        var handle_storage: [MAX_LAYER_PROGRAM_HANDLES]runtime_contract.TensorHandle = undefined;
        var view_storage: [MAX_LAYER_PROGRAM_HANDLES]runtime_contract.TensorViewDesc = undefined;
        const built_handles = try buildLayerProgramInstructionHandles(insn, ctx, &handle_storage, &view_storage);
        var state_blocks = try layerProgramStateBlocksForInstruction(insn, ctx);
        _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &ctx.compiled_plan.plan, state_blocks.slice());
        try adapter(
            &rt_ctx,
            insn,
            built_handles.registers,
            built_handles.views,
            state_blocks.slice(),
            &.{},
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
            .compiled_plan = &compiled_plan,
            .instruction_ops = lw.instruction_ops,
            .op_index = 0,
            .layer_idx = layer_idx,
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
            .bindings = try buildLayerProgramRuntimeBindings(&compiled_plan.plan, lw, config, weight_handles),
        };
        _ = try runtime_contract.collectBuiltinStateFlags(&compiled_plan.plan);
        try bindLayerProgramStateDescriptors(&exec_ctx, &compiled_plan.plan);

        for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
            exec_ctx.op_index = op_index;
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

        if (lw.compiled_plan == null) {
            log.warn("inference", "Metal block missing compiled layer plan", .{
                .layer = layer_idx,
                .kind = @intFromEnum(lw.kind),
            });
            return error.UnsupportedModel;
        }
        try validateCompiledLayerProgram(lw, layer_idx);
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

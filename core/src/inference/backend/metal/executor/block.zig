//! Metal backend block executor.
//!
//! Centralizes single-layer lazy graph assembly so model-level orchestration
//! can delegate layer work through a stable `TransformerBlock.forward` surface.

const std = @import("std");
const compute = @import("../../../../compute/root.zig");
const layer_ops = @import("../../../../models/layer_ops.zig");
const op_types = @import("../../../../models/op_types.zig");
const tensor = @import("../../../../tensor.zig");
const opcode_map = @import("../../../../models/plan/opcode_map.zig");
const log = @import("../../../../log.zig");
const trace = @import("../../../../xray/trace.zig");
const runtime_contract = @import("../../../runtime_contract/root.zig");
const runtime_graph = @import("../runtime_graph.zig");
const weights_mod = @import("weights.zig");
const attention_kernel = @import("../kernels/attention.zig");
const ffn_kernel = @import("../kernels/ffn.zig");
const gated_delta_kernel = @import("../kernels/gated_delta.zig");
const mamba_kernel = @import("../kernels/mamba.zig");
const mla_kernel = @import("../kernels/mla_attention.zig");
const moe_kernel = @import("../kernels/moe.zig");
const norm_kernel = @import("../kernels/norm.zig");
const shortconv_kernel = @import("../kernels/shortconv.zig");
const vision_adapters = @import("../../../vision_program_adapters.zig");
const mlx_graph = compute.metal.graph;
// Optional dispatch observability. Keep disabled by default so production
// execution adds zero atomic overhead in the token loop.
const enable_dispatch_observability: bool = false;
var layer_program_dispatch_counters = runtime_contract.DispatchCounters{};

pub const Cache = runtime_graph.Cache;
pub const GatedDeltaCache = runtime_graph.GatedDeltaCache;
pub const ShortConvCache = runtime_graph.ShortConvCache;
pub const MambaCache = runtime_graph.MambaCache;
const ModelConfig = tensor.ModelConfig;
const WeightHandles = weights_mod.WeightHandles;
const LayerWeights = WeightHandles.LayerWeights;

pub const TransformerBlock = struct {
    const MaxLayerProgramStateBindings = 256;

    const AttentionRuntimeBinding = union(enum) {
        mla: mla_kernel.MLAttention,
        multihead: attention_kernel.MultiHeadAttention,
    };

    const LayerProgramStateBinding = struct {
        handle: runtime_contract.StateBlockHandle,
    };

    const LayerRuntimeMetadata = struct {
        model_config: ModelConfig,
        residual_multiplier: f32,
        use_gelu: bool,
        attention_multiplier: f32,
        attention_storage_kind: LayerWeights.AttentionStorageKind,
        gated_delta_storage_kind: LayerWeights.GatedDeltaStorageKind,
        shortconv_storage_kind: LayerWeights.ShortConvStorageKind,
        ffn_storage_kind: LayerWeights.FfnStorageKind,
        mla_storage_kind: LayerWeights.MLAStorageKind,
        mamba_storage_kind: LayerWeights.MambaStorageKind,
        mla_config: ?WeightHandles.MLAConfig,
        gated_delta_d_conv: usize,
        gated_delta_n_heads: usize,
        gated_delta_n_key_heads: usize,
        gated_delta_d_head: usize,
        shortconv_d_conv: usize,
        shortconv_conv_dim: usize,
        moe_router_group_size: usize,
        moe_expert_group_size: usize,
        moe_num_experts: usize,
        moe_experts_per_token: usize,
        mamba_d_state: usize,
        mamba_d_conv: usize,
        mamba_n_heads: usize,
        mamba_d_head: usize,
        mamba_n_groups: usize,
        mamba_gate_up_layout: u8,
    };

    const LayerProgramExecutionContext = struct {
        compiled_plan: *const runtime_contract.CompiledPlan,
        layer_weights: *const LayerWeights,
        layer_idx: usize,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: []mlx_graph.ArrayHandle,
        register_to_slot_map: []const u8,
        instruction_handles: []runtime_contract.TensorHandle,
        instruction_views: []runtime_contract.TensorViewDesc,
        runtime_meta: LayerRuntimeMetadata,
        resolved_weight_ptrs: []const ?*anyopaque,
        rmsnorm_weight_order: [6]?mlx_graph.ArrayHandle = [_]?mlx_graph.ArrayHandle{null} ** 6,
        rmsnorm_weight_count: usize = 0,
        rmsnorm_fallback_logged: bool = false,
        state_bindings: [MaxLayerProgramStateBindings]?LayerProgramStateBinding = [_]?LayerProgramStateBinding{null} ** MaxLayerProgramStateBindings,
        state_binding_count: usize = 0,
    };

    const BuiltLayerProgramHandles = struct {
        registers: []runtime_contract.TensorHandle,
        views: []const runtime_contract.TensorViewDesc,
    };

    const LayerProgramInstructionStateBlocks = struct {
        handles: [1]runtime_contract.StateBlockHandle = undefined,
        len: usize = 0,

        fn slice(self: *LayerProgramInstructionStateBlocks) []runtime_contract.StateBlockHandle {
            return self.handles[0..self.len];
        }
    };

    fn bindLayerProgramState(
        ctx: *LayerProgramExecutionContext,
        state_block: runtime_contract.StateBlockHandle,
    ) !void {
        var idx: usize = 0;
        while (idx < ctx.state_binding_count) : (idx += 1) {
            const existing = ctx.state_bindings[idx] orelse continue;
            if (existing.handle.id == state_block.id) {
                ctx.state_bindings[idx] = .{ .handle = state_block };
                return;
            }
        }
        if (ctx.state_binding_count >= ctx.state_bindings.len) return error.InvalidStateDescriptorBinding;
        ctx.state_bindings[ctx.state_binding_count] = .{ .handle = state_block };
        ctx.state_binding_count += 1;
    }

    fn layerProgramStateBinding(
        ctx: *const LayerProgramExecutionContext,
        state_id: u8,
    ) ?LayerProgramStateBinding {
        var idx: usize = 0;
        while (idx < ctx.state_binding_count) : (idx += 1) {
            const binding = ctx.state_bindings[idx] orelse continue;
            if (binding.handle.id == state_id) return binding;
        }
        return null;
    }

    fn bindLayerProgramStateDescriptors(
        ctx: *LayerProgramExecutionContext,
        plan: *const runtime_contract.ExecutionPlan,
        bound_state_blocks: []const runtime_contract.StateBlockHandle,
    ) !void {
        ctx.state_bindings = [_]?LayerProgramStateBinding{null} ** MaxLayerProgramStateBindings;
        ctx.state_binding_count = 0;
        for (plan.state_descs) |state_desc| {
            const state_block = runtime_contract.findStateBlock(bound_state_blocks, state_desc.id) orelse {
                return error.InvalidStateDescriptorBinding;
            };
            if (state_block.align_bytes < state_desc.align_bytes) return error.InvalidStateDescriptorBinding;
            if (state_desc.size_bytes > 0 and state_block.size < state_desc.size_bytes) return error.InvalidStateDescriptorBinding;
            try bindLayerProgramState(ctx, state_block.*);
        }
    }

    const layer_program_required_opcodes = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .mla_attention,
        .gated_delta_net,
        .shortconv,
        .swiglu,
        .moe,
        .mamba_mixer,
        .residual_add,
    } ++ vision_adapters.required_opcodes;

    const layer_program_adapter_table: runtime_contract.AdapterTable = blk: {
        var table: runtime_contract.AdapterTable = [_]?runtime_contract.KernelAdapterFn{null} ** 256;
        table[@intFromEnum(opcode_map.Opcode.rmsnorm)] = layerProgramNormRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.multihead_attention)] = layerProgramAttentionRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.mla_attention)] = layerProgramAttentionRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.gated_delta_net)] = layerProgramGatedDeltaRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.shortconv)] = layerProgramShortConvRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.swiglu)] = layerProgramSwiGluRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.moe)] = layerProgramMoeRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.mamba_mixer)] = layerProgramMambaRuntimeAdapter;
        table[@intFromEnum(opcode_map.Opcode.residual_add)] = layerProgramResidualAddRuntimeAdapter;

        // Vision opcodes — dispatched via vision_program_adapters.runVisionProgram
        for (vision_adapters.required_opcodes) |opcode| {
            table[@intFromEnum(opcode)] = vision_adapters.adapter_table[@intFromEnum(opcode)];
        }

        break :blk table;
    };

    const layer_program_adapter_capabilities: runtime_contract.AdapterCapabilities = blk: {
        var caps: runtime_contract.AdapterCapabilities = [_]runtime_contract.AdapterCapability{.{
            .supports_batch = false,
            .supports_graph_emit = true,
            .max_batch_size = 1,
        }} ** 256;

        for (layer_program_required_opcodes) |opcode| {
            caps[@intFromEnum(opcode)] = .{
                .supports_batch = false,
                .supports_graph_emit = true,
                .max_batch_size = 1,
            };
        }
        break :blk caps;
    };

    comptime {
        runtime_contract.assertAdapterTableCoverage(
            layer_program_adapter_table,
            layer_program_required_opcodes,
            "metal.executor.block.layer_program_adapter_table",
        );
    }

    fn layerProgramExecutionState(ctx: *runtime_contract.ExecutionContext) !*LayerProgramExecutionContext {
        const raw_state = ctx.workspace.any orelse return error.InvalidDispatchState;
        return @ptrCast(@alignCast(raw_state));
    }

    fn inferTraceSeqLen(state: *const LayerProgramExecutionContext) u32 {
        var shape_buffer: [8]usize = undefined;
        const rank = mlx_graph.getShape(state.residual.*, &shape_buffer);
        if (rank >= 2 and shape_buffer[1] > 0 and shape_buffer[1] <= std.math.maxInt(u32)) {
            return @intCast(shape_buffer[1]);
        }
        return 1;
    }

    fn traceShapeBsd(seq_len: u32, dim: u32) [4]u32 {
        return .{ 1, seq_len, dim, 0 };
    }

    fn traceTokenIndex(seq_len: u32) u32 {
        if (seq_len == 0) return 0;
        return seq_len - 1;
    }

    fn saturatingU32(value: usize) u32 {
        return @intCast(@min(value, std.math.maxInt(u32)));
    }

    fn tracePositionForPoint(point: trace.TracePoint, pos_offset: usize, seq_len: u32) u32 {
        if (seq_len == 0) return 0;
        return switch (point) {
            // Match CPU attention decode traces:
            // - q/k/v/embed_pos use cache position before qk/softmax update.
            .attn_q, .attn_k, .attn_v, .embed_pos => if (seq_len == 1)
                saturatingU32(pos_offset)
            else
                seq_len,
            // - qk/weights/attn_out use cache position after appending decode token.
            .attn_qk, .attn_weights, .attn_out => if (seq_len == 1)
                saturatingU32(pos_offset + 1)
            else
                seq_len,
            // Norm/FFN/residual-style points use sequence length in CPU traces.
            else => seq_len,
        };
    }

    fn emitLayerProgramTracePoint(
        state: *const LayerProgramExecutionContext,
        point: trace.TracePoint,
        shape: [4]u32,
        ndim: u8,
        kernel_name: []const u8,
    ) void {
        if (!trace.isEnabled()) return;
        // block_out must carry real host data; marker-pointer traces are invalid.
        if (point == .block_out) return;
        var marker = [_]f32{0.0};
        const seq_len = inferTraceSeqLen(state);
        trace.emit(
            point,
            @intCast(state.layer_idx),
            traceTokenIndex(seq_len),
            tracePositionForPoint(point, state.pos_offset, seq_len),
            @ptrCast(marker[0..].ptr),
            .f32,
            shape,
            ndim,
            kernel_name,
        );
    }

    fn emitLayerProgramModelWidthHostTracePoint(
        state: *const LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        point: trace.TracePoint,
        kernel_name: []const u8,
    ) void {
        if (!trace.isEnabled()) return;

        const io = instructionIoSlices(insn, registers) catch return;
        if (io.outputs.len == 0) return;
        const output = arraySlotFromHandle(io.outputs[0]).*;
        if (output == null) return;

        var output_shape: [8]usize = undefined;
        const output_rank = mlx_graph.getShape(output, &output_shape);
        if (output_rank == 0) return;

        var element_count: usize = 1;
        for (0..output_rank) |axis| {
            element_count = std.math.mul(usize, element_count, output_shape[axis]) catch return;
        }
        if (element_count == 0) return;

        const MappedShape = struct {
            batch: usize,
            seq_len: usize,
            width: usize,
        };
        const mapped: MappedShape = switch (output_rank) {
            3 => .{
                .batch = output_shape[0],
                .seq_len = output_shape[1],
                .width = output_shape[2],
            },
            2 => .{
                .batch = @as(usize, 1),
                .seq_len = output_shape[0],
                .width = output_shape[1],
            },
            1 => .{
                .batch = @as(usize, 1),
                .seq_len = @as(usize, 1),
                .width = output_shape[0],
            },
            else => return,
        };
        if (mapped.seq_len == 0 or mapped.width == 0) return;
        const seq_len_u32: u32 = @intCast(@min(mapped.seq_len, @as(usize, std.math.maxInt(u32))));
        const batch_u32: u32 = @intCast(@min(mapped.batch, @as(usize, std.math.maxInt(u32))));
        const width_u32: u32 = @intCast(@min(mapped.width, @as(usize, std.math.maxInt(u32))));

        mlx_graph.eval(&.{output});
        const host_buf = std.heap.c_allocator.alloc(f32, element_count) catch return;
        defer std.heap.c_allocator.free(host_buf);
        mlx_graph.copyToHost(output, host_buf);

        trace.emit(
            point,
            @intCast(state.layer_idx),
            traceTokenIndex(seq_len_u32),
            tracePositionForPoint(point, state.pos_offset, seq_len_u32),
            @ptrCast(host_buf.ptr),
            .f32,
            .{ batch_u32, seq_len_u32, width_u32, 0 },
            3,
            kernel_name,
        );
    }

    fn emitLayerProgramPerTokenHostTracePoint(
        state: *const LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        point: trace.TracePoint,
        kernel_name: []const u8,
    ) void {
        if (!trace.isEnabled()) return;

        const io = instructionIoSlices(insn, registers) catch return;
        if (io.outputs.len == 0) return;
        const output = arraySlotFromHandle(io.outputs[0]).*;
        if (output == null) return;

        var output_shape: [8]usize = undefined;
        const output_rank = mlx_graph.getShape(output, &output_shape);
        if (output_rank == 0) return;

        var seq_len: usize = 0;
        var width: usize = 0;
        switch (output_rank) {
            3 => {
                seq_len = output_shape[1];
                width = output_shape[2];
            },
            2 => {
                seq_len = output_shape[0];
                width = output_shape[1];
            },
            1 => {
                seq_len = 1;
                width = output_shape[0];
            },
            else => return,
        }
        if (seq_len == 0 or width == 0) return;

        for (0..seq_len) |token_idx| {
            const token_handle = if (output_rank == 1) output else blk: {
                var starts: [8]c_int = [_]c_int{0} ** 8;
                var ends: [8]c_int = [_]c_int{0} ** 8;
                for (0..output_rank) |axis| {
                    ends[axis] = @intCast(output_shape[axis]);
                }
                const seq_axis: usize = if (output_rank == 3) 1 else 0;
                starts[seq_axis] = @intCast(token_idx);
                ends[seq_axis] = @intCast(token_idx + 1);
                break :blk mlx_graph.mlx_lazy_slice(output, &starts, &ends, output_rank);
            };
            defer if (token_handle != output) mlx_graph.freeArray(token_handle);

            mlx_graph.eval(&.{token_handle});
            const host_buf = std.heap.c_allocator.alloc(f32, width) catch return;
            defer std.heap.c_allocator.free(host_buf);
            mlx_graph.copyToHost(token_handle, host_buf);

            trace.emit(
                point,
                @intCast(state.layer_idx),
                0,
                @intCast(token_idx),
                @ptrCast(host_buf.ptr),
                .f32,
                .{ 1, 1, @intCast(width), 0 },
                3,
                kernel_name,
            );
        }
    }

    fn emitArrayPerTokenHostTracePoint(
        state: *const LayerProgramExecutionContext,
        output: mlx_graph.ArrayHandle,
        point: trace.TracePoint,
        kernel_name: []const u8,
    ) void {
        if (!trace.isEnabled()) return;
        if (output == null) return;

        var output_shape: [8]usize = undefined;
        const output_rank = mlx_graph.getShape(output, &output_shape);
        if (output_rank == 0) return;

        var seq_len: usize = 0;
        var width: usize = 0;
        switch (output_rank) {
            3 => {
                seq_len = output_shape[1];
                width = output_shape[2];
            },
            2 => {
                seq_len = output_shape[0];
                width = output_shape[1];
            },
            1 => {
                seq_len = 1;
                width = output_shape[0];
            },
            else => return,
        }
        if (seq_len == 0 or width == 0) return;

        for (0..seq_len) |token_idx| {
            const token_handle = if (output_rank == 1) output else blk: {
                var starts: [8]c_int = [_]c_int{0} ** 8;
                var ends: [8]c_int = [_]c_int{0} ** 8;
                for (0..output_rank) |axis| {
                    ends[axis] = @intCast(output_shape[axis]);
                }
                const seq_axis: usize = if (output_rank == 3) 1 else 0;
                starts[seq_axis] = @intCast(token_idx);
                ends[seq_axis] = @intCast(token_idx + 1);
                break :blk mlx_graph.mlx_lazy_slice(output, &starts, &ends, output_rank);
            };
            defer if (token_handle != output) mlx_graph.freeArray(token_handle);

            mlx_graph.eval(&.{token_handle});
            const host_buf = std.heap.c_allocator.alloc(f32, width) catch return;
            defer std.heap.c_allocator.free(host_buf);
            mlx_graph.copyToHost(token_handle, host_buf);

            trace.emit(
                point,
                @intCast(state.layer_idx),
                0,
                @intCast(token_idx),
                @ptrCast(host_buf.ptr),
                .f32,
                .{ 1, 1, @intCast(width), 0 },
                3,
                kernel_name,
            );
        }
    }

    fn inferNormTracePoint(state: *const LayerProgramExecutionContext, insn: *const runtime_contract.Instruction) trace.TracePoint {
        const plan = &state.compiled_plan.plan;
        var op_index: ?usize = null;
        for (plan.instructions, 0..) |*candidate, idx| {
            if (candidate == insn) {
                op_index = idx;
                break;
            }
        }
        const op_idx = op_index orelse return .layer_ffn_norm;
        if (op_idx + 1 < plan.instructions.len) {
            const next_opcode = plan.instructions[op_idx + 1].opcode;
            if (next_opcode == .multihead_attention or next_opcode == .mla_attention or next_opcode == .shortconv or next_opcode == .gated_delta_net) {
                return .layer_attn_norm;
            }
        }
        return .layer_ffn_norm;
    }

    fn layerProgramStateBlocksForInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
    ) !LayerProgramInstructionStateBlocks {
        var blocks = LayerProgramInstructionStateBlocks{};
        const state_id = insn.state_block_id orelse return blocks;
        const binding = layerProgramStateBinding(ctx, state_id) orelse return error.InvalidStateDescriptorBinding;
        const state_block = binding.handle;
        if (comptime std.debug.runtime_safety) {
            const descriptor = runtime_contract.findStateDescriptor(&ctx.compiled_plan.plan, state_id) orelse {
                return error.UnknownStateDescriptorId;
            };
            if (state_block.align_bytes < descriptor.align_bytes) return error.InvalidStateDescriptorBinding;
            if (descriptor.size_bytes > 0 and state_block.size < descriptor.size_bytes) return error.InvalidStateDescriptorBinding;
        }
        blocks.handles[0] = state_block;
        blocks.len = 1;
        return blocks;
    }

    pub fn validateCompiledLayerProgram(lw: *const LayerWeights, layer_idx: usize) !void {
        const compiled_plan = lw.compiled_plan orelse return error.UnsupportedModel;
        runtime_contract.validateExecutionPlanForBlockKind(&compiled_plan.plan, lw.kind) catch |err| {
            log.warn("inference", "Metal compiled layer plan fails block-kind validation", .{
                .layer = layer_idx,
                .kind = @intFromEnum(lw.kind),
                .reason = @errorName(err),
            });
            return error.UnsupportedModel;
        };
        if (runtime_contract.firstUnsupportedInstructionOpcode(&compiled_plan.plan, layer_program_adapter_table)) |unsupported| {
            log.warn("inference", "Metal compiled layer plan contains unsupported opcode", .{
                .layer = layer_idx,
                .op_index = unsupported.instruction_index,
                .kind = @intFromEnum(lw.kind),
                .opcode = @intFromEnum(unsupported.opcode),
            });
            return error.UnsupportedModel;
        }
    }

    fn bufferSlotForRegister(
        reg: runtime_contract.RegisterRef,
        residual: *mlx_graph.ArrayHandle,
        slot_buffers: []mlx_graph.ArrayHandle,
        register_to_slot_map: []const u8,
    ) !*mlx_graph.ArrayHandle {
        const reg_idx = runtime_contract.registerToIndex(reg);
        if (reg_idx == 0) return residual;
        if (reg_idx >= register_to_slot_map.len) return error.NotImplemented;
        const slot_idx = register_to_slot_map[reg_idx];
        if (slot_idx == std.math.maxInt(u8) or slot_idx >= slot_buffers.len) return error.NotImplemented;
        return &slot_buffers[slot_idx];
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

    fn instructionWeightSlice(
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) ![]const runtime_contract.TensorHandle {
        const io_count = insn.inputs.len + insn.outputs.len;
        if (registers.len < io_count) return error.InvalidInstructionBinding;
        const weights = registers[io_count..];
        if (weights.len != insn.weights.len) return error.InvalidWeightRefCount;
        return weights;
    }

    fn quantizedWeightFromHandle(handle: runtime_contract.TensorHandle) *const WeightHandles.QuantizedWeight {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn arrayWeightFromHandle(handle: runtime_contract.TensorHandle) *const mlx_graph.ArrayHandle {
        return @ptrCast(@alignCast(handle.ptr));
    }

    fn optionalArrayWeightFromHandle(handle: runtime_contract.TensorHandle) ?mlx_graph.ArrayHandle {
        const value = arrayWeightFromHandle(handle).*;
        if (value == null) return null;
        return value;
    }

    fn layerProgramWeightHandlePtr(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
        slot_idx: usize,
    ) !*anyopaque {
        if (slot_idx >= insn.weights.len) return error.InvalidWeightRefIndex;
        const ref_index = insn.weights[slot_idx].index;
        if (ref_index >= ctx.resolved_weight_ptrs.len) return error.InvalidWeightRefIndex;
        if (ctx.resolved_weight_ptrs[ref_index]) |ptr| return ptr;
        const binding_key: WeightHandles.LayerProgramWeightBindingKey = if (ref_index < ctx.layer_weights.weight_binding_keys.len)
            ctx.layer_weights.weight_binding_keys[ref_index]
        else
            .invalid;
        log.warn("inference", "Metal layer-program unresolved weight pointer", .{
            .layer = ctx.layer_idx,
            .opcode = @intFromEnum(insn.opcode),
            .weight_ref_index = ref_index,
            .weight_slot = slot_idx,
            .binding_key = @tagName(binding_key),
            .weights_len = insn.weights.len,
            .resolved_len = ctx.resolved_weight_ptrs.len,
        });
        if (isRecoverableMissingLayerProgramWeight(binding_key)) {
            return weights_mod.missingOptionalWeightPtr();
        }
        return error.InvalidWeightBindingName;
    }

    fn isRecoverableMissingLayerProgramWeight(
        key: WeightHandles.LayerProgramWeightBindingKey,
    ) bool {
        return switch (key) {
            .norm_weight,
            .norm_bias,
            .attention_q_norm,
            .attention_k_norm,
            .attention_q_bias,
            .attention_k_bias,
            .attention_v_bias,
            .attention_o_bias,
            .attention_attn_sinks,
            .swiglu_w1_bias,
            .swiglu_w2_bias,
            .moe_router_bias,
            .moe_gate_bias,
            .moe_up_bias,
            .moe_down_bias,
            .moe_router_scales,
            .moe_router_quant_bias,
            .mamba_conv_bias,
            .mamba_dt_bias,
            .mamba_norm_weight,
            .mamba_gate_up,
            .mamba_down_proj,
            .gated_delta_conv_bias,
            .gated_delta_dt_bias,
            .gated_delta_norm_weight,
            .shortconv_conv_bias,
            => true,
            else => false,
        };
    }

    fn isMissingOptionalWeightHandle(handle: runtime_contract.TensorHandle) bool {
        return handle.ptr == weights_mod.missingOptionalWeightPtr();
    }

    fn instructionParams(
        insn: *const runtime_contract.Instruction,
        compiled_plan: *const runtime_contract.CompiledPlan,
        param_storage: *[1]runtime_contract.ParamBlock,
    ) ![]const runtime_contract.ParamBlock {
        const param_id = insn.param_block_id orelse return &.{};
        if (param_id >= compiled_plan.param_blocks.len) return error.MissingParamBlock;
        param_storage[0] = compiled_plan.param_blocks[param_id];
        return param_storage[0..1];
    }

    fn requireInstructionStateValue(
        comptime T: type,
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !T {
        const state_block = (try runtime_contract.requireInstructionStateBlockForPlan(
            insn,
            &state.compiled_plan.plan,
            state_blocks,
        )) orelse return error.InvalidStateDescriptorBinding;
        const direct_ptr: *const T = runtime_contract.stateValueFromBlock(*const T, state_block) orelse {
            return error.InvalidStateDescriptorBinding;
        };
        return direct_ptr.*;
    }

    fn instructionStateValueOrNull(
        comptime T: type,
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        state_blocks: []const runtime_contract.StateBlockHandle,
    ) !?T {
        if (insn.state_block_id == null) return null;
        return try requireInstructionStateValue(T, state, insn, state_blocks);
    }

    fn buildLayerProgramInstructionHandles(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
        handle_storage: []runtime_contract.TensorHandle,
        view_storage: []runtime_contract.TensorViewDesc,
    ) !BuiltLayerProgramHandles {
        var handle_count: usize = 0;
        for (insn.inputs) |reg| {
            if (handle_count >= handle_storage.len or handle_count >= view_storage.len) return error.InvalidInstructionBinding;
            const slot = try bufferSlotForRegister(reg, ctx.residual, ctx.slot_buffers, ctx.register_to_slot_map);
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(slot),
            };
            handle_count += 1;
        }
        for (insn.outputs) |reg| {
            if (handle_count >= handle_storage.len or handle_count >= view_storage.len) return error.InvalidInstructionBinding;
            const slot = try bufferSlotForRegister(reg, ctx.residual, ctx.slot_buffers, ctx.register_to_slot_map);
            handle_storage[handle_count] = .{
                .register = reg,
                .ptr = @ptrCast(slot),
            };
            handle_count += 1;
        }
        for (insn.weights, 0..) |_, slot_idx| {
            if (handle_count >= handle_storage.len or handle_count >= view_storage.len) return error.InvalidInstructionBinding;
            const weight_ptr = try layerProgramWeightHandlePtr(insn, ctx, slot_idx);
            handle_storage[handle_count] = .{
                .register = runtime_contract.registerFromIndex(@intCast(slot_idx)),
                .ptr = weight_ptr,
            };
            handle_count += 1;
        }

        return .{
            .registers = handle_storage[0..handle_count],
            .views = view_storage[0..handle_count],
        };
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

    fn countRmsNormInstructions(plan: *const runtime_contract.ExecutionPlan) usize {
        var count: usize = 0;
        for (plan.instructions) |insn| {
            if (insn.opcode == .rmsnorm) count += 1;
        }
        return count;
    }

    fn shouldIncludeQkNormInRmsOrder(
        layer: *const LayerWeights,
        rmsnorm_instruction_count: usize,
    ) bool {
        var has_named_attention_qk_norm = false;
        for (layer.weight_binding_keys) |key| {
            if (key == .attention_q_norm or key == .attention_k_norm) {
                has_named_attention_qk_norm = true;
                break;
            }
        }

        const base_rmsnorm_count: usize = 2 +
            (if (layer.pre_ffn_norm != null) @as(usize, 1) else 0) +
            (if (layer.post_ffn_norm != null) @as(usize, 1) else 0);

        return !has_named_attention_qk_norm and
            layer.q_norm != null and
            layer.k_norm != null and
            rmsnorm_instruction_count > base_rmsnorm_count;
    }

    fn buildRmsNormWeightOrder(state: *LayerProgramExecutionContext) void {
        state.rmsnorm_weight_order = [_]?mlx_graph.ArrayHandle{null} ** 6;
        state.rmsnorm_weight_count = 0;

        const layer = state.layer_weights;
        const rmsnorm_instruction_count = countRmsNormInstructions(&state.compiled_plan.plan);
        const include_qk_norm_in_order = shouldIncludeQkNormInRmsOrder(
            layer,
            rmsnorm_instruction_count,
        );

        state.rmsnorm_weight_order[state.rmsnorm_weight_count] = layer.ln1_weight;
        state.rmsnorm_weight_count += 1;

        if (include_qk_norm_in_order) {
            if (layer.q_norm) |w| {
                state.rmsnorm_weight_order[state.rmsnorm_weight_count] = w;
                state.rmsnorm_weight_count += 1;
            }
            if (layer.k_norm) |w| {
                state.rmsnorm_weight_order[state.rmsnorm_weight_count] = w;
                state.rmsnorm_weight_count += 1;
            }
        }

        state.rmsnorm_weight_order[state.rmsnorm_weight_count] = layer.ln2_weight;
        state.rmsnorm_weight_count += 1;

        if (layer.pre_ffn_norm) |w| {
            state.rmsnorm_weight_order[state.rmsnorm_weight_count] = w;
            state.rmsnorm_weight_count += 1;
        }
        if (layer.post_ffn_norm) |w| {
            state.rmsnorm_weight_order[state.rmsnorm_weight_count] = w;
            state.rmsnorm_weight_count += 1;
        }
    }

    fn normalizeRmsNormWeightForFeatureDim(
        weight: mlx_graph.ArrayHandle,
        feature_dim: usize,
    ) ?mlx_graph.ArrayHandle {
        if (weight == null or feature_dim == 0) return null;
        var shape_buf: [8]usize = undefined;
        const rank = mlx_graph.getShape(weight, &shape_buf);
        if (rank == 0) return weight;

        var normalized = weight;
        var dim: usize = 0;
        if (rank == 1) {
            dim = shape_buf[0];
        } else {
            var flat_count: usize = 1;
            for (0..rank) |axis| {
                flat_count *= shape_buf[axis];
            }
            if (flat_count == 0) return null;
            const flat_shape = [_]usize{flat_count};
            normalized = mlx_graph.mlx_lazy_reshape(weight, &flat_shape, 1);
            dim = flat_count;
        }

        if (dim == feature_dim) return normalized;
        if (feature_dim == dim * 2) return normalized;
        if (dim > feature_dim and dim % feature_dim == 0) {
            const starts = [_]c_int{0};
            const ends = [_]c_int{@intCast(feature_dim)};
            return mlx_graph.mlx_lazy_slice(normalized, &starts, &ends, 1);
        }
        return null;
    }

    fn selectRmsNormWeightByFeatureWidth(
        state: *const LayerProgramExecutionContext,
        feature_dim: usize,
    ) ?mlx_graph.ArrayHandle {
        const candidates = [_]?mlx_graph.ArrayHandle{
            state.layer_weights.ln1_weight,
            state.layer_weights.ln2_weight,
            state.layer_weights.q_norm,
            state.layer_weights.k_norm,
            state.layer_weights.pre_ffn_norm,
            state.layer_weights.post_ffn_norm,
        };
        for (candidates) |candidate_opt| {
            const candidate = candidate_opt orelse continue;
            if (normalizeRmsNormWeightForFeatureDim(candidate, feature_dim)) |normalized| {
                return normalized;
            }
        }
        return null;
    }

    fn firstRank1RmsNormWeight(state: *const LayerProgramExecutionContext) ?mlx_graph.ArrayHandle {
        for (state.rmsnorm_weight_order[0..state.rmsnorm_weight_count]) |candidate_opt| {
            const candidate = candidate_opt orelse continue;
            var shape_buf: [8]usize = undefined;
            const rank = mlx_graph.getShape(candidate, &shape_buf);
            if (rank == 1 or rank == 0) return candidate;
        }
        return null;
    }

    fn sanitizeAttentionNormWeight(
        weight_opt: ?mlx_graph.ArrayHandle,
        expected_width: usize,
    ) ?mlx_graph.ArrayHandle {
        const weight = weight_opt orelse return null;
        if (expected_width == 0) return null;

        var shape_buf: [8]usize = undefined;
        const rank = mlx_graph.getShape(weight, &shape_buf);
        if (rank == 0) return null;

        if (rank == 1) {
            const dim = shape_buf[0];
            if (dim == expected_width) return weight;
            if (dim > expected_width) {
                const starts = [_]c_int{0};
                const ends = [_]c_int{@intCast(expected_width)};
                return mlx_graph.mlx_lazy_slice(weight, &starts, &ends, 1);
            }
            return null;
        }

        var flat_count: usize = 1;
        for (0..rank) |axis| {
            flat_count *= shape_buf[axis];
        }
        if (flat_count < expected_width) return null;

        const flat_shape = [_]usize{flat_count};
        const flat = mlx_graph.mlx_lazy_reshape(weight, &flat_shape, 1);
        if (flat_count == expected_width) return flat;
        const starts = [_]c_int{0};
        const ends = [_]c_int{@intCast(expected_width)};
        return mlx_graph.mlx_lazy_slice(flat, &starts, &ends, 1);
    }

    fn synthesizeRmsNormWeightFromInput(
        input: mlx_graph.ArrayHandle,
        input_shape: *const [8]usize,
        input_rank: usize,
    ) !mlx_graph.ArrayHandle {
        if (input_rank == 0) return error.InvalidTensorType;
        if (input_rank > 8) return error.InvalidTensorType;
        // Construct a tensor whose flattened element count equals feature width.
        // For rank>1, slice the first element along all prefix axes and keep the
        // full feature axis, then map to ones. This yields shape [1,...,1,D].
        const seed = if (input_rank == 1) input else blk: {
            var starts: [8]c_int = [_]c_int{0} ** 8;
            var ends: [8]c_int = [_]c_int{0} ** 8;
            for (0..input_rank) |axis| {
                ends[axis] = if (axis + 1 == input_rank)
                    @intCast(input_shape[axis])
                else
                    1;
            }
            break :blk mlx_graph.mlx_lazy_slice(input, &starts, &ends, input_rank);
        };
        const zeros = mlx_graph.mlx_lazy_multiply_scalar(seed, 0.0);
        const ones = mlx_graph.mlx_add_one(zeros);
        const flat_shape = [_]usize{input_shape[input_rank - 1]};
        return mlx_graph.mlx_lazy_reshape(ones, &flat_shape, 1);
    }

    fn synthesizeUnitRmsNormWeight(feature_dim: usize) mlx_graph.ArrayHandle {
        const shape = [_]usize{feature_dim};
        return mlx_graph.mlx_lazy_full(&shape, 1, 1.0);
    }

    const ResolvedRmsNormWeight = struct {
        weight: mlx_graph.ArrayHandle,
        rank: usize,
        dim: usize,
    };

    fn rmsNormInstructionOrdinal(
        state: *const LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
    ) ?usize {
        var ordinal: usize = 0;
        for (state.compiled_plan.plan.instructions) |*candidate| {
            if (candidate.opcode == .rmsnorm) {
                if (candidate == insn) return ordinal;
                ordinal += 1;
            } else if (candidate == insn) {
                return null;
            }
        }
        return null;
    }

    fn resolveRmsNormWeightHandle(
        state: *const LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        fallback_weight: mlx_graph.ArrayHandle,
    ) !mlx_graph.ArrayHandle {
        const ordinal = rmsNormInstructionOrdinal(state, insn) orelse return error.InvalidInstructionBinding;
        if (ordinal < state.rmsnorm_weight_count) {
            return state.rmsnorm_weight_order[ordinal] orelse return error.MissingWeight;
        }
        return fallback_weight orelse error.MissingWeight;
    }

    fn resolveUsableRmsNormWeight(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        fallback_bound_weight: mlx_graph.ArrayHandle,
        input: mlx_graph.ArrayHandle,
        input_shape: *const [8]usize,
        input_rank: usize,
        feature_dim: usize,
    ) !ResolvedRmsNormWeight {
        // Use instruction-bound weight handles as source of truth. Ordered
        // fallback is only used when bindings are incomplete.
        var selected_weight = fallback_bound_weight;
        if (selected_weight == null) {
            selected_weight = resolveRmsNormWeightHandle(state, insn, fallback_bound_weight) catch fallback_bound_weight;
        }
        if (selected_weight == null) {
            if (!state.rmsnorm_fallback_logged) {
                log.warn("inference", "Metal RMSNorm missing instruction-bound weight; synthesizing fallback", .{
                    .layer = state.layer_idx,
                    .feature_dim = feature_dim,
                });
                state.rmsnorm_fallback_logged = true;
            }
            selected_weight = try synthesizeRmsNormWeightFromInput(input, input_shape, input_rank);
        }
        if (normalizeRmsNormWeightForFeatureDim(selected_weight, feature_dim)) |normalized| {
            selected_weight = normalized;
        }
        var selected_shape: [8]usize = undefined;
        var selected_rank = mlx_graph.getShape(selected_weight, &selected_shape);
        var selected_dim: usize = if (selected_rank >= 1) selected_shape[selected_rank - 1] else feature_dim;

        if (selected_rank != 0 and (selected_rank != 1 or (selected_dim != feature_dim and selected_dim * 2 != feature_dim))) {
            if (!state.rmsnorm_fallback_logged) {
                log.warn("inference", "Metal RMSNorm instruction-bound weight incompatible with input width; synthesizing fallback", .{
                    .layer = state.layer_idx,
                    .feature_dim = feature_dim,
                    .selected_dim = selected_dim,
                    .selected_rank = selected_rank,
                });
                state.rmsnorm_fallback_logged = true;
            }
            selected_weight = try synthesizeRmsNormWeightFromInput(input, input_shape, input_rank);
            if (normalizeRmsNormWeightForFeatureDim(selected_weight, feature_dim)) |normalized| {
                selected_weight = normalized;
            }
            selected_rank = mlx_graph.getShape(selected_weight, &selected_shape);
            selected_dim = if (selected_rank >= 1) selected_shape[selected_rank - 1] else feature_dim;
        }

        if (selected_rank != 0 and selected_dim != feature_dim and selected_dim * 2 != feature_dim) {
            if (!state.rmsnorm_fallback_logged) {
                log.warn("inference", "Metal RMSNorm selected incompatible width; synthesizing fallback", .{
                    .layer = state.layer_idx,
                    .feature_dim = feature_dim,
                    .selected_dim = selected_dim,
                });
                state.rmsnorm_fallback_logged = true;
            }
            selected_weight = try synthesizeRmsNormWeightFromInput(input, input_shape, input_rank);
            if (normalizeRmsNormWeightForFeatureDim(selected_weight, feature_dim)) |normalized| {
                selected_weight = normalized;
            }
            selected_rank = mlx_graph.getShape(selected_weight, &selected_shape);
            selected_dim = if (selected_rank >= 1) selected_shape[selected_rank - 1] else feature_dim;
        }
        return .{
            .weight = selected_weight,
            .rank = selected_rank,
            .dim = selected_dim,
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

    fn runGatedDeltaKernel(
        input: mlx_graph.ArrayHandle,
        gated_delta: gated_delta_kernel.GatedDeltaKernel,
        layer_idx: usize,
        gated_delta_cache: ?GatedDeltaCache,
        trace_state: ?*gated_delta_kernel.GatedDeltaState,
    ) !mlx_graph.ArrayHandle {
        var kernel = gated_delta;
        var gd_state = gated_delta_kernel.GatedDeltaState{
            .cache = gated_delta_cache,
            .layer_idx = layer_idx,
            .trace_capture_enabled = trace_state != null,
        };
        var gd_scratch = gated_delta_kernel.GatedDeltaScratch{};
        var gd_matmul = gated_delta_kernel.MatmulScratch{};
        var gd_out: mlx_graph.ArrayHandle = undefined;
        try kernel.forward(
            input,
            &gd_out,
            &gd_state,
            &gd_scratch,
            &gd_matmul,
        );
        if (trace_state) |out_state| {
            out_state.* = gd_state;
        }
        return gd_out;
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

    fn runSwiGluKernel(
        input: mlx_graph.ArrayHandle,
        swiglu: ffn_kernel.SwiGLU,
    ) !mlx_graph.ArrayHandle {
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
        return ffn_result;
    }

    fn canFuseDenseRmsNormSwiGlu(
        state: *const LayerProgramExecutionContext,
        norm_insn: *const runtime_contract.Instruction,
        swiglu_insn: *const runtime_contract.Instruction,
    ) bool {
        if (norm_insn.opcode != opcode_map.Opcode.rmsnorm) return false;
        if (swiglu_insn.opcode != opcode_map.Opcode.swiglu) return false;
        if (norm_insn.inputs.len != 1 or norm_insn.outputs.len != 1) return false;
        if (swiglu_insn.inputs.len != 1 or swiglu_insn.outputs.len != 1) return false;
        if (norm_insn.weights.len != runtime_contract.expectedWeightRefCount(norm_insn.opcode)) return false;
        if (swiglu_insn.weights.len != runtime_contract.expectedWeightRefCount(swiglu_insn.opcode)) return false;
        if (norm_insn.outputs[0] != swiglu_insn.inputs[0]) return false;
        return state.runtime_meta.ffn_storage_kind == .dense;
    }

    fn canFuseQuantizedRmsNormSwiGlu(
        state: *const LayerProgramExecutionContext,
        norm_insn: *const runtime_contract.Instruction,
        swiglu_insn: *const runtime_contract.Instruction,
    ) bool {
        if (norm_insn.opcode != opcode_map.Opcode.rmsnorm) return false;
        if (swiglu_insn.opcode != opcode_map.Opcode.swiglu) return false;
        if (norm_insn.inputs.len != 1 or norm_insn.outputs.len != 1) return false;
        if (swiglu_insn.inputs.len != 1 or swiglu_insn.outputs.len != 1) return false;
        if (norm_insn.weights.len != runtime_contract.expectedWeightRefCount(norm_insn.opcode)) return false;
        if (swiglu_insn.weights.len != runtime_contract.expectedWeightRefCount(swiglu_insn.opcode)) return false;
        if (norm_insn.outputs[0] != swiglu_insn.inputs[0]) return false;
        return state.runtime_meta.ffn_storage_kind == .quantized;
    }

    fn runDenseRmsNormSwiGluFusion(
        state: *LayerProgramExecutionContext,
        norm_insn: *const runtime_contract.Instruction,
        swiglu_insn: *const runtime_contract.Instruction,
    ) !void {
        const norm_handles = try buildLayerProgramInstructionHandles(
            norm_insn,
            state,
            state.instruction_handles,
            state.instruction_views,
        );
        const norm_io = try instructionIoSlices(norm_insn, norm_handles.registers);
        const norm_weights = try instructionWeightSlice(norm_insn, norm_handles.registers);
        const norm_input = arraySlotFromHandle(norm_io.inputs[0]).*;
        var norm_input_shape: [8]usize = undefined;
        const norm_input_rank = mlx_graph.getShape(norm_input, &norm_input_shape);
        if (norm_input_rank == 0) return error.InvalidTensorType;
        const feature_dim = norm_input_shape[norm_input_rank - 1];
        const fallback_norm_weight: mlx_graph.ArrayHandle = if (norm_weights.len > 0)
            arrayWeightFromHandle(norm_weights[0]).*
        else
            null;
        const resolved_norm_weight = try resolveUsableRmsNormWeight(
            state,
            norm_insn,
            fallback_norm_weight,
            norm_input,
            &norm_input_shape,
            norm_input_rank,
            feature_dim,
        );
        if (resolved_norm_weight.rank != 1) {
            try layerProgramNormAdapter(state, norm_insn, norm_handles.registers);
            const swiglu_handles_unfused = try buildLayerProgramInstructionHandles(
                swiglu_insn,
                state,
                state.instruction_handles,
                state.instruction_views,
            );
            try layerProgramSwiGluAdapter(state, swiglu_insn, swiglu_handles_unfused.registers);
            return;
        }
        const norm_weight = resolved_norm_weight.weight;
        const swiglu_handles = try buildLayerProgramInstructionHandles(
            swiglu_insn,
            state,
            state.instruction_handles,
            state.instruction_views,
        );
        const swiglu_io = try instructionIoSlices(swiglu_insn, swiglu_handles.registers);
        const swiglu_weights = try instructionWeightSlice(swiglu_insn, swiglu_handles.registers);
        var output: mlx_graph.ArrayHandle = undefined;
        ffn_kernel.forwardRmsNormFusedBf16(
            norm_input,
            norm_weight,
            state.runtime_meta.model_config.norm_eps,
            arrayWeightFromHandle(swiglu_weights[0]).*,
            arrayWeightFromHandle(swiglu_weights[1]).*,
            arrayWeightFromHandle(swiglu_weights[2]).*,
            &output,
        );
        arraySlotFromHandle(swiglu_io.outputs[0]).* = output;
    }

    fn runQuantizedRmsNormSwiGluFusion(
        state: *LayerProgramExecutionContext,
        norm_insn: *const runtime_contract.Instruction,
        swiglu_insn: *const runtime_contract.Instruction,
    ) !void {
        const norm_handles = try buildLayerProgramInstructionHandles(
            norm_insn,
            state,
            state.instruction_handles,
            state.instruction_views,
        );
        const norm_io = try instructionIoSlices(norm_insn, norm_handles.registers);
        const norm_weights = try instructionWeightSlice(norm_insn, norm_handles.registers);
        const norm_input = arraySlotFromHandle(norm_io.inputs[0]).*;
        var norm_input_shape: [8]usize = undefined;
        const norm_input_rank = mlx_graph.getShape(norm_input, &norm_input_shape);
        if (norm_input_rank == 0) return error.InvalidTensorType;
        const feature_dim = norm_input_shape[norm_input_rank - 1];
        const fallback_norm_weight: mlx_graph.ArrayHandle = if (norm_weights.len > 0)
            arrayWeightFromHandle(norm_weights[0]).*
        else
            null;
        const resolved_norm_weight = try resolveUsableRmsNormWeight(
            state,
            norm_insn,
            fallback_norm_weight,
            norm_input,
            &norm_input_shape,
            norm_input_rank,
            feature_dim,
        );
        if (resolved_norm_weight.rank != 1) {
            try layerProgramNormAdapter(state, norm_insn, norm_handles.registers);
            const swiglu_handles_unfused = try buildLayerProgramInstructionHandles(
                swiglu_insn,
                state,
                state.instruction_handles,
                state.instruction_views,
            );
            try layerProgramSwiGluAdapter(state, swiglu_insn, swiglu_handles_unfused.registers);
            return;
        }
        const norm_weight = resolved_norm_weight.weight;
        const swiglu_handles = try buildLayerProgramInstructionHandles(
            swiglu_insn,
            state,
            state.instruction_handles,
            state.instruction_views,
        );
        const swiglu_io = try instructionIoSlices(swiglu_insn, swiglu_handles.registers);
        const swiglu_weights = try instructionWeightSlice(swiglu_insn, swiglu_handles.registers);
        var output: mlx_graph.ArrayHandle = undefined;
        try ffn_kernel.forwardRmsNormFusedQuantized(
            norm_input,
            norm_weight,
            state.runtime_meta.model_config.norm_eps,
            state.runtime_meta.use_gelu,
            quantizedWeightFromHandle(swiglu_weights[0]).*,
            quantizedWeightFromHandle(swiglu_weights[1]).*,
            quantizedWeightFromHandle(swiglu_weights[2]).*,
            &output,
        );
        arraySlotFromHandle(swiglu_io.outputs[0]).* = output;
    }

    fn runMoeKernel(
        input: mlx_graph.ArrayHandle,
        moe_binding: moe_kernel.MoEFFN,
    ) !mlx_graph.ArrayHandle {
        var ffn = moe_binding;
        var moe_scratch = moe_kernel.MoEScratch{};
        var moe_matmul_scratch = moe_kernel.MatmulScratch{};
        var moe_out: mlx_graph.ArrayHandle = undefined;
        try ffn.forward(
            input,
            &moe_out,
            &moe_scratch,
            &moe_matmul_scratch,
        );
        return moe_out;
    }

    fn layerProgramNormAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        var output: mlx_graph.ArrayHandle = undefined;
        var input_shape: [8]usize = undefined;
        const input_rank = mlx_graph.getShape(input, &input_shape);
        if (input_rank == 0) return error.InvalidTensorType;
        const feature_dim = input_shape[input_rank - 1];

        const fallback_bound_weight: mlx_graph.ArrayHandle = if (weight_handles.len > 0)
            arrayWeightFromHandle(weight_handles[0]).*
        else
            null;
        const resolved_norm_weight = try resolveUsableRmsNormWeight(
            state,
            insn,
            fallback_bound_weight,
            input,
            &input_shape,
            input_rank,
            feature_dim,
        );
        var selected_weight = resolved_norm_weight.weight;
        var selected_dim = resolved_norm_weight.dim;
        var weight_shape: [8]usize = undefined;
        var weight_rank = mlx_graph.getShape(selected_weight, &weight_shape);
        if (weight_rank != 1) {
            if (normalizeRmsNormWeightForFeatureDim(selected_weight, feature_dim)) |normalized| {
                selected_weight = normalized;
                weight_rank = mlx_graph.getShape(selected_weight, &weight_shape);
                selected_dim = if (weight_rank >= 1) weight_shape[weight_rank - 1] else feature_dim;
            }
        }
        if (weight_rank != 1) {
            if (!state.rmsnorm_fallback_logged) {
                log.warn("inference", "Metal RMSNorm adapter forcing 1D fallback weight", .{
                    .layer = state.layer_idx,
                    .feature_dim = feature_dim,
                    .weight_rank = weight_rank,
                });
                state.rmsnorm_fallback_logged = true;
            }
            selected_weight = synthesizeUnitRmsNormWeight(feature_dim);
            weight_rank = 1;
            selected_dim = feature_dim;
        }

        const norm = norm_kernel.RMSNorm{
            .weight = selected_weight,
            .eps = state.runtime_meta.model_config.norm_eps,
        };
        // Query-gated models can carry packed Q projections [q | gate] where
        // RMSNorm weight is defined only for q. In that case normalize the
        // first half and keep gate half untouched.
        if (input_rank >= 1 and weight_rank == 1) {
            const norm_dim = selected_dim;
            if (norm_dim > 0 and feature_dim == norm_dim * 2) {
                var starts_first: [8]c_int = [_]c_int{0} ** 8;
                var ends_first: [8]c_int = [_]c_int{0} ** 8;
                var starts_second: [8]c_int = [_]c_int{0} ** 8;
                var ends_second: [8]c_int = [_]c_int{0} ** 8;
                if (input_rank > starts_first.len) return error.InvalidTensorType;
                for (0..input_rank) |axis| {
                    const dim_i32: c_int = @intCast(input_shape[axis]);
                    ends_first[axis] = dim_i32;
                    ends_second[axis] = dim_i32;
                }
                ends_first[input_rank - 1] = @intCast(norm_dim);
                starts_second[input_rank - 1] = @intCast(norm_dim);

                const q_half = mlx_graph.mlx_lazy_slice(input, &starts_first, &ends_first, input_rank);
                const gate_half = mlx_graph.mlx_lazy_slice(input, &starts_second, &ends_second, input_rank);
                var q_normed: mlx_graph.ArrayHandle = undefined;
                norm.forward(q_half, &q_normed);
                output = mlx_graph.mlx_lazy_concatenate(q_normed, gate_half, input_rank - 1);
                arraySlotFromHandle(io.outputs[0]).* = output;
                return;
            }
        }
        norm.forward(input, &output);
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramAttentionAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        cache: ?Cache,
        query_gate: bool,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        }
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        var attention_binding: AttentionRuntimeBinding = blk: {
            if (state.runtime_meta.mla_storage_kind != .missing) {
                const mla_cfg = state.runtime_meta.mla_config orelse return error.MissingField;
                break :blk .{
                    .mla = .{
                        .n_heads = @intCast(state.runtime_meta.model_config.n_heads),
                        .rope_theta = state.runtime_meta.model_config.rope_theta,
                        .norm_eps = state.runtime_meta.model_config.norm_eps,
                        .q_lora_rank = mla_cfg.q_lora_rank,
                        .kv_lora_rank = mla_cfg.kv_lora_rank,
                        .qk_head_dim = mla_cfg.qk_head_dim,
                        .qk_rope_head_dim = mla_cfg.qk_rope_head_dim,
                        .qk_nope_head_dim = mla_cfg.qk_nope_head_dim,
                        .v_head_dim = mla_cfg.v_head_dim,
                        .q_a_proj = null,
                        .q_b_proj = null,
                        .kv_a_proj = null,
                        .kv_b_proj = null,
                        .q_a_proj_bf16 = null,
                        .q_b_proj_bf16 = null,
                        .kv_a_proj_bf16 = null,
                        .kv_b_proj_bf16 = null,
                        .q_a_norm = null,
                        .kv_a_norm = null,
                        .o_proj = null,
                        .o_proj_bf16 = null,
                    },
                };
            }
            break :blk .{
                .multihead = .{
                    .n_heads = @intCast(state.runtime_meta.model_config.n_heads),
                    .n_kv_heads = @intCast(state.runtime_meta.model_config.n_kv_groups),
                    .head_dim = @intCast(state.runtime_meta.model_config.head_dim),
                    .rope_theta = state.runtime_meta.model_config.rope_theta,
                    .norm_eps = state.runtime_meta.model_config.norm_eps,
                    .query_pre_attn_scalar = state.runtime_meta.model_config.query_pre_attn_scalar,
                    .attention_multiplier = state.runtime_meta.attention_multiplier,
                    .query_gate = query_gate,
                    .q_proj = null,
                    .k_proj = null,
                    .v_proj = null,
                    .o_proj = null,
                    .q_proj_bf16 = null,
                    .k_proj_bf16 = null,
                    .v_proj_bf16 = null,
                    .o_proj_bf16 = null,
                    .q_norm = null,
                    .k_norm = null,
                    .q_bias = null,
                    .k_bias = null,
                    .v_bias = null,
                    .o_bias = null,
                    .attn_sinks = null,
                },
            };
        };
        switch (attention_binding) {
            .multihead => |*multihead| {
                switch (state.runtime_meta.attention_storage_kind) {
                    .quantized => {
                        multihead.q_proj = quantizedWeightFromHandle(weight_handles[0]).*;
                        multihead.k_proj = quantizedWeightFromHandle(weight_handles[1]).*;
                        multihead.v_proj = quantizedWeightFromHandle(weight_handles[2]).*;
                        multihead.o_proj = quantizedWeightFromHandle(weight_handles[3]).*;
                    },
                    .mixed_qkv_quantized_o_dense => {
                        multihead.q_proj = quantizedWeightFromHandle(weight_handles[0]).*;
                        multihead.k_proj = quantizedWeightFromHandle(weight_handles[1]).*;
                        multihead.v_proj = quantizedWeightFromHandle(weight_handles[2]).*;
                        multihead.o_proj_bf16 = arrayWeightFromHandle(weight_handles[3]).*;
                    },
                    .dense => {
                        multihead.q_proj_bf16 = arrayWeightFromHandle(weight_handles[0]).*;
                        multihead.k_proj_bf16 = arrayWeightFromHandle(weight_handles[1]).*;
                        multihead.v_proj_bf16 = arrayWeightFromHandle(weight_handles[2]).*;
                        multihead.o_proj_bf16 = arrayWeightFromHandle(weight_handles[3]).*;
                    },
                    else => return error.InvalidTensorType,
                }
                if (std.posix.getenv("TALU_DBG_ATTN_NORM") != null) {
                    const q_norm_raw = optionalArrayWeightFromHandle(weight_handles[4]);
                    const k_norm_raw = optionalArrayWeightFromHandle(weight_handles[5]);
                    var q_shape: [8]usize = undefined;
                    var k_shape: [8]usize = undefined;
                    const q_rank = if (q_norm_raw != null) mlx_graph.getShape(q_norm_raw.?, &q_shape) else @as(usize, 0);
                    const k_rank = if (k_norm_raw != null) mlx_graph.getShape(k_norm_raw.?, &k_shape) else @as(usize, 0);
                    std.debug.print(
                        "DBG_ATTN_NORM layer={} q_rank={} q_last_dim={} k_rank={} k_last_dim={} head_dim={} weight_slots={}\n",
                        .{
                            state.layer_idx,
                            q_rank,
                            if (q_rank > 0) q_shape[q_rank - 1] else @as(usize, 0),
                            k_rank,
                            if (k_rank > 0) k_shape[k_rank - 1] else @as(usize, 0),
                            state.runtime_meta.model_config.head_dim,
                            weight_handles.len,
                        },
                    );
                }
                multihead.q_norm = sanitizeAttentionNormWeight(
                    optionalArrayWeightFromHandle(weight_handles[4]),
                    @intCast(state.runtime_meta.model_config.head_dim),
                );
                multihead.k_norm = sanitizeAttentionNormWeight(
                    optionalArrayWeightFromHandle(weight_handles[5]),
                    @intCast(state.runtime_meta.model_config.head_dim),
                );
                multihead.q_bias = optionalArrayWeightFromHandle(weight_handles[6]);
                multihead.k_bias = optionalArrayWeightFromHandle(weight_handles[7]);
                multihead.v_bias = optionalArrayWeightFromHandle(weight_handles[8]);
                multihead.o_bias = optionalArrayWeightFromHandle(weight_handles[9]);
                multihead.attn_sinks = optionalArrayWeightFromHandle(weight_handles[10]);
            },
            .mla => |*mla| {
                switch (state.runtime_meta.mla_storage_kind) {
                    .quantized => {
                        mla.q_a_proj = quantizedWeightFromHandle(weight_handles[0]).*;
                        mla.q_b_proj = quantizedWeightFromHandle(weight_handles[2]).*;
                        mla.kv_a_proj = quantizedWeightFromHandle(weight_handles[3]).*;
                        mla.kv_b_proj = quantizedWeightFromHandle(weight_handles[5]).*;
                        mla.o_proj = quantizedWeightFromHandle(weight_handles[6]).*;
                    },
                    .dense => {
                        mla.q_a_proj_bf16 = arrayWeightFromHandle(weight_handles[0]).*;
                        mla.q_b_proj_bf16 = arrayWeightFromHandle(weight_handles[2]).*;
                        mla.kv_a_proj_bf16 = arrayWeightFromHandle(weight_handles[3]).*;
                        mla.kv_b_proj_bf16 = arrayWeightFromHandle(weight_handles[5]).*;
                        mla.o_proj_bf16 = arrayWeightFromHandle(weight_handles[6]).*;
                    },
                    else => return error.InvalidTensorType,
                }
                mla.q_a_norm = optionalArrayWeightFromHandle(weight_handles[1]) orelse return error.MissingField;
                mla.kv_a_norm = optionalArrayWeightFromHandle(weight_handles[4]) orelse return error.MissingField;
            },
        }
        const output = try runAttentionKernel(
            input,
            attention_binding,
            state.layer_idx,
            cache,
            state.pos_offset,
            state.runtime_rope_cos_handle,
            state.runtime_rope_sin_handle,
            state.runtime_rope_dim,
        );
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramShortConvAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        shortconv_cache: ?ShortConvCache,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        }
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        var shortconv_binding = shortconv_kernel.ShortConvKernel{
            .in_proj = null,
            .out_proj = null,
            .in_proj_bf16 = null,
            .out_proj_bf16 = null,
            .conv_weight = null,
            .conv_bias = null,
            .d_conv = state.runtime_meta.shortconv_d_conv,
            .conv_dim = state.runtime_meta.shortconv_conv_dim,
        };
        switch (state.runtime_meta.shortconv_storage_kind) {
            .quantized => {
                shortconv_binding.in_proj = quantizedWeightFromHandle(weight_handles[0]).*;
                shortconv_binding.out_proj = quantizedWeightFromHandle(weight_handles[2]).*;
            },
            .dense => {
                shortconv_binding.in_proj_bf16 = arrayWeightFromHandle(weight_handles[0]).*;
                shortconv_binding.out_proj_bf16 = arrayWeightFromHandle(weight_handles[2]).*;
            },
            else => return error.InvalidTensorType,
        }
        shortconv_binding.conv_weight = arrayWeightFromHandle(weight_handles[1]).*;
        shortconv_binding.conv_bias = optionalArrayWeightFromHandle(weight_handles[3]);
        const output = try runShortConvKernel(input, shortconv_binding, state.layer_idx, shortconv_cache);
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramGatedDeltaAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        gated_delta_cache: ?GatedDeltaCache,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        }
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        var gated_delta_binding = gated_delta_kernel.GatedDeltaKernel{
            .d_conv = state.runtime_meta.gated_delta_d_conv,
            .n_heads = state.runtime_meta.gated_delta_n_heads,
            .n_key_heads = state.runtime_meta.gated_delta_n_key_heads,
            .d_head = state.runtime_meta.gated_delta_d_head,
            .in_proj = null,
            .in_proj_bf16 = null,
            .conv_weight = null,
            .conv_bias = null,
            .a_log = null,
            .dt_bias = null,
            .norm_weight = null,
            .out_proj = null,
            .out_proj_bf16 = null,
        };
        switch (state.runtime_meta.gated_delta_storage_kind) {
            .quantized => {
                gated_delta_binding.in_proj = quantizedWeightFromHandle(weight_handles[0]).*;
                gated_delta_binding.out_proj = quantizedWeightFromHandle(weight_handles[3]).*;
            },
            .dense => {
                gated_delta_binding.in_proj_bf16 = arrayWeightFromHandle(weight_handles[0]).*;
                gated_delta_binding.out_proj_bf16 = arrayWeightFromHandle(weight_handles[3]).*;
            },
            else => return error.InvalidTensorType,
        }
        gated_delta_binding.conv_weight = arrayWeightFromHandle(weight_handles[1]).*;
        gated_delta_binding.a_log = arrayWeightFromHandle(weight_handles[2]).*;
        gated_delta_binding.conv_bias = optionalArrayWeightFromHandle(weight_handles[4]);
        gated_delta_binding.dt_bias = optionalArrayWeightFromHandle(weight_handles[5]);
        gated_delta_binding.norm_weight = optionalArrayWeightFromHandle(weight_handles[6]);
        var gd_trace_state: gated_delta_kernel.GatedDeltaState = .{
            .cache = gated_delta_cache,
            .layer_idx = state.layer_idx,
        };
        // XRAY ACCEPTABLE USE:
        // Trace state is capture-only. It can mirror intermediates to host but
        // MUST NOT change the selected kernel path or arithmetic.
        const gd_trace_state_ptr: ?*gated_delta_kernel.GatedDeltaState = if (trace.isEnabled())
            &gd_trace_state
        else
            null;
        const output = try runGatedDeltaKernel(
            input,
            gated_delta_binding,
            state.layer_idx,
            gated_delta_cache,
            gd_trace_state_ptr,
        );
        arraySlotFromHandle(io.outputs[0]).* = output;
        if (trace.isEnabled()) {
            const captured_handles: [4]?mlx_graph.ArrayHandle = .{
                gd_trace_state.trace_in_proj,
                gd_trace_state.trace_conv,
                gd_trace_state.trace_ssm,
                gd_trace_state.trace_norm,
            };
            defer {
                // Capture kernels may alias returned intermediates. Free only
                // unique non-output handles to avoid accidental double-free/UAF.
                var seen: [4]mlx_graph.ArrayHandle = undefined;
                var seen_count: usize = 0;
                for (captured_handles) |maybe_handle| {
                    const handle = maybe_handle orelse continue;
                    if (handle == output) continue;
                    var duplicate = false;
                    for (seen[0..seen_count]) |existing| {
                        if (existing == handle) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        seen[seen_count] = handle;
                        seen_count += 1;
                        mlx_graph.freeArray(handle);
                    }
                }
            }
            if (gd_trace_state.trace_in_proj) |handle| {
                emitArrayPerTokenHostTracePoint(state, handle, .gdelta_in_proj, "metal_gdelta_in_proj_host");
            }
            if (gd_trace_state.trace_conv) |handle| {
                emitArrayPerTokenHostTracePoint(state, handle, .gdelta_conv, "metal_gdelta_conv_host");
            }
            if (gd_trace_state.trace_ssm) |handle| {
                emitArrayPerTokenHostTracePoint(state, handle, .gdelta_ssm, "metal_gdelta_ssm_host");
            }
            if (gd_trace_state.trace_norm) |handle| {
                emitArrayPerTokenHostTracePoint(state, handle, .gdelta_norm, "metal_gdelta_norm_host");
            }
            emitArrayPerTokenHostTracePoint(state, output, .gdelta_out, "metal_gdelta_out_host");
        }
    }

    fn layerProgramSwiGluAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        }
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        var swiglu_binding = ffn_kernel.SwiGLU{
            .use_gelu = state.runtime_meta.use_gelu,
            .w1 = null,
            .w2 = null,
            .w3 = null,
            .w1_bf16 = null,
            .w2_bf16 = null,
            .w3_bf16 = null,
        };
        switch (state.runtime_meta.ffn_storage_kind) {
            .quantized => {
                swiglu_binding.w1 = quantizedWeightFromHandle(weight_handles[0]).*;
                swiglu_binding.w3 = quantizedWeightFromHandle(weight_handles[1]).*;
                swiglu_binding.w2 = quantizedWeightFromHandle(weight_handles[2]).*;
            },
            .dense => {
                swiglu_binding.w1_bf16 = arrayWeightFromHandle(weight_handles[0]).*;
                swiglu_binding.w3_bf16 = arrayWeightFromHandle(weight_handles[1]).*;
                swiglu_binding.w2_bf16 = arrayWeightFromHandle(weight_handles[2]).*;
            },
            else => return error.InvalidTensorType,
        }
        const output = try runSwiGluKernel(input, swiglu_binding);
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramMoeAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        }
        const input = arraySlotFromHandle(io.inputs[0]).*;
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        var runtime_moe = WeightHandles.MoEWeights{
            .router_w = arrayWeightFromHandle(weight_handles[0]).*,
            .router_s = optionalArrayWeightFromHandle(weight_handles[11]),
            .router_b = optionalArrayWeightFromHandle(weight_handles[12]),
            .router_bias = optionalArrayWeightFromHandle(weight_handles[4]),
            .gate_w = arrayWeightFromHandle(weight_handles[1]).*,
            .gate_s = arrayWeightFromHandle(weight_handles[5]).*,
            .up_w = arrayWeightFromHandle(weight_handles[2]).*,
            .up_s = arrayWeightFromHandle(weight_handles[6]).*,
            .down_w = arrayWeightFromHandle(weight_handles[3]).*,
            .down_s = arrayWeightFromHandle(weight_handles[7]).*,
            .gate_bias = optionalArrayWeightFromHandle(weight_handles[8]),
            .up_bias = optionalArrayWeightFromHandle(weight_handles[9]),
            .down_bias = optionalArrayWeightFromHandle(weight_handles[10]),
            .router_group_size = state.runtime_meta.moe_router_group_size,
            .expert_group_size = state.runtime_meta.moe_expert_group_size,
            .num_experts = state.runtime_meta.moe_num_experts,
            .experts_per_token = state.runtime_meta.moe_experts_per_token,
        };
        const output = try runMoeKernel(input, .{ .weights = &runtime_moe });
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramMambaAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        mamba_cache: ?MambaCache,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
        }
        const weight_handles = try instructionWeightSlice(insn, registers);
        if (weight_handles.len != runtime_contract.expectedWeightRefCount(insn.opcode)) return error.InvalidWeightRefCount;
        const input = arraySlotFromHandle(io.inputs[0]).*;
        var mamba_binding = mamba_kernel.MambaKernel{
            .d_state = state.runtime_meta.mamba_d_state,
            .d_conv = state.runtime_meta.mamba_d_conv,
            .n_heads = state.runtime_meta.mamba_n_heads,
            .d_head = state.runtime_meta.mamba_d_head,
            .n_groups = state.runtime_meta.mamba_n_groups,
            .use_gelu = state.runtime_meta.use_gelu,
            .residual_multiplier = state.runtime_meta.residual_multiplier,
            .norm_eps = state.runtime_meta.model_config.norm_eps,
            .gate_up_layout = state.runtime_meta.mamba_gate_up_layout,
            .ln1_weight = null,
            .in_proj = null,
            .in_proj_bf16 = null,
            .conv_weight = null,
            .conv_bias = null,
            .a_log = null,
            .d_skip = null,
            .dt_bias = null,
            .norm_weight = null,
            .out_proj = null,
            .out_proj_bf16 = null,
            .ln2_weight = null,
            .gate_up = null,
            .gate_up_bf16 = null,
            .down_proj = null,
            .down_proj_bf16 = null,
        };
        switch (state.runtime_meta.mamba_storage_kind) {
            .quantized => {
                mamba_binding.in_proj = quantizedWeightFromHandle(weight_handles[0]).*;
                mamba_binding.out_proj = quantizedWeightFromHandle(weight_handles[4]).*;
            },
            .dense => {
                mamba_binding.in_proj_bf16 = arrayWeightFromHandle(weight_handles[0]).*;
                mamba_binding.out_proj_bf16 = arrayWeightFromHandle(weight_handles[4]).*;
            },
            else => return error.InvalidTensorType,
        }
        mamba_binding.conv_weight = arrayWeightFromHandle(weight_handles[1]).*;
        mamba_binding.a_log = arrayWeightFromHandle(weight_handles[2]).*;
        mamba_binding.d_skip = arrayWeightFromHandle(weight_handles[3]).*;
        mamba_binding.ln1_weight = arrayWeightFromHandle(weight_handles[8]).*;
        mamba_binding.ln2_weight = arrayWeightFromHandle(weight_handles[9]).*;
        mamba_binding.conv_bias = optionalArrayWeightFromHandle(weight_handles[5]);
        mamba_binding.dt_bias = optionalArrayWeightFromHandle(weight_handles[6]);
        mamba_binding.norm_weight = optionalArrayWeightFromHandle(weight_handles[7]);
        if (!isMissingOptionalWeightHandle(weight_handles[10])) {
            switch (state.runtime_meta.mamba_storage_kind) {
                .quantized => mamba_binding.gate_up = quantizedWeightFromHandle(weight_handles[10]).*,
                .dense => mamba_binding.gate_up_bf16 = optionalArrayWeightFromHandle(weight_handles[10]),
                else => return error.InvalidTensorType,
            }
        }
        if (!isMissingOptionalWeightHandle(weight_handles[11])) {
            switch (state.runtime_meta.mamba_storage_kind) {
                .quantized => mamba_binding.down_proj = quantizedWeightFromHandle(weight_handles[11]).*,
                .dense => mamba_binding.down_proj_bf16 = optionalArrayWeightFromHandle(weight_handles[11]),
                else => return error.InvalidTensorType,
            }
        }
        const output = try runMambaKernel(
            input,
            mamba_binding,
            state.layer_idx,
            mamba_cache,
        );
        arraySlotFromHandle(io.outputs[0]).* = output;
    }

    fn layerProgramResidualAddAdapter(
        state: *LayerProgramExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        scale_kind: layer_ops.ResidualScale,
    ) !void {
        const io = try instructionIoSlices(insn, registers);
        if (comptime std.debug.runtime_safety) {
            if (io.inputs.len < 2) return error.InvalidInstructionBinding;
        }
        const residual_slot = arraySlotFromHandle(io.inputs[0]);
        const branch = arraySlotFromHandle(io.inputs[1]).*;
        const scale = residualScale(scale_kind, state.runtime_meta.residual_multiplier);
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
        _: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        try layerProgramNormAdapter(
            state,
            insn,
            registers,
        );
        const point = inferNormTracePoint(state, insn);
        emitLayerProgramModelWidthHostTracePoint(
            state,
            insn,
            registers,
            point,
            "metal_rmsnorm_host",
        );
    }

    fn layerProgramAttentionRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        const cache = try instructionStateValueOrNull(
            Cache,
            state,
            insn,
            state_blocks,
        );
        const seq_len = inferTraceSeqLen(state);
        const q_dim: u32 = if (insn.opcode == .multihead_attention)
            @intCast(state.runtime_meta.model_config.n_heads * state.runtime_meta.model_config.head_dim)
        else
            @intCast(state.runtime_meta.model_config.d_model);
        emitLayerProgramTracePoint(
            state,
            .attn_q,
            traceShapeBsd(seq_len, q_dim),
            3,
            "metal_attention_q",
        );
        const query_gate = if (insn.opcode == .multihead_attention) blk: {
            const p = try runtime_contract.paramAs(
                runtime_contract.AttentionKernelParam,
                params,
                .multihead_attention,
            );
            break :blk p.query_gate != 0;
        } else false;
        try layerProgramAttentionAdapter(
            state,
            insn,
            registers,
            cache,
            query_gate,
        );
        emitLayerProgramTracePoint(
            state,
            .attn_out,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_attention_out",
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
        const shortconv_cache = try requireInstructionStateValue(
            ShortConvCache,
            state,
            insn,
            state_blocks,
        );
        const seq_len = inferTraceSeqLen(state);
        emitLayerProgramTracePoint(
            state,
            .conv_in_proj,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.shortconv_conv_dim * 3)),
            3,
            "metal_shortconv_in_proj",
        );
        try layerProgramShortConvAdapter(
            state,
            insn,
            registers,
            shortconv_cache,
        );
        emitLayerProgramTracePoint(
            state,
            .conv_out_proj,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_shortconv_out_proj",
        );
    }

    fn layerProgramGatedDeltaRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        state_blocks: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        const p = try runtime_contract.paramAs(runtime_contract.GatedDeltaKernelParam, params, .gated_delta_net);
        const d_inner = std.math.mul(u32, p.n_heads, p.d_head) catch return error.InvalidParamBlockABI;
        if (p.d_inner != d_inner) return error.InvalidParamBlockABI;
        if (p.d_conv != @as(u32, @intCast(state.runtime_meta.gated_delta_d_conv)) or
            p.n_heads != @as(u32, @intCast(state.runtime_meta.gated_delta_n_heads)) or
            p.d_head != @as(u32, @intCast(state.runtime_meta.gated_delta_d_head)))
        {
            return error.InvalidParamBlockABI;
        }
        const gated_delta_cache = try requireInstructionStateValue(
            GatedDeltaCache,
            state,
            insn,
            state_blocks,
        );
        const seq_len = inferTraceSeqLen(state);
        try layerProgramGatedDeltaAdapter(
            state,
            insn,
            registers,
            gated_delta_cache,
        );
        emitLayerProgramTracePoint(
            state,
            .block_out,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_gated_delta_out",
        );
    }

    fn layerProgramSwiGluRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        _: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        const seq_len = inferTraceSeqLen(state);
        emitLayerProgramTracePoint(
            state,
            .ffn_gate,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_ff)),
            3,
            "metal_ffn_gate",
        );
        try layerProgramSwiGluAdapter(
            state,
            insn,
            registers,
        );
        emitLayerProgramTracePoint(
            state,
            .ffn_down,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_ffn_down",
        );
    }

    fn layerProgramMoeRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        _: []runtime_contract.StateBlockHandle,
        _: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        const seq_len = inferTraceSeqLen(state);
        emitLayerProgramTracePoint(
            state,
            .ffn_gate,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_ff)),
            3,
            "metal_moe_gate",
        );
        try layerProgramMoeAdapter(
            state,
            insn,
            registers,
        );
        emitLayerProgramTracePoint(
            state,
            .ffn_down,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_moe_down",
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
        const mamba_cache = try requireInstructionStateValue(
            MambaCache,
            state,
            insn,
            state_blocks,
        );
        const seq_len = inferTraceSeqLen(state);
        try layerProgramMambaAdapter(
            state,
            insn,
            registers,
            mamba_cache,
        );
        emitLayerProgramTracePoint(
            state,
            .mamba_out,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_mamba_out",
        );
    }

    fn layerProgramResidualAddRuntimeAdapter(
        ctx: *runtime_contract.ExecutionContext,
        insn: *const runtime_contract.Instruction,
        registers: []runtime_contract.TensorHandle,
        _: []const runtime_contract.TensorViewDesc,
        _: []runtime_contract.StateBlockHandle,
        params: []const runtime_contract.ParamBlock,
    ) !void {
        const state = try layerProgramExecutionState(ctx);
        const p = try runtime_contract.paramAs(runtime_contract.ResidualAddParam, params, .residual_add);
        const scale_kind: layer_ops.ResidualScale = switch (p.scale_tag) {
            0 => .one,
            1 => .residual_multiplier,
            2 => .{ .literal = @bitCast(p.scale_literal) },
            else => return error.InvalidParamBlockABI,
        };
        try layerProgramResidualAddAdapter(
            state,
            insn,
            registers,
            scale_kind,
        );
        const seq_len = inferTraceSeqLen(state);
        emitLayerProgramTracePoint(
            state,
            .block_out,
            traceShapeBsd(seq_len, @intCast(state.runtime_meta.model_config.d_model)),
            3,
            "metal_residual_add",
        );
    }

    fn dispatchLayerProgramInstruction(
        insn: *const runtime_contract.Instruction,
        ctx: *LayerProgramExecutionContext,
        rt_ctx: *runtime_contract.ExecutionContext,
    ) !void {
        const adapter = layer_program_adapter_table[@intFromEnum(insn.opcode)].?;
        rt_ctx.workspace.any = @ptrCast(ctx);
        if (comptime std.debug.runtime_safety) {
            try runtime_contract.validateBatchCapability(
                layer_program_adapter_capabilities[@intFromEnum(insn.opcode)],
                rt_ctx.batch_size,
            );
        }
        if (enable_dispatch_observability) runtime_contract.recordExecutionDispatch(rt_ctx, insn.opcode);
        const built_handles = try buildLayerProgramInstructionHandles(
            insn,
            ctx,
            ctx.instruction_handles,
            ctx.instruction_views,
        );
        var state_blocks = try layerProgramStateBlocksForInstruction(insn, ctx);
        var param_storage: [1]runtime_contract.ParamBlock = undefined;
        if (comptime std.debug.runtime_safety) {
            _ = try runtime_contract.requireInstructionStateBlockForPlan(insn, &ctx.compiled_plan.plan, state_blocks.slice());
        }
        try adapter(
            rt_ctx,
            insn,
            built_handles.registers,
            built_handles.views,
            state_blocks.slice(),
            try instructionParams(insn, ctx.compiled_plan, &param_storage),
        );
    }

    fn forwardWithProgram(
        hidden: mlx_graph.ArrayHandle,
        lw: *const LayerWeights,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        layer_idx: usize,
        state_blocks: []const runtime_contract.StateBlockHandle,
        pos_offset: usize,
        runtime_rope_cos_handle: mlx_graph.ArrayHandle,
        runtime_rope_sin_handle: mlx_graph.ArrayHandle,
        runtime_rope_dim: usize,
    ) !mlx_graph.ArrayHandle {
        const compiled_plan = lw.compiled_plan orelse return error.NotImplemented;
        var residual = hidden;
        const required_slot_count = blk: {
            var required: usize = 0;
            for (lw.register_to_slot_map) |slot_idx| {
                if (slot_idx == std.math.maxInt(u8)) continue;
                const next = @as(usize, slot_idx) + 1;
                if (next > required) required = next;
            }
            break :blk required;
        };
        if (required_slot_count > lw.slot_scratch.len) return error.InvalidInstructionBinding;
        for (lw.slot_scratch[0..required_slot_count]) |*slot| {
            slot.* = hidden;
        }
        const slot_buffers = lw.slot_scratch[0..required_slot_count];
        const active_slots: [1]usize = .{0};
        const sequence_lengths: [1]u32 = .{0};
        var rt_ctx = runtime_contract.ExecutionContext{
            .mode = .decode,
            .active_slots = active_slots[0..],
            .sequence_lengths = sequence_lengths[0..],
            .batch_size = 1,
            .dispatch_counters = if (enable_dispatch_observability) &layer_program_dispatch_counters else null,
            .workspace = .{ .any = null },
        };
        if (comptime std.debug.runtime_safety) {
            try runtime_contract.validateExecutionContext(&rt_ctx);
        }
        var exec_ctx = LayerProgramExecutionContext{
            .compiled_plan = &compiled_plan,
            .layer_weights = lw,
            .layer_idx = layer_idx,
            .pos_offset = pos_offset,
            .runtime_rope_cos_handle = runtime_rope_cos_handle,
            .runtime_rope_sin_handle = runtime_rope_sin_handle,
            .runtime_rope_dim = runtime_rope_dim,
            .residual = &residual,
            .slot_buffers = slot_buffers,
            .register_to_slot_map = lw.register_to_slot_map,
            .instruction_handles = lw.instruction_handle_scratch,
            .instruction_views = lw.instruction_view_scratch,
            .runtime_meta = .{
                .model_config = config,
                .residual_multiplier = weight_handles.residual_multiplier,
                .use_gelu = weight_handles.use_gelu,
                .attention_multiplier = weight_handles.attention_multiplier,
                .attention_storage_kind = lw.attentionStorageKind(),
                .gated_delta_storage_kind = lw.gatedDeltaStorageKind(),
                .shortconv_storage_kind = lw.shortconvStorageKind(),
                .ffn_storage_kind = lw.ffnStorageKind(),
                .mla_storage_kind = lw.mlaStorageKind(),
                .mamba_storage_kind = lw.mambaStorageKind(),
                .mla_config = lw.mla_config,
                .gated_delta_d_conv = lw.gated_delta_d_conv,
                .gated_delta_n_heads = lw.gated_delta_n_heads,
                .gated_delta_n_key_heads = lw.gated_delta_n_key_heads,
                .gated_delta_d_head = lw.gated_delta_d_head,
                .shortconv_d_conv = lw.shortconv_d_conv,
                .shortconv_conv_dim = lw.shortconv_conv_dim,
                .moe_router_group_size = if (lw.moe) |moe| moe.router_group_size else 0,
                .moe_expert_group_size = if (lw.moe) |moe| moe.expert_group_size else 0,
                .moe_num_experts = if (lw.moe) |moe| moe.num_experts else 0,
                .moe_experts_per_token = if (lw.moe) |moe| moe.experts_per_token else 0,
                .mamba_d_state = lw.mamba_d_state,
                .mamba_d_conv = lw.mamba_d_conv,
                .mamba_n_heads = lw.mamba_n_heads,
                .mamba_d_head = lw.mamba_d_head,
                .mamba_n_groups = lw.mamba_n_groups,
                .mamba_gate_up_layout = @intFromEnum(lw.mamba_gate_up_layout),
            },
            .resolved_weight_ptrs = lw.weight_ptr_scratch,
        };
        buildRmsNormWeightOrder(&exec_ctx);
        if (exec_ctx.resolved_weight_ptrs.len != lw.weight_binding_keys.len) {
            return error.InvalidWeightRefCount;
        }
        try bindLayerProgramStateDescriptors(&exec_ctx, &compiled_plan.plan, state_blocks);

        var insn_idx: usize = 0;
        // XRAY ACCEPTABLE USE:
        // Fusion policy is a production performance decision. Verify/tracing
        // must not alter fusion on/off.
        const allow_rms_swiglu_fusion = std.posix.getenv("TALU_METAL_DISABLE_RMS_SWIGLU_FUSION") == null;
        while (insn_idx < compiled_plan.plan.instructions.len) {
            const insn = &compiled_plan.plan.instructions[insn_idx];
            if (insn_idx + 1 < compiled_plan.plan.instructions.len) {
                const next_insn = &compiled_plan.plan.instructions[insn_idx + 1];
                if (allow_rms_swiglu_fusion and canFuseDenseRmsNormSwiGlu(&exec_ctx, insn, next_insn)) {
                    if (enable_dispatch_observability) {
                        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
                        runtime_contract.recordExecutionDispatch(&rt_ctx, next_insn.opcode);
                    }
                    try runDenseRmsNormSwiGluFusion(&exec_ctx, insn, next_insn);
                    insn_idx += 2;
                    continue;
                }
                if (allow_rms_swiglu_fusion and canFuseQuantizedRmsNormSwiGlu(&exec_ctx, insn, next_insn)) {
                    if (enable_dispatch_observability) {
                        runtime_contract.recordExecutionDispatch(&rt_ctx, insn.opcode);
                        runtime_contract.recordExecutionDispatch(&rt_ctx, next_insn.opcode);
                    }
                    try runQuantizedRmsNormSwiGluFusion(&exec_ctx, insn, next_insn);
                    insn_idx += 2;
                    continue;
                }
            }
            try dispatchLayerProgramInstruction(insn, &exec_ctx, &rt_ctx);
            insn_idx += 1;
        }

        const final_register = runtime_contract.planFinalOutputRegister(&compiled_plan.plan);
        const final_slot = try bufferSlotForRegister(final_register, &residual, slot_buffers, lw.register_to_slot_map);
        return final_slot.*;
    }

    pub fn forward(
        hidden: mlx_graph.ArrayHandle,
        layer_weights: *const LayerWeights,
        layer_idx: usize,
        config: ModelConfig,
        weight_handles: *const WeightHandles,
        state_blocks: []const runtime_contract.StateBlockHandle,
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
        return forwardWithProgram(
            hidden,
            lw,
            config,
            weight_handles,
            layer_idx,
            state_blocks,
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
        return projectLogitsFromNormedHidden(final_normed, weight_handles);
    }

    pub fn projectLogitsFromNormedHidden(
        final_normed: mlx_graph.ArrayHandle,
        weight_handles: anytype,
    ) mlx_graph.ArrayHandle {
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
            break :blk mlx_graph.mlx_lazy_matmul(final_normed, weight_handles.lm_head.?);
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
        var hidden_shape: [8]usize = undefined;
        const hidden_rank = mlx_graph.getShape(hidden, &hidden_shape);
        const feature_dim: usize = if (hidden_rank >= 1) hidden_shape[hidden_rank - 1] else 0;
        var final_norm_weight = weight_handles.ln_final;
        if (feature_dim > 0) {
            if (normalizeRmsNormWeightForFeatureDim(final_norm_weight, feature_dim)) |normalized| {
                final_norm_weight = normalized;
            } else {
                final_norm_weight = synthesizeUnitRmsNormWeight(feature_dim);
            }
        }
        var final_weight_shape: [8]usize = undefined;
        if (mlx_graph.getShape(final_norm_weight, &final_weight_shape) != 1 and feature_dim > 0) {
            final_norm_weight = synthesizeUnitRmsNormWeight(feature_dim);
        }
        const final_norm = norm_kernel.RMSNorm{
            .weight = final_norm_weight,
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
    try std.testing.expectEqual(layer_ops.BufferId.residual, layer_ops.finalOutputBuffer(&program));
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
    try std.testing.expectEqual(layer_ops.BufferId.norm_out, layer_ops.finalOutputBuffer(&program));
}

test "layer program contract accepts kernel-add programs" {
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
    try std.testing.expect(
        runtime_contract.firstLayerProgramCompatibilityIssue(
            &program,
            .attention_mlp,
            TransformerBlock.layer_program_adapter_table,
        ) == null,
    );
}

test "canFuseDenseRmsNormSwiGlu accepts direct dense norm to swiglu handoff" {
    var ctx = std.mem.zeroes(TransformerBlock.LayerProgramExecutionContext);
    ctx.runtime_meta.ffn_storage_kind = .dense;

    const norm_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const norm_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const norm_weights = [_]runtime_contract.WeightRef{
        .{ .index = 0 },
        .{ .index = 1 },
    };
    const swiglu_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const swiglu_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const swiglu_weights = [_]runtime_contract.WeightRef{
        .{ .index = 2 },
        .{ .index = 3 },
        .{ .index = 4 },
        .{ .index = 5 },
        .{ .index = 6 },
    };
    const norm_insn = runtime_contract.Instruction{
        .opcode = .rmsnorm,
        .inputs = &norm_inputs,
        .outputs = &norm_outputs,
        .weights = &norm_weights,
        .param_block_id = null,
        .state_block_id = null,
    };
    const swiglu_insn = runtime_contract.Instruction{
        .opcode = .swiglu,
        .inputs = &swiglu_inputs,
        .outputs = &swiglu_outputs,
        .weights = &swiglu_weights,
        .param_block_id = null,
        .state_block_id = null,
    };

    try std.testing.expect(TransformerBlock.canFuseDenseRmsNormSwiGlu(&ctx, &norm_insn, &swiglu_insn));
}

test "canFuseDenseRmsNormSwiGlu rejects mismatched registers" {
    var ctx = std.mem.zeroes(TransformerBlock.LayerProgramExecutionContext);
    ctx.runtime_meta.ffn_storage_kind = .dense;

    const norm_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const norm_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const norm_weights = [_]runtime_contract.WeightRef{
        .{ .index = 0 },
        .{ .index = 1 },
    };
    const swiglu_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(3)};
    const swiglu_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const swiglu_weights = [_]runtime_contract.WeightRef{
        .{ .index = 2 },
        .{ .index = 3 },
        .{ .index = 4 },
        .{ .index = 5 },
        .{ .index = 6 },
    };
    const norm_insn = runtime_contract.Instruction{
        .opcode = .rmsnorm,
        .inputs = &norm_inputs,
        .outputs = &norm_outputs,
        .weights = &norm_weights,
        .param_block_id = null,
        .state_block_id = null,
    };
    const swiglu_insn = runtime_contract.Instruction{
        .opcode = .swiglu,
        .inputs = &swiglu_inputs,
        .outputs = &swiglu_outputs,
        .weights = &swiglu_weights,
        .param_block_id = null,
        .state_block_id = null,
    };

    try std.testing.expect(!TransformerBlock.canFuseDenseRmsNormSwiGlu(&ctx, &norm_insn, &swiglu_insn));
}

test "canFuseQuantizedRmsNormSwiGlu accepts direct quantized norm to swiglu handoff" {
    var ctx = std.mem.zeroes(TransformerBlock.LayerProgramExecutionContext);
    ctx.runtime_meta.ffn_storage_kind = .quantized;

    const norm_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const norm_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const norm_weights = [_]runtime_contract.WeightRef{
        .{ .index = 0 },
        .{ .index = 1 },
    };
    const swiglu_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const swiglu_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const swiglu_weights = [_]runtime_contract.WeightRef{
        .{ .index = 2 },
        .{ .index = 3 },
        .{ .index = 4 },
        .{ .index = 5 },
        .{ .index = 6 },
    };
    const norm_insn = runtime_contract.Instruction{
        .opcode = .rmsnorm,
        .inputs = &norm_inputs,
        .outputs = &norm_outputs,
        .weights = &norm_weights,
        .param_block_id = null,
        .state_block_id = null,
    };
    const swiglu_insn = runtime_contract.Instruction{
        .opcode = .swiglu,
        .inputs = &swiglu_inputs,
        .outputs = &swiglu_outputs,
        .weights = &swiglu_weights,
        .param_block_id = null,
        .state_block_id = null,
    };

    try std.testing.expect(TransformerBlock.canFuseQuantizedRmsNormSwiGlu(&ctx, &norm_insn, &swiglu_insn));
}

test "canFuseQuantizedRmsNormSwiGlu rejects dense storage kind" {
    var ctx = std.mem.zeroes(TransformerBlock.LayerProgramExecutionContext);
    ctx.runtime_meta.ffn_storage_kind = .dense;

    const norm_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(0)};
    const norm_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const norm_weights = [_]runtime_contract.WeightRef{
        .{ .index = 0 },
        .{ .index = 1 },
    };
    const swiglu_inputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(1)};
    const swiglu_outputs = [_]runtime_contract.RegisterRef{runtime_contract.registerFromIndex(2)};
    const swiglu_weights = [_]runtime_contract.WeightRef{
        .{ .index = 2 },
        .{ .index = 3 },
        .{ .index = 4 },
        .{ .index = 5 },
        .{ .index = 6 },
    };
    const norm_insn = runtime_contract.Instruction{
        .opcode = .rmsnorm,
        .inputs = &norm_inputs,
        .outputs = &norm_outputs,
        .weights = &norm_weights,
        .param_block_id = null,
        .state_block_id = null,
    };
    const swiglu_insn = runtime_contract.Instruction{
        .opcode = .swiglu,
        .inputs = &swiglu_inputs,
        .outputs = &swiglu_outputs,
        .weights = &swiglu_weights,
        .param_block_id = null,
        .state_block_id = null,
    };

    try std.testing.expect(!TransformerBlock.canFuseQuantizedRmsNormSwiGlu(&ctx, &norm_insn, &swiglu_insn));
}

test "layer_program_adapter_table covers Metal LayerOp execution subset" {
    const supported = [_]opcode_map.Opcode{
        .rmsnorm,
        .multihead_attention,
        .mla_attention,
        .gated_delta_net,
        .swiglu,
        .moe,
        .mamba_mixer,
        .shortconv,
        .residual_add,
    };
    for (supported) |opcode| {
        try std.testing.expect(TransformerBlock.layer_program_adapter_table[@intFromEnum(opcode)] != null);
    }

    // Vision opcodes are now registered in the main table
    for (vision_adapters.required_opcodes) |opcode| {
        try std.testing.expect(TransformerBlock.layer_program_adapter_table[@intFromEnum(opcode)] != null);
    }

    try std.testing.expect(TransformerBlock.layer_program_adapter_table[@intFromEnum(opcode_map.Opcode.mul_scalar)] == null);
}

test "layer program contract rejects unsupported primitive ops" {
    const program = [_]layer_ops.LayerOp{
        .{ .mul_scalar = .{
            .in = .residual,
            .out = .residual,
            .scalar = 0.5,
        } },
    };
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        TransformerBlock.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .unsupported_opcode => |unsupported| try std.testing.expectEqual(opcode_map.Opcode.mul_scalar, unsupported.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "layer program contract rejects stateful opcode bound to wrong block kind" {
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
    const issue = runtime_contract.firstLayerProgramCompatibilityIssue(
        &program,
        .attention_mlp,
        TransformerBlock.layer_program_adapter_table,
    ) orelse return error.TestUnexpectedResult;
    switch (issue) {
        .state_mismatch => |mismatch| try std.testing.expectEqual(opcode_map.Opcode.shortconv, mismatch.opcode),
        else => return error.TestUnexpectedResult,
    }
}

test "shouldIncludeQkNormInRmsOrder includes qk norms when only norm slots exist" {
    var layer = std.mem.zeroes(LayerWeights);
    layer.q_norm = @as(mlx_graph.ArrayHandle, @ptrFromInt(1));
    layer.k_norm = @as(mlx_graph.ArrayHandle, @ptrFromInt(2));
    layer.weight_binding_keys = &[_]WeightHandles.LayerProgramWeightBindingKey{
        .norm_weight,
        .norm_weight,
        .norm_weight,
        .norm_weight,
    };
    try std.testing.expect(
        TransformerBlock.shouldIncludeQkNormInRmsOrder(&layer, 4),
    );
}

test "shouldIncludeQkNormInRmsOrder disables qk norms when explicit qk slots exist" {
    var layer = std.mem.zeroes(LayerWeights);
    layer.q_norm = @as(mlx_graph.ArrayHandle, @ptrFromInt(1));
    layer.k_norm = @as(mlx_graph.ArrayHandle, @ptrFromInt(2));
    layer.weight_binding_keys = &[_]WeightHandles.LayerProgramWeightBindingKey{
        .norm_weight,
        .attention_q_norm,
        .attention_k_norm,
        .norm_weight,
    };
    try std.testing.expect(
        !TransformerBlock.shouldIncludeQkNormInRmsOrder(&layer, 4),
    );
}

test "tracePositionForPoint prefill uses sequence length across points" {
    const seq_len: u32 = 14;
    const pos_offset: usize = 0;
    try std.testing.expectEqual(@as(u32, 14), TransformerBlock.tracePositionForPoint(.layer_attn_norm, pos_offset, seq_len));
    try std.testing.expectEqual(@as(u32, 14), TransformerBlock.tracePositionForPoint(.attn_q, pos_offset, seq_len));
    try std.testing.expectEqual(@as(u32, 14), TransformerBlock.tracePositionForPoint(.attn_out, pos_offset, seq_len));
}

test "tracePositionForPoint decode matches CPU trace semantics" {
    const seq_len: u32 = 1;
    const pos_offset: usize = 14;
    try std.testing.expectEqual(@as(u32, 1), TransformerBlock.tracePositionForPoint(.layer_attn_norm, pos_offset, seq_len));
    try std.testing.expectEqual(@as(u32, 14), TransformerBlock.tracePositionForPoint(.attn_q, pos_offset, seq_len));
    try std.testing.expectEqual(@as(u32, 15), TransformerBlock.tracePositionForPoint(.attn_out, pos_offset, seq_len));
    try std.testing.expectEqual(@as(u32, 1), TransformerBlock.tracePositionForPoint(.ffn_down, pos_offset, seq_len));
}

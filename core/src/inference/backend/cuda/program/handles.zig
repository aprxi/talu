//! Layer program dispatch, adapter implementations, kernel initialization.
//!
//! Contains the per-instruction adapter implementations (norm, attention,
//! short conv, gated delta, SwiGLU, residual add), the layer program
//! dispatch loop, tryExecuteLayerProgram, runAttentionContext, kernel
//! function resolution, and CPU ROPE initialization.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const build_options = @import("build_options");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const models = @import("models_pkg");
const layer_ops = models.layer_ops;
const op_types = models.op_types;
const opcode_map = models.plan.opcode_map;
const rope_scaling = models.rope_scaling;
const runtime_contract = @import("runtime_contract_pkg");
const attention_mod = @import("../attention_path.zig");
const smoke_checks = @import("../selftest.zig");
const cpu_kernels = @import("../../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;

// --- Types from engine.zig (mutual import) ---
const engine = @import("../engine.zig");
const CudaBackend = engine.CudaBackend;
const LayerProgramExecutionContext = CudaBackend.LayerProgramExecutionContext;
const BuiltLayerProgramHandles = CudaBackend.BuiltLayerProgramHandles;
const LayerProgramInstructionStateBlocks = CudaBackend.LayerProgramInstructionStateBlocks;
const layer_program_adapter_table = CudaBackend.layer_program_adapter_table;
const traceShapeBsd = CudaBackend.traceShapeBsd;
const traceTokenIndex = CudaBackend.traceTokenIndex;
const tracePositionForPoint = CudaBackend.tracePositionForPoint;
const layerProgramExecutionState = CudaBackend.layerProgramExecutionState;
const layer_program_adapter_capabilities = CudaBackend.layer_program_adapter_capabilities;

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
const DeviceTensor = engine_types.DeviceTensor;
const LinearWeight = engine_types.LinearWeight;
const BlockRuntimeLayer = engine_types.BlockRuntimeLayer;
const BlockRuntime = engine_types.BlockRuntime;
const RuntimeBuffers = engine_types.RuntimeBuffers;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LayerAttentionExecConfig = engine_types.LayerAttentionExecConfig;
const ShortConvBlockRuntime = engine_types.ShortConvBlockRuntime;
const ShortConvExecConfig = engine_types.ShortConvExecConfig;
const GatedDeltaBlockRuntime = engine_types.GatedDeltaBlockRuntime;
const KvRuntimeState = engine_types.KvRuntimeState;
const ShortConvRuntimeState = engine_types.ShortConvRuntimeState;
const GatedDeltaRuntimeState = engine_types.GatedDeltaRuntimeState;
const AttentionKernelSet = engine_types.AttentionKernelSet;
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KernelSlot = engine_types.KernelSlot;
const RequiredKernel = engine_types.RequiredKernel;
const required_kernels = engine_types.required_kernels;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_fused_attention_f16_kv = engine_types.enable_fused_attention_f16_kv;
const max_fused_attention_f16_kv_seq_len = engine_types.max_fused_attention_f16_kv_seq_len;
const max_supported_fused_f16_kv_head_dim = engine_types.max_supported_fused_f16_kv_head_dim;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const attention_policy_config = engine_types.attention_policy_config;
const missing_device_tensor = engine_types.missing_device_tensor;
const expectedAttentionQProjectionDim = engine_types.expectedAttentionQProjectionDim;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;
const GaffineU4LinearWeight = engine_types.GaffineU4LinearWeight;
const GaffineU8LinearWeight = engine_types.GaffineU8LinearWeight;
const Nvfp4LinearWeight = engine_types.Nvfp4LinearWeight;
const AttentionPath = engine_types.AttentionPath;
const gqa_prefill_f16_dynamic_smem_bytes: u32 = 65536;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("../operators/root.zig");

// --- Mixer functions from engine_mixers.zig ---
const engine_mixers = @import("../operators/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;

const ResolvedAttentionShape = struct {
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
};

fn resolveAttentionShapeForInstruction(
    cfg: *const LayerAttentionExecConfig,
    q_norm_weight: ?*const DeviceTensor,
    k_norm_weight: ?*const DeviceTensor,
    ctx: *const LayerProgramExecutionContext,
) !ResolvedAttentionShape {
    var head_dim_u32 = ctx.head_dim_u32;
    if (q_norm_weight) |q_norm| {
        head_dim_u32 = @intCast(q_norm.rows);
    } else if (k_norm_weight) |k_norm| {
        head_dim_u32 = @intCast(k_norm.rows);
    }
    if (head_dim_u32 == 0) return error.InvalidInstructionBinding;
    const head_dim_usize: usize = @intCast(head_dim_u32);

    var n_heads_u32 = ctx.n_heads_u32;
    if ((cfg.q_dim % head_dim_usize) == 0) {
        n_heads_u32 = @intCast(cfg.q_dim / head_dim_usize);
    } else if (q_norm_weight != null) {
        return error.InvalidInstructionBinding;
    }

    var n_kv_heads_u32 = ctx.n_kv_heads_u32;
    if ((cfg.kv_dim % head_dim_usize) == 0) {
        n_kv_heads_u32 = @intCast(cfg.kv_dim / head_dim_usize);
    } else if (k_norm_weight != null) {
        return error.InvalidInstructionBinding;
    }
    if (n_heads_u32 == 0 or n_kv_heads_u32 == 0 or (n_heads_u32 % n_kv_heads_u32) != 0) {
        return error.InvalidInstructionBinding;
    }

    var rope_dim_u32 = @min(ctx.rope_dim_u32, head_dim_u32);
    // Mixed-attention models use proportional rotary width for full-attention
    // layers, but sliding-window layers run local RoPE across the full per-head
    // width.
    if (cfg.sliding_window > 0 and ctx.backend.loaded.config.global_head_dim > 0) {
        rope_dim_u32 = head_dim_u32;
    }

    return .{
        .head_dim_u32 = head_dim_u32,
        .rope_dim_u32 = rope_dim_u32,
        .n_heads_u32 = n_heads_u32,
        .n_kv_heads_u32 = n_kv_heads_u32,
    };
}

pub fn layerProgramStateBlocksForInstruction(
    insn: *const runtime_contract.Instruction,
    ctx: *LayerProgramExecutionContext,
) !LayerProgramInstructionStateBlocks {
    var blocks = LayerProgramInstructionStateBlocks{};
    const state_id = insn.state_block_id orelse return blocks;
    const descriptor = runtime_contract.findStateDescriptor(&ctx.layer.compiled_plan.?.plan, state_id) orelse {
        return error.UnknownStateDescriptorId;
    };
    const slot_block = runtime_contract.findStateBlock(
        ctx.backend.slotStateBlocks(ctx.slot_index),
        state_id,
    ) orelse return error.InvalidStateDescriptorBinding;
    if (slot_block.align_bytes < descriptor.align_bytes) return error.InvalidStateDescriptorBinding;
    if (descriptor.size_bytes > 0 and slot_block.size < descriptor.size_bytes) return error.InvalidStateDescriptorBinding;
    blocks.handles[0] = slot_block.*;
    blocks.len = 1;
    return blocks;
}

pub fn bufferFromTensorHandle(handle: runtime_contract.TensorHandle) *compute.cuda.Buffer {
    return @ptrCast(@alignCast(handle.ptr));
}

pub fn instructionIoSlices(
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

pub fn instructionWeightSlice(
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
) ![]const runtime_contract.TensorHandle {
    const io_count = insn.inputs.len + insn.outputs.len;
    if (registers.len < io_count) return error.InvalidInstructionBinding;
    const weights = registers[io_count..];
    if (weights.len != insn.weights.len) return error.InvalidWeightRefCount;
    return weights;
}

pub fn layerProgramInstructionHandleCapacity(plan: *const runtime_contract.ExecutionPlan) usize {
    var max_handles: usize = 0;
    for (plan.instructions) |insn| {
        const handle_count = insn.inputs.len + insn.outputs.len + insn.weights.len;
        if (handle_count > max_handles) max_handles = handle_count;
    }
    return max_handles;
}

pub fn deviceTensorFromWeightHandle(handle: runtime_contract.TensorHandle) *const DeviceTensor {
    return @ptrCast(@alignCast(handle.ptr));
}

pub fn optionalDeviceTensorFromWeightHandle(handle: runtime_contract.TensorHandle) ?*const DeviceTensor {
    const value: *const DeviceTensor = @ptrCast(@alignCast(handle.ptr));
    if (value == &missing_device_tensor) return null;
    return value;
}

pub fn linearWeightFromWeightHandle(handle: runtime_contract.TensorHandle) *const LinearWeight {
    return @ptrCast(@alignCast(handle.ptr));
}

pub fn decodeResidualScaleFromParams(params: []const runtime_contract.ParamBlock) !layer_ops.ResidualScale {
    const p = try runtime_contract.paramAs(runtime_contract.ResidualAddParam, params, .residual_add);
    return switch (p.scale_tag) {
        0 => .one,
        1 => .residual_multiplier,
        2 => .{ .literal = @bitCast(p.scale_literal) },
        else => error.InvalidParamBlockABI,
    };
}

pub fn requireLayerProgramRuntimeState(
    ctx: *LayerProgramExecutionContext,
    insn: *const runtime_contract.Instruction,
    state_blocks: []const runtime_contract.StateBlockHandle,
) !*BlockRuntimeLayer {
    const state_id = insn.state_block_id orelse return ctx.layer;
    if (runtime_contract.findStateBlock(state_blocks, state_id) == null) {
        return error.InvalidStateDescriptorBinding;
    }
    return ctx.layer;
}

pub fn requireStateValue(
    comptime T: type,
    state_blocks: []const runtime_contract.StateBlockHandle,
    state_id: u8,
) !*T {
    const block = runtime_contract.findStateBlock(state_blocks, state_id) orelse {
        return error.InvalidStateDescriptorBinding;
    };
    const value = runtime_contract.stateValueFromBlock(*T, block) orelse {
        return error.InvalidStateDescriptorBinding;
    };
    return value;
}

pub fn requireAttentionRuntimeBinding(state: *const KvRuntimeState, layer_index: usize) !*LayerAttentionRuntime {
    if (layer_index >= state.block_runtime.blocks.len) {
        log.warn("inference", "requireAttentionRuntimeBinding OOB", .{
            .layer_index = layer_index,
            .blocks_len = state.block_runtime.blocks.len,
            .runtime_kind = state.runtime_kind,
            .slot_index = state.slot_index,
        });
        return error.InvalidInstructionIndex;
    }
    return state.block_runtime.blocks[layer_index].attention_binding orelse error.InvalidStateDescriptorBinding;
}

pub fn requireShortConvRuntimeBinding(state: *const ShortConvRuntimeState, layer_index: usize) !*ShortConvBlockRuntime {
    if (layer_index >= state.block_runtime.blocks.len) return error.InvalidInstructionIndex;
    return state.block_runtime.blocks[layer_index].shortconv_binding orelse error.InvalidStateDescriptorBinding;
}

pub fn requireGatedDeltaRuntimeBinding(state: *const GatedDeltaRuntimeState, layer_index: usize) !*GatedDeltaBlockRuntime {
    if (layer_index >= state.block_runtime.blocks.len) return error.InvalidInstructionIndex;
    return state.block_runtime.blocks[layer_index].gated_delta_binding orelse error.InvalidStateDescriptorBinding;
}

pub fn instructionParams(
    insn: *const runtime_contract.Instruction,
    compiled: *const runtime_contract.CompiledPlan,
    storage: *[1]runtime_contract.ParamBlock,
) ![]const runtime_contract.ParamBlock {
    const param_id = insn.param_block_id orelse return &.{};
    if (param_id >= compiled.param_blocks.len) return error.MissingParamBlock;
    storage[0] = compiled.param_blocks[param_id];
    return storage[0..1];
}

pub fn tensorViewDescForCudaBuffer() runtime_contract.TensorViewDesc {
    return .{
        .dtype = .f32,
        .rank = 0,
        .shape = .{ 0, 0, 0, 0 },
        .stride_elems = .{ 0, 0, 0, 0 },
        .layout = .backend_native,
    };
}

pub fn layerProgramWeightHandlePtr(ctx: *LayerProgramExecutionContext, slot_idx: usize) !*anyopaque {
    if (ctx.op_index + 1 >= ctx.layer.instruction_weight_offsets.len) return error.InvalidInstructionBinding;
    const start: usize = ctx.layer.instruction_weight_offsets[ctx.op_index];
    const end: usize = ctx.layer.instruction_weight_offsets[ctx.op_index + 1];
    if (end < start) return error.InvalidInstructionBinding;
    const count = end - start;
    if (slot_idx >= count) return error.InvalidWeightRefCount;
    const idx = start + slot_idx;
    if (idx >= ctx.layer.instruction_weight_ptrs.len) return error.InvalidInstructionBinding;
    return ctx.layer.instruction_weight_ptrs[idx] orelse error.MissingWeight;
}

pub fn buildLayerProgramInstructionHandles(
    self: anytype,
    insn: *const runtime_contract.Instruction,
    ctx: *LayerProgramExecutionContext,
    handle_storage: []runtime_contract.TensorHandle,
    view_storage: []runtime_contract.TensorViewDesc,
) !BuiltLayerProgramHandles {
    var handle_count: usize = 0;
    var view_count: usize = 0;

    for (insn.inputs) |reg| {
        if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
        const reg_idx = runtime_contract.registerToIndex(reg);
        const input = engine_ops.programBuffer(self, reg_idx, ctx) orelse return error.UnsupportedModel;
        handle_storage[handle_count] = .{
            .register = reg,
            .ptr = @ptrCast(input),
        };
        view_storage[view_count] = tensorViewDescForCudaBuffer();
        handle_count += 1;
        view_count += 1;
    }
    for (insn.outputs) |reg| {
        if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
        const reg_idx = runtime_contract.registerToIndex(reg);
        const output = engine_ops.programBuffer(self, reg_idx, ctx) orelse return error.UnsupportedModel;
        handle_storage[handle_count] = .{
            .register = reg,
            .ptr = @ptrCast(output),
        };
        view_storage[view_count] = tensorViewDescForCudaBuffer();
        handle_count += 1;
        view_count += 1;
    }
    for (insn.weights, 0..) |_, slot_idx| {
        if (handle_count >= handle_storage.len) return error.InvalidInstructionBinding;
        const weight_ptr = try layerProgramWeightHandlePtr(ctx, slot_idx);
        handle_storage[handle_count] = .{
            .register = runtime_contract.registerFromIndex(@intCast(slot_idx)),
            .ptr = weight_ptr,
        };
        view_storage[view_count] = tensorViewDescForCudaBuffer();
        handle_count += 1;
        view_count += 1;
    }

    return .{
        .registers = handle_storage[0..handle_count],
        .views = view_storage[0..view_count],
    };
}

pub fn recordLayerProgramDispatch(self: anytype, opcode: opcode_map.Opcode) void {
    const opcode_idx = @intFromEnum(opcode);
    self.layer_program_dispatch_total[opcode_idx] +%= 1;
    if (enable_dispatch_observability) {
        self.runtime_dispatch_counters.record(opcode);
    }
}

pub fn prefillDispatchDelta(self: anytype, opcode: opcode_map.Opcode) u64 {
    const opcode_idx = @intFromEnum(opcode);
    return self.layer_program_dispatch_total[opcode_idx] - self.prefill_dispatch_window_start[opcode_idx];
}

pub fn prefillDispatchTotal(self: anytype) u64 {
    var total: u64 = 0;
    for (0..self.layer_program_dispatch_total.len) |idx| {
        total += self.layer_program_dispatch_total[idx] - self.prefill_dispatch_window_start[idx];
    }
    return total;
}

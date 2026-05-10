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

const handles = @import("handles.zig");
const requireShortConvRuntimeBinding = handles.requireShortConvRuntimeBinding;
const requireGatedDeltaRuntimeBinding = handles.requireGatedDeltaRuntimeBinding;
const instructionWeightSlice = handles.instructionWeightSlice;
const instructionParams = handles.instructionParams;
const linearWeightFromWeightHandle = handles.linearWeightFromWeightHandle;
const deviceTensorFromWeightHandle = handles.deviceTensorFromWeightHandle;
const optionalDeviceTensorFromWeightHandle = handles.optionalDeviceTensorFromWeightHandle;
const instructionIoSlices = handles.instructionIoSlices;
const bufferFromTensorHandle = handles.bufferFromTensorHandle;
const requireStateValue = handles.requireStateValue;

pub fn layerProgramShortConvAdapter(
    self: anytype,
    layer: *BlockRuntimeLayer,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    state_blocks: []const runtime_contract.StateBlockHandle,
    ctx: *LayerProgramExecutionContext,
) !void {
    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
    const weight_handles = try instructionWeightSlice(insn, registers);
    if (weight_handles.len != 4) return error.InvalidWeightRefCount;
    const input = bufferFromTensorHandle(io.inputs[0]);
    const output = bufferFromTensorHandle(io.outputs[0]);
    const cfg = try layer.instructionShortConvRef(ctx.op_index);
    const in_proj = linearWeightFromWeightHandle(weight_handles[0]).*;
    const conv_weight = deviceTensorFromWeightHandle(weight_handles[1]).*;
    const out_proj = linearWeightFromWeightHandle(weight_handles[2]).*;
    const conv_bias = optionalDeviceTensorFromWeightHandle(weight_handles[3]);
    if (in_proj.cols() != (3 * cfg.conv_dim)) return error.InvalidInstructionBinding;
    if (out_proj.cols() != self.d_model) return error.InvalidInstructionBinding;
    const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
    const shortconv_state = try requireStateValue(ShortConvRuntimeState, state_blocks, state_id);
    if (shortconv_state.runtime_kind != runtime_contract.state_runtime_kind_shortconv_cache) {
        return error.InvalidStateDescriptorBinding;
    }
    const shortconv_binding = try requireShortConvRuntimeBinding(shortconv_state, ctx.layer_index);

    // Batched decode path: per-slot shortconv state from slot_kv_states.
    if (ctx.batch_info) |batch| {
        var row_idx: usize = 0;
        while (row_idx < ctx.active_rows_u32) : (row_idx += 1) {
            const slot_idx = batch.slot_indices[row_idx];
            var sc_state = &self.slot_kv_states[slot_idx].sc[batch.sc_layer_index];
            var input_row = try logicalF32RowSlice(
                input,
                @intCast(ctx.active_rows_u32),
                row_idx,
                in_proj.rows(),
            );
            var output_row = try logicalF32RowSlice(
                output,
                @intCast(ctx.active_rows_u32),
                row_idx,
                out_proj.cols(),
            );
            try engine_mixers.runShortConvMixerStep(
                self,
                cfg,
                &sc_state.conv,
                &in_proj,
                &out_proj,
                &conv_weight,
                conv_bias,
                &input_row,
                &output_row,
                ctx.shortconv_step_function,
            );
        }
        return;
    }

    // Single-row fast path (decode, single token).
    if (ctx.active_rows_u32 <= 1) {
        try engine_mixers.runShortConvMixerStep(
            self,
            cfg,
            &shortconv_binding.conv_state,
            &in_proj,
            &out_proj,
            &conv_weight,
            conv_bias,
            input,
            output,
            ctx.shortconv_step_function,
        );
        return;
    }

    // Non-batched multi-row path (prefill): row loop with layer-local state.
    var row_idx: usize = 0;
    while (row_idx < ctx.active_rows_u32) : (row_idx += 1) {
        var input_row = try logicalF32RowSlice(
            input,
            @intCast(ctx.active_rows_u32),
            row_idx,
            in_proj.rows(),
        );
        var output_row = try logicalF32RowSlice(
            output,
            @intCast(ctx.active_rows_u32),
            row_idx,
            out_proj.cols(),
        );
        try engine_mixers.runShortConvMixerStep(
            self,
            cfg,
            &shortconv_binding.conv_state,
            &in_proj,
            &out_proj,
            &conv_weight,
            conv_bias,
            &input_row,
            &output_row,
            ctx.shortconv_step_function,
        );
    }
}

pub fn layerProgramGatedDeltaAdapter(
    self: anytype,
    layer: *BlockRuntimeLayer,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    state_blocks: []const runtime_contract.StateBlockHandle,
    params: []const runtime_contract.ParamBlock,
    ctx: *LayerProgramExecutionContext,
) !void {
    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
    _ = try runtime_contract.paramAs(runtime_contract.GatedDeltaKernelParam, params, .gated_delta_net);
    const input = bufferFromTensorHandle(io.inputs[0]);
    const output = bufferFromTensorHandle(io.outputs[0]);
    const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
    const gated_delta_state = try requireStateValue(GatedDeltaRuntimeState, state_blocks, state_id);
    if (gated_delta_state.runtime_kind != runtime_contract.state_runtime_kind_gated_delta_cache) {
        return error.InvalidStateDescriptorBinding;
    }
    const binding = try requireGatedDeltaRuntimeBinding(gated_delta_state, ctx.layer_index);
    const expected_rows: usize = @intCast(ctx.active_rows_u32);
    const expected_bytes = std.math.mul(usize, expected_rows, self.d_model * @sizeOf(f32)) catch return error.InvalidArgument;
    if (input.size != expected_bytes or output.size != expected_bytes) {
        log.warn("inference", "CUDA gated-delta row count mismatch", .{
            .layer_index = ctx.layer_index,
            .op_index = ctx.op_index,
            .expected_rows = expected_rows,
            .input_bytes = input.size,
            .output_bytes = output.size,
            .expected_bytes = expected_bytes,
            .d_model = self.d_model,
        });
        return error.InvalidInstructionBinding;
    }
    if (ctx.batch_info) |batch| {
        try engine_mixers.runBatchedDecodeGatedDeltaMixer(
            self,
            binding,
            input,
            output,
            ctx,
            batch,
        );
        _ = layer;
        return;
    }
    try engine_mixers.runGatedDeltaMixerStep(
        self,
        binding,
        input,
        output,
        expected_rows,
    );
    _ = layer;
}

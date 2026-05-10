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
const engine_types = @import("../runtime/_types_impl.zig");
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
const decodeResidualScaleFromParams = handles.decodeResidualScaleFromParams;
const instructionWeightSlice = handles.instructionWeightSlice;
const instructionParams = handles.instructionParams;
const optionalDeviceTensorFromWeightHandle = handles.optionalDeviceTensorFromWeightHandle;
const deviceTensorFromWeightHandle = handles.deviceTensorFromWeightHandle;
const instructionIoSlices = handles.instructionIoSlices;
const bufferFromTensorHandle = handles.bufferFromTensorHandle;
const common = @import("common.zig");
const tryFuseResidualAddIntoNextRmsnorm = common.tryFuseResidualAddIntoNextRmsnorm;
const residualScaleFactor = common.residualScaleFactor;
const runResidualAddRmsnormRowsStrideAware = common.runResidualAddRmsnormRowsStrideAware;
const standaloneLayerScalarOutputScale = common.standaloneLayerScalarOutputScale;

pub fn layerProgramNormAdapter(
    self: anytype,
    _: *BlockRuntimeLayer,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    ctx: *LayerProgramExecutionContext,
) !void {
    if (self.skip_next_rmsnorm) {
        self.skip_next_rmsnorm = false;
        return;
    }
    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
    const weight_handles = try instructionWeightSlice(insn, registers);
    if (weight_handles.len != 2) return error.InvalidWeightRefCount;
    const input = bufferFromTensorHandle(io.inputs[0]);
    const output = bufferFromTensorHandle(io.outputs[0]);
    const weight = deviceTensorFromWeightHandle(weight_handles[0]);
    const rmsnorm_start_ns: i128 = std.time.nanoTimestamp();
    engine_ops.runRmsnormRowsStrideAware(self, input, &weight.buffer, output, ctx.active_rows_u32, ctx.d_model_u32) catch |err| {
        if (err == error.InvalidArgument) {
            const expected_io_bytes = std.math.mul(usize, @as(usize, ctx.active_rows_u32), @as(usize, ctx.d_model_u32) * @sizeOf(f32)) catch 0;
            const expected_weight_bytes = std.math.mul(usize, @as(usize, ctx.d_model_u32), @sizeOf(f32)) catch 0;
            log.warn("inference", "CUDA rmsnorm adapter invalid args", .{
                .layer_index = ctx.layer_index,
                .op_index = ctx.op_index,
                .input_bytes = input.size,
                .output_bytes = output.size,
                .weight_bytes = weight.buffer.size,
                .expected_io_bytes = expected_io_bytes,
                .expected_weight_bytes = expected_weight_bytes,
                .d_model = ctx.d_model_u32,
                .seq_len = ctx.seq_len_u32,
            });
        }
        return err;
    };
    const rmsnorm_elapsed_i128: i128 = std.time.nanoTimestamp() - rmsnorm_start_ns;
    const rmsnorm_elapsed_ns: u64 = if (rmsnorm_elapsed_i128 > 0) @intCast(rmsnorm_elapsed_i128) else 0;
    self.nvfp4_phase_counters.recordRmsnorm(rmsnorm_elapsed_ns);
}

pub fn layerProgramResidualAddAdapter(
    self: anytype,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    scale: layer_ops.ResidualScale,
    ctx: *LayerProgramExecutionContext,
) !void {
    if (self.skip_next_residual_add) {
        self.skip_next_residual_add = false;
        return;
    }
    const residual_add_start_ns: i128 = std.time.nanoTimestamp();
    if (try tryFuseResidualAddIntoNextRmsnorm(self, insn, registers, scale, ctx)) {
        const fused_elapsed_i128: i128 = std.time.nanoTimestamp() - residual_add_start_ns;
        const fused_elapsed_ns: u64 = if (fused_elapsed_i128 > 0) @intCast(fused_elapsed_i128) else 0;
        self.nvfp4_phase_counters.recordResidualAdd(fused_elapsed_ns);
        return;
    }
    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len < 2 or io.outputs.len == 0) return error.InvalidInstructionBinding;
    const residual_src = bufferFromTensorHandle(io.inputs[0]);
    const residual = bufferFromTensorHandle(io.outputs[0]);
    const branch = bufferFromTensorHandle(io.inputs[1]);
    const output_scale = standaloneLayerScalarOutputScale(self, ctx);
    try engine_ops.addResidualWithScaleRowsStrideAware(
        self,
        residual,
        residual_src,
        branch,
        ctx.active_rows_u32,
        ctx.d_model_u32,
        scale,
        output_scale,
    );
    const residual_add_elapsed_i128: i128 = std.time.nanoTimestamp() - residual_add_start_ns;
    const residual_add_elapsed_ns: u64 = if (residual_add_elapsed_i128 > 0) @intCast(residual_add_elapsed_i128) else 0;
    self.nvfp4_phase_counters.recordResidualAdd(residual_add_elapsed_ns);
}

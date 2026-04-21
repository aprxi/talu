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
const tensor = @import("tensor_pkg");
const dtype = @import("dtype_pkg");
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
const adapters_attention = @import("adapters_attention.zig");
const adapters_ffn = @import("adapters_ffn.zig");
const adapters_moe = @import("adapters_moe.zig");
const adapters_norm_residual = @import("adapters_norm_residual.zig");
const adapters_stateful = @import("adapters_stateful.zig");
const recordLayerProgramDispatch = handles.recordLayerProgramDispatch;
const buildLayerProgramInstructionHandles = handles.buildLayerProgramInstructionHandles;
const instructionParams = handles.instructionParams;
const layerProgramStateBlocksForInstruction = handles.layerProgramStateBlocksForInstruction;
const layerProgramInstructionHandleCapacity = handles.layerProgramInstructionHandleCapacity;
const layerProgramAttentionAdapter = adapters_attention.layerProgramAttentionAdapter;
const layerProgramSwiGluAdapter = adapters_ffn.layerProgramSwiGluAdapter;
const layerProgramMoEAdapter = adapters_moe.layerProgramMoEAdapter;
const layerProgramNormAdapter = adapters_norm_residual.layerProgramNormAdapter;
const layerProgramResidualAddAdapter = adapters_norm_residual.layerProgramResidualAddAdapter;
const layerProgramShortConvAdapter = adapters_stateful.layerProgramShortConvAdapter;
const layerProgramGatedDeltaAdapter = adapters_stateful.layerProgramGatedDeltaAdapter;

pub fn emitLayerProgramTracePoint(
    ctx: *LayerProgramExecutionContext,
    point: trace.TracePoint,
    shape: [4]u32,
    ndim: u8,
    kernel_name: []const u8,
    output: ?*const compute.cuda.Buffer,
) void {
    if (!trace.isEnabled()) return;
    var marker = [_]f32{0.0};
    // CUDA timing emits may exist without a host-materialized tensor. Only mark
    // the emission as f32 after a successful host download so verification never
    // computes stats from synthetic marker storage.
    var emit_dtype: trace.DType = .u8;
    const parity_prefill_seq_len_u32: u32 = if (ctx.backend.parity_prefill_seq_len > 0)
        @intCast(ctx.backend.parity_prefill_seq_len)
    else
        ctx.trace_seq_len_u32;
    const parity_prefill_active = parity_prefill_seq_len_u32 > 1 and
        ctx.active_rows_u32 == parity_prefill_seq_len_u32 and
        ctx.backend.parityPrefillBufferForPoint(point) != null;
    const logical_seq_len: u32 = ctx.trace_seq_len_u32;
    const shape_seq_len: u32 = if (parity_prefill_active)
        parity_prefill_seq_len_u32
    else
        ctx.active_rows_u32;
    const position_seq_len: u32 = if (parity_prefill_active)
        parity_prefill_seq_len_u32
    else
        ctx.seq_len_u32;
    const emission_token: u32 = if (parity_prefill_active) 0 else traceTokenIndex(logical_seq_len);
    var emit_shape = shape;
    if (ndim >= 2) emit_shape[1] = shape_seq_len;
    var data_ptr: [*]const u8 = @ptrCast(marker[0..].ptr);
    var emit_kernel_name = kernel_name;
    var host_kernel_name_buf: [96]u8 = undefined;
    if (output) |buffer| {
        const width = @as(usize, emit_shape[ndim - 1]);
        if (parity_prefill_active) {
            if (ctx.backend.parityPrefillBufferForPoint(point)) |prefill_buffer| {
                const parity_seq_len = ctx.backend.parity_prefill_seq_len;
                if (parity_seq_len == 0) return;
                if (width > 0 and ctx.layer_index < ctx.backend.block_runtime.blocks.len) {
                    const per_layer = parity_seq_len * width;
                    ctx.backend.ensureTraceCheckpointHostCapacity(per_layer) catch return;
                    const layer_host = ctx.backend.trace_checkpoint_host[0..per_layer];
                    engine_mixers.downloadRowsF32StrideAware(ctx.backend, buffer, parity_seq_len, width, layer_host) catch |err| {
                        const warn_idx = @intFromEnum(point);
                        if (!ctx.backend.parity_checkpoint_warned[warn_idx]) {
                            ctx.backend.parity_checkpoint_warned[warn_idx] = true;
                            log.warn("inference", "CUDA parity checkpoint download failed", .{
                                .point = point.name(),
                                .layer_index = ctx.layer_index,
                                .position = ctx.trace_pos_offset,
                                .seq_len = parity_seq_len,
                                .width = width,
                                .element_count = per_layer,
                                .buffer_bytes = buffer.size,
                                .reason = @errorName(err),
                            });
                        }
                        return;
                    };
                    const dst_start = ctx.layer_index * per_layer;
                    @memcpy(prefill_buffer[dst_start .. dst_start + per_layer], layer_host);
                    emit_kernel_name = std.fmt.bufPrint(&host_kernel_name_buf, "{s}_host", .{kernel_name}) catch kernel_name;
                    data_ptr = @ptrCast(layer_host.ptr);
                    emit_dtype = .f32;
                }
            } else {
                return;
            }
        } else {
            var element_count: usize = 1;
            for (0..ndim) |dim_idx| {
                // Trace downloads must match the emitted checkpoint shape, not
                // logical sequence length metadata.
                element_count *= @as(usize, emit_shape[dim_idx]);
            }
            if (element_count > 0) {
                ctx.backend.ensureTraceCheckpointHostCapacity(element_count) catch return;
                const host = ctx.backend.trace_checkpoint_host[0..element_count];
                buffer.download(&ctx.backend.device, std.mem.sliceAsBytes(host)) catch |err| {
                    const warn_idx = @intFromEnum(point);
                    if (!ctx.backend.parity_checkpoint_warned[warn_idx]) {
                        ctx.backend.parity_checkpoint_warned[warn_idx] = true;
                        log.warn("inference", "CUDA checkpoint download failed", .{
                            .point = point.name(),
                            .layer_index = ctx.layer_index,
                            .position = ctx.trace_pos_offset,
                            .element_count = element_count,
                            .buffer_bytes = buffer.size,
                            .reason = @errorName(err),
                        });
                    }
                    return;
                };
                data_ptr = @ptrCast(host.ptr);
                emit_dtype = .f32;
                emit_kernel_name = std.fmt.bufPrint(&host_kernel_name_buf, "{s}_host", .{kernel_name}) catch kernel_name;
            }
        }
    }
    trace.emit(
        point,
        @intCast(ctx.layer_index),
        emission_token,
        tracePositionForPoint(point, ctx.trace_pos_offset, position_seq_len),
        data_ptr,
        emit_dtype,
        emit_shape,
        ndim,
        emit_kernel_name,
    );
}

pub fn inferNormTracePoint(layer: *const BlockRuntimeLayer, op_index: usize) trace.TracePoint {
    const compiled = layer.compiled_plan orelse return .layer_ffn_norm;
    if (op_index + 1 < compiled.plan.instructions.len) {
        var idx = op_index + 1;
        while (idx < compiled.plan.instructions.len) : (idx += 1) {
            const next_opcode = compiled.plan.instructions[idx].opcode;
            switch (next_opcode) {
                .multihead_attention, .mla_attention, .shortconv, .gated_delta_net => return .layer_attn_norm,
                .swiglu => return .layer_ffn_norm,
                .residual_add => continue,
                else => continue,
            }
        }
    }
    return .layer_ffn_norm;
}

pub fn dispatchLayerProgramInstruction(
    self: anytype,
    insn: *const runtime_contract.Instruction,
    ctx: *LayerProgramExecutionContext,
) !void {
    const adapter = layer_program_adapter_table[@intFromEnum(insn.opcode)].?;
    recordLayerProgramDispatch(self, insn.opcode);

    var active_slots: [1]usize = .{0};
    var seq_lengths: [1]u32 = .{ctx.active_rows_u32};
    var rt_ctx = runtime_contract.ExecutionContext{
        .mode = if (ctx.active_rows_u32 > 1) .prefill else .decode,
        .active_slots = active_slots[0..],
        .sequence_lengths = seq_lengths[0..],
        .batch_size = 1,
        .dispatch_counters = if (enable_dispatch_observability) &self.runtime_dispatch_counters else null,
        .stream_or_queue = null,
        .workspace = .{ .any = @ptrCast(ctx) },
    };
    try runtime_contract.validateBatchCapability(
        layer_program_adapter_capabilities[@intFromEnum(insn.opcode)],
        rt_ctx.batch_size,
    );
    const built_handles = try buildLayerProgramInstructionHandles(
        self,
        insn,
        ctx,
        ctx.instruction_handles,
        ctx.instruction_views,
    );
    var param_storage: [1]runtime_contract.ParamBlock = undefined;
    const params = try instructionParams(insn, &ctx.layer.compiled_plan.?, &param_storage);
    var state_blocks = try layerProgramStateBlocksForInstruction(insn, ctx);
    _ = try runtime_contract.requireInstructionStateBlockForPlan(
        insn,
        &ctx.layer.compiled_plan.?.plan,
        state_blocks.slice(),
    );

    try adapter(
        &rt_ctx,
        insn,
        built_handles.registers,
        built_handles.views,
        state_blocks.slice(),
        params,
    );
}

pub fn tryExecuteLayerProgram(
    self: anytype,
    layer: *BlockRuntimeLayer,
    slot_index: usize,
    layer_index: usize,
    d_model_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
    active_rows_u32: u32,
    seq_len_u32: u32,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    position: usize,
    position_u32: u32,
    global_rope_theta: f32,
    local_rope_theta: f32,
    rope_function: compute.cuda.Function,
    copy_function: compute.cuda.Function,
    cast_f32_to_f16_function: ?compute.cuda.Function,
    kv_write_f16_function: ?compute.cuda.Function,
    rope_store_f16_function: ?compute.cuda.Function,
    shortconv_step_function: compute.cuda.Function,
    attention_kernels: AttentionKernelSet,
    batch_info: ?*const BatchDecodeInfo,
) !compute.cuda.Buffer {
    const prev_backend = trace.setBackendContext(.cuda);
    defer _ = trace.setBackendContext(prev_backend);
    const compiled_plan = layer.compiled_plan orelse return error.UnsupportedModel;
    const required_slot_count = blk: {
        var required: usize = 0;
        for (layer.register_to_slot_map) |slot_idx| {
            if (slot_idx == BlockRuntimeLayer.invalid_slot) continue;
            const next = @as(usize, slot_idx) + 1;
            if (next > required) required = next;
        }
        break :blk required;
    };
    if (required_slot_count > self.layer_program_slot_buffers.len) return error.UnsupportedModel;
    const handle_capacity = layerProgramInstructionHandleCapacity(&compiled_plan.plan);
    if (handle_capacity > self.layer_program_instruction_handle_scratch.len) {
        self.layer_program_instruction_handle_scratch = try self.allocator.realloc(
            self.layer_program_instruction_handle_scratch,
            handle_capacity,
        );
    }
    if (handle_capacity > self.layer_program_instruction_view_scratch.len) {
        self.layer_program_instruction_view_scratch = try self.allocator.realloc(
            self.layer_program_instruction_view_scratch,
            handle_capacity,
        );
    }
    if (required_slot_count > self.layer_program_slot_view_scratch.len) {
        self.layer_program_slot_view_scratch = try self.allocator.realloc(
            self.layer_program_slot_view_scratch,
            required_slot_count,
        );
    }
    const instruction_handles = self.layer_program_instruction_handle_scratch[0..handle_capacity];
    const instruction_views = self.layer_program_instruction_view_scratch[0..handle_capacity];
    const slot_buffer_views = self.layer_program_slot_view_scratch[0..required_slot_count];
    const active_input_bytes = std.math.mul(usize, @as(usize, active_rows_u32), @as(usize, d_model_u32) * @sizeOf(f32)) catch return error.InvalidArgument;
    const input_view = try bufferSlice(&self.runtime_buffers.input_dev, 0, active_input_bytes);
    for (slot_buffer_views, 0..) |*view, slot_idx| {
        const width = layer.slot_width_hints[slot_idx];
        const bytes = std.math.mul(usize, @as(usize, active_rows_u32), width * @sizeOf(f32)) catch return error.InvalidArgument;
        view.* = try bufferSlice(self.layer_program_slot_ptrs[slot_idx], 0, bytes);
    }
    var exec_ctx = LayerProgramExecutionContext{
        .backend = self,
        .layer = layer,
        .slot_index = slot_index,
        .layer_index = layer_index,
        .op_index = 0,
        .d_model_u32 = d_model_u32,
        .head_dim_u32 = head_dim_u32,
        .rope_dim_u32 = rope_dim_u32,
        .n_heads_u32 = n_heads_u32,
        .n_kv_heads_u32 = n_kv_heads_u32,
        .active_rows_u32 = active_rows_u32,
        .seq_len_u32 = seq_len_u32,
        .trace_seq_len_u32 = trace_seq_len_u32,
        .trace_pos_offset = trace_pos_offset,
        .position = position,
        .position_u32 = position_u32,
        .global_rope_theta = global_rope_theta,
        .local_rope_theta = local_rope_theta,
        .rope_function = rope_function,
        .copy_function = copy_function,
        .cast_f32_to_f16_function = cast_f32_to_f16_function,
        .kv_write_f16_function = kv_write_f16_function,
        .rope_store_f16_function = rope_store_f16_function,
        .shortconv_step_function = shortconv_step_function,
        .attention_kernels = attention_kernels,
        .register_to_slot_map = layer.register_to_slot_map,
        .input_view = input_view,
        .slot_buffers = slot_buffer_views,
        .instruction_handles = instruction_handles,
        .instruction_views = instruction_views,
        .batch_info = batch_info,
    };

    for (compiled_plan.plan.instructions, 0..) |insn, op_index| {
        exec_ctx.op_index = op_index;
        dispatchLayerProgramInstruction(self, &insn, &exec_ctx) catch |err| {
            log.warn("inference", "CUDA layer program instruction failed", .{
                .layer_index = layer_index,
                .op_index = op_index,
                .opcode = @tagName(insn.opcode),
                .seq_len = seq_len_u32,
                .position = position,
                .reason = @errorName(err),
            });
            return err;
        };
    }

    const final_register = runtime_contract.planFinalOutputRegister(&compiled_plan.plan);
    const final_register_idx = runtime_contract.registerToIndex(final_register);
    if (final_register_idx != 0) {
        const final_buf = engine_ops.programBuffer(self, final_register_idx, &exec_ctx) orelse return error.UnsupportedModel;
        const row_elems_usize: usize = @intCast(d_model_u32);
        const row_bytes = std.math.mul(usize, row_elems_usize, @sizeOf(f32)) catch return error.InvalidArgument;
        const slot_idx = if (final_register_idx < layer.register_to_slot_map.len)
            layer.register_to_slot_map[final_register_idx]
        else
            BlockRuntimeLayer.invalid_slot;
        const final_row_width = if (slot_idx == BlockRuntimeLayer.invalid_slot or slot_idx >= layer.slot_width_hints.len)
            row_elems_usize
        else
            layer.slot_width_hints[slot_idx];

        // Final output may live in a widened temporary slot. For multi-row prefill,
        // copy row-by-row using slot stride so later rows do not alias padding.
        if (final_row_width == row_elems_usize) {
            try compute.cuda.copy.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                copy_function,
                final_buf,
                &self.runtime_buffers.input_dev,
                std.math.mul(u32, active_rows_u32, d_model_u32) catch return error.InvalidArgument,
            );
        } else {
            const src_row_stride = std.math.mul(usize, final_row_width, @sizeOf(f32)) catch return error.InvalidArgument;
            const row_count: usize = @intCast(active_rows_u32);
            var row_idx: usize = 0;
            while (row_idx < row_count) : (row_idx += 1) {
                const src_offset = std.math.mul(usize, row_idx, src_row_stride) catch return error.InvalidArgument;
                const dst_offset = std.math.mul(usize, row_idx, row_bytes) catch return error.InvalidArgument;
                var src_row = try bufferSlice(final_buf, src_offset, row_bytes);
                var dst_row = try bufferSlice(&self.runtime_buffers.input_dev, dst_offset, row_bytes);
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function,
                    &src_row,
                    &dst_row,
                    d_model_u32,
                );
            }
        }
        return final_buf.*;
    }
    return exec_ctx.input_view;
}

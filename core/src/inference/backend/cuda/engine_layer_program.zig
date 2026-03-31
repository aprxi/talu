//! Layer program dispatch, adapter implementations, kernel initialization.
//!
//! Contains the per-instruction adapter implementations (norm, attention,
//! short conv, gated delta, SwiGLU, residual add), the layer program
//! dispatch loop, tryExecuteLayerProgram, runAttentionContext, kernel
//! function resolution, and CPU ROPE initialization.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const build_options = @import("build_options");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");
const models = @import("../../../models/root.zig");
const layer_ops = models.layer_ops;
const op_types = models.op_types;
const opcode_map = models.plan.opcode_map;
const rope_scaling = models.rope_scaling;
const runtime_contract = @import("../../runtime_contract/root.zig");
const attention_mod = @import("attention.zig");
const smoke_checks = @import("smoke_checks.zig");
const cpu_kernels = @import("../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;

// --- Types from engine.zig (mutual import) ---
const engine = @import("engine.zig");
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
const engine_types = @import("engine_types.zig");
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
const AttentionPath = engine_types.AttentionPath;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("engine_ops.zig");

// --- Mixer functions from engine_mixers.zig ---
const engine_mixers = @import("engine_mixers.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const bufferSlice = engine_weights.bufferSlice;

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
    if (layer_index >= state.block_runtime.blocks.len) return error.InvalidInstructionIndex;
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

pub fn initCpuRuntimeRopeHandles(self: anytype) !void {
    if (self.loaded.position_embeddings != null) return;
    if (self.rope_dim == 0) return;

    var global_freqs = try rope_scaling.materializeInverseFrequencies(
        self.allocator,
        self.rope_dim,
        self.loaded.config.rope_theta,
        self.loaded.config.rope_scaling,
    );
    defer global_freqs.deinit(self.allocator);

    const global_rope = try self.allocator.create(cpu_kernels.RoPE);
    errdefer self.allocator.destroy(global_rope);
    global_rope.* = try cpu_kernels.RoPE.initFromInvFreq(
        self.allocator,
        self.rope_dim,
        @intCast(self.max_seq_len),
        global_freqs.inv_freq,
        global_freqs.attention_scaling,
    );
    self.cpu_rope_global = global_rope;

    if (self.loaded.config.rope_local_theta > 0 and self.loaded.config.sliding_window > 0) {
        var local_freqs = try rope_scaling.materializeInverseFrequencies(
            self.allocator,
            self.rope_dim,
            self.loaded.config.rope_local_theta,
            self.loaded.config.rope_scaling,
        );
        defer local_freqs.deinit(self.allocator);

        const local_rope = try self.allocator.create(cpu_kernels.RoPE);
        errdefer self.allocator.destroy(local_rope);
        local_rope.* = try cpu_kernels.RoPE.initFromInvFreq(
            self.allocator,
            self.rope_dim,
            @intCast(self.max_seq_len),
            local_freqs.inv_freq,
            local_freqs.attention_scaling,
        );
        self.cpu_rope_local = local_rope;
    }
}

pub fn assignCpuRuntimeRopeToAttentionFallbacks(self: anytype) void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.attention_binding orelse continue;
        if (block.cpu_kernel) |*kernel| {
            kernel.rope = if (kernel.sliding_window > 0 and self.cpu_rope_local != null)
                self.cpu_rope_local
            else
                self.cpu_rope_global;
        }
    }
}

fn residualScaleFactor(self: anytype, scale: layer_ops.ResidualScale) f32 {
    return switch (scale) {
        .one => 1.0,
        .residual_multiplier => self.loaded.config.residual_multiplier,
        .literal => |literal| literal,
    };
}

fn runResidualAddRmsnormRowsStrideAware(
    self: anytype,
    fused_fn: compute.cuda.Function,
    residual_out: *compute.cuda.Buffer,
    norm_out: *compute.cuda.Buffer,
    residual_in: *const compute.cuda.Buffer,
    branch: *const compute.cuda.Buffer,
    weight: *const compute.cuda.Buffer,
    residual_scale: f32,
    rows: u32,
    cols: u32,
) !void {
    if (rows == 0 or cols == 0) return error.InvalidArgument;
    const packed_count = std.math.mul(u32, rows, cols) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, @as(usize, packed_count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (residual_out.size < packed_bytes or
        norm_out.size < packed_bytes or
        residual_in.size < packed_bytes or
        branch.size < packed_bytes)
    {
        return error.InvalidInstructionBinding;
    }

    var residual_out_stride_elems: u32 = cols;
    var norm_out_stride_elems: u32 = cols;
    var residual_in_stride_elems: u32 = cols;
    var branch_stride_elems: u32 = cols;

    if (!(residual_out.size == packed_bytes and
        norm_out.size == packed_bytes and
        residual_in.size == packed_bytes and
        branch.size == packed_bytes))
    {
        const row_count: usize = @intCast(rows);
        if (residual_out.size % row_count != 0 or
            norm_out.size % row_count != 0 or
            residual_in.size % row_count != 0 or
            branch.size % row_count != 0)
        {
            return error.InvalidInstructionBinding;
        }
        const row_bytes = std.math.mul(usize, @as(usize, cols), @sizeOf(f32)) catch return error.InvalidArgument;
        const residual_out_stride = residual_out.size / row_count;
        const norm_out_stride = norm_out.size / row_count;
        const residual_in_stride = residual_in.size / row_count;
        const branch_stride = branch.size / row_count;
        if (residual_out_stride < row_bytes or
            norm_out_stride < row_bytes or
            residual_in_stride < row_bytes or
            branch_stride < row_bytes)
        {
            return error.InvalidInstructionBinding;
        }
        if ((residual_out_stride % @sizeOf(f32)) != 0 or
            (norm_out_stride % @sizeOf(f32)) != 0 or
            (residual_in_stride % @sizeOf(f32)) != 0 or
            (branch_stride % @sizeOf(f32)) != 0)
        {
            return error.InvalidInstructionBinding;
        }

        residual_out_stride_elems = std.math.cast(u32, residual_out_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
        norm_out_stride_elems = std.math.cast(u32, norm_out_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
        residual_in_stride_elems = std.math.cast(u32, residual_in_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
        branch_stride_elems = std.math.cast(u32, branch_stride / @sizeOf(f32)) orelse return error.InvalidArgument;
    }

    try compute.cuda.residual_scaled_rmsnorm_rows_strided.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        fused_fn,
        residual_out,
        norm_out,
        residual_in,
        branch,
        weight,
        residual_scale,
        rows,
        cols,
        residual_out_stride_elems,
        norm_out_stride_elems,
        residual_in_stride_elems,
        branch_stride_elems,
        self.norm_eps,
        self.loaded.runtime.weight_offset,
    );
}

fn tryFuseResidualAddIntoNextRmsnorm(
    self: anytype,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    scale: layer_ops.ResidualScale,
    ctx: *LayerProgramExecutionContext,
) !bool {
    const fused_fn = self.residual_scaled_rmsnorm_rows_strided_function orelse return false;
    const compiled = ctx.layer.compiled_plan orelse return error.UnsupportedModel;
    if (ctx.op_index + 1 >= compiled.plan.instructions.len) return false;

    const next_insn = &compiled.plan.instructions[ctx.op_index + 1];
    if (next_insn.opcode != .rmsnorm) return false;
    if (insn.outputs.len != 1 or next_insn.inputs.len != 1 or next_insn.outputs.len != 1) return false;
    if (insn.outputs[0] != next_insn.inputs[0]) return false;

    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len < 2 or io.outputs.len == 0) return error.InvalidInstructionBinding;
    const residual_src = bufferFromTensorHandle(io.inputs[0]);
    const residual_dst = bufferFromTensorHandle(io.outputs[0]);
    const branch = bufferFromTensorHandle(io.inputs[1]);

    const norm_out_reg = runtime_contract.registerToIndex(next_insn.outputs[0]);
    const norm_out = engine_ops.programBuffer(self, norm_out_reg, ctx) orelse return error.UnsupportedModel;
    const norm_weight = try ctx.layer.instructionNormWeightRef(ctx.op_index + 1);
    try runResidualAddRmsnormRowsStrideAware(
        self,
        fused_fn,
        residual_dst,
        norm_out,
        residual_src,
        branch,
        &norm_weight.buffer,
        residualScaleFactor(self, scale),
        ctx.active_rows_u32,
        ctx.d_model_u32,
    );
    self.skip_next_rmsnorm = true;
    return true;
}

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
}

pub fn layerProgramAttentionAdapter(
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
    if (weight_handles.len != 11) return error.InvalidWeightRefCount;
    const input = bufferFromTensorHandle(io.inputs[0]);
    const output = bufferFromTensorHandle(io.outputs[0]);
    const cfg = try layer.instructionAttentionRef(ctx.op_index);
    const q_proj = linearWeightFromWeightHandle(weight_handles[0]).*;
    const k_proj = linearWeightFromWeightHandle(weight_handles[1]).*;
    const v_proj = linearWeightFromWeightHandle(weight_handles[2]).*;
    const o_proj = linearWeightFromWeightHandle(weight_handles[3]).*;
    const q_norm_weight = optionalDeviceTensorFromWeightHandle(weight_handles[4]);
    const k_norm_weight = optionalDeviceTensorFromWeightHandle(weight_handles[5]);
    if (q_proj.cols() != expectedAttentionQProjectionDim(cfg)) return error.InvalidInstructionBinding;
    if (k_proj.cols() != cfg.kv_dim or v_proj.cols() != cfg.kv_dim) return error.InvalidInstructionBinding;
    const state_id = insn.state_block_id orelse return error.InvalidStateDescriptorBinding;
    const kv_state = try requireStateValue(KvRuntimeState, state_blocks, state_id);
    if (kv_state.runtime_kind != runtime_contract.state_runtime_kind_kv_cache) {
        return error.InvalidStateDescriptorBinding;
    }
    const attention_binding = try requireAttentionRuntimeBinding(kv_state, ctx.layer_index);
    // Batched decode: N tokens at different positions/slots, GEMM projections.
    if (ctx.batch_info) |batch| {
        // Reuse the fused concat-I8 QKV cache for batched decode as well.
        // Previously this was only enabled on non-batched prefill routes,
        // forcing decode batch rows through slower U4 QKV projection kernels.
        self.active_qkv_concat = if (attention_binding.qkv_i8_concat.pointer != 0)
            .{ .i8_buf = attention_binding.qkv_i8_concat, .scales_buf = attention_binding.qkv_scales_concat, .dims = attention_binding.qkv_concat_dims }
        else
            null;
        defer self.active_qkv_concat = null;
        try engine_mixers.runBatchedDecodeAttentionMixer(
            self,
            cfg,
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            q_norm_weight,
            k_norm_weight,
            input,
            output,
            ctx,
            batch,
        );
        return;
    }

    if (ctx.active_rows_u32 <= 1) {
        const residual_buf: ?compute.cuda.Buffer =
            if (self.loaded.config.residual_multiplier == 1.0) ctx.input_view else null;
        try engine_mixers.runAttentionMixerStep(
            self,
            cfg,
            &attention_binding.k_cache,
            &attention_binding.v_cache,
            &attention_binding.k_scale,
            &attention_binding.v_scale,
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            q_norm_weight,
            k_norm_weight,
            input,
            output,
            ctx.d_model_u32,
            ctx.head_dim_u32,
            ctx.rope_dim_u32,
            ctx.n_heads_u32,
            ctx.n_kv_heads_u32,
            ctx.seq_len_u32,
            ctx.position,
            ctx.position_u32,
            ctx.global_rope_theta,
            ctx.local_rope_theta,
            ctx.rope_function,
            ctx.copy_function,
            ctx.cast_f32_to_f16_function,
            ctx.kv_write_f16_function,
            ctx.rope_store_f16_function,
            ctx.attention_kernels,
            residual_buf,
        );
        return;
    }

    // Provide concat I8 QKV cache for fused prefill GEMM.
    self.active_qkv_concat = if (attention_binding.qkv_i8_concat.pointer != 0)
        .{ .i8_buf = attention_binding.qkv_i8_concat, .scales_buf = attention_binding.qkv_scales_concat, .dims = attention_binding.qkv_concat_dims }
    else
        null;
    defer self.active_qkv_concat = null;

    if (!cfg.query_gate) {
        try engine_mixers.runAttentionMixerPrefillBatchedNoQueryGate(
            self,
            cfg,
            &attention_binding.k_cache,
            &attention_binding.v_cache,
            &attention_binding.k_scale,
            &attention_binding.v_scale,
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            q_norm_weight,
            k_norm_weight,
            input,
            output,
            ctx.d_model_u32,
            ctx.head_dim_u32,
            ctx.rope_dim_u32,
            ctx.n_heads_u32,
            ctx.n_kv_heads_u32,
            ctx.seq_len_u32,
            ctx.global_rope_theta,
            ctx.local_rope_theta,
            ctx.rope_function,
            ctx.copy_function,
            ctx.cast_f32_to_f16_function,
            ctx.kv_write_f16_function,
            ctx.rope_store_f16_function,
            ctx.attention_kernels,
        );
        return;
    }

    try engine_mixers.runAttentionMixerPrefillBatchedWithQueryGate(
        self,
        cfg,
        &attention_binding.k_cache,
        &attention_binding.v_cache,
        &attention_binding.k_scale,
        &attention_binding.v_scale,
        &q_proj,
        &k_proj,
        &v_proj,
        &o_proj,
        q_norm_weight,
        k_norm_weight,
        input,
        output,
        ctx.d_model_u32,
        ctx.head_dim_u32,
        ctx.rope_dim_u32,
        ctx.n_heads_u32,
        ctx.n_kv_heads_u32,
        ctx.seq_len_u32,
        ctx.global_rope_theta,
        ctx.local_rope_theta,
        ctx.rope_function,
        ctx.copy_function,
        ctx.cast_f32_to_f16_function,
        ctx.kv_write_f16_function,
        ctx.rope_store_f16_function,
        ctx.attention_kernels,
    );
}

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

pub fn layerProgramSwiGluAdapter(
    self: anytype,
    _: *BlockRuntimeLayer,
    insn: *const runtime_contract.Instruction,
    registers: []runtime_contract.TensorHandle,
    ctx: *LayerProgramExecutionContext,
) !void {
    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len != 1 or io.outputs.len != 1) return error.InvalidInstructionBinding;
    const input = bufferFromTensorHandle(io.inputs[0]);
    const output = bufferFromTensorHandle(io.outputs[0]);
    const weight_handles = try instructionWeightSlice(insn, registers);
    if (weight_handles.len != 5) return error.InvalidWeightRefCount;
    const gate_weight = linearWeightFromWeightHandle(weight_handles[0]);
    const up_weight = linearWeightFromWeightHandle(weight_handles[1]);
    const down_weight = linearWeightFromWeightHandle(weight_handles[2]);
    const gate_bias = optionalDeviceTensorFromWeightHandle(weight_handles[3]);
    const down_bias = optionalDeviceTensorFromWeightHandle(weight_handles[4]);
    const d_ff = gate_weight.cols();
    if (up_weight.cols() != d_ff) return error.InvalidInstructionBinding;
    if (down_weight.rows() != d_ff) return error.InvalidInstructionBinding;
    const d_ff_u32: u32 = @intCast(d_ff);
    const residual_buf: ?compute.cuda.Buffer =
        if (ctx.active_rows_u32 <= 1 and self.loaded.config.residual_multiplier == 1.0)
            ctx.input_view
        else
            null;
    try engine_mixers.runFfnStep(
        self,
        input,
        @intCast(ctx.active_rows_u32),
        gate_weight,
        up_weight,
        down_weight,
        gate_bias,
        down_bias,
        d_ff_u32,
        output,
        residual_buf,
    );
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
    if (try tryFuseResidualAddIntoNextRmsnorm(self, insn, registers, scale, ctx)) {
        return;
    }
    const io = try instructionIoSlices(insn, registers);
    if (io.inputs.len < 2 or io.outputs.len == 0) return error.InvalidInstructionBinding;
    const residual_src = bufferFromTensorHandle(io.inputs[0]);
    const residual = bufferFromTensorHandle(io.outputs[0]);
    const branch = bufferFromTensorHandle(io.inputs[1]);
    try engine_ops.addResidualWithScaleRowsStrideAware(
        self,
        residual,
        residual_src,
        branch,
        ctx.active_rows_u32,
        ctx.d_model_u32,
        scale,
    );
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
    const instruction_handles = try self.allocator.alloc(runtime_contract.TensorHandle, handle_capacity);
    defer if (instruction_handles.len > 0) self.allocator.free(instruction_handles);
    const instruction_views = try self.allocator.alloc(runtime_contract.TensorViewDesc, handle_capacity);
    defer if (instruction_views.len > 0) self.allocator.free(instruction_views);
    const active_input_bytes = std.math.mul(usize, @as(usize, active_rows_u32), @as(usize, d_model_u32) * @sizeOf(f32)) catch return error.InvalidArgument;
    const input_view = try bufferSlice(&self.runtime_buffers.input_dev, 0, active_input_bytes);
    const slot_buffer_views = try self.allocator.alloc(compute.cuda.Buffer, required_slot_count);
    defer if (slot_buffer_views.len > 0) self.allocator.free(slot_buffer_views);
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

pub fn runAttentionContext(
    self: anytype,
    cfg: *const LayerAttentionExecConfig,
    q_stage: *const compute.cuda.Buffer,
    context_stage: *compute.cuda.Buffer,
    k_cache: *const compute.cuda.Buffer,
    v_cache: *const compute.cuda.Buffer,
    k_scale: *const compute.cuda.Buffer,
    v_scale: *const compute.cuda.Buffer,
    kernels: AttentionKernelSet,
    seq_len_u32: u32,
    head_dim_u32: u32,
    kv_dim_u32: u32,
    kv_groups_u32: u32,
    rope_dim_u32: u32,
    position_u32: u32,
    theta: f32,
) !AttentionPath {
    var effective_seq_len_u32 = seq_len_u32;
    var k_cache_view = k_cache.*;
    var v_cache_view = v_cache.*;
    var k_scale_view = k_scale.*;
    var v_scale_view = v_scale.*;

    if (cfg.sliding_window > 0 and cfg.is_causal) {
        const window_u32 = std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32);
        if (effective_seq_len_u32 > window_u32) {
            const kv_elem_bytes: usize = self.kv_cache_dtype.elementBytes();
            const row_bytes = std.math.mul(usize, @as(usize, kv_dim_u32), kv_elem_bytes) catch return error.InvalidArgument;
            const start_row = effective_seq_len_u32 - window_u32;
            const start_offset = std.math.mul(usize, @as(usize, start_row), row_bytes) catch return error.InvalidArgument;
            k_cache_view = try bufferSlice(k_cache, start_offset, k_cache.size - start_offset);
            v_cache_view = try bufferSlice(v_cache, start_offset, v_cache.size - start_offset);
            // Slice scale buffers for i8 KV cache.
            const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
            const scale_row_bytes = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
            const scale_start = std.math.mul(usize, @as(usize, start_row), scale_row_bytes) catch return error.InvalidArgument;
            if (k_scale.size > scale_start) {
                k_scale_view = try bufferSlice(k_scale, scale_start, k_scale.size - scale_start);
                v_scale_view = try bufferSlice(v_scale, scale_start, v_scale.size - scale_start);
            }
            effective_seq_len_u32 = window_u32;
        }
    }

    switch (self.kv_cache_dtype) {
        .f16 => {
            if (!cfg.query_gate and attention_mod.useFusedHeadsF16Kv(
                attention_policy_config,
                seq_len_u32,
                cfg.sliding_window,
                cfg.is_causal,
                head_dim_u32,
                kernels.attn_fused_heads_f16_kv_function != null,
            )) {
                try compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_f16_kv_function.?,
                    q_stage,
                    &k_cache_view,
                    &v_cache_view,
                    context_stage,
                    @intCast(self.n_heads),
                    effective_seq_len_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    head_dim_u32,
                    self.attention_scale,
                    rope_dim_u32,
                    position_u32,
                    theta,
                );
                return .fused_heads_f16_kv;
            }

            const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
            const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();
            try compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                q_stage,
                &k_cache_view,
                attn_scores_dev,
                @intCast(self.n_heads),
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
                self.attention_scale,
            );
            try compute.cuda.softmax_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
                attn_scores_dev,
                attn_probs_dev,
                @intCast(self.n_heads),
                effective_seq_len_u32,
            );
            try compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &v_cache_view,
                context_stage,
                @intCast(self.n_heads),
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            );
            return .heads_f16_kv;
        },
        .i8 => {
            const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
            if (!cfg.query_gate and attention_mod.useFusedHeadsF16Kv(
                attention_policy_config,
                seq_len_u32,
                cfg.sliding_window,
                cfg.is_causal,
                head_dim_u32,
                kernels.attn_fused_heads_i8_kv_function != null,
            )) {
                try compute.cuda.attn_fused_heads_i8_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_i8_kv_function.?,
                    q_stage,
                    &k_cache_view,
                    &v_cache_view,
                    &k_scale_view,
                    &v_scale_view,
                    context_stage,
                    @intCast(self.n_heads),
                    n_kv_heads_u32,
                    effective_seq_len_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    head_dim_u32,
                    self.attention_scale,
                    rope_dim_u32,
                    position_u32,
                    theta,
                );
                return .fused_heads_i8_kv;
            }

            const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
            const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();
            try compute.cuda.attn_scores_heads_i8_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_i8_kv_function orelse return error.CudaKernelUnavailable,
                q_stage,
                &k_cache_view,
                &k_scale_view,
                attn_scores_dev,
                @intCast(self.n_heads),
                n_kv_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
                self.attention_scale,
            );
            try compute.cuda.softmax_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
                attn_scores_dev,
                attn_probs_dev,
                @intCast(self.n_heads),
                effective_seq_len_u32,
            );
            try compute.cuda.attn_weighted_sum_heads_i8_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_i8_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &v_cache_view,
                &v_scale_view,
                context_stage,
                @intCast(self.n_heads),
                n_kv_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            );
            return .heads_i8_kv;
        },
    }
}

pub fn initKernelFunctions(self: anytype) !void {
    if (!self.device.supportsModuleLaunch()) return;

    try self.kernel_registry.loadEmbeddedModule(compute.cuda.vector_add.embedded_module);
    const sideload_loaded = tryLoadSideloadModule(
        self,
    ) catch |err| blk: {
        log.warn("inference", "CUDA sideload unavailable; using embedded PTX", .{
            .reason = @errorName(err),
        });
        break :blk false;
    };
    if (sideload_loaded) {
        log.info("inference", "CUDA sideload kernel module active", .{});
    }

    try resolveRequiredKernels(
        self,
    );

    // Optional fusion kernel. If unavailable (e.g. stale sideload payload),
    // keep the canonical residual_add + rmsnorm split path.
    if (self.kernel_registry.resolveFunction(
        compute.cuda.residual_scaled_rmsnorm_rows_strided.op_name,
        compute.cuda.residual_scaled_rmsnorm_rows_strided.embedded_symbol,
    )) |resolved| {
        self.residual_scaled_rmsnorm_rows_strided_function = resolved.function;
        self.residual_scaled_rmsnorm_rows_strided_source = resolved.source;
    } else |_| {
        self.residual_scaled_rmsnorm_rows_strided_function = null;
        self.residual_scaled_rmsnorm_rows_strided_source = null;
    }
}

/// Pre-dequantize all gaffine_u8 weights to persistent F16 and I8 device buffers.
/// F16 cache: eliminates per-prefill dequant kernel overhead.
/// I8 cache: enables cuBLAS INT8 tensor core GEMM for prefill.
pub fn warmupDequantF16Cache(self: anytype) !void {
    // Resolve INT8 GEMM helper kernels (optional — graceful degradation to F16 path).
    if (self.kernel_registry.resolveFunction("quantize_f32_to_i8", "talu_quantize_f32_to_i8")) |resolved| {
        self.quantize_f32_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_i32_gaffine", "talu_dequant_i32_gaffine")) |resolved| {
        self.dequant_i32_gaffine_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("i8_rowsum", "talu_i8_rowsum")) |resolved| {
        self.i8_rowsum_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("u8_xor_to_i8", "talu_u8_xor_to_i8")) |resolved| {
        self.u8_xor_to_i8_function = resolved.function;
    } else |_| {}
    // Symmetric INT8 kernels.
    if (self.kernel_registry.resolveFunction("i8_matvec_f32", "talu_i8_matvec_f32")) |resolved| {
        self.i8_matvec_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("i8_matvec_qkv_f32", "talu_i8_matvec_qkv_f32")) |resolved| {
        self.i8_matvec_qkv_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("i8_matvec_gate_up_silu_f32", "talu_i8_matvec_gate_up_silu_f32")) |resolved| {
        self.i8_matvec_gate_up_silu_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("gaffine_u8_to_i8", "talu_gaffine_u8_to_i8")) |resolved| {
        self.gaffine_u8_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("gaffine_u4_to_i8", "talu_gaffine_u4_to_i8")) |resolved| {
        self.gaffine_u4_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f16_to_i8", "talu_quantize_f16_to_i8")) |resolved| {
        self.quantize_f16_to_i8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_i8_simple", "talu_quantize_f32_to_i8_simple")) |resolved| {
        self.quantize_f32_to_i8_simple_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_i32_scales", "talu_dequant_i32_scales")) |resolved| {
        self.dequant_i32_scales_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_i32_scales_split3", "talu_dequant_i32_scales_split3")) |resolved| {
        self.dequant_i32_scales_split3_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_fp8_e4m3", "talu_quantize_f32_to_fp8_e4m3")) |resolved| {
        self.quantize_f32_to_fp8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("quantize_f32_to_mxfp8", "talu_quantize_f32_to_mxfp8")) |resolved| {
        self.quantize_f32_to_mxfp8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_mxfp8_to_bf16", "talu_dequant_mxfp8_to_bf16")) |resolved| {
        self.mxfp8_dequant_to_bf16_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("scale_rows_f32", "talu_scale_rows_f32")) |resolved| {
        self.scale_rows_f32_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_f32", "talu_fp8_e4m3_matvec_f32")) |resolved| {
        self.fp8_matvec_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("dequant_fp8_e4m3_to_bf16", "talu_dequant_fp8_e4m3_to_bf16")) |resolved| {
        self.fp8_dequant_to_bf16_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_silu_f32", "talu_fp8_e4m3_matvec_gate_up_silu_f32")) |resolved| {
        self.fp8_matvec_gate_up_silu_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_f32", "talu_fp8_e4m3_matvec_gate_up_f32")) |resolved| {
        self.fp8_matvec_gate_up_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_f32_tile8", "talu_fp8_e4m3_matvec_f32_tile8")) |resolved| {
        self.fp8_matvec_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_silu_f32_tile8", "talu_fp8_e4m3_matvec_gate_up_silu_f32_tile8")) |resolved| {
        self.fp8_matvec_gate_up_silu_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("fp8_e4m3_matvec_gate_up_f32_tile8", "talu_fp8_e4m3_matvec_gate_up_f32_tile8")) |resolved| {
        self.fp8_matvec_gate_up_tile8_function = resolved.function;
    } else |_| {}

    if (self.kernel_registry.resolveFunction("mxfp8_matvec_f32", "talu_mxfp8_matvec_f32")) |resolved| {
        self.mxfp8_matvec_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_f32_tile8", "talu_mxfp8_matvec_f32_tile8")) |resolved| {
        self.mxfp8_matvec_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_silu_f32", "talu_mxfp8_matvec_gate_up_silu_f32")) |resolved| {
        self.mxfp8_matvec_gate_up_silu_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_silu_f32_tile8", "talu_mxfp8_matvec_gate_up_silu_f32_tile8")) |resolved| {
        self.mxfp8_matvec_gate_up_silu_tile8_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_f32", "talu_mxfp8_matvec_gate_up_f32")) |resolved| {
        self.mxfp8_matvec_gate_up_function = resolved.function;
    } else |_| {}
    if (self.kernel_registry.resolveFunction("mxfp8_matvec_gate_up_f32_tile8", "talu_mxfp8_matvec_gate_up_f32_tile8")) |resolved| {
        self.mxfp8_matvec_gate_up_tile8_function = resolved.function;
    } else |_| {}

    const has_u8_dequant = self.gaffine_u8_dequant_f16_function != null;
    const has_u4_dequant = self.gaffine_u4_dequant_f16_function != null;
    const has_u4_to_i8 = self.gaffine_u4_to_i8_function != null;
    const has_u8_to_i8 = self.gaffine_u8_to_i8_function != null;
    if (!has_u8_dequant and !has_u4_dequant and !has_u4_to_i8 and !has_u8_to_i8) {
        return;
    }

    var total_bytes: usize = 0;
    var weight_count: usize = 0;

    // Helper to launch a fused gaffine→I8 dequant kernel for a single weight.
    // Shared by both U8 and U4 paths — only the kernel function differs.
    const launchFusedToI8 = struct {
        fn run(
            backend: *CudaBackend,
            fused_fn: compute.cuda.Function,
            w: *GaffineU4LinearWeight,
            bytes_out: *usize,
        ) void {
            const weight_elems = std.math.mul(usize, w.rows, w.cols) catch return;
            if (weight_elems == 0) return;
            const i8_bytes = weight_elems;
            const scale_bytes = std.math.mul(usize, w.cols, @sizeOf(f32)) catch return;

            var i8_buf = backend.device.allocBuffer(i8_bytes) catch return;
            var scale_buf = backend.device.allocBuffer(scale_bytes) catch {
                i8_buf.deinit(&backend.device);
                return;
            };

            // Launch: grid=(out_dim=w.cols), block=(256)
            backend.kernel_arg_pack.reset();
            backend.kernel_arg_pack.appendBufferPtr(&w.packed_data) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&w.scales) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&w.biases) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&i8_buf) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendBufferPtr(&scale_buf) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, @intCast(w.rows)) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, w.group_size) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            backend.kernel_arg_pack.appendScalar(u32, w.scales_dtype_tag) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };
            compute.cuda.launch.launchWithFamily(&backend.device, fused_fn, .{
                .grid_x = @intCast(w.cols),
                .block_x = 256,
            }, &backend.kernel_arg_pack, .other) catch {
                scale_buf.deinit(&backend.device);
                i8_buf.deinit(&backend.device);
                return;
            };

            w.dequant_i8_cache = i8_buf;
            w.mean_scale_cache = scale_buf;
            bytes_out.* += i8_bytes + scale_bytes;
        }
    }.run;

    // Helper to create I8 cache for a single gaffine_u8 weight.
    // Prefers fused U8→I8 kernel (no F16 intermediate), falls back to F16→I8.
    const dequantU8Weight = struct {
        fn run(
            backend: *CudaBackend,
            w: *GaffineU8LinearWeight,
            bytes_out: *usize,
        ) void {
            // Try fused U8→I8 path (no F16 intermediate, saves ~50% VRAM).
            if (backend.gaffine_u8_to_i8_function) |fused_fn| {
                launchFusedToI8(backend, fused_fn, w, bytes_out);
                return;
            }

            // Fallback: dequant to F16 cache (for F16 GEMM path).
            const dequant_f16_fn = backend.gaffine_u8_dequant_f16_function orelse return;
            const weight_elems = std.math.mul(usize, w.rows, w.cols) catch return;
            if (weight_elems == 0) return;
            const weight_f16_bytes = std.math.mul(usize, weight_elems, @sizeOf(u16)) catch return;
            var cache_buf = backend.device.allocBuffer(weight_f16_bytes) catch return;
            errdefer cache_buf.deinit(&backend.device);

            compute.cuda.gaffine_u8_dequantize_f16.runWithFunction(
                &backend.kernel_arg_pack,
                &backend.device,
                dequant_f16_fn,
                &w.packed_data,
                &w.scales,
                &w.biases,
                &cache_buf,
                @intCast(w.cols),
                @intCast(w.rows),
                w.group_size,
                w.scales_dtype_tag,
            ) catch {
                cache_buf.deinit(&backend.device);
                return;
            };

            w.dequant_f16_cache = cache_buf;
            bytes_out.* += weight_f16_bytes;
        }
    }.run;

    // Helper to process a LinearWeight if it is gaffine_u8 or gaffine_u4.
    // Both get I8 caches for INT8 tensor core prefill (2x vs F16 GEMM).
    const maybeProcess = struct {
        fn run(
            backend: *CudaBackend,
            weight: *LinearWeight,
            bytes_out: *usize,
            count_out: *usize,
        ) void {
            switch (weight.*) {
                .gaffine_u8 => |*w| {
                    dequantU8Weight(backend, w, bytes_out);
                    count_out.* += 1;
                },
                .gaffine_u4 => |*w| {
                    if (backend.gaffine_u4_to_i8_function) |fused_fn| {
                        launchFusedToI8(backend, fused_fn, w, bytes_out);
                        count_out.* += 1;
                    }
                },
                else => {},
            }
        }
    }.run;

    // Helper to process an optional LinearWeight.
    const maybeProcessOpt = struct {
        fn run(
            backend: *CudaBackend,
            weight_opt: *?LinearWeight,
            bytes_out: *usize,
            count_out: *usize,
        ) void {
            if (weight_opt.*) |*w| {
                maybeProcess(backend, w, bytes_out, count_out);
            }
        }
    }.run;

    for (self.block_runtime.blocks) |*layer| {
        if (layer.attention_runtime) |*attn| {
            maybeProcess(self, &attn.q_proj, &total_bytes, &weight_count);
            maybeProcess(self, &attn.k_proj, &total_bytes, &weight_count);
            maybeProcess(self, &attn.v_proj, &total_bytes, &weight_count);
            maybeProcess(self, &attn.o_proj, &total_bytes, &weight_count);
            maybeProcess(self, &attn.w1, &total_bytes, &weight_count);
            maybeProcess(self, &attn.w2, &total_bytes, &weight_count);
            maybeProcess(self, &attn.w3, &total_bytes, &weight_count);
        }
        if (layer.shortconv_runtime) |*sc| {
            maybeProcess(self, &sc.in_proj, &total_bytes, &weight_count);
            maybeProcess(self, &sc.out_proj, &total_bytes, &weight_count);
            maybeProcessOpt(self, &sc.ffn_w1, &total_bytes, &weight_count);
            maybeProcessOpt(self, &sc.ffn_w2, &total_bytes, &weight_count);
            maybeProcessOpt(self, &sc.ffn_w3, &total_bytes, &weight_count);
        }
        if (layer.gated_delta_runtime) |*gd| {
            maybeProcess(self, &gd.in_proj, &total_bytes, &weight_count);
            maybeProcess(self, &gd.out_proj, &total_bytes, &weight_count);
            maybeProcessOpt(self, &gd.ffn_w1, &total_bytes, &weight_count);
            maybeProcessOpt(self, &gd.ffn_w2, &total_bytes, &weight_count);
            maybeProcessOpt(self, &gd.ffn_w3, &total_bytes, &weight_count);
        }
    }

    // Projection weight (lm_head).
    maybeProcess(self, &self.runtime_buffers.projection_weight, &total_bytes, &weight_count);

    // Build concatenated I8 QKV caches for fused prefill GEMM.
    // This merges Q+K+V I8 weights into one contiguous buffer so prefill
    // can run a single large GEMM instead of 3 separate ones.
    const I8CacheRef = struct { i8_buf: compute.cuda.Buffer, scales_buf: compute.cuda.Buffer };
    const getI8Cache = struct {
        fn get(weight: *const LinearWeight) ?I8CacheRef {
            return switch (weight.*) {
                .gaffine_u4 => |w| if (w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0)
                    .{ .i8_buf = w.dequant_i8_cache, .scales_buf = w.mean_scale_cache }
                else
                    null,
                .gaffine_u8 => |w| if (w.dequant_i8_cache.pointer != 0 and w.mean_scale_cache.pointer != 0)
                    .{ .i8_buf = w.dequant_i8_cache, .scales_buf = w.mean_scale_cache }
                else
                    null,
                else => null,
            };
        }
    }.get;

    for (self.block_runtime.blocks) |*layer| {
        const attn = &(layer.attention_runtime orelse continue);
        const q_ref = getI8Cache(&attn.q_proj) orelse continue;
        const k_ref = getI8Cache(&attn.k_proj) orelse continue;
        const v_ref = getI8Cache(&attn.v_proj) orelse continue;

        const in_dim = attn.q_proj.rows();
        if (in_dim != attn.k_proj.rows() or in_dim != attn.v_proj.rows()) continue;
        if (in_dim == 0) continue;

        const q_dim: u32 = @intCast(attn.q_proj.cols());
        const k_dim: u32 = @intCast(attn.k_proj.cols());
        const v_dim: u32 = @intCast(attn.v_proj.cols());

        const total_dim: usize = @as(usize, q_dim) + k_dim + v_dim;
        const i8_bytes = std.math.mul(usize, total_dim, in_dim) catch continue;
        const scales_bytes = std.math.mul(usize, total_dim, @sizeOf(f32)) catch continue;

        var concat_i8 = self.device.allocBuffer(i8_bytes) catch continue;
        var concat_scales = self.device.allocBuffer(scales_bytes) catch {
            concat_i8.deinit(&self.device);
            continue;
        };

        // D2D copy each weight's I8 cache into the concatenated buffer.
        const q_i8_bytes = @as(usize, q_dim) * in_dim;
        const k_i8_bytes = @as(usize, k_dim) * in_dim;
        const v_i8_bytes = @as(usize, v_dim) * in_dim;
        const q_scale_bytes = @as(usize, q_dim) * @sizeOf(f32);
        const k_scale_bytes = @as(usize, k_dim) * @sizeOf(f32);
        const v_scale_bytes = @as(usize, v_dim) * @sizeOf(f32);

        const ok = blk: {
            var dst = bufferSlice(&concat_i8, 0, q_i8_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &q_ref.i8_buf, q_i8_bytes) catch break :blk false;
            dst = bufferSlice(&concat_i8, q_i8_bytes, k_i8_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &k_ref.i8_buf, k_i8_bytes) catch break :blk false;
            dst = bufferSlice(&concat_i8, q_i8_bytes + k_i8_bytes, v_i8_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &v_ref.i8_buf, v_i8_bytes) catch break :blk false;

            dst = bufferSlice(&concat_scales, 0, q_scale_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &q_ref.scales_buf, q_scale_bytes) catch break :blk false;
            dst = bufferSlice(&concat_scales, q_scale_bytes, k_scale_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &k_ref.scales_buf, k_scale_bytes) catch break :blk false;
            dst = bufferSlice(&concat_scales, q_scale_bytes + k_scale_bytes, v_scale_bytes) catch break :blk false;
            dst.copyFrom(&self.device, &v_ref.scales_buf, v_scale_bytes) catch break :blk false;
            break :blk true;
        };

        if (ok) {
            attn.qkv_i8_concat = concat_i8;
            attn.qkv_scales_concat = concat_scales;
            attn.qkv_concat_dims = .{ q_dim, k_dim, v_dim };
            total_bytes += i8_bytes + scales_bytes;
        } else {
            concat_scales.deinit(&self.device);
            concat_i8.deinit(&self.device);
        }
    }

    self.dequant_cache_bytes = total_bytes;
    if (weight_count > 0) {
        try self.device.synchronize();
        log.info("inference", "CUDA gaffine dequant cache ready", .{
            .weights = weight_count,
            .cache_mib = total_bytes / (1024 * 1024),
        });
    }
}

pub fn resolveRequiredKernels(self: anytype) !void {
    for (required_kernels) |kernel| {
        const resolved = self.kernel_registry.resolveFunction(
            kernel.op_name,
            kernel.embedded_symbol,
        ) catch |err| {
            log.warn("inference", "CUDA kernel resolve failed", .{
                .op = kernel.op_name,
                .symbol = kernel.embedded_symbol,
                .reason = @errorName(err),
            });
            return err;
        };
        assignResolvedKernel(self, kernel.slot, resolved);
    }
}

pub fn assignResolvedKernel(
    self: anytype,
    slot: KernelSlot,
    resolved: compute.cuda.registry.ResolvedFunction,
) void {
    switch (slot) {
        .vector_add => {
            self.vector_add_function = resolved.function;
            self.vector_add_source = resolved.source;
        },
        .vector_add_scaled => {
            self.vector_add_scaled_function = resolved.function;
            self.vector_add_scaled_source = resolved.source;
        },
        .vector_add_rows_strided => {
            self.vector_add_rows_strided_function = resolved.function;
            self.vector_add_rows_strided_source = resolved.source;
        },
        .vector_add_scaled_rows_strided => {
            self.vector_add_scaled_rows_strided_function = resolved.function;
            self.vector_add_scaled_rows_strided_source = resolved.source;
        },
        .mul => {
            self.mul_function = resolved.function;
            self.mul_source = resolved.source;
        },
        .copy => {
            self.copy_function = resolved.function;
            self.copy_source = resolved.source;
        },
        .copy_u16 => {
            self.copy_u16_function = resolved.function;
            self.copy_u16_source = resolved.source;
        },
        .cast_f32_to_f16 => {
            self.cast_f32_to_f16_function = resolved.function;
            self.cast_f32_to_f16_source = resolved.source;
        },
        .cast_f32_to_bf16 => {
            self.cast_f32_to_bf16_function = resolved.function;
            self.cast_f32_to_bf16_source = resolved.source;
        },
        .embedding_lookup_f32 => {
            self.embedding_lookup_f32_function = resolved.function;
            self.embedding_lookup_f32_source = resolved.source;
        },
        .embedding_lookup_u16 => {
            self.embedding_lookup_u16_function = resolved.function;
            self.embedding_lookup_u16_source = resolved.source;
        },
        .embedding_lookup_u16_rows => {
            self.embedding_lookup_u16_rows_function = resolved.function;
            self.embedding_lookup_u16_rows_source = resolved.source;
        },
        .embedding_lookup_gaffine_u4 => {
            self.embedding_lookup_gaffine_u4_function = resolved.function;
            self.embedding_lookup_gaffine_u4_source = resolved.source;
        },
        .kv_write_f16 => {
            self.kv_write_f16_function = resolved.function;
            self.kv_write_f16_source = resolved.source;
        },
        .kv_write_f16_rows => {
            self.kv_write_f16_rows_function = resolved.function;
            self.kv_write_f16_rows_source = resolved.source;
        },
        .kv_write_f16_rows_ptrs => {
            self.kv_write_f16_rows_ptrs_function = resolved.function;
            self.kv_write_f16_rows_ptrs_source = resolved.source;
        },
        .rmsnorm => {
            self.rmsnorm_function = resolved.function;
            self.rmsnorm_source = resolved.source;
        },
        .rmsnorm_rows_strided => {
            self.rmsnorm_rows_strided_function = resolved.function;
            self.rmsnorm_rows_strided_source = resolved.source;
        },
        .rope => {
            self.rope_function = resolved.function;
            self.rope_source = resolved.source;
        },
        .rope_store_f16 => {
            self.rope_store_f16_function = resolved.function;
            self.rope_store_f16_source = resolved.source;
        },
        .attn_scores_heads_f32 => {
            self.attn_scores_heads_f32_function = resolved.function;
            self.attn_scores_heads_f32_source = resolved.source;
        },
        .attn_scores_heads_f16_kv => {
            self.attn_scores_heads_f16_kv_function = resolved.function;
            self.attn_scores_heads_f16_kv_source = resolved.source;
        },
        .attn_fused_heads_f16_kv => {
            self.attn_fused_heads_f16_kv_function = resolved.function;
            self.attn_fused_heads_f16_kv_source = resolved.source;
        },
        .attn_fused_decode_heads_f16_kv_ptrs => {
            self.attn_fused_decode_heads_f16_kv_ptrs_function = resolved.function;
            self.attn_fused_decode_heads_f16_kv_ptrs_source = resolved.source;
        },
        .attn_fused_prefill_heads_f16_kv => {
            self.attn_fused_prefill_heads_f16_kv_function = resolved.function;
            self.attn_fused_prefill_heads_f16_kv_source = resolved.source;
        },
        .attn_fused_prefill_heads_f16_kv_gqa => {
            self.attn_fused_prefill_heads_f16_kv_gqa_function = resolved.function;
            self.attn_fused_prefill_heads_f16_kv_gqa_source = resolved.source;
        },
        .causal_attn_softmax_f32 => {
            self.causal_attn_softmax_f32_function = resolved.function;
            self.causal_attn_softmax_f32_source = resolved.source;
        },
        .softmax_rows => {
            self.softmax_rows_function = resolved.function;
            self.softmax_rows_source = resolved.source;
        },
        .attn_weighted_sum_heads_f32 => {
            self.attn_weighted_sum_heads_f32_function = resolved.function;
            self.attn_weighted_sum_heads_f32_source = resolved.source;
        },
        .attn_weighted_sum_heads_f16_kv => {
            self.attn_weighted_sum_heads_f16_kv_function = resolved.function;
            self.attn_weighted_sum_heads_f16_kv_source = resolved.source;
        },
        .rope_rows_ptrs => {
            self.rope_rows_ptrs_function = resolved.function;
            self.rope_rows_ptrs_source = resolved.source;
        },
        .attn_scores_heads_f16_kv_ptrs => {
            self.attn_scores_heads_f16_kv_ptrs_function = resolved.function;
            self.attn_scores_heads_f16_kv_ptrs_source = resolved.source;
        },
        .softmax_rows_dynamic_cols_ptrs => {
            self.softmax_rows_dynamic_cols_ptrs_function = resolved.function;
            self.softmax_rows_dynamic_cols_ptrs_source = resolved.source;
        },
        .attn_weighted_sum_heads_f16_kv_ptrs => {
            self.attn_weighted_sum_heads_f16_kv_ptrs_function = resolved.function;
            self.attn_weighted_sum_heads_f16_kv_ptrs_source = resolved.source;
        },
        .kv_write_i8 => {
            self.kv_write_i8_function = resolved.function;
            self.kv_write_i8_source = resolved.source;
        },
        .kv_write_i8_rows => {
            self.kv_write_i8_rows_function = resolved.function;
            self.kv_write_i8_rows_source = resolved.source;
        },
        .kv_write_i8_rows_ptrs => {
            self.kv_write_i8_rows_ptrs_function = resolved.function;
            self.kv_write_i8_rows_ptrs_source = resolved.source;
        },
        .rope_store_i8 => {
            self.rope_store_i8_function = resolved.function;
            self.rope_store_i8_source = resolved.source;
        },
        .attn_scores_heads_i8_kv => {
            self.attn_scores_heads_i8_kv_function = resolved.function;
            self.attn_scores_heads_i8_kv_source = resolved.source;
        },
        .attn_weighted_sum_heads_i8_kv => {
            self.attn_weighted_sum_heads_i8_kv_function = resolved.function;
            self.attn_weighted_sum_heads_i8_kv_source = resolved.source;
        },
        .attn_fused_heads_i8_kv => {
            self.attn_fused_heads_i8_kv_function = resolved.function;
            self.attn_fused_heads_i8_kv_source = resolved.source;
        },
        .attn_fused_decode_heads_i8_kv_ptrs => {
            self.attn_fused_decode_heads_i8_kv_ptrs_function = resolved.function;
            self.attn_fused_decode_heads_i8_kv_ptrs_source = resolved.source;
        },
        .attn_fused_prefill_heads_i8_kv => {
            self.attn_fused_prefill_heads_i8_kv_function = resolved.function;
            self.attn_fused_prefill_heads_i8_kv_source = resolved.source;
        },
        .attn_fused_prefill_heads_i8_kv_gqa => {
            self.attn_fused_prefill_heads_i8_kv_gqa_function = resolved.function;
            self.attn_fused_prefill_heads_i8_kv_gqa_source = resolved.source;
        },
        .attn_scores_heads_i8_kv_ptrs => {
            self.attn_scores_heads_i8_kv_ptrs_function = resolved.function;
            self.attn_scores_heads_i8_kv_ptrs_source = resolved.source;
        },
        .attn_weighted_sum_heads_i8_kv_ptrs => {
            self.attn_weighted_sum_heads_i8_kv_ptrs_function = resolved.function;
            self.attn_weighted_sum_heads_i8_kv_ptrs_source = resolved.source;
        },
        .silu => {
            self.silu_function = resolved.function;
            self.silu_source = resolved.source;
        },
        .silu_mul => {
            self.silu_mul_function = resolved.function;
            self.silu_mul_source = resolved.source;
        },
        .gelu_mul => {
            self.gelu_mul_function = resolved.function;
            self.gelu_mul_source = resolved.source;
        },
        .shortconv_step => {
            self.shortconv_step_function = resolved.function;
            self.shortconv_step_source = resolved.source;
        },
        .gated_attention_compact_q => {
            self.gated_attention_compact_q_function = resolved.function;
            self.gated_attention_compact_q_source = resolved.source;
        },
        .gated_attention_output_gate => {
            self.gated_attention_output_gate_function = resolved.function;
            self.gated_attention_output_gate_source = resolved.source;
        },
        .gated_delta_conv => {
            self.gated_delta_conv_function = resolved.function;
            self.gated_delta_conv_source = resolved.source;
        },
        .gated_delta_conv_silu => {
            self.gated_delta_conv_silu_function = resolved.function;
            self.gated_delta_conv_silu_source = resolved.source;
        },
        .gated_delta_conv_silu_rows => {
            self.gated_delta_conv_silu_rows_function = resolved.function;
            self.gated_delta_conv_silu_rows_source = resolved.source;
        },
        .gated_delta_conv_silu_rows_ptrs => {
            self.gated_delta_conv_silu_rows_ptrs_function = resolved.function;
            self.gated_delta_conv_silu_rows_ptrs_source = resolved.source;
        },
        .gated_delta_advance_ring_heads => {
            self.gated_delta_advance_ring_heads_function = resolved.function;
            self.gated_delta_advance_ring_heads_source = resolved.source;
        },
        .gated_delta_qk_norm => {
            self.gated_delta_qk_norm_function = resolved.function;
            self.gated_delta_qk_norm_source = resolved.source;
        },
        .gated_delta_ssm => {
            self.gated_delta_ssm_function = resolved.function;
            self.gated_delta_ssm_source = resolved.source;
        },
        .gated_delta_ssm_rows => {
            self.gated_delta_ssm_rows_function = resolved.function;
            self.gated_delta_ssm_rows_source = resolved.source;
        },
        .gated_delta_ssm_rows_ptrs => {
            self.gated_delta_ssm_rows_ptrs_function = resolved.function;
            self.gated_delta_ssm_rows_ptrs_source = resolved.source;
        },
        .gated_delta_ssm_rows_i8 => {
            self.gated_delta_ssm_rows_i8_function = resolved.function;
            self.gated_delta_ssm_rows_i8_source = resolved.source;
        },
        .gated_delta_ssm_rows_ptrs_i8 => {
            self.gated_delta_ssm_rows_ptrs_i8_function = resolved.function;
            self.gated_delta_ssm_rows_ptrs_i8_source = resolved.source;
        },
        .gated_delta_rmsnorm_silu_mul => {
            self.gated_delta_rmsnorm_silu_mul_function = resolved.function;
            self.gated_delta_rmsnorm_silu_mul_source = resolved.source;
        },
        .gated_delta_rmsnorm_silu_mul_rows => {
            self.gated_delta_rmsnorm_silu_mul_rows_function = resolved.function;
            self.gated_delta_rmsnorm_silu_mul_rows_source = resolved.source;
        },
        .argmax => {
            self.argmax_function = resolved.function;
            self.argmax_source = resolved.source;
        },
        .matmul_f16 => {
            self.matmul_f16_function = resolved.function;
            self.matmul_f16_source = resolved.source;
        },
        .matmul_bf16 => {
            self.matmul_bf16_function = resolved.function;
            self.matmul_bf16_source = resolved.source;
        },
        .matvec_f16 => {
            self.matvec_f16_function = resolved.function;
            self.matvec_f16_source = resolved.source;
        },
        .matvec_bf16 => {
            self.matvec_bf16_function = resolved.function;
            self.matvec_bf16_source = resolved.source;
        },
        .matvec_gate_up_f16 => {
            self.matvec_gate_up_f16_function = resolved.function;
            self.matvec_gate_up_f16_source = resolved.source;
        },
        .matvec_gate_up_bf16 => {
            self.matvec_gate_up_bf16_function = resolved.function;
            self.matvec_gate_up_bf16_source = resolved.source;
        },
        .matvec_gate_up_silu_f16 => {
            self.matvec_gate_up_silu_f16_function = resolved.function;
            self.matvec_gate_up_silu_f16_source = resolved.source;
        },
        .matvec_gate_up_silu_bf16 => {
            self.matvec_gate_up_silu_bf16_function = resolved.function;
            self.matvec_gate_up_silu_bf16_source = resolved.source;
        },
        .matvec_qkv_f16 => {
            self.matvec_qkv_f16_function = resolved.function;
            self.matvec_qkv_f16_source = resolved.source;
        },
        .matvec_qkv_bf16 => {
            self.matvec_qkv_bf16_function = resolved.function;
            self.matvec_qkv_bf16_source = resolved.source;
        },
        .gaffine_u4_matvec => {
            self.gaffine_u4_matvec_function = resolved.function;
            self.gaffine_u4_matvec_source = resolved.source;
        },
        .gaffine_u4_matvec_tile8 => {
            self.gaffine_u4_matvec_tile8_function = resolved.function;
            self.gaffine_u4_matvec_tile8_source = resolved.source;
        },
        .gaffine_u8_matvec => {
            self.gaffine_u8_matvec_function = resolved.function;
            self.gaffine_u8_matvec_source = resolved.source;
        },
        .gaffine_u4_matvec_gate_up => {
            self.gaffine_u4_matvec_gate_up_function = resolved.function;
            self.gaffine_u4_matvec_gate_up_source = resolved.source;
        },
        .gaffine_u4_matvec_qkv => {
            self.gaffine_u4_matvec_qkv_function = resolved.function;
            self.gaffine_u4_matvec_qkv_source = resolved.source;
        },
        .gaffine_u4_matvec_qkv_tile8 => {
            self.gaffine_u4_matvec_qkv_tile8_function = resolved.function;
            self.gaffine_u4_matvec_qkv_tile8_source = resolved.source;
        },
        .gaffine_u8_matvec_qkv => {
            self.gaffine_u8_matvec_qkv_function = resolved.function;
            self.gaffine_u8_matvec_qkv_source = resolved.source;
        },
        .gaffine_u8_matvec_gate_up => {
            self.gaffine_u8_matvec_gate_up_function = resolved.function;
            self.gaffine_u8_matvec_gate_up_source = resolved.source;
        },
        .gaffine_u4_matvec_gate_up_silu => {
            self.gaffine_u4_matvec_gate_up_silu_function = resolved.function;
            self.gaffine_u4_matvec_gate_up_silu_source = resolved.source;
        },
        .gaffine_u4_matvec_gate_up_silu_tile8 => {
            self.gaffine_u4_matvec_gate_up_silu_tile8_function = resolved.function;
            self.gaffine_u4_matvec_gate_up_silu_tile8_source = resolved.source;
        },
        .gaffine_u8_matvec_gate_up_silu => {
            self.gaffine_u8_matvec_gate_up_silu_function = resolved.function;
            self.gaffine_u8_matvec_gate_up_silu_source = resolved.source;
        },
        .gaffine_u4_dequant_f16 => {
            self.gaffine_u4_dequant_f16_function = resolved.function;
            self.gaffine_u4_dequant_f16_source = resolved.source;
        },
        .gaffine_u8_dequant_f16 => {
            self.gaffine_u8_dequant_f16_function = resolved.function;
            self.gaffine_u8_dequant_f16_source = resolved.source;
        },
    }
}

pub fn tryLoadSideloadModule(self: anytype) !bool {
    const base_url_raw = std.process.getEnvVarOwned(self.allocator, compute.cuda.sideload.kernel_base_url_env) catch |err| switch (err) {
        error.EnvironmentVariableNotFound => return false,
        else => return err,
    };
    defer self.allocator.free(base_url_raw);
    const base_url = std.mem.trim(u8, base_url_raw, " \t\r\n");
    if (base_url.len == 0) return false;

    const capability = self.device.computeCapability() catch |err| switch (err) {
        error.CudaQueryUnavailable => return false,
        else => return err,
    };
    const arch = try compute.cuda.sideload.archTag(self.allocator, capability.major, capability.minor);
    defer self.allocator.free(arch);

    const cache_dir = try compute.cuda.sideload.resolveCacheDir(self.allocator);
    defer self.allocator.free(cache_dir);
    try compute.cuda.sideload.ensureCacheDir(cache_dir);

    const manifest_bytes = try compute.cuda.sideload.loadOrFetchManifest(
        self.allocator,
        cache_dir,
        arch,
        base_url,
    );
    defer self.allocator.free(manifest_bytes);
    var parsed_manifest = try compute.cuda.manifest.parse(self.allocator, manifest_bytes);
    defer parsed_manifest.deinit();
    try compute.cuda.manifest.ensureCompatible(
        parsed_manifest.manifest,
        arch,
        compute.cuda.manifest.kernel_abi_version,
    );

    const artifact_bytes = try compute.cuda.sideload.loadOrFetchArtifact(
        self.allocator,
        cache_dir,
        arch,
        base_url,
        parsed_manifest.manifest.sha256,
    );
    defer self.allocator.free(artifact_bytes);

    try self.kernel_registry.loadSideloadModule(
        manifest_bytes,
        artifact_bytes,
        arch,
        compute.cuda.manifest.kernel_abi_version,
    );
    log.info("inference", "CUDA sideload payload loaded", .{
        .arch = arch,
        .cache_dir = cache_dir,
    });
    return true;
}

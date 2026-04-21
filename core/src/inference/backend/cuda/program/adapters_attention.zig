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
const requireAttentionRuntimeBinding = handles.requireAttentionRuntimeBinding;
const optionalDeviceTensorFromWeightHandle = handles.optionalDeviceTensorFromWeightHandle;
const linearWeightFromWeightHandle = handles.linearWeightFromWeightHandle;
const instructionWeightSlice = handles.instructionWeightSlice;
const instructionParams = handles.instructionParams;
const instructionIoSlices = handles.instructionIoSlices;
const bufferFromTensorHandle = handles.bufferFromTensorHandle;
const requireStateValue = handles.requireStateValue;
const common = @import("common.zig");
const finishAttentionRecord = common.finishAttentionRecord;
const deviceTensorFromWeightHandle = handles.deviceTensorFromWeightHandle;

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
    const shape = try resolveAttentionShapeForInstruction(cfg, q_norm_weight, k_norm_weight, ctx);
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
        var local_ctx = ctx.*;
        local_ctx.head_dim_u32 = shape.head_dim_u32;
        local_ctx.rope_dim_u32 = shape.rope_dim_u32;
        local_ctx.n_heads_u32 = shape.n_heads_u32;
        local_ctx.n_kv_heads_u32 = shape.n_kv_heads_u32;
        engine_mixers.runBatchedDecodeAttentionMixer(
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
            &local_ctx,
            batch,
        ) catch |err| {
            log.warn("inference", "CUDA layer-program batched decode attention failed", .{
                .layer_index = ctx.layer_index,
                .op_index = ctx.op_index,
                .active_rows = ctx.active_rows_u32,
                .q_dim = cfg.q_dim,
                .q_projection_dim = cfg.q_projection_dim,
                .kv_dim = cfg.kv_dim,
                .head_dim = shape.head_dim_u32,
                .rope_dim = shape.rope_dim_u32,
                .n_heads = shape.n_heads_u32,
                .n_kv_heads = shape.n_kv_heads_u32,
                .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
                .reason = @errorName(err),
            });
            return err;
        };
        return;
    }

    // Provide concat I8 QKV cache for fused prefill GEMM.
    self.active_qkv_concat = if (attention_binding.qkv_i8_concat.pointer != 0)
        .{ .i8_buf = attention_binding.qkv_i8_concat, .scales_buf = attention_binding.qkv_scales_concat, .dims = attention_binding.qkv_concat_dims }
    else
        null;
    defer self.active_qkv_concat = null;

    // Resolve read K/V cache for this attention layer.
    // Same-device sharing: read from the local source layer's binding.
    // Cross-device mirror: read from block_runtime.mirror_kv.
    // No sharing: read from own cache.
    var read_k_cache = &attention_binding.k_cache;
    var read_v_cache = &attention_binding.v_cache;
    var read_k_scale = &attention_binding.k_scale;
    var read_v_scale = &attention_binding.v_scale;
    if (attention_binding.kv_shared_source_layer) |src_layer| {
        const src_binding = try requireAttentionRuntimeBinding(kv_state, src_layer);
        read_k_cache = &src_binding.k_cache;
        read_v_cache = &src_binding.v_cache;
        read_k_scale = &src_binding.k_scale;
        read_v_scale = &src_binding.v_scale;
    } else if (attention_binding.kv_shared_source_slot_kv_index) |src_idx| {
        const n_real = kv_state.block_runtime.attention_block_count;
        if (src_idx >= n_real and src_idx - n_real < kv_state.block_runtime.mirror_kv.len) {
            const mk = &kv_state.block_runtime.mirror_kv[src_idx - n_real];
            read_k_cache = &mk.k;
            read_v_cache = &mk.v;
            read_k_scale = &mk.k_scale;
            read_v_scale = &mk.v_scale;
        }
    }

    if (!cfg.query_gate) {
        engine_mixers.runAttentionMixerPrefillBatchedNoQueryGate(
            self,
            cfg,
            &attention_binding.k_cache,
            &attention_binding.v_cache,
            &attention_binding.k_scale,
            &attention_binding.v_scale,
            read_k_cache,
            read_v_cache,
            read_k_scale,
            read_v_scale,
            &q_proj,
            &k_proj,
            &v_proj,
            &o_proj,
            q_norm_weight,
            k_norm_weight,
            input,
            output,
            ctx.d_model_u32,
            shape.head_dim_u32,
            shape.rope_dim_u32,
            shape.n_heads_u32,
            shape.n_kv_heads_u32,
            ctx.seq_len_u32,
            ctx.global_rope_theta,
            ctx.local_rope_theta,
            ctx.rope_function,
            ctx.copy_function,
            ctx.cast_f32_to_f16_function,
            ctx.kv_write_f16_function,
            ctx.rope_store_f16_function,
            ctx.attention_kernels,
        ) catch |err| {
            log.warn("inference", "CUDA layer-program prefill attention failed", .{
                .layer_index = ctx.layer_index,
                .op_index = ctx.op_index,
                .active_rows = ctx.active_rows_u32,
                .seq_len = ctx.seq_len_u32,
                .q_dim = cfg.q_dim,
                .q_projection_dim = cfg.q_projection_dim,
                .kv_dim = cfg.kv_dim,
                .head_dim = shape.head_dim_u32,
                .rope_dim = shape.rope_dim_u32,
                .n_heads = shape.n_heads_u32,
                .n_kv_heads = shape.n_kv_heads_u32,
                .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
                .reason = @errorName(err),
            });
            return err;
        };
        return;
    }

    engine_mixers.runAttentionMixerPrefillBatchedWithQueryGate(
        self,
        cfg,
        &attention_binding.k_cache,
        &attention_binding.v_cache,
        &attention_binding.k_scale,
        &attention_binding.v_scale,
        read_k_cache,
        read_v_cache,
        read_k_scale,
        read_v_scale,
        &q_proj,
        &k_proj,
        &v_proj,
        &o_proj,
        q_norm_weight,
        k_norm_weight,
        input,
        output,
        ctx.d_model_u32,
        shape.head_dim_u32,
        shape.rope_dim_u32,
        shape.n_heads_u32,
        shape.n_kv_heads_u32,
        ctx.seq_len_u32,
        ctx.global_rope_theta,
        ctx.local_rope_theta,
        ctx.rope_function,
        ctx.copy_function,
        ctx.cast_f32_to_f16_function,
        ctx.kv_write_f16_function,
        ctx.rope_store_f16_function,
        ctx.attention_kernels,
    ) catch |err| {
        log.warn("inference", "CUDA layer-program prefill attention(query-gate) failed", .{
            .layer_index = ctx.layer_index,
            .op_index = ctx.op_index,
            .active_rows = ctx.active_rows_u32,
            .seq_len = ctx.seq_len_u32,
            .q_dim = cfg.q_dim,
            .q_projection_dim = cfg.q_projection_dim,
            .kv_dim = cfg.kv_dim,
            .head_dim = shape.head_dim_u32,
            .rope_dim = shape.rope_dim_u32,
            .n_heads = shape.n_heads_u32,
            .n_kv_heads = shape.n_kv_heads_u32,
            .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
            .reason = @errorName(err),
        });
        return err;
    };
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
    n_heads_u32: u32,
    attention_scale: f32,
    rope_dim_u32: u32,
    position_u32: u32,
    theta: f32,
) !AttentionPath {
    const attention_start_ns: i128 = std.time.nanoTimestamp();
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
            const n_kv_heads_u32 = n_heads_u32 / kv_groups_u32;
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
                compute.cuda.attn_fused_heads_f16_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_f16_kv_function.?,
                    q_stage,
                    &k_cache_view,
                    &v_cache_view,
                    context_stage,
                    n_heads_u32,
                    effective_seq_len_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    head_dim_u32,
                    attention_scale,
                    rope_dim_u32,
                    position_u32,
                    theta,
                ) catch |err| {
                    log.warn("inference", "CUDA attention fused_heads_f16_kv launch failed", .{
                        .seq_len = effective_seq_len_u32,
                        .head_dim = head_dim_u32,
                        .kv_dim = kv_dim_u32,
                        .kv_groups = kv_groups_u32,
                        .rope_dim = rope_dim_u32,
                        .position = position_u32,
                        .reason = @errorName(err),
                    });
                    return err;
                };
                return finishAttentionRecord(self, .fused_heads_f16_kv, attention_start_ns, cfg.is_causal);
            }

            const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
            const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();
            compute.cuda.attn_scores_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                q_stage,
                &k_cache_view,
                attn_scores_dev,
                n_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
                attention_scale,
            ) catch |err| {
                log.warn("inference", "CUDA attention scores_heads_f16_kv launch failed", .{
                    .seq_len = effective_seq_len_u32,
                    .head_dim = head_dim_u32,
                    .kv_dim = kv_dim_u32,
                    .kv_groups = kv_groups_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
            compute.cuda.softmax_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
                attn_scores_dev,
                attn_probs_dev,
                n_heads_u32,
                effective_seq_len_u32,
            ) catch |err| {
                log.warn("inference", "CUDA attention softmax_rows launch failed", .{
                    .rows = @as(u32, n_heads_u32),
                    .cols = effective_seq_len_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
            compute.cuda.attn_weighted_sum_heads_f16_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &v_cache_view,
                context_stage,
                n_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            ) catch |err| {
                log.warn("inference", "CUDA attention weighted_sum_heads_f16_kv launch failed", .{
                    .seq_len = effective_seq_len_u32,
                    .head_dim = head_dim_u32,
                    .kv_dim = kv_dim_u32,
                    .kv_groups = kv_groups_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
            return finishAttentionRecord(self, .heads_f16_kv, attention_start_ns, cfg.is_causal);
        },
        .i8 => {
            const n_kv_heads_u32 = n_heads_u32 / kv_groups_u32;
            if (!cfg.query_gate and attention_mod.useFusedHeadsF16Kv(
                attention_policy_config,
                seq_len_u32,
                cfg.sliding_window,
                cfg.is_causal,
                head_dim_u32,
                kernels.attn_fused_heads_i8_kv_function != null,
            )) {
                const fused_i8_ok = blk: {
                    compute.cuda.attn_fused_heads_i8_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernels.attn_fused_heads_i8_kv_function.?,
                        q_stage,
                        &k_cache_view,
                        &v_cache_view,
                        &k_scale_view,
                        &v_scale_view,
                        context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        effective_seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_u32,
                        theta,
                    ) catch |err| {
                        if (err == error.CudaKernelLaunchFailed) {
                            log.warn("inference", "CUDA attention fused_heads_i8_kv launch failed; falling back to separate i8 attention", .{
                                .seq_len = effective_seq_len_u32,
                                .head_dim = head_dim_u32,
                                .kv_dim = kv_dim_u32,
                                .kv_groups = kv_groups_u32,
                                .rope_dim = rope_dim_u32,
                                .position = position_u32,
                            });
                            break :blk false;
                        }
                        return err;
                    };
                    break :blk true;
                };
                if (fused_i8_ok) return finishAttentionRecord(self, .fused_heads_i8_kv, attention_start_ns, cfg.is_causal);
            }

            const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
            const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();
            compute.cuda.attn_scores_heads_i8_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_i8_kv_function orelse return error.CudaKernelUnavailable,
                q_stage,
                &k_cache_view,
                &k_scale_view,
                attn_scores_dev,
                n_heads_u32,
                n_kv_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
                attention_scale,
            ) catch |err| {
                log.warn("inference", "CUDA attention scores_heads_i8_kv launch failed", .{
                    .seq_len = effective_seq_len_u32,
                    .head_dim = head_dim_u32,
                    .kv_dim = kv_dim_u32,
                    .kv_groups = kv_groups_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
            compute.cuda.softmax_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
                attn_scores_dev,
                attn_probs_dev,
                n_heads_u32,
                effective_seq_len_u32,
            ) catch |err| {
                log.warn("inference", "CUDA attention softmax_rows(i8) launch failed", .{
                    .rows = @as(u32, n_heads_u32),
                    .cols = effective_seq_len_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
            compute.cuda.attn_weighted_sum_heads_i8_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_i8_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &v_cache_view,
                &v_scale_view,
                context_stage,
                n_heads_u32,
                n_kv_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            ) catch |err| {
                log.warn("inference", "CUDA attention weighted_sum_heads_i8_kv launch failed", .{
                    .seq_len = effective_seq_len_u32,
                    .head_dim = head_dim_u32,
                    .kv_dim = kv_dim_u32,
                    .kv_groups = kv_groups_u32,
                    .reason = @errorName(err),
                });
                return err;
            };
            return finishAttentionRecord(self, .heads_i8_kv, attention_start_ns, cfg.is_causal);
        },
        .fp8 => {
            const n_kv_heads_u32 = n_heads_u32 / kv_groups_u32;
            if (!cfg.query_gate and attention_mod.useFusedHeadsF16Kv(
                attention_policy_config,
                seq_len_u32,
                cfg.sliding_window,
                cfg.is_causal,
                head_dim_u32,
                kernels.attn_fused_heads_fp8_kv_function != null,
            )) {
                try compute.cuda.attn_fused_heads_fp8_kv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kernels.attn_fused_heads_fp8_kv_function.?,
                    q_stage,
                    &k_cache_view,
                    &v_cache_view,
                    &k_scale_view,
                    &v_scale_view,
                    context_stage,
                    n_heads_u32,
                    n_kv_heads_u32,
                    effective_seq_len_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    head_dim_u32,
                    attention_scale,
                    rope_dim_u32,
                    position_u32,
                    theta,
                );
                return finishAttentionRecord(self, .fused_heads_fp8_kv, attention_start_ns, cfg.is_causal);
            }

            const attn_scores_dev = try self.runtime_buffers.requireAttentionScoresDev();
            const attn_probs_dev = try self.runtime_buffers.requireAttentionProbsDev();
            try compute.cuda.attn_scores_heads_fp8_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_scores_heads_fp8_kv_function orelse return error.CudaKernelUnavailable,
                q_stage,
                &k_cache_view,
                &k_scale_view,
                attn_scores_dev,
                n_heads_u32,
                n_kv_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
                attention_scale,
            );
            try compute.cuda.softmax_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.softmax_rows_function orelse return error.CudaKernelUnavailable,
                attn_scores_dev,
                attn_probs_dev,
                n_heads_u32,
                effective_seq_len_u32,
            );
            try compute.cuda.attn_weighted_sum_heads_fp8_kv.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                kernels.attn_weighted_sum_heads_fp8_kv_function orelse return error.CudaKernelUnavailable,
                attn_probs_dev,
                &v_cache_view,
                &v_scale_view,
                context_stage,
                n_heads_u32,
                n_kv_heads_u32,
                effective_seq_len_u32,
                kv_dim_u32,
                kv_groups_u32,
                head_dim_u32,
            );
            return finishAttentionRecord(self, .heads_fp8_kv, attention_start_ns, cfg.is_causal);
        },
    }
}

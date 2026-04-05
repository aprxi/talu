//! Mixer (attention, short conv, gated delta) and FFN step functions.
//!
//! Contains attention mixer steps (single-token, batched decode, prefill),
//! short convolution mixer, gated delta mixer, FFN step, and supporting
//! workspace management. Functions use `self: anytype` to avoid circular
//! imports with engine.zig.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const dtype = @import("../../../dtype.zig");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");
const attention_mod = @import("attention.zig");
const cpu_kernels = @import("../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;

// --- Shared types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const LayerAttentionExecConfig = engine_types.LayerAttentionExecConfig;
const LayerAttentionRuntime = engine_types.LayerAttentionRuntime;
const LinearWeight = engine_types.LinearWeight;
const DeviceTensor = engine_types.DeviceTensor;
const AttentionKernelSet = engine_types.AttentionKernelSet;
const ShortConvBlockRuntime = engine_types.ShortConvBlockRuntime;
const ShortConvExecConfig = engine_types.ShortConvExecConfig;
const GatedDeltaBlockRuntime = engine_types.GatedDeltaBlockRuntime;
const bufferF32RowCount = engine_types.bufferF32RowCount;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_fused_attention_f16_kv = engine_types.enable_fused_attention_f16_kv;
const max_fused_attention_f16_kv_seq_len = engine_types.max_fused_attention_f16_kv_seq_len;
const max_supported_fused_f16_kv_head_dim = engine_types.max_supported_fused_f16_kv_head_dim;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const attention_policy_config = engine_types.attention_policy_config;
const min_flash_decode_blocks_default: u32 = 8;
const min_flash_decode_blocks_low_kv_heads: u32 = 1024;

fn debugKernelSyncEnabled() bool {
    const raw = std.posix.getenv("TALU_CUDA_DEBUG_SYNC") orelse return false;
    return std.mem.eql(u8, raw, "1") or std.ascii.eqlIgnoreCase(raw, "true");
}

fn effectiveLayerRopeTheta(
    self: anytype,
    cfg: *const LayerAttentionExecConfig,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    global_rope_theta: f32,
    local_rope_theta: f32,
) f32 {
    _ = head_dim_u32;
    if (cfg.sliding_window > 0) return local_rope_theta;
    if (global_rope_theta <= 0.0 or rope_dim_u32 == 0) return global_rope_theta;
    if (self.loaded.config.global_head_dim <= 0) return global_rope_theta;

    const global_head_dim: f32 = @floatFromInt(self.loaded.config.global_head_dim);
    const rope_dim: f32 = @floatFromInt(rope_dim_u32);
    if (global_head_dim < rope_dim or global_head_dim <= 0.0) return global_rope_theta;

    // Proportional RoPE models (Gemma4) define frequencies over global_head_dim,
    // while CUDA kernels parameterize frequencies by rope_dim. Adjust theta so
    // theta^(2i/rope_dim) matches the intended theta^(2i/global_head_dim).
    const exponent = rope_dim / global_head_dim;
    return std.math.pow(f32, global_rope_theta, exponent);
}

fn attentionScaleForHeadDim(self: anytype, head_dim_u32: u32) f32 {
    if (self.loaded.config.attention_multiplier > 0.0 or self.loaded.config.query_pre_attn_scalar > 0.0) {
        return self.attention_scale;
    }
    if (head_dim_u32 == 0) return self.attention_scale;
    return 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim_u32)));
}

fn applyValueNormInPlace(
    self: anytype,
    values: *compute.cuda.Buffer,
    rows: usize,
    n_kv_heads_u32: u32,
    head_dim_u32: u32,
) !void {
    if (!self.loaded.config.use_v_norm) return;
    if (head_dim_u32 == 0) return error.InvalidArgument;
    const head_dim: usize = @intCast(head_dim_u32);
    if (head_dim > self.d_model or head_dim > self.runtime_buffers.hidden_host.len) return error.InvalidArgument;

    const norm_weight_bytes = std.math.mul(usize, head_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    // Use attn_context_dev as scratch (unused before attention context computation).
    var norm_weight_dev = bufferSlice(&self.runtime_buffers.attn_context_dev, 0, norm_weight_bytes) catch return error.InvalidArgument;

    const norm_weight_host = self.runtime_buffers.hidden_host[0..head_dim];
    @memset(norm_weight_host, 1.0);
    norm_weight_dev.upload(&self.device, std.mem.sliceAsBytes(norm_weight_host)) catch |err| {
        // Distinguish prior async error from genuine upload failure.
        const has_prior_error: u8 = if (self.device.synchronize()) 0 else |_| 1;
        log.warn("inference", "v_norm weight upload failed", .{
            .head_dim = head_dim_u32,
            .buf_size = self.runtime_buffers.attn_context_dev.size,
            .upload_bytes = norm_weight_bytes,
            .prior_async_error = has_prior_error,
            .reason = @errorName(err),
        });
        return err;
    };

    const v_norm_rows = std.math.mul(u32, @intCast(rows), n_kv_heads_u32) catch return error.InvalidArgument;
    compute.cuda.rmsnorm.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.rmsnorm_function orelse return error.CudaKernelUnavailable,
        values,
        &norm_weight_dev,
        values,
        v_norm_rows,
        head_dim_u32,
        self.norm_eps,
        0.0,
    ) catch |err| {
        log.warn("inference", "v_norm rmsnorm kernel failed", .{
            .rows = v_norm_rows,
            .head_dim = head_dim_u32,
            .reason = @errorName(err),
        });
        return err;
    };
}

const Tensor = tensor.Tensor;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("engine_ops.zig");

// --- Forward pass from engine_forward.zig ---
const engine_forward = @import("engine_forward.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const bufferSlice = engine_weights.bufferSlice;
const resizeScratchBuffer = engine_weights.resizeScratchBuffer;

pub fn runAttentionMixerStep(
    self: anytype,
    cfg: *const LayerAttentionExecConfig,
    k_cache: *const compute.cuda.Buffer,
    v_cache: *const compute.cuda.Buffer,
    k_scale: *const compute.cuda.Buffer,
    v_scale: *const compute.cuda.Buffer,
    read_k_cache: *const compute.cuda.Buffer,
    read_v_cache: *const compute.cuda.Buffer,
    read_k_scale: *const compute.cuda.Buffer,
    read_v_scale: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    o_proj: *const LinearWeight,
    q_norm_weight: ?*const DeviceTensor,
    k_norm_weight: ?*const DeviceTensor,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    d_model_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
    seq_len_u32: u32,
    position: usize,
    position_u32: u32,
    global_rope_theta: f32,
    local_rope_theta: f32,
    rope_function: compute.cuda.Function,
    _: compute.cuda.Function,
    cast_f32_to_f16_function: ?compute.cuda.Function,
    kv_write_f16_function: ?compute.cuda.Function,
    rope_store_f16_function: ?compute.cuda.Function,
    attention_kernels: AttentionKernelSet,
    residual_buf: ?compute.cuda.Buffer,
) !void {
    const layer_rope_theta = effectiveLayerRopeTheta(
        self,
        cfg,
        head_dim_u32,
        rope_dim_u32,
        global_rope_theta,
        local_rope_theta,
    );
    const attention_scale = attentionScaleForHeadDim(self, head_dim_u32);
    const stage_rows = bufferF32RowCount(input, @intCast(d_model_u32)) catch |err| {
        log.warn("inference", "CUDA attention staged row count invalid", .{
            .seq_len = seq_len_u32,
            .input_bytes = input.size,
            .d_model = d_model_u32,
            .reason = @errorName(err),
        });
        return err;
    };
    const q_stage_bytes = std.math.mul(usize, stage_rows, cfg.q_projection_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const q_values_bytes = std.math.mul(usize, stage_rows, cfg.q_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const kv_stage_bytes = std.math.mul(usize, stage_rows, cfg.kv_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const context_stage_bytes = std.math.mul(usize, stage_rows, o_proj.rows() * @sizeOf(f32)) catch return error.InvalidArgument;
    var q_projection_stage = if (cfg.query_gate)
        try bufferSlice(&self.runtime_buffers.query_gate_proj_dev, 0, q_stage_bytes)
    else
        try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_stage_bytes);
    var attn_q_stage = q_projection_stage;
    var q_values_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_values_bytes);
    var attn_k_stage = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, kv_stage_bytes);
    var attn_v_stage = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, kv_stage_bytes);
    var attn_context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_stage_bytes);
    _ = engine_ops.runQkvProjection(self, input, q_proj, k_proj, v_proj, stage_rows, &q_projection_stage) catch |err| {
        log.warn("inference", "CUDA attention qkv projection failed", .{
            .seq_len = seq_len_u32,
            .stage_rows = stage_rows,
            .q_dim = cfg.q_dim,
            .q_projection_dim = cfg.q_projection_dim,
            .kv_dim = cfg.kv_dim,
            .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
            .reason = @errorName(err),
        });
        return err;
    };
    if (debugKernelSyncEnabled()) {
        self.device.synchronize() catch |err| {
            log.warn("inference", "CUDA debug sync failed after decode qkv", .{
                .head_dim = head_dim_u32,
                .n_heads = n_heads_u32,
                .n_kv_heads = n_kv_heads_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    if (cfg.query_gate) {
        engine_ops.compactQueryGateProjection(
            self,
            stage_rows,
            cfg.q_dim,
            cfg.q_projection_dim,
            n_heads_u32,
            head_dim_u32,
            &q_projection_stage,
            &q_values_stage,
        ) catch |err| {
            log.warn("inference", "CUDA attention query-gate compact failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .q_dim = cfg.q_dim,
                .q_projection_dim = cfg.q_projection_dim,
                .reason = @errorName(err),
            });
            return err;
        };
        attn_q_stage = q_values_stage;
    }
    if (q_norm_weight) |q_norm_value| {
        const q_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_heads_u32) catch return error.InvalidArgument;
        compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_q_stage,
            &q_norm_value.buffer,
            &attn_q_stage,
            q_norm_rows,
            head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        ) catch |err| {
            log.warn("inference", "CUDA attention q_norm failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .n_heads = n_heads_u32,
                .head_dim = head_dim_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    if (debugKernelSyncEnabled()) {
        self.device.synchronize() catch |err| {
            log.warn("inference", "CUDA debug sync failed after decode q_norm", .{
                .head_dim = head_dim_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    if (k_norm_weight) |k_norm_value| {
        const k_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_kv_heads_u32) catch return error.InvalidArgument;
        compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_k_stage,
            &k_norm_value.buffer,
            &attn_k_stage,
            k_norm_rows,
            head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        ) catch |err| {
            log.warn("inference", "CUDA attention k_norm failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .n_kv_heads = n_kv_heads_u32,
                .head_dim = head_dim_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    if (debugKernelSyncEnabled()) {
        self.device.synchronize() catch |err| {
            log.warn("inference", "CUDA debug sync failed after decode qk_norm/v_norm", .{
                .head_dim = head_dim_u32,
                .n_heads = n_heads_u32,
                .n_kv_heads = n_kv_heads_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    try applyValueNormInPlace(self, &attn_v_stage, stage_rows, n_kv_heads_u32, head_dim_u32);
    const use_fused_attention_heads_f16_kv = (!cfg.query_gate) and attention_mod.useFusedHeadsF16Kv(
        attention_policy_config,
        seq_len_u32,
        cfg.sliding_window,
        cfg.is_causal,
        head_dim_u32,
        attention_kernels.attn_fused_heads_f16_kv_function != null,
    );
    if (!use_fused_attention_heads_f16_kv) {
        compute.cuda.rope.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            rope_function,
            &attn_q_stage,
            n_heads_u32,
            head_dim_u32,
            rope_dim_u32,
            position_u32,
            layer_rope_theta,
        ) catch |err| {
            log.warn("inference", "CUDA attention q rope failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .position = position_u32,
                .n_heads = n_heads_u32,
                .head_dim = head_dim_u32,
                .rope_dim = rope_dim_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    const use_k_write_fused = switch (self.kv_cache_dtype) {
        .f16 => kv_write_f16_function != null or rope_store_f16_function != null,
        .i8 => self.kv_write_i8_function != null,
        .fp8 => self.kv_write_fp8_function != null,
    };
    if (!use_k_write_fused) {
        compute.cuda.rope.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            rope_function,
            &attn_k_stage,
            n_kv_heads_u32,
            head_dim_u32,
            rope_dim_u32,
            position_u32,
            layer_rope_theta,
        ) catch |err| {
            log.warn("inference", "CUDA attention k rope failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .position = position_u32,
                .n_kv_heads = n_kv_heads_u32,
                .head_dim = head_dim_u32,
                .rope_dim = rope_dim_u32,
                .reason = @errorName(err),
            });
            return err;
        };
    }

    const kv_elem_bytes: usize = self.kv_cache_dtype.elementBytes();
    const kv_row_bytes = std.math.mul(usize, cfg.kv_dim, kv_elem_bytes) catch return error.InvalidArgument;
    const kv_row_offset = std.math.mul(usize, position, kv_row_bytes) catch return error.InvalidArgument;
    var k_row = bufferSlice(k_cache, kv_row_offset, kv_row_bytes) catch |err| {
        log.warn("inference", "CUDA attention k cache slice failed", .{
            .seq_len = seq_len_u32,
            .position = position_u32,
            .kv_dim = cfg.kv_dim,
            .row_offset = kv_row_offset,
            .row_bytes = kv_row_bytes,
            .cache_bytes = k_cache.size,
            .reason = @errorName(err),
        });
        return err;
    };
    var v_row = bufferSlice(v_cache, kv_row_offset, kv_row_bytes) catch |err| {
        log.warn("inference", "CUDA attention v cache slice failed", .{
            .seq_len = seq_len_u32,
            .position = position_u32,
            .kv_dim = cfg.kv_dim,
            .row_offset = kv_row_offset,
            .row_bytes = kv_row_bytes,
            .cache_bytes = v_cache.size,
            .reason = @errorName(err),
        });
        return err;
    };
    switch (self.kv_cache_dtype) {
        .f16 => {
            if (kv_write_f16_function) |kv_write_f16| {
                try compute.cuda.kv_write_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    kv_write_f16,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_row,
                    &v_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            } else if (rope_store_f16_function) |rope_store_f16| {
                try compute.cuda.rope_store_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_store_f16,
                    &attn_k_stage,
                    &k_row,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_v_stage,
                    &v_row,
                    @intCast(cfg.kv_dim),
                );
            } else {
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_k_stage,
                    &k_row,
                    @intCast(cfg.kv_dim),
                );
                try compute.cuda.cast_f32_to_f16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_v_stage,
                    &v_row,
                    @intCast(cfg.kv_dim),
                );
            }
        },
        .i8 => {
            const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
            const scale_offset = std.math.mul(usize, position, scale_row_bytes) catch return error.InvalidArgument;
            var k_scale_row = try bufferSlice(k_scale, scale_offset, scale_row_bytes);
            var v_scale_row = try bufferSlice(v_scale, scale_offset, scale_row_bytes);
            try compute.cuda.kv_write_i8.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.kv_write_i8_function orelse return error.CudaKernelUnavailable,
                &attn_k_stage,
                &attn_v_stage,
                &k_row,
                &v_row,
                &k_scale_row,
                &v_scale_row,
                n_kv_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                position_u32,
                layer_rope_theta,
            );
        },
        .fp8 => {
            const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
            const scale_offset = std.math.mul(usize, position, scale_row_bytes) catch return error.InvalidArgument;
            var k_scale_row = try bufferSlice(k_scale, scale_offset, scale_row_bytes);
            var v_scale_row = try bufferSlice(v_scale, scale_offset, scale_row_bytes);
            try compute.cuda.kv_write_fp8.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.kv_write_fp8_function orelse return error.CudaKernelUnavailable,
                &attn_k_stage,
                &attn_v_stage,
                &k_row,
                &v_row,
                &k_scale_row,
                &v_scale_row,
                n_kv_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                position_u32,
                layer_rope_theta,
            );
        },
    }

    const kv_groups_u32: u32 = n_heads_u32 / n_kv_heads_u32;
    const kv_dim_u32: u32 = @intCast(cfg.kv_dim);
    _ = self.runAttentionContext(
        cfg,
        &attn_q_stage,
        &attn_context_stage,
        read_k_cache,
        read_v_cache,
        read_k_scale,
        read_v_scale,
        attention_kernels,
        seq_len_u32,
        head_dim_u32,
        kv_dim_u32,
        kv_groups_u32,
        n_heads_u32,
        attention_scale,
        rope_dim_u32,
        position_u32,
        layer_rope_theta,
    ) catch |err| {
        log.warn("inference", "CUDA attention context failed", .{
            .seq_len = seq_len_u32,
            .stage_rows = stage_rows,
            .position = position_u32,
            .q_dim = cfg.q_dim,
            .kv_dim = cfg.kv_dim,
            .query_gate = @as(u8, @intFromBool(cfg.query_gate)),
            .reason = @errorName(err),
        });
        return err;
    };
    if (cfg.query_gate) {
        engine_ops.applyQueryGateToContextInPlace(
            self,
            stage_rows,
            cfg.q_dim,
            cfg.q_projection_dim,
            n_heads_u32,
            head_dim_u32,
        ) catch |err| {
            log.warn("inference", "CUDA attention query-gate output failed", .{
                .seq_len = seq_len_u32,
                .stage_rows = stage_rows,
                .q_dim = cfg.q_dim,
                .q_projection_dim = cfg.q_projection_dim,
                .reason = @errorName(err),
            });
            return err;
        };
    }
    if (residual_buf) |rb| {
        self.pending_residual_add_buf = rb;
    }
    engine_ops.linearForwardRows(self, &attn_context_stage, stage_rows, o_proj, output) catch |err| {
        log.warn("inference", "CUDA attention output projection failed", .{
            .seq_len = seq_len_u32,
            .stage_rows = stage_rows,
            .o_proj_in_dim = o_proj.rows(),
            .o_proj_out_dim = o_proj.cols(),
            .context_bytes = attn_context_stage.size,
            .output_bytes = output.size,
            .reason = @errorName(err),
        });
        return err;
    };
}

/// Batched decode attention: QKV GEMM for N rows, per-row RoPE + KV write + attention,
/// then O GEMM for N rows. Each row uses its own slot's KV cache.
pub fn runBatchedDecodeAttentionMixer(
    self: anytype,
    cfg: *const LayerAttentionExecConfig,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    o_proj: *const LinearWeight,
    q_norm_weight: ?*const DeviceTensor,
    k_norm_weight: ?*const DeviceTensor,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    ctx: anytype,
    batch: anytype,
) !void {
    const n_rows = ctx.active_rows_u32;
    const n: usize = @intCast(n_rows);
    const layer_rope_theta = effectiveLayerRopeTheta(
        self,
        cfg,
        ctx.head_dim_u32,
        ctx.rope_dim_u32,
        ctx.global_rope_theta,
        ctx.local_rope_theta,
    );
    const attention_scale = attentionScaleForHeadDim(self, ctx.head_dim_u32);
    const q_stage_bytes = n * cfg.q_projection_dim * @sizeOf(f32);
    const q_values_bytes = n * cfg.q_dim * @sizeOf(f32);
    const kv_stage_bytes = n * cfg.kv_dim * @sizeOf(f32);
    const context_stage_bytes = n * o_proj.rows() * @sizeOf(f32);
    var q_projection_stage = if (cfg.query_gate)
        try bufferSlice(&self.runtime_buffers.query_gate_proj_dev, 0, q_stage_bytes)
    else
        try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_stage_bytes);
    var attn_q_stage = q_projection_stage;
    var q_values_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_values_bytes);
    var attn_k_stage = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, kv_stage_bytes);
    var attn_v_stage = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, kv_stage_bytes);
    var attn_context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_stage_bytes);
    const current_attention_binding = self.block_runtime.blocks[ctx.layer_index].attention_binding orelse return error.InvalidStateDescriptorBinding;
    const shared_source_slot_kv_index = current_attention_binding.kv_shared_source_slot_kv_index;

    // Step 1: QKV projection for all N rows.
    _ = try engine_ops.runQkvProjection(self, input, q_proj, k_proj, v_proj, n, &q_projection_stage);
    if (cfg.query_gate) {
        try engine_ops.compactQueryGateProjection(self, n, cfg.q_dim, cfg.q_projection_dim, ctx.n_heads_u32, ctx.head_dim_u32, &q_projection_stage, &q_values_stage);
        attn_q_stage = q_values_stage;
    }

    // Step 2: Q/K norms for all N rows (position-independent).
    if (q_norm_weight) |q_norm_value| {
        const q_norm_rows = @as(u32, @intCast(n)) * ctx.n_heads_u32;
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_q_stage,
            &q_norm_value.buffer,
            &attn_q_stage,
            q_norm_rows,
            ctx.head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        );
    }
    if (k_norm_weight) |k_norm_value| {
        const k_norm_rows = @as(u32, @intCast(n)) * ctx.n_kv_heads_u32;
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_k_stage,
            &k_norm_value.buffer,
            &attn_k_stage,
            k_norm_rows,
            ctx.head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        );
    }
    if (current_attention_binding.use_v_norm) {
        try applyValueNormInPlace(self, &attn_v_stage, n, ctx.n_kv_heads_u32, ctx.head_dim_u32);
    }

    // Step 3: Per-row RoPE + KV write, then decode attention.
    const kv_elem_bytes: usize = self.kv_cache_dtype.elementBytes();
    const kv_row_bytes = cfg.kv_dim * kv_elem_bytes;
    const kv_groups: u32 = ctx.n_heads_u32 / ctx.n_kv_heads_u32;
    const kv_dim_u32: u32 = @intCast(cfg.kv_dim);
    const use_k_write_fused = switch (self.kv_cache_dtype) {
        .f16 => ctx.kv_write_f16_function != null or ctx.rope_store_f16_function != null,
        .i8 => self.kv_write_i8_function != null,
        .fp8 => self.kv_write_fp8_function != null,
    };
    const use_batched_kv_write = switch (self.kv_cache_dtype) {
        .f16 => (ctx.kv_write_f16_function != null) and (self.kv_write_f16_rows_ptrs_function != null),
        .i8 => self.kv_write_i8_rows_ptrs_function != null,
        .fp8 => self.kv_write_fp8_rows_ptrs_function != null,
    };
    const use_batched_fused_decode_attention = switch (self.kv_cache_dtype) {
        .f16 => self.attn_fused_decode_heads_f16_kv_ptrs_function != null,
        .i8 => self.attn_fused_decode_heads_i8_kv_ptrs_function != null,
        .fp8 => self.attn_fused_decode_heads_fp8_kv_ptrs_function != null,
    };
    const flash_decode_available = switch (self.kv_cache_dtype) {
        .f16 => self.flash_decode_f16_function != null,
        .i8 => self.flash_decode_i8_function != null,
        .fp8 => self.flash_decode_fp8_function != null,
    };
    // Flash decode is better when there is enough parallel work from
    // (n_kv_heads × batch_rows). For low-block launches, non-flash batched
    // decode kernels usually sustain better occupancy.
    const use_flash_decode = shouldUseFlashDecodePath(
        kv_groups,
        ctx.head_dim_u32,
        ctx.n_kv_heads_u32,
        n_rows,
        flash_decode_available,
    );
    const use_batched_separate_decode_attention = switch (self.kv_cache_dtype) {
        .f16 => (self.rope_rows_ptrs_function != null) and
            (self.attn_scores_heads_f16_kv_ptrs_function != null) and
            (self.softmax_rows_dynamic_cols_ptrs_function != null) and
            (self.attn_weighted_sum_heads_f16_kv_ptrs_function != null),
        .i8 => (self.rope_rows_ptrs_function != null) and
            (self.attn_scores_heads_i8_kv_ptrs_function != null) and
            (self.softmax_rows_dynamic_cols_ptrs_function != null) and
            (self.attn_weighted_sum_heads_i8_kv_ptrs_function != null),
        .fp8 => (self.rope_rows_ptrs_function != null) and
            (self.attn_scores_heads_fp8_kv_ptrs_function != null) and
            (self.softmax_rows_dynamic_cols_ptrs_function != null) and
            (self.attn_weighted_sum_heads_fp8_kv_ptrs_function != null),
    };
    const sliding_window_u32: u32 = if (cfg.is_causal and cfg.sliding_window > 0)
        (std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32))
    else
        0;
    const ptr_bytes = std.math.mul(usize, n, @sizeOf(u64)) catch return error.InvalidArgument;
    const idx_bytes = std.math.mul(usize, n, @sizeOf(u32)) catch return error.InvalidArgument;
    const attn_table_row_offset = std.math.mul(usize, batch.attn_layer_index, batch.attn_ptrs_row_stride) catch return error.InvalidArgument;
    const attn_table_byte_offset = std.math.mul(usize, attn_table_row_offset, @sizeOf(u64)) catch return error.InvalidArgument;
    var write_key_cache_ptrs_dev = try bufferSlice(batch.attn_key_cache_ptrs_table_dev, attn_table_byte_offset, ptr_bytes);
    var write_value_cache_ptrs_dev = try bufferSlice(batch.attn_value_cache_ptrs_table_dev, attn_table_byte_offset, ptr_bytes);
    var write_k_scale_ptrs_dev = try bufferSlice(batch.attn_k_scale_ptrs_table_dev, attn_table_byte_offset, ptr_bytes);
    var write_v_scale_ptrs_dev = try bufferSlice(batch.attn_v_scale_ptrs_table_dev, attn_table_byte_offset, ptr_bytes);
    var read_key_cache_ptrs_dev = write_key_cache_ptrs_dev;
    var read_value_cache_ptrs_dev = write_value_cache_ptrs_dev;
    var read_k_scale_ptrs_dev = write_k_scale_ptrs_dev;
    var read_v_scale_ptrs_dev = write_v_scale_ptrs_dev;
    var decode_positions_dev = try bufferSlice(&self.runtime_buffers.decode_positions_dev, 0, idx_bytes);
    var decode_seq_lens_dev = try bufferSlice(&self.runtime_buffers.decode_seq_lens_dev, 0, idx_bytes);

    if (shared_source_slot_kv_index) |read_slot_kv_index| {
        var read_key_ptrs_host = self.runtime_buffers.decode_key_cache_ptrs_host[0..n];
        var read_value_ptrs_host = self.runtime_buffers.decode_value_cache_ptrs_host[0..n];
        for (0..n) |row_i| {
            const slot_idx = batch.slot_indices[row_i];
            const read_entry = self.slot_kv_states[slot_idx].kv[read_slot_kv_index];
            read_key_ptrs_host[row_i] = read_entry.k.pointer;
            read_value_ptrs_host[row_i] = read_entry.v.pointer;
        }
        read_key_cache_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_key_cache_ptrs_dev, 0, ptr_bytes);
        read_value_cache_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_value_cache_ptrs_dev, 0, ptr_bytes);
        try read_key_cache_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(read_key_ptrs_host));
        try read_value_cache_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(read_value_ptrs_host));

        switch (self.kv_cache_dtype) {
            .f16 => {},
            .i8, .fp8 => {
                var read_k_scale_ptrs_host = self.runtime_buffers.decode_attn_k_scale_ptrs_table_host[0..n];
                var read_v_scale_ptrs_host = self.runtime_buffers.decode_attn_v_scale_ptrs_table_host[0..n];
                for (0..n) |row_i| {
                    const slot_idx = batch.slot_indices[row_i];
                    const read_entry = self.slot_kv_states[slot_idx].kv[read_slot_kv_index];
                    read_k_scale_ptrs_host[row_i] = read_entry.k_scale.pointer;
                    read_v_scale_ptrs_host[row_i] = read_entry.v_scale.pointer;
                }
                read_k_scale_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_attn_k_scale_ptrs_table_dev, 0, ptr_bytes);
                read_v_scale_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_attn_v_scale_ptrs_table_dev, 0, ptr_bytes);
                try read_k_scale_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(read_k_scale_ptrs_host));
                try read_v_scale_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(read_v_scale_ptrs_host));
            },
        }
    }

    // Flash decode preferred (GQA-aware, fewest KV reads), then batched
    // separate (graph-compatible), then fused fallback.
    const use_batched_decode_attention = use_flash_decode or use_batched_separate_decode_attention or use_batched_fused_decode_attention;
    const can_skip_row_decode_prep = shared_source_slot_kv_index == null and use_batched_kv_write and use_batched_decode_attention;
    if (!can_skip_row_decode_prep) {
        for (0..n) |row_i| {
            const slot_idx = batch.slot_indices[row_i];
            const position = batch.positions[row_i];
            const position_u32: u32 = @intCast(position);
            const seq_len_u32 = batch.seq_lens[row_i];

            // Get per-row Q, K, V slices.
            var q_row = try logicalF32RowSlice(&attn_q_stage, n, row_i, cfg.q_dim);
            var k_row_stage = try logicalF32RowSlice(&attn_k_stage, n, row_i, cfg.kv_dim);
            var v_row_stage = try logicalF32RowSlice(&attn_v_stage, n, row_i, cfg.kv_dim);
            const use_fused_attention_heads_f16_kv = use_flash_decode or use_batched_fused_decode_attention or
                ((!cfg.query_gate) and attention_mod.useFusedHeadsF16Kv(
                    attention_policy_config,
                    seq_len_u32,
                    cfg.sliding_window,
                    cfg.is_causal,
                    ctx.head_dim_u32,
                    ctx.attention_kernels.attn_fused_heads_f16_kv_function != null,
                ));

            // RoPE on Q (per-row position).
            if (!use_fused_attention_heads_f16_kv) {
                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    ctx.rope_function,
                    &q_row,
                    ctx.n_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            }

            if (use_batched_kv_write) continue;
            const kv_entry = self.slot_kv_states[slot_idx].kv[batch.attn_layer_index];

            // RoPE on K (per-row position).
            if (!use_k_write_fused) {
                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    ctx.rope_function,
                    &k_row_stage,
                    ctx.n_kv_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            }

            // KV cache write for this slot.
            const kv_offset = position * kv_row_bytes;
            var k_cache_row = try bufferSlice(&kv_entry.k, kv_offset, kv_row_bytes);
            var v_cache_row = try bufferSlice(&kv_entry.v, kv_offset, kv_row_bytes);
            switch (self.kv_cache_dtype) {
                .f16 => {
                    if (ctx.kv_write_f16_function) |kv_write_f16| {
                        try compute.cuda.kv_write_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kv_write_f16,
                            &k_row_stage,
                            &v_row_stage,
                            &k_cache_row,
                            &v_cache_row,
                            ctx.n_kv_heads_u32,
                            ctx.head_dim_u32,
                            ctx.rope_dim_u32,
                            position_u32,
                            layer_rope_theta,
                        );
                    } else if (ctx.rope_store_f16_function) |rope_store_f16| {
                        try compute.cuda.rope_store_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            rope_store_f16,
                            &k_row_stage,
                            &k_cache_row,
                            ctx.n_kv_heads_u32,
                            ctx.head_dim_u32,
                            ctx.rope_dim_u32,
                            position_u32,
                            layer_rope_theta,
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            ctx.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &v_row_stage,
                            &v_cache_row,
                            @intCast(cfg.kv_dim),
                        );
                    } else {
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            ctx.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &k_row_stage,
                            &k_cache_row,
                            @intCast(cfg.kv_dim),
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            ctx.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &v_row_stage,
                            &v_cache_row,
                            @intCast(cfg.kv_dim),
                        );
                    }
                },
                .i8 => {
                    const scale_row_bytes: usize = @as(usize, ctx.n_kv_heads_u32) * @sizeOf(f32);
                    const scale_off = std.math.mul(usize, position, scale_row_bytes) catch return error.InvalidArgument;
                    var k_sc = try bufferSlice(&kv_entry.k_scale, scale_off, scale_row_bytes);
                    var v_sc = try bufferSlice(&kv_entry.v_scale, scale_off, scale_row_bytes);
                    try compute.cuda.kv_write_i8.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.kv_write_i8_function orelse return error.CudaKernelUnavailable,
                        &k_row_stage,
                        &v_row_stage,
                        &k_cache_row,
                        &v_cache_row,
                        &k_sc,
                        &v_sc,
                        ctx.n_kv_heads_u32,
                        ctx.head_dim_u32,
                        ctx.rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                },
                .fp8 => {
                    const scale_row_bytes: usize = @as(usize, ctx.n_kv_heads_u32) * @sizeOf(f32);
                    const scale_off = std.math.mul(usize, position, scale_row_bytes) catch return error.InvalidArgument;
                    var k_sc = try bufferSlice(&kv_entry.k_scale, scale_off, scale_row_bytes);
                    var v_sc = try bufferSlice(&kv_entry.v_scale, scale_off, scale_row_bytes);
                    try compute.cuda.kv_write_fp8.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.kv_write_fp8_function orelse return error.CudaKernelUnavailable,
                        &k_row_stage,
                        &v_row_stage,
                        &k_cache_row,
                        &v_cache_row,
                        &k_sc,
                        &v_sc,
                        ctx.n_kv_heads_u32,
                        ctx.head_dim_u32,
                        ctx.rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                },
            }
        }
    }

    if (use_batched_kv_write) {
        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.kv_write_f16_rows_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_f16_rows_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_k_stage,
                    &attn_v_stage,
                    &write_key_cache_ptrs_dev,
                    &write_value_cache_ptrs_dev,
                    &decode_positions_dev,
                    ctx.n_kv_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    n_rows,
                    kv_dim_u32,
                    layer_rope_theta,
                );
            },
            .i8 => {
                try compute.cuda.kv_write_i8_rows_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_i8_rows_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_k_stage,
                    &attn_v_stage,
                    &write_key_cache_ptrs_dev,
                    &write_value_cache_ptrs_dev,
                    &write_k_scale_ptrs_dev,
                    &write_v_scale_ptrs_dev,
                    &decode_positions_dev,
                    ctx.n_kv_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    n_rows,
                    kv_dim_u32,
                    layer_rope_theta,
                );
            },
            .fp8 => {
                try compute.cuda.kv_write_fp8_rows_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_fp8_rows_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_k_stage,
                    &attn_v_stage,
                    &write_key_cache_ptrs_dev,
                    &write_value_cache_ptrs_dev,
                    &write_k_scale_ptrs_dev,
                    &write_v_scale_ptrs_dev,
                    &decode_positions_dev,
                    ctx.n_kv_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    n_rows,
                    kv_dim_u32,
                    layer_rope_theta,
                );
            },
        }
    }

    if (use_flash_decode) {
        // Flash decode with split-K: partition sequence across blocks for
        // occupancy when n_kv_heads is small. GQA-aware, does RoPE on Q
        // internally, reads KV once per KV head for all grouped Q heads.
        const gate_proj: ?*const compute.cuda.Buffer = if (cfg.query_gate) &q_projection_stage else null;
        const gate_proj_stride: u32 = if (cfg.query_gate) @intCast(cfg.q_projection_dim) else 0;
        const n_seq_chunks = compute.cuda.flash_decode.computeSeqChunks(ctx.n_kv_heads_u32, n_rows);

        // Lazy-allocate partial buffers for split-K when needed.
        var partial_m_buf: compute.cuda.Buffer = undefined;
        var partial_s_buf: compute.cuda.Buffer = undefined;
        var partial_out_buf: compute.cuda.Buffer = undefined;
        var partial_m_ptr: ?*compute.cuda.Buffer = null;
        var partial_s_ptr: ?*compute.cuda.Buffer = null;
        var partial_out_ptr: ?*compute.cuda.Buffer = null;
        if (n_seq_chunks > 1) {
            const entries = @as(usize, n_rows) * @as(usize, ctx.n_heads_u32) * @as(usize, n_seq_chunks);
            const ms_bytes = entries * @sizeOf(f32);
            const out_bytes = entries * @as(usize, ctx.head_dim_u32) * @sizeOf(f32);
            const total_bytes = ms_bytes + ms_bytes + out_bytes;
            if (self.flash_decode_partial_dev == null or self.flash_decode_partial_dev.?.size < total_bytes) {
                if (self.flash_decode_partial_dev) |*buf| buf.deinit(&self.device);
                self.flash_decode_partial_dev = try self.device.allocBuffer(total_bytes);
            }
            const base_ptr = self.flash_decode_partial_dev.?.pointer;
            partial_m_buf = .{ .pointer = base_ptr, .size = ms_bytes };
            partial_s_buf = .{ .pointer = base_ptr + ms_bytes, .size = ms_bytes };
            partial_out_buf = .{ .pointer = base_ptr + 2 * ms_bytes, .size = out_bytes };
            partial_m_ptr = &partial_m_buf;
            partial_s_ptr = &partial_s_buf;
            partial_out_ptr = &partial_out_buf;
        }

        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.flash_decode.runNoScales(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.flash_decode_f16_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_value_cache_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    ctx.rope_dim_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    gate_proj,
                    gate_proj_stride,
                    n_seq_chunks,
                    partial_m_ptr,
                    partial_s_ptr,
                    partial_out_ptr,
                );
            },
            .i8 => {
                try compute.cuda.flash_decode.runWithScales(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.flash_decode_i8_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_value_cache_ptrs_dev,
                    &read_k_scale_ptrs_dev,
                    &read_v_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    ctx.rope_dim_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    gate_proj,
                    gate_proj_stride,
                    n_seq_chunks,
                    partial_m_ptr,
                    partial_s_ptr,
                    partial_out_ptr,
                );
            },
            .fp8 => {
                try compute.cuda.flash_decode.runWithScales(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.flash_decode_fp8_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_value_cache_ptrs_dev,
                    &read_k_scale_ptrs_dev,
                    &read_v_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    ctx.rope_dim_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    gate_proj,
                    gate_proj_stride,
                    n_seq_chunks,
                    partial_m_ptr,
                    partial_s_ptr,
                    partial_out_ptr,
                );
            },
        }

        // Launch reduce kernel to merge split-K partials.
        if (n_seq_chunks > 1) {
            try compute.cuda.flash_decode.runReduce(
                &self.kernel_arg_pack,
                &self.device,
                self.flash_decode_reduce_function orelse return error.CudaKernelUnavailable,
                &attn_context_stage,
                &partial_m_buf,
                &partial_s_buf,
                &partial_out_buf,
                n_rows,
                ctx.n_heads_u32,
                ctx.head_dim_u32,
                n_seq_chunks,
                gate_proj,
                gate_proj_stride,
            );
        }
    } else if (use_batched_separate_decode_attention) {
        // Batched separate attention: same high-occupancy kernel design as
        // per-row path, but reads position/seq_len from device buffers and
        // KV cache from pointer tables — fully graph-compatible.
        const max_seq_len = self.runtime_buffers.batched_attn_max_seq_len;

        // Q RoPE for all rows (reads positions from device buffer).
        try compute.cuda.rope_rows_ptrs.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rope_rows_ptrs_function orelse return error.CudaKernelUnavailable,
            &attn_q_stage,
            &decode_positions_dev,
            n_rows,
            ctx.n_heads_u32,
            ctx.head_dim_u32,
            ctx.rope_dim_u32,
            layer_rope_theta,
        );

        // Scores: Q*K dot products with pointer-table KV cache.
        var scores_dev = self.runtime_buffers.batched_attn_scores_dev;
        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.attn_scores_heads_f16_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_scores_heads_f16_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &scores_dev,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    max_seq_len,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    sliding_window_u32,
                );
            },
            .i8 => {
                try compute.cuda.attn_scores_heads_i8_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_scores_heads_i8_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &scores_dev,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_k_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    max_seq_len,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    sliding_window_u32,
                );
            },
            .fp8 => {
                try compute.cuda.attn_scores_heads_fp8_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_scores_heads_fp8_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &scores_dev,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_k_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    max_seq_len,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    sliding_window_u32,
                );
            },
        }

        // Softmax with dynamic per-row column counts.
        try compute.cuda.softmax_rows_dynamic_cols_ptrs.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.softmax_rows_dynamic_cols_ptrs_function orelse return error.CudaKernelUnavailable,
            &scores_dev,
            &decode_seq_lens_dev,
            &decode_positions_dev,
            n_rows,
            ctx.n_heads_u32,
            max_seq_len,
            sliding_window_u32,
        );

        // Weighted sum: probs * V with pointer-table KV cache.
        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.attn_weighted_sum_heads_f16_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_weighted_sum_heads_f16_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &scores_dev,
                    &read_value_cache_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    max_seq_len,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    sliding_window_u32,
                );
            },
            .i8 => {
                try compute.cuda.attn_weighted_sum_heads_i8_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_weighted_sum_heads_i8_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &scores_dev,
                    &read_value_cache_ptrs_dev,
                    &read_v_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    max_seq_len,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    sliding_window_u32,
                );
            },
            .fp8 => {
                try compute.cuda.attn_weighted_sum_heads_fp8_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_weighted_sum_heads_fp8_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &scores_dev,
                    &read_value_cache_ptrs_dev,
                    &read_v_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    max_seq_len,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    sliding_window_u32,
                );
            },
        }
    } else if (use_batched_fused_decode_attention) {
        // Fused path fallback: single kernel does RoPE + scores + softmax +
        // weighted_sum + optional gate fusion.
        const gate_proj: ?*const compute.cuda.Buffer = if (cfg.query_gate) &q_projection_stage else null;
        const gate_proj_stride: u32 = if (cfg.query_gate) @intCast(cfg.q_projection_dim) else 0;

        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.attn_fused_decode_heads_f16_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_fused_decode_heads_f16_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_value_cache_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    ctx.rope_dim_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    gate_proj,
                    gate_proj_stride,
                );
            },
            .i8 => {
                try compute.cuda.attn_fused_decode_heads_i8_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_fused_decode_heads_i8_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_value_cache_ptrs_dev,
                    &read_k_scale_ptrs_dev,
                    &read_v_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    ctx.rope_dim_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    gate_proj,
                    gate_proj_stride,
                );
            },
            .fp8 => {
                try compute.cuda.attn_fused_decode_heads_fp8_kv_ptrs.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.attn_fused_decode_heads_fp8_kv_ptrs_function orelse return error.CudaKernelUnavailable,
                    &attn_context_stage,
                    &attn_q_stage,
                    &read_key_cache_ptrs_dev,
                    &read_value_cache_ptrs_dev,
                    &read_k_scale_ptrs_dev,
                    &read_v_scale_ptrs_dev,
                    &decode_seq_lens_dev,
                    &decode_positions_dev,
                    n_rows,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.head_dim_u32,
                    attention_scale,
                    ctx.rope_dim_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    gate_proj,
                    gate_proj_stride,
                );
            },
        }
    } else {
        for (0..n) |row_i| {
            const slot_idx = batch.slot_indices[row_i];
            const position_u32: u32 = @intCast(batch.positions[row_i]);
            const seq_len_u32 = batch.seq_lens[row_i];
            const kv_entry = self.slot_kv_states[slot_idx].kv[batch.attn_layer_index];
            const read_entry = if (shared_source_slot_kv_index) |read_slot_kv_index|
                self.slot_kv_states[slot_idx].kv[read_slot_kv_index]
            else
                kv_entry;
            var q_row = try logicalF32RowSlice(&attn_q_stage, n, row_i, cfg.q_dim);
            var context_row = try logicalF32RowSlice(&attn_context_stage, n, row_i, o_proj.rows());
            _ = try self.runAttentionContext(
                cfg,
                &q_row,
                &context_row,
                &read_entry.k,
                &read_entry.v,
                &read_entry.k_scale,
                &read_entry.v_scale,
                ctx.attention_kernels,
                seq_len_u32,
                ctx.head_dim_u32,
                kv_dim_u32,
                kv_groups,
                ctx.n_heads_u32,
                attention_scale,
                ctx.rope_dim_u32,
                position_u32,
                layer_rope_theta,
            );
        }
    }

    // Step 4: Query gate — the fused path handles gate internally; batched
    // separate and per-row paths need a separate gate kernel.
    const fused_path_active = use_flash_decode or (use_batched_fused_decode_attention and !use_batched_separate_decode_attention);
    if (cfg.query_gate and !fused_path_active) {
        try engine_ops.applyQueryGateToContextInPlace(self, n, cfg.q_dim, cfg.q_projection_dim, ctx.n_heads_u32, ctx.head_dim_u32);
    }

    // Step 5: O projection GEMM for all N rows.
    try engine_ops.linearForwardRows(self, &attn_context_stage, n, o_proj, output);
}

pub fn runShortConvMixerStep(
    self: anytype,
    cfg: *const ShortConvExecConfig,
    conv_state: *compute.cuda.Buffer,
    in_proj: *const LinearWeight,
    out_proj: *const LinearWeight,
    conv_weight_time_major: *const DeviceTensor,
    conv_bias: ?*const DeviceTensor,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    shortconv_step_function: compute.cuda.Function,
) !void {
    const rows = try bufferF32RowCount(input, in_proj.rows());
    try engine_ops.linearForward(self, input, in_proj, &self.runtime_buffers.shortconv_proj_dev);
    const conv_bytes = std.math.mul(usize, cfg.conv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    var b_gate = try bufferSlice(&self.runtime_buffers.shortconv_proj_dev, 0, conv_bytes);
    var c_gate = try bufferSlice(&self.runtime_buffers.shortconv_proj_dev, conv_bytes, conv_bytes);
    var x_proj = try bufferSlice(&self.runtime_buffers.shortconv_proj_dev, conv_bytes * 2, conv_bytes);

    try compute.cuda.shortconv_step.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        shortconv_step_function,
        &b_gate,
        &c_gate,
        &x_proj,
        conv_state,
        &conv_weight_time_major.buffer,
        if (conv_bias) |w| &w.buffer else null,
        &self.runtime_buffers.shortconv_conv_dev,
        @intCast(cfg.conv_dim),
        @intCast(cfg.d_conv),
    );
    try engine_ops.linearForwardRows(self, &self.runtime_buffers.shortconv_conv_dev, rows, out_proj, output);
}

pub fn downloadRowsF32StrideAware(
    self: anytype,
    src: *const compute.cuda.Buffer,
    rows: usize,
    row_width: usize,
    dst: []f32,
) !void {
    if (rows == 0 or row_width == 0) return error.InvalidArgument;
    const logical_elements = std.math.mul(usize, rows, row_width) catch return error.InvalidArgument;
    if (dst.len < logical_elements) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, row_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, logical_elements, @sizeOf(f32)) catch return error.InvalidArgument;
    if (src.size < packed_bytes) return error.InvalidInstructionBinding;

    if (rows == 1 or src.size == packed_bytes) {
        return src.download(&self.device, std.mem.sliceAsBytes(dst[0..logical_elements]));
    }

    if (src.size % rows != 0) return error.InvalidInstructionBinding;
    const src_row_stride = src.size / rows;
    if (src_row_stride < row_bytes) return error.InvalidInstructionBinding;

    var row_idx: usize = 0;
    while (row_idx < rows) : (row_idx += 1) {
        const src_offset = std.math.mul(usize, row_idx, src_row_stride) catch return error.InvalidArgument;
        var src_row = try bufferSlice(src, src_offset, row_bytes);
        const dst_start = std.math.mul(usize, row_idx, row_width) catch return error.InvalidArgument;
        const dst_row = dst[dst_start .. dst_start + row_width];
        try src_row.download(&self.device, std.mem.sliceAsBytes(dst_row));
    }
}

pub fn uploadRowsF32StrideAware(
    self: anytype,
    src: []const f32,
    rows: usize,
    row_width: usize,
    dst: *compute.cuda.Buffer,
) !void {
    if (rows == 0 or row_width == 0) return error.InvalidArgument;
    const logical_elements = std.math.mul(usize, rows, row_width) catch return error.InvalidArgument;
    if (src.len < logical_elements) return error.InvalidArgument;
    const row_bytes = std.math.mul(usize, row_width, @sizeOf(f32)) catch return error.InvalidArgument;
    const packed_bytes = std.math.mul(usize, logical_elements, @sizeOf(f32)) catch return error.InvalidArgument;
    if (dst.size < packed_bytes) return error.InvalidInstructionBinding;

    if (rows == 1 or dst.size == packed_bytes) {
        return dst.upload(&self.device, std.mem.sliceAsBytes(src[0..logical_elements]));
    }

    if (dst.size % rows != 0) return error.InvalidInstructionBinding;
    const dst_row_stride = dst.size / rows;
    if (dst_row_stride < row_bytes) return error.InvalidInstructionBinding;

    var row_idx: usize = 0;
    while (row_idx < rows) : (row_idx += 1) {
        const dst_offset = std.math.mul(usize, row_idx, dst_row_stride) catch return error.InvalidArgument;
        var dst_row = try bufferSlice(dst, dst_offset, row_bytes);
        const src_start = std.math.mul(usize, row_idx, row_width) catch return error.InvalidArgument;
        const src_row = src[src_start .. src_start + row_width];
        try dst_row.upload(&self.device, std.mem.sliceAsBytes(src_row));
    }
}

pub fn runGatedDeltaMixerStep(
    self: anytype,
    block: *GatedDeltaBlockRuntime,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    seq_len: usize,
) !void {
    const d_model = self.d_model;
    const cfg = block.kernel.config;
    const d_inner: usize = @as(usize, cfg.n_heads) * @as(usize, cfg.d_head);
    const d_conv: usize = cfg.d_conv;
    const n_v_heads: usize = cfg.n_heads;
    const d_head: usize = cfg.d_head;
    const proj_len = block.in_proj.cols();
    const qkv_len = blk: {
        const values = block.kernel.weights.conv1d_weight.asSlice(f32);
        if (values.len == 0 or d_conv == 0 or (values.len % d_conv) != 0) return error.InvalidShape;
        break :blk values.len / d_conv;
    };
    const minimum_proj = d_inner + (2 * n_v_heads);
    if (proj_len <= minimum_proj) return error.InvalidShape;
    if (block.state.conv_state.len < qkv_len * d_conv) return error.InvalidShape;

    const proj_element_count = std.math.mul(usize, seq_len, proj_len) catch return error.InvalidArgument;
    const proj_bytes = std.math.mul(usize, proj_element_count, @sizeOf(f32)) catch return error.InvalidArgument;
    const ssm_element_count = std.math.mul(usize, seq_len, d_inner) catch return error.InvalidArgument;
    const ssm_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
    const norm_stage_bytes = std.math.mul(usize, ssm_element_count, @sizeOf(f32)) catch return error.InvalidArgument;

    var proj_dev = try bufferSlice(&self.runtime_buffers.gdelta_proj_dev, 0, proj_bytes);
    var norm_stage_dev = try bufferSlice(&self.runtime_buffers.gdelta_ssm_dev, 0, norm_stage_bytes);
    engine_ops.linearForwardRows(self, input, seq_len, &block.in_proj, &proj_dev) catch |err| {
        const in_proj_kind = switch (block.in_proj) {
            .dense_f32 => "dense_f32",
            .dense_u16 => "dense_u16",
            .gaffine_u4 => "gaffine_u4",
            .gaffine_u8 => "gaffine_u8",
            .fp8 => "fp8",
            .mxfp8 => "mxfp8",
            .nvfp4 => "nvfp4",
        };
        log.warn("inference", "CUDA gated-delta in_proj failed", .{
            .seq_len = seq_len,
            .kind = in_proj_kind,
            .reason = @errorName(err),
        });
        return err;
    };
    const prev_trace_position_offset = block.kernel.trace_position_offset;
    block.kernel.trace_position_offset = if (self.parity_prefill_seq_len > 1 and seq_len == 1)
        self.parity_prefill_token_index
    else
        0;
    defer block.kernel.trace_position_offset = prev_trace_position_offset;

    const trace_pos_offset = block.kernel.trace_position_offset;
    const trace_enabled = trace.isEnabled();
    if (trace_enabled) {
        try engine_forward.ensureGatedDeltaHostStageCapacity(self, @max(proj_element_count, ssm_element_count));
        const proj_host = self.gated_delta_stage_input_host[0..proj_element_count];
        try self.device.synchronize();
        downloadRowsF32StrideAware(self, &proj_dev, seq_len, proj_len, proj_host) catch |err| {
            log.warn("inference", "CUDA gated-delta projection download failed", .{
                .seq_len = seq_len,
                .d_model = d_model,
                .proj_len = proj_len,
                .proj_bytes = proj_dev.size,
                .stage_bytes = proj_element_count * @sizeOf(f32),
                .reason = @errorName(err),
            });
            return err;
        };
        const prev_backend = trace.setBackendContext(.cuda);
        defer _ = trace.setBackendContext(prev_backend);
        for (0..seq_len) |t| {
            const proj_row = proj_host[t * proj_len ..][0..proj_len];
            trace.emit(
                .gdelta_in_proj,
                block.kernel.layer_idx,
                0,
                @intCast(trace_pos_offset + t),
                @ptrCast(proj_row.ptr),
                .f32,
                .{ 1, 1, @intCast(proj_len), 0 },
                3,
                "cuda_gdelta_in_proj_host",
            );
        }
    }

    if (qkv_len <= d_inner) return error.InvalidShape;
    const qk_total = qkv_len - d_inner;
    if ((qk_total % 2) != 0) return error.InvalidShape;
    const qk_inner = qk_total / 2;
    if ((qk_inner % d_head) != 0) return error.InvalidShape;
    const n_qk_heads = qk_inner / d_head;
    if (n_qk_heads == 0 or (n_v_heads % n_qk_heads) != 0) return error.InvalidShape;

    const qkv_bytes = std.math.mul(usize, qkv_len, @sizeOf(f32)) catch return error.InvalidArgument;
    const z_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
    const beta_bytes = std.math.mul(usize, n_v_heads, @sizeOf(f32)) catch return error.InvalidArgument;
    const a_bytes = beta_bytes;
    const beta_offset_elems = qkv_len + d_inner;
    const a_offset_elems = beta_offset_elems + n_v_heads;
    const quantized_ssm_state = block.ssm_state_format == .i8_per_column_scale;
    const use_rows_conv_silu = if (quantized_ssm_state)
        self.gated_delta_conv_silu_rows_function != null
    else
        seq_len > 1 and self.gated_delta_conv_silu_rows_function != null;
    const use_token_conv_silu = !trace_enabled and !quantized_ssm_state and self.gated_delta_conv_silu_function != null;
    const use_rows_ssm = if (quantized_ssm_state)
        self.gated_delta_ssm_rows_i8_function != null
    else
        seq_len > 1 and self.gated_delta_ssm_rows_function != null;
    if (quantized_ssm_state and !use_rows_conv_silu) return error.CudaKernelUnavailable;
    if (quantized_ssm_state and !use_rows_ssm) return error.CudaKernelUnavailable;
    const head_bytes = std.math.mul(usize, d_head, @sizeOf(f32)) catch return error.InvalidArgument;
    const fused_norm_gate_supported = self.gated_delta_rmsnorm_silu_mul_function != null and
        ((block.norm_weight.buffer.size == head_bytes) or (block.norm_weight.buffer.size == ssm_bytes));
    const rows_norm_weight_stride: u32 = if (block.norm_weight.buffer.size == head_bytes)
        0
    else
        @intCast(d_head);
    const use_rows_norm_gate = use_rows_ssm and
        fused_norm_gate_supported and
        self.gated_delta_rmsnorm_silu_mul_rows_function != null;

    if (use_rows_conv_silu) {
        var qkv_all_dev = try bufferSlice(&proj_dev, 0, proj_bytes);
        try compute.cuda.gated_delta_conv_silu_rows.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.gated_delta_conv_silu_rows_function.?,
            &qkv_all_dev,
            &block.conv_state_dev,
            &block.conv_weight_time_major.buffer,
            if (block.conv_bias) |*bias| &bias.buffer else null,
            &qkv_all_dev,
            @intCast(qkv_len),
            @intCast(d_conv),
            block.conv_ring_head,
            @intCast(seq_len),
            @intCast(proj_len),
        );
        block.conv_ring_head = @intCast((@as(usize, block.conv_ring_head) + seq_len) % d_conv);
    }
    if (use_rows_ssm) {
        var qkv_all_dev = try bufferSlice(&proj_dev, 0, proj_bytes);
        if (quantized_ssm_state) {
            try compute.cuda.gated_delta_ssm_rows_i8.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.gated_delta_ssm_rows_i8_function.?,
                &qkv_all_dev,
                &block.a_log.buffer,
                if (block.dt_bias) |*bias| &bias.buffer else null,
                &block.ssm_state_dev,
                &norm_stage_dev,
                @intCast(n_qk_heads),
                @intCast(n_v_heads),
                @intCast(d_head),
                @intCast(seq_len),
                @intCast(proj_len),
                @intCast(beta_offset_elems),
                @intCast(a_offset_elems),
                @intCast(d_inner),
                block.ssm_state_scales_offset,
            );
        } else {
            try compute.cuda.gated_delta_ssm_rows.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.gated_delta_ssm_rows_function.?,
                &qkv_all_dev,
                &block.a_log.buffer,
                if (block.dt_bias) |*bias| &bias.buffer else null,
                &block.ssm_state_dev,
                &norm_stage_dev,
                @intCast(n_qk_heads),
                @intCast(n_v_heads),
                @intCast(d_head),
                @intCast(seq_len),
                @intCast(proj_len),
                @intCast(beta_offset_elems),
                @intCast(a_offset_elems),
                @intCast(d_inner),
            );
        }
    }
    if (use_rows_norm_gate) {
        const rows_total = std.math.mul(usize, seq_len, n_v_heads) catch return error.InvalidArgument;
        try compute.cuda.gated_delta_rmsnorm_silu_mul_rows.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.gated_delta_rmsnorm_silu_mul_rows_function.?,
            &norm_stage_dev,
            &proj_dev,
            &block.norm_weight.buffer,
            &norm_stage_dev,
            @intCast(rows_total),
            @intCast(d_head),
            @intCast(n_v_heads),
            @intCast(proj_len),
            @intCast(qkv_len),
            1.0e-6,
            rows_norm_weight_stride,
        );
    }
    const use_fully_batched_gdelta = !trace_enabled and use_rows_conv_silu and use_rows_ssm and use_rows_norm_gate;
    if (use_fully_batched_gdelta) {
        try engine_ops.linearForwardRows(self, &norm_stage_dev, seq_len, &block.out_proj, output);
        return;
    }

    for (0..seq_len) |t| {
        const ssm_host_row = block.scratch.getSsmOutput(d_inner);

        var proj_row_dev = try logicalF32RowSlice(&proj_dev, seq_len, t, proj_len);

        var qkv_dev = try bufferSlice(&proj_row_dev, 0, qkv_bytes);
        if (!use_rows_conv_silu) {
            if (use_token_conv_silu) {
                try compute.cuda.gated_delta_conv_silu.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.gated_delta_conv_silu_function.?,
                    &qkv_dev,
                    &block.conv_state_dev,
                    &block.conv_weight_time_major.buffer,
                    if (block.conv_bias) |*bias| &bias.buffer else null,
                    &qkv_dev,
                    @intCast(qkv_len),
                    @intCast(d_conv),
                    block.conv_ring_head,
                );
            } else {
                try compute.cuda.gated_delta_conv.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.gated_delta_conv_function orelse return error.CudaKernelUnavailable,
                    &qkv_dev,
                    &block.conv_state_dev,
                    &block.conv_weight_time_major.buffer,
                    if (block.conv_bias) |*bias| &bias.buffer else null,
                    &qkv_dev,
                    @intCast(qkv_len),
                    @intCast(d_conv),
                    block.conv_ring_head,
                );
            }
            block.conv_ring_head = if (block.conv_ring_head + 1 >= @as(u32, @intCast(d_conv)))
                0
            else
                block.conv_ring_head + 1;
        }
        if (trace_enabled) {
            const conv_host = self.gated_delta_stage_input_host[0..qkv_len];
            try qkv_dev.download(&self.device, std.mem.sliceAsBytes(conv_host));
            const prev_backend = trace.setBackendContext(.cuda);
            defer _ = trace.setBackendContext(prev_backend);
            trace.emit(
                .gdelta_conv,
                block.kernel.layer_idx,
                0,
                @intCast(trace_pos_offset + t),
                @ptrCast(conv_host.ptr),
                .f32,
                .{ 1, 1, @intCast(qkv_len), 0 },
                3,
                null,
            );
        }
        if (!use_rows_conv_silu and !use_token_conv_silu) {
            try compute.cuda.silu.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.silu_function orelse return error.CudaKernelUnavailable,
                &qkv_dev,
                &qkv_dev,
                @intCast(qkv_len),
            );
        }
        var z_dev = try bufferSlice(&proj_row_dev, qkv_bytes, z_bytes);
        var beta_dev = try bufferSlice(&proj_row_dev, qkv_bytes + z_bytes, beta_bytes);
        var a_dev = try bufferSlice(&proj_row_dev, qkv_bytes + z_bytes + beta_bytes, a_bytes);
        const norm_stage_offset = std.math.mul(usize, t, ssm_bytes) catch return error.InvalidArgument;
        var norm_dev = try bufferSlice(&norm_stage_dev, norm_stage_offset, ssm_bytes);
        if (!use_rows_ssm) {
            try compute.cuda.gated_delta_ssm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.gated_delta_ssm_function orelse return error.CudaKernelUnavailable,
                &qkv_dev,
                &beta_dev,
                &a_dev,
                &block.a_log.buffer,
                if (block.dt_bias) |*bias| &bias.buffer else null,
                &block.ssm_state_dev,
                &norm_dev,
                @intCast(n_qk_heads),
                @intCast(n_v_heads),
                @intCast(d_head),
            );
        }
        if (trace_enabled) {
            try norm_dev.download(&self.device, std.mem.sliceAsBytes(ssm_host_row));
            const prev_gpu_backend = trace.setBackendContext(.cuda);
            defer _ = trace.setBackendContext(prev_gpu_backend);
            trace.emit(
                .gdelta_ssm,
                block.kernel.layer_idx,
                0,
                @intCast(trace_pos_offset + t),
                @ptrCast(ssm_host_row.ptr),
                .f32,
                .{ 1, 1, @intCast(d_inner), 0 },
                3,
                null,
            );
        }

        if (fused_norm_gate_supported and !use_rows_norm_gate) {
            try compute.cuda.gated_delta_rmsnorm_silu_mul.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.gated_delta_rmsnorm_silu_mul_function.?,
                &norm_dev,
                &z_dev,
                &block.norm_weight.buffer,
                &norm_dev,
                @intCast(n_v_heads),
                @intCast(d_head),
                1.0e-6,
                rows_norm_weight_stride,
            );
        } else if (!fused_norm_gate_supported) {
            if (block.norm_weight.buffer.size == head_bytes) {
                try compute.cuda.rmsnorm.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                    &norm_dev,
                    &block.norm_weight.buffer,
                    &norm_dev,
                    @intCast(n_v_heads),
                    @intCast(d_head),
                    1.0e-6,
                    0.0,
                );
            } else if (block.norm_weight.buffer.size == ssm_bytes) {
                for (0..n_v_heads) |head_idx| {
                    const head_offset_bytes = std.math.mul(usize, head_idx * d_head, @sizeOf(f32)) catch return error.InvalidArgument;
                    var norm_head_dev = try bufferSlice(&norm_dev, head_offset_bytes, head_bytes);
                    var weight_head_dev = try bufferSlice(&block.norm_weight.buffer, head_offset_bytes, head_bytes);
                    try compute.cuda.rmsnorm.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                        &norm_head_dev,
                        &weight_head_dev,
                        &norm_head_dev,
                        1,
                        @intCast(d_head),
                        1.0e-6,
                        0.0,
                    );
                }
            } else {
                return error.InvalidShape;
            }
            try compute.cuda.silu_mul.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.silu_mul_function orelse return error.CudaKernelUnavailable,
                &z_dev,
                &norm_dev,
                &norm_dev,
                @intCast(d_inner),
            );
        }
        if (trace_enabled) {
            try norm_dev.download(&self.device, std.mem.sliceAsBytes(ssm_host_row));
            const prev_gpu_backend = trace.setBackendContext(.cuda);
            defer _ = trace.setBackendContext(prev_gpu_backend);
            trace.emit(
                .gdelta_norm,
                block.kernel.layer_idx,
                0,
                @intCast(trace_pos_offset + t),
                @ptrCast(ssm_host_row.ptr),
                .f32,
                .{ 1, 1, @intCast(d_inner), 0 },
                3,
                null,
            );
        }
    }
    try engine_ops.linearForwardRows(self, &norm_stage_dev, seq_len, &block.out_proj, output);
    if (trace_enabled) {
        if (self.gated_delta_stage_output_host.len < d_model) {
            if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
            self.gated_delta_stage_output_host = try self.allocator.alloc(f32, d_model);
        }
        for (0..seq_len) |t| {
            var output_row = try logicalF32RowSlice(output, seq_len, t, d_model);
            try output_row.download(&self.device, std.mem.sliceAsBytes(self.gated_delta_stage_output_host[0..d_model]));
            const prev_gpu_backend = trace.setBackendContext(.cuda);
            defer _ = trace.setBackendContext(prev_gpu_backend);
            trace.emit(
                .gdelta_out,
                block.kernel.layer_idx,
                0,
                @intCast(trace_pos_offset + t),
                @ptrCast(self.gated_delta_stage_output_host.ptr),
                .f32,
                .{ 1, 1, @intCast(d_model), 0 },
                3,
                "cuda_gdelta_out_host",
            );
        }
    }
    // Tracing downloads intermediate tensors to host. Keep strict ordering
    // there, but avoid per-token global sync in normal inference.
    if (trace_enabled) {
        try self.device.synchronize();
    }
}

/// Batched decode path for gated delta layers: N slots × 1 token each.
/// in_proj and out_proj use GEMM for N rows; conv/SSM/norm run in rows
/// kernels with per-slot state pointers.
pub fn runBatchedDecodeGatedDeltaMixer(
    self: anytype,
    block: *GatedDeltaBlockRuntime,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    ctx: anytype,
    batch: anytype,
) !void {
    const n_rows = ctx.active_rows_u32;
    const n: usize = @intCast(n_rows);
    const cfg = block.kernel.config;
    const d_inner: usize = @as(usize, cfg.n_heads) * @as(usize, cfg.d_head);
    const d_conv: usize = cfg.d_conv;
    const n_v_heads: usize = cfg.n_heads;
    const d_head: usize = cfg.d_head;
    const proj_len = block.in_proj.cols();
    const qkv_len = blk: {
        const values = block.kernel.weights.conv1d_weight.asSlice(f32);
        if (values.len == 0 or d_conv == 0 or (values.len % d_conv) != 0) return error.InvalidShape;
        break :blk values.len / d_conv;
    };
    const minimum_proj = d_inner + (2 * n_v_heads);
    if (proj_len <= minimum_proj) return error.InvalidShape;

    // Buffer sizes for N rows.
    const proj_element_count = std.math.mul(usize, n, proj_len) catch return error.InvalidArgument;
    const proj_bytes = std.math.mul(usize, proj_element_count, @sizeOf(f32)) catch return error.InvalidArgument;
    const ssm_bytes = std.math.mul(usize, d_inner, @sizeOf(f32)) catch return error.InvalidArgument;
    const norm_total_bytes = std.math.mul(usize, n * d_inner, @sizeOf(f32)) catch return error.InvalidArgument;

    var proj_dev = try bufferSlice(&self.runtime_buffers.gdelta_proj_dev, 0, proj_bytes);
    var norm_stage_dev = try bufferSlice(&self.runtime_buffers.gdelta_ssm_dev, 0, norm_total_bytes);

    // Step 1: in_proj GEMM for all N rows.
    try engine_ops.linearForwardRows(self, input, n, &block.in_proj, &proj_dev);

    // QKV shape validation.
    if (qkv_len <= d_inner) return error.InvalidShape;
    const qk_total = qkv_len - d_inner;
    if ((qk_total % 2) != 0) return error.InvalidShape;
    const qk_inner = qk_total / 2;
    if ((qk_inner % d_head) != 0) return error.InvalidShape;
    const n_qk_heads = qk_inner / d_head;
    if (n_qk_heads == 0 or (n_v_heads % n_qk_heads) != 0) return error.InvalidShape;

    const beta_offset_elems = qkv_len + d_inner;
    const a_offset_elems = beta_offset_elems + n_v_heads;
    const ptr_bytes = std.math.mul(usize, n, @sizeOf(u64)) catch return error.InvalidArgument;
    const idx_bytes = std.math.mul(usize, n, @sizeOf(u32)) catch return error.InvalidArgument;
    const gd_table_row_offset = std.math.mul(usize, batch.gd_layer_index, batch.gd_ptrs_row_stride) catch return error.InvalidArgument;
    const gd_table_ptr_byte_offset = std.math.mul(usize, gd_table_row_offset, @sizeOf(u64)) catch return error.InvalidArgument;
    const gd_table_idx_byte_offset = std.math.mul(usize, gd_table_row_offset, @sizeOf(u32)) catch return error.InvalidArgument;
    const head_bytes = std.math.mul(usize, d_head, @sizeOf(f32)) catch return error.InvalidArgument;
    const norm_weight_supported = (block.norm_weight.buffer.size == head_bytes) or (block.norm_weight.buffer.size == ssm_bytes);
    if (!norm_weight_supported) return error.InvalidShape;
    const conv_rows_ptrs_function = self.gated_delta_conv_silu_rows_ptrs_function orelse return error.CudaKernelUnavailable;
    const quantized_ssm_state = block.ssm_state_format == .i8_per_column_scale;
    const norm_rows_function = self.gated_delta_rmsnorm_silu_mul_rows_function orelse return error.CudaKernelUnavailable;
    const rows_norm_weight_stride: u32 = if (block.norm_weight.buffer.size == head_bytes)
        0
    else
        @intCast(d_head);

    // Step 2: Conv + SSM + norm/gate for all rows with per-slot state pointers.
    var conv_state_ptrs_dev = try bufferSlice(batch.gd_conv_state_ptrs_table_dev, gd_table_ptr_byte_offset, ptr_bytes);
    var ssm_state_ptrs_dev = try bufferSlice(batch.gd_ssm_state_ptrs_table_dev, gd_table_ptr_byte_offset, ptr_bytes);
    var conv_ring_heads_dev = try bufferSlice(batch.gd_conv_ring_heads_table_dev, gd_table_idx_byte_offset, idx_bytes);
    const conv_ring_heads_host = self.runtime_buffers.decode_gd_conv_ring_heads_table_host[gd_table_row_offset..][0..n];

    try compute.cuda.gated_delta_conv_silu_rows_ptrs.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        conv_rows_ptrs_function,
        &proj_dev,
        &conv_state_ptrs_dev,
        &conv_ring_heads_dev,
        &block.conv_weight_time_major.buffer,
        if (block.conv_bias) |*bias| &bias.buffer else null,
        &proj_dev,
        @intCast(qkv_len),
        @intCast(d_conv),
        n_rows,
        @intCast(proj_len),
    );

    try compute.cuda.gated_delta_conv_silu_rows_ptrs.runAdvanceWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.gated_delta_advance_ring_heads_function orelse return error.CudaKernelUnavailable,
        &conv_ring_heads_dev,
        @intCast(d_conv),
        n_rows,
    );

    const d_conv_u32: u32 = @intCast(d_conv);
    for (0..n) |row_i| {
        const slot_idx = batch.slot_indices[row_i];
        const gd_state = &self.slot_kv_states[slot_idx].gd[batch.gd_layer_index];
        const ring_head = conv_ring_heads_host[row_i];
        gd_state.conv_ring_head = if (ring_head + 1 >= d_conv_u32) 0 else ring_head + 1;
    }

    if (quantized_ssm_state) {
        const ssm_rows_ptrs_i8_function = self.gated_delta_ssm_rows_ptrs_i8_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.gated_delta_ssm_rows_ptrs_i8.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            ssm_rows_ptrs_i8_function,
            &proj_dev,
            &ssm_state_ptrs_dev,
            &block.a_log.buffer,
            if (block.dt_bias) |*bias| &bias.buffer else null,
            &norm_stage_dev,
            @intCast(n_qk_heads),
            @intCast(n_v_heads),
            @intCast(d_head),
            n_rows,
            @intCast(proj_len),
            @intCast(beta_offset_elems),
            @intCast(a_offset_elems),
            @intCast(d_inner),
            block.ssm_state_scales_offset,
        );
    } else {
        const ssm_rows_ptrs_function = self.gated_delta_ssm_rows_ptrs_function orelse return error.CudaKernelUnavailable;
        try compute.cuda.gated_delta_ssm_rows_ptrs.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            ssm_rows_ptrs_function,
            &proj_dev,
            &ssm_state_ptrs_dev,
            &block.a_log.buffer,
            if (block.dt_bias) |*bias| &bias.buffer else null,
            &norm_stage_dev,
            @intCast(n_qk_heads),
            @intCast(n_v_heads),
            @intCast(d_head),
            n_rows,
            @intCast(proj_len),
            @intCast(beta_offset_elems),
            @intCast(a_offset_elems),
            @intCast(d_inner),
        );
    }

    const rows_total = std.math.mul(usize, n, n_v_heads) catch return error.InvalidArgument;
    try compute.cuda.gated_delta_rmsnorm_silu_mul_rows.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        norm_rows_function,
        &norm_stage_dev,
        &proj_dev,
        &block.norm_weight.buffer,
        &norm_stage_dev,
        @intCast(rows_total),
        @intCast(d_head),
        @intCast(n_v_heads),
        @intCast(proj_len),
        @intCast(qkv_len),
        1.0e-6,
        rows_norm_weight_stride,
    );

    // Step 3: out_proj GEMM for all N rows.
    try engine_ops.linearForwardRows(self, &norm_stage_dev, n, &block.out_proj, output);
}

pub fn runAttentionMixerStepCpu(
    self: anytype,
    block: *LayerAttentionRuntime,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    seq_len: usize,
) !void {
    const kernel = &(block.cpu_kernel orelse return error.InvalidInstructionBinding);
    const cache = &(block.cpu_cache orelse return error.InvalidInstructionBinding);
    const scratch = &(block.cpu_scratch orelse return error.InvalidInstructionBinding);
    const matmul_scratch = &(block.cpu_matmul_scratch orelse return error.InvalidInstructionBinding);
    const element_count = std.math.mul(usize, seq_len, self.d_model) catch return error.InvalidArgument;
    try engine_forward.ensureGatedDeltaHostStageCapacity(self, element_count);
    const input_host = self.gated_delta_stage_input_host[0..element_count];
    const output_host = self.gated_delta_stage_output_host[0..element_count];
    try self.device.synchronize();
    try downloadRowsF32StrideAware(self, input, seq_len, self.d_model, input_host);
    kernel.position_delta = self.slot_rope_position_deltas[self.active_kv_slot];
    var input_view = Tensor.view3DSlice(input_host, seq_len, self.d_model);
    var output_view = Tensor.view3DSlice(output_host, seq_len, self.d_model);
    const use_cache = attentionFallbackUsesCache(seq_len);
    // Attention fallback executes on CPU tensors; preserve host-readable
    // trace semantics even when wrapped by the CUDA execution route.
    const prev_backend = trace.setBackendContext(.cpu);
    defer _ = trace.setBackendContext(prev_backend);
    try kernel.forward(
        &input_view,
        &output_view,
        cache,
        scratch,
        matmul_scratch,
        use_cache,
    );
    try uploadRowsF32StrideAware(self, output_host, seq_len, self.d_model, output);
    try self.device.synchronize();
}

pub fn runAttentionMixerPrefillBatchedNoQueryGate(
    self: anytype,
    cfg: *const LayerAttentionExecConfig,
    k_cache: *const compute.cuda.Buffer,
    v_cache: *const compute.cuda.Buffer,
    k_scale: *const compute.cuda.Buffer,
    v_scale: *const compute.cuda.Buffer,
    read_k_cache: *const compute.cuda.Buffer,
    read_v_cache: *const compute.cuda.Buffer,
    read_k_scale: *const compute.cuda.Buffer,
    read_v_scale: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    o_proj: *const LinearWeight,
    q_norm_weight: ?*const DeviceTensor,
    k_norm_weight: ?*const DeviceTensor,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    d_model_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
    seq_len_u32: u32,
    global_rope_theta: f32,
    local_rope_theta: f32,
    rope_function: compute.cuda.Function,
    copy_function: compute.cuda.Function,
    cast_f32_to_f16_function: ?compute.cuda.Function,
    kv_write_f16_function: ?compute.cuda.Function,
    rope_store_f16_function: ?compute.cuda.Function,
    attention_kernels: AttentionKernelSet,
) !void {
    if (cfg.query_gate) return error.InvalidInstructionBinding;

    const stage_rows = try bufferF32RowCount(input, @intCast(d_model_u32));
    if (stage_rows == 0) return error.InvalidInstructionBinding;
    if (stage_rows > @as(usize, seq_len_u32)) return error.InvalidInstructionBinding;

    // Position base for chunked prefill: this chunk writes KV at
    // [position_base, position_base + stage_rows) in the cache.
    const position_base_u32: u32 = seq_len_u32 - @as(u32, @intCast(stage_rows));
    const attention_scale = attentionScaleForHeadDim(self, head_dim_u32);
    if (stage_rows == 1) {
        const position: usize = @intCast(position_base_u32);
        var input_row = try logicalF32RowSlice(input, stage_rows, 0, @intCast(d_model_u32));
        var output_row = try logicalF32RowSlice(output, stage_rows, 0, o_proj.cols());
        try runAttentionMixerStep(
            self,
            cfg,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            read_k_cache,
            read_v_cache,
            read_k_scale,
            read_v_scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm_weight,
            k_norm_weight,
            &input_row,
            &output_row,
            d_model_u32,
            head_dim_u32,
            rope_dim_u32,
            n_heads_u32,
            n_kv_heads_u32,
            seq_len_u32,
            position,
            position_base_u32,
            global_rope_theta,
            local_rope_theta,
            rope_function,
            copy_function,
            cast_f32_to_f16_function,
            kv_write_f16_function,
            rope_store_f16_function,
            attention_kernels,
            null,
        );
        return;
    }

    const layer_rope_theta = effectiveLayerRopeTheta(
        self,
        cfg,
        head_dim_u32,
        rope_dim_u32,
        global_rope_theta,
        local_rope_theta,
    );
    const q_stage_bytes = std.math.mul(usize, stage_rows, cfg.q_projection_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const kv_stage_bytes = std.math.mul(usize, stage_rows, cfg.kv_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const context_stage_bytes = std.math.mul(usize, stage_rows, o_proj.rows() * @sizeOf(f32)) catch return error.InvalidArgument;
    var attn_q_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_stage_bytes);
    var attn_k_stage = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, kv_stage_bytes);
    var attn_v_stage = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, kv_stage_bytes);
    var attn_context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_stage_bytes);

    _ = engine_ops.runQkvProjection(self, input, q_proj, k_proj, v_proj, stage_rows, &self.runtime_buffers.attn_q_dev) catch |err| {
        log.warn("inference", "CUDA prefill QKV projection failed", .{
            .rows = stage_rows,
            .q_dim = cfg.q_dim,
            .q_projection_dim = cfg.q_projection_dim,
            .kv_dim = cfg.kv_dim,
            .reason = @errorName(err),
        });
        return err;
    };
    if (debugKernelSyncEnabled()) {
        self.device.synchronize() catch |err| {
            log.warn("inference", "CUDA debug sync failed after prefill qkv", .{
                .reason = @errorName(err),
            });
            return err;
        };
    }

    if (q_norm_weight) |q_norm_value| {
        const q_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_heads_u32) catch return error.InvalidArgument;
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_q_stage,
            &q_norm_value.buffer,
            &attn_q_stage,
            q_norm_rows,
            head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        );
    }
    if (k_norm_weight) |k_norm_value| {
        const k_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_kv_heads_u32) catch return error.InvalidArgument;
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_k_stage,
            &k_norm_value.buffer,
            &attn_k_stage,
            k_norm_rows,
            head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        );
    }
    try applyValueNormInPlace(self, &attn_v_stage, stage_rows, n_kv_heads_u32, head_dim_u32);
    if (debugKernelSyncEnabled()) {
        self.device.synchronize() catch |err| {
            log.warn("inference", "CUDA debug sync failed after prefill qk_norm", .{
                .reason = @errorName(err),
            });
            return err;
        };
    }

    const kv_row_f32_bytes = std.math.mul(usize, cfg.kv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    const kv_elem_bytes: usize = self.kv_cache_dtype.elementBytes();
    const kv_row_bytes = std.math.mul(usize, cfg.kv_dim, kv_elem_bytes) catch return error.InvalidArgument;
    const kv_dim_u32: u32 = @intCast(cfg.kv_dim);
    const kv_groups_u32: u32 = n_heads_u32 / n_kv_heads_u32;
    const use_k_write_fused = switch (self.kv_cache_dtype) {
        .f16 => kv_write_f16_function != null or rope_store_f16_function != null,
        .i8 => self.kv_write_i8_function != null,
        .fp8 => self.kv_write_fp8_function != null,
    };
    const can_flash_prefill = (head_dim_u32 <= 256) and cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => self.flash_prefill_f16_function != null,
        .i8 => self.flash_prefill_i8_function != null,
        .fp8 => self.flash_prefill_fp8_function != null,
    };
    const can_fused_prefill_attn = cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => attention_kernels.attn_fused_prefill_heads_f16_kv_function != null,
        .i8 => attention_kernels.attn_fused_prefill_heads_i8_kv_function != null,
        .fp8 => attention_kernels.attn_fused_prefill_heads_fp8_kv_function != null,
    };
    const can_any_fused_prefill = can_flash_prefill or can_fused_prefill_attn;
    const can_batched_kv_write_prefill = can_any_fused_prefill and switch (self.kv_cache_dtype) {
        .f16 => kv_write_f16_function != null and self.kv_write_f16_rows_function != null,
        .i8 => self.kv_write_i8_rows_function != null,
        .fp8 => self.kv_write_fp8_rows_function != null,
    };

    if (can_batched_kv_write_prefill) {
        const kv_cache_offset = std.math.mul(usize, @as(usize, position_base_u32), kv_row_bytes) catch return error.InvalidArgument;
        var k_cache_out = k_cache.*;
        k_cache_out.pointer += kv_cache_offset;
        k_cache_out.size -= kv_cache_offset;
        var v_cache_out = v_cache.*;
        v_cache_out.pointer += kv_cache_offset;
        v_cache_out.size -= kv_cache_offset;
        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.kv_write_f16_rows.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_f16_rows_function.?,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_cache_out,
                    &v_cache_out,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(stage_rows),
                    @intCast(cfg.kv_dim),
                    position_base_u32,
                    layer_rope_theta,
                );
            },
            .i8 => {
                const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset = std.math.mul(usize, @as(usize, position_base_u32), scale_row_bytes) catch return error.InvalidArgument;
                var k_scale_out = k_scale.*;
                k_scale_out.pointer += scale_offset;
                k_scale_out.size -= scale_offset;
                var v_scale_out = v_scale.*;
                v_scale_out.pointer += scale_offset;
                v_scale_out.size -= scale_offset;
                try compute.cuda.kv_write_i8_rows.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_i8_rows_function.?,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_cache_out,
                    &v_cache_out,
                    &k_scale_out,
                    &v_scale_out,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(stage_rows),
                    @intCast(cfg.kv_dim),
                    position_base_u32,
                    layer_rope_theta,
                );
            },
            .fp8 => {
                const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset = std.math.mul(usize, @as(usize, position_base_u32), scale_row_bytes) catch return error.InvalidArgument;
                var k_scale_out = k_scale.*;
                k_scale_out.pointer += scale_offset;
                k_scale_out.size -= scale_offset;
                var v_scale_out = v_scale.*;
                v_scale_out.pointer += scale_offset;
                v_scale_out.size -= scale_offset;
                try compute.cuda.kv_write_fp8_rows.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_fp8_rows_function.?,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_cache_out,
                    &v_cache_out,
                    &k_scale_out,
                    &v_scale_out,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(stage_rows),
                    @intCast(cfg.kv_dim),
                    position_base_u32,
                    layer_rope_theta,
                );
            },
        }
    } else {
        var row_idx: usize = 0;
        while (row_idx < stage_rows) : (row_idx += 1) {
            const kv_offset_f32 = std.math.mul(usize, row_idx, kv_row_f32_bytes) catch return error.InvalidArgument;
            var k_row_in = try bufferSlice(&attn_k_stage, kv_offset_f32, kv_row_f32_bytes);
            var v_row_in = try bufferSlice(&attn_v_stage, kv_offset_f32, kv_row_f32_bytes);

            const position_u32: u32 = position_base_u32 + @as(u32, @intCast(row_idx));
            if (!use_k_write_fused) {
                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_function,
                    &k_row_in,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            }

            const cache_row = @as(usize, position_base_u32) + row_idx;
            const kv_offset = std.math.mul(usize, cache_row, kv_row_bytes) catch return error.InvalidArgument;
            var k_row_out = try bufferSlice(k_cache, kv_offset, kv_row_bytes);
            var v_row_out = try bufferSlice(v_cache, kv_offset, kv_row_bytes);

            switch (self.kv_cache_dtype) {
                .f16 => {
                    if (kv_write_f16_function) |kv_write_f16| {
                        try compute.cuda.kv_write_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kv_write_f16,
                            &k_row_in,
                            &v_row_in,
                            &k_row_out,
                            &v_row_out,
                            n_kv_heads_u32,
                            head_dim_u32,
                            rope_dim_u32,
                            position_u32,
                            layer_rope_theta,
                        );
                    } else if (rope_store_f16_function) |rope_store_f16| {
                        try compute.cuda.rope_store_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            rope_store_f16,
                            &k_row_in,
                            &k_row_out,
                            n_kv_heads_u32,
                            head_dim_u32,
                            rope_dim_u32,
                            position_u32,
                            layer_rope_theta,
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &v_row_in,
                            &v_row_out,
                            @intCast(cfg.kv_dim),
                        );
                    } else {
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &k_row_in,
                            &k_row_out,
                            @intCast(cfg.kv_dim),
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &v_row_in,
                            &v_row_out,
                            @intCast(cfg.kv_dim),
                        );
                    }
                },
                .i8 => {
                    const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                    const scale_offset = std.math.mul(usize, cache_row, scale_row_bytes) catch return error.InvalidArgument;
                    var k_scale_row = try bufferSlice(k_scale, scale_offset, scale_row_bytes);
                    var v_scale_row = try bufferSlice(v_scale, scale_offset, scale_row_bytes);
                    try compute.cuda.kv_write_i8.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.kv_write_i8_function orelse return error.CudaKernelUnavailable,
                        &k_row_in,
                        &v_row_in,
                        &k_row_out,
                        &v_row_out,
                        &k_scale_row,
                        &v_scale_row,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                },
                .fp8 => {
                    const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                    const scale_offset = std.math.mul(usize, cache_row, scale_row_bytes) catch return error.InvalidArgument;
                    var k_scale_row = try bufferSlice(k_scale, scale_offset, scale_row_bytes);
                    var v_scale_row = try bufferSlice(v_scale, scale_offset, scale_row_bytes);
                    try compute.cuda.kv_write_fp8.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.kv_write_fp8_function orelse return error.CudaKernelUnavailable,
                        &k_row_in,
                        &v_row_in,
                        &k_row_out,
                        &v_row_out,
                        &k_scale_row,
                        &v_scale_row,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                },
            }
            if (!can_any_fused_prefill) {
                const q_row_bytes = std.math.mul(usize, cfg.q_projection_dim, @sizeOf(f32)) catch return error.InvalidArgument;
                const ctx_row_bytes = std.math.mul(usize, o_proj.rows(), @sizeOf(f32)) catch return error.InvalidArgument;
                const q_offset = std.math.mul(usize, row_idx, q_row_bytes) catch return error.InvalidArgument;
                const ctx_offset = std.math.mul(usize, row_idx, ctx_row_bytes) catch return error.InvalidArgument;
                var q_row = try bufferSlice(&attn_q_stage, q_offset, q_row_bytes);
                var ctx_row = try bufferSlice(&attn_context_stage, ctx_offset, ctx_row_bytes);
                const effective_seq_len_u32: u32 = position_u32 + 1;
                const use_fused_attention_heads_f16_kv = attention_mod.useFusedHeadsF16Kv(
                    attention_policy_config,
                    effective_seq_len_u32,
                    cfg.sliding_window,
                    cfg.is_causal,
                    head_dim_u32,
                    attention_kernels.attn_fused_heads_f16_kv_function != null,
                );
                if (!use_fused_attention_heads_f16_kv) {
                    try compute.cuda.rope.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        rope_function,
                        &q_row,
                        n_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                }
                _ = try self.runAttentionContext(
                    cfg,
                    &q_row,
                    &ctx_row,
                    read_k_cache,
                    read_v_cache,
                    read_k_scale,
                    read_v_scale,
                    attention_kernels,
                    effective_seq_len_u32,
                    head_dim_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    n_heads_u32,
                    attention_scale,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            }
        }
    }
    if (debugKernelSyncEnabled()) {
        self.device.synchronize() catch |err| {
            log.warn("inference", "CUDA debug sync failed after prefill kv-write", .{
                .reason = @errorName(err),
            });
            return err;
        };
    }

    if (can_any_fused_prefill) {
        const sliding_window_u32 = std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32);
        switch (self.kv_cache_dtype) {
            .f16 => {
                const use_gqa = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_f16_kv_gqa_function != null;

                // GEMM-based attention: cuBLAS strided batched GEMM for Q×K^T and
                // probs×V.  Preferred for prefill because cuBLAS leverages tensor
                // cores and reads K/V once per KV head instead of once per Q row.
                const can_gemm_attn = use_gqa and
                    attention_kernels.causal_attn_softmax_f32_function != null;

                if (can_flash_prefill) {
                    try compute.cuda.flash_prefill.runF16(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.flash_prefill_f16_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &attn_context_stage,
                        n_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                } else if (can_gemm_attn) {
                    // Apply RoPE to Q (per-row, each row has a different position).
                    // The fused kernels do this internally; the GEMM path needs it upfront.
                    const q_row_dim = cfg.q_projection_dim;
                    const q_row_bytes_rope = std.math.mul(usize, q_row_dim, @sizeOf(f32)) catch return error.InvalidArgument;
                    var rope_row_idx: usize = 0;
                    while (rope_row_idx < stage_rows) : (rope_row_idx += 1) {
                        const q_row_offset = std.math.mul(usize, rope_row_idx, q_row_bytes_rope) catch return error.InvalidArgument;
                        var q_row = try bufferSlice(&attn_q_stage, q_row_offset, q_row_bytes_rope);
                        const pos = position_base_u32 + @as(u32, @intCast(rope_row_idx));
                        try compute.cuda.rope.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            rope_function,
                            &q_row,
                            n_heads_u32,
                            head_dim_u32,
                            rope_dim_u32,
                            pos,
                            layer_rope_theta,
                        );
                    }

                    // GEMM-based attention per KV head.
                    var scores_buf = try ensureAttnScoresWorkspace(
                        self,
                        kv_groups_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                    );
                    const q_ld: usize = cfg.q_projection_dim;
                    const kv_ld: usize = cfg.kv_dim;
                    const ctx_ld: usize = o_proj.rows();
                    const hd: usize = head_dim_u32;
                    const sl: usize = seq_len_u32;
                    const qr: usize = stage_rows;
                    const n_kv: u32 = n_heads_u32 / kv_groups_u32;

                    // Cast Q from f32 to f16 for tensor-core GEMM (f16×f16→f32).
                    const q_f16_elems = std.math.mul(usize, qr, cfg.q_projection_dim) catch return error.InvalidArgument;
                    const q_f16_bytes = std.math.mul(usize, q_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
                    const probs_f16_elems = std.math.mul(usize, std.math.mul(usize, kv_groups_u32, qr) catch return error.InvalidArgument, sl) catch return error.InvalidArgument;
                    const probs_f16_bytes = std.math.mul(usize, probs_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
                    var u16_ws = try ensureAttnU16Workspace(self, q_f16_bytes + probs_f16_bytes);
                    var q_f16_buf = try bufferSlice(&u16_ws, 0, q_f16_bytes);
                    var probs_f16_buf = try bufferSlice(&u16_ws, q_f16_bytes, probs_f16_bytes);
                    try compute.cuda.cast_f32_to_f16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                        &attn_q_stage,
                        &q_f16_buf,
                        @intCast(q_f16_elems),
                    );

                    var kv_h: u32 = 0;
                    while (kv_h < n_kv) : (kv_h += 1) {
                        const k_ptr = read_k_cache.pointer + @as(usize, kv_h) * hd * @sizeOf(u16);
                        const v_ptr = read_v_cache.pointer + @as(usize, kv_h) * hd * @sizeOf(u16);
                        const q_f16_ptr = q_f16_buf.pointer + @as(usize, kv_h) * kv_groups_u32 * hd * @sizeOf(u16);
                        const out_ptr = attn_context_stage.pointer + @as(usize, kv_h) * kv_groups_u32 * hd * @sizeOf(f32);

                        // Q × K^T → scores [kv_groups × q_rows × seq_len].
                        // f16 × f16 → f32 via tensor cores.
                        var g: u32 = 0;
                        while (g < kv_groups_u32) : (g += 1) {
                            try self.blas.gemmU16(
                                &self.device,
                                true, // transa=T for K
                                sl, // m = seq_len
                                qr, // n = q_rows
                                hd, // k = head_dim
                                attention_scale,
                                k_ptr,
                                kv_ld, // lda = kv_dim
                                q_f16_ptr + @as(usize, g) * hd * @sizeOf(u16),
                                q_ld, // ldb = q_projection_dim
                                0.0,
                                scores_buf.pointer + @as(usize, g) * qr * sl * @sizeOf(f32),
                                sl, // ldc = seq_len
                            );
                        }

                        // Causal mask + softmax (operates on f32 scores).
                        try compute.cuda.causal_attn_softmax_f32.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            attention_kernels.causal_attn_softmax_f32_function.?,
                            &scores_buf,
                            kv_groups_u32 * @as(u32, @intCast(stage_rows)),
                            seq_len_u32,
                            @intCast(stage_rows),
                            position_base_u32,
                            sliding_window_u32,
                        );

                        // Cast probs from f32 to f16 for the V GEMM.
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &scores_buf,
                            &probs_f16_buf,
                            @intCast(probs_f16_elems),
                        );

                        // probs × V → output [kv_groups × q_rows × head_dim].
                        // f16 × f16 → f32 via tensor cores.
                        g = 0;
                        while (g < kv_groups_u32) : (g += 1) {
                            try self.blas.gemmU16(
                                &self.device,
                                false, // transa=N for V
                                hd, // m = head_dim
                                qr, // n = q_rows
                                sl, // k = seq_len
                                1.0,
                                v_ptr,
                                kv_ld, // lda = kv_dim
                                probs_f16_buf.pointer + @as(usize, g) * qr * sl * @sizeOf(u16),
                                sl, // ldb = seq_len
                                0.0,
                                out_ptr + @as(usize, g) * hd * @sizeOf(f32),
                                ctx_ld, // ldc = context_dim
                            );
                        }
                    }
                } else if (use_gqa) {
                    try compute.cuda.attn_fused_prefill_heads_f16_kv_gqa.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_f16_kv_gqa_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &attn_context_stage,
                        n_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                } else {
                    try compute.cuda.attn_fused_prefill_heads_f16_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_f16_kv_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &attn_context_stage,
                        n_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                }
            },
            .i8 => {
                const use_gqa_i8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_i8_kv_gqa_function != null;
                const scale_row_bytes_attn: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset_attn = std.math.mul(usize, 0, scale_row_bytes_attn) catch return error.InvalidArgument;
                var k_scale_attn = try bufferSlice(read_k_scale, scale_offset_attn, read_k_scale.size);
                var v_scale_attn = try bufferSlice(read_v_scale, scale_offset_attn, read_v_scale.size);
                if (can_flash_prefill) {
                    try compute.cuda.flash_prefill.runWithScales(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.flash_prefill_i8_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                } else if (use_gqa_i8) {
                    try compute.cuda.attn_fused_prefill_heads_i8_kv_gqa.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_i8_kv_gqa_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                } else {
                    try compute.cuda.attn_fused_prefill_heads_i8_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_i8_kv_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                }
            },
            .fp8 => {
                const use_gqa_fp8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_fp8_kv_gqa_function != null;
                const scale_row_bytes_attn_fp8: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset_attn_fp8 = std.math.mul(usize, 0, scale_row_bytes_attn_fp8) catch return error.InvalidArgument;
                var k_scale_attn_fp8 = try bufferSlice(read_k_scale, scale_offset_attn_fp8, read_k_scale.size);
                var v_scale_attn_fp8 = try bufferSlice(read_v_scale, scale_offset_attn_fp8, read_v_scale.size);
                if (can_flash_prefill) {
                    try compute.cuda.flash_prefill.runWithScales(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.flash_prefill_fp8_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                } else if (use_gqa_fp8) {
                    try compute.cuda.attn_fused_prefill_heads_fp8_kv_gqa.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_fp8_kv_gqa_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                } else {
                    try compute.cuda.attn_fused_prefill_heads_fp8_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_fp8_kv_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                    );
                }
            },
        }
    }

    engine_ops.linearForwardRows(self, &attn_context_stage, stage_rows, o_proj, output) catch |err| {
        log.warn("inference", "CUDA prefill attention O projection failed", .{
            .rows = stage_rows,
            .in_dim = o_proj.rows(),
            .out_dim = o_proj.cols(),
            .reason = @errorName(err),
        });
        return err;
    };
}

pub fn runAttentionMixerPrefillBatchedWithQueryGate(
    self: anytype,
    cfg: *const LayerAttentionExecConfig,
    k_cache: *const compute.cuda.Buffer,
    v_cache: *const compute.cuda.Buffer,
    k_scale: *const compute.cuda.Buffer,
    v_scale: *const compute.cuda.Buffer,
    read_k_cache: *const compute.cuda.Buffer,
    read_v_cache: *const compute.cuda.Buffer,
    read_k_scale: *const compute.cuda.Buffer,
    read_v_scale: *const compute.cuda.Buffer,
    q_proj: *const LinearWeight,
    k_proj: *const LinearWeight,
    v_proj: *const LinearWeight,
    o_proj: *const LinearWeight,
    q_norm_weight: ?*const DeviceTensor,
    k_norm_weight: ?*const DeviceTensor,
    input: *const compute.cuda.Buffer,
    output: *compute.cuda.Buffer,
    d_model_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
    seq_len_u32: u32,
    global_rope_theta: f32,
    local_rope_theta: f32,
    rope_function: compute.cuda.Function,
    copy_function: compute.cuda.Function,
    cast_f32_to_f16_function: ?compute.cuda.Function,
    kv_write_f16_function: ?compute.cuda.Function,
    rope_store_f16_function: ?compute.cuda.Function,
    attention_kernels: AttentionKernelSet,
) !void {
    if (!cfg.query_gate) return error.InvalidInstructionBinding;

    const stage_rows = try bufferF32RowCount(input, @intCast(d_model_u32));
    if (stage_rows == 0) return error.InvalidInstructionBinding;
    if (stage_rows > @as(usize, seq_len_u32)) return error.InvalidInstructionBinding;

    // Position base for chunked prefill: this chunk writes KV at
    // [position_base, position_base + stage_rows) in the cache.
    const position_base_u32: u32 = seq_len_u32 - @as(u32, @intCast(stage_rows));
    const attention_scale = attentionScaleForHeadDim(self, head_dim_u32);
    if (stage_rows == 1) {
        const position: usize = @intCast(position_base_u32);
        var input_row = try logicalF32RowSlice(input, stage_rows, 0, @intCast(d_model_u32));
        var output_row = try logicalF32RowSlice(output, stage_rows, 0, o_proj.cols());
        try runAttentionMixerStep(
            self,
            cfg,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            read_k_cache,
            read_v_cache,
            read_k_scale,
            read_v_scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm_weight,
            k_norm_weight,
            &input_row,
            &output_row,
            d_model_u32,
            head_dim_u32,
            rope_dim_u32,
            n_heads_u32,
            n_kv_heads_u32,
            seq_len_u32,
            position,
            position_base_u32,
            global_rope_theta,
            local_rope_theta,
            rope_function,
            copy_function,
            cast_f32_to_f16_function,
            kv_write_f16_function,
            rope_store_f16_function,
            attention_kernels,
            null,
        );
        return;
    }

    const layer_rope_theta = effectiveLayerRopeTheta(
        self,
        cfg,
        head_dim_u32,
        rope_dim_u32,
        global_rope_theta,
        local_rope_theta,
    );

    // Q projection output goes into query_gate_proj_dev (q_projection_dim per row).
    const q_proj_stage_bytes = std.math.mul(usize, stage_rows, cfg.q_projection_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    var q_projection_stage = try bufferSlice(&self.runtime_buffers.query_gate_proj_dev, 0, q_proj_stage_bytes);

    // After compact, Q values (q_dim per row) go into attn_q_dev.
    const q_values_bytes = std.math.mul(usize, stage_rows, cfg.q_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const kv_stage_bytes = std.math.mul(usize, stage_rows, cfg.kv_dim * @sizeOf(f32)) catch return error.InvalidArgument;
    const context_stage_bytes = std.math.mul(usize, stage_rows, o_proj.rows() * @sizeOf(f32)) catch return error.InvalidArgument;
    var attn_q_stage = try bufferSlice(&self.runtime_buffers.attn_q_dev, 0, q_values_bytes);
    var attn_k_stage = try bufferSlice(&self.runtime_buffers.attn_k_dev, 0, kv_stage_bytes);
    var attn_v_stage = try bufferSlice(&self.runtime_buffers.attn_v_dev, 0, kv_stage_bytes);
    var attn_context_stage = try bufferSlice(&self.runtime_buffers.attn_context_dev, 0, context_stage_bytes);

    // Batched QKV projection — Q output goes into query_gate_proj_dev.
    _ = engine_ops.runQkvProjection(self, input, q_proj, k_proj, v_proj, stage_rows, &self.runtime_buffers.query_gate_proj_dev) catch |err| {
        log.warn("inference", "CUDA prefill QKV projection(query-gate) failed", .{
            .rows = stage_rows,
            .q_dim = cfg.q_dim,
            .q_projection_dim = cfg.q_projection_dim,
            .kv_dim = cfg.kv_dim,
            .reason = @errorName(err),
        });
        return err;
    };

    // Extract Q values and save gate for later.
    try engine_ops.compactQueryGateProjection(self, stage_rows, cfg.q_dim, cfg.q_projection_dim, n_heads_u32, head_dim_u32, &q_projection_stage, &attn_q_stage);

    if (q_norm_weight) |q_norm_value| {
        const q_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_heads_u32) catch return error.InvalidArgument;
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_q_stage,
            &q_norm_value.buffer,
            &attn_q_stage,
            q_norm_rows,
            head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        );
    }
    if (k_norm_weight) |k_norm_value| {
        const k_norm_rows = std.math.mul(u32, @intCast(stage_rows), n_kv_heads_u32) catch return error.InvalidArgument;
        try compute.cuda.rmsnorm.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            self.rmsnorm_function orelse return error.CudaKernelUnavailable,
            &attn_k_stage,
            &k_norm_value.buffer,
            &attn_k_stage,
            k_norm_rows,
            head_dim_u32,
            self.norm_eps,
            self.loaded.runtime.qk_norm_weight_offset,
        );
    }
    try applyValueNormInPlace(self, &attn_v_stage, stage_rows, n_kv_heads_u32, head_dim_u32);

    const kv_row_f32_bytes = std.math.mul(usize, cfg.kv_dim, @sizeOf(f32)) catch return error.InvalidArgument;
    const kv_elem_bytes: usize = self.kv_cache_dtype.elementBytes();
    const kv_row_bytes = std.math.mul(usize, cfg.kv_dim, kv_elem_bytes) catch return error.InvalidArgument;
    const kv_dim_u32: u32 = @intCast(cfg.kv_dim);
    const kv_groups_u32: u32 = n_heads_u32 / n_kv_heads_u32;
    const use_k_write_fused = switch (self.kv_cache_dtype) {
        .f16 => kv_write_f16_function != null or rope_store_f16_function != null,
        .i8 => self.kv_write_i8_function != null,
        .fp8 => self.kv_write_fp8_function != null,
    };
    const can_flash_prefill = (head_dim_u32 <= 256) and cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => self.flash_prefill_f16_function != null,
        .i8 => self.flash_prefill_i8_function != null,
        .fp8 => self.flash_prefill_fp8_function != null,
    };
    const can_fused_prefill_attn = cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => attention_kernels.attn_fused_prefill_heads_f16_kv_function != null,
        .i8 => attention_kernels.attn_fused_prefill_heads_i8_kv_function != null,
        .fp8 => attention_kernels.attn_fused_prefill_heads_fp8_kv_function != null,
    };
    const can_any_fused_prefill = can_flash_prefill or can_fused_prefill_attn;
    const can_batched_kv_write_prefill = can_any_fused_prefill and switch (self.kv_cache_dtype) {
        .f16 => kv_write_f16_function != null and self.kv_write_f16_rows_function != null,
        .i8 => self.kv_write_i8_rows_function != null,
        .fp8 => self.kv_write_fp8_rows_function != null,
    };

    if (can_batched_kv_write_prefill) {
        const kv_cache_offset = std.math.mul(usize, @as(usize, position_base_u32), kv_row_bytes) catch return error.InvalidArgument;
        var k_cache_out = k_cache.*;
        k_cache_out.pointer += kv_cache_offset;
        k_cache_out.size -= kv_cache_offset;
        var v_cache_out = v_cache.*;
        v_cache_out.pointer += kv_cache_offset;
        v_cache_out.size -= kv_cache_offset;
        switch (self.kv_cache_dtype) {
            .f16 => {
                try compute.cuda.kv_write_f16_rows.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_f16_rows_function.?,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_cache_out,
                    &v_cache_out,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(stage_rows),
                    @intCast(cfg.kv_dim),
                    position_base_u32,
                    layer_rope_theta,
                );
            },
            .i8 => {
                const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset = std.math.mul(usize, @as(usize, position_base_u32), scale_row_bytes) catch return error.InvalidArgument;
                var k_scale_out = k_scale.*;
                k_scale_out.pointer += scale_offset;
                k_scale_out.size -= scale_offset;
                var v_scale_out = v_scale.*;
                v_scale_out.pointer += scale_offset;
                v_scale_out.size -= scale_offset;
                try compute.cuda.kv_write_i8_rows.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_i8_rows_function.?,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_cache_out,
                    &v_cache_out,
                    &k_scale_out,
                    &v_scale_out,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(stage_rows),
                    @intCast(cfg.kv_dim),
                    position_base_u32,
                    layer_rope_theta,
                );
            },
            .fp8 => {
                const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset = std.math.mul(usize, @as(usize, position_base_u32), scale_row_bytes) catch return error.InvalidArgument;
                var k_scale_out = k_scale.*;
                k_scale_out.pointer += scale_offset;
                k_scale_out.size -= scale_offset;
                var v_scale_out = v_scale.*;
                v_scale_out.pointer += scale_offset;
                v_scale_out.size -= scale_offset;
                try compute.cuda.kv_write_fp8_rows.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    self.kv_write_fp8_rows_function.?,
                    &attn_k_stage,
                    &attn_v_stage,
                    &k_cache_out,
                    &v_cache_out,
                    &k_scale_out,
                    &v_scale_out,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    @intCast(stage_rows),
                    @intCast(cfg.kv_dim),
                    position_base_u32,
                    layer_rope_theta,
                );
            },
        }
    } else {
        var row_idx: usize = 0;
        while (row_idx < stage_rows) : (row_idx += 1) {
            const kv_offset_f32 = std.math.mul(usize, row_idx, kv_row_f32_bytes) catch return error.InvalidArgument;
            var k_row_in = try bufferSlice(&attn_k_stage, kv_offset_f32, kv_row_f32_bytes);
            var v_row_in = try bufferSlice(&attn_v_stage, kv_offset_f32, kv_row_f32_bytes);

            const position_u32: u32 = position_base_u32 + @as(u32, @intCast(row_idx));
            if (!use_k_write_fused) {
                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_function,
                    &k_row_in,
                    n_kv_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            }

            const cache_row = @as(usize, position_base_u32) + row_idx;
            const kv_offset = std.math.mul(usize, cache_row, kv_row_bytes) catch return error.InvalidArgument;
            var k_row_out = try bufferSlice(k_cache, kv_offset, kv_row_bytes);
            var v_row_out = try bufferSlice(v_cache, kv_offset, kv_row_bytes);

            switch (self.kv_cache_dtype) {
                .f16 => {
                    if (kv_write_f16_function) |kv_write_f16| {
                        try compute.cuda.kv_write_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kv_write_f16,
                            &k_row_in,
                            &v_row_in,
                            &k_row_out,
                            &v_row_out,
                            n_kv_heads_u32,
                            head_dim_u32,
                            rope_dim_u32,
                            position_u32,
                            layer_rope_theta,
                        );
                    } else if (rope_store_f16_function) |rope_store_f16| {
                        try compute.cuda.rope_store_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            rope_store_f16,
                            &k_row_in,
                            &k_row_out,
                            n_kv_heads_u32,
                            head_dim_u32,
                            rope_dim_u32,
                            position_u32,
                            layer_rope_theta,
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &v_row_in,
                            &v_row_out,
                            @intCast(cfg.kv_dim),
                        );
                    } else {
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &k_row_in,
                            &k_row_out,
                            @intCast(cfg.kv_dim),
                        );
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &v_row_in,
                            &v_row_out,
                            @intCast(cfg.kv_dim),
                        );
                    }
                },
                .i8 => {
                    const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                    const scale_offset = std.math.mul(usize, cache_row, scale_row_bytes) catch return error.InvalidArgument;
                    var k_scale_row = try bufferSlice(k_scale, scale_offset, scale_row_bytes);
                    var v_scale_row = try bufferSlice(v_scale, scale_offset, scale_row_bytes);
                    try compute.cuda.kv_write_i8.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.kv_write_i8_function orelse return error.CudaKernelUnavailable,
                        &k_row_in,
                        &v_row_in,
                        &k_row_out,
                        &v_row_out,
                        &k_scale_row,
                        &v_scale_row,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                },
                .fp8 => {
                    const scale_row_bytes: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                    const scale_offset = std.math.mul(usize, cache_row, scale_row_bytes) catch return error.InvalidArgument;
                    var k_scale_row = try bufferSlice(k_scale, scale_offset, scale_row_bytes);
                    var v_scale_row = try bufferSlice(v_scale, scale_offset, scale_row_bytes);
                    try compute.cuda.kv_write_fp8.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.kv_write_fp8_function orelse return error.CudaKernelUnavailable,
                        &k_row_in,
                        &v_row_in,
                        &k_row_out,
                        &v_row_out,
                        &k_scale_row,
                        &v_scale_row,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_u32,
                        layer_rope_theta,
                    );
                },
            }
            if (!can_any_fused_prefill) {
                const q_row_bytes = std.math.mul(usize, cfg.q_dim, @sizeOf(f32)) catch return error.InvalidArgument;
                const ctx_row_bytes = std.math.mul(usize, o_proj.rows(), @sizeOf(f32)) catch return error.InvalidArgument;
                const q_offset = std.math.mul(usize, row_idx, q_row_bytes) catch return error.InvalidArgument;
                const ctx_offset = std.math.mul(usize, row_idx, ctx_row_bytes) catch return error.InvalidArgument;
                var q_row = try bufferSlice(&attn_q_stage, q_offset, q_row_bytes);
                var ctx_row = try bufferSlice(&attn_context_stage, ctx_offset, ctx_row_bytes);
                const effective_seq_len_u32: u32 = position_u32 + 1;
                // For query_gate, always apply RoPE on Q — the fused single-head
                // attention path is not used when query_gate is true.
                try compute.cuda.rope.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    rope_function,
                    &q_row,
                    n_heads_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
                _ = try self.runAttentionContext(
                    cfg,
                    &q_row,
                    &ctx_row,
                    read_k_cache,
                    read_v_cache,
                    read_k_scale,
                    read_v_scale,
                    attention_kernels,
                    effective_seq_len_u32,
                    head_dim_u32,
                    kv_dim_u32,
                    kv_groups_u32,
                    n_heads_u32,
                    attention_scale,
                    rope_dim_u32,
                    position_u32,
                    layer_rope_theta,
                );
            }
        }
    }

    if (can_any_fused_prefill) {
        const sliding_window_u32_wq = std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32);
        switch (self.kv_cache_dtype) {
            .f16 => {
                const use_gqa = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_f16_kv_gqa_function != null;

                // GEMM-based attention (see NoQueryGate path for full commentary).
                const can_gemm_attn = use_gqa and
                    attention_kernels.causal_attn_softmax_f32_function != null;

                if (can_flash_prefill) {
                    try compute.cuda.flash_prefill.runF16(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.flash_prefill_f16_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &attn_context_stage,
                        n_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                } else if (can_gemm_attn) {
                    // Apply RoPE to Q (per-row).
                    const q_row_dim_wq = cfg.q_dim;
                    const q_row_bytes_rope_wq = std.math.mul(usize, q_row_dim_wq, @sizeOf(f32)) catch return error.InvalidArgument;
                    var rope_row_idx: usize = 0;
                    while (rope_row_idx < stage_rows) : (rope_row_idx += 1) {
                        const q_row_offset = std.math.mul(usize, rope_row_idx, q_row_bytes_rope_wq) catch return error.InvalidArgument;
                        var q_row = try bufferSlice(&attn_q_stage, q_row_offset, q_row_bytes_rope_wq);
                        const pos = position_base_u32 + @as(u32, @intCast(rope_row_idx));
                        try compute.cuda.rope.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            rope_function,
                            &q_row,
                            n_heads_u32,
                            head_dim_u32,
                            rope_dim_u32,
                            pos,
                            layer_rope_theta,
                        );
                    }

                    var scores_buf = try ensureAttnScoresWorkspace(
                        self,
                        kv_groups_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                    );
                    const q_ld: usize = cfg.q_dim;
                    const kv_ld: usize = cfg.kv_dim;
                    const ctx_ld: usize = o_proj.rows();
                    const hd: usize = head_dim_u32;
                    const sl: usize = seq_len_u32;
                    const qr: usize = stage_rows;
                    const n_kv: u32 = n_heads_u32 / kv_groups_u32;

                    // Cast Q from f32 to f16 for tensor-core GEMM.
                    const q_f16_elems = std.math.mul(usize, qr, cfg.q_dim) catch return error.InvalidArgument;
                    const q_f16_bytes = std.math.mul(usize, q_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
                    const probs_f16_elems = std.math.mul(usize, std.math.mul(usize, kv_groups_u32, qr) catch return error.InvalidArgument, sl) catch return error.InvalidArgument;
                    const probs_f16_bytes = std.math.mul(usize, probs_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
                    var u16_ws = try ensureAttnU16Workspace(self, q_f16_bytes + probs_f16_bytes);
                    var q_f16_buf = try bufferSlice(&u16_ws, 0, q_f16_bytes);
                    var probs_f16_buf = try bufferSlice(&u16_ws, q_f16_bytes, probs_f16_bytes);
                    try compute.cuda.cast_f32_to_f16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                        &attn_q_stage,
                        &q_f16_buf,
                        @intCast(q_f16_elems),
                    );

                    var kv_h: u32 = 0;
                    while (kv_h < n_kv) : (kv_h += 1) {
                        const k_ptr = read_k_cache.pointer + @as(usize, kv_h) * hd * @sizeOf(u16);
                        const v_ptr = read_v_cache.pointer + @as(usize, kv_h) * hd * @sizeOf(u16);
                        const q_f16_ptr = q_f16_buf.pointer + @as(usize, kv_h) * kv_groups_u32 * hd * @sizeOf(u16);
                        const out_ptr = attn_context_stage.pointer + @as(usize, kv_h) * kv_groups_u32 * hd * @sizeOf(f32);

                        // Q × K^T: f16 × f16 → f32.
                        var g: u32 = 0;
                        while (g < kv_groups_u32) : (g += 1) {
                            try self.blas.gemmU16(
                                &self.device,
                                true,
                                sl,
                                qr,
                                hd,
                                attention_scale,
                                k_ptr,
                                kv_ld,
                                q_f16_ptr + @as(usize, g) * hd * @sizeOf(u16),
                                q_ld,
                                0.0,
                                scores_buf.pointer + @as(usize, g) * qr * sl * @sizeOf(f32),
                                sl,
                            );
                        }

                        // Causal mask + softmax (f32).
                        try compute.cuda.causal_attn_softmax_f32.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            attention_kernels.causal_attn_softmax_f32_function.?,
                            &scores_buf,
                            kv_groups_u32 * @as(u32, @intCast(stage_rows)),
                            seq_len_u32,
                            @intCast(stage_rows),
                            position_base_u32,
                            sliding_window_u32_wq,
                        );

                        // Cast probs from f32 to f16 for V GEMM.
                        try compute.cuda.cast_f32_to_f16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            cast_f32_to_f16_function orelse return error.CudaKernelUnavailable,
                            &scores_buf,
                            &probs_f16_buf,
                            @intCast(probs_f16_elems),
                        );

                        // probs × V: f16 × f16 → f32.
                        g = 0;
                        while (g < kv_groups_u32) : (g += 1) {
                            try self.blas.gemmU16(
                                &self.device,
                                false,
                                hd,
                                qr,
                                sl,
                                1.0,
                                v_ptr,
                                kv_ld,
                                probs_f16_buf.pointer + @as(usize, g) * qr * sl * @sizeOf(u16),
                                sl,
                                0.0,
                                out_ptr + @as(usize, g) * hd * @sizeOf(f32),
                                ctx_ld,
                            );
                        }
                    }
                } else if (use_gqa) {
                    try compute.cuda.attn_fused_prefill_heads_f16_kv_gqa.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_f16_kv_gqa_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &attn_context_stage,
                        n_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                } else {
                    try compute.cuda.attn_fused_prefill_heads_f16_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_f16_kv_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &attn_context_stage,
                        n_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                }
            },
            .i8 => {
                const use_gqa_i8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_i8_kv_gqa_function != null;
                var k_scale_attn = try bufferSlice(read_k_scale, 0, read_k_scale.size);
                var v_scale_attn = try bufferSlice(read_v_scale, 0, read_v_scale.size);
                if (can_flash_prefill) {
                    try compute.cuda.flash_prefill.runWithScales(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.flash_prefill_i8_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                } else if (use_gqa_i8) {
                    try compute.cuda.attn_fused_prefill_heads_i8_kv_gqa.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_i8_kv_gqa_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                } else {
                    try compute.cuda.attn_fused_prefill_heads_i8_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_i8_kv_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                }
            },
            .fp8 => {
                const use_gqa_fp8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_fp8_kv_gqa_function != null;
                var k_scale_attn_fp8 = try bufferSlice(read_k_scale, 0, read_k_scale.size);
                var v_scale_attn_fp8 = try bufferSlice(read_v_scale, 0, read_v_scale.size);
                if (can_flash_prefill) {
                    try compute.cuda.flash_prefill.runWithScales(
                        &self.kernel_arg_pack,
                        &self.device,
                        self.flash_prefill_fp8_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                } else if (use_gqa_fp8) {
                    try compute.cuda.attn_fused_prefill_heads_fp8_kv_gqa.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_fp8_kv_gqa_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                } else {
                    try compute.cuda.attn_fused_prefill_heads_fp8_kv.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        attention_kernels.attn_fused_prefill_heads_fp8_kv_function.?,
                        &attn_q_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        &attn_context_stage,
                        n_heads_u32,
                        n_kv_heads_u32,
                        @intCast(stage_rows),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        head_dim_u32,
                        attention_scale,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                    );
                }
            },
        }
    }

    // Apply output gate before O projection.
    try engine_ops.applyQueryGateToContextInPlace(self, stage_rows, cfg.q_dim, cfg.q_projection_dim, n_heads_u32, head_dim_u32);

    engine_ops.linearForwardRows(self, &attn_context_stage, stage_rows, o_proj, output) catch |err| {
        log.warn("inference", "CUDA prefill attention O projection(query-gate) failed", .{
            .rows = stage_rows,
            .in_dim = o_proj.rows(),
            .out_dim = o_proj.cols(),
            .reason = @errorName(err),
        });
        return err;
    };
}

/// Ensure the attention scores workspace is large enough for GEMM-based
/// prefill attention.  Layout: [kv_groups * q_rows * seq_len] f32.
/// Grow-only, never shrinks.
pub fn ensureAttnScoresWorkspace(
    self: anytype,
    kv_groups: u32,
    q_rows: u32,
    seq_len: u32,
) !compute.cuda.Buffer {
    const total_rows = std.math.mul(usize, @as(usize, kv_groups), @as(usize, q_rows)) catch return error.InvalidArgument;
    const total_elems = std.math.mul(usize, total_rows, @as(usize, seq_len)) catch return error.InvalidArgument;
    const required_bytes = std.math.mul(usize, total_elems, @sizeOf(f32)) catch return error.InvalidArgument;

    if (self.attn_scores_workspace_dev == null or self.attn_scores_workspace_dev.?.size < required_bytes) {
        if (self.fixed_alloc_mode) return error.OutOfMemory;
        if (self.attn_scores_workspace_dev) |*buf| buf.deinit(&self.device);
        self.attn_scores_workspace_dev = try self.device.allocBuffer(required_bytes);
    }
    return compute.cuda.Buffer{
        .pointer = self.attn_scores_workspace_dev.?.pointer,
        .size = required_bytes,
    };
}

/// Ensure the u16 workspace for GEMM attention is large enough.
/// Returns a buffer of at least `required_bytes`.
pub fn ensureAttnU16Workspace(self: anytype, required_bytes: usize) !compute.cuda.Buffer {
    if (self.attn_u16_workspace_dev == null or self.attn_u16_workspace_dev.?.size < required_bytes) {
        if (self.fixed_alloc_mode) return error.OutOfMemory;
        if (self.attn_u16_workspace_dev) |*buf| buf.deinit(&self.device);
        self.attn_u16_workspace_dev = try self.device.allocBuffer(required_bytes);
    }
    return compute.cuda.Buffer{
        .pointer = self.attn_u16_workspace_dev.?.pointer,
        .size = required_bytes,
    };
}

pub fn attentionFallbackUsesCache(seq_len: usize) bool {
    return seq_len == 1;
}

fn shouldUseFlashDecodePath(
    kv_groups: u32,
    head_dim: u32,
    n_kv_heads: u32,
    n_rows: u32,
    flash_decode_available: bool,
) bool {
    if (std.posix.getenv("TALU_NO_FLASH_DECODE") != null) return false;
    if (!flash_decode_available) return false;
    if (kv_groups == 0 or kv_groups > 4) return false;
    if (head_dim == 0 or head_dim > 256) return false;
    const flash_blocks = std.math.mul(u32, n_kv_heads, n_rows) catch return false;
    const min_blocks = if (n_kv_heads <= 2) min_flash_decode_blocks_low_kv_heads else min_flash_decode_blocks_default;
    return flash_blocks >= min_blocks;
}

test "shouldUseFlashDecodePath keeps low-kv decode conservative" {
    try std.testing.expect(!shouldUseFlashDecodePath(4, 256, 2, 8, true));
    try std.testing.expect(!shouldUseFlashDecodePath(4, 256, 2, 511, true));
    try std.testing.expect(shouldUseFlashDecodePath(4, 256, 2, 512, true));
}

test "shouldUseFlashDecodePath accepts sufficient parallelism" {
    try std.testing.expect(shouldUseFlashDecodePath(4, 256, 8, 4, true));
    try std.testing.expect(shouldUseFlashDecodePath(2, 128, 16, 2, true));
}

test "shouldUseFlashDecodePath rejects unsupported geometry" {
    try std.testing.expect(!shouldUseFlashDecodePath(8, 256, 8, 8, true));
    try std.testing.expect(!shouldUseFlashDecodePath(4, 512, 8, 8, true));
    try std.testing.expect(!shouldUseFlashDecodePath(4, 256, 8, 8, false));
}

pub fn applyBiasF32(
    self: anytype,
    target: *compute.cuda.Buffer,
    bias: *const DeviceTensor,
    count: u32,
) !void {
    const element_count = std.math.mul(usize, bias.rows, bias.cols) catch return error.InvalidArgument;
    const count_usize: usize = @intCast(count);
    if (element_count != count_usize) return error.InvalidInstructionBinding;
    const expected_bytes = std.math.mul(usize, @as(usize, count), @sizeOf(f32)) catch return error.InvalidArgument;
    if (bias.buffer.size != expected_bytes) return error.InvalidInstructionBinding;
    const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
    try compute.cuda.vector_add.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        vector_add_function,
        target,
        &bias.buffer,
        target,
        count,
    );
}

pub fn runFfnStep(
    self: anytype,
    input: *const compute.cuda.Buffer,
    rows: usize,
    gate_weight: *const LinearWeight,
    up_weight: *const LinearWeight,
    down_weight: *const LinearWeight,
    gate_bias: ?*const DeviceTensor,
    down_bias: ?*const DeviceTensor,
    d_ff: u32,
    output: *compute.cuda.Buffer,
    residual_buf: ?compute.cuda.Buffer,
) !void {
    const activation_count = std.math.mul(u32, @intCast(rows), d_ff) catch return error.InvalidArgument;
    const activation_bytes = std.math.mul(usize, @as(usize, activation_count), @sizeOf(f32)) catch return error.InvalidArgument;
    const prefer_split_i8_gate_up = self.gaffine_u4_decode_i8_enabled and
        self.i8_blas_supported and
        rows >= 4 and rows <= self.max_batch_size and
        engine_ops.linearWeightHasI8Cache(gate_weight) and
        engine_ops.linearWeightHasI8Cache(up_weight);
    const fused_gate_up_silu = gate_bias == null and !prefer_split_i8_gate_up and ((try engine_ops.tryFusedDenseU16GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )) or (try engine_ops.tryFusedMxfp8GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )) or (try engine_ops.tryFusedFp8GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )) or (try engine_ops.tryFusedNvfp4GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )) or (try engine_ops.tryFusedGaffineU4GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )) or (try engine_ops.tryFusedI8GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )) or (try engine_ops.tryFusedGaffineU8GateUpSiluForward(
        self,
        input,
        gate_weight,
        up_weight,
        rows,
        d_ff,
    )));
    if (!fused_gate_up_silu) {
        _ = try engine_ops.runGateUpProjectionWithWeights(self, input, gate_weight, up_weight, rows);
        if (gate_bias) |bias| {
            if (rows != 1) return error.UnsupportedModel;
            try applyBiasF32(self, &self.runtime_buffers.ffn_gate_dev, bias, d_ff);
        }
        try engine_ops.runFfnActivationMul(self, activation_count);
    }
    var mul_in = try bufferSlice(&self.runtime_buffers.ffn_mul_dev, 0, activation_bytes);
    if (residual_buf) |rb| {
        self.pending_residual_add_buf = rb;
    }
    try engine_ops.linearForwardRows(self, &mul_in, rows, down_weight, output);
    if (down_bias) |bias| {
        if (rows != 1) return error.UnsupportedModel;
        const d_model = std.math.cast(u32, down_weight.cols()) orelse return error.InvalidArgument;
        try applyBiasF32(self, output, bias, d_model);
    }
}

//! Attention prefill paths for the CUDA inference backend.

const workspace = @import("workspace.zig");
const ensureAttnScoresWorkspace = workspace.ensureAttnScoresWorkspace;
const ensureAttnU16Workspace = workspace.ensureAttnU16Workspace;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const attention_mod = @import("../../attention_path.zig");
const cpu_kernels = @import("../../../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;

// --- Shared types from engine_types.zig ---
const engine_types = @import("../../runtime/_types_impl.zig");
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
const MoEWeightRefs = engine_types.MoEWeightRefs;
const AttentionPath = engine_types.AttentionPath;
const enable_fused_attention_f16_kv = engine_types.enable_fused_attention_f16_kv;
const max_fused_attention_f16_kv_seq_len = engine_types.max_fused_attention_f16_kv_seq_len;
const max_supported_fused_f16_kv_head_dim = engine_types.max_supported_fused_f16_kv_head_dim;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const attention_policy_config = engine_types.attention_policy_config;
const min_flash_decode_blocks_default: u32 = 8;
const min_flash_decode_blocks_low_kv_heads: u32 = 1024;

fn debugKernelSyncEnabled() bool {
    const raw = @import("env_pkg").getenv("TALU_CUDA_DEBUG_SYNC") orelse return false;
    return std.mem.eql(u8, raw, "1") or std.ascii.eqlIgnoreCase(raw, "true");
}

fn phaseEventTimingEnabled(self: anytype) bool {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "phase_event_timing_enabled")) {
        return self.phase_event_timing_enabled;
    }
    return false;
}

fn recordAttentionPhase(self: anytype, path: AttentionPath, start_ns: i128, is_causal: bool) void {
    const SelfType = @TypeOf(self.*);
    var elapsed_ns: u64 = 0;
    if (comptime @hasField(SelfType, "phase_attention_start_event")) {
        if (phaseEventTimingEnabled(self)) {
            if (self.phase_attention_start_event) |start_evt| {
                if (self.phase_attention_stop_event) |stop_evt| {
                    self.device.recordEvent(stop_evt, self.compute_stream) catch {};
                    self.device.synchronizeEvent(stop_evt) catch {};
                    elapsed_ns = self.device.elapsedEventNs(start_evt, stop_evt) catch 0;
                }
            }
        }
    }
    if (elapsed_ns == 0) {
        const elapsed_i128 = std.time.nanoTimestamp() - start_ns;
        elapsed_ns = if (elapsed_i128 > 0) @intCast(elapsed_i128) else 0;
    }
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordAttention(path, elapsed_ns);
        self.nvfp4_phase_counters.recordAttentionCausality(is_causal);
        self.nvfp4_phase_counters.recordAttentionBatchedPrefill();
    }
}

fn recordDecodeAttentionPhase(self: anytype, path: AttentionPath, start_ns: i128, is_causal: bool) void {
    const SelfType = @TypeOf(self.*);
    var elapsed_ns: u64 = 0;
    if (comptime @hasField(SelfType, "phase_attention_start_event")) {
        if (phaseEventTimingEnabled(self)) {
            if (self.phase_attention_start_event) |start_evt| {
                if (self.phase_attention_stop_event) |stop_evt| {
                    self.device.recordEvent(stop_evt, self.compute_stream) catch {};
                    self.device.synchronizeEvent(stop_evt) catch {};
                    elapsed_ns = self.device.elapsedEventNs(start_evt, stop_evt) catch 0;
                }
            }
        }
    }
    if (elapsed_ns == 0) {
        const elapsed_i128 = std.time.nanoTimestamp() - start_ns;
        elapsed_ns = if (elapsed_i128 > 0) @intCast(elapsed_i128) else 0;
    }
    if (comptime @hasField(SelfType, "nvfp4_phase_counters")) {
        self.nvfp4_phase_counters.recordAttention(path, elapsed_ns);
        self.nvfp4_phase_counters.recordAttentionCausality(is_causal);
    }
}

fn applyPrefillRopeRows(
    self: anytype,
    q_stage: *compute.cuda.Buffer,
    stage_rows: usize,
    row_stride_elems: usize,
    n_heads_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    position_base_u32: u32,
    theta: f32,
    rope_function: compute.cuda.Function,
) !void {
    const expected_row_stride = std.math.mul(usize, @as(usize, n_heads_u32), @as(usize, head_dim_u32)) catch return error.InvalidArgument;
    if (self.rope_rows_ptrs_function) |rope_rows_ptrs_fn| {
        if (row_stride_elems == expected_row_stride) {
            const positions_bytes = std.math.mul(usize, stage_rows, @sizeOf(u32)) catch return error.InvalidArgument;
            var positions_dev = try bufferSlice(&self.runtime_buffers.prefill_tokens_dev, 0, positions_bytes);
            const rows_u32: u32 = @intCast(stage_rows);
            const need_upload = self.prefill_rope_positions_cached_dirty or
                !self.prefill_rope_positions_cached_valid or
                self.prefill_rope_positions_cached_rows != rows_u32 or
                self.prefill_rope_positions_cached_base != position_base_u32;
            if (need_upload) {
                if (self.prefill_rope_positions_host.len < stage_rows) {
                    if (self.prefill_rope_positions_host.len > 0) self.allocator.free(self.prefill_rope_positions_host);
                    self.prefill_rope_positions_host = try self.allocator.alloc(u32, stage_rows);
                }
                const positions_host = self.prefill_rope_positions_host[0..stage_rows];
                var row: usize = 0;
                while (row < stage_rows) : (row += 1) {
                    positions_host[row] = position_base_u32 + @as(u32, @intCast(row));
                }
                try positions_dev.upload(&self.device, std.mem.sliceAsBytes(positions_host));
                self.prefill_rope_positions_cached_valid = true;
                self.prefill_rope_positions_cached_dirty = false;
                self.prefill_rope_positions_cached_rows = rows_u32;
                self.prefill_rope_positions_cached_base = position_base_u32;
            }
            try compute.cuda.rope_rows_ptrs.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rope_rows_ptrs_fn,
                q_stage,
                &positions_dev,
                @intCast(stage_rows),
                n_heads_u32,
                head_dim_u32,
                rope_dim_u32,
                theta,
            );
            return;
        }
    }

    // Fallback for non-standard row stride layouts.
    const q_row_bytes_rope = std.math.mul(usize, row_stride_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    var rope_row_idx: usize = 0;
    while (rope_row_idx < stage_rows) : (rope_row_idx += 1) {
        const q_row_offset = std.math.mul(usize, rope_row_idx, q_row_bytes_rope) catch return error.InvalidArgument;
        var q_row = try bufferSlice(q_stage, q_row_offset, q_row_bytes_rope);
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
            theta,
        );
    }
}

const LowBitKvKind = enum {
    i8,
    fp8,
};

fn runPrefillAttentionLowBitViaF16Gemm(
    self: anytype,
    kind: LowBitKvKind,
    attn_q_stage: *compute.cuda.Buffer,
    attn_context_stage: *compute.cuda.Buffer,
    read_k_cache: *const compute.cuda.Buffer,
    read_v_cache: *const compute.cuda.Buffer,
    read_k_scale: *const compute.cuda.Buffer,
    read_v_scale: *const compute.cuda.Buffer,
    q_rows: usize,
    q_row_stride: usize,
    ctx_row_stride: usize,
    seq_len_u32: u32,
    kv_dim_u32: u32,
    kv_groups_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    position_base_u32: u32,
    sliding_window_u32: u32,
    layer_rope_theta: f32,
    attention_scale: f32,
    rope_function: compute.cuda.Function,
    cast_f32_to_f16_function: ?compute.cuda.Function,
    causal_softmax_f32_fn: compute.cuda.Function,
) !void {
    const cast_f32_to_f16_fn = cast_f32_to_f16_function orelse self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable;
    const n_kv: u32 = n_heads_u32 / kv_groups_u32;
    const q_f16_elems = std.math.mul(usize, q_rows, q_row_stride) catch return error.InvalidArgument;
    const probs_f16_elems = std.math.mul(usize, std.math.mul(usize, n_kv, q_rows) catch return error.InvalidArgument, seq_len_u32) catch return error.InvalidArgument;
    const kv_f16_elems = std.math.mul(usize, @as(usize, seq_len_u32), @as(usize, kv_dim_u32)) catch return error.InvalidArgument;

    const q_f16_bytes = std.math.mul(usize, q_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const probs_f16_bytes = std.math.mul(usize, probs_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
    const kv_f16_bytes = std.math.mul(usize, kv_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;

    var u16_ws = try ensureAttnU16Workspace(
        self,
        q_f16_bytes + probs_f16_bytes + kv_f16_bytes + kv_f16_bytes,
    );
    var q_f16_buf = try bufferSlice(&u16_ws, 0, q_f16_bytes);
    var probs_f16_buf = try bufferSlice(&u16_ws, q_f16_bytes, probs_f16_bytes);
    var k_f16_buf = try bufferSlice(&u16_ws, q_f16_bytes + probs_f16_bytes, kv_f16_bytes);
    var v_f16_buf = try bufferSlice(&u16_ws, q_f16_bytes + probs_f16_bytes + kv_f16_bytes, kv_f16_bytes);

    switch (kind) {
        .i8 => {
            try compute.cuda.dequant_kv_i8_to_f16.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.dequant_kv_i8_to_f16_function orelse return error.CudaKernelUnavailable,
                read_k_cache,
                read_v_cache,
                read_k_scale,
                read_v_scale,
                &k_f16_buf,
                &v_f16_buf,
                seq_len_u32,
                n_kv_heads_u32,
                kv_dim_u32,
                head_dim_u32,
            );
        },
        .fp8 => {
            try compute.cuda.dequant_kv_fp8_to_f16.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.dequant_kv_fp8_to_f16_function orelse return error.CudaKernelUnavailable,
                read_k_cache,
                read_v_cache,
                read_k_scale,
                read_v_scale,
                &k_f16_buf,
                &v_f16_buf,
                seq_len_u32,
                n_kv_heads_u32,
                kv_dim_u32,
                head_dim_u32,
            );
        },
    }

    try applyPrefillRopeRows(
        self,
        attn_q_stage,
        q_rows,
        q_row_stride,
        n_heads_u32,
        head_dim_u32,
        rope_dim_u32,
        position_base_u32,
        layer_rope_theta,
        rope_function,
    );

    try compute.cuda.cast_f32_to_f16.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        cast_f32_to_f16_fn,
        attn_q_stage,
        &q_f16_buf,
        @intCast(q_f16_elems),
    );

    var scores_buf = try ensureAttnScoresWorkspace(
        self,
        n_kv,
        @intCast(q_rows),
        seq_len_u32,
    );

    const q_ld: usize = q_row_stride;
    const kv_ld: usize = kv_dim_u32;
    const hd: usize = head_dim_u32;
    const sl: usize = seq_len_u32;
    const qr: usize = q_rows;

    var group_idx: u32 = 0;
    while (group_idx < kv_groups_u32) : (group_idx += 1) {
        const q_f16_ptr = q_f16_buf.pointer + @as(usize, group_idx) * hd * @sizeOf(u16);
        const out_ptr = attn_context_stage.pointer + @as(usize, group_idx) * hd * @sizeOf(f32);

        try self.blas.gemmU16StridedBatched(
            &self.device,
            true,
            sl,
            qr,
            hd,
            attention_scale,
            k_f16_buf.pointer,
            kv_ld,
            hd,
            q_f16_ptr,
            q_ld,
            kv_groups_u32 * hd,
            0.0,
            scores_buf.pointer,
            sl,
            qr * sl,
            n_kv,
        );

        try compute.cuda.causal_attn_softmax_f32.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            causal_softmax_f32_fn,
            &scores_buf,
            n_kv * @as(u32, @intCast(q_rows)),
            seq_len_u32,
            @intCast(q_rows),
            position_base_u32,
            sliding_window_u32,
        );

        try compute.cuda.cast_f32_to_f16.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            cast_f32_to_f16_fn,
            &scores_buf,
            &probs_f16_buf,
            @intCast(probs_f16_elems),
        );

        try self.blas.gemmU16StridedBatched(
            &self.device,
            false,
            hd,
            qr,
            sl,
            1.0,
            v_f16_buf.pointer,
            kv_ld,
            hd,
            probs_f16_buf.pointer,
            sl,
            qr * sl,
            0.0,
            out_ptr,
            ctx_row_stride,
            kv_groups_u32 * hd,
            n_kv,
        );
    }
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

    // Some proportional RoPE configurations define frequencies over global_head_dim,
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
const engine_ops = @import("../root.zig");

// --- Forward pass from engine_forward.zig ---
const engine_forward = @import("../../exec/common.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../../weights/root.zig");
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
    const attention_kernel_applies_q_rope = (!cfg.query_gate) and attention_mod.useFusedHeadsF16Kv(
        attention_policy_config,
        seq_len_u32,
        cfg.sliding_window,
        cfg.is_causal,
        head_dim_u32,
        attention_kernels.attn_fused_heads_f16_kv_function != null,
    );
    if (!attention_kernel_applies_q_rope) {
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
    const can_lowbit_gemm_prefill = cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => false,
        .i8 => self.dequant_kv_i8_to_f16_function != null and (cast_f32_to_f16_function != null or self.cast_f32_to_f16_function != null) and
            ((attention_kernels.causal_attn_softmax_f32_function != null) or (self.causal_attn_softmax_f32_function != null)),
        .fp8 => self.dequant_kv_fp8_to_f16_function != null and (cast_f32_to_f16_function != null or self.cast_f32_to_f16_function != null) and
            ((attention_kernels.causal_attn_softmax_f32_function != null) or (self.causal_attn_softmax_f32_function != null)),
    };
    const can_any_fused_prefill = can_flash_prefill or can_fused_prefill_attn or can_lowbit_gemm_prefill;
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
                const attention_context_path_applies_q_rope = attention_mod.useFusedHeadsF16Kv(
                    attention_policy_config,
                    effective_seq_len_u32,
                    cfg.sliding_window,
                    cfg.is_causal,
                    head_dim_u32,
                    attention_kernels.attn_fused_heads_f16_kv_function != null,
                );
                if (!attention_context_path_applies_q_rope) {
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
        const attention_start_ns: i128 = std.time.nanoTimestamp();
        if (comptime @hasField(@TypeOf(self.*), "phase_attention_start_event")) {
            if (phaseEventTimingEnabled(self)) {
                if (self.phase_attention_start_event) |start_evt| {
                    self.device.recordEvent(start_evt, self.compute_stream) catch {};
                }
            }
        }
        var attention_path: AttentionPath = .heads_f32_kv;
        const sliding_window_u32 = std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32);
        switch (self.kv_cache_dtype) {
            .f16 => {
                const use_gqa = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_f16_kv_gqa_function != null;

                // Current f16 prefill route contract:
                // Keep GEMM attention as the selected baseline route in this branch.
                // Flash/fused prefill kernels remain as non-selected alternatives.
                const causal_softmax_f32_fn = attention_kernels.causal_attn_softmax_f32_function orelse self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
                const use_gemm_prefill_attention = true;
                if (use_gemm_prefill_attention) {
                    // Apply RoPE to Q for all rows (each row has a different position).
                    // Use batched row kernel when row stride matches n_heads*head_dim.
                    try applyPrefillRopeRows(
                        self,
                        &attn_q_stage,
                        stage_rows,
                        cfg.q_projection_dim,
                        n_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_base_u32,
                        layer_rope_theta,
                        rope_function,
                    );

                    // GEMM-based attention per KV head.
                    const n_kv: u32 = n_heads_u32 / kv_groups_u32;
                    var scores_buf = try ensureAttnScoresWorkspace(
                        self,
                        n_kv,
                        @intCast(stage_rows),
                        seq_len_u32,
                    );
                    const q_ld: usize = cfg.q_projection_dim;
                    const kv_ld: usize = cfg.kv_dim;
                    const ctx_ld: usize = o_proj.rows();
                    const hd: usize = head_dim_u32;
                    const sl: usize = seq_len_u32;
                    const qr: usize = stage_rows;
                    const q_f16_elems = std.math.mul(usize, qr, cfg.q_projection_dim) catch return error.InvalidArgument;
                    const q_f16_bytes = std.math.mul(usize, q_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
                    const probs_f16_elems = std.math.mul(usize, std.math.mul(usize, n_kv, qr) catch return error.InvalidArgument, sl) catch return error.InvalidArgument;
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

                    var group_idx: u32 = 0;
                    while (group_idx < kv_groups_u32) : (group_idx += 1) {
                        const k_ptr = read_k_cache.pointer;
                        const v_ptr = read_v_cache.pointer;
                        const q_f16_ptr = q_f16_buf.pointer + @as(usize, group_idx) * hd * @sizeOf(u16);
                        const out_ptr = attn_context_stage.pointer + @as(usize, group_idx) * hd * @sizeOf(f32);

                        // Q × K^T over all KV heads for this query group.
                        // Batch dimension is n_kv (each batch = one KV head).
                        try self.blas.gemmU16StridedBatched(
                            &self.device,
                            true, // transa=T for K
                            sl, // m = seq_len
                            qr, // n = q_rows
                            hd, // k = head_dim
                            attention_scale,
                            k_ptr,
                            kv_ld, // lda = kv_dim
                            hd, // strideA in u16 elements (next KV head)
                            q_f16_ptr,
                            q_ld, // ldb = q_projection_dim
                            kv_groups_u32 * hd, // strideB in u16 elements (next KV head in this group)
                            0.0,
                            scores_buf.pointer,
                            sl, // ldc = seq_len
                            qr * sl, // strideC in f32 elements
                            n_kv,
                        );

                        // Causal mask + softmax over [n_kv * q_rows, seq_len].
                        try compute.cuda.causal_attn_softmax_f32.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            causal_softmax_f32_fn,
                            &scores_buf,
                            n_kv * @as(u32, @intCast(stage_rows)),
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

                        // probs × V over all KV heads for this query group.
                        try self.blas.gemmU16StridedBatched(
                            &self.device,
                            false, // transa=N for V
                            hd, // m = head_dim
                            qr, // n = q_rows
                            sl, // k = seq_len
                            1.0,
                            v_ptr,
                            kv_ld, // lda = kv_dim
                            hd, // strideA in u16 elements (next KV head)
                            probs_f16_buf.pointer,
                            sl, // ldb = seq_len
                            qr * sl, // strideB in u16 elements
                            0.0,
                            out_ptr,
                            ctx_ld, // ldc = context_dim
                            kv_groups_u32 * hd, // strideC in f32 elements (next KV head in this group)
                            n_kv,
                        );
                    }
                    attention_path = .heads_f16_kv;
                } else if (can_flash_prefill) {
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
                    attention_path = .fused_heads_f16_kv;
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
                    attention_path = .fused_heads_f16_kv;
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
                    attention_path = .fused_heads_f16_kv;
                }
            },
            .i8 => {
                const use_gqa_i8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_i8_kv_gqa_function != null;
                const scale_row_bytes_attn: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset_attn = std.math.mul(usize, 0, scale_row_bytes_attn) catch return error.InvalidArgument;
                var k_scale_attn = try bufferSlice(read_k_scale, scale_offset_attn, read_k_scale.size);
                var v_scale_attn = try bufferSlice(read_v_scale, scale_offset_attn, read_v_scale.size);
                if (can_lowbit_gemm_prefill) {
                    const causal_softmax_f32_fn = attention_kernels.causal_attn_softmax_f32_function orelse
                        self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
                    try runPrefillAttentionLowBitViaF16Gemm(
                        self,
                        .i8,
                        &attn_q_stage,
                        &attn_context_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        stage_rows,
                        cfg.q_projection_dim,
                        o_proj.rows(),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        n_heads_u32,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                        attention_scale,
                        rope_function,
                        cast_f32_to_f16_function,
                        causal_softmax_f32_fn,
                    );
                    attention_path = .heads_lowbit_bridge_f16_kv;
                } else if (can_flash_prefill) {
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
                    attention_path = .heads_i8_kv;
                } else if (use_gqa_i8) {
                    const gqa_i8_ok = blk: {
                        compute.cuda.attn_fused_prefill_heads_i8_kv_gqa.runWithFunction(
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
                        ) catch |err| {
                            if (err == error.CudaKernelLaunchFailed or err == error.InvalidArgument) {
                                log.warn("inference", "CUDA fused prefill i8 GQA launch failed; falling back to non-GQA fused prefill", .{
                                    .q_rows = stage_rows,
                                    .seq_len = seq_len_u32,
                                    .head_dim = head_dim_u32,
                                    .kv_groups = kv_groups_u32,
                                    .rope_dim = rope_dim_u32,
                                    .reason = @errorName(err),
                                });
                                break :blk false;
                            }
                            return err;
                        };
                        break :blk true;
                    };
                    if (!gqa_i8_ok) {
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
                    attention_path = .fused_heads_i8_kv;
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
                    attention_path = .fused_heads_i8_kv;
                }
            },
            .fp8 => {
                const use_gqa_fp8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_fp8_kv_gqa_function != null;
                const scale_row_bytes_attn_fp8: usize = @as(usize, n_kv_heads_u32) * @sizeOf(f32);
                const scale_offset_attn_fp8 = std.math.mul(usize, 0, scale_row_bytes_attn_fp8) catch return error.InvalidArgument;
                var k_scale_attn_fp8 = try bufferSlice(read_k_scale, scale_offset_attn_fp8, read_k_scale.size);
                var v_scale_attn_fp8 = try bufferSlice(read_v_scale, scale_offset_attn_fp8, read_v_scale.size);
                if (can_lowbit_gemm_prefill) {
                    const causal_softmax_f32_fn = attention_kernels.causal_attn_softmax_f32_function orelse
                        self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
                    try runPrefillAttentionLowBitViaF16Gemm(
                        self,
                        .fp8,
                        &attn_q_stage,
                        &attn_context_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        stage_rows,
                        cfg.q_projection_dim,
                        o_proj.rows(),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        n_heads_u32,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32,
                        layer_rope_theta,
                        attention_scale,
                        rope_function,
                        cast_f32_to_f16_function,
                        causal_softmax_f32_fn,
                    );
                    attention_path = .heads_lowbit_bridge_f16_kv;
                } else if (can_flash_prefill) {
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
                    attention_path = .heads_fp8_kv;
                } else if (use_gqa_fp8) {
                    const gqa_fp8_ok = blk: {
                        compute.cuda.attn_fused_prefill_heads_fp8_kv_gqa.runWithFunction(
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
                        ) catch |err| {
                            if (err == error.CudaKernelLaunchFailed or err == error.InvalidArgument) {
                                log.warn("inference", "CUDA fused prefill fp8 GQA launch failed; falling back to non-GQA fused prefill", .{
                                    .q_rows = stage_rows,
                                    .seq_len = seq_len_u32,
                                    .head_dim = head_dim_u32,
                                    .kv_groups = kv_groups_u32,
                                    .rope_dim = rope_dim_u32,
                                    .reason = @errorName(err),
                                });
                                break :blk false;
                            }
                            return err;
                        };
                        break :blk true;
                    };
                    if (!gqa_fp8_ok) {
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
                    attention_path = .fused_heads_fp8_kv;
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
                    attention_path = .fused_heads_fp8_kv;
                }
            },
        }
        recordAttentionPhase(self, attention_path, attention_start_ns, cfg.is_causal);
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
    const can_lowbit_gemm_prefill = cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => false,
        .i8 => self.dequant_kv_i8_to_f16_function != null and (cast_f32_to_f16_function != null or self.cast_f32_to_f16_function != null) and
            ((attention_kernels.causal_attn_softmax_f32_function != null) or (self.causal_attn_softmax_f32_function != null)),
        .fp8 => self.dequant_kv_fp8_to_f16_function != null and (cast_f32_to_f16_function != null or self.cast_f32_to_f16_function != null) and
            ((attention_kernels.causal_attn_softmax_f32_function != null) or (self.causal_attn_softmax_f32_function != null)),
    };
    const can_any_fused_prefill = can_flash_prefill or can_fused_prefill_attn or can_lowbit_gemm_prefill;
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
        const attention_start_ns: i128 = std.time.nanoTimestamp();
        if (comptime @hasField(@TypeOf(self.*), "phase_attention_start_event")) {
            if (phaseEventTimingEnabled(self)) {
                if (self.phase_attention_start_event) |start_evt| {
                    self.device.recordEvent(start_evt, self.compute_stream) catch {};
                }
            }
        }
        var attention_path: AttentionPath = .heads_f32_kv;
        const sliding_window_u32_wq = std.math.cast(u32, cfg.sliding_window) orelse std.math.maxInt(u32);
        switch (self.kv_cache_dtype) {
            .f16 => {
                const use_gqa = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_f16_kv_gqa_function != null;

                // Current f16 prefill route contract:
                // Keep GEMM attention as the selected baseline route in this branch.
                // Flash/fused prefill kernels remain as non-selected alternatives.
                const causal_softmax_f32_fn = attention_kernels.causal_attn_softmax_f32_function orelse self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
                const use_gemm_prefill_attention = true;
                if (use_gemm_prefill_attention) {
                    // Apply RoPE to Q for all rows.
                    try applyPrefillRopeRows(
                        self,
                        &attn_q_stage,
                        stage_rows,
                        cfg.q_dim,
                        n_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_base_u32,
                        layer_rope_theta,
                        rope_function,
                    );

                    const n_kv: u32 = n_heads_u32 / kv_groups_u32;
                    var scores_buf = try ensureAttnScoresWorkspace(
                        self,
                        n_kv,
                        @intCast(stage_rows),
                        seq_len_u32,
                    );
                    const q_ld: usize = cfg.q_dim;
                    const kv_ld: usize = cfg.kv_dim;
                    const ctx_ld: usize = o_proj.rows();
                    const hd: usize = head_dim_u32;
                    const sl: usize = seq_len_u32;
                    const qr: usize = stage_rows;
                    const q_f16_elems = std.math.mul(usize, qr, cfg.q_dim) catch return error.InvalidArgument;
                    const q_f16_bytes = std.math.mul(usize, q_f16_elems, @sizeOf(u16)) catch return error.InvalidArgument;
                    const probs_f16_elems = std.math.mul(usize, std.math.mul(usize, n_kv, qr) catch return error.InvalidArgument, sl) catch return error.InvalidArgument;
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

                    var group_idx: u32 = 0;
                    while (group_idx < kv_groups_u32) : (group_idx += 1) {
                        const k_ptr = read_k_cache.pointer;
                        const v_ptr = read_v_cache.pointer;
                        const q_f16_ptr = q_f16_buf.pointer + @as(usize, group_idx) * hd * @sizeOf(u16);
                        const out_ptr = attn_context_stage.pointer + @as(usize, group_idx) * hd * @sizeOf(f32);

                        // Q × K^T over all KV heads for this query group.
                        try self.blas.gemmU16StridedBatched(
                            &self.device,
                            true,
                            sl,
                            qr,
                            hd,
                            attention_scale,
                            k_ptr,
                            kv_ld,
                            hd,
                            q_f16_ptr,
                            q_ld,
                            kv_groups_u32 * hd,
                            0.0,
                            scores_buf.pointer,
                            sl,
                            qr * sl,
                            n_kv,
                        );

                        // Causal mask + softmax over [n_kv * q_rows, seq_len].
                        try compute.cuda.causal_attn_softmax_f32.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            causal_softmax_f32_fn,
                            &scores_buf,
                            n_kv * @as(u32, @intCast(stage_rows)),
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

                        // probs × V over all KV heads for this query group.
                        try self.blas.gemmU16StridedBatched(
                            &self.device,
                            false,
                            hd,
                            qr,
                            sl,
                            1.0,
                            v_ptr,
                            kv_ld,
                            hd,
                            probs_f16_buf.pointer,
                            sl,
                            qr * sl,
                            0.0,
                            out_ptr,
                            ctx_ld,
                            kv_groups_u32 * hd,
                            n_kv,
                        );
                    }
                    attention_path = .heads_f16_kv;
                } else if (can_flash_prefill) {
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
                    attention_path = .fused_heads_f16_kv;
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
                    attention_path = .fused_heads_f16_kv;
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
                    attention_path = .fused_heads_f16_kv;
                }
            },
            .i8 => {
                const use_gqa_i8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_i8_kv_gqa_function != null;
                var k_scale_attn = try bufferSlice(read_k_scale, 0, read_k_scale.size);
                var v_scale_attn = try bufferSlice(read_v_scale, 0, read_v_scale.size);
                if (can_lowbit_gemm_prefill) {
                    const causal_softmax_f32_fn = attention_kernels.causal_attn_softmax_f32_function orelse
                        self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
                    try runPrefillAttentionLowBitViaF16Gemm(
                        self,
                        .i8,
                        &attn_q_stage,
                        &attn_context_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn,
                        &v_scale_attn,
                        stage_rows,
                        cfg.q_dim,
                        o_proj.rows(),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        n_heads_u32,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                        attention_scale,
                        rope_function,
                        cast_f32_to_f16_function,
                        causal_softmax_f32_fn,
                    );
                    attention_path = .heads_lowbit_bridge_f16_kv;
                } else if (can_flash_prefill) {
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
                    attention_path = .heads_i8_kv;
                } else if (use_gqa_i8) {
                    const gqa_i8_ok = blk: {
                        compute.cuda.attn_fused_prefill_heads_i8_kv_gqa.runWithFunction(
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
                        ) catch |err| {
                            if (err == error.CudaKernelLaunchFailed or err == error.InvalidArgument) {
                                log.warn("inference", "CUDA fused prefill i8 GQA launch failed; falling back to non-GQA fused prefill", .{
                                    .q_rows = stage_rows,
                                    .seq_len = seq_len_u32,
                                    .head_dim = head_dim_u32,
                                    .kv_groups = kv_groups_u32,
                                    .rope_dim = rope_dim_u32,
                                    .reason = @errorName(err),
                                });
                                break :blk false;
                            }
                            return err;
                        };
                        break :blk true;
                    };
                    if (!gqa_i8_ok) {
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
                    attention_path = .fused_heads_i8_kv;
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
                    attention_path = .fused_heads_i8_kv;
                }
            },
            .fp8 => {
                const use_gqa_fp8 = kv_groups_u32 >= 2 and
                    attention_kernels.attn_fused_prefill_heads_fp8_kv_gqa_function != null;
                var k_scale_attn_fp8 = try bufferSlice(read_k_scale, 0, read_k_scale.size);
                var v_scale_attn_fp8 = try bufferSlice(read_v_scale, 0, read_v_scale.size);
                if (can_lowbit_gemm_prefill) {
                    const causal_softmax_f32_fn = attention_kernels.causal_attn_softmax_f32_function orelse
                        self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
                    try runPrefillAttentionLowBitViaF16Gemm(
                        self,
                        .fp8,
                        &attn_q_stage,
                        &attn_context_stage,
                        read_k_cache,
                        read_v_cache,
                        &k_scale_attn_fp8,
                        &v_scale_attn_fp8,
                        stage_rows,
                        cfg.q_dim,
                        o_proj.rows(),
                        seq_len_u32,
                        kv_dim_u32,
                        kv_groups_u32,
                        n_heads_u32,
                        n_kv_heads_u32,
                        head_dim_u32,
                        rope_dim_u32,
                        position_base_u32,
                        sliding_window_u32_wq,
                        layer_rope_theta,
                        attention_scale,
                        rope_function,
                        cast_f32_to_f16_function,
                        causal_softmax_f32_fn,
                    );
                    attention_path = .heads_lowbit_bridge_f16_kv;
                } else if (can_flash_prefill) {
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
                    attention_path = .heads_fp8_kv;
                } else if (use_gqa_fp8) {
                    const gqa_fp8_ok = blk: {
                        compute.cuda.attn_fused_prefill_heads_fp8_kv_gqa.runWithFunction(
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
                        ) catch |err| {
                            if (err == error.CudaKernelLaunchFailed or err == error.InvalidArgument) {
                                log.warn("inference", "CUDA fused prefill fp8 GQA launch failed; falling back to non-GQA fused prefill", .{
                                    .q_rows = stage_rows,
                                    .seq_len = seq_len_u32,
                                    .head_dim = head_dim_u32,
                                    .kv_groups = kv_groups_u32,
                                    .rope_dim = rope_dim_u32,
                                    .reason = @errorName(err),
                                });
                                break :blk false;
                            }
                            return err;
                        };
                        break :blk true;
                    };
                    if (!gqa_fp8_ok) {
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
                    attention_path = .fused_heads_fp8_kv;
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
                    attention_path = .fused_heads_fp8_kv;
                }
            },
        }
        recordAttentionPhase(self, attention_path, attention_start_ns, cfg.is_causal);
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

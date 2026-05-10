//! Attention decode paths for the CUDA inference backend.

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
const engine_types = @import("../../runtime/root.zig");
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
    const can_lowbit_gemm_decode = cfg.is_causal and switch (self.kv_cache_dtype) {
        .f16 => false,
        .i8 => self.dequant_kv_i8_to_f16_function != null and
            (ctx.cast_f32_to_f16_function != null or self.cast_f32_to_f16_function != null) and
            (self.causal_attn_softmax_f32_function != null),
        .fp8 => self.dequant_kv_fp8_to_f16_function != null and
            (ctx.cast_f32_to_f16_function != null or self.cast_f32_to_f16_function != null) and
            (self.causal_attn_softmax_f32_function != null),
    };
    const flash_decode_available = switch (self.kv_cache_dtype) {
        .f16 => self.flash_decode_f16_function != null,
        .i8 => self.flash_decode_i8_function != null,
        .fp8 => self.flash_decode_fp8_function != null,
    };
    var max_seq_len_u32: u32 = 0;
    for (batch.seq_lens[0..n]) |len_u32| {
        max_seq_len_u32 = @max(max_seq_len_u32, len_u32);
    }
    // Flash decode is better when there is enough parallel work from
    // (n_kv_heads × batch_rows). Do not force low-bit single-row flash:
    // on Blackwell this has regressed vs pointer-table decode for NVFP4 flows.
    const allow_lowbit_single_row_flash = false;
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
    // On Blackwell SM120, low-bit decode has shown better end-to-end latency
    // with pointer-table separate decode than fused ptrs at rows==1.
    // Keep fused ptrs available as an explicit alternate route, but do not prefer it by default.
    const prefer_batched_fused_decode = switch (self.kv_cache_dtype) {
        .i8, .fp8 => false,
        .f16 => false,
    };
    const use_batched_fused_decode_path = use_batched_fused_decode_attention and
        (prefer_batched_fused_decode or !use_batched_separate_decode_attention);
    const use_batched_separate_decode_path = use_batched_separate_decode_attention and
        !use_batched_fused_decode_path;
    // Low-bit GEMM bridge dequantizes KV to f16 each decode step, so prefer it
    // only when native low-bit decode kernels are absent.
    const use_lowbit_gemm_decode_path = can_lowbit_gemm_decode and n_rows == 1 and
        !use_flash_decode and !use_batched_separate_decode_path and !use_batched_fused_decode_path;
    if (@import("env_pkg").getenv("TALU_CUDA_LOG_DECODE_PATH") != null) {
        const route = if (use_lowbit_gemm_decode_path)
            "lowbit_gemm_bridge"
        else if (use_flash_decode)
            "flash"
        else if (use_batched_fused_decode_path)
            "fused_ptrs"
        else if (use_batched_separate_decode_path)
            "separate_ptrs"
        else
            "row_loop";
        const route_id: u32 = if (use_lowbit_gemm_decode_path)
            5
        else if (use_flash_decode)
            1
        else if (use_batched_fused_decode_path)
            2
        else if (use_batched_separate_decode_path)
            3
        else
            4;
        log.warn("inference", "CUDA decode attention route", .{
            .kv_dtype = @tagName(self.kv_cache_dtype),
            .rows = n_rows,
            .head_dim = ctx.head_dim_u32,
            .n_heads = ctx.n_heads_u32,
            .n_kv_heads = ctx.n_kv_heads_u32,
            .route = route,
            .route_id = route_id,
            .prefer_batched_fused_decode_u32 = @as(u32, @intFromBool(prefer_batched_fused_decode)),
            .allow_lowbit_single_row_flash_u32 = @as(u32, @intFromBool(allow_lowbit_single_row_flash)),
            .n_rows_eq_1_u32 = @as(u32, @intFromBool(n_rows == 1)),
            .lowbit_gemm_bridge_u32 = @as(u32, @intFromBool(use_lowbit_gemm_decode_path)),
            .flash_u32 = @as(u32, @intFromBool(use_flash_decode)),
            .separate_u32 = @as(u32, @intFromBool(use_batched_separate_decode_path)),
            .fused_u32 = @as(u32, @intFromBool(use_batched_fused_decode_path)),
            .flash_available_u32 = @as(u32, @intFromBool(flash_decode_available)),
            .separate_available_u32 = @as(u32, @intFromBool(use_batched_separate_decode_attention)),
            .fused_available_u32 = @as(u32, @intFromBool(use_batched_fused_decode_attention)),
            .lowbit_gemm_decode_available_u32 = @as(u32, @intFromBool(can_lowbit_gemm_decode)),
        });
    }
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
        const attn_layers = self.block_runtime.attention_block_count;
        if (read_slot_kv_index < attn_layers) {
            // Fast path: source KV is another local attention layer.
            // Reuse the prebuilt device pointer tables and avoid per-layer
            // host table rebuild/upload in decode.
            const read_row_offset = std.math.mul(usize, read_slot_kv_index, batch.attn_ptrs_row_stride) catch return error.InvalidArgument;
            const read_byte_offset = std.math.mul(usize, read_row_offset, @sizeOf(u64)) catch return error.InvalidArgument;
            read_key_cache_ptrs_dev = try bufferSlice(batch.attn_key_cache_ptrs_table_dev, read_byte_offset, ptr_bytes);
            read_value_cache_ptrs_dev = try bufferSlice(batch.attn_value_cache_ptrs_table_dev, read_byte_offset, ptr_bytes);
            switch (self.kv_cache_dtype) {
                .f16 => {},
                .i8, .fp8 => {
                    read_k_scale_ptrs_dev = try bufferSlice(batch.attn_k_scale_ptrs_table_dev, read_byte_offset, ptr_bytes);
                    read_v_scale_ptrs_dev = try bufferSlice(batch.attn_v_scale_ptrs_table_dev, read_byte_offset, ptr_bytes);
                },
            }
        } else {
            // Cross-device mirror source: table storage is not layer-indexed in
            // decode_attn_*_ptrs_table_dev, so build one compact pointer row.
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
    }

    // Flash decode preferred (GQA-aware, fewest KV reads), then batched
    // separate (graph-compatible), then fused.
    const use_batched_decode_attention = use_lowbit_gemm_decode_path or use_flash_decode or use_batched_separate_decode_path or use_batched_fused_decode_path;
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
            const decode_attention_path_applies_q_rope = use_flash_decode or use_batched_fused_decode_attention or
                ((!cfg.query_gate) and attention_mod.useFusedHeadsF16Kv(
                    attention_policy_config,
                    seq_len_u32,
                    cfg.sliding_window,
                    cfg.is_causal,
                    ctx.head_dim_u32,
                    ctx.attention_kernels.attn_fused_heads_f16_kv_function != null,
                ));

            // RoPE on Q (per-row position).
            if (!decode_attention_path_applies_q_rope) {
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

    const record_decode_attention = use_lowbit_gemm_decode_path or use_flash_decode or use_batched_separate_decode_path or use_batched_fused_decode_path;
    const decode_attention_start_ns: i128 = if (record_decode_attention) std.time.nanoTimestamp() else 0;

    if (use_lowbit_gemm_decode_path) {
        const causal_softmax_f32_fn = self.causal_attn_softmax_f32_function orelse return error.CudaKernelUnavailable;
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
            switch (self.kv_cache_dtype) {
                .f16 => unreachable,
                .i8 => try runPrefillAttentionLowBitViaF16Gemm(
                    self,
                    .i8,
                    &q_row,
                    &context_row,
                    &read_entry.k,
                    &read_entry.v,
                    &read_entry.k_scale,
                    &read_entry.v_scale,
                    1,
                    cfg.q_dim,
                    o_proj.rows(),
                    seq_len_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    position_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    attention_scale,
                    ctx.rope_function,
                    ctx.cast_f32_to_f16_function,
                    causal_softmax_f32_fn,
                ),
                .fp8 => try runPrefillAttentionLowBitViaF16Gemm(
                    self,
                    .fp8,
                    &q_row,
                    &context_row,
                    &read_entry.k,
                    &read_entry.v,
                    &read_entry.k_scale,
                    &read_entry.v_scale,
                    1,
                    cfg.q_dim,
                    o_proj.rows(),
                    seq_len_u32,
                    kv_dim_u32,
                    kv_groups,
                    ctx.n_heads_u32,
                    ctx.n_kv_heads_u32,
                    ctx.head_dim_u32,
                    ctx.rope_dim_u32,
                    position_u32,
                    sliding_window_u32,
                    layer_rope_theta,
                    attention_scale,
                    ctx.rope_function,
                    ctx.cast_f32_to_f16_function,
                    causal_softmax_f32_fn,
                ),
            }
        }
        recordDecodeAttentionPhase(self, .heads_lowbit_bridge_f16_kv, decode_attention_start_ns, cfg.is_causal);
    } else if (use_flash_decode) {
        // Flash decode with split-K: partition sequence across blocks for
        // occupancy when n_kv_heads is small. GQA-aware, does RoPE on Q
        // internally, reads KV once per KV head for all grouped Q heads.
        const gate_proj: ?*const compute.cuda.Buffer = if (cfg.query_gate) &q_projection_stage else null;
        const gate_proj_stride: u32 = if (cfg.query_gate) @intCast(cfg.q_projection_dim) else 0;
        const prefer_split_k = switch (self.kv_cache_dtype) {
            // For single-row decode, split-K often adds more launch/reduce
            // overhead than it saves; keep low-bit flash in single-chunk mode.
            .i8, .fp8 => n_rows > 1,
            .f16 => false,
        };
        const n_seq_chunks = compute.cuda.flash_decode.computeSeqChunks(
            ctx.n_kv_heads_u32,
            n_rows,
            max_seq_len_u32,
            prefer_split_k,
        );

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
        const decode_path: AttentionPath = switch (self.kv_cache_dtype) {
            .f16 => .heads_f16_kv,
            .i8 => .heads_i8_kv,
            .fp8 => .heads_fp8_kv,
        };
        recordDecodeAttentionPhase(self, decode_path, decode_attention_start_ns, cfg.is_causal);
    } else if (use_batched_separate_decode_path) {
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
        const decode_path: AttentionPath = switch (self.kv_cache_dtype) {
            .f16 => .heads_f16_kv,
            .i8 => .heads_i8_kv,
            .fp8 => .heads_fp8_kv,
        };
        recordDecodeAttentionPhase(self, decode_path, decode_attention_start_ns, cfg.is_causal);
    } else if (use_batched_fused_decode_path) {
        // Fused path: single kernel does RoPE + scores + softmax +
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
        const decode_path: AttentionPath = switch (self.kv_cache_dtype) {
            .f16 => .fused_heads_f16_kv,
            .i8 => .fused_heads_i8_kv,
            .fp8 => .fused_heads_fp8_kv,
        };
        recordDecodeAttentionPhase(self, decode_path, decode_attention_start_ns, cfg.is_causal);
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
    const fused_path_active = use_flash_decode or use_batched_fused_decode_path;
    if (cfg.query_gate and !fused_path_active) {
        try engine_ops.applyQueryGateToContextInPlace(self, n, cfg.q_dim, cfg.q_projection_dim, ctx.n_heads_u32, ctx.head_dim_u32);
    }

    // Step 5: O projection GEMM for all N rows.
    try engine_ops.linearForwardRows(self, &attn_context_stage, n, o_proj, output);
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
    if (@import("env_pkg").getenv("TALU_NO_FLASH_DECODE") != null) return false;
    if (!flash_decode_available) return false;
    // Single-row decode does not provide enough row-level parallelism to
    // amortize split-K flash decode overhead; separate decode attention is
    // consistently faster for this shape.
    if (n_rows <= 1) return false;
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
    try std.testing.expect(!shouldUseFlashDecodePath(2, 128, 16, 1, true));
    try std.testing.expect(shouldUseFlashDecodePath(4, 256, 8, 4, true));
    try std.testing.expect(shouldUseFlashDecodePath(2, 128, 16, 2, true));
}

test "shouldUseFlashDecodePath rejects unsupported geometry" {
    try std.testing.expect(!shouldUseFlashDecodePath(8, 256, 8, 8, true));
    try std.testing.expect(!shouldUseFlashDecodePath(4, 512, 8, 8, true));
    try std.testing.expect(!shouldUseFlashDecodePath(4, 256, 8, 8, false));
}

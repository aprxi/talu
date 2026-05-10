//! Gated-delta operator helpers for the CUDA inference backend.

const workspace = @import("attention/workspace.zig");
const ensureAttnScoresWorkspace = workspace.ensureAttnScoresWorkspace;
const ensureAttnU16Workspace = workspace.ensureAttnU16Workspace;
const shortconv = @import("shortconv.zig");
const downloadRowsF32StrideAware = shortconv.downloadRowsF32StrideAware;

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const dtype = @import("compute_pkg").dtype;
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const attention_mod = @import("../attention_path.zig");
const cpu_kernels = @import("../../cpu/kernels/root.zig");
const cpu_conv1d = compute.cpu.conv1d_depthwise;
const cpu_gated_delta = compute.cpu.gated_delta;

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
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
const engine_ops = @import("root.zig");

// --- Forward pass from engine_forward.zig ---
const engine_forward = @import("../exec/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const resizeScratchBuffer = engine_weights.resizeScratchBuffer;

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

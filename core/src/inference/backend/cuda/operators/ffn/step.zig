//! FFN execution steps for the CUDA inference backend.

const workspace = @import("../attention/workspace.zig");
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

    // Scalar row path for non-standard row stride layouts.
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
    return engine_types.resolveRuntimeAttentionScale(self.loaded.config, self.attention_scale, @intCast(head_dim_u32));
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
    )) or (try engine_ops.tryFusedNvfp4GateUpGeluForward(
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

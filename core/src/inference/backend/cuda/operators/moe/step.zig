//! Mixture-of-experts execution for the CUDA inference backend.

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

pub fn runMoEFusedStep(
    self: anytype,
    input: *const compute.cuda.Buffer,
    rows: u32,
    moe: *const MoEWeightRefs,
    output: *compute.cuda.Buffer,
) !void {
    const d_model: u32 = @intCast(self.loaded.config.d_model);
    const d_model_bytes: usize = @as(usize, d_model) * @sizeOf(f32);
    const shared_d_ff = moe.shared_d_ff;
    const expert_d_ff = moe.expert_d_ff;
    const num_experts = moe.num_experts;
    const experts_per_token = moe.experts_per_token;
    const norm_weight_offset = self.loaded.runtime.weight_offset;
    const rmsnorm_fn = self.rmsnorm_function orelse return error.CudaKernelUnavailable;
    const vector_add_fn = self.vector_add_function orelse return error.CudaKernelUnavailable;
    const vector_add_scaled_fn = self.vector_add_scaled_function orelse return error.CudaKernelUnavailable;

    // Select the configured activation kernel for expert FFNs.
    const act_fn = if (moe.use_gelu)
        (self.gelu_mul_function orelse return error.CudaKernelUnavailable)
    else
        (self.silu_mul_function orelse return error.CudaKernelUnavailable);

    const max_experts = 256;
    const max_topk = 16;
    if (num_experts > max_experts or experts_per_token > max_topk) return error.UnsupportedModel;
    if (experts_per_token == 0) return error.InvalidArgument;

    var router_logits_host: [max_experts]f32 = undefined;
    var selected_indices: [max_topk]u32 = undefined;
    var selected_weights: [max_topk]f32 = undefined;
    var per_expert_scale_host: [max_experts]f32 = undefined;

    const ne: usize = @as(usize, num_experts);
    const ept: usize = @as(usize, experts_per_token);

    // Download per-expert scales once when present.
    if (moe.router_per_expert_scale) |pes| {
        try pes.buffer.download(
            &self.device,
            std.mem.sliceAsBytes(per_expert_scale_host[0..ne]),
        );
    }

    const shared_d_ff_bytes: usize = @as(usize, shared_d_ff) * @sizeOf(f32);
    const expert_gate_up_bytes: usize = @as(usize, 2 * expert_d_ff) * @sizeOf(f32);
    const expert_d_ff_bytes: usize = @as(usize, expert_d_ff) * @sizeOf(f32);
    const router_logits_bytes: usize = ne * @sizeOf(f32);

    for (0..@as(usize, rows)) |row_idx| {
        const row_offset: usize = row_idx * d_model_bytes;
        var input_row = try bufferSlice(input, row_offset, d_model_bytes);
        var output_row = try bufferSlice(output, row_offset, d_model_bytes);

        // ===== 1. Shared MLP =====
        // Input to shared MLP: RMSNorm(input) if pre_ffn_norm present, else input directly
        var shared_mlp_input = &input_row;
        if (moe.pre_ffn_norm) |pfn| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_fn,
                &input_row,
                &pfn.buffer,
                &self.runtime_buffers.norm_out_dev,
                1,
                d_model,
                self.norm_eps,
                norm_weight_offset,
            );
            shared_mlp_input = &self.runtime_buffers.norm_out_dev;
        }

        // gate_proj(input) → ffn_gate_dev
        var gate_out = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, shared_d_ff_bytes);
        try engine_ops.linearForwardRows(self, shared_mlp_input, 1, &moe.shared_gate, &gate_out);

        // up_proj(input) → ffn_up_dev
        var up_out = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, shared_d_ff_bytes);
        try engine_ops.linearForwardRows(self, shared_mlp_input, 1, &moe.shared_up, &up_out);

        // act(gate) * up → ffn_mul_dev
        var mul_out = try bufferSlice(&self.runtime_buffers.ffn_mul_dev, 0, shared_d_ff_bytes);
        if (moe.use_gelu) {
            try compute.cuda.gelu_mul.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                act_fn,
                &gate_out,
                &up_out,
                &mul_out,
                shared_d_ff,
            );
        } else {
            try compute.cuda.silu_mul.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                act_fn,
                &gate_out,
                &up_out,
                &mul_out,
                shared_d_ff,
            );
        }

        // down_proj(mul) → ffn_down_dev
        try engine_ops.linearForwardRows(self, &mul_out, 1, &moe.shared_down, &self.runtime_buffers.ffn_down_dev);

        // Optional post-shared norm.
        if (moe.post_shared_norm) |psn| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_fn,
                &self.runtime_buffers.ffn_down_dev,
                &psn.buffer,
                &self.runtime_buffers.ffn_down_dev,
                1,
                d_model,
                self.norm_eps,
                norm_weight_offset,
            );
        }

        // Optional sigmoid gate for shared expert output.
        if (moe.shared_expert_gate) |seg| {
            // Compute dot(input, gate_weight) on GPU via matmul: [1, d_model] x [d_model, 1] -> [1, 1]
            var scalar_dev = try bufferSlice(&self.runtime_buffers.ffn_up_dev, 0, @sizeOf(f32));
            try engine_ops.linearForwardRows(self, &input_row, 1, &seg, &scalar_dev);
            // Download scalar, compute sigmoid on host
            var gate_scalar: [1]f32 = undefined;
            try scalar_dev.download(&self.device, std.mem.sliceAsBytes(&gate_scalar));
            const gate_value: f32 = 1.0 / (1.0 + @exp(-gate_scalar[0]));
            // Scale shared output: ffn_down_dev *= gate_value
            try compute.cuda.vector_add_scaled.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                vector_add_scaled_fn,
                &self.runtime_buffers.ffn_down_dev,
                &self.runtime_buffers.ffn_down_dev,
                &self.runtime_buffers.ffn_down_dev,
                gate_value - 1.0,
                d_model,
            );
        }

        // ===== 2. Router =====
        // Router input may be normalized and scaled before projection.
        if (moe.router_input_scale) |ris| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_fn,
                &input_row,
                &ris.buffer,
                &self.runtime_buffers.norm_out_dev,
                1,
                d_model,
                self.norm_eps,
                0.0,
            );
            if (moe.router_scalar != 1.0) {
                try compute.cuda.vector_add_scaled.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_scaled_fn,
                    &self.runtime_buffers.norm_out_dev,
                    &self.runtime_buffers.norm_out_dev,
                    &self.runtime_buffers.norm_out_dev,
                    moe.router_scalar - 1.0,
                    d_model,
                );
            }
        }
        const router_input = if (moe.router_input_scale != null) &self.runtime_buffers.norm_out_dev else &input_row;

        // Router projection → logits
        var router_logits_dev = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, router_logits_bytes);
        try engine_ops.linearForwardRows(self, router_input, 1, &moe.router_proj, &router_logits_dev);

        // Download logits to host for softmax + topk
        try router_logits_dev.download(
            &self.device,
            std.mem.sliceAsBytes(router_logits_host[0..ne]),
        );

        // Softmax
        {
            var max_logit: f32 = -std.math.inf(f32);
            for (router_logits_host[0..ne]) |l| {
                if (l > max_logit) max_logit = l;
            }
            var sum_exp: f32 = 0.0;
            for (router_logits_host[0..ne]) |*l| {
                l.* = @exp(l.* - max_logit);
                sum_exp += l.*;
            }
            if (sum_exp > 0.0) {
                const inv_sum = 1.0 / sum_exp;
                for (router_logits_host[0..ne]) |*l| l.* *= inv_sum;
            }
        }

        // Top-k selection (greedy)
        for (0..ept) |k| {
            var best_idx: u32 = 0;
            var best_val: f32 = -std.math.inf(f32);
            for (router_logits_host[0..ne], 0..) |val, i| {
                if (val > best_val) {
                    var already = false;
                    for (selected_indices[0..k]) |prev| {
                        if (prev == @as(u32, @intCast(i))) {
                            already = true;
                            break;
                        }
                    }
                    if (!already) {
                        best_val = val;
                        best_idx = @intCast(i);
                    }
                }
            }
            selected_indices[k] = best_idx;
            selected_weights[k] = best_val;
        }

        // Renormalize selected weights
        {
            var weight_sum: f32 = 0.0;
            for (selected_weights[0..ept]) |w| weight_sum += w;
            if (weight_sum > 0.0) {
                const inv_sum = 1.0 / weight_sum;
                for (selected_weights[0..ept]) |*w| w.* *= inv_sum;
            }
        }

        // Apply per-expert scale when present.
        if (moe.router_per_expert_scale != null) {
            for (selected_indices[0..ept], selected_weights[0..ept]) |idx, *w| {
                if (idx < num_experts) {
                    w.* *= per_expert_scale_host[@as(usize, idx)];
                }
            }
        }

        // ===== 3. Expert path =====
        // Expert input may be normalized before projection.
        var expert_input = &input_row;
        if (moe.pre_expert_norm) |pen| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_fn,
                &input_row,
                &pen.buffer,
                &self.runtime_buffers.norm_out_dev,
                1,
                d_model,
                self.norm_eps,
                norm_weight_offset,
            );
            expert_input = &self.runtime_buffers.norm_out_dev;
        }

        // Run selected experts and accumulate weighted output
        for (0..ept) |e_idx| {
            const expert_id = selected_indices[e_idx];
            const weight = selected_weights[e_idx];
            if (expert_id >= num_experts) continue;

            // gate_up_proj(input) → ffn_gate_dev [2*expert_d_ff]
            var expert_gate_up = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, expert_gate_up_bytes);
            try engine_ops.linearForwardRows(self, expert_input, 1, &moe.expert_gate_up[@as(usize, expert_id)], &expert_gate_up);

            // Split: gate = [0..d_ff], up = [d_ff..2*d_ff]
            var expert_gate = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, 0, expert_d_ff_bytes);
            var expert_up = try bufferSlice(&self.runtime_buffers.ffn_gate_dev, expert_d_ff_bytes, expert_d_ff_bytes);

            // act(gate) * up → ffn_mul_dev
            var expert_mul = try bufferSlice(&self.runtime_buffers.ffn_mul_dev, 0, expert_d_ff_bytes);
            if (moe.use_gelu) {
                try compute.cuda.gelu_mul.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    act_fn,
                    &expert_gate,
                    &expert_up,
                    &expert_mul,
                    expert_d_ff,
                );
            } else {
                try compute.cuda.silu_mul.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    act_fn,
                    &expert_gate,
                    &expert_up,
                    &expert_mul,
                    expert_d_ff,
                );
            }

            // down_proj(mul) → deepstack_add_dev (temp, d_model-sized)
            try engine_ops.linearForwardRows(self, &expert_mul, 1, &moe.expert_down[@as(usize, expert_id)], &self.runtime_buffers.deepstack_add_dev);

            // Accumulate: first expert bootstraps accum, rest add
            if (e_idx == 0) {
                try compute.cuda.vector_add_scaled.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_scaled_fn,
                    &self.runtime_buffers.deepstack_add_dev,
                    &self.runtime_buffers.deepstack_add_dev,
                    &self.runtime_buffers.attn_out_dev,
                    weight - 1.0,
                    d_model,
                );
            } else {
                try compute.cuda.vector_add_scaled.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    vector_add_scaled_fn,
                    &self.runtime_buffers.attn_out_dev,
                    &self.runtime_buffers.deepstack_add_dev,
                    &self.runtime_buffers.attn_out_dev,
                    weight,
                    d_model,
                );
            }
        }

        // Optional post-expert norm.
        if (moe.post_expert_norm) |pon| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_fn,
                &self.runtime_buffers.attn_out_dev,
                &pon.buffer,
                &self.runtime_buffers.attn_out_dev,
                1,
                d_model,
                self.norm_eps,
                norm_weight_offset,
            );
        }

        // ===== 4. Combine: output = shared_out + expert_accum =====
        try compute.cuda.vector_add.runWithFunction(
            &self.kernel_arg_pack,
            &self.device,
            vector_add_fn,
            &self.runtime_buffers.ffn_down_dev,
            &self.runtime_buffers.attn_out_dev,
            &output_row,
            d_model,
        );

        // Optional post-combine norm.
        if (moe.post_combine_norm) |pcn| {
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                rmsnorm_fn,
                &output_row,
                &pcn.buffer,
                &output_row,
                1,
                d_model,
                self.norm_eps,
                norm_weight_offset,
            );
        }
    }
}

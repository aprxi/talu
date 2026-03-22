//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("../../../compute/root.zig");
const tensor = @import("../../../tensor.zig");
const log = @import("../../../log.zig");
const trace = @import("../../../xray/trace.zig");
const staged_orchestrator = @import("../staged_orchestrator.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const kv_cache_dtype_fp16 = engine_types.kv_cache_dtype_fp16;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const AttentionKernelSet = engine_types.AttentionKernelSet;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("engine_ops.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("engine_weights.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;
const tryPopulateHiddenFromToken = engine_weights.tryPopulateHiddenFromToken;

const saturatingU64FromU128 = engine_types.saturatingU64FromU128;
const logicalF32RowSlice = engine_types.logicalF32RowSlice;

fn topologyModeTag(self: anytype) ?[]const u8 {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "topology_mode")) return null;
    return @tagName(self.topology_mode);
}

fn topologyModeIs(self: anytype, comptime expected: []const u8) bool {
    const tag = topologyModeTag(self) orelse return false;
    return std.mem.eql(u8, tag, expected);
}

fn typeHasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

fn executeCpuStage0LayerRange(
    stage0: anytype,
    token: u32,
    position: usize,
    slot_index: usize,
    split_layer: usize,
    ensure_kv_capacity: bool,
) !void {
    const Stage0Type = @TypeOf(stage0.*);
    if (comptime !typeHasDecl(Stage0Type, "computePrototypeLogitsWithLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try stage0.computePrototypeLogitsWithLayerRange(
        token,
        position,
        slot_index,
        null,
        0,
        split_layer,
        false,
        false,
        ensure_kv_capacity,
        false,
    );
}

fn pipelineActivationByteCountFor(self: anytype) !usize {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "pipelineActivationByteCount")) {
        return self.pipelineActivationByteCount();
    }
    return std.math.mul(usize, self.d_model, @sizeOf(f32)) catch error.InvalidArgument;
}

/// Transfer multiple rows of pipeline activations from self.input_dev to dst.input_dev.
/// For P2P: single bulk memcpyPeerAsync + sync.
/// For host-staged: loops row-by-row through the small staging buffer, using
/// bufferSlice offsets into each engine's input_dev.
fn transferPipelineActivationMultiRow(self: anytype, dst: anytype, total_bytes: usize, row_bytes: usize) !void {
    if (total_bytes == 0) return;
    switch (self.pipeline_transfer_mode) {
        .peer_to_peer => {
            try self.device.memcpyPeerAsync(
                dst.runtime_buffers.input_dev.pointer,
                dst.device.context,
                self.runtime_buffers.input_dev.pointer,
                self.device.context,
                total_bytes,
                self.compute_stream,
            );
            if (self.compute_stream) |stream| {
                try self.device.synchronizeStream(stream);
            } else {
                try self.device.synchronize();
            }
        },
        .host_staged => {
            const staging = self.pipeline_host_staging orelse return error.PipelineTransferNotInitialized;
            // Staging buffer is sized for one row. Transfer row-by-row.
            if (row_bytes > staging.len) return error.PipelineTransferBufferTooSmall;
            var offset: usize = 0;
            while (offset < total_bytes) {
                const chunk = @min(row_bytes, total_bytes - offset);
                var src_slice = try bufferSlice(&self.runtime_buffers.input_dev, offset, chunk);
                try src_slice.download(&self.device, staging[0..chunk]);
                var dst_slice = try bufferSlice(&dst.runtime_buffers.input_dev, offset, chunk);
                try dst_slice.upload(&dst.device, staging[0..chunk]);
                offset += chunk;
            }
        },
        .none => return error.InvalidTopologyConfig,
    }
}

/// Batched prefill for pipeline2 topology. Processes all tokens through stage0
/// layers in chunks, bulk-transfers activations to stage1, then processes through
/// stage1 layers. Eliminates the per-token sync/transfer of the old path.
fn computeBatchedPrefillPipeline2(
    self: anytype,
    stage1: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    if (comptime !@hasField(@TypeOf(stage1.*), "device")) return error.InvalidArgument;

    const previous_launch_phase0 = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase0);
    const previous_launch_phase1 = stage1.device.setLaunchPhase(.prefill);
    defer _ = stage1.device.setLaunchPhase(previous_launch_phase1);

    const total_rows = tokens.len;

    try ensureKvCapacity(self, total_rows);
    try ensureKvCapacity(stage1, total_rows);
    try resetShortConvStates(self);
    try resetShortConvStates(stage1);
    resetAttentionCpuStates(self);
    resetAttentionCpuStates(stage1);
    resetGatedDeltaStates(self);
    resetGatedDeltaStates(stage1);

    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const d_model_u32: u32 = @intCast(self.d_model);
    const head_dim_u32: u32 = @intCast(self.head_dim);
    const rope_dim_u32: u32 = @intCast(self.rope_dim);
    const n_heads_u32: u32 = @intCast(self.n_heads);
    const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;

    const attn_kernels_0 = AttentionKernelSet{
        .attn_scores_heads_f32_function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable),
        .attn_weighted_sum_heads_f32_function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable),
        .attn_scores_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null,
        .softmax_rows_function = self.softmax_rows_function,
        .attn_weighted_sum_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            (self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null,
        .attn_fused_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            self.attn_fused_heads_f16_kv_function
        else
            null,
        .attn_fused_prefill_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            self.attn_fused_prefill_heads_f16_kv_function
        else
            null,
        .attn_fused_prefill_heads_f16_kv_gqa_function = if (kv_cache_dtype_fp16)
            self.attn_fused_prefill_heads_f16_kv_gqa_function
        else
            null,
        .causal_attn_softmax_f32_function = if (kv_cache_dtype_fp16)
            self.causal_attn_softmax_f32_function
        else
            null,
    };
    const attn_kernels_1 = AttentionKernelSet{
        .attn_scores_heads_f32_function = if (kv_cache_dtype_fp16)
            null
        else
            (stage1.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable),
        .attn_weighted_sum_heads_f32_function = if (kv_cache_dtype_fp16)
            null
        else
            (stage1.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable),
        .attn_scores_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            (stage1.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null,
        .softmax_rows_function = stage1.softmax_rows_function,
        .attn_weighted_sum_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            (stage1.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null,
        .attn_fused_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            stage1.attn_fused_heads_f16_kv_function
        else
            null,
        .attn_fused_prefill_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            stage1.attn_fused_prefill_heads_f16_kv_function
        else
            null,
        .attn_fused_prefill_heads_f16_kv_gqa_function = if (kv_cache_dtype_fp16)
            stage1.attn_fused_prefill_heads_f16_kv_gqa_function
        else
            null,
        .causal_attn_softmax_f32_function = if (kv_cache_dtype_fp16)
            stage1.causal_attn_softmax_f32_function
        else
            null,
    };

    const chunk_cap = @min(self.prefill_chunk_rows_cap, stage1.prefill_chunk_rows_cap);

    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, chunk_cap);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];

        // ── Stage 0: embedding + layers on GPU0 ──
        try self.runtime_buffers.ensureRowCapacity(&self.device, rows, self.fixed_alloc_mode);
        try self.ensureLayerProgramSlotRowCapacity(rows, self.fixed_alloc_mode);

        // Embedding lookup on stage0 (duplicated from single-GPU prefill path).
        var used_device_lookup = false;
        if (enable_device_embedding_lookup and self.runtime_buffers.embedding_lookup != null) {
            const lookup = &self.runtime_buffers.embedding_lookup.?;
            switch (lookup.kind) {
                .f16, .bf16 => {
                    if (self.embedding_lookup_u16_rows_function) |kernel| {
                        const token_bytes = std.math.mul(usize, rows, @sizeOf(u32)) catch return error.InvalidArgument;
                        var token_ids_dev = try bufferSlice(&self.runtime_buffers.prefill_tokens_dev, 0, token_bytes);
                        try token_ids_dev.upload(&self.device, std.mem.sliceAsBytes(chunk_tokens));
                        try compute.cuda.embedding_lookup_u16_rows.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &self.runtime_buffers.input_dev,
                            &lookup.buffer,
                            &token_ids_dev,
                            @intCast(rows),
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            lookup.layout_tag,
                            switch (lookup.kind) {
                                .f16 => compute.cuda.embedding_lookup_u16_rows.dtype_f16,
                                .bf16 => compute.cuda.embedding_lookup_u16_rows.dtype_bf16,
                                else => unreachable,
                            },
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                else => {},
            }

            if (!used_device_lookup) {
                var device_lookup_ok = true;
                var row_idx: usize = 0;
                fill_rows: while (row_idx < chunk_tokens.len) : (row_idx += 1) {
                    const row_offset = std.math.mul(usize, row_idx, row_bytes) catch return error.InvalidArgument;
                    var input_row = try bufferSlice(&self.runtime_buffers.input_dev, row_offset, row_bytes);
                    const token = chunk_tokens[row_idx];
                    switch (lookup.kind) {
                        .f32 => {
                            if (self.embedding_lookup_f32_function) |kernel| {
                                try compute.cuda.embedding_lookup_f32.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    lookup.dim0,
                                    lookup.dim1,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.layout_tag,
                                    lookup.multiplier,
                                );
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                        .f16 => {
                            if (self.embedding_lookup_u16_function) |kernel| {
                                try compute.cuda.embedding_lookup_u16.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    lookup.dim0,
                                    lookup.dim1,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.layout_tag,
                                    compute.cuda.embedding_lookup_u16.dtype_f16,
                                    lookup.multiplier,
                                );
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                        .bf16 => {
                            if (self.embedding_lookup_u16_function) |kernel| {
                                try compute.cuda.embedding_lookup_u16.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    lookup.dim0,
                                    lookup.dim1,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.layout_tag,
                                    compute.cuda.embedding_lookup_u16.dtype_bf16,
                                    lookup.multiplier,
                                );
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                        .gaffine_u4 => {
                            if (self.embedding_lookup_gaffine_u4_function) |kernel| {
                                if (lookup.scales) |*scales_buf| {
                                    if (lookup.biases) |*biases_buf| {
                                        try compute.cuda.embedding_lookup_gaffine_u4.runWithFunction(
                                            &self.kernel_arg_pack,
                                            &self.device,
                                            kernel,
                                            &input_row,
                                            &lookup.buffer,
                                            scales_buf,
                                            biases_buf,
                                            lookup.dim0,
                                            lookup.hidden_dim,
                                            token,
                                            lookup.group_size,
                                            lookup.scales_dtype_tag,
                                            lookup.multiplier,
                                        );
                                    } else {
                                        device_lookup_ok = false;
                                        break :fill_rows;
                                    }
                                } else {
                                    device_lookup_ok = false;
                                    break :fill_rows;
                                }
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                    }
                }
                used_device_lookup = device_lookup_ok;
            }
        }

        if (!used_device_lookup) {
            const hidden_count = std.math.mul(usize, rows, self.d_model) catch return error.InvalidArgument;
            const hidden_host = try self.allocator.alloc(f32, hidden_count);
            defer self.allocator.free(hidden_host);
            try populatePrefillHiddenFromTokens(self.loaded, chunk_tokens, self.d_model, hidden_host, null);
            try self.runtime_buffers.input_dev.upload(&self.device, std.mem.sliceAsBytes(hidden_host));
        }

        const active_rows_u32: u32 = @intCast(rows);
        const seq_len_u32: u32 = @intCast(pos_base + rows);
        const last_position = pos_base + rows - 1;
        const last_position_u32: u32 = @intCast(last_position);

        // Stage 0 layer loop.
        {
            var layer_idx: usize = 0;
            while (layer_idx < self.block_runtime.blocks.len) : (layer_idx += 1) {
                const layer = &self.block_runtime.blocks[layer_idx];
                _ = try self.tryExecuteLayerProgram(
                    layer,
                    slot_index,
                    layer_idx,
                    d_model_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    n_heads_u32,
                    n_kv_heads_u32,
                    active_rows_u32,
                    seq_len_u32,
                    seq_len_u32,
                    pos_base,
                    last_position,
                    last_position_u32,
                    global_rope_theta,
                    local_rope_theta,
                    self.rope_function orelse return error.CudaKernelUnavailable,
                    self.copy_function orelse return error.CudaKernelUnavailable,
                    if (kv_cache_dtype_fp16)
                        (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
                    else
                        null,
                    if (kv_cache_dtype_fp16) self.kv_write_f16_function else null,
                    if (kv_cache_dtype_fp16) self.rope_store_f16_function else null,
                    self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_0,
                    null,
                );
            }
        }

        // ── Stage 1 buffer setup (must precede transfer into input_dev) ──
        try stage1.runtime_buffers.ensureRowCapacity(&stage1.device, rows, stage1.fixed_alloc_mode);
        try stage1.ensureLayerProgramSlotRowCapacity(rows, stage1.fixed_alloc_mode);

        // ── Bulk transfer stage0 → stage1 ──
        const transfer_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
        try transferPipelineActivationMultiRow(self, stage1, transfer_bytes, row_bytes);

        var stage1_final_hidden = stage1.runtime_buffers.input_dev;
        {
            var layer_idx: usize = 0;
            while (layer_idx < stage1.block_runtime.blocks.len) : (layer_idx += 1) {
                const layer = &stage1.block_runtime.blocks[layer_idx];
                stage1_final_hidden = try stage1.tryExecuteLayerProgram(
                    layer,
                    slot_index,
                    layer_idx,
                    d_model_u32,
                    head_dim_u32,
                    rope_dim_u32,
                    n_heads_u32,
                    n_kv_heads_u32,
                    active_rows_u32,
                    seq_len_u32,
                    seq_len_u32,
                    pos_base,
                    last_position,
                    last_position_u32,
                    global_rope_theta,
                    local_rope_theta,
                    stage1.rope_function orelse return error.CudaKernelUnavailable,
                    stage1.copy_function orelse return error.CudaKernelUnavailable,
                    if (kv_cache_dtype_fp16)
                        (stage1.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
                    else
                        null,
                    if (kv_cache_dtype_fp16) stage1.kv_write_f16_function else null,
                    if (kv_cache_dtype_fp16) stage1.rope_store_f16_function else null,
                    stage1.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_1,
                    null,
                );
            }
        }

        // ── Logits from last chunk (on stage1) ──
        if (pos_base + rows >= total_rows) {
            const last_row_in_chunk = rows - 1;
            const last_offset = std.math.mul(usize, last_row_in_chunk, row_bytes) catch return error.InvalidArgument;
            var last_hidden = try bufferSlice(&stage1_final_hidden, last_offset, row_bytes);
            var last_norm = try bufferSlice(&stage1.runtime_buffers.norm_out_dev, 0, row_bytes);
            try compute.cuda.rmsnorm.runWithFunction(
                &stage1.kernel_arg_pack,
                &stage1.device,
                stage1.rmsnorm_function orelse return error.CudaKernelUnavailable,
                &last_hidden,
                &stage1.runtime_buffers.norm_weight_dev,
                &last_norm,
                1,
                @intCast(stage1.d_model),
                stage1.norm_eps,
                stage1.loaded.runtime.weight_offset,
            );
            if (trace.isEnabled()) {
                try last_norm.download(&stage1.device, std.mem.sliceAsBytes(stage1.runtime_buffers.hidden_host));
                trace.emitFinal(
                    .final_norm,
                    @intCast(last_position),
                    1,
                    @ptrCast(stage1.runtime_buffers.hidden_host.ptr),
                    .f32,
                    .{ @intCast(stage1.d_model), 0, 0, 0 },
                    1,
                    "cuda_pipeline2_final_norm_host",
                );
            }

            try engine_ops.linearForwardRows(stage1, &last_norm, 1, &stage1.runtime_buffers.projection_weight, &stage1.runtime_buffers.logits_dev);
            try stage1.runtime_buffers.logits_dev.download(&stage1.device, std.mem.sliceAsBytes(stage1.runtime_buffers.projected_logits_host));
            if (trace.isEnabled()) {
                const rows128: u128 = 1;
                const d_model128: u128 = @intCast(stage1.d_model);
                const vocab128: u128 = @intCast(stage1.runtime_buffers.projected_vocab);
                const total_flops = saturatingU64FromU128(2 * rows128 * d_model128 * vocab128);
                const total_bytes_lm = saturatingU64FromU128(
                    rows128 * d_model128 * @sizeOf(f32) +
                        @as(u128, stage1.runtime_buffers.projection_weight.byteSize()) +
                        rows128 * vocab128 * @sizeOf(f32),
                );
                const kernel_name = switch (stage1.runtime_buffers.projection_weight) {
                    .dense_f32 => "matmul_lm_head_f32_host",
                    .dense_u16 => |w| switch (w.dtype) {
                        .bf16 => "matmul_lm_head_bf16_host",
                        .f16 => "matmul_lm_head_f16_host",
                    },
                    .gaffine_u4 => "matmul_lm_head_gaffine_u4_host",
                    .gaffine_u8 => "matmul_lm_head_gaffine_u8_host",
                };
                trace.emitFinalWithWork(
                    .lm_head,
                    @intCast(last_position),
                    0,
                    @ptrCast(stage1.runtime_buffers.projected_logits_host.ptr),
                    .f32,
                    .{ @intCast(stage1.runtime_buffers.projected_vocab), 0, 0, 0 },
                    1,
                    kernel_name,
                    .{ .flops = total_flops, .bytes = total_bytes_lm },
                );
            }

            if (stage1.runtime_buffers.projected_vocab == logits_out.len) {
                @memcpy(logits_out, stage1.runtime_buffers.projected_logits_host);
            } else {
                @memset(logits_out, -1.0e9);
                @memcpy(logits_out[0..stage1.runtime_buffers.projected_vocab], stage1.runtime_buffers.projected_logits_host);
            }
            if (stage1.loaded.config.logits_scaling != 1.0) {
                for (logits_out) |*v| {
                    v.* /= stage1.loaded.config.logits_scaling;
                }
                if (trace.isEnabled()) {
                    trace.emitFinal(
                        .logits_scaled,
                        @intCast(last_position),
                        0,
                        @ptrCast(logits_out.ptr),
                        .f32,
                        .{ @intCast(stage1.vocab_size), 0, 0, 0 },
                        1,
                        null,
                    );
                }
            }
        }

        pos_base += rows;
    }
}

fn runPipeline2WithPipelineRuntime(
    self: anytype,
    stage1_backend: anytype,
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    activation_byte_count: usize,
) !void {
    const Ctx = struct {
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_opt: ?[]f32,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
    };
    const Stage0 = struct {
        backend: @TypeOf(self),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const local_layer_limit = layer_end - layer_start;
            try computeGpuPrototypeLogitsWithLayerLimit(
                stage.backend,
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                null,
                local_layer_limit,
                false,
                false,
                stage.ctx.ensure_kv_capacity,
                stage.ctx.trace_seq_len_u32,
                stage.ctx.trace_pos_offset,
                null,
                null,
                null,
                false,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            try stage.backend.runtime_buffers.input_dev.download(&stage.backend.device, host_buf[0..byte_count]);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
                return stage.backend.uploadPipelineActivationFromHost(stage.ctx.slot_index, host_buf[0..byte_count], byte_count);
            }
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.upload(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.backend.compute_stream) |stream| {
                try stage.backend.device.synchronizeStream(stream);
                return;
            }
            try stage.backend.device.synchronize();
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    const Stage1 = struct {
        backend: @TypeOf(stage1_backend),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const local_layer_limit = layer_end - layer_start;
            try computeGpuPrototypeLogitsWithLayerLimit(
                stage.backend,
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                stage.ctx.logits_out_opt,
                local_layer_limit,
                stage.ctx.compute_logits,
                stage.ctx.download_logits,
                stage.ctx.ensure_kv_capacity,
                stage.ctx.trace_seq_len_u32,
                stage.ctx.trace_pos_offset,
                null,
                null,
                null,
                true,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.download(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
                return stage.backend.uploadPipelineActivationFromHost(stage.ctx.slot_index, host_buf[0..byte_count], byte_count);
            }
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.upload(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.backend.compute_stream) |stream| {
                try stage.backend.device.synchronizeStream(stream);
                return;
            }
            try stage.backend.device.synchronize();
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    const Transfer = struct {
        owner: @TypeOf(self),

        pub fn transfer(t: *@This(), src: *Stage0, dst: *Stage1, byte_count: usize) anyerror!void {
            _ = src;
            try t.owner.transferPipelineActivation(dst.backend, byte_count);
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    var ctx = Ctx{
        .token = token,
        .position = position,
        .slot_index = slot_index,
        .logits_out_opt = logits_out_opt,
        .compute_logits = compute_logits,
        .download_logits = download_logits,
        .ensure_kv_capacity = ensure_kv_capacity,
        .trace_seq_len_u32 = trace_seq_len_u32,
        .trace_pos_offset = trace_pos_offset,
    };
    try staged_orchestrator.executeTwoStageForward(
        Stage0,
        Stage1,
        Transfer,
        .{ .backend = self, .ctx = &ctx },
        .{ .backend = stage1_backend, .ctx = &ctx },
        self.split_layer,
        self.split_layer + stage1_backend.block_runtime.blocks.len,
        &.{},
        &.{},
        activation_byte_count,
        null,
        .{ .owner = self },
    );
}

fn runCpuGpuWithPipelineRuntime(
    self: anytype,
    cpu_stage0_backend: anytype,
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    activation_byte_count: usize,
) !void {
    const Ctx = struct {
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_opt: ?[]f32,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
    };
    const Stage0 = struct {
        backend: @TypeOf(cpu_stage0_backend),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            try stage.backend.computePrototypeLogitsWithLayerRange(
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                null,
                layer_start,
                layer_end,
                false,
                false,
                stage.ctx.ensure_kv_capacity,
                false,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const src = stage.backend.slotActivationBytes(stage.ctx.slot_index);
            if (byte_count > src.len) return error.InvalidArgument;
            @memcpy(host_buf[0..byte_count], src[0..byte_count]);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const dst = stage.backend.slotActivationBytesMut(stage.ctx.slot_index);
            if (byte_count > dst.len) return error.InvalidArgument;
            @memcpy(dst[0..byte_count], host_buf[0..byte_count]);
        }

        pub fn synchronize(_: *@This()) anyerror!void {}

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    const Stage1 = struct {
        backend: @TypeOf(self),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const local_layer_limit = layer_end - layer_start;
            try computeGpuPrototypeLogitsWithLayerLimit(
                stage.backend,
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                stage.ctx.logits_out_opt,
                local_layer_limit,
                stage.ctx.compute_logits,
                stage.ctx.download_logits,
                stage.ctx.ensure_kv_capacity,
                stage.ctx.trace_seq_len_u32,
                stage.ctx.trace_pos_offset,
                null,
                null,
                null,
                true,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.download(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
                return stage.backend.uploadPipelineActivationFromHost(stage.ctx.slot_index, host_buf[0..byte_count], byte_count);
            }
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.upload(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.backend.compute_stream) |stream| {
                try stage.backend.device.synchronizeStream(stream);
                return;
            }
            try stage.backend.device.synchronize();
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    var ctx = Ctx{
        .token = token,
        .position = position,
        .slot_index = slot_index,
        .logits_out_opt = logits_out_opt,
        .compute_logits = compute_logits,
        .download_logits = download_logits,
        .ensure_kv_capacity = ensure_kv_capacity,
        .trace_seq_len_u32 = trace_seq_len_u32,
        .trace_pos_offset = trace_pos_offset,
    };
    try staged_orchestrator.executeTwoStageForward(
        Stage0,
        Stage1,
        null,
        .{ .backend = cpu_stage0_backend, .ctx = &ctx },
        .{ .backend = self, .ctx = &ctx },
        self.split_layer,
        self.split_layer + self.block_runtime.blocks.len,
        &.{},
        &.{},
        activation_byte_count,
        self.pipeline_host_staging,
        {},
    );
}

fn runCpuGpuGpuWithPipelineRuntime(
    self: anytype,
    cpu_stage0_backend: anytype,
    gpu_stage1_backend: anytype,
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    activation01_byte_count: usize,
    activation12_byte_count: usize,
) !void {
    const Ctx = struct {
        token: u32,
        position: usize,
        slot_index: usize,
        logits_out_opt: ?[]f32,
        compute_logits: bool,
        download_logits: bool,
        ensure_kv_capacity: bool,
        trace_seq_len_u32: u32,
        trace_pos_offset: usize,
    };
    const Stage0 = struct {
        backend: @TypeOf(cpu_stage0_backend),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            try stage.backend.computePrototypeLogitsWithLayerRange(
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                null,
                layer_start,
                layer_end,
                false,
                false,
                stage.ctx.ensure_kv_capacity,
                false,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const src = stage.backend.slotActivationBytes(stage.ctx.slot_index);
            if (byte_count > src.len) return error.InvalidArgument;
            @memcpy(host_buf[0..byte_count], src[0..byte_count]);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const dst = stage.backend.slotActivationBytesMut(stage.ctx.slot_index);
            if (byte_count > dst.len) return error.InvalidArgument;
            @memcpy(dst[0..byte_count], host_buf[0..byte_count]);
        }

        pub fn synchronize(_: *@This()) anyerror!void {}

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    const Stage1 = struct {
        backend: @TypeOf(gpu_stage1_backend),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const local_layer_limit = layer_end - layer_start;
            try computeGpuPrototypeLogitsWithLayerLimit(
                stage.backend,
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                null,
                local_layer_limit,
                false,
                false,
                stage.ctx.ensure_kv_capacity,
                stage.ctx.trace_seq_len_u32,
                stage.ctx.trace_pos_offset,
                null,
                null,
                null,
                true,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.download(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
                return stage.backend.uploadPipelineActivationFromHost(stage.ctx.slot_index, host_buf[0..byte_count], byte_count);
            }
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.upload(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.backend.compute_stream) |stream| {
                try stage.backend.device.synchronizeStream(stream);
                return;
            }
            try stage.backend.device.synchronize();
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    const Stage2 = struct {
        backend: @TypeOf(self),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (input.len != 0) return error.InvalidArgument;
            if (layer_end < layer_start) return error.InvalidArgument;
            const local_layer_limit = layer_end - layer_start;
            try computeGpuPrototypeLogitsWithLayerLimit(
                stage.backend,
                stage.ctx.token,
                stage.ctx.position,
                stage.ctx.slot_index,
                stage.ctx.logits_out_opt,
                local_layer_limit,
                stage.ctx.compute_logits,
                stage.ctx.download_logits,
                stage.ctx.ensure_kv_capacity,
                stage.ctx.trace_seq_len_u32,
                stage.ctx.trace_pos_offset,
                null,
                null,
                null,
                true,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.download(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            if (byte_count > host_buf.len) return error.InvalidArgument;
            const BackendType = @TypeOf(stage.backend.*);
            if (comptime @hasDecl(BackendType, "uploadPipelineActivationFromHost")) {
                return stage.backend.uploadPipelineActivationFromHost(stage.ctx.slot_index, host_buf[0..byte_count], byte_count);
            }
            if (comptime @hasField(BackendType, "runtime_buffers") and @hasField(BackendType, "device")) {
                return stage.backend.runtime_buffers.input_dev.upload(&stage.backend.device, host_buf[0..byte_count]);
            }
            return error.InvalidTopologyConfig;
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            if (stage.backend.compute_stream) |stream| {
                try stage.backend.device.synchronizeStream(stream);
                return;
            }
            try stage.backend.device.synchronize();
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };

    var ctx = Ctx{
        .token = token,
        .position = position,
        .slot_index = slot_index,
        .logits_out_opt = logits_out_opt,
        .compute_logits = compute_logits,
        .download_logits = download_logits,
        .ensure_kv_capacity = ensure_kv_capacity,
        .trace_seq_len_u32 = trace_seq_len_u32,
        .trace_pos_offset = trace_pos_offset,
    };
    try staged_orchestrator.executeThreeStageForward(
        Stage0,
        Stage1,
        Stage2,
        .{ .backend = cpu_stage0_backend, .ctx = &ctx },
        .{ .backend = gpu_stage1_backend, .ctx = &ctx },
        .{ .backend = self, .ctx = &ctx },
        self.split_layer,
        self.pipelineSplitLayerStage2(),
        self.pipelineSplitLayerStage2() + self.block_runtime.blocks.len,
        &.{},
        &.{},
        &.{},
        activation01_byte_count,
        activation12_byte_count,
        self.pipeline_host_staging,
        self.pipeline_host_staging_stage12,
    );
}

pub fn computeGpuPrototypeLogitsWithLayerLimit(
    self: anytype,
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    layer_limit: usize,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    hidden_override: ?[]const f32,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
    use_preloaded_input: bool,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "pipelineStage1") and
        @hasDecl(SelfType, "transferPipelineActivation"))
    {
        if (topologyModeIs(self, "pipeline2") and
            layer_limit == self.block_runtime.blocks.len and
            compute_logits and
            !use_preloaded_input)
        {
            if (comptime @hasDecl(SelfType, "pipelineActivationByteCount")) {
                if (hidden_override == null and deepstack_layer_features_opt == null and deepstack_feature_index_opt == null) {
                    var runtime_stage1 = self.pipelineStage1() orelse return error.InvalidTopologyConfig;
                    if (runtime_stage1.state_descriptor_count > 0) {
                        try runtime_stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                    }
                    runtime_stage1.activateKvSlot(slot_index);
                    const runtime_activation_bytes = try pipelineActivationByteCountFor(self);
                    try runPipeline2WithPipelineRuntime(
                        self,
                        runtime_stage1,
                        token,
                        position,
                        slot_index,
                        logits_out_opt,
                        compute_logits,
                        download_logits,
                        ensure_kv_capacity,
                        trace_seq_len_u32,
                        trace_pos_offset,
                        runtime_activation_bytes,
                    );
                    // Stage1 computed logits on stage1's device. When the caller
                    // keeps logits on-device (download_logits=false, used by
                    // decodeStreaming + selectNextTokenFromDeviceLogitsImpl),
                    // copy them to stage0's device buffer.
                    if (comptime @hasField(SelfType, "runtime_buffers")) {
                        if (compute_logits and !download_logits) {
                            const proj_vocab = self.runtime_buffers.projected_vocab;
                            const host_logits = std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host[0..proj_vocab]);
                            try runtime_stage1.runtime_buffers.logits_dev.download(
                                &runtime_stage1.device,
                                host_logits,
                            );
                            try self.runtime_buffers.logits_dev.upload(
                                &self.device,
                                host_logits,
                            );
                        }
                    }
                    return;
                }
            }
            var stage1_deepstack_layer_features_opt: ?[]const []const f32 = null;
            if (deepstack_layer_features_opt) |deepstack_layer_features| {
                if (self.split_layer < deepstack_layer_features.len) {
                    stage1_deepstack_layer_features_opt = deepstack_layer_features[self.split_layer..];
                }
            }
            var stage1 = self.pipelineStage1() orelse return error.InvalidTopologyConfig;
            if (stage1.state_descriptor_count > 0) {
                try stage1.mirrorSlotStateBlocksFrom(self, slot_index);
            }
            stage1.activateKvSlot(slot_index);
            try computeGpuPrototypeLogitsWithLayerLimit(
                self,
                token,
                position,
                slot_index,
                null,
                layer_limit,
                false,
                false,
                ensure_kv_capacity,
                trace_seq_len_u32,
                trace_pos_offset,
                hidden_override,
                deepstack_layer_features_opt,
                deepstack_feature_index_opt,
                false,
            );
            const activation_bytes = try pipelineActivationByteCountFor(self);
            try self.transferPipelineActivation(stage1, activation_bytes);
            try computeGpuPrototypeLogitsWithLayerLimit(
                stage1,
                token,
                position,
                slot_index,
                logits_out_opt,
                stage1.block_runtime.blocks.len,
                compute_logits,
                download_logits,
                ensure_kv_capacity,
                trace_seq_len_u32,
                trace_pos_offset,
                null,
                stage1_deepstack_layer_features_opt,
                deepstack_feature_index_opt,
                true,
            );
            // Stage1 computed logits on stage1's device. When the caller
            // keeps logits on-device (download_logits=false, used by
            // decodeStreaming + selectNextTokenFromDeviceLogitsImpl),
            // copy them to stage0's device buffer.
            if (comptime @hasField(SelfType, "runtime_buffers")) {
                if (compute_logits and !download_logits) {
                    const proj_vocab = self.runtime_buffers.projected_vocab;
                    const host_logits = std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host[0..proj_vocab]);
                    try stage1.runtime_buffers.logits_dev.download(
                        &stage1.device,
                        host_logits,
                    );
                    try self.runtime_buffers.logits_dev.upload(
                        &self.device,
                        host_logits,
                    );
                }
            }
            return;
        }
    }
    if (comptime @hasDecl(SelfType, "pipelineCpuStage0") and
        @hasDecl(SelfType, "pipelineSplitLayer") and
        @hasDecl(SelfType, "transferPipelineActivationFromCpu"))
    {
        if (comptime @hasDecl(SelfType, "pipelineStage1") and
            @hasDecl(SelfType, "pipelineSplitLayerStage2") and
            @hasField(SelfType, "pipeline_host_staging_stage12"))
        {
            if (topologyModeIs(self, "cpu_gpu_gpu") and
                layer_limit == self.block_runtime.blocks.len and
                compute_logits and
                !use_preloaded_input)
            {
                if (hidden_override != null or deepstack_layer_features_opt != null or deepstack_feature_index_opt != null) {
                    return error.InvalidTopologyConfig;
                }
                const stage0 = self.pipelineCpuStage0() orelse return error.InvalidTopologyConfig;
                var stage1 = self.pipelineStage1() orelse return error.InvalidTopologyConfig;
                const split_layer = self.pipelineSplitLayer();
                const split_layer_stage2 = self.pipelineSplitLayerStage2();
                if (split_layer == 0 or split_layer_stage2 <= split_layer) return error.InvalidTopologyConfig;
                if (stage1.state_descriptor_count > 0) {
                    try stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                }
                stage1.activateKvSlot(slot_index);
                self.activateKvSlot(slot_index);
                if (comptime @hasDecl(SelfType, "pipelineActivationByteCount")) {
                    const runtime_activation01_bytes = try pipelineActivationByteCountFor(self);
                    try runCpuGpuGpuWithPipelineRuntime(
                        self,
                        stage0,
                        stage1,
                        token,
                        position,
                        slot_index,
                        logits_out_opt,
                        compute_logits,
                        download_logits,
                        ensure_kv_capacity,
                        trace_seq_len_u32,
                        trace_pos_offset,
                        runtime_activation01_bytes,
                        runtime_activation01_bytes,
                    );
                    return;
                }
                if (comptime !@hasDecl(@TypeOf(stage1.*), "transferPipelineActivationFromCpu") or
                    !@hasDecl(@TypeOf(stage1.*), "transferPipelineActivation"))
                {
                    return error.InvalidTopologyConfig;
                }
                try executeCpuStage0LayerRange(
                    stage0,
                    token,
                    position,
                    slot_index,
                    split_layer,
                    ensure_kv_capacity,
                );
                const activation_bytes = try pipelineActivationByteCountFor(self);
                try stage1.transferPipelineActivationFromCpu(stage0, slot_index, activation_bytes);
                try computeGpuPrototypeLogitsWithLayerLimit(
                    stage1,
                    token,
                    position,
                    slot_index,
                    null,
                    stage1.block_runtime.blocks.len,
                    false,
                    false,
                    ensure_kv_capacity,
                    trace_seq_len_u32,
                    trace_pos_offset,
                    null,
                    null,
                    null,
                    true,
                );
                try stage1.transferPipelineActivation(self, activation_bytes);
                return computeGpuPrototypeLogitsWithLayerLimit(
                    self,
                    token,
                    position,
                    slot_index,
                    logits_out_opt,
                    self.block_runtime.blocks.len,
                    compute_logits,
                    download_logits,
                    ensure_kv_capacity,
                    trace_seq_len_u32,
                    trace_pos_offset,
                    null,
                    null,
                    null,
                    true,
                );
            }
        }
        if (topologyModeIs(self, "cpu_gpu") and
            layer_limit == self.block_runtime.blocks.len and
            compute_logits and
            !use_preloaded_input)
        {
            if (hidden_override != null or deepstack_layer_features_opt != null or deepstack_feature_index_opt != null) {
                return error.InvalidTopologyConfig;
            }
            const stage0 = self.pipelineCpuStage0() orelse return error.InvalidTopologyConfig;
            const split_layer = self.pipelineSplitLayer();
            if (split_layer == 0) return error.InvalidTopologyConfig;
            if (comptime @hasDecl(SelfType, "pipelineActivationByteCount")) {
                self.activateKvSlot(slot_index);
                const runtime_activation_bytes = try pipelineActivationByteCountFor(self);
                try runCpuGpuWithPipelineRuntime(
                    self,
                    stage0,
                    token,
                    position,
                    slot_index,
                    logits_out_opt,
                    compute_logits,
                    download_logits,
                    ensure_kv_capacity,
                    trace_seq_len_u32,
                    trace_pos_offset,
                    runtime_activation_bytes,
                );
                return;
            }
            try executeCpuStage0LayerRange(
                stage0,
                token,
                position,
                slot_index,
                split_layer,
                ensure_kv_capacity,
            );
            const activation_bytes = try pipelineActivationByteCountFor(self);
            try self.transferPipelineActivationFromCpu(stage0, slot_index, activation_bytes);
            return computeGpuPrototypeLogitsWithLayerLimit(
                self,
                token,
                position,
                slot_index,
                logits_out_opt,
                layer_limit,
                compute_logits,
                download_logits,
                ensure_kv_capacity,
                trace_seq_len_u32,
                trace_pos_offset,
                null,
                null,
                null,
                true,
            );
        }
    }
    if (comptime @hasDecl(SelfType, "computeGpuPrototypeLogitsWithLayerLimitTestHook")) {
        return self.computeGpuPrototypeLogitsWithLayerLimitTestHook(
            token,
            position,
            slot_index,
            logits_out_opt,
            layer_limit,
            compute_logits,
            download_logits,
            ensure_kv_capacity,
            trace_seq_len_u32,
            trace_pos_offset,
            hidden_override,
            deepstack_layer_features_opt,
            deepstack_feature_index_opt,
            use_preloaded_input,
        );
    }

    const previous_launch_phase = self.device.setLaunchPhase(.decode);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);
    if (!compute_logits and download_logits) return error.InvalidArgument;
    if (use_preloaded_input and hidden_override != null) return error.InvalidArgument;
    if (deepstack_feature_index_opt != null and deepstack_layer_features_opt == null) return error.InvalidArgument;
    if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
    if (download_logits) {
        const logits_out = logits_out_opt orelse return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
    }
    if (position >= self.max_seq_len) return error.InvalidArgument;
    if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;
    if (position == 0) {
        try resetShortConvStates(
            self,
        );
        resetAttentionCpuStates(
            self,
        );
        resetGatedDeltaStates(
            self,
        );
    }
    if (ensure_kv_capacity) {
        try ensureKvCapacity(self, position + 1);
    }

    const vector_add_function = self.vector_add_function orelse return error.CudaKernelUnavailable;
    if (self.loaded.config.residual_multiplier != 1.0 and self.vector_add_scaled_function == null) {
        return error.CudaKernelUnavailable;
    }
    if (self.loaded.config.use_gelu) {
        if (self.gelu_mul_function == null) return error.CudaKernelUnavailable;
    } else {
        if (self.silu_mul_function == null) return error.CudaKernelUnavailable;
    }
    const shortconv_step_function = self.shortconv_step_function orelse return error.CudaKernelUnavailable;
    const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
    const embedding_lookup_f32_function = self.embedding_lookup_f32_function;
    const embedding_lookup_u16_function = self.embedding_lookup_u16_function;
    const embedding_lookup_gaffine_u4_function = self.embedding_lookup_gaffine_u4_function;
    const cast_f32_to_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const kv_write_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        self.kv_write_f16_function
    else
        null;
    const rope_store_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        self.rope_store_f16_function
    else
        null;
    const rope_function = self.rope_function orelse return error.CudaKernelUnavailable;
    const attn_scores_heads_f32_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        null
    else
        (self.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable);
    const attn_scores_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const attn_fused_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        self.attn_fused_heads_f16_kv_function
    else
        null;
    const attn_fused_prefill_heads_f16_kv_function: ?compute.cuda.Function = null;
    const softmax_rows_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.softmax_rows_function orelse return error.CudaKernelUnavailable)
    else
        self.softmax_rows_function;
    const attn_weighted_sum_heads_f32_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        null
    else
        (self.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable);
    const attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const d_model_u32: u32 = @intCast(self.d_model);
    const head_dim_u32: u32 = @intCast(self.head_dim);
    const rope_dim_u32: u32 = @intCast(self.rope_dim);
    const n_heads_u32: u32 = @intCast(self.n_heads);
    const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
    const seq_len_u32: u32 = @intCast(position + 1);
    const position_u32: u32 = @intCast(position);
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    var input_row = try bufferSlice(&self.runtime_buffers.input_dev, 0, row_bytes);
    var norm_out_row = try bufferSlice(&self.runtime_buffers.norm_out_dev, 0, row_bytes);
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;

    if (use_preloaded_input) {
        // Stage boundary handoff path: input_dev is already populated by caller.
    } else if (hidden_override) |hidden| {
        if (hidden.len != self.d_model) return error.InvalidArgument;
        @memcpy(self.runtime_buffers.hidden_host, hidden);
        try input_row.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
    } else {
        var used_device_lookup = false;
        if (enable_device_embedding_lookup and self.runtime_buffers.embedding_lookup != null) {
            const lookup = &self.runtime_buffers.embedding_lookup.?;
            switch (lookup.kind) {
                .f32 => {
                    if (embedding_lookup_f32_function) |kernel| {
                        try compute.cuda.embedding_lookup_f32.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &input_row,
                            &lookup.buffer,
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            token,
                            lookup.layout_tag,
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                .f16 => {
                    if (embedding_lookup_u16_function) |kernel| {
                        try compute.cuda.embedding_lookup_u16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &input_row,
                            &lookup.buffer,
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            token,
                            lookup.layout_tag,
                            compute.cuda.embedding_lookup_u16.dtype_f16,
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                .bf16 => {
                    if (embedding_lookup_u16_function) |kernel| {
                        try compute.cuda.embedding_lookup_u16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &input_row,
                            &lookup.buffer,
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            token,
                            lookup.layout_tag,
                            compute.cuda.embedding_lookup_u16.dtype_bf16,
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                .gaffine_u4 => {
                    if (embedding_lookup_gaffine_u4_function) |kernel| {
                        if (lookup.scales) |*scales_buf| {
                            if (lookup.biases) |*biases_buf| {
                                try compute.cuda.embedding_lookup_gaffine_u4.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    scales_buf,
                                    biases_buf,
                                    lookup.dim0,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.group_size,
                                    lookup.scales_dtype_tag,
                                    lookup.multiplier,
                                );
                                used_device_lookup = true;
                            }
                        }
                    }
                },
            }
        }
        if (!used_device_lookup) {
            const used_model_embeddings = tryPopulateHiddenFromToken(self.loaded, token, self.runtime_buffers.hidden_host) catch |err| switch (err) {
                error.InvalidArgument => return error.InvalidArgument,
                else => return err,
            };
            if (!used_model_embeddings) {
                log.warn("inference", "CUDA embedding layout unsupported", .{
                    .token = token,
                    .embed_shape_0 = self.loaded.token_embeddings.shape[0],
                    .embed_shape_1 = self.loaded.token_embeddings.shape[1],
                    .embed_dtype = @tagName(self.loaded.token_embeddings.dtype),
                    .embed_ndim = self.loaded.token_embeddings.n_dims,
                });
                return error.UnsupportedModel;
            }
            if (self.loaded.config.embedding_multiplier != 1.0) {
                for (self.runtime_buffers.hidden_host) |*v| {
                    v.* *= self.loaded.config.embedding_multiplier;
                }
            }
            try input_row.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
        }
    }

    // CUDA graph capture: record all GPU kernels (layer loop + final norm + lm_head)
    // into a graph, then replay with near-zero scheduling gaps between kernels.
    // Requires: named stream, no tracing (trace downloads break capture), no deepstack
    // (sync memcpy during capture is not stream-ordered).
    var graph_capture_active = false;
    if (self.compute_stream != null and compute_logits and
        !trace.isEnabled() and deepstack_layer_features_opt == null)
    {
        if (self.device.streamBeginCapture(self.compute_stream.?)) {
            graph_capture_active = true;
        } else |_| {}
    }
    errdefer if (graph_capture_active) {
        _ = self.device.streamEndCapture(self.compute_stream.?) catch {};
    };

    var final_hidden = input_row;
    var layer_idx: usize = 0;
    while (layer_idx < layer_limit) : (layer_idx += 1) {
        const layer = &self.block_runtime.blocks[layer_idx];
        const attention_kernels = AttentionKernelSet{
            .attn_scores_heads_f32_function = attn_scores_heads_f32_function,
            .attn_weighted_sum_heads_f32_function = attn_weighted_sum_heads_f32_function,
            .attn_scores_heads_f16_kv_function = attn_scores_heads_f16_kv_function,
            .softmax_rows_function = softmax_rows_function,
            .attn_weighted_sum_heads_f16_kv_function = attn_weighted_sum_heads_f16_kv_function,
            .attn_fused_heads_f16_kv_function = attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = null,
            .causal_attn_softmax_f32_function = null,
        };
        final_hidden = try self.tryExecuteLayerProgram(
            layer,
            slot_index,
            layer_idx,
            d_model_u32,
            head_dim_u32,
            rope_dim_u32,
            n_heads_u32,
            n_kv_heads_u32,
            1,
            seq_len_u32,
            trace_seq_len_u32,
            trace_pos_offset,
            position,
            position_u32,
            global_rope_theta,
            local_rope_theta,
            rope_function,
            copy_function,
            cast_f32_to_f16_function,
            kv_write_f16_function,
            rope_store_f16_function,
            shortconv_step_function,
            attention_kernels,
            null,
        );
        // Deepstack: per-request feature vector addition between layer program
        // dispatches. Operates outside the per-instruction adapter table — same
        // pattern as embedding lookup and final logit projection.
        if (deepstack_layer_features_opt) |deepstack_layer_features| {
            if (deepstack_feature_index_opt) |deepstack_feature_index| {
                if (layer_idx < deepstack_layer_features.len) {
                    const layer_features = deepstack_layer_features[layer_idx];
                    const feature_rows = std.math.divExact(usize, layer_features.len, self.d_model) catch {
                        log.warn("inference", "CUDA deepstack add skipped: invalid layer feature stride", .{
                            .layer_index = layer_idx,
                            .feature_len = layer_features.len,
                            .d_model = self.d_model,
                        });
                        continue;
                    };
                    if (deepstack_feature_index >= feature_rows) {
                        log.warn("inference", "CUDA deepstack add skipped: feature row index out of range", .{
                            .layer_index = layer_idx,
                            .feature_index = deepstack_feature_index,
                            .feature_rows = feature_rows,
                        });
                        continue;
                    }
                    const row_start = std.math.mul(usize, deepstack_feature_index, self.d_model) catch {
                        log.warn("inference", "CUDA deepstack add skipped: row offset overflow", .{
                            .layer_index = layer_idx,
                            .feature_index = deepstack_feature_index,
                            .d_model = self.d_model,
                        });
                        continue;
                    };
                    const feature_row = layer_features[row_start .. row_start + self.d_model];
                    try self.runtime_buffers.deepstack_add_dev.upload(&self.device, std.mem.sliceAsBytes(feature_row));
                    try compute.cuda.vector_add.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        vector_add_function,
                        &self.runtime_buffers.input_dev,
                        &self.runtime_buffers.deepstack_add_dev,
                        &self.runtime_buffers.input_dev,
                        d_model_u32,
                    );
                    final_hidden = self.runtime_buffers.input_dev;
                }
            }
        }
    }
    if (!compute_logits) {
        // Stage handoff invariant: when skipping logits, publish the final hidden
        // row to input_dev so the next stage can consume a deterministic buffer.
        if (final_hidden.pointer != self.runtime_buffers.input_dev.pointer) {
            try self.runtime_buffers.input_dev.copyFrom(&self.device, &final_hidden, row_bytes);
        }
        return;
    }

    try compute.cuda.rmsnorm.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.rmsnorm_function orelse return error.CudaKernelUnavailable,
        &final_hidden,
        &self.runtime_buffers.norm_weight_dev,
        &norm_out_row,
        1,
        @intCast(self.d_model),
        self.norm_eps,
        self.loaded.runtime.weight_offset,
    );
    if (trace.isEnabled()) {
        try norm_out_row.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
        trace.emitFinal(
            .final_norm,
            @intCast(position),
            1,
            @ptrCast(self.runtime_buffers.hidden_host.ptr),
            .f32,
            .{ @intCast(self.d_model), 0, 0, 0 },
            1,
            "cuda_final_norm_host",
        );
    }

    try engine_ops.linearForwardRows(self, &norm_out_row, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);

    // End graph capture: instantiate/update exec, then launch the captured graph.
    if (graph_capture_active) {
        const new_graph = self.device.streamEndCapture(self.compute_stream.?) catch
            return error.CudaGraphCaptureFailed;
        defer self.device.graphDestroy(new_graph);

        if (self.decode_graph_exec) |exec| {
            self.device.graphExecUpdate(exec, new_graph) catch {
                // Topology changed — re-instantiate.
                self.device.graphExecDestroy(exec);
                self.decode_graph_exec = self.device.graphInstantiate(new_graph) catch
                    return error.CudaGraphInstantiateFailed;
            };
        } else {
            self.decode_graph_exec = self.device.graphInstantiate(new_graph) catch
                return error.CudaGraphInstantiateFailed;
        }
        try self.device.graphLaunch(self.decode_graph_exec.?, self.compute_stream);
    }

    if (!download_logits) return;

    const logits_out = logits_out_opt.?;
    try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));

    if (trace.isEnabled()) {
        const rows128: u128 = 1;
        const d_model128: u128 = @intCast(self.d_model);
        const vocab128: u128 = @intCast(self.runtime_buffers.projected_vocab);
        const total_flops = saturatingU64FromU128(2 * rows128 * d_model128 * vocab128);
        const total_bytes = saturatingU64FromU128(
            rows128 * d_model128 * @sizeOf(f32) +
                @as(u128, self.runtime_buffers.projection_weight.byteSize()) +
                rows128 * vocab128 * @sizeOf(f32),
        );
        const kernel_name = switch (self.runtime_buffers.projection_weight) {
            .dense_f32 => "matmul_lm_head_f32_host",
            .dense_u16 => |w| switch (w.dtype) {
                .bf16 => "matmul_lm_head_bf16_host",
                .f16 => "matmul_lm_head_f16_host",
            },
            .gaffine_u4 => "matmul_lm_head_gaffine_u4_host",
            .gaffine_u8 => "matmul_lm_head_gaffine_u8_host",
        };
        trace.emitFinalWithWork(
            .lm_head,
            @intCast(position),
            0,
            @ptrCast(self.runtime_buffers.projected_logits_host.ptr),
            .f32,
            .{ @intCast(self.runtime_buffers.projected_vocab), 0, 0, 0 },
            1,
            kernel_name,
            .{ .flops = total_flops, .bytes = total_bytes },
        );
    }

    if (self.runtime_buffers.projected_vocab == logits_out.len) {
        @memcpy(logits_out, self.runtime_buffers.projected_logits_host);
    } else {
        @memset(logits_out, -1.0e9);
        @memcpy(logits_out[0..self.runtime_buffers.projected_vocab], self.runtime_buffers.projected_logits_host);
    }
    if (self.loaded.config.logits_scaling != 1.0) {
        for (logits_out) |*v| {
            v.* /= self.loaded.config.logits_scaling;
        }
        if (trace.isEnabled()) {
            trace.emitFinal(
                .logits_scaled,
                @intCast(position),
                0,
                @ptrCast(logits_out.ptr),
                .f32,
                .{ @intCast(self.vocab_size), 0, 0, 0 },
                1,
                null,
            );
        }
    }
}

/// Batched decode: process N tokens at different positions/slots together.
/// Uses GEMM (not GEMV) for layer projections, sharing weight reads
/// across sequences for ~Nx throughput.
pub fn computeBatchedDecodeLogits(
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
) !void {
    const SelfType = @TypeOf(self.*);
    const n_usize = tokens.len;
    if (n_usize == 0) return;
    if (n_usize > self.max_batch_size) return error.InvalidArgument;
    const n: u32 = @intCast(n_usize);
    if (topologyModeIs(self, "pipeline2") or topologyModeIs(self, "cpu_gpu") or topologyModeIs(self, "cpu_gpu_gpu")) {
        for (0..n_usize) |i| {
            self.activateKvSlot(slot_indices[i]);
            try computeGpuPrototypeLogitsWithLayerLimit(
                self,
                tokens[i],
                positions[i],
                slot_indices[i],
                self.slotLogits(slot_indices[i]),
                self.block_runtime.blocks.len,
                true,
                true,
                true,
                1,
                positions[i],
                null,
                null,
                null,
                false,
            );
        }
        return;
    }
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    const previous_launch_phase = self.device.setLaunchPhase(.decode);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);

    // Validate all slots, positions, and state block bindings.
    for (slot_indices) |slot_idx| {
        if (!self.slotIndexSupported(slot_idx)) return error.InvalidArgument;
    }
    for (positions) |pos| {
        if (pos >= self.max_seq_len) return error.InvalidArgument;
    }
    if (self.state_descriptor_count > 0) {
        for (slot_indices) |slot_idx| {
            try self.ensureSlotStateBlocksBoundForScheduler(slot_idx);
        }
    }

    // Ensure scratch buffers fit N rows.
    try self.runtime_buffers.ensureRowCapacity(&self.device, n_usize, self.fixed_alloc_mode);
    try self.ensureLayerProgramSlotRowCapacity(n_usize, self.fixed_alloc_mode);

    // Ensure KV capacity for each slot via activate/ensure/save.
    for (0..n_usize) |i| {
        self.activateKvSlot(slot_indices[i]);
        try ensureKvCapacity(self, positions[i] + 1);
    }
    self.saveActiveKvSlot();

    // Extract kernel functions (same set as single-token path).
    const shortconv_step_function = self.shortconv_step_function orelse return error.CudaKernelUnavailable;
    const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
    const cast_f32_to_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const kv_write_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        self.kv_write_f16_function
    else
        null;
    const rope_store_f16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        self.rope_store_f16_function
    else
        null;
    const rope_function = self.rope_function orelse return error.CudaKernelUnavailable;
    const attn_scores_heads_f32_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        null
    else
        (self.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable);
    const attn_scores_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const attn_fused_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        self.attn_fused_heads_f16_kv_function
    else
        null;
    const softmax_rows_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.softmax_rows_function orelse return error.CudaKernelUnavailable)
    else
        self.softmax_rows_function;
    const attn_weighted_sum_heads_f32_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        null
    else
        (self.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable);
    const attn_weighted_sum_heads_f16_kv_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const d_model_u32: u32 = @intCast(self.d_model);
    const head_dim_u32: u32 = @intCast(self.head_dim);
    const rope_dim_u32: u32 = @intCast(self.rope_dim);
    const n_heads_u32: u32 = @intCast(self.n_heads);
    const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;
    const attention_kernels = AttentionKernelSet{
        .attn_scores_heads_f32_function = attn_scores_heads_f32_function,
        .attn_weighted_sum_heads_f32_function = attn_weighted_sum_heads_f32_function,
        .attn_scores_heads_f16_kv_function = attn_scores_heads_f16_kv_function,
        .softmax_rows_function = softmax_rows_function,
        .attn_weighted_sum_heads_f16_kv_function = attn_weighted_sum_heads_f16_kv_function,
        .attn_fused_heads_f16_kv_function = attn_fused_heads_f16_kv_function,
        .attn_fused_prefill_heads_f16_kv_function = null,
        .attn_fused_prefill_heads_f16_kv_gqa_function = null,
        .causal_attn_softmax_f32_function = null,
    };

    // Embedding lookup: N tokens → N rows of input_dev.
    const embedding_lookup_f32_function = self.embedding_lookup_f32_function;
    const embedding_lookup_u16_function = self.embedding_lookup_u16_function;
    const embedding_lookup_gaffine_u4_function = self.embedding_lookup_gaffine_u4_function;
    for (0..n_usize) |i| {
        var input_row = try bufferSlice(&self.runtime_buffers.input_dev, i * row_bytes, row_bytes);
        var used_device_lookup = false;
        if (enable_device_embedding_lookup and self.runtime_buffers.embedding_lookup != null) {
            const lookup = &self.runtime_buffers.embedding_lookup.?;
            switch (lookup.kind) {
                .f32 => {
                    if (embedding_lookup_f32_function) |kernel| {
                        try compute.cuda.embedding_lookup_f32.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &input_row,
                            &lookup.buffer,
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            tokens[i],
                            lookup.layout_tag,
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                .f16 => {
                    if (embedding_lookup_u16_function) |kernel| {
                        try compute.cuda.embedding_lookup_u16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &input_row,
                            &lookup.buffer,
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            tokens[i],
                            lookup.layout_tag,
                            compute.cuda.embedding_lookup_u16.dtype_f16,
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                .bf16 => {
                    if (embedding_lookup_u16_function) |kernel| {
                        try compute.cuda.embedding_lookup_u16.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &input_row,
                            &lookup.buffer,
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            tokens[i],
                            lookup.layout_tag,
                            compute.cuda.embedding_lookup_u16.dtype_bf16,
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                .gaffine_u4 => {
                    if (embedding_lookup_gaffine_u4_function) |kernel| {
                        if (lookup.scales) |*scales_buf| {
                            if (lookup.biases) |*biases_buf| {
                                try compute.cuda.embedding_lookup_gaffine_u4.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    scales_buf,
                                    biases_buf,
                                    lookup.dim0,
                                    lookup.hidden_dim,
                                    tokens[i],
                                    lookup.group_size,
                                    lookup.scales_dtype_tag,
                                    lookup.multiplier,
                                );
                                used_device_lookup = true;
                            }
                        }
                    }
                },
            }
        }
        if (!used_device_lookup) {
            const used_model = tryPopulateHiddenFromToken(self.loaded, tokens[i], self.runtime_buffers.hidden_host) catch |err| switch (err) {
                error.InvalidArgument => return error.InvalidArgument,
                else => return err,
            };
            if (!used_model) return error.UnsupportedModel;
            if (self.loaded.config.embedding_multiplier != 1.0) {
                for (self.runtime_buffers.hidden_host) |*v| {
                    v.* *= self.loaded.config.embedding_multiplier;
                }
            }
            try input_row.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
        }
    }

    // Build BatchDecodeInfo with per-row seq_lens.
    const seq_lens = try self.allocator.alloc(u32, n_usize);
    defer self.allocator.free(seq_lens);
    for (0..n_usize) |i| {
        seq_lens[i] = @intCast(positions[i] + 1);
    }
    var batch_info = BatchDecodeInfo{
        .slot_indices = slot_indices,
        .positions = positions,
        .seq_lens = seq_lens,
        .attn_layer_index = 0,
        .gd_layer_index = 0,
        .sc_layer_index = 0,
    };

    // Layer loop with N active rows and batch_info.
    var final_hidden = try bufferSlice(&self.runtime_buffers.input_dev, 0, n_usize * row_bytes);
    const layer_limit = self.block_runtime.blocks.len;
    for (0..layer_limit) |layer_idx| {
        const layer = &self.block_runtime.blocks[layer_idx];
        final_hidden = try self.tryExecuteLayerProgram(
            layer,
            slot_indices[0],
            layer_idx,
            d_model_u32,
            head_dim_u32,
            rope_dim_u32,
            n_heads_u32,
            n_kv_heads_u32,
            n, // active_rows_u32
            1,
            1,
            0,
            0,
            0,
            global_rope_theta,
            local_rope_theta,
            rope_function,
            copy_function,
            cast_f32_to_f16_function,
            kv_write_f16_function,
            rope_store_f16_function,
            shortconv_step_function,
            attention_kernels,
            &batch_info,
        );
        // Advance state layer indices for the type of mixer this layer used.
        if (layer.attention_binding != null) {
            batch_info.attn_layer_index += 1;
        }
        if (layer.gated_delta_binding != null) {
            batch_info.gd_layer_index += 1;
        }
        if (layer.shortconv_binding != null) {
            batch_info.sc_layer_index += 1;
        }
    }

    // Final norm for all N rows.
    const norm_out_bytes = n_usize * self.d_model * @sizeOf(f32);
    var norm_out = try bufferSlice(&self.runtime_buffers.norm_out_dev, 0, norm_out_bytes);
    try compute.cuda.rmsnorm.runWithFunction(
        &self.kernel_arg_pack,
        &self.device,
        self.rmsnorm_function orelse return error.CudaKernelUnavailable,
        &final_hidden,
        &self.runtime_buffers.norm_weight_dev,
        &norm_out,
        n,
        d_model_u32,
        self.norm_eps,
        self.loaded.runtime.weight_offset,
    );

    // LM head + download per row (logits_dev is sized for 1 row).
    const vocab = self.runtime_buffers.projected_vocab;
    for (0..n_usize) |i| {
        var norm_row = try logicalF32RowSlice(&norm_out, n_usize, i, self.d_model);
        try engine_ops.linearForwardRows(self, &norm_row, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);
        try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));
        const slot_logits = self.slotLogits(slot_indices[i]);
        if (vocab == slot_logits.len) {
            @memcpy(slot_logits, self.runtime_buffers.projected_logits_host);
        } else {
            @memset(slot_logits, -1.0e9);
            @memcpy(slot_logits[0..vocab], self.runtime_buffers.projected_logits_host);
        }
        if (self.loaded.config.logits_scaling != 1.0) {
            for (slot_logits) |*v| {
                v.* /= self.loaded.config.logits_scaling;
            }
        }
    }
}

pub fn computeGpuPrototypePrefillLogitsWithLayerLimit(
    self: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
    layer_limit: usize,
) !void {
    const SelfType = @TypeOf(self.*);

    // Pipeline2: batched prefill through stage0 → bulk transfer → stage1.
    // Uses comptime guard because anytype monomorphization requires all referenced
    // methods to exist on the type, even in runtime-unreachable branches.
    if (topologyModeIs(self, "pipeline2") and layer_limit == self.block_runtime.blocks.len) {
        if (tokens.len == 0) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;

        if (comptime @hasDecl(SelfType, "pipelineStage1")) {
            var stage1 = self.pipelineStage1() orelse return error.InvalidTopologyConfig;
            if (stage1.state_descriptor_count > 0) try stage1.mirrorSlotStateBlocksFrom(self, slot_index);
            stage1.activateKvSlot(slot_index);
            self.activateKvSlot(slot_index);
            return computeBatchedPrefillPipeline2(self, stage1, tokens, slot_index, logits_out);
        }
    }

    // Multi-stage topologies without batched prefill: token-by-token fallback.
    if ((topologyModeIs(self, "pipeline2") or topologyModeIs(self, "cpu_gpu") or topologyModeIs(self, "cpu_gpu_gpu")) and
        layer_limit == self.block_runtime.blocks.len)
    {
        if (tokens.len == 0) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;

        var token_index: usize = 0;
        while (token_index < tokens.len) : (token_index += 1) {
            const should_download = token_index + 1 == tokens.len;
            try computeGpuPrototypeLogitsWithLayerLimit(
                self,
                tokens[token_index],
                token_index,
                slot_index,
                if (should_download) logits_out else null,
                layer_limit,
                true,
                should_download,
                true,
                @intCast(token_index + 1),
                token_index,
                null,
                null,
                null,
                false,
            );
        }
        return;
    }

    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);
    if (tokens.len == 0) return error.InvalidArgument;
    if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
    if (logits_out.len != self.vocab_size) return error.InvalidArgument;
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
    if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;

    const total_rows = tokens.len;
    try ensureKvCapacity(self, total_rows);
    try resetShortConvStates(
        self,
    );
    resetAttentionCpuStates(
        self,
    );
    resetGatedDeltaStates(
        self,
    );

    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const d_model_u32: u32 = @intCast(self.d_model);
    const head_dim_u32: u32 = @intCast(self.head_dim);
    const rope_dim_u32: u32 = @intCast(self.rope_dim);
    const n_heads_u32: u32 = @intCast(self.n_heads);
    const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;
    const attention_kernels = AttentionKernelSet{
        .attn_scores_heads_f32_function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_scores_heads_f32_function orelse return error.CudaKernelUnavailable),
        .attn_weighted_sum_heads_f32_function = if (kv_cache_dtype_fp16)
            null
        else
            (self.attn_weighted_sum_heads_f32_function orelse return error.CudaKernelUnavailable),
        .attn_scores_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            (self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null,
        .softmax_rows_function = self.softmax_rows_function,
        .attn_weighted_sum_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            (self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable)
        else
            null,
        .attn_fused_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            self.attn_fused_heads_f16_kv_function
        else
            null,
        .attn_fused_prefill_heads_f16_kv_function = if (kv_cache_dtype_fp16)
            self.attn_fused_prefill_heads_f16_kv_function
        else
            null,
        .attn_fused_prefill_heads_f16_kv_gqa_function = if (kv_cache_dtype_fp16)
            self.attn_fused_prefill_heads_f16_kv_gqa_function
        else
            null,
        .causal_attn_softmax_f32_function = if (kv_cache_dtype_fp16)
            self.causal_attn_softmax_f32_function
        else
            null,
    };

    // Chunked prefill: process in chunks through all layers, building
    // KV cache incrementally. Keeps scratch buffer allocations bounded.
    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, self.prefill_chunk_rows_cap);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];
        try self.runtime_buffers.ensureRowCapacity(&self.device, rows, self.fixed_alloc_mode);
        try self.ensureLayerProgramSlotRowCapacity(rows, self.fixed_alloc_mode);
        var used_device_lookup = false;
        if (enable_device_embedding_lookup and self.runtime_buffers.embedding_lookup != null) {
            const lookup = &self.runtime_buffers.embedding_lookup.?;
            switch (lookup.kind) {
                .f16, .bf16 => {
                    if (self.embedding_lookup_u16_rows_function) |kernel| {
                        const token_bytes = std.math.mul(usize, rows, @sizeOf(u32)) catch return error.InvalidArgument;
                        var token_ids_dev = try bufferSlice(&self.runtime_buffers.prefill_tokens_dev, 0, token_bytes);
                        try token_ids_dev.upload(&self.device, std.mem.sliceAsBytes(chunk_tokens));
                        try compute.cuda.embedding_lookup_u16_rows.runWithFunction(
                            &self.kernel_arg_pack,
                            &self.device,
                            kernel,
                            &self.runtime_buffers.input_dev,
                            &lookup.buffer,
                            &token_ids_dev,
                            @intCast(rows),
                            lookup.dim0,
                            lookup.dim1,
                            lookup.hidden_dim,
                            lookup.layout_tag,
                            switch (lookup.kind) {
                                .f16 => compute.cuda.embedding_lookup_u16_rows.dtype_f16,
                                .bf16 => compute.cuda.embedding_lookup_u16_rows.dtype_bf16,
                                else => unreachable,
                            },
                            lookup.multiplier,
                        );
                        used_device_lookup = true;
                    }
                },
                else => {},
            }

            if (!used_device_lookup) {
                var device_lookup_ok = true;
                var row_idx: usize = 0;
                fill_rows: while (row_idx < chunk_tokens.len) : (row_idx += 1) {
                    const row_offset = std.math.mul(usize, row_idx, row_bytes) catch return error.InvalidArgument;
                    var input_row = try bufferSlice(&self.runtime_buffers.input_dev, row_offset, row_bytes);
                    const token = chunk_tokens[row_idx];
                    switch (lookup.kind) {
                        .f32 => {
                            if (self.embedding_lookup_f32_function) |kernel| {
                                try compute.cuda.embedding_lookup_f32.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    lookup.dim0,
                                    lookup.dim1,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.layout_tag,
                                    lookup.multiplier,
                                );
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                        .f16 => {
                            if (self.embedding_lookup_u16_function) |kernel| {
                                try compute.cuda.embedding_lookup_u16.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    lookup.dim0,
                                    lookup.dim1,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.layout_tag,
                                    compute.cuda.embedding_lookup_u16.dtype_f16,
                                    lookup.multiplier,
                                );
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                        .bf16 => {
                            if (self.embedding_lookup_u16_function) |kernel| {
                                try compute.cuda.embedding_lookup_u16.runWithFunction(
                                    &self.kernel_arg_pack,
                                    &self.device,
                                    kernel,
                                    &input_row,
                                    &lookup.buffer,
                                    lookup.dim0,
                                    lookup.dim1,
                                    lookup.hidden_dim,
                                    token,
                                    lookup.layout_tag,
                                    compute.cuda.embedding_lookup_u16.dtype_bf16,
                                    lookup.multiplier,
                                );
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                        .gaffine_u4 => {
                            if (self.embedding_lookup_gaffine_u4_function) |kernel| {
                                if (lookup.scales) |*scales_buf| {
                                    if (lookup.biases) |*biases_buf| {
                                        try compute.cuda.embedding_lookup_gaffine_u4.runWithFunction(
                                            &self.kernel_arg_pack,
                                            &self.device,
                                            kernel,
                                            &input_row,
                                            &lookup.buffer,
                                            scales_buf,
                                            biases_buf,
                                            lookup.dim0,
                                            lookup.hidden_dim,
                                            token,
                                            lookup.group_size,
                                            lookup.scales_dtype_tag,
                                            lookup.multiplier,
                                        );
                                    } else {
                                        device_lookup_ok = false;
                                        break :fill_rows;
                                    }
                                } else {
                                    device_lookup_ok = false;
                                    break :fill_rows;
                                }
                            } else {
                                device_lookup_ok = false;
                                break :fill_rows;
                            }
                        },
                    }
                }
                used_device_lookup = device_lookup_ok;
            }
        }

        if (!used_device_lookup) {
            const hidden_count = std.math.mul(usize, rows, self.d_model) catch return error.InvalidArgument;
            const hidden_host = try self.allocator.alloc(f32, hidden_count);
            defer self.allocator.free(hidden_host);
            try populatePrefillHiddenFromTokens(self.loaded, chunk_tokens, self.d_model, hidden_host, null);
            try self.runtime_buffers.input_dev.upload(&self.device, std.mem.sliceAsBytes(hidden_host));
        }

        const active_rows_u32: u32 = @intCast(rows);
        const seq_len_u32: u32 = @intCast(pos_base + rows);
        const last_position = pos_base + rows - 1;
        const last_position_u32: u32 = @intCast(last_position);

        var final_hidden_rows = self.runtime_buffers.input_dev;
        var layer_idx: usize = 0;
        while (layer_idx < layer_limit) : (layer_idx += 1) {
            const layer = &self.block_runtime.blocks[layer_idx];
            final_hidden_rows = try self.tryExecuteLayerProgram(
                layer,
                slot_index,
                layer_idx,
                d_model_u32,
                head_dim_u32,
                rope_dim_u32,
                n_heads_u32,
                n_kv_heads_u32,
                active_rows_u32,
                seq_len_u32,
                seq_len_u32,
                pos_base,
                last_position,
                last_position_u32,
                global_rope_theta,
                local_rope_theta,
                self.rope_function orelse return error.CudaKernelUnavailable,
                self.copy_function orelse return error.CudaKernelUnavailable,
                if (kv_cache_dtype_fp16)
                    (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
                else
                    null,
                if (kv_cache_dtype_fp16) self.kv_write_f16_function else null,
                if (kv_cache_dtype_fp16) self.rope_store_f16_function else null,
                self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                attention_kernels,
                null,
            );
        }

        // Extract logits from the last row of the final chunk.
        if (pos_base + rows >= total_rows) {
            const last_row_in_chunk = rows - 1;
            const last_offset = std.math.mul(usize, last_row_in_chunk, row_bytes) catch return error.InvalidArgument;
            var last_hidden = try bufferSlice(&final_hidden_rows, last_offset, row_bytes);
            var last_norm = try bufferSlice(&self.runtime_buffers.norm_out_dev, 0, row_bytes);
            try compute.cuda.rmsnorm.runWithFunction(
                &self.kernel_arg_pack,
                &self.device,
                self.rmsnorm_function orelse return error.CudaKernelUnavailable,
                &last_hidden,
                &self.runtime_buffers.norm_weight_dev,
                &last_norm,
                1,
                @intCast(self.d_model),
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );
            if (trace.isEnabled()) {
                try last_norm.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.hidden_host));
                trace.emitFinal(
                    .final_norm,
                    @intCast(last_position),
                    1,
                    @ptrCast(self.runtime_buffers.hidden_host.ptr),
                    .f32,
                    .{ @intCast(self.d_model), 0, 0, 0 },
                    1,
                    "cuda_final_norm_host",
                );
            }

            try engine_ops.linearForwardRows(self, &last_norm, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);
            try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));
            if (trace.isEnabled()) {
                const rows128: u128 = 1;
                const d_model128: u128 = @intCast(self.d_model);
                const vocab128: u128 = @intCast(self.runtime_buffers.projected_vocab);
                const total_flops = saturatingU64FromU128(2 * rows128 * d_model128 * vocab128);
                const total_bytes = saturatingU64FromU128(
                    rows128 * d_model128 * @sizeOf(f32) +
                        @as(u128, self.runtime_buffers.projection_weight.byteSize()) +
                        rows128 * vocab128 * @sizeOf(f32),
                );
                const kernel_name = switch (self.runtime_buffers.projection_weight) {
                    .dense_f32 => "matmul_lm_head_f32_host",
                    .dense_u16 => |w| switch (w.dtype) {
                        .bf16 => "matmul_lm_head_bf16_host",
                        .f16 => "matmul_lm_head_f16_host",
                    },
                    .gaffine_u4 => "matmul_lm_head_gaffine_u4_host",
                    .gaffine_u8 => "matmul_lm_head_gaffine_u8_host",
                };
                trace.emitFinalWithWork(
                    .lm_head,
                    @intCast(last_position),
                    0,
                    @ptrCast(self.runtime_buffers.projected_logits_host.ptr),
                    .f32,
                    .{ @intCast(self.runtime_buffers.projected_vocab), 0, 0, 0 },
                    1,
                    kernel_name,
                    .{ .flops = total_flops, .bytes = total_bytes },
                );
            }

            if (self.runtime_buffers.projected_vocab == logits_out.len) {
                @memcpy(logits_out, self.runtime_buffers.projected_logits_host);
            } else {
                @memset(logits_out, -1.0e9);
                @memcpy(logits_out[0..self.runtime_buffers.projected_vocab], self.runtime_buffers.projected_logits_host);
            }
            if (self.loaded.config.logits_scaling != 1.0) {
                for (logits_out) |*v| {
                    v.* /= self.loaded.config.logits_scaling;
                }
                if (trace.isEnabled()) {
                    trace.emitFinal(
                        .logits_scaled,
                        @intCast(last_position),
                        0,
                        @ptrCast(logits_out.ptr),
                        .f32,
                        .{ @intCast(self.vocab_size), 0, 0, 0 },
                        1,
                        null,
                    );
                }
            }
        }

        pos_base += rows;
    }
}

pub fn ensureKvCapacity(self: anytype, required_tokens: usize) !void {
    if (required_tokens == 0) return;
    if (required_tokens > self.max_seq_len) return error.InvalidArgument;
    const copy_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        null
    else
        (self.copy_function orelse return error.CudaKernelUnavailable);
    const copy_u16_function: ?compute.cuda.Function = if (kv_cache_dtype_fp16)
        (self.copy_u16_function orelse return error.CudaKernelUnavailable)
    else
        null;

    for (self.block_runtime.blocks) |*layer| {
        const block = layer.attention_binding orelse continue;
        if (required_tokens <= block.kv_capacity) continue;
        if (self.fixed_alloc_mode) return error.OutOfMemory;

        var new_capacity = block.kv_capacity;
        if (new_capacity == 0) new_capacity = 1;
        while (new_capacity < required_tokens) {
            const doubled = std.math.mul(usize, new_capacity, 2) catch self.max_seq_len;
            const next = if (doubled > new_capacity) doubled else self.max_seq_len;
            new_capacity = @min(self.max_seq_len, next);
            if (new_capacity == self.max_seq_len) break;
        }
        if (new_capacity < required_tokens) return error.InvalidArgument;

        var new_kv_pair = try self.allocKvPair(new_capacity, block.kv_dim);
        errdefer {
            new_kv_pair.v.deinit(&self.device);
            new_kv_pair.k.deinit(&self.device);
        }

        if (block.kv_capacity > 0) {
            const old_elems = std.math.mul(usize, block.kv_capacity, block.kv_dim) catch return error.InvalidArgument;
            const old_count_u32: u32 = @intCast(old_elems);
            if (kv_cache_dtype_fp16) {
                try compute.cuda.copy_u16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_u16_function.?,
                    &block.k_cache,
                    &new_kv_pair.k,
                    old_count_u32,
                );
                try compute.cuda.copy_u16.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_u16_function.?,
                    &block.v_cache,
                    &new_kv_pair.v,
                    old_count_u32,
                );
            } else {
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function.?,
                    &block.k_cache,
                    &new_kv_pair.k,
                    old_count_u32,
                );
                try compute.cuda.copy.runWithFunction(
                    &self.kernel_arg_pack,
                    &self.device,
                    copy_function.?,
                    &block.v_cache,
                    &new_kv_pair.v,
                    old_count_u32,
                );
            }
        }

        block.k_cache.deinit(&self.device);
        block.v_cache.deinit(&self.device);
        block.k_cache = new_kv_pair.k;
        block.v_cache = new_kv_pair.v;
        block.kv_capacity = new_capacity;
    }
}

pub fn resetShortConvStates(self: anytype) !void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.shortconv_binding orelse continue;
        const elems = std.math.divExact(usize, block.conv_state.size, @sizeOf(f32)) catch return error.InvalidArgument;
        const zeros = try self.allocator.alloc(f32, elems);
        defer self.allocator.free(zeros);
        @memset(zeros, 0.0);
        try block.conv_state.upload(&self.device, std.mem.sliceAsBytes(zeros));
    }
}

pub fn resetGatedDeltaStates(self: anytype) void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.gated_delta_binding orelse continue;
        block.state.reset();
        block.conv_ring_head = 0;
        const conv_elems = std.math.divExact(usize, block.conv_state_dev.size, @sizeOf(f32)) catch continue;
        const conv_zeros = self.allocator.alloc(f32, conv_elems) catch continue;
        defer self.allocator.free(conv_zeros);
        @memset(conv_zeros, 0.0);
        block.conv_state_dev.upload(&self.device, std.mem.sliceAsBytes(conv_zeros)) catch {};
        const ssm_elems = std.math.divExact(usize, block.ssm_state_dev.size, @sizeOf(f32)) catch continue;
        const zeros = self.allocator.alloc(f32, ssm_elems) catch continue;
        defer self.allocator.free(zeros);
        @memset(zeros, 0.0);
        block.ssm_state_dev.upload(&self.device, std.mem.sliceAsBytes(zeros)) catch {};
    }
}

pub fn resetAttentionCpuStates(self: anytype) void {
    for (self.block_runtime.blocks) |*layer| {
        const block = layer.attention_binding orelse continue;
        if (block.cpu_cache) |*cache| cache.resetCache();
    }
}

pub fn ensureGatedDeltaHostStageCapacity(self: anytype, elements: usize) !void {
    if (elements == 0) return error.InvalidArgument;
    if (self.gated_delta_stage_input_host.len < elements) {
        if (self.gated_delta_stage_input_host.len > 0) self.allocator.free(self.gated_delta_stage_input_host);
        self.gated_delta_stage_input_host = try self.allocator.alloc(f32, elements);
    }
    if (self.gated_delta_stage_mid_host.len < elements) {
        if (self.gated_delta_stage_mid_host.len > 0) self.allocator.free(self.gated_delta_stage_mid_host);
        self.gated_delta_stage_mid_host = try self.allocator.alloc(f32, elements);
    }
    if (self.gated_delta_stage_output_host.len < elements) {
        if (self.gated_delta_stage_output_host.len > 0) self.allocator.free(self.gated_delta_stage_output_host);
        self.gated_delta_stage_output_host = try self.allocator.alloc(f32, elements);
    }
}

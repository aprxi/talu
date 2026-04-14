//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("tensor_pkg");
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const staged_orchestrator = @import("../staged_orchestrator.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("engine_types.zig");
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KvCacheDtype = engine_types.KvCacheDtype;
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

/// Download and log first N f32 values + L2 norm from a device buffer.
/// Gated by TALU_DUMP_HIDDEN env var. Uses log.warn so it survives ReleaseFast.
fn dumpHiddenState(
    self: anytype,
    buf: *const compute.cuda.Buffer,
    global_layer_idx: usize,
    label: []const u8,
    d_model: usize,
    rows: usize,
) void {
    const dump_env = std.posix.getenv("TALU_DUMP_HIDDEN");
    if (dump_env == null) return;
    _ = rows;

    // Sync the entire CUDA context first.
    self.device.synchronize() catch |e| {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{ .layer = global_layer_idx, .label = label, .reason = "sync_err", .err = @intFromError(e) });
        return;
    };

    // Download full row to hidden_host using raw cu_memcpy_dtoh.
    const n = @min(d_model, self.runtime_buffers.hidden_host.len);
    if (n == 0) return;
    const download_bytes = n * @sizeOf(f32);
    if (buf.pointer == 0 or buf.size < download_bytes) {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{
            .layer = global_layer_idx,
            .label = label,
            .reason = "bad_buf",
            .ptr = buf.pointer,
            .buf_size = buf.size,
            .need = download_bytes,
        });
        return;
    }

    self.device.makeCurrent() catch |e| {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{ .layer = global_layer_idx, .label = label, .reason = "make_current_err", .err = @intFromError(e) });
        return;
    };

    const host_ptr: *anyopaque = @ptrCast(self.runtime_buffers.hidden_host.ptr);
    const rc = self.device.api.cu_memcpy_dtoh(host_ptr, buf.pointer, download_bytes);
    if (rc != 0) {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{
            .layer = global_layer_idx,
            .label = label,
            .reason = "cu_memcpy_rc",
            .rc = rc,
            .dev_ptr = buf.pointer,
            .buf_size = buf.size,
            .download_bytes = download_bytes,
        });
        return;
    }

    const host = self.runtime_buffers.hidden_host[0..n];
    var sum: f64 = 0.0;
    for (host) |v| sum += @as(f64, v) * @as(f64, v);
    const l2_norm: f32 = @floatCast(@sqrt(sum));

    const n8 = @min(8, n);
    var v: [8]f32 = .{0} ** 8;
    @memcpy(v[0..n8], host[0..n8]);

    log.warn("inference", "DUMP_HIDDEN", .{
        .layer = global_layer_idx,
        .label = label,
        .l2_norm = l2_norm,
        .v0 = v[0],
        .v1 = v[1],
        .v2 = v[2],
        .v3 = v[3],
        .v4 = v[4],
        .v5 = v[5],
        .v6 = v[6],
        .v7 = v[7],
    });
}

fn applyHostLogitsPostProcess(
    logits: []f32,
    logits_scaling: f32,
    final_logit_softcapping: f32,
) void {
    if (logits_scaling != 1.0) {
        for (logits) |*value| {
            value.* /= logits_scaling;
        }
    }
    if (final_logit_softcapping > 0.0) {
        for (logits) |*value| {
            value.* = std.math.tanh(value.* / final_logit_softcapping) * final_logit_softcapping;
        }
    }
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

/// Upload CPU source layer KV data to GPU mirror buffers for cross-device
/// KV sharing. After CPU forward populates KV cache for source layers, this
/// converts f32 to the GPU KV dtype and uploads to GPU mirror entries so
/// GPU attention layers can read shared KV.
fn uploadCpuKvToMirrors(
    gpu_backend: anytype,
    cpu_backend: anytype,
    slot_index: usize,
    pos_start: usize,
    n_positions: usize,
) !void {
    const GpuType = @TypeOf(gpu_backend.*);
    if (comptime !@hasField(GpuType, "block_runtime")) return;
    const BrtType = @TypeOf(gpu_backend.block_runtime);
    if (comptime !@hasField(BrtType, "replicated_kv_sources")) return;
    const replicated = gpu_backend.block_runtime.replicated_kv_sources;
    if (replicated.len == 0) return;

    const mirrors = gpu_backend.block_runtime.mirror_kv;
    const head_dim = gpu_backend.head_dim;
    const allocator = gpu_backend.allocator;

    // Only f16 KV cache is supported for cross-device mirrors.
    if (gpu_backend.kv_cache_dtype != .f16) return error.UnsupportedModel;

    for (replicated, 0..) |src, mi| {
        const mk = &mirrors[mi];
        const cpu_kv = cpu_backend.kv_cache.getLayer(src.global_layer_idx);
        const n_kv_heads = src.kv_dim / head_dim;
        const kv_dim = src.kv_dim;
        const total_elems = n_positions * kv_dim;
        const total_bytes = total_elems * @sizeOf(f16);

        const staging_f16 = try allocator.alloc(f16, total_elems);
        defer allocator.free(staging_f16);

        // Convert + transpose K: CPU [slot][head][pos][dim] → GPU [pos][head*dim]
        for (0..n_positions) |pi| {
            const pos = pos_start + pi;
            const row_off = pi * kv_dim;
            for (0..n_kv_heads) |kh| {
                const cpu_k = cpu_kv.getK(slot_index, kh, pos);
                const dst_off = row_off + kh * head_dim;
                for (0..head_dim) |di| {
                    staging_f16[dst_off + di] = @floatCast(cpu_k[di]);
                }
            }
        }
        const gpu_k_offset = pos_start * kv_dim * @sizeOf(f16);
        const k_slice = try bufferSlice(&mk.k, gpu_k_offset, total_bytes);
        try k_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));

        // Convert + transpose V.
        for (0..n_positions) |pi| {
            const pos = pos_start + pi;
            const row_off = pi * kv_dim;
            for (0..n_kv_heads) |kh| {
                const cpu_v = cpu_kv.getV(slot_index, kh, pos);
                const dst_off = row_off + kh * head_dim;
                for (0..head_dim) |di| {
                    staging_f16[dst_off + di] = @floatCast(cpu_v[di]);
                }
            }
        }
        const gpu_v_offset = pos_start * kv_dim * @sizeOf(f16);
        const v_slice = try bufferSlice(&mk.v, gpu_v_offset, total_bytes);
        try v_slice.upload(&gpu_backend.device, std.mem.sliceAsBytes(staging_f16));
    }
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
fn transferPipelineActivationMultiRow(self: anytype, dst: anytype, total_bytes: usize) !void {
    if (total_bytes == 0) return;
    switch (self.pipeline_transfer_mode) {
        .peer_to_peer => {
            // Prefer event-based non-blocking transfer to avoid host stalls.
            if (self.pipeline_stage0_event) |event| {
                try self.device.recordEvent(event, self.compute_stream);
                try dst.device.streamWaitEvent(dst.compute_stream, event);
                try dst.device.makeCurrent();
                try self.device.memcpyPeerAsync(
                    dst.runtime_buffers.input_dev.pointer,
                    dst.device.context,
                    self.runtime_buffers.input_dev.pointer,
                    self.device.context,
                    total_bytes,
                    dst.compute_stream,
                );
            } else {
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
            }
        },
        .host_staged => {
            const staging = self.pipeline_host_staging orelse return error.PipelineTransferNotInitialized;
            if (staging.len == 0) return error.PipelineTransferBufferTooSmall;
            var offset: usize = 0;
            while (offset < total_bytes) {
                const chunk = @min(staging.len, total_bytes - offset);
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

/// Transfer stage1->stage2 activations for cpu+gpu+gpu prefill.
/// Uses a single peer copy when possible; otherwise uses host staging in
/// chunks up to `pipeline_host_staging_stage12.len`.
fn transferPipelineActivationStage12MultiRow(self: anytype, src: anytype, total_bytes: usize) !void {
    if (total_bytes == 0) return;

    if (self.device.canAccessPeer(&src.device)) {
        // Best effort: peer access can already be enabled or unavailable at runtime.
        self.device.enablePeerAccess(&src.device) catch {};
        src.device.enablePeerAccess(&self.device) catch {};
        if (src.device.memcpyPeerAsync(
            self.runtime_buffers.input_dev.pointer,
            self.device.context,
            src.runtime_buffers.input_dev.pointer,
            src.device.context,
            total_bytes,
            src.compute_stream,
        )) {
            if (src.compute_stream) |stream| {
                try src.device.synchronizeStream(stream);
            } else {
                try src.device.synchronize();
            }
            return;
        } else |_| {
            // Fall through to host-staged transfer.
        }
    }

    const staging = self.pipeline_host_staging_stage12 orelse return error.PipelineTransferNotInitialized;
    if (staging.len == 0) return error.PipelineTransferBufferTooSmall;

    var offset: usize = 0;
    while (offset < total_bytes) {
        const chunk = @min(staging.len, total_bytes - offset);
        var src_slice = try bufferSlice(&src.runtime_buffers.input_dev, offset, chunk);
        try src_slice.download(&src.device, staging[0..chunk]);
        var dst_slice = try bufferSlice(&self.runtime_buffers.input_dev, offset, chunk);
        try dst_slice.upload(&self.device, staging[0..chunk]);
        offset += chunk;
    }
}

fn buildAttentionKernelSet(backend: anytype) !AttentionKernelSet {
    return switch (backend.kv_cache_dtype) {
        .f16 => .{
            .attn_scores_heads_f16_kv_function = backend.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = backend.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = backend.attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = backend.attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = backend.attn_fused_prefill_heads_f16_kv_gqa_function,
            .softmax_rows_function = backend.softmax_rows_function,
            .causal_attn_softmax_f32_function = backend.causal_attn_softmax_f32_function,
        },
        .i8 => .{
            .attn_scores_heads_i8_kv_function = backend.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = backend.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = backend.attn_fused_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_function = backend.attn_fused_prefill_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_gqa_function = backend.attn_fused_prefill_heads_i8_kv_gqa_function,
            .softmax_rows_function = backend.softmax_rows_function,
            .causal_attn_softmax_f32_function = backend.causal_attn_softmax_f32_function,
        },
        .fp8 => .{
            .attn_scores_heads_fp8_kv_function = backend.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = backend.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = backend.attn_fused_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_function = backend.attn_fused_prefill_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_gqa_function = backend.attn_fused_prefill_heads_fp8_kv_gqa_function,
            .softmax_rows_function = backend.softmax_rows_function,
            .causal_attn_softmax_f32_function = backend.causal_attn_softmax_f32_function,
        },
    };
}

/// Batched prefill for cpu_gpu topology. Runs all tokens through CPU layers
/// [0, split_layer) in one batched pass, then uploads activations in chunks
/// to the GPU for processing through GPU layers.
fn computeBatchedPrefillCpuGpu(
    self: anytype,
    cpu_stage0: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;

    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);

    const total_rows = tokens.len;
    const d_model = self.d_model;

    // ── GPU setup ──
    try ensureKvCapacity(self, total_rows);
    try resetShortConvStates(self);
    resetAttentionCpuStates(self);
    resetGatedDeltaStates(self);

    const row_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const d_model_u32: u32 = @intCast(d_model);
    const head_dim_u32: u32 = @intCast(self.head_dim);
    const rope_dim_u32: u32 = @intCast(self.rope_dim);
    const n_heads_u32: u32 = @intCast(self.n_heads);
    const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;

    const attention_kernels: AttentionKernelSet = switch (self.kv_cache_dtype) {
        .f16 => .{
            .attn_scores_heads_f16_kv_function = self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = self.attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = self.attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = self.attn_fused_prefill_heads_f16_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
        .i8 => .{
            .attn_scores_heads_i8_kv_function = self.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = self.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = self.attn_fused_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_function = self.attn_fused_prefill_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_gqa_function = self.attn_fused_prefill_heads_i8_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
        .fp8 => .{
            .attn_scores_heads_fp8_kv_function = self.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = self.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = self.attn_fused_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_function = self.attn_fused_prefill_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_gqa_function = self.attn_fused_prefill_heads_fp8_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
    };

    // ── CPU stage0: batched embed + forward through [0, split_layer) ──
    const prefill_buffer = try self.allocator.alloc(f32, total_rows * d_model);
    defer self.allocator.free(prefill_buffer);

    // For Gemma4 per-layer-input: capture source embeddings (raw scaled
    // embed_tokens output) before CPU layers modify the hidden states.
    const gemma4_active = comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch");
    const has_gemma4 = gemma4_active and self.gemma4_per_layer != null;
    const source_embeddings_host: ?[]f32 = if (has_gemma4)
        try self.allocator.alloc(f32, total_rows * d_model)
    else
        null;
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    try cpu_stage0.prefillSlotLayerRange(slot_index, tokens, prefill_buffer, self.split_layer, source_embeddings_host);

    // Upload CPU source layer KV to GPU mirror buffers for cross-device sharing.
    try uploadCpuKvToMirrors(self, cpu_stage0, slot_index, 0, total_rows);

    // ── GPU stage1: chunked forward through GPU layers ──
    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, self.prefill_chunk_rows_cap);

        try self.runtime_buffers.ensureRowCapacity(&self.device, rows, self.fixed_alloc_mode);
        try self.ensureLayerProgramSlotRowCapacity(rows, self.fixed_alloc_mode);

        // Upload chunk from CPU host buffer to GPU input_dev.
        const chunk_offset = pos_base * d_model;
        const chunk_f32s = rows * d_model;
        try self.runtime_buffers.input_dev.upload(&self.device, std.mem.sliceAsBytes(prefill_buffer[chunk_offset..][0..chunk_f32s]));

        const active_rows_u32: u32 = @intCast(rows);
        const seq_len_u32: u32 = @intCast(pos_base + rows);
        const last_position = pos_base + rows - 1;
        const last_position_u32: u32 = @intCast(last_position);

        var final_hidden_rows = self.runtime_buffers.input_dev;
        var gemma_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (source_embeddings_host) |se_host| {
            // Pipeline mode: upload source embeddings from CPU to deepstack_add_dev.
            const se_chunk_offset = pos_base * d_model;
            const se_chunk_f32s = rows * d_model;
            const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
            var se_dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, se_bytes);
            try se_dst.upload(&self.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
            gemma_source_embeddings_opt = se_dst;
        } else if (comptime @hasDecl(SelfType, "maybeCaptureGemma4SourceEmbeddings")) {
            gemma_source_embeddings_opt = try self.maybeCaptureGemma4SourceEmbeddings(rows);
        }
        var layer_idx: usize = 0;
        while (layer_idx < self.block_runtime.blocks.len) : (layer_idx += 1) {
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
                if (self.kv_cache_dtype == .f16) (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                if (self.kv_cache_dtype == .f16) self.kv_write_f16_function else null,
                if (self.kv_cache_dtype == .f16) self.rope_store_f16_function else null,
                self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                attention_kernels,
                null,
            );
            dumpHiddenState(self, &self.runtime_buffers.input_dev, self.split_layer + layer_idx, "post_layer", self.d_model, 1);
            if (comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch")) {
                if (gemma_source_embeddings_opt) |*gemma_source_embeddings| {
                    const chunk_tokens = tokens[pos_base .. pos_base + rows];
                    try self.applyGemma4PerLayerBranch(
                        layer_idx,
                        chunk_tokens,
                        gemma_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden_rows = self.runtime_buffers.input_dev;
                    dumpHiddenState(self, &self.runtime_buffers.input_dev, self.split_layer + layer_idx, "post_ple", self.d_model, 1);
                } else if (comptime @hasDecl(SelfType, "applyStandaloneLayerScalar")) {
                    try self.applyStandaloneLayerScalar(layer_idx, &self.runtime_buffers.input_dev, rows);
                }
            }
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
                @intCast(d_model),
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );
            try engine_ops.linearForwardRows(self, &last_norm, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);
            try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));

            if (self.runtime_buffers.projected_vocab == logits_out.len) {
                @memcpy(logits_out, self.runtime_buffers.projected_logits_host);
            } else {
                @memset(logits_out, -1.0e9);
                @memcpy(logits_out[0..self.runtime_buffers.projected_vocab], self.runtime_buffers.projected_logits_host);
            }
            applyHostLogitsPostProcess(logits_out, self.loaded.config.logits_scaling, self.loaded.config.final_logit_softcapping);
        }

        pos_base += rows;
    }
}

/// Batched prefill for cpu_gpu_gpu topology. CPU stage0 processes all tokens
/// through layers [0, split_layer). Then GPU1 (gpu_stage1, middle layers) and
/// GPU2 (self, last layers) process their layer ranges in chunks with bulk
/// transfer between them.
///
/// Stage mapping:
///   self        = GPU2 (stage 2, layers [split_stage2, total))
///   gpu_stage1  = GPU1 (stage 1, layers [split, split_stage2))
///   cpu_stage0  = CPU  (stage 0, layers [0, split))
fn computeBatchedPrefillCpuGpuGpu(
    self: anytype,
    cpu_stage0: anytype,
    gpu_stage1: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    if (comptime !@hasField(@TypeOf(gpu_stage1.*), "device")) return error.InvalidArgument;

    const previous_launch_phase1 = gpu_stage1.device.setLaunchPhase(.prefill);
    defer _ = gpu_stage1.device.setLaunchPhase(previous_launch_phase1);
    const previous_launch_phase2 = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase2);

    const total_rows = tokens.len;
    const d_model = self.d_model;

    // ── GPU1 + GPU2 setup ──
    try ensureKvCapacity(gpu_stage1, total_rows);
    try ensureKvCapacity(self, total_rows);
    try resetShortConvStates(gpu_stage1);
    try resetShortConvStates(self);
    resetAttentionCpuStates(gpu_stage1);
    resetAttentionCpuStates(self);
    resetGatedDeltaStates(gpu_stage1);
    resetGatedDeltaStates(self);

    const row_bytes = std.math.mul(usize, d_model, @sizeOf(f32)) catch return error.InvalidArgument;
    const d_model_u32: u32 = @intCast(d_model);
    const head_dim_u32: u32 = @intCast(self.head_dim);
    const rope_dim_u32: u32 = @intCast(self.rope_dim);
    const n_heads_u32: u32 = @intCast(self.n_heads);
    const n_kv_heads_u32: u32 = @intCast(self.n_kv_heads);
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;

    const attn_kernels_1 = buildAttentionKernelSet(gpu_stage1) catch return error.CudaKernelUnavailable;
    const attn_kernels_2 = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;

    // Use min of both GPU chunk caps.
    const chunk_cap = @min(self.prefill_chunk_rows_cap, gpu_stage1.prefill_chunk_rows_cap);

    // ── CPU stage0: batched embed + forward through [0, split_layer) ──
    const prefill_buffer = try self.allocator.alloc(f32, total_rows * d_model);
    defer self.allocator.free(prefill_buffer);

    // For Gemma4 per-layer-input: capture source embeddings from CPU.
    const gemma4_active_1 = comptime @hasDecl(@TypeOf(gpu_stage1.*), "applyGemma4PerLayerBranch");
    const gemma4_active_2 = comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch");
    const has_gemma4_1 = gemma4_active_1 and gpu_stage1.gemma4_per_layer != null;
    const has_gemma4_2 = gemma4_active_2 and self.gemma4_per_layer != null;
    const need_source_embeddings = has_gemma4_1 or has_gemma4_2;
    const source_embeddings_host: ?[]f32 = if (need_source_embeddings)
        try self.allocator.alloc(f32, total_rows * d_model)
    else
        null;
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    try cpu_stage0.prefillSlotLayerRange(slot_index, tokens, prefill_buffer, self.split_layer, source_embeddings_host);

    // ── GPU1 → GPU2: chunked forward ──
    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, chunk_cap);

        const active_rows_u32: u32 = @intCast(rows);
        const seq_len_u32: u32 = @intCast(pos_base + rows);
        const last_position = pos_base + rows - 1;
        const last_position_u32: u32 = @intCast(last_position);

        // GPU1: upload CPU output + layer loop.
        try gpu_stage1.runtime_buffers.ensureRowCapacity(&gpu_stage1.device, rows, gpu_stage1.fixed_alloc_mode);
        try gpu_stage1.ensureLayerProgramSlotRowCapacity(rows, gpu_stage1.fixed_alloc_mode);

        const chunk_offset = pos_base * d_model;
        const chunk_f32s = rows * d_model;
        try gpu_stage1.runtime_buffers.input_dev.upload(&gpu_stage1.device, std.mem.sliceAsBytes(prefill_buffer[chunk_offset..][0..chunk_f32s]));

        // Upload source embeddings for GPU1 Gemma4 per-layer branch.
        var gemma_source_embeddings_1: ?compute.cuda.Buffer = null;
        if (has_gemma4_1) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&gpu_stage1.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&gpu_stage1.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                gemma_source_embeddings_1 = se_dst;
            }
        }

        {
            var layer_idx: usize = 0;
            while (layer_idx < gpu_stage1.block_runtime.blocks.len) : (layer_idx += 1) {
                const layer = &gpu_stage1.block_runtime.blocks[layer_idx];
                _ = try gpu_stage1.tryExecuteLayerProgram(
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
                    gpu_stage1.rope_function orelse return error.CudaKernelUnavailable,
                    gpu_stage1.copy_function orelse return error.CudaKernelUnavailable,
                    if (gpu_stage1.kv_cache_dtype == .f16) (gpu_stage1.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                    if (gpu_stage1.kv_cache_dtype == .f16) gpu_stage1.kv_write_f16_function else null,
                    if (gpu_stage1.kv_cache_dtype == .f16) gpu_stage1.rope_store_f16_function else null,
                    gpu_stage1.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_1,
                    null,
                );
                if (comptime gemma4_active_1) {
                    if (gemma_source_embeddings_1) |*gemma_se| {
                        const chunk_tokens = tokens[pos_base .. pos_base + rows];
                        try gpu_stage1.applyGemma4PerLayerBranch(
                            layer_idx,
                            chunk_tokens,
                            gemma_se,
                            &gpu_stage1.runtime_buffers.input_dev,
                        );
                    } else if (comptime @hasDecl(@TypeOf(gpu_stage1.*), "applyStandaloneLayerScalar")) {
                        try gpu_stage1.applyStandaloneLayerScalar(layer_idx, &gpu_stage1.runtime_buffers.input_dev, rows);
                    }
                }
            }
        }

        // GPU2: transfer from GPU1 + layer loop.
        try self.runtime_buffers.ensureRowCapacity(&self.device, rows, self.fixed_alloc_mode);
        try self.ensureLayerProgramSlotRowCapacity(rows, self.fixed_alloc_mode);

        // Bulk stage1→stage2 transfer for the full chunk.
        const transfer_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
        try transferPipelineActivationStage12MultiRow(self, gpu_stage1, transfer_bytes);

        // Upload source embeddings for GPU2 Gemma4 per-layer branch.
        var gemma_source_embeddings_2: ?compute.cuda.Buffer = null;
        if (has_gemma4_2) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&self.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                gemma_source_embeddings_2 = se_dst;
            }
        }

        var final_hidden_rows = self.runtime_buffers.input_dev;
        {
            var layer_idx: usize = 0;
            while (layer_idx < self.block_runtime.blocks.len) : (layer_idx += 1) {
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
                    if (self.kv_cache_dtype == .f16) (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                    if (self.kv_cache_dtype == .f16) self.kv_write_f16_function else null,
                    if (self.kv_cache_dtype == .f16) self.rope_store_f16_function else null,
                    self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_2,
                    null,
                );
                if (comptime gemma4_active_2) {
                    if (gemma_source_embeddings_2) |*gemma_se| {
                        const chunk_tokens = tokens[pos_base .. pos_base + rows];
                        try self.applyGemma4PerLayerBranch(
                            layer_idx,
                            chunk_tokens,
                            gemma_se,
                            &self.runtime_buffers.input_dev,
                        );
                        final_hidden_rows = self.runtime_buffers.input_dev;
                    } else if (comptime @hasDecl(SelfType, "applyStandaloneLayerScalar")) {
                        try self.applyStandaloneLayerScalar(layer_idx, &self.runtime_buffers.input_dev, rows);
                    }
                }
            }
        }

        // Logits from last chunk (on GPU2).
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
                @intCast(d_model),
                self.norm_eps,
                self.loaded.runtime.weight_offset,
            );
            try engine_ops.linearForwardRows(self, &last_norm, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);
            try self.runtime_buffers.logits_dev.download(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.projected_logits_host));

            if (self.runtime_buffers.projected_vocab == logits_out.len) {
                @memcpy(logits_out, self.runtime_buffers.projected_logits_host);
            } else {
                @memset(logits_out, -1.0e9);
                @memcpy(logits_out[0..self.runtime_buffers.projected_vocab], self.runtime_buffers.projected_logits_host);
            }
            applyHostLogitsPostProcess(logits_out, self.loaded.config.logits_scaling, self.loaded.config.final_logit_softcapping);
        }

        pos_base += rows;
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

    const attn_kernels_0 = switch (self.kv_cache_dtype) {
        .f16 => AttentionKernelSet{
            .attn_scores_heads_f16_kv_function = self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = self.attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = self.attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = self.attn_fused_prefill_heads_f16_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
        .i8 => AttentionKernelSet{
            .attn_scores_heads_i8_kv_function = self.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = self.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = self.attn_fused_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_function = self.attn_fused_prefill_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_gqa_function = self.attn_fused_prefill_heads_i8_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
        .fp8 => AttentionKernelSet{
            .attn_scores_heads_fp8_kv_function = self.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = self.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = self.attn_fused_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_function = self.attn_fused_prefill_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_gqa_function = self.attn_fused_prefill_heads_fp8_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
    };
    const attn_kernels_1 = switch (self.kv_cache_dtype) {
        .f16 => AttentionKernelSet{
            .attn_scores_heads_f16_kv_function = stage1.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = stage1.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = stage1.attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = stage1.attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = stage1.attn_fused_prefill_heads_f16_kv_gqa_function,
            .softmax_rows_function = stage1.softmax_rows_function,
            .causal_attn_softmax_f32_function = stage1.causal_attn_softmax_f32_function,
        },
        .i8 => AttentionKernelSet{
            .attn_scores_heads_i8_kv_function = self.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = self.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = self.attn_fused_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_function = self.attn_fused_prefill_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_gqa_function = self.attn_fused_prefill_heads_i8_kv_gqa_function,
            .softmax_rows_function = stage1.softmax_rows_function,
            .causal_attn_softmax_f32_function = stage1.causal_attn_softmax_f32_function,
        },
        .fp8 => AttentionKernelSet{
            .attn_scores_heads_fp8_kv_function = self.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = self.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = self.attn_fused_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_function = self.attn_fused_prefill_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_gqa_function = self.attn_fused_prefill_heads_fp8_kv_gqa_function,
            .softmax_rows_function = stage1.softmax_rows_function,
            .causal_attn_softmax_f32_function = stage1.causal_attn_softmax_f32_function,
        },
    };

    const chunk_cap = @min(self.prefill_chunk_rows_cap, stage1.prefill_chunk_rows_cap);

    // ── Gemma4 per-layer input: compute source embeddings on host ──
    const Stage1Type = @TypeOf(stage1.*);
    const has_gemma4_0 = comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch");
    const has_gemma4_1 = comptime @hasDecl(Stage1Type, "applyGemma4PerLayerBranch");
    const gemma4_active_0 = has_gemma4_0 and (if (comptime @hasField(SelfType, "gemma4_per_layer")) self.gemma4_per_layer != null else false);
    const gemma4_active_1 = has_gemma4_1 and (if (comptime @hasField(Stage1Type, "gemma4_per_layer")) stage1.gemma4_per_layer != null else false);
    const need_source_embeddings = gemma4_active_0 or gemma4_active_1;
    const d_model = self.d_model;
    const source_embeddings_host: ?[]f32 = if (need_source_embeddings)
        try self.allocator.alloc(f32, total_rows * d_model)
    else
        null;
    defer if (source_embeddings_host) |buf| self.allocator.free(buf);

    if (source_embeddings_host) |se_host| {
        try populatePrefillHiddenFromTokens(self.loaded, tokens, d_model, se_host, null);
    }

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
                        self.prefill_rope_positions_cached_dirty = true;
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

        // Upload source embeddings for this chunk to stage0's deepstack_add_dev.
        var gemma_source_embeddings_0: ?compute.cuda.Buffer = null;
        if (gemma4_active_0) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&self.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                gemma_source_embeddings_0 = se_dst;
            }
        }

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
                    if (self.kv_cache_dtype == .f16) (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                    if (self.kv_cache_dtype == .f16) self.kv_write_f16_function else null,
                    if (self.kv_cache_dtype == .f16) self.rope_store_f16_function else null,
                    self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_0,
                    null,
                );
                if (comptime has_gemma4_0) {
                    if (gemma_source_embeddings_0) |*gemma_se| {
                        try self.applyGemma4PerLayerBranch(
                            layer_idx,
                            chunk_tokens,
                            gemma_se,
                            &self.runtime_buffers.input_dev,
                        );
                    } else if (comptime @hasDecl(SelfType, "applyStandaloneLayerScalar")) {
                        try self.applyStandaloneLayerScalar(layer_idx, &self.runtime_buffers.input_dev, rows);
                    }
                }
            }
        }

        // ── Stage 1 buffer setup (must precede transfer into input_dev) ──
        try stage1.runtime_buffers.ensureRowCapacity(&stage1.device, rows, stage1.fixed_alloc_mode);
        try stage1.ensureLayerProgramSlotRowCapacity(rows, stage1.fixed_alloc_mode);

        // ── Bulk transfer stage0 → stage1 ──
        const transfer_bytes = std.math.mul(usize, rows, row_bytes) catch return error.InvalidArgument;
        try transferPipelineActivationMultiRow(self, stage1, transfer_bytes);

        // Upload source embeddings for this chunk to stage1's deepstack_add_dev.
        var gemma_source_embeddings_1: ?compute.cuda.Buffer = null;
        if (gemma4_active_1) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&stage1.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&stage1.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                gemma_source_embeddings_1 = se_dst;
            }
        }

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
                    if (self.kv_cache_dtype == .f16) (stage1.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                    if (self.kv_cache_dtype == .f16) stage1.kv_write_f16_function else null,
                    if (self.kv_cache_dtype == .f16) stage1.rope_store_f16_function else null,
                    stage1.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_1,
                    null,
                );
                if (comptime has_gemma4_1) {
                    if (gemma_source_embeddings_1) |*gemma_se| {
                        try stage1.applyGemma4PerLayerBranch(
                            layer_idx,
                            chunk_tokens,
                            gemma_se,
                            &stage1.runtime_buffers.input_dev,
                        );
                        stage1_final_hidden = stage1.runtime_buffers.input_dev;
                    } else if (comptime @hasDecl(Stage1Type, "applyStandaloneLayerScalar")) {
                        try stage1.applyStandaloneLayerScalar(layer_idx, &stage1.runtime_buffers.input_dev, rows);
                    }
                }
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
                    .fp8 => "matmul_lm_head_fp8_host",
                    .mxfp8 => "matmul_lm_head_mxfp8_host",
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
            applyHostLogitsPostProcess(
                logits_out,
                stage1.loaded.config.logits_scaling,
                stage1.loaded.config.final_logit_softcapping,
            );
            if (stage1.loaded.config.logits_scaling != 1.0 and trace.isEnabled()) {
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

        /// Event-based transfer records an event on stage0's stream and makes
        /// stage1's stream wait on it, so the caller can skip explicit stage0
        /// synchronization.
        pub fn handlesStageSync(t: *const @This()) bool {
            return t.owner.pipeline_stage0_event != null;
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
        gpu_backend: @TypeOf(self),
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

        pub fn synchronize(stage: *@This()) anyerror!void {
            // Upload CPU source layer KV to GPU mirrors after CPU forward,
            // before GPU stage1 reads KV in attention.
            try uploadCpuKvToMirrors(stage.gpu_backend, stage.backend, stage.ctx.slot_index, stage.ctx.position, 1);
        }

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
    };
    const Stage1 = struct {
        backend: @TypeOf(self),
        ctx: *const Ctx,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (std.posix.getenv("TALU_DUMP_HIDDEN") != null) {
                log.warn("inference", "STAGE1_EXEC", .{
                    .layer_start = layer_start,
                    .layer_end = layer_end,
                    .input_len = input.len,
                });
            }
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
        .{ .backend = cpu_stage0_backend, .gpu_backend = self, .ctx = &ctx },
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
                if (hidden_override == null and deepstack_layer_features_opt == null and deepstack_feature_index_opt == null)
                {
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
                    // In device-only decode mode, keep logits resident on stage1.
                    // Pipeline-aware token-selection/top-k extraction consumes
                    // stage1 logits directly to avoid full-vocab stage1→host→stage0
                    // copies on every token.
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
            // In device-only decode mode, keep logits resident on stage1.
            // Pipeline-aware token-selection/top-k extraction consumes stage1
            // logits directly.
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
            try uploadCpuKvToMirrors(self, stage0, slot_index, position, 1);
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
    // Sync slot_kv_states after KV growth so that loadKvSlot at the end of
    // this function preserves the new pointers/capacity. Without this, the
    // token-by-token prefill loop loses KV data on every iteration because
    // loadKvSlot overwrites block_runtime from stale slot_kv_states.
    self.saveActiveKvSlot();

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
    const is_f16_kv = self.kv_cache_dtype == .f16;
    const cast_f32_to_f16_function: ?compute.cuda.Function = if (is_f16_kv)
        (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const kv_write_f16_function: ?compute.cuda.Function = if (is_f16_kv) self.kv_write_f16_function else null;
    const rope_store_f16_function: ?compute.cuda.Function = if (is_f16_kv) self.rope_store_f16_function else null;
    const rope_function = self.rope_function orelse return error.CudaKernelUnavailable;
    const softmax_rows_function = self.softmax_rows_function;
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

    var gemma_source_embeddings_opt: ?compute.cuda.Buffer = null;
    if (comptime @hasDecl(SelfType, "maybeCaptureGemma4SourceEmbeddings")) {
        if (use_preloaded_input) {
            // Pipeline mode: input_dev has post-CPU-layer hidden states, not raw
            // embeddings. Look up the raw embedding on host and upload.
            if (comptime @hasDecl(SelfType, "captureGemma4SourceEmbeddingsForPipeline")) {
                gemma_source_embeddings_opt = self.captureGemma4SourceEmbeddingsForPipeline(token) catch |err| blk: {
                    log.warn("inference", "CUDA captureGemma4SourceEmbeddingsForPipeline failed", .{
                        .reason = @errorName(err),
                        .token = token,
                    });
                    break :blk null;
                };
            }
        } else {
            gemma_source_embeddings_opt = try self.maybeCaptureGemma4SourceEmbeddings(1);
        }
    }
    const gemma_token_id_single = [_]u32{token};

    // Pipeline stage boundary: input_dev was populated by a host→device copy on
    // the default stream (cu_memcpy_htod). Synchronize compute_stream so kernels
    // on the named stream see the uploaded data before executing.
    if (use_preloaded_input) {
        if (self.compute_stream) |stream| {
            try self.device.synchronizeStream(stream);
        }
    }

    // Build BatchDecodeInfo for the single-token decode step. This routes
    // attention through the proven batched decode path (batched separate or
    // flash decode) instead of the legacy runAttentionMixerStep path which
    // has double-RoPE and dead-code bugs.
    const slot_indices_single = [_]usize{slot_index};
    const positions_single = [_]usize{position};
    self.runtime_buffers.decode_seq_lens_host[0] = seq_len_u32;
    self.runtime_buffers.decode_positions_host[0] = position_u32;
    var proto_seq_lens_dev = try bufferSlice(&self.runtime_buffers.decode_seq_lens_dev, 0, @sizeOf(u32));
    var proto_positions_dev = try bufferSlice(&self.runtime_buffers.decode_positions_dev, 0, @sizeOf(u32));
    try proto_seq_lens_dev.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.decode_seq_lens_host[0..1]));
    try proto_positions_dev.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.decode_positions_host[0..1]));

    // Set batched attention max seq_len tier (power-of-2 ceiling).
    const model_max_u32: u32 = @intCast(self.max_seq_len);
    self.runtime_buffers.batched_attn_max_seq_len = @min(
        std.math.ceilPowerOfTwo(u32, seq_len_u32) catch model_max_u32,
        model_max_u32,
    );

    // Populate attention pointer tables from block_runtime (authoritative
    // after ensureKvCapacity which may have grown KV buffers).
    const attn_layers = self.block_runtime.attention_block_count;
    if (attn_layers > 0) {
        const attn_key_ptrs_host = self.runtime_buffers.decode_attn_key_cache_ptrs_table_host[0..attn_layers];
        const attn_value_ptrs_host = self.runtime_buffers.decode_attn_value_cache_ptrs_table_host[0..attn_layers];
        const attn_k_scale_ptrs_host = self.runtime_buffers.decode_attn_k_scale_ptrs_table_host[0..attn_layers];
        const attn_v_scale_ptrs_host = self.runtime_buffers.decode_attn_v_scale_ptrs_table_host[0..attn_layers];
        var attn_idx: usize = 0;
        for (self.block_runtime.blocks) |blk| {
            const binding = blk.attention_binding orelse continue;
            attn_key_ptrs_host[attn_idx] = binding.k_cache.pointer;
            attn_value_ptrs_host[attn_idx] = binding.v_cache.pointer;
            attn_k_scale_ptrs_host[attn_idx] = binding.k_scale.pointer;
            attn_v_scale_ptrs_host[attn_idx] = binding.v_scale.pointer;
            attn_idx += 1;
        }
        const attn_ptr_bytes = std.math.mul(usize, attn_layers, @sizeOf(u64)) catch return error.InvalidArgument;
        var attn_key_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_attn_key_cache_ptrs_table_dev, 0, attn_ptr_bytes);
        var attn_value_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_attn_value_cache_ptrs_table_dev, 0, attn_ptr_bytes);
        var attn_k_scale_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_attn_k_scale_ptrs_table_dev, 0, attn_ptr_bytes);
        var attn_v_scale_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_attn_v_scale_ptrs_table_dev, 0, attn_ptr_bytes);
        try attn_key_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(attn_key_ptrs_host));
        try attn_value_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(attn_value_ptrs_host));
        try attn_k_scale_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(attn_k_scale_ptrs_host));
        try attn_v_scale_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(attn_v_scale_ptrs_host));
    }

    // Populate gated delta pointer tables from block_runtime.
    const gd_layers = self.block_runtime.gated_delta_block_count;
    if (gd_layers > 0) {
        const gd_conv_ptrs_host = self.runtime_buffers.decode_gd_conv_state_ptrs_table_host[0..gd_layers];
        const gd_ssm_ptrs_host = self.runtime_buffers.decode_gd_ssm_state_ptrs_table_host[0..gd_layers];
        const gd_ring_heads_host = self.runtime_buffers.decode_gd_conv_ring_heads_table_host[0..gd_layers];
        var gd_idx: usize = 0;
        for (self.block_runtime.blocks) |blk| {
            const binding = blk.gated_delta_binding orelse continue;
            gd_conv_ptrs_host[gd_idx] = binding.conv_state_dev.pointer;
            gd_ssm_ptrs_host[gd_idx] = binding.ssm_state_dev.pointer;
            gd_ring_heads_host[gd_idx] = binding.conv_ring_head;
            gd_idx += 1;
        }
        const gd_ptr_bytes = std.math.mul(usize, gd_layers, @sizeOf(u64)) catch return error.InvalidArgument;
        const gd_idx_bytes = std.math.mul(usize, gd_layers, @sizeOf(u32)) catch return error.InvalidArgument;
        var gd_conv_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_gd_conv_state_ptrs_table_dev, 0, gd_ptr_bytes);
        var gd_ssm_ptrs_dev = try bufferSlice(&self.runtime_buffers.decode_gd_ssm_state_ptrs_table_dev, 0, gd_ptr_bytes);
        var gd_ring_heads_dev = try bufferSlice(&self.runtime_buffers.decode_gd_conv_ring_heads_table_dev, 0, gd_idx_bytes);
        try gd_conv_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(gd_conv_ptrs_host));
        try gd_ssm_ptrs_dev.upload(&self.device, std.mem.sliceAsBytes(gd_ssm_ptrs_host));
        try gd_ring_heads_dev.upload(&self.device, std.mem.sliceAsBytes(gd_ring_heads_host));
    }

    var batch_info = BatchDecodeInfo{
        .slot_indices = &slot_indices_single,
        .positions = &positions_single,
        .seq_lens = self.runtime_buffers.decode_seq_lens_host[0..1],
        .attn_ptrs_row_stride = 1,
        .attn_key_cache_ptrs_table_dev = &self.runtime_buffers.decode_attn_key_cache_ptrs_table_dev,
        .attn_value_cache_ptrs_table_dev = &self.runtime_buffers.decode_attn_value_cache_ptrs_table_dev,
        .gd_ptrs_row_stride = 1,
        .gd_conv_state_ptrs_table_dev = &self.runtime_buffers.decode_gd_conv_state_ptrs_table_dev,
        .gd_ssm_state_ptrs_table_dev = &self.runtime_buffers.decode_gd_ssm_state_ptrs_table_dev,
        .gd_conv_ring_heads_table_dev = &self.runtime_buffers.decode_gd_conv_ring_heads_table_dev,
        .attn_k_scale_ptrs_table_dev = &self.runtime_buffers.decode_attn_k_scale_ptrs_table_dev,
        .attn_v_scale_ptrs_table_dev = &self.runtime_buffers.decode_attn_v_scale_ptrs_table_dev,
        .attn_layer_index = 0,
        .gd_layer_index = 0,
        .sc_layer_index = 0,
    };

    // Invalidate batched decode pointer table cache (prototype path
    // populates tables for a single row; batched path must re-upload).
    if (comptime @hasField(SelfType, "decode_ptr_tables_dirty")) {
        self.decode_ptr_tables_dirty = true;
    }

    // CUDA graph capture: record all GPU kernels (layer loop + final norm + lm_head)
    // into a graph, then replay with near-zero scheduling gaps between kernels.
    // Graph capture eliminates per-kernel launch overhead by recording the layer
    // sequence and replaying it as a single launch.  Re-captures every decode step
    // to pick up changed arguments (position, seq_len); graphExecUpdate patches
    // the existing exec in-place when topology matches (very fast).
    // Restricted to single-row decode: prefill has synchronous KV operations and
    // varying row counts that break graph topology stability.
    // Pipeline2 stage1 excluded: receives activation via external transfer.
    // Persistent graph replay: on the first decode step, capture the layer
    // sequence into a CUDA graph and instantiate it. On subsequent steps,
    // replay the existing graph without re-capture — the position/seq_len
    // data lives in device buffers updated before this function, so the graph
    // topology is identical every token. All setup (embedding lookup, pointer
    // table uploads) runs before the graph region and before persistent replay.
    const event_timing_enabled = if (comptime @hasField(@TypeOf(self.*), "phase_event_timing_enabled"))
        self.phase_event_timing_enabled
    else
        false;
    const no_graph = std.posix.getenv("TALU_NO_GRAPH") != null;
    const graph_eligible = self.compute_stream != null and
        !trace.isEnabled() and deepstack_layer_features_opt == null and
        !event_timing_enabled and
        !no_graph;

    var graph_capture_active = false;
    if (graph_eligible) {
        if (self.device.streamBeginCapture(self.compute_stream.?)) {
            graph_capture_active = true;
        } else |_| {}
    }
    errdefer if (graph_capture_active) {
        _ = self.device.streamEndCapture(self.compute_stream.?) catch {};
    };

    var final_hidden = input_row;

    if (std.posix.getenv("TALU_DUMP_HIDDEN") != null) {
        log.warn("inference", "GPU_LOOP_ENTRY", .{
            .layer_limit = layer_limit,
            .split_layer = self.split_layer,
            .blocks_len = self.block_runtime.blocks.len,
            .use_preloaded = use_preloaded_input,
            .d_model = self.d_model,
        });
    }

    var layer_idx: usize = 0;
    while (layer_idx < layer_limit) : (layer_idx += 1) {
        const layer = &self.block_runtime.blocks[layer_idx];
        const attention_kernels: AttentionKernelSet = switch (self.kv_cache_dtype) {
            .f16 => .{
                .attn_scores_heads_f16_kv_function = self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                .attn_weighted_sum_heads_f16_kv_function = self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
                .attn_fused_heads_f16_kv_function = self.attn_fused_heads_f16_kv_function,
                .attn_fused_prefill_heads_f16_kv_function = self.attn_fused_prefill_heads_f16_kv_function,
                .attn_fused_prefill_heads_f16_kv_gqa_function = self.attn_fused_prefill_heads_f16_kv_gqa_function,
                .softmax_rows_function = softmax_rows_function,
                .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
            },
            .i8 => .{
                .attn_scores_heads_i8_kv_function = self.attn_scores_heads_i8_kv_function,
                .attn_weighted_sum_heads_i8_kv_function = self.attn_weighted_sum_heads_i8_kv_function,
                .attn_fused_heads_i8_kv_function = self.attn_fused_heads_i8_kv_function,
                .attn_fused_prefill_heads_i8_kv_function = self.attn_fused_prefill_heads_i8_kv_function,
                .attn_fused_prefill_heads_i8_kv_gqa_function = self.attn_fused_prefill_heads_i8_kv_gqa_function,
                .softmax_rows_function = softmax_rows_function,
                .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
            },
            .fp8 => .{
                .attn_scores_heads_fp8_kv_function = self.attn_scores_heads_fp8_kv_function,
                .attn_weighted_sum_heads_fp8_kv_function = self.attn_weighted_sum_heads_fp8_kv_function,
                .attn_fused_heads_fp8_kv_function = self.attn_fused_heads_fp8_kv_function,
                .attn_fused_prefill_heads_fp8_kv_function = self.attn_fused_prefill_heads_fp8_kv_function,
                .attn_fused_prefill_heads_fp8_kv_gqa_function = self.attn_fused_prefill_heads_fp8_kv_gqa_function,
                .softmax_rows_function = softmax_rows_function,
                .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
            },
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
            &batch_info,
        );
        if (layer.attention_binding != null) batch_info.attn_layer_index += 1;
        if (layer.gated_delta_binding != null) batch_info.gd_layer_index += 1;
        if (layer.shortconv_binding != null) batch_info.sc_layer_index += 1;
        dumpHiddenState(self, &self.runtime_buffers.input_dev, self.split_layer + layer_idx, "post_layer", self.d_model, 1);
        if (comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch")) {
            if (gemma_source_embeddings_opt) |*gemma_source_embeddings| {
                try self.applyGemma4PerLayerBranch(
                    layer_idx,
                    gemma_token_id_single[0..],
                    gemma_source_embeddings,
                    &self.runtime_buffers.input_dev,
                );
                final_hidden = self.runtime_buffers.input_dev;
                dumpHiddenState(self, &self.runtime_buffers.input_dev, self.split_layer + layer_idx, "post_ple", self.d_model, 1);
            } else if (comptime @hasDecl(SelfType, "applyStandaloneLayerScalar")) {
                try self.applyStandaloneLayerScalar(layer_idx, &self.runtime_buffers.input_dev, 1);
            }
        }
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

    // Sync block_runtime with slot_kv_states after the batched mixer may
    // have updated GD state (conv_ring_head) directly in slot_kv_states,
    // bypassing block_runtime. Same pattern as computeBatchedDecodeLogits.
    self.loadKvSlot(slot_index);

    if (!compute_logits) {
        // Stage handoff invariant: when skipping logits, publish the final hidden
        // row to input_dev so the next stage can consume a deterministic buffer.
        if (final_hidden.pointer != self.runtime_buffers.input_dev.pointer) {
            try self.runtime_buffers.input_dev.copyFrom(&self.device, &final_hidden, row_bytes);
        }
        // Finalize graph capture for non-logit stage (Pipeline2 stage0).
        // The copyFrom above uses async DtoD when a stream is set, so it is
        // captured into the graph alongside the layer kernels.
        if (graph_capture_active) {
            const new_graph = self.device.streamEndCapture(self.compute_stream.?) catch
                return error.CudaGraphCaptureFailed;
            defer self.device.graphDestroy(new_graph);
            if (self.decode_graph_exec) |exec| {
                self.device.graphExecUpdate(exec, new_graph) catch {
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
            .fp8 => "matmul_lm_head_fp8_host",
            .mxfp8 => "matmul_lm_head_mxfp8_host",
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
    applyHostLogitsPostProcess(
        logits_out,
        self.loaded.config.logits_scaling,
        self.loaded.config.final_logit_softcapping,
    );
    if (self.loaded.config.logits_scaling != 1.0 and trace.isEnabled()) {
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

/// Batched decode: process N tokens at different positions/slots together.
/// Uses GEMM (not GEMV) for layer projections, sharing weight reads
/// across sequences for ~Nx throughput.
const BatchedDecodeOutputMode = enum {
    host_logits,
    device_only,
};

fn computeBatchedDecodeLogitsWithMode(
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    output_mode: BatchedDecodeOutputMode,
) !void {
    const SelfType = @TypeOf(self.*);
    const gemma4_branch_active = if (comptime @hasField(SelfType, "gemma4_per_layer"))
        self.gemma4_per_layer != null
    else
        false;
    const n_usize = tokens.len;
    if (n_usize == 0) return;
    if (n_usize > self.max_batch_size) return error.InvalidArgument;
    const n: u32 = @intCast(n_usize);
    const force_prototype = std.posix.getenv("TALU_FORCE_PROTOTYPE") != null;
    if (force_prototype or topologyModeIs(self, "pipeline2") or topologyModeIs(self, "cpu_gpu") or topologyModeIs(self, "cpu_gpu_gpu")) {
        const emit_host_logits = output_mode == .host_logits;
        for (0..n_usize) |i| {
            self.activateKvSlot(slot_indices[i]);
            const logits_target: ?[]f32 = if (emit_host_logits) blk: {
                if (comptime !@hasDecl(SelfType, "slotLogits")) return error.InvalidTopologyConfig;
                break :blk self.slotLogits(slot_indices[i]);
            } else null;
            try computeGpuPrototypeLogitsWithLayerLimit(
                self,
                tokens[i],
                positions[i],
                slot_indices[i],
                logits_target,
                self.block_runtime.blocks.len,
                true,
                emit_host_logits,
                true,
                1,
                positions[i],
                null,
                null,
                null,
                false,
            );
            if (!emit_host_logits) continue;
            // Sync batch host buffer: the decode loop reads logits via
            // batchedHostLogitsRow() which returns projected_logits_batch_host.
            // Copy from slotLogits (the authoritative source after the prototype
            // call). For pipeline2, self.runtime_buffers.projected_logits_host is
            // stale because the logit projection runs on pipeline_backend1, not self.
            // slotLogits already has post-processing applied (line ~2599), so we
            // must NOT re-apply applyHostLogitsPostProcess here.
            if (comptime @hasField(SelfType, "runtime_buffers") and @hasDecl(SelfType, "slotLogits")) {
                const vocab = self.runtime_buffers.projected_vocab;
                const row_start = std.math.mul(usize, i, vocab) catch continue;
                const row_end = std.math.add(usize, row_start, vocab) catch continue;
                if (row_end <= self.runtime_buffers.projected_logits_batch_host.len) {
                    const slot_logits = self.slotLogits(slot_indices[i]);
                    @memcpy(
                        self.runtime_buffers.projected_logits_batch_host[row_start..row_end],
                        slot_logits[0..vocab],
                    );
                }
            }
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

    // Ensure KV capacity for each slot only when growth is required.
    var require_kv_growth = false;
    for (0..n_usize) |i| {
        const required_tokens = positions[i] + 1;
        const slot_state = &self.slot_kv_states[slot_indices[i]];
        var slot_min_capacity = self.max_seq_len;
        if (slot_state.kv.len > 0) {
            slot_min_capacity = slot_state.kv[0].capacity;
            for (slot_state.kv[1..]) |kv_entry| {
                if (kv_entry.capacity < slot_min_capacity) slot_min_capacity = kv_entry.capacity;
            }
        }
        if (required_tokens > slot_min_capacity) {
            require_kv_growth = true;
            break;
        }
    }
    if (require_kv_growth) {
        for (0..n_usize) |i| {
            self.activateKvSlot(slot_indices[i]);
            try ensureKvCapacity(self, positions[i] + 1);
        }
        self.saveActiveKvSlot();
    }

    // Extract kernel functions (same set as single-token path).
    const shortconv_step_function = self.shortconv_step_function orelse return error.CudaKernelUnavailable;
    const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
    const is_f16_kv_bd = self.kv_cache_dtype == .f16;
    const cast_f32_to_f16_function: ?compute.cuda.Function = if (is_f16_kv_bd)
        (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable)
    else
        null;
    const kv_write_f16_function: ?compute.cuda.Function = if (is_f16_kv_bd) self.kv_write_f16_function else null;
    const rope_store_f16_function: ?compute.cuda.Function = if (is_f16_kv_bd) self.rope_store_f16_function else null;
    const rope_function = self.rope_function orelse return error.CudaKernelUnavailable;
    const softmax_rows_function = self.softmax_rows_function;
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
    const attention_kernels: AttentionKernelSet = switch (self.kv_cache_dtype) {
        .f16 => .{
            .attn_scores_heads_f16_kv_function = self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = self.attn_fused_heads_f16_kv_function,
            .softmax_rows_function = softmax_rows_function,
        },
        .i8 => .{
            .attn_scores_heads_i8_kv_function = self.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = self.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = self.attn_fused_heads_i8_kv_function,
            .softmax_rows_function = softmax_rows_function,
        },
        .fp8 => .{
            .attn_scores_heads_fp8_kv_function = self.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = self.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = self.attn_fused_heads_fp8_kv_function,
            .softmax_rows_function = softmax_rows_function,
        },
    };

    // Embedding lookup: N tokens → N rows of input_dev.
    const embedding_lookup_f32_function = self.embedding_lookup_f32_function;
    const embedding_lookup_u16_function = self.embedding_lookup_u16_function;
    const embedding_lookup_gaffine_u4_function = self.embedding_lookup_gaffine_u4_function;
    var used_rows_device_lookup = false;
    if (enable_device_embedding_lookup and self.runtime_buffers.embedding_lookup != null) {
        const lookup = &self.runtime_buffers.embedding_lookup.?;
        switch (lookup.kind) {
            .f16, .bf16 => {
                if (self.embedding_lookup_u16_rows_function) |kernel| {
                    const token_bytes = std.math.mul(usize, n_usize, @sizeOf(u32)) catch return error.InvalidArgument;
                    var token_ids_dev = try bufferSlice(&self.runtime_buffers.prefill_tokens_dev, 0, token_bytes);
                    self.prefill_rope_positions_cached_dirty = true;
                    try token_ids_dev.upload(&self.device, std.mem.sliceAsBytes(tokens));
                    var input_rows = try bufferSlice(&self.runtime_buffers.input_dev, 0, n_usize * row_bytes);
                    try compute.cuda.embedding_lookup_u16_rows.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        kernel,
                        &input_rows,
                        &lookup.buffer,
                        &token_ids_dev,
                        @intCast(n_usize),
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
                    used_rows_device_lookup = true;
                }
            },
            else => {},
        }
    }

    if (!used_rows_device_lookup) {
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
    }

    var slot_set_matches_cache = false;
    if (comptime @hasField(SelfType, "decode_ptr_tables_cached_rows") and @hasField(SelfType, "decode_ptr_tables_cached_slots")) {
        if (self.decode_ptr_tables_cached_rows == n_usize and self.decode_ptr_tables_cached_slots.len >= n_usize) {
            slot_set_matches_cache = true;
            for (slot_indices, 0..) |slot_idx, i| {
                if (self.decode_ptr_tables_cached_slots[i] != slot_idx) {
                    slot_set_matches_cache = false;
                    break;
                }
            }
        }
    }
    const pointer_tables_dirty = if (comptime @hasField(SelfType, "decode_ptr_tables_dirty"))
        self.decode_ptr_tables_dirty
    else
        true;
    const refresh_pointer_tables = pointer_tables_dirty or !slot_set_matches_cache;

    // Build per-row decode metadata once for this decode step.
    const seq_lens = self.runtime_buffers.decode_seq_lens_host[0..n_usize];
    const positions_host = self.runtime_buffers.decode_positions_host[0..n_usize];
    for (0..n_usize) |i| {
        seq_lens[i] = @intCast(positions[i] + 1);
        positions_host[i] = @intCast(positions[i]);
    }
    const decode_idx_bytes = std.math.mul(usize, n_usize, @sizeOf(u32)) catch return error.InvalidArgument;
    var decode_seq_lens_dev = try bufferSlice(&self.runtime_buffers.decode_seq_lens_dev, 0, decode_idx_bytes);
    var decode_positions_dev = try bufferSlice(&self.runtime_buffers.decode_positions_dev, 0, decode_idx_bytes);
    if (refresh_pointer_tables) {
        try decode_seq_lens_dev.upload(&self.device, std.mem.sliceAsBytes(seq_lens));
        try decode_positions_dev.upload(&self.device, std.mem.sliceAsBytes(positions_host));
    }

    const attn_layers = self.block_runtime.attention_block_count;
    const attn_table_entries = std.math.mul(usize, n_usize, attn_layers) catch return error.InvalidArgument;
    if (attn_table_entries > 0 and refresh_pointer_tables) {
        var attn_key_ptrs_table_host = self.runtime_buffers.decode_attn_key_cache_ptrs_table_host[0..attn_table_entries];
        var attn_value_ptrs_table_host = self.runtime_buffers.decode_attn_value_cache_ptrs_table_host[0..attn_table_entries];
        var attn_k_scale_ptrs_table_host = self.runtime_buffers.decode_attn_k_scale_ptrs_table_host[0..attn_table_entries];
        var attn_v_scale_ptrs_table_host = self.runtime_buffers.decode_attn_v_scale_ptrs_table_host[0..attn_table_entries];
        var attn_layer_idx: usize = 0;
        for (self.block_runtime.blocks) |layer| {
            if (layer.attention_binding == null) continue;
            const table_row_offset = std.math.mul(usize, attn_layer_idx, n_usize) catch return error.InvalidArgument;
            for (0..n_usize) |row_i| {
                const slot_idx = slot_indices[row_i];
                const kv_entry = self.slot_kv_states[slot_idx].kv[attn_layer_idx];
                const dst_idx = table_row_offset + row_i;
                attn_key_ptrs_table_host[dst_idx] = kv_entry.k.pointer;
                attn_value_ptrs_table_host[dst_idx] = kv_entry.v.pointer;
                attn_k_scale_ptrs_table_host[dst_idx] = kv_entry.k_scale.pointer;
                attn_v_scale_ptrs_table_host[dst_idx] = kv_entry.v_scale.pointer;
            }
            attn_layer_idx += 1;
        }
        const attn_table_ptr_bytes = std.math.mul(usize, attn_table_entries, @sizeOf(u64)) catch return error.InvalidArgument;
        var attn_key_ptrs_table_dev = try bufferSlice(&self.runtime_buffers.decode_attn_key_cache_ptrs_table_dev, 0, attn_table_ptr_bytes);
        var attn_value_ptrs_table_dev = try bufferSlice(&self.runtime_buffers.decode_attn_value_cache_ptrs_table_dev, 0, attn_table_ptr_bytes);
        var attn_k_scale_ptrs_table_dev = try bufferSlice(&self.runtime_buffers.decode_attn_k_scale_ptrs_table_dev, 0, attn_table_ptr_bytes);
        var attn_v_scale_ptrs_table_dev = try bufferSlice(&self.runtime_buffers.decode_attn_v_scale_ptrs_table_dev, 0, attn_table_ptr_bytes);
        try attn_key_ptrs_table_dev.upload(&self.device, std.mem.sliceAsBytes(attn_key_ptrs_table_host));
        try attn_value_ptrs_table_dev.upload(&self.device, std.mem.sliceAsBytes(attn_value_ptrs_table_host));
        try attn_k_scale_ptrs_table_dev.upload(&self.device, std.mem.sliceAsBytes(attn_k_scale_ptrs_table_host));
        try attn_v_scale_ptrs_table_dev.upload(&self.device, std.mem.sliceAsBytes(attn_v_scale_ptrs_table_host));
    }

    const gd_layers = self.block_runtime.gated_delta_block_count;
    const gd_table_entries = std.math.mul(usize, n_usize, gd_layers) catch return error.InvalidArgument;
    if (gd_table_entries > 0) {
        var gd_ring_heads_table_host = self.runtime_buffers.decode_gd_conv_ring_heads_table_host[0..gd_table_entries];
        // Ring heads table: always populate host-side (needed by engine_mixers
        // shadow increment), but only upload to device on batch change.  The
        // conv kernel increments ring heads in-place on device, so successive
        // steps with the same batch require no transfer.
        var gd_layer_idx: usize = 0;
        for (self.block_runtime.blocks) |layer| {
            if (layer.gated_delta_binding == null) continue;
            const table_row_offset = std.math.mul(usize, gd_layer_idx, n_usize) catch return error.InvalidArgument;
            for (0..n_usize) |row_i| {
                const slot_idx = slot_indices[row_i];
                const gd_entry = self.slot_kv_states[slot_idx].gd[gd_layer_idx];
                const dst_idx = table_row_offset + row_i;
                gd_ring_heads_table_host[dst_idx] = gd_entry.conv_ring_head;
            }
            gd_layer_idx += 1;
        }
        if (refresh_pointer_tables) {
            var gd_conv_ptrs_table_host = self.runtime_buffers.decode_gd_conv_state_ptrs_table_host[0..gd_table_entries];
            var gd_ssm_ptrs_table_host = self.runtime_buffers.decode_gd_ssm_state_ptrs_table_host[0..gd_table_entries];
            gd_layer_idx = 0;
            for (self.block_runtime.blocks) |layer_| {
                if (layer_.gated_delta_binding == null) continue;
                const table_row_offset = std.math.mul(usize, gd_layer_idx, n_usize) catch return error.InvalidArgument;
                for (0..n_usize) |row_i| {
                    const slot_idx = slot_indices[row_i];
                    const gd_entry = self.slot_kv_states[slot_idx].gd[gd_layer_idx];
                    const dst_idx = table_row_offset + row_i;
                    gd_conv_ptrs_table_host[dst_idx] = gd_entry.conv.pointer;
                    gd_ssm_ptrs_table_host[dst_idx] = gd_entry.ssm.pointer;
                }
                gd_layer_idx += 1;
            }
            const gd_table_ptr_bytes = std.math.mul(usize, gd_table_entries, @sizeOf(u64)) catch return error.InvalidArgument;
            const gd_table_idx_bytes = std.math.mul(usize, gd_table_entries, @sizeOf(u32)) catch return error.InvalidArgument;
            var gd_conv_ptrs_table_dev = try bufferSlice(&self.runtime_buffers.decode_gd_conv_state_ptrs_table_dev, 0, gd_table_ptr_bytes);
            var gd_ssm_ptrs_table_dev = try bufferSlice(&self.runtime_buffers.decode_gd_ssm_state_ptrs_table_dev, 0, gd_table_ptr_bytes);
            var gd_ring_heads_table_dev = try bufferSlice(&self.runtime_buffers.decode_gd_conv_ring_heads_table_dev, 0, gd_table_idx_bytes);
            try gd_conv_ptrs_table_dev.upload(&self.device, std.mem.sliceAsBytes(gd_conv_ptrs_table_host));
            try gd_ssm_ptrs_table_dev.upload(&self.device, std.mem.sliceAsBytes(gd_ssm_ptrs_table_host));
            try gd_ring_heads_table_dev.upload(&self.device, std.mem.sliceAsBytes(gd_ring_heads_table_host));
        }
    }

    if (refresh_pointer_tables and comptime @hasField(SelfType, "decode_ptr_tables_cached_rows") and @hasField(SelfType, "decode_ptr_tables_cached_slots")) {
        if (self.decode_ptr_tables_cached_slots.len >= n_usize) {
            @memcpy(self.decode_ptr_tables_cached_slots[0..n_usize], slot_indices);
            self.decode_ptr_tables_cached_rows = n_usize;
        } else {
            self.decode_ptr_tables_cached_rows = 0;
        }
    }
    if (refresh_pointer_tables and comptime @hasField(SelfType, "decode_ptr_tables_dirty")) {
        self.decode_ptr_tables_dirty = false;
    }

    // Logits output setup (common to replay and normal paths).
    const vocab = self.runtime_buffers.projected_vocab;
    const logits_elems = std.math.mul(usize, n_usize, vocab) catch return error.InvalidArgument;
    const logits_bytes = std.math.mul(usize, logits_elems, @sizeOf(f32)) catch return error.InvalidArgument;
    var logits_batch = try bufferSlice(&self.runtime_buffers.logits_dev, 0, logits_bytes);

    // Compute seq_len tier for batched attention grid dimensions.
    // Using the model's full max_seq_len would launch millions of empty blocks
    // and cause poor cache utilization from large output strides. Instead, use
    // a power-of-2 tier based on the current max seq_len across batch rows.
    if (comptime @hasField(SelfType, "batched_decode_graph_seq_tier")) {
        var current_max_seq: u32 = 0;
        for (0..n_usize) |i| current_max_seq = @max(current_max_seq, seq_lens[i]);
        if (current_max_seq > self.batched_decode_graph_seq_tier) {
            // Seq_len exceeded the tier used at graph capture — invalidate.
            if (comptime @hasField(SelfType, "batched_decode_graph_exec")) {
                if (self.batched_decode_graph_exec) |exec| {
                    self.device.graphExecDestroy(exec);
                    self.batched_decode_graph_exec = null;
                }
            }
            // New tier: next power of 2, capped at model max.
            // Use self.max_seq_len (immutable) — not batched_attn_max_seq_len
            // which gets overwritten with the tier value each step.
            const model_max: u32 = @intCast(self.max_seq_len);
            self.batched_decode_graph_seq_tier = @min(
                std.math.ceilPowerOfTwo(u32, current_max_seq) catch model_max,
                model_max,
            );
        }
        // Set the tier for mixer grid dimensions (buffer is allocated for model max).
        self.runtime_buffers.batched_attn_max_seq_len = self.batched_decode_graph_seq_tier;
    }

    // CUDA graph: capture-once-replay-many. Three states:
    // 1. refresh_pointer_tables=true → run normally, no capture (batch changed).
    // 2. First steady-state step (no graph exec) → capture + instantiate + launch.
    // 3. Subsequent steady-state steps → graphLaunch only + host shadow bookkeeping.
    compute: {
        // State 3: steady-state replay — cached graph exec + stable batch.
        if (comptime @hasField(SelfType, "batched_decode_graph_exec")) {
            if (self.compute_stream != null and !trace.isEnabled() and !refresh_pointer_tables and !gemma4_branch_active) {
                if (self.batched_decode_graph_exec) |exec| {
                    try self.device.graphLaunch(exec, self.compute_stream.?);

                    // Host shadow: advance GDN ring heads to match device-side advance.
                    var gd_layer_idx: usize = 0;
                    for (self.block_runtime.blocks) |blk| {
                        if (blk.gated_delta_binding == null) continue;
                        const d_conv: u32 = @intCast(blk.gated_delta_binding.?.kernel.config.d_conv);
                        for (0..n_usize) |row_i| {
                            const gd_state = &self.slot_kv_states[slot_indices[row_i]].gd[gd_layer_idx];
                            gd_state.conv_ring_head = if (gd_state.conv_ring_head + 1 >= d_conv) 0 else gd_state.conv_ring_head + 1;
                        }
                        gd_layer_idx += 1;
                    }
                    self.loadKvSlot(self.active_kv_slot);
                    break :compute;
                }
            }
        }

        // States 1 and 2: normal execution with optional graph capture.
        // Capture only on first steady-state step (state 2).
        var graph_capture_active = false;
        if (comptime @hasField(SelfType, "batched_decode_graph_exec")) {
            const event_timing_enabled = if (comptime @hasField(SelfType, "phase_event_timing_enabled"))
                self.phase_event_timing_enabled
            else
                false;
            if (self.compute_stream != null and !trace.isEnabled() and !refresh_pointer_tables and !gemma4_branch_active and !event_timing_enabled) {
                try self.device.streamBeginCapture(self.compute_stream.?);
                graph_capture_active = true;
            }
        }
        errdefer if (graph_capture_active) {
            _ = self.device.streamEndCapture(self.compute_stream.?) catch {};
        };

        var batch_info = BatchDecodeInfo{
            .slot_indices = slot_indices,
            .positions = positions,
            .seq_lens = seq_lens,
            .attn_ptrs_row_stride = n_usize,
            .attn_key_cache_ptrs_table_dev = &self.runtime_buffers.decode_attn_key_cache_ptrs_table_dev,
            .attn_value_cache_ptrs_table_dev = &self.runtime_buffers.decode_attn_value_cache_ptrs_table_dev,
            .gd_ptrs_row_stride = n_usize,
            .gd_conv_state_ptrs_table_dev = &self.runtime_buffers.decode_gd_conv_state_ptrs_table_dev,
            .gd_ssm_state_ptrs_table_dev = &self.runtime_buffers.decode_gd_ssm_state_ptrs_table_dev,
            .gd_conv_ring_heads_table_dev = &self.runtime_buffers.decode_gd_conv_ring_heads_table_dev,
            .attn_k_scale_ptrs_table_dev = &self.runtime_buffers.decode_attn_k_scale_ptrs_table_dev,
            .attn_v_scale_ptrs_table_dev = &self.runtime_buffers.decode_attn_v_scale_ptrs_table_dev,
            .attn_layer_index = 0,
            .gd_layer_index = 0,
            .sc_layer_index = 0,
        };

        const layer_seq_len_u32: u32 = seq_lens[0];
        const layer_position: usize = positions[0];
        const layer_position_u32: u32 = @intCast(layer_position);

        // Multi-row layer loop: all N tokens processed together through each layer.
        var final_hidden = try bufferSlice(&self.runtime_buffers.input_dev, 0, n_usize * row_bytes);
        var gemma_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (comptime @hasDecl(SelfType, "maybeCaptureGemma4SourceEmbeddings")) {
            gemma_source_embeddings_opt = try self.maybeCaptureGemma4SourceEmbeddings(n_usize);
        }
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
                n,
                layer_seq_len_u32,
                layer_seq_len_u32,
                0,
                layer_position,
                layer_position_u32,
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
            if (comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch")) {
                if (gemma_source_embeddings_opt) |*gemma_source_embeddings| {
                    try self.applyGemma4PerLayerBranch(
                        layer_idx,
                        tokens,
                        gemma_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden = self.runtime_buffers.input_dev;
                } else if (comptime @hasDecl(SelfType, "applyStandaloneLayerScalar")) {
                    try self.applyStandaloneLayerScalar(layer_idx, &self.runtime_buffers.input_dev, n_usize);
                }
            }
            if (layer.attention_binding != null) batch_info.attn_layer_index += 1;
            if (layer.gated_delta_binding != null) batch_info.gd_layer_index += 1;
            if (layer.shortconv_binding != null) batch_info.sc_layer_index += 1;
        }

        // Final norm for all N rows.
        final_hidden = try bufferSlice(&self.runtime_buffers.input_dev, 0, n_usize * row_bytes);
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

        // LM head for all rows.
        try engine_ops.linearForwardRows(self, &norm_out, n_usize, &self.runtime_buffers.projection_weight, &logits_batch);

        // Keep decode metadata resident on device across token steps.
        if (comptime @hasDecl(SelfType, "incrementDecodeMetadataInPlace")) {
            try self.incrementDecodeMetadataInPlace(&decode_seq_lens_dev, &decode_positions_dev, n_usize);
        }

        // End graph capture: instantiate exec, then launch the captured graph.
        if (graph_capture_active) {
            const new_graph = self.device.streamEndCapture(self.compute_stream.?) catch
                return error.CudaGraphCaptureFailed;
            graph_capture_active = false;
            defer self.device.graphDestroy(new_graph);

            if (comptime @hasField(SelfType, "batched_decode_graph_exec")) {
                self.batched_decode_graph_exec = self.device.graphInstantiate(new_graph) catch
                    return error.CudaGraphInstantiateFailed;
                try self.device.graphLaunch(self.batched_decode_graph_exec.?, self.compute_stream.?);
            }
        }

        // Sync block_runtime with the active slot's updated state (host operation).
        self.loadKvSlot(self.active_kv_slot);
    }

    if (output_mode == .host_logits) {
        const logits_batch_host = self.runtime_buffers.projected_logits_batch_host[0..logits_elems];
        try logits_batch.download(&self.device, std.mem.sliceAsBytes(logits_batch_host));

        applyHostLogitsPostProcess(
            logits_batch_host,
            self.loaded.config.logits_scaling,
            self.loaded.config.final_logit_softcapping,
        );
    }
}

pub fn computeBatchedDecodeLogits(
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
) !void {
    return computeBatchedDecodeLogitsWithMode(self, tokens, slot_indices, positions, .host_logits);
}

pub fn computeBatchedDecodeLogitsDeviceOnly(
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
) !void {
    return computeBatchedDecodeLogitsWithMode(self, tokens, slot_indices, positions, .device_only);
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
    if (topologyModeIs(self, "pipeline2") and layer_limit == self.block_runtime.blocks.len)
    {
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

    // cpu_gpu / cpu_gpu_gpu: batched CPU stage0 → chunked GPU stage(s).
    if ((topologyModeIs(self, "cpu_gpu") or topologyModeIs(self, "cpu_gpu_gpu")) and
        layer_limit == self.block_runtime.blocks.len)
    {
        if (tokens.len == 0) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;

        if (comptime @hasDecl(SelfType, "pipelineCpuStage0")) {
            const cpu_stage0 = self.pipelineCpuStage0() orelse return error.InvalidTopologyConfig;
            self.activateKvSlot(slot_index);

            // cpu_gpu_gpu: CPU → GPU0 → GPU1 (3-stage).
            if (topologyModeIs(self, "cpu_gpu_gpu")) {
                if (comptime @hasDecl(SelfType, "pipelineStage1")) {
                    var gpu_stage1 = self.pipelineStage1() orelse return error.InvalidTopologyConfig;
                    if (gpu_stage1.state_descriptor_count > 0) try gpu_stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                    gpu_stage1.activateKvSlot(slot_index);
                    return computeBatchedPrefillCpuGpuGpu(self, cpu_stage0, gpu_stage1, tokens, slot_index, logits_out);
                }
            }

            // cpu_gpu: CPU → GPU (2-stage).
            return computeBatchedPrefillCpuGpu(self, cpu_stage0, tokens, slot_index, logits_out);
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
    const attention_kernels: AttentionKernelSet = switch (self.kv_cache_dtype) {
        .f16 => .{
            .attn_scores_heads_f16_kv_function = self.attn_scores_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_weighted_sum_heads_f16_kv_function = self.attn_weighted_sum_heads_f16_kv_function orelse return error.CudaKernelUnavailable,
            .attn_fused_heads_f16_kv_function = self.attn_fused_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_function = self.attn_fused_prefill_heads_f16_kv_function,
            .attn_fused_prefill_heads_f16_kv_gqa_function = self.attn_fused_prefill_heads_f16_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
        .i8 => .{
            .attn_scores_heads_i8_kv_function = self.attn_scores_heads_i8_kv_function,
            .attn_weighted_sum_heads_i8_kv_function = self.attn_weighted_sum_heads_i8_kv_function,
            .attn_fused_heads_i8_kv_function = self.attn_fused_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_function = self.attn_fused_prefill_heads_i8_kv_function,
            .attn_fused_prefill_heads_i8_kv_gqa_function = self.attn_fused_prefill_heads_i8_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
        .fp8 => .{
            .attn_scores_heads_fp8_kv_function = self.attn_scores_heads_fp8_kv_function,
            .attn_weighted_sum_heads_fp8_kv_function = self.attn_weighted_sum_heads_fp8_kv_function,
            .attn_fused_heads_fp8_kv_function = self.attn_fused_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_function = self.attn_fused_prefill_heads_fp8_kv_function,
            .attn_fused_prefill_heads_fp8_kv_gqa_function = self.attn_fused_prefill_heads_fp8_kv_gqa_function,
            .softmax_rows_function = self.softmax_rows_function,
            .causal_attn_softmax_f32_function = self.causal_attn_softmax_f32_function,
        },
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
                        self.prefill_rope_positions_cached_dirty = true;
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
        var gemma_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (comptime @hasDecl(SelfType, "maybeCaptureGemma4SourceEmbeddings")) {
            gemma_source_embeddings_opt = try self.maybeCaptureGemma4SourceEmbeddings(rows);
        }
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
                if (self.kv_cache_dtype == .f16) (self.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                if (self.kv_cache_dtype == .f16) self.kv_write_f16_function else null,
                if (self.kv_cache_dtype == .f16) self.rope_store_f16_function else null,
                self.shortconv_step_function orelse return error.CudaKernelUnavailable,
                attention_kernels,
                null,
            );
            if (comptime @hasDecl(SelfType, "applyGemma4PerLayerBranch")) {
                if (gemma_source_embeddings_opt) |*gemma_source_embeddings| {
                    try self.applyGemma4PerLayerBranch(
                        layer_idx,
                        chunk_tokens,
                        gemma_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden_rows = self.runtime_buffers.input_dev;
                } else if (comptime @hasDecl(SelfType, "applyStandaloneLayerScalar")) {
                    try self.applyStandaloneLayerScalar(layer_idx, &self.runtime_buffers.input_dev, rows);
                }
            }
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
                    .fp8 => "matmul_lm_head_fp8_host",
                    .mxfp8 => "matmul_lm_head_mxfp8_host",
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
            applyHostLogitsPostProcess(
                logits_out,
                self.loaded.config.logits_scaling,
                self.loaded.config.final_logit_softcapping,
            );
            if (self.loaded.config.logits_scaling != 1.0 and trace.isEnabled()) {
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

        pos_base += rows;
    }
}

pub fn ensureKvCapacity(self: anytype, required_tokens: usize) !void {
    if (required_tokens == 0) return;
    if (required_tokens > self.max_seq_len) return error.InvalidArgument;
    const copy_function = self.copy_function orelse return error.CudaKernelUnavailable;
    const copy_u16_function = self.copy_u16_function orelse return error.CudaKernelUnavailable;

    var grew_any = false;
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
            if (new_kv_pair.v_scale.pointer != 0) new_kv_pair.v_scale.deinit(&self.device);
            if (new_kv_pair.k_scale.pointer != 0) new_kv_pair.k_scale.deinit(&self.device);
            new_kv_pair.v.deinit(&self.device);
            new_kv_pair.k.deinit(&self.device);
        }

        if (block.kv_capacity > 0) {
            const old_elems = std.math.mul(usize, block.kv_capacity, block.kv_dim) catch return error.InvalidArgument;
            switch (self.kv_cache_dtype) {
                .f16 => {
                    const old_count_u32: u32 = @intCast(old_elems);
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.k_cache,
                        &new_kv_pair.k,
                        old_count_u32,
                    );
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.v_cache,
                        &new_kv_pair.v,
                        old_count_u32,
                    );
                },
                .i8, .fp8 => {
                    // i8/fp8 cache: copy bytes via copy_u16 with halved count.
                    const old_u16_count: u32 = @intCast((old_elems + 1) / 2);
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.k_cache,
                        &new_kv_pair.k,
                        old_u16_count,
                    );
                    try compute.cuda.copy_u16.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_u16_function,
                        &block.v_cache,
                        &new_kv_pair.v,
                        old_u16_count,
                    );
                    // Copy scale buffers (f32 elements: capacity * n_kv_heads).
                    const old_scale_elems = std.math.mul(usize, block.kv_capacity, self.n_kv_heads) catch return error.InvalidArgument;
                    const old_scale_count_u32: u32 = @intCast(old_scale_elems);
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function,
                        &block.k_scale,
                        &new_kv_pair.k_scale,
                        old_scale_count_u32,
                    );
                    try compute.cuda.copy.runWithFunction(
                        &self.kernel_arg_pack,
                        &self.device,
                        copy_function,
                        &block.v_scale,
                        &new_kv_pair.v_scale,
                        old_scale_count_u32,
                    );
                },
            }
        }

        if (block.v_scale.pointer != 0) block.v_scale.deinit(&self.device);
        if (block.k_scale.pointer != 0) block.k_scale.deinit(&self.device);
        block.k_cache.deinit(&self.device);
        block.v_cache.deinit(&self.device);
        block.k_cache = new_kv_pair.k;
        block.v_cache = new_kv_pair.v;
        block.k_scale = new_kv_pair.k_scale;
        block.v_scale = new_kv_pair.v_scale;
        block.kv_capacity = new_capacity;
        grew_any = true;
    }
    if (grew_any) {
        const SelfType = @TypeOf(self.*);
        if (comptime @hasField(SelfType, "decode_ptr_tables_dirty")) {
            self.decode_ptr_tables_dirty = true;
            if (comptime @hasField(SelfType, "decode_ptr_tables_cached_rows")) {
                self.decode_ptr_tables_cached_rows = 0;
            }
        }
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
        const BlockType = @TypeOf(block.*);
        if (comptime @hasDecl(BlockType, "ssmStateDataBytes") and
            @hasDecl(BlockType, "ssmStateScalesCount") and
            @hasField(BlockType, "ssm_state_format") and
            @hasField(BlockType, "ssm_state_scales_offset"))
        {
            const ssm_data_bytes = block.ssmStateDataBytes() catch continue;
            switch (block.ssm_state_format) {
                .f32 => {
                    const ssm_elems = std.math.divExact(usize, ssm_data_bytes, @sizeOf(f32)) catch continue;
                    const zeros = self.allocator.alloc(f32, ssm_elems) catch continue;
                    defer self.allocator.free(zeros);
                    @memset(zeros, 0.0);
                    block.ssm_state_dev.upload(&self.device, std.mem.sliceAsBytes(zeros)) catch {};
                },
                .i8_per_column_scale => {
                    const zeros_i8 = self.allocator.alloc(i8, ssm_data_bytes) catch continue;
                    defer self.allocator.free(zeros_i8);
                    @memset(zeros_i8, 0);
                    var ssm_i8_dev = bufferSlice(&block.ssm_state_dev, 0, ssm_data_bytes) catch continue;
                    ssm_i8_dev.upload(&self.device, std.mem.sliceAsBytes(zeros_i8)) catch {};

                    const scale_count = block.ssmStateScalesCount();
                    if (scale_count > 0) {
                        const scale_bytes = std.math.mul(usize, scale_count, @sizeOf(f32)) catch continue;
                        const scales = self.allocator.alloc(f32, scale_count) catch continue;
                        defer self.allocator.free(scales);
                        @memset(scales, 1.0);
                        var scales_dev = bufferSlice(&block.ssm_state_dev, @as(usize, block.ssm_state_scales_offset), scale_bytes) catch continue;
                        scales_dev.upload(&self.device, std.mem.sliceAsBytes(scales)) catch {};
                    }
                },
            }
        } else {
            const ssm_elems = std.math.divExact(usize, block.ssm_state_dev.size, @sizeOf(f32)) catch continue;
            const zeros = self.allocator.alloc(f32, ssm_elems) catch continue;
            defer self.allocator.free(zeros);
            @memset(zeros, 0.0);
            block.ssm_state_dev.upload(&self.device, std.mem.sliceAsBytes(zeros)) catch {};
        }
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

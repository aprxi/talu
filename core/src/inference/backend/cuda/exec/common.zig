//! Shared helpers for CUDA execution routes.
//!
//! Route entrypoints live in sibling files. This module is deliberately limited
//! to helper logic used by more than one route.

const std = @import("std");
const compute = @import("compute_pkg");
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const per_layer_branch_feature = @import("../per_layer_branch.zig");

const engine_ops = @import("../operators/root.zig");

const engine_types = @import("../runtime/root.zig");
const AttentionKernelSet = engine_types.AttentionKernelSet;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const saturatingU64FromU128 = engine_types.saturatingU64FromU128;

const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;

const kv_capacity = @import("kv_capacity.zig");
const resets = @import("resets.zig");
const ensureKvCapacity = kv_capacity.ensureKvCapacity;
const resetShortConvStates = resets.resetShortConvStates;
const resetGatedDeltaStates = resets.resetGatedDeltaStates;
const resetAttentionCpuStates = resets.resetAttentionCpuStates;

pub fn typeHasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

pub fn logDecodeInventoryOnce(self: anytype, mode_label: []const u8, token_count: usize, batch_rows: usize) void {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasField(SelfType, "decode_inventory_logged") and @hasDecl(SelfType, "logDecodeInventorySummaryImpl")) {
        if (!self.decode_inventory_logged) {
            self.logDecodeInventorySummaryImpl(mode_label, token_count, batch_rows);
            self.decode_inventory_logged = true;
        }
    }
}

/// Download and log first N f32 values + L2 norm from a device buffer.
/// Gated by TALU_DUMP_HIDDEN env var. Uses log.warn so it survives ReleaseFast.
pub fn dumpHiddenState(
    self: anytype,
    buf: *const compute.cuda.Buffer,
    global_layer_idx: usize,
    label: []const u8,
    d_model: usize,
    rows: usize,
) void {
    const dump_env = @import("env_pkg").getenv("TALU_DUMP_HIDDEN");
    if (dump_env == null) return;
    _ = rows;

    // Sync the entire CUDA context first.
    self.device.synchronize() catch |err| {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{ .layer = global_layer_idx, .label = label, .reason = "sync_err", .err = @intFromError(err) });
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

    self.device.makeCurrent() catch |err| {
        log.warn("inference", "DUMP_HIDDEN_SKIP", .{ .layer = global_layer_idx, .label = label, .reason = "make_current_err", .err = @intFromError(err) });
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
    for (host) |value| sum += @as(f64, value) * @as(f64, value);
    const l2_norm: f32 = @floatCast(@sqrt(sum));

    const n8 = @min(8, n);
    var values: [8]f32 = .{0} ** 8;
    @memcpy(values[0..n8], host[0..n8]);

    log.warn("inference", "DUMP_HIDDEN", .{
        .layer = global_layer_idx,
        .label = label,
        .l2_norm = l2_norm,
        .v0 = values[0],
        .v1 = values[1],
        .v2 = values[2],
        .v3 = values[3],
        .v4 = values[4],
        .v5 = values[5],
        .v6 = values[6],
        .v7 = values[7],
    });
}

pub fn applyHostLogitsPostProcess(
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

pub const PrefillMath = struct {
    d_model: usize,
    row_bytes: usize,
    d_model_u32: u32,
    head_dim_u32: u32,
    rope_dim_u32: u32,
    n_heads_u32: u32,
    n_kv_heads_u32: u32,
    global_rope_theta: f32,
    local_rope_theta: f32,
};

pub const PrefillChunkContext = struct {
    rows: usize,
    active_rows_u32: u32,
    seq_len_u32: u32,
    last_position: usize,
    last_position_u32: u32,
    pos_base: usize,
};

pub fn prefillMath(self: anytype) !PrefillMath {
    const global_rope_theta: f32 = if (self.loaded.config.rope_theta > 1.0) self.loaded.config.rope_theta else 10000.0;
    const local_rope_theta: f32 = if (self.loaded.config.rope_local_theta > 1.0 and self.loaded.config.sliding_window > 0)
        self.loaded.config.rope_local_theta
    else
        global_rope_theta;
    return .{
        .d_model = self.d_model,
        .row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument,
        .d_model_u32 = @intCast(self.d_model),
        .head_dim_u32 = @intCast(self.head_dim),
        .rope_dim_u32 = @intCast(self.rope_dim),
        .n_heads_u32 = @intCast(self.n_heads),
        .n_kv_heads_u32 = @intCast(self.n_kv_heads),
        .global_rope_theta = global_rope_theta,
        .local_rope_theta = local_rope_theta,
    };
}

pub fn prefillChunkContext(pos_base: usize, rows: usize) !PrefillChunkContext {
    const last_position = std.math.add(usize, pos_base, rows - 1) catch return error.InvalidArgument;
    return .{
        .rows = rows,
        .active_rows_u32 = @intCast(rows),
        .seq_len_u32 = @intCast(pos_base + rows),
        .last_position = last_position,
        .last_position_u32 = @intCast(last_position),
        .pos_base = pos_base,
    };
}

pub fn prepareCudaPrefillBackend(backend: anytype, total_rows: usize) !void {
    try ensureKvCapacity(backend, total_rows);
    try resetShortConvStates(backend);
    resetAttentionCpuStates(backend);
    resetGatedDeltaStates(backend);
}

pub fn localActivationByteCountFor(self: anytype) !usize {
    const SelfType = @TypeOf(self.*);
    if (comptime @hasDecl(SelfType, "localActivationByteCount")) {
        return self.localActivationByteCount();
    }
    return std.math.mul(usize, self.d_model, @sizeOf(f32)) catch error.InvalidArgument;
}

pub fn buildAttentionKernelSet(backend: anytype) !AttentionKernelSet {
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

pub fn populateCudaPrefillInputRows(
    self: anytype,
    chunk_tokens: []const u32,
    rows: usize,
    row_bytes: usize,
) !void {
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
}

pub fn executeGpuPrefillLayers(
    backend: anytype,
    slot_index: usize,
    chunk_tokens: []const u32,
    math: PrefillMath,
    chunk: PrefillChunkContext,
    attention_kernels: AttentionKernelSet,
    per_layer_source_embeddings_opt: ?compute.cuda.Buffer,
    branch_enabled: bool,
    dump_layer_offset: ?usize,
    layer_limit: usize,
) !compute.cuda.Buffer {
    var final_hidden_rows = backend.runtime_buffers.input_dev;
    var source_embeddings = per_layer_source_embeddings_opt;
    var layer_idx: usize = 0;
    while (layer_idx < layer_limit) : (layer_idx += 1) {
        const layer = &backend.block_runtime.blocks[layer_idx];
        final_hidden_rows = try backend.tryExecuteLayerProgram(
            layer,
            slot_index,
            layer_idx,
            math.d_model_u32,
            math.head_dim_u32,
            math.rope_dim_u32,
            math.n_heads_u32,
            math.n_kv_heads_u32,
            chunk.active_rows_u32,
            chunk.seq_len_u32,
            chunk.seq_len_u32,
            chunk.pos_base,
            chunk.last_position,
            chunk.last_position_u32,
            math.global_rope_theta,
            math.local_rope_theta,
            backend.rope_function orelse return error.CudaKernelUnavailable,
            backend.copy_function orelse return error.CudaKernelUnavailable,
            if (backend.kv_cache_dtype == .f16) (backend.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
            if (backend.kv_cache_dtype == .f16) backend.kv_write_f16_function else null,
            if (backend.kv_cache_dtype == .f16) backend.rope_store_f16_function else null,
            backend.shortconv_step_function orelse return error.CudaKernelUnavailable,
            attention_kernels,
            null,
        );
        if (dump_layer_offset) |offset| {
            dumpHiddenState(backend, &backend.runtime_buffers.input_dev, offset + layer_idx, "post_layer", backend.d_model, 1);
        }
        if (branch_enabled) {
            if (source_embeddings) |*per_layer_source_embeddings| {
                try per_layer_branch_feature.applyPerLayerBranch(
                    backend,
                    layer_idx,
                    chunk_tokens,
                    per_layer_source_embeddings,
                    &backend.runtime_buffers.input_dev,
                );
                final_hidden_rows = backend.runtime_buffers.input_dev;
                if (dump_layer_offset) |offset| {
                    dumpHiddenState(backend, &backend.runtime_buffers.input_dev, offset + layer_idx, "post_ple", backend.d_model, 1);
                }
            } else if (per_layer_branch_feature.hasStandaloneLayerScalars(backend)) {
                try per_layer_branch_feature.applyStandaloneLayerScalar(backend, layer_idx, &backend.runtime_buffers.input_dev, chunk.rows);
            }
        }
    }
    return final_hidden_rows;
}

fn writeProjectedLogits(backend: anytype, logits_out: []f32) void {
    if (backend.runtime_buffers.projected_vocab == logits_out.len) {
        @memcpy(logits_out, backend.runtime_buffers.projected_logits_host);
    } else {
        @memset(logits_out, -1.0e9);
        @memcpy(logits_out[0..backend.runtime_buffers.projected_vocab], backend.runtime_buffers.projected_logits_host);
    }
}

fn emitLmHeadTrace(backend: anytype, last_position: usize) void {
    const rows128: u128 = 1;
    const d_model128: u128 = @intCast(backend.d_model);
    const vocab128: u128 = @intCast(backend.runtime_buffers.projected_vocab);
    const total_flops = saturatingU64FromU128(2 * rows128 * d_model128 * vocab128);
    const total_bytes_lm = saturatingU64FromU128(
        rows128 * d_model128 * @sizeOf(f32) +
            @as(u128, backend.runtime_buffers.projection_weight.byteSize()) +
            rows128 * vocab128 * @sizeOf(f32),
    );
    const kernel_name = switch (backend.runtime_buffers.projection_weight) {
        .dense_f32 => "matmul_lm_head_f32_host",
        .dense_u16 => |w| switch (w.dtype) {
            .bf16 => "matmul_lm_head_bf16_host",
            .f16 => "matmul_lm_head_f16_host",
        },
        .gaffine_u4 => "matmul_lm_head_gaffine_u4_host",
        .gaffine_u8 => "matmul_lm_head_gaffine_u8_host",
        .fp8 => "matmul_lm_head_fp8_host",
        .mxfp8 => "matmul_lm_head_mxfp8_host",
        .nvfp4 => "matmul_lm_head_nvfp4_host",
    };
    trace.emitFinalWithWork(
        .lm_head,
        @intCast(last_position),
        0,
        @ptrCast(backend.runtime_buffers.projected_logits_host.ptr),
        .f32,
        .{ @intCast(backend.runtime_buffers.projected_vocab), 0, 0, 0 },
        1,
        kernel_name,
        .{ .flops = total_flops, .bytes = total_bytes_lm },
    );
}

pub fn projectFinalLogitsFromCudaStage(
    backend: anytype,
    final_hidden_rows: compute.cuda.Buffer,
    rows: usize,
    row_bytes: usize,
    last_position: usize,
    logits_out: []f32,
    trace_final_norm_label: ?[]const u8,
) !void {
    var hidden_rows = final_hidden_rows;
    const last_row_in_chunk = rows - 1;
    const last_offset = std.math.mul(usize, last_row_in_chunk, row_bytes) catch return error.InvalidArgument;
    var last_hidden = try bufferSlice(&hidden_rows, last_offset, row_bytes);
    var last_norm = try bufferSlice(&backend.runtime_buffers.norm_out_dev, 0, row_bytes);
    try compute.cuda.rmsnorm.runWithFunction(
        &backend.kernel_arg_pack,
        &backend.device,
        backend.rmsnorm_function orelse return error.CudaKernelUnavailable,
        &last_hidden,
        &backend.runtime_buffers.norm_weight_dev,
        &last_norm,
        1,
        @intCast(backend.d_model),
        backend.norm_eps,
        backend.loaded.runtime.weight_offset,
    );
    if (trace_final_norm_label) |label| {
        if (trace.isEnabled()) {
            try last_norm.download(&backend.device, std.mem.sliceAsBytes(backend.runtime_buffers.hidden_host));
            trace.emitFinal(
                .final_norm,
                @intCast(last_position),
                1,
                @ptrCast(backend.runtime_buffers.hidden_host.ptr),
                .f32,
                .{ @intCast(backend.d_model), 0, 0, 0 },
                1,
                label,
            );
        }
    }

    try engine_ops.linearForwardRows(backend, &last_norm, 1, &backend.runtime_buffers.projection_weight, &backend.runtime_buffers.logits_dev);
    try backend.runtime_buffers.logits_dev.download(&backend.device, std.mem.sliceAsBytes(backend.runtime_buffers.projected_logits_host));
    if (trace_final_norm_label != null and trace.isEnabled()) {
        emitLmHeadTrace(backend, last_position);
    }

    writeProjectedLogits(backend, logits_out);
    applyHostLogitsPostProcess(
        logits_out,
        backend.loaded.config.logits_scaling,
        backend.loaded.config.final_logit_softcapping,
    );
    if (trace_final_norm_label != null and backend.loaded.config.logits_scaling != 1.0 and trace.isEnabled()) {
        trace.emitFinal(
            .logits_scaled,
            @intCast(last_position),
            0,
            @ptrCast(logits_out.ptr),
            .f32,
            .{ @intCast(backend.vocab_size), 0, 0, 0 },
            1,
            null,
        );
    }
}

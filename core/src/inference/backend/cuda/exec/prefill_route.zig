//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const tensor = @import("compute_pkg").tensor;
const log = @import("log_pkg");
const trace = @import("xray_pkg").trace;
const per_layer_branch_feature = @import("../per_layer_branch.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/root.zig");
const BatchDecodeInfo = engine_types.BatchDecodeInfo;
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_dispatch_observability = engine_types.enable_dispatch_observability;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const AttentionKernelSet = engine_types.AttentionKernelSet;

// --- Compute ops from engine_ops.zig ---
const engine_ops = @import("../operators/root.zig");

// --- Utilities from engine_weights.zig ---
const engine_weights = @import("../weights/root.zig");
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

fn rejectUnsupportedStagedPrefillRoute(topology_tag: ?[]const u8) !void {
    const tag = topology_tag orelse return;
    if (std.mem.eql(u8, tag, "pipeline2") or
        std.mem.eql(u8, tag, "cpu_gpu") or
        std.mem.eql(u8, tag, "cpu_gpu_gpu"))
    {
        return error.UnsupportedModel;
    }
}

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
const common = @import("common.zig");
const kv_capacity = @import("kv_capacity.zig");
const resets = @import("resets.zig");
const stage_adapters = @import("stage_adapters.zig");
const staged_prefill = @import("staged_prefill.zig");
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const dumpHiddenState = common.dumpHiddenState;
const applyHostLogitsPostProcess = common.applyHostLogitsPostProcess;
const executeCpuStage0LayerRange = common.executeCpuStage0LayerRange;
const executeLocalPrefillCudaCuda = staged_prefill.executeLocalPrefillCudaCuda;
const executeLocalPrefillCpuCuda = staged_prefill.executeLocalPrefillCpuCuda;
const executeLocalPrefillCpuCudaCuda = staged_prefill.executeLocalPrefillCpuCudaCuda;
const ensureKvCapacity = kv_capacity.ensureKvCapacity;
const resetShortConvStates = resets.resetShortConvStates;
const resetGatedDeltaStates = resets.resetGatedDeltaStates;
const resetAttentionCpuStates = resets.resetAttentionCpuStates;

pub fn resolveStagedPrefillChunkRows(total_rows: usize, requested_cap: usize, env_override: bool) usize {
    const clamped = @max(@as(usize, 1), @min(total_rows, requested_cap));
    if (env_override) return clamped;
    // Empirical tuning on Blackwell NVFP4:
    // medium prefill lengths benefit from a slightly smaller staged chunk.
    if (total_rows >= 384 and total_rows <= 640 and clamped >= 254) return 254;
    return clamped;
}

pub fn executePrefillWithLayerLimit(
    self: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
    layer_limit: usize,
) !void {
    const SelfType = @TypeOf(self.*);

    if (comptime @hasDecl(SelfType, "executePrefillWithLayerLimitTestHook")) {
        return self.executePrefillWithLayerLimitTestHook(tokens, slot_index, logits_out, layer_limit);
    }
    if (comptime @hasDecl(SelfType, "executeDecodeWithLayerLimitTestHook") and
        !@hasDecl(SelfType, "localCpuStage0"))
    {
        if (tokens.len == 0) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;
        if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;

        for (tokens, 0..) |token, position| {
            const is_last = position + 1 == tokens.len;
            try self.executeDecodeWithLayerLimitTestHook(
                token,
                position,
                slot_index,
                if (is_last) logits_out else null,
                layer_limit,
                true,
                is_last,
                true,
                @intCast(position + 1),
                position,
                null,
                null,
                null,
                false,
            );
        }
        return;
    }

    // Local multi-stage prefill is selected from the bridge-validated local
    // pipeline shape. The CUDA backend only binds concrete local adapters.
    if (layer_limit == self.block_runtime.blocks.len) {
        if (try stage_adapters.localPipelinePlacementKind(self)) |placement_kind| {
            if (tokens.len == 0) return error.InvalidArgument;
            if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
            if (logits_out.len != self.vocab_size) return error.InvalidArgument;
            if (tokens.len > self.max_seq_len) return error.InvalidArgument;

            switch (placement_kind) {
                .cuda_cuda => {
                    if (comptime @hasDecl(SelfType, "localCudaStage1")) {
                        var stage1 = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
                        if (stage1.state_descriptor_count > 0) try stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                        stage1.activateKvSlot(slot_index);
                        self.activateKvSlot(slot_index);
                        return executeLocalPrefillCudaCuda(self, stage1, tokens, slot_index, logits_out);
                    }
                },
                .cpu_cuda => {
                    if (comptime @hasDecl(SelfType, "localCpuStage0")) {
                        const cpu_stage0 = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
                        self.activateKvSlot(slot_index);
                        return executeLocalPrefillCpuCuda(self, cpu_stage0, tokens, slot_index, logits_out);
                    }
                },
                .cpu_cuda_cuda => {
                    if (comptime @hasDecl(SelfType, "localCpuStage0") and @hasDecl(SelfType, "localCudaStage1")) {
                        const cpu_stage0 = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
                        var gpu_stage1 = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
                        self.activateKvSlot(slot_index);
                        if (gpu_stage1.state_descriptor_count > 0) try gpu_stage1.mirrorSlotStateBlocksFrom(self, slot_index);
                        gpu_stage1.activateKvSlot(slot_index);
                        return executeLocalPrefillCpuCudaCuda(self, cpu_stage0, gpu_stage1, tokens, slot_index, logits_out);
                    }
                },
                .generic_local_chain => {},
            }
            return error.InvalidTopologyConfig;
        }
    }

    if ((topologyModeIs(self, "pipeline2") or topologyModeIs(self, "cpu_gpu") or topologyModeIs(self, "cpu_gpu_gpu")) and
        layer_limit == self.block_runtime.blocks.len)
    {
        if (tokens.len == 0) return error.InvalidArgument;
        if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
        if (tokens.len > self.max_seq_len) return error.InvalidArgument;
        return rejectUnsupportedStagedPrefillRoute(topologyModeTag(self));
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
        var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
            per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, rows);
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
            if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
                if (per_layer_source_embeddings_opt) |*per_layer_source_embeddings| {
                    try per_layer_branch_feature.applyPerLayerBranch(
                        self,
                        layer_idx,
                        chunk_tokens,
                        per_layer_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden_rows = self.runtime_buffers.input_dev;
                } else if (per_layer_branch_feature.hasStandaloneLayerScalars(self)) {
                    try per_layer_branch_feature.applyStandaloneLayerScalar(self, layer_idx, &self.runtime_buffers.input_dev, rows);
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
                    .nvfp4 => "matmul_lm_head_nvfp4_host",
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

test "rejectUnsupportedStagedPrefillRoute rejects staged prefill route" {
    try std.testing.expectError(
        error.UnsupportedModel,
        rejectUnsupportedStagedPrefillRoute("pipeline2"),
    );
}

test "rejectUnsupportedStagedPrefillRoute allows single-device tag" {
    try rejectUnsupportedStagedPrefillRoute("single");
    try rejectUnsupportedStagedPrefillRoute(null);
}

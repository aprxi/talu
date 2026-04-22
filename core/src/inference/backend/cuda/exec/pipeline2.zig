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
const staged_orchestrator = @import("../../staged_orchestrator.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

// --- Shared types from engine_types.zig ---
const engine_types = @import("../runtime/_types_impl.zig");
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

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
const common = @import("common.zig");
const kv_capacity = @import("kv_capacity.zig");
const decode_route = @import("decode_route.zig");
const resets = @import("resets.zig");
const transfers = @import("transfers.zig");
const resolveStagedPrefillChunkRows = @import("prefill_route.zig").resolveStagedPrefillChunkRows;
const transferPipelineActivationMultiRow = transfers.transferPipelineActivationMultiRow;
const dumpHiddenState = common.dumpHiddenState;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const applyHostLogitsPostProcess = common.applyHostLogitsPostProcess;
const ensureKvCapacity = kv_capacity.ensureKvCapacity;
const resetShortConvStates = resets.resetShortConvStates;
const resetGatedDeltaStates = resets.resetGatedDeltaStates;
const resetAttentionCpuStates = resets.resetAttentionCpuStates;
const ensureGatedDeltaHostStageCapacity = resets.ensureGatedDeltaHostStageCapacity;
const computeGpuPrototypeLogitsWithLayerLimit = decode_route.computeGpuPrototypeLogitsWithLayerLimit;

pub fn computeBatchedPrefillPipeline2(
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

    const attn_kernels_0 = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;
    const attn_kernels_1 = buildAttentionKernelSet(stage1) catch return error.CudaKernelUnavailable;

    const staged_chunk_cap_base = @min(self.prefill_chunk_rows_cap, stage1.prefill_chunk_rows_cap);
    const chunk_cap = resolveStagedPrefillChunkRows(
        total_rows,
        staged_chunk_cap_base,
        @import("env_pkg").getenv("TALU_CUDA_PREFILL_CHUNK_ROWS") != null,
    );

    // ── per-layer branch input: compute source embeddings on host ──
    const Stage1Type = @TypeOf(stage1.*);
    const has_per_layer_branch_0 = per_layer_branch_feature.hasPerLayerBranchRuntime(self);
    const has_per_layer_branch_1 = per_layer_branch_feature.hasPerLayerBranchRuntime(stage1);
    const per_layer_branch_active_0 = has_per_layer_branch_0 and (if (comptime @hasField(SelfType, "per_layer_branch_runtime")) self.per_layer_branch_runtime != null else false);
    const per_layer_branch_active_1 = has_per_layer_branch_1 and (if (comptime @hasField(Stage1Type, "per_layer_branch_runtime")) stage1.per_layer_branch_runtime != null else false);
    const need_source_embeddings = per_layer_branch_active_0 or per_layer_branch_active_1;
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
        var per_layer_source_embeddings_0: ?compute.cuda.Buffer = null;
        if (per_layer_branch_active_0) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&self.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                per_layer_source_embeddings_0 = se_dst;
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
                if (has_per_layer_branch_0) {
                    if (per_layer_source_embeddings_0) |*per_layer_se| {
                        try per_layer_branch_feature.applyPerLayerBranch(
                            self,
                            layer_idx,
                            chunk_tokens,
                            per_layer_se,
                            &self.runtime_buffers.input_dev,
                        );
                    } else if (per_layer_branch_feature.hasStandaloneLayerScalars(self)) {
                        try per_layer_branch_feature.applyStandaloneLayerScalar(self, layer_idx, &self.runtime_buffers.input_dev, rows);
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
        var per_layer_source_embeddings_1: ?compute.cuda.Buffer = null;
        if (per_layer_branch_active_1) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&stage1.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&stage1.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                per_layer_source_embeddings_1 = se_dst;
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
                    if (stage1.kv_cache_dtype == .f16) (stage1.cast_f32_to_f16_function orelse return error.CudaKernelUnavailable) else null,
                    if (stage1.kv_cache_dtype == .f16) stage1.kv_write_f16_function else null,
                    if (stage1.kv_cache_dtype == .f16) stage1.rope_store_f16_function else null,
                    stage1.shortconv_step_function orelse return error.CudaKernelUnavailable,
                    attn_kernels_1,
                    null,
                );
                if (has_per_layer_branch_1) {
                    if (per_layer_source_embeddings_1) |*per_layer_se| {
                        try per_layer_branch_feature.applyPerLayerBranch(
                            stage1,
                            layer_idx,
                            chunk_tokens,
                            per_layer_se,
                            &stage1.runtime_buffers.input_dev,
                        );
                        stage1_final_hidden = stage1.runtime_buffers.input_dev;
                    } else if (per_layer_branch_feature.hasStandaloneLayerScalars(stage1)) {
                        try per_layer_branch_feature.applyStandaloneLayerScalar(stage1, layer_idx, &stage1.runtime_buffers.input_dev, rows);
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
                    .nvfp4 => "matmul_lm_head_nvfp4_host",
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

pub fn runPipeline2WithPipelineRuntime(
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
            computeGpuPrototypeLogitsWithLayerLimit(
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
            ) catch |err| {
                log.warn("inference", "CUDA pipeline2 stage0 executeLayers failed", .{
                    .slot = stage.ctx.slot_index,
                    .position = stage.ctx.position,
                    .layer_start = layer_start,
                    .layer_end = layer_end,
                    .local_layer_limit = local_layer_limit,
                    .ensure_kv_capacity = @as(u8, @intFromBool(stage.ctx.ensure_kv_capacity)),
                    .reason = @errorName(err),
                });
                return err;
            };
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
            computeGpuPrototypeLogitsWithLayerLimit(
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
            ) catch |err| {
                log.warn("inference", "CUDA pipeline2 stage1 executeLayers failed", .{
                    .slot = stage.ctx.slot_index,
                    .position = stage.ctx.position,
                    .layer_start = layer_start,
                    .layer_end = layer_end,
                    .local_layer_limit = local_layer_limit,
                    .ensure_kv_capacity = @as(u8, @intFromBool(stage.ctx.ensure_kv_capacity)),
                    .reason = @errorName(err),
                });
                return err;
            };
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

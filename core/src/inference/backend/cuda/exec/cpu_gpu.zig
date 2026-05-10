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
const uploadCpuKvToMirrors = transfers.uploadCpuKvToMirrors;
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

pub fn computeBatchedPrefillCpuGpu(
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

    // For per-layer branch-input: capture source embeddings (raw scaled
    // embed_tokens output) before CPU layers modify the hidden states.
    const per_layer_branch_active = per_layer_branch_feature.hasPerLayerBranchRuntime(self);
    const has_per_layer_branch = per_layer_branch_active and self.per_layer_branch_runtime != null;
    const source_embeddings_host: ?[]f32 = if (has_per_layer_branch)
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
        var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (source_embeddings_host) |se_host| {
            // Pipeline mode: upload source embeddings from CPU to deepstack_add_dev.
            const se_chunk_offset = pos_base * d_model;
            const se_chunk_f32s = rows * d_model;
            const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
            var se_dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, se_bytes);
            try se_dst.upload(&self.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
            per_layer_source_embeddings_opt = se_dst;
        } else if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
            per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, rows);
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
            if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
                if (per_layer_source_embeddings_opt) |*per_layer_source_embeddings| {
                    const chunk_tokens = tokens[pos_base .. pos_base + rows];
                    try per_layer_branch_feature.applyPerLayerBranch(
                        self,
                        layer_idx,
                        chunk_tokens,
                        per_layer_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden_rows = self.runtime_buffers.input_dev;
                    dumpHiddenState(self, &self.runtime_buffers.input_dev, self.split_layer + layer_idx, "post_ple", self.d_model, 1);
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

pub fn runCpuGpuWithPipelineRuntime(
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
            if (@import("env_pkg").getenv("TALU_DUMP_HIDDEN") != null) {
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

//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const models = @import("models_pkg");
const tensor = @import("compute_pkg").tensor;
const log = @import("log_pkg");
const bridge = @import("../../../bridge/root.zig");
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

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
const common = @import("common.zig");
const kv_capacity = @import("kv_capacity.zig");
const decode_route = @import("decode_route.zig");
const resets = @import("resets.zig");
const stage_adapters = @import("stage_adapters.zig");
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
const executeDecodeWithLayerLimit = decode_route.executeDecodeWithLayerLimit;

fn usizeToU64(value: usize, comptime err: anyerror) !u64 {
    return std.math.cast(u64, value) orelse return err;
}

fn cpuGpuFrameId(slot_id: u64, sequence_start: u64) !bridge.TensorFrameInstanceId {
    const raw = ((slot_id & 0xffff_ffff) << 32) | ((sequence_start +% 1) & 0xffff_ffff);
    return bridge.TensorFrameInstanceId.init(if (raw == 0) 1 else raw);
}

fn cpuGpuTensorFramePlanRef(self: anytype) !*const bridge.TensorFramePlanRef {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "cpu_gpu_tensor_frame_plan_ref")) return error.InvalidTopologyConfig;
    if (self.cpu_gpu_tensor_frame_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

fn cpuGpuLocalStageRunnerPlanRef(self: anytype) !*const bridge.LocalStageRunnerPlanRef {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "cpu_gpu_local_stage_runner_plan_ref")) return error.InvalidTopologyConfig;
    if (self.cpu_gpu_local_stage_runner_plan_ref) |*plan_ref| return plan_ref;
    return error.InvalidTopologyConfig;
}

fn cpuGpuPayloadLocationHint(self: anytype) !?bridge.TensorFramePayloadLocationHint {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return null;
    const ordinal = std.math.cast(u16, self.device.ordinal()) orelse return error.InvalidTopologyConfig;
    return .{ .cuda = ordinal };
}

fn cpuGpuSlotId(slot_index: usize) !u64 {
    return std.math.add(u64, try usizeToU64(slot_index, error.InvalidSlotId), 1) catch return error.InvalidSlotId;
}

fn cpuGpuRequestId(self: anytype, slot_index: usize) !u64 {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "slot_request_ids")) return error.InvalidRequestId;
    if (slot_index >= self.slot_request_ids.len) return error.InvalidSlotId;
    return self.slot_request_ids[slot_index] orelse error.InvalidRequestId;
}

fn buildCpuGpuActivationFrame(
    self: anytype,
    step_kind: bridge.TensorFrameStepKind,
    slot_index: usize,
    sequence_start: usize,
    token_count: usize,
    batch_entries: *[1]bridge.TensorFrameBatchEntry,
) !bridge.TensorFrameMetadata {
    const plan_ref = try cpuGpuTensorFramePlanRef(self);
    const boundary_index: usize = 0;
    const contract = try bridge.selectedBoundaryTensorContract(
        plan_ref,
        boundary_index,
        self.pipeline_boundary_dtype,
        self.pipeline_boundary_layout,
        .negotiated,
    );
    const hidden_size = try usizeToU64(self.d_model, error.InvalidHiddenSize);
    const token_count_u64 = try usizeToU64(token_count, error.InvalidSequenceRange);
    const tensor_desc = try bridge.TensorFrameTensorDesc.contiguousActivation(
        self.pipeline_boundary_dtype,
        .{ 1, token_count_u64, hidden_size, 0 },
    );
    const slot_id = try cpuGpuSlotId(slot_index);
    const sequence_start_u64 = try usizeToU64(sequence_start, error.InvalidSequenceRange);
    batch_entries[0] = .{
        .batch_index = 0,
        .request_id = try cpuGpuRequestId(self, slot_index),
        .slot_id = slot_id,
        .sequence_start = sequence_start_u64,
        .token_count = token_count_u64,
    };

    const args = bridge.ActivationFrameArgs{
        .frame_id = try cpuGpuFrameId(slot_id, sequence_start_u64),
        .plan_ref = plan_ref,
        .boundary_index = boundary_index,
        .selected_contract = &contract,
        .shape_context = .{
            .expected_hidden_size = hidden_size,
            .expected_step_kind = step_kind,
        },
        .tensor = tensor_desc,
        .batch = .{ .entries = batch_entries },
        .payload = .{
            .byte_count = tensor_desc.payload_byte_count,
            .location_hint = try cpuGpuPayloadLocationHint(self),
            .ownership = .borrowed_until_next_stage_call,
            .lifetime = .step_scoped,
        },
    };
    return switch (step_kind) {
        .decode => bridge.activationDecodeFrame(args),
        .prefill => bridge.activationPrefillFrame(args),
    };
}

fn buildCpuGpuDecodeFrame(
    self: anytype,
    ctx: *const stage_adapters.DecodeContext,
    batch_entries: *[1]bridge.TensorFrameBatchEntry,
) !bridge.TensorFrameMetadata {
    return buildCpuGpuActivationFrame(self, .decode, ctx.slot_index, ctx.position, 1, batch_entries);
}

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
        const chunk_bytes = std.mem.sliceAsBytes(prefill_buffer[chunk_offset..][0..chunk_f32s]);
        var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
        const activation_metadata = try buildCpuGpuActivationFrame(self, .prefill, slot_index, pos_base, rows, &batch_entries);
        try bridge.validateTensorFrameForPlanBoundary(&activation_metadata, try cpuGpuTensorFramePlanRef(self), 0);
        try bridge.validatePayloadBufferLength(&activation_metadata, chunk_bytes.len);
        try self.runtime_buffers.input_dev.upload(&self.device, chunk_bytes);

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
    const Stage0 = struct {
        backend: @TypeOf(cpu_stage0_backend),
        gpu_backend: @TypeOf(self),
        ctx: *const stage_adapters.DecodeContext,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            try stage_adapters.validateEmptyInput(input);
            try stage_adapters.executeCpuDecodeLayerRange(stage.backend, stage.ctx, layer_start, layer_end, false);
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try stage_adapters.downloadCpuActivation(stage.backend, stage.ctx.slot_index, host_buf, byte_count);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try stage_adapters.uploadCpuActivation(stage.backend, stage.ctx.slot_index, host_buf, byte_count);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            // Upload CPU source layer KV to GPU mirrors after CPU forward,
            // before GPU stage1 reads KV in attention.
            try uploadCpuKvToMirrors(stage.gpu_backend, stage.backend, stage.ctx.slot_index, stage.ctx.position, 1);
        }
    };
    const Stage1 = struct {
        backend: @TypeOf(self),
        ctx: *const stage_adapters.DecodeContext,

        pub fn executeLayers(stage: *@This(), input: []const u8, layer_start: usize, layer_end: usize) anyerror!void {
            if (@import("env_pkg").getenv("TALU_DUMP_HIDDEN") != null) {
                log.warn("inference", "STAGE1_EXEC", .{
                    .layer_start = layer_start,
                    .layer_end = layer_end,
                    .input_len = input.len,
                });
            }
            try stage_adapters.validateEmptyInput(input);
            try stage_adapters.executeCudaDecodeLayerRange(
                executeDecodeWithLayerLimit,
                stage.backend,
                stage.ctx,
                layer_start,
                layer_end,
                stage.ctx.logits_out_opt,
                stage.ctx.compute_logits,
                stage.ctx.download_logits,
                true,
            );
        }

        pub fn downloadActivation(stage: *@This(), host_buf: []u8, byte_count: usize) anyerror!void {
            try stage_adapters.downloadCudaActivation(stage.backend, host_buf, byte_count);
        }

        pub fn uploadActivation(stage: *@This(), host_buf: []const u8, byte_count: usize) anyerror!void {
            try stage_adapters.uploadCudaActivation(stage.backend, stage.ctx.slot_index, host_buf, byte_count);
        }

        pub fn synchronize(stage: *@This()) anyerror!void {
            try stage_adapters.synchronizeCudaBackend(stage.backend);
        }
    };
    var ctx = stage_adapters.DecodeContext{
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
    var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
    const activation_metadata = try buildCpuGpuDecodeFrame(self, &ctx, &batch_entries);
    const runner_plan_ref = try cpuGpuLocalStageRunnerPlanRef(self);
    var touched_stage_scratch: [2]bridge.LocalStageTouchedRef = undefined;
    const result = try bridge.executeLocalStageBoundary(
        Stage0,
        Stage1,
        .{ .backend = cpu_stage0_backend, .gpu_backend = self, .ctx = &ctx },
        .{ .backend = self, .ctx = &ctx },
        .{
            .plan_ref = runner_plan_ref,
            .step = .{
                .boundary_index = 0,
                .step_kind = .decode,
                .metadata = activation_metadata,
                .activation_byte_count = activation_byte_count,
                .expected_request_id = batch_entries[0].request_id,
                .expected_slot_id = batch_entries[0].slot_id,
            },
            .host_staging = self.pipeline_host_staging,
            .touched_stage_scratch = &touched_stage_scratch,
        },
    );
    switch (result) {
        .success => {},
        .failure => |failure| return failure.source_error orelse error.UnknownRunnerFailure,
    }
}

test "buildCpuGpuDecodeFrame binds cpu cuda handoff to TensorFramePlanRef" {
    const MockCudaDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return 7;
        }
    };
    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        split_layer: usize = 2,
        d_model: usize = 4,
        pipeline_boundary_dtype: bridge.BoundaryDType = .f32,
        pipeline_boundary_layout: bridge.BoundaryLayout = .row_major,
        cpu_gpu_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
        slot_request_ids: [3]?u64 = .{ null, null, 456 },
        block_runtime: BlockRuntimeMock = .{},
        device: MockCudaDevice = .{},
    };
    const plan_boundaries = [_]bridge.TensorFrameBoundaryRef{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }};
    const plan_ref = bridge.TensorFramePlanRef{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundaries = &plan_boundaries,
    };
    var backend = MockBackend{
        .cpu_gpu_tensor_frame_plan_ref = plan_ref,
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
    };
    const ctx = stage_adapters.DecodeContext{
        .token = 123,
        .position = 9,
        .slot_index = 2,
        .logits_out_opt = null,
        .compute_logits = false,
        .download_logits = false,
        .ensure_kv_capacity = true,
        .trace_seq_len_u32 = 10,
        .trace_pos_offset = 9,
    };
    var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
    const metadata = try buildCpuGpuDecodeFrame(&backend, &ctx, &batch_entries);

    try bridge.validateTensorFrameForPlanBoundary(&metadata, &plan_ref, 0);
    try std.testing.expectEqual(bridge.TensorFrameRole.activation, metadata.role);
    try std.testing.expectEqual(bridge.TensorFrameStepKind.decode, metadata.step_kind);
    try std.testing.expectEqual(@as(u64, 16), metadata.payload.byte_count);
    try std.testing.expectEqual(@as(u64, 9), metadata.batch.entries[0].sequence_start);
    try std.testing.expectEqual(@as(u64, 3), metadata.batch.entries[0].slot_id);
    try std.testing.expectEqual(@as(u64, 456), metadata.batch.entries[0].request_id);
    try std.testing.expectEqual(bridge.TensorFramePayloadLocationHint{ .cuda = 7 }, metadata.payload.location_hint.?);
}

test "buildCpuGpuActivationFrame validates prefill chunk metadata" {
    const MockCudaDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return 7;
        }
    };
    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        split_layer: usize = 2,
        d_model: usize = 4,
        pipeline_boundary_dtype: bridge.BoundaryDType = .f32,
        pipeline_boundary_layout: bridge.BoundaryLayout = .row_major,
        cpu_gpu_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
        slot_request_ids: [1]?u64 = .{789},
        block_runtime: BlockRuntimeMock = .{},
        device: MockCudaDevice = .{},
    };
    const plan_boundaries = [_]bridge.TensorFrameBoundaryRef{.{
        .boundary_index = 0,
        .source_stage_id = 0,
        .target_stage_id = 1,
        .producer_layer_start = 0,
        .producer_layer_end = 2,
        .consumer_layer_start = 2,
        .consumer_layer_end = 4,
    }};
    const plan_ref = bridge.TensorFramePlanRef{
        .allocator = std.testing.allocator,
        .identity = .{
            .graph_digest = [_]u8{1} ** 32,
            .graph_contract_version = 1,
            .stage_plan_contract_version = 1,
            .stage_plan_id = .{ .digest = [_]u8{2} ** 32 },
        },
        .boundaries = &plan_boundaries,
    };
    var backend = MockBackend{ .cpu_gpu_tensor_frame_plan_ref = plan_ref };
    var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
    const metadata = try buildCpuGpuActivationFrame(&backend, .prefill, 0, 4, 3, &batch_entries);

    try bridge.validateTensorFrameForPlanBoundary(&metadata, &plan_ref, 0);
    try bridge.validatePayloadBufferLength(&metadata, 3 * 4 * @sizeOf(f32));
    try std.testing.expectEqual(bridge.TensorFrameStepKind.prefill, metadata.step_kind);
    try std.testing.expectEqual(@as(u64, 789), metadata.batch.entries[0].request_id);
    try std.testing.expectEqual(@as(u64, 4), metadata.batch.entries[0].sequence_start);
    try std.testing.expectEqual(@as(u64, 3), metadata.batch.entries[0].token_count);
}

test "runCpuGpuWithPipelineRuntime rejects stale activation byte count before transfer" {
    const d_model: usize = 4;

    const MockCudaDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return 0;
        }
    };

    const TraceState = struct {
        stage0_execute_calls: usize = 0,
        stage0_slot_bytes_calls: usize = 0,
        stage1_upload_calls: usize = 0,
        stage1_execute_calls: usize = 0,
    };

    const CpuStage0Mock = struct {
        trace_state: *TraceState,
        activation: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn executeDecodeLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = logits_out_opt;
            _ = layer_start;
            _ = layer_end;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            self.trace_state.stage0_execute_calls += 1;
            @memset(self.activation[0..], 1.0);
        }

        pub fn slotActivationBytes(self: *@This(), _: usize) []const u8 {
            self.trace_state.stage0_slot_bytes_calls += 1;
            return std.mem.sliceAsBytes(self.activation[0..]);
        }

        pub fn slotActivationBytesMut(self: *@This(), _: usize) []u8 {
            return std.mem.sliceAsBytes(self.activation[0..]);
        }
    };

    const Stage1Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        trace_state: *TraceState,
        split_layer: usize = 2,
        d_model: usize = d_model,
        pipeline_boundary_dtype: bridge.BoundaryDType = .f32,
        pipeline_boundary_layout: bridge.BoundaryLayout = .row_major,
        pipeline_host_staging: ?[]align(64) u8 = null,
        cpu_gpu_stage_plan: ?models.stage_plan.StagePlan = null,
        cpu_gpu_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
        cpu_gpu_placement_plan: ?bridge.PlacementPlan = null,
        cpu_gpu_local_stage_runner_plan_ref: ?bridge.LocalStageRunnerPlanRef = null,
        slot_request_ids: [1]?u64 = .{456},
        block_runtime: BlockRuntimeMock = .{},
        device: MockCudaDevice = .{},

        pub fn uploadPipelineActivationFromHost(
            self: *@This(),
            slot_index: usize,
            host_buf: []const u8,
            byte_count: usize,
        ) !void {
            _ = slot_index;
            _ = host_buf;
            _ = byte_count;
            self.trace_state.stage1_upload_calls += 1;
        }

        pub fn executeDecodeWithLayerLimitTestHook(
            self: *@This(),
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
            _ = token;
            _ = position;
            _ = slot_index;
            _ = logits_out_opt;
            _ = layer_limit;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            _ = use_preloaded_input;
            self.trace_state.stage1_execute_calls += 1;
        }
    };

    var trace_state = TraceState{};
    var cpu_stage0 = CpuStage0Mock{ .trace_state = &trace_state };
    var staging: [d_model * @sizeOf(f32) + 8]u8 align(64) = undefined;
    var stage1 = Stage1Mock{
        .trace_state = &trace_state,
        .pipeline_host_staging = staging[0..],
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
    };
    try initMockCpuGpuLocalRunnerContracts(std.testing.allocator, &stage1);
    defer deinitMockCpuGpuLocalRunnerContracts(&stage1);

    try std.testing.expectError(
        error.PayloadBufferLengthMismatch,
        runCpuGpuWithPipelineRuntime(
            &stage1,
            &cpu_stage0,
            7,
            5,
            0,
            null,
            false,
            false,
            true,
            6,
            5,
            d_model * @sizeOf(f32) + 4,
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage0_execute_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage0_slot_bytes_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage1_upload_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage1_execute_calls);
}

test "runCpuGpuWithPipelineRuntime routes decode through LocalStageRunnerPlanRef" {
    const d_model: usize = 4;

    const MockCudaDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return 0;
        }
    };

    const TraceState = struct {
        stage0_execute_calls: usize = 0,
        stage0_slot_bytes_calls: usize = 0,
        stage1_upload_calls: usize = 0,
        stage1_execute_calls: usize = 0,
    };

    const CpuStage0Mock = struct {
        trace_state: *TraceState,
        activation: [d_model]f32 = [_]f32{0.0} ** d_model,

        pub fn executeDecodeLayerRange(
            self: *@This(),
            token: u32,
            position: usize,
            slot_index: usize,
            logits_out_opt: ?[]f32,
            layer_start: usize,
            layer_end: usize,
            compute_logits: bool,
            download_logits: bool,
            ensure_kv_capacity: bool,
            use_preloaded_input: bool,
        ) !void {
            _ = token;
            _ = position;
            _ = slot_index;
            _ = logits_out_opt;
            _ = layer_start;
            _ = layer_end;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = use_preloaded_input;
            self.trace_state.stage0_execute_calls += 1;
            @memset(self.activation[0..], 1.0);
        }

        pub fn slotActivationBytes(self: *@This(), _: usize) []const u8 {
            self.trace_state.stage0_slot_bytes_calls += 1;
            return std.mem.sliceAsBytes(self.activation[0..]);
        }

        pub fn slotActivationBytesMut(self: *@This(), _: usize) []u8 {
            return std.mem.sliceAsBytes(self.activation[0..]);
        }
    };

    const Stage1Mock = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        trace_state: *TraceState,
        split_layer: usize = 2,
        d_model: usize = d_model,
        pipeline_boundary_dtype: bridge.BoundaryDType = .f32,
        pipeline_boundary_layout: bridge.BoundaryLayout = .row_major,
        pipeline_host_staging: ?[]align(64) u8 = null,
        cpu_gpu_stage_plan: ?models.stage_plan.StagePlan = null,
        cpu_gpu_tensor_frame_plan_ref: ?bridge.TensorFramePlanRef = null,
        cpu_gpu_placement_plan: ?bridge.PlacementPlan = null,
        cpu_gpu_local_stage_runner_plan_ref: ?bridge.LocalStageRunnerPlanRef = null,
        slot_request_ids: [1]?u64 = .{456},
        block_runtime: BlockRuntimeMock = .{},
        device: MockCudaDevice = .{},

        pub fn uploadPipelineActivationFromHost(
            self: *@This(),
            slot_index: usize,
            host_buf: []const u8,
            byte_count: usize,
        ) !void {
            _ = slot_index;
            _ = host_buf;
            _ = byte_count;
            self.trace_state.stage1_upload_calls += 1;
        }

        pub fn executeDecodeWithLayerLimitTestHook(
            self: *@This(),
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
            _ = token;
            _ = position;
            _ = slot_index;
            _ = logits_out_opt;
            _ = layer_limit;
            _ = compute_logits;
            _ = download_logits;
            _ = ensure_kv_capacity;
            _ = trace_seq_len_u32;
            _ = trace_pos_offset;
            _ = hidden_override;
            _ = deepstack_layer_features_opt;
            _ = deepstack_feature_index_opt;
            _ = use_preloaded_input;
            self.trace_state.stage1_execute_calls += 1;
        }
    };

    var trace_state = TraceState{};
    var cpu_stage0 = CpuStage0Mock{ .trace_state = &trace_state };
    var staging: [d_model * @sizeOf(f32) + 8]u8 align(64) = undefined;
    var stage1 = Stage1Mock{
        .trace_state = &trace_state,
        .pipeline_host_staging = staging[0..],
        .block_runtime = .{ .blocks = &[_]u8{ 0, 0 } },
    };
    try initMockCpuGpuLocalRunnerContracts(std.testing.allocator, &stage1);
    defer deinitMockCpuGpuLocalRunnerContracts(&stage1);
    if (stage1.cpu_gpu_local_stage_runner_plan_ref) |*plan_ref| {
        plan_ref.plan_id.digest[0] ^= 1;
    }

    try std.testing.expectError(
        error.LocalStageRunnerPlanFingerprintMismatch,
        runCpuGpuWithPipelineRuntime(
            &stage1,
            &cpu_stage0,
            7,
            5,
            0,
            null,
            false,
            false,
            true,
            6,
            5,
            d_model * @sizeOf(f32),
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage0_execute_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage0_slot_bytes_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage1_upload_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage1_execute_calls);
}

fn initMockCpuGpuLocalRunnerContracts(allocator: std.mem.Allocator, backend: anytype) !void {
    const total_layers = backend.split_layer + backend.block_runtime.blocks.len;
    var manifest = try cpuGpuRouteTestManifest(allocator, total_layers, backend.d_model);
    defer manifest.deinit();
    var architecture = cpuGpuRouteTestArch();
    var config = cpuGpuRouteTestConfig(total_layers, backend.d_model);
    const split_points = [_]usize{backend.split_layer};

    var plan = try models.stage_plan.buildStagePlan(allocator, .{
        .n_layers = total_layers,
        .split_points = &split_points,
        .architecture = &architecture,
        .model_config = &config,
        .manifest = &manifest,
        .partition_constraints = .{ .decoder_cuts_allowed = true },
    });
    errdefer plan.deinit();

    var plan_ref = try bridge.TensorFramePlanRef.fromStagePlan(allocator, &plan);
    errdefer plan_ref.deinit();

    var placement = try buildMockCpuGpuPlacementPlan(allocator, backend, &plan);
    errdefer placement.deinit();

    var runner = try bridge.buildLocalStageRunnerPlanRef(allocator, .{
        .stage_plan = &plan,
        .tensor_frame_plan_ref = &plan_ref,
        .placement_plan = &placement,
    });
    errdefer runner.deinit();

    backend.cpu_gpu_stage_plan = plan;
    backend.cpu_gpu_tensor_frame_plan_ref = plan_ref;
    backend.cpu_gpu_placement_plan = placement;
    backend.cpu_gpu_local_stage_runner_plan_ref = runner;
}

fn deinitMockCpuGpuLocalRunnerContracts(backend: anytype) void {
    if (backend.cpu_gpu_local_stage_runner_plan_ref) |*plan_ref| {
        plan_ref.deinit();
        backend.cpu_gpu_local_stage_runner_plan_ref = null;
    }
    if (backend.cpu_gpu_placement_plan) |*placement| {
        placement.deinit();
        backend.cpu_gpu_placement_plan = null;
    }
    if (backend.cpu_gpu_tensor_frame_plan_ref) |*plan_ref| {
        plan_ref.deinit();
        backend.cpu_gpu_tensor_frame_plan_ref = null;
    }
    if (backend.cpu_gpu_stage_plan) |*plan| {
        plan.deinit();
        backend.cpu_gpu_stage_plan = null;
    }
}

fn buildMockCpuGpuPlacementPlan(
    allocator: std.mem.Allocator,
    backend: anytype,
    plan: *const models.stage_plan.StagePlan,
) !bridge.PlacementPlan {
    if (plan.stages.len != 2 or plan.boundaries.len != 1) return error.InvalidTopologyConfig;
    const element_bytes: usize = switch (backend.pipeline_boundary_dtype) {
        .bf16, .f16 => @sizeOf(u16),
        .f32 => @sizeOf(f32),
    };
    const activation_bytes = std.math.mul(usize, backend.d_model, element_bytes) catch return error.InvalidArgument;
    const activation_bytes_u64 = std.math.cast(u64, activation_bytes) orelse return error.InvalidArgument;
    const cpu_host_id = bridge.HostId{ .value = 1 };
    const cuda_host_id = bridge.HostId{ .value = 2 };

    const cpu_frames = [_]bridge.HostFrameCapability{.{
        .endpoint_role = .producer,
        .step_kind = .decode,
        .dtype = backend.pipeline_boundary_dtype,
        .layout = backend.pipeline_boundary_layout,
        .handoff_mode = .local_in_process,
        .max_batch_entries = 1,
        .max_token_count_per_frame = 1,
        .max_activation_payload_bytes = activation_bytes_u64,
    }};
    const cuda_frames = [_]bridge.HostFrameCapability{.{
        .endpoint_role = .consumer,
        .step_kind = .decode,
        .dtype = backend.pipeline_boundary_dtype,
        .layout = backend.pipeline_boundary_layout,
        .handoff_mode = .local_in_process,
        .max_batch_entries = 1,
        .max_token_count_per_frame = 1,
        .max_activation_payload_bytes = activation_bytes_u64,
    }};
    var cpu_capability = try bridge.buildHostCapability(allocator, .{
        .host_id = cpu_host_id,
        .backend_kind = .cpu,
        .reachability_kind = .local_in_process,
        .supported_graph_contract_versions = &.{plan.graph_identity.graph_contract_version},
        .supported_stage_plan_contract_versions = &.{plan.stage_contract_version},
        .frame_capabilities = &cpu_frames,
    });
    defer cpu_capability.deinit();
    var cuda_capability = try bridge.buildHostCapability(allocator, .{
        .host_id = cuda_host_id,
        .backend_kind = .cuda,
        .reachability_kind = .local_in_process,
        .supported_graph_contract_versions = &.{plan.graph_identity.graph_contract_version},
        .supported_stage_plan_contract_versions = &.{plan.stage_contract_version},
        .frame_capabilities = &cuda_frames,
    });
    defer cuda_capability.deinit();

    const cpu_resident = [_]bridge.ResidentStageEntry{mockResidentEntryFromStage(plan.stages[0])};
    const cuda_resident = [_]bridge.ResidentStageEntry{mockResidentEntryFromStage(plan.stages[1])};
    var cpu_residency = try bridge.buildHostResidencySnapshot(allocator, .{
        .host_id = cpu_host_id,
        .plan = plan,
        .resident_stages = &cpu_resident,
    });
    defer cpu_residency.deinit();
    var cuda_residency = try bridge.buildHostResidencySnapshot(allocator, .{
        .host_id = cuda_host_id,
        .plan = plan,
        .resident_stages = &cuda_resident,
    });
    defer cuda_residency.deinit();

    const boundary = plan.boundaries[0];
    const bindings = [_]bridge.StageHostBinding{
        .{ .stage_id = boundary.source_stage_id, .host_id = cpu_host_id },
        .{ .stage_id = boundary.target_stage_id, .host_id = cuda_host_id },
    };
    const profiles = [_]bridge.BoundaryFrameProfile{.{
        .boundary_index = 0,
        .source_stage_id = boundary.source_stage_id,
        .target_stage_id = boundary.target_stage_id,
        .step_kind = .decode,
        .dtype = backend.pipeline_boundary_dtype,
        .layout = backend.pipeline_boundary_layout,
        .max_batch_entries = 1,
        .max_token_count_per_frame = 1,
        .max_activation_payload_bytes = activation_bytes_u64,
        .handoff_mode = .local_in_process,
    }};

    return bridge.buildPlacementPlan(allocator, .{
        .plan = plan,
        .required_step_kinds = &.{.decode},
        .host_capabilities = &.{ cpu_capability, cuda_capability },
        .host_residency_snapshots = &.{ cpu_residency, cuda_residency },
        .stage_host_bindings = &bindings,
        .boundary_frame_profiles = &profiles,
        .allowed_reachability = &.{.local_in_process},
    });
}

fn mockResidentEntryFromStage(stage: models.stage_plan.StagePlanStage) bridge.ResidentStageEntry {
    return .{
        .stage_id = stage.id,
        .layer_start = stage.layer_start,
        .layer_end = stage.layer_end,
        .owned_roles = stage.owned_roles,
        .residency = stage.residency,
    };
}

fn cpuGpuRouteTestConfig(layer_count: usize, d_model: usize) models.config.ModelConfig {
    return .{
        .vocab_size = 64,
        .d_model = @intCast(d_model),
        .n_layers = @intCast(layer_count),
        .n_heads = 2,
        .n_kv_groups = 2,
        .d_ff = @intCast(d_model * 4),
        .max_seq_len = 32,
        .head_dim = @intCast(d_model / 2),
        .rope_theta = 10000,
        .norm_eps = 0.00001,
        .gaffine_group_size = 0,
        .tie_word_embeddings = false,
    };
}

fn cpuGpuRouteTestArch() models.op_types.Architecture {
    return .{
        .name = "cpu_gpu_route_local_runner_test",
        .model_types = &.{"cpu_gpu_route_local_runner_test"},
    };
}

fn cpuGpuRouteTestManifest(
    allocator: std.mem.Allocator,
    layer_count: usize,
    d_model: usize,
) !models.manifest.ModelManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_allocator = arena.allocator();
    const entry_count = layer_count + 3;
    const entries = try arena_allocator.alloc(models.manifest.TensorManifestEntry, entry_count);
    const embed_shape = try arena_allocator.dupe(usize, &[_]usize{ 64, d_model });
    const layer_shape = try arena_allocator.dupe(usize, &[_]usize{ d_model, d_model });
    const norm_shape = try arena_allocator.dupe(usize, &[_]usize{d_model});
    const lm_head_shape = try arena_allocator.dupe(usize, &[_]usize{ 64, d_model });
    const embed_bytes = 64 * d_model * @sizeOf(f32);
    const layer_bytes = d_model * d_model * @sizeOf(f32);
    const norm_bytes = d_model * @sizeOf(f32);

    entries[0] = .{
        .name = "model.embed_tokens.weight",
        .dtype = .f32,
        .shape = embed_shape,
        .checkpoint_bytes = embed_bytes,
        .role = .token_embeddings,
        .weight_id = "token_embeddings",
        .status = .architecture_weight,
    };
    for (0..layer_count) |layer_index| {
        entries[layer_index + 1] = .{
            .name = "model.layers.self_attn.q_proj.weight",
            .dtype = .f32,
            .shape = layer_shape,
            .checkpoint_bytes = layer_bytes,
            .role = .decoder_layer,
            .layer_index = layer_index,
            .weight_id = "self_attn.q_proj.weight",
            .status = .architecture_weight,
        };
    }
    entries[layer_count + 1] = .{
        .name = "model.norm.weight",
        .dtype = .f32,
        .shape = norm_shape,
        .checkpoint_bytes = norm_bytes,
        .role = .final_norm,
        .weight_id = "ln_final",
        .status = .architecture_weight,
    };
    entries[layer_count + 2] = .{
        .name = "lm_head.weight",
        .dtype = .f32,
        .shape = lm_head_shape,
        .checkpoint_bytes = embed_bytes,
        .role = .lm_head,
        .weight_id = "lm_head",
        .status = .architecture_weight,
    };

    var role_bytes = [_]usize{0} ** models.manifest.role_count;
    var total_bytes: usize = 0;
    for (entries) |entry| {
        total_bytes += entry.checkpoint_bytes;
        role_bytes[@intFromEnum(entry.role)] += entry.checkpoint_bytes;
    }

    return .{
        .arena = arena,
        .architecture_id = "cpu_gpu_route_local_runner_test",
        .layer_count = layer_count,
        .entries = entries,
        .total_checkpoint_bytes = total_bytes,
        .role_bytes = role_bytes,
    };
}

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

fn u32FromUsize(value: usize, comptime err: anyerror) !u32 {
    return std.math.cast(u32, value) orelse return err;
}

fn cpuGpuFrameId(slot_index: u32, position: u32) u64 {
    return (@as(u64, slot_index) << 32) | @as(u64, position);
}

fn cpuGpuGraphId(split_layer: u32, total_layers: u32) u64 {
    return (@as(u64, split_layer) << 32) | @as(u64, total_layers);
}

fn cpuGpuRequestId(slot_index: u32, token: u32) u64 {
    return (@as(u64, slot_index) << 32) | @as(u64, token);
}

fn cpuGpuActivationDevice(self: anytype) !bridge.TensorFrameDevice {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidTopologyConfig;
    const ordinal = std.math.cast(u16, self.device.ordinal()) orelse return error.InvalidTopologyConfig;
    return .{ .cuda = ordinal };
}

fn buildCpuGpuActivationFrame(
    self: anytype,
    ctx: *const stage_adapters.DecodeContext,
) !bridge.TensorFrameMetadata {
    const slot_index_u32 = try u32FromUsize(ctx.slot_index, error.InvalidArgument);
    const position_u32 = try u32FromUsize(ctx.position, error.InvalidSequenceRange);
    const split_layer_u32 = try u32FromUsize(self.split_layer, error.InvalidLayerRange);
    const total_layers = self.split_layer + self.block_runtime.blocks.len;
    const total_layers_u32 = try u32FromUsize(total_layers, error.InvalidLayerRange);
    const d_model_u64 = std.math.cast(u64, self.d_model) orelse return error.InvalidTensorShape;
    const shape = try bridge.TensorFrameShape.contiguous(3, .{ 1, 1, d_model_u64, 0 });

    return bridge.activationHandoffFrame(.{
        .frame_id = cpuGpuFrameId(slot_index_u32, position_u32),
        .graph_id = cpuGpuGraphId(split_layer_u32, total_layers_u32),
        .request_id = cpuGpuRequestId(slot_index_u32, ctx.token),
        .source = .{ .stage_id = 0, .backend = .cpu },
        .target = .{ .stage_id = 1, .backend = .cuda },
        .producer_layer_start = 0,
        .producer_layer_end = split_layer_u32,
        .consumer_layer_start = split_layer_u32,
        .consumer_layer_end = total_layers_u32,
        .dtype = self.pipeline_boundary_dtype,
        .layout = self.pipeline_boundary_layout,
        .shape = shape,
        .device = try cpuGpuActivationDevice(self),
        .sequence_start = position_u32,
        .sequence_len = 1,
        .batch_size = 1,
        .slot_index = slot_index_u32,
        .ownership = .borrowed_until_next_stage_call,
        .lifetime = .step_scoped,
    });
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

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
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

        pub fn deinit(_: *@This(), _: std.mem.Allocator) void {}
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
    const activation_metadata = try buildCpuGpuActivationFrame(self, &ctx);
    try bridge.executeLocalDecodeHandoff(
        Stage0,
        Stage1,
        .{ .backend = cpu_stage0_backend, .gpu_backend = self, .ctx = &ctx },
        .{ .backend = self, .ctx = &ctx },
        .{
            .metadata = activation_metadata,
            .activation_byte_count = activation_byte_count,
            .host_staging = self.pipeline_host_staging,
        },
    );
}

test "buildCpuGpuActivationFrame describes cpu cuda decode handoff" {
    const MockCudaDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return 7;
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        split_layer: usize = 3,
        d_model: usize = 8,
        pipeline_boundary_dtype: bridge.BoundaryDType = .f32,
        pipeline_boundary_layout: bridge.BoundaryLayout = .row_major,
        block_runtime: BlockRuntimeMock = .{},
        device: MockCudaDevice = .{},
    };

    var backend = MockBackend{
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

    const metadata = try buildCpuGpuActivationFrame(&backend, &ctx);
    try std.testing.expectEqual(bridge.TensorFrameRole.activation, metadata.role);
    try std.testing.expectEqual(@as(u64, 32), metadata.byte_count);
    try std.testing.expectEqual(@as(u64, 0x0000_0002_0000_0009), metadata.frame_id);
    try std.testing.expectEqual(@as(u64, 0x0000_0003_0000_0005), metadata.graph_id);
    try std.testing.expectEqual(@as(u64, 0x0000_0002_0000_007b), metadata.request_id);
    try std.testing.expectEqual(bridge.TensorFrameDevice{ .cuda = 7 }, metadata.device);
    try std.testing.expectEqual(@as(u32, 9), metadata.sequence_start);
    try std.testing.expectEqual(@as(u32, 1), metadata.sequence_len);
    try std.testing.expectEqual(@as(?u32, 2), metadata.slot_index);
}

test "buildCpuGpuActivationFrame rejects cuda ordinal outside tensor frame range" {
    const MockCudaDevice = struct {
        pub fn ordinal(_: *const @This()) usize {
            return @as(usize, std.math.maxInt(u16)) + 1;
        }
    };

    const MockBackend = struct {
        const BlockRuntimeMock = struct {
            blocks: []const u8 = &.{},
        };

        split_layer: usize = 3,
        d_model: usize = 8,
        pipeline_boundary_dtype: bridge.BoundaryDType = .f32,
        pipeline_boundary_layout: bridge.BoundaryLayout = .row_major,
        block_runtime: BlockRuntimeMock = .{},
        device: MockCudaDevice = .{},
    };

    var backend = MockBackend{
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

    try std.testing.expectError(error.InvalidTopologyConfig, buildCpuGpuActivationFrame(&backend, &ctx));
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

    try std.testing.expectError(
        error.InvalidTensorByteCount,
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
    try std.testing.expectEqual(@as(usize, 1), trace_state.stage0_execute_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage0_slot_bytes_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage1_upload_calls);
    try std.testing.expectEqual(@as(usize, 0), trace_state.stage1_execute_calls);
}

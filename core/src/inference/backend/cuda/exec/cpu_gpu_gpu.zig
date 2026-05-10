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
const transferPipelineActivationStage12MultiRow = transfers.transferPipelineActivationStage12MultiRow;
const dumpHiddenState = common.dumpHiddenState;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const applyHostLogitsPostProcess = common.applyHostLogitsPostProcess;
const ensureKvCapacity = kv_capacity.ensureKvCapacity;
const resetShortConvStates = resets.resetShortConvStates;
const resetGatedDeltaStates = resets.resetGatedDeltaStates;
const resetAttentionCpuStates = resets.resetAttentionCpuStates;
const ensureGatedDeltaHostStageCapacity = resets.ensureGatedDeltaHostStageCapacity;
const computeGpuPrototypeLogitsWithLayerLimit = decode_route.computeGpuPrototypeLogitsWithLayerLimit;

pub fn computeBatchedPrefillCpuGpuGpu(
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
    const staged_chunk_cap_base = @min(self.prefill_chunk_rows_cap, gpu_stage1.prefill_chunk_rows_cap);
    const chunk_cap = resolveStagedPrefillChunkRows(
        total_rows,
        staged_chunk_cap_base,
        @import("env_pkg").getenv("TALU_CUDA_PREFILL_CHUNK_ROWS") != null,
    );

    // ── CPU stage0: batched embed + forward through [0, split_layer) ──
    const prefill_buffer = try self.allocator.alloc(f32, total_rows * d_model);
    defer self.allocator.free(prefill_buffer);

    // For per-layer branch-input: capture source embeddings from CPU.
    const per_layer_branch_active_1 = per_layer_branch_feature.hasPerLayerBranchRuntime(gpu_stage1);
    const per_layer_branch_active_2 = per_layer_branch_feature.hasPerLayerBranchRuntime(self);
    const has_per_layer_branch_1 = per_layer_branch_active_1 and gpu_stage1.per_layer_branch_runtime != null;
    const has_per_layer_branch_2 = per_layer_branch_active_2 and self.per_layer_branch_runtime != null;
    const need_source_embeddings = has_per_layer_branch_1 or has_per_layer_branch_2;
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

        // Upload source embeddings for GPU1 per-layer branch branch.
        var per_layer_source_embeddings_1: ?compute.cuda.Buffer = null;
        if (has_per_layer_branch_1) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&gpu_stage1.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&gpu_stage1.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                per_layer_source_embeddings_1 = se_dst;
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
                if (per_layer_branch_active_1) {
                    if (per_layer_source_embeddings_1) |*per_layer_se| {
                        const chunk_tokens = tokens[pos_base .. pos_base + rows];
                        try per_layer_branch_feature.applyPerLayerBranch(
                            gpu_stage1,
                            layer_idx,
                            chunk_tokens,
                            per_layer_se,
                            &gpu_stage1.runtime_buffers.input_dev,
                        );
                    } else if (per_layer_branch_feature.hasStandaloneLayerScalars(gpu_stage1)) {
                        try per_layer_branch_feature.applyStandaloneLayerScalar(gpu_stage1, layer_idx, &gpu_stage1.runtime_buffers.input_dev, rows);
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

        // Upload source embeddings for GPU2 per-layer branch branch.
        var per_layer_source_embeddings_2: ?compute.cuda.Buffer = null;
        if (has_per_layer_branch_2) {
            if (source_embeddings_host) |se_host| {
                const se_chunk_offset = pos_base * d_model;
                const se_chunk_f32s = rows * d_model;
                const se_bytes = std.math.mul(usize, se_chunk_f32s, @sizeOf(f32)) catch return error.InvalidArgument;
                var se_dst = try bufferSlice(&self.runtime_buffers.deepstack_add_dev, 0, se_bytes);
                try se_dst.upload(&self.device, std.mem.sliceAsBytes(se_host[se_chunk_offset..][0..se_chunk_f32s]));
                per_layer_source_embeddings_2 = se_dst;
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
                if (per_layer_branch_active_2) {
                    if (per_layer_source_embeddings_2) |*per_layer_se| {
                        const chunk_tokens = tokens[pos_base .. pos_base + rows];
                        try per_layer_branch_feature.applyPerLayerBranch(
                            self,
                            layer_idx,
                            chunk_tokens,
                            per_layer_se,
                            &self.runtime_buffers.input_dev,
                        );
                        final_hidden_rows = self.runtime_buffers.input_dev;
                    } else if (per_layer_branch_feature.hasStandaloneLayerScalars(self)) {
                        try per_layer_branch_feature.applyStandaloneLayerScalar(self, layer_idx, &self.runtime_buffers.input_dev, rows);
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

pub fn runCpuGpuGpuWithPipelineRuntime(
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

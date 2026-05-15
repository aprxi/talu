//! Bridge-driven local staged execution for CUDA-backed local topologies.
//!
//! CPU->CUDA, CUDA->CUDA, and CPU->CUDA->CUDA are placement instances of this local stage-chain route.

const std = @import("std");
const compute = @import("compute_pkg");
const trace = @import("xray_pkg").trace;
const bridge = @import("../../../bridge/root.zig");
const transport = @import("../../../transport/root.zig");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

const engine_types = @import("../runtime/root.zig");
const KvCacheDtype = engine_types.KvCacheDtype;
const enable_device_embedding_lookup = engine_types.enable_device_embedding_lookup;
const AttentionKernelSet = engine_types.AttentionKernelSet;

const engine_ops = @import("../operators/root.zig");

const engine_weights = @import("../weights/root.zig");
const bufferSlice = engine_weights.bufferSlice;
const populatePrefillHiddenFromTokens = engine_weights.populatePrefillHiddenFromTokens;

const saturatingU64FromU128 = engine_types.saturatingU64FromU128;

const common = @import("common.zig");
const kv_capacity = @import("kv_capacity.zig");
const resets = @import("resets.zig");
const stage_adapters = @import("stage_adapters.zig");
const resolveStagedPrefillChunkRows = @import("prefill_route.zig").resolveStagedPrefillChunkRows;
const uploadCpuKvToMirrors = transport.uploadCpuKvToCudaMirrors;
const dumpHiddenState = common.dumpHiddenState;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const applyHostLogitsPostProcess = common.applyHostLogitsPostProcess;
const ensureKvCapacity = kv_capacity.ensureKvCapacity;
const resetShortConvStates = resets.resetShortConvStates;
const resetGatedDeltaStates = resets.resetGatedDeltaStates;
const resetAttentionCpuStates = resets.resetAttentionCpuStates;

fn executeLocalPrefillBoundaryChain(
    root_backend: anytype,
    allocator: ?std.mem.Allocator,
    comptime step_kind: bridge.TensorFrameStepKind,
    comptime Source: type,
    comptime Target: type,
    source: *Source,
    target: *Target,
    placement_plan: *const bridge.PlacementPlan,
    state_ownership_plan: ?*const bridge.StageStateOwnershipPlan,
    metadata: *const bridge.TensorFrameMetadata,
    image: *const bridge.BoundaryByteImageRef,
    staging: ?[]align(64) u8,
    allow_borrow: bool,
    local_peer_copy: bool,
) !void {
    var stages = [_]bridge.LocalStageChainStage{
        bridge.localStageAdapter(Source, metadata.boundary.source_stage_id, source),
        bridge.localStageAdapter(Target, metadata.boundary.target_stage_id, target),
    };
    const boundaries = [_]bridge.LocalStageChainBoundaryStep{.{
        .boundary_index = metadata.boundary.boundary_index,
        .step_kind = step_kind,
        .metadata = metadata,
        .image = image,
        .staging = staging,
        .allow_borrow = allow_borrow,
        .local_device_peer_copy_available = local_peer_copy,
    }};
    try bridge.executeLocalStageChain(.{
        .allocator = allocator,
        .plan_ref = try stage_adapters.localTopologyRunnerPlanRef(root_backend),
        .placement_plan = placement_plan,
        .state_ownership_plan = state_ownership_plan,
        .stages = stages[0..],
        .boundaries = boundaries[0..],
    });
}

pub fn executeLocalPrefillCpuCuda(
    self: anytype,
    cpu_stage0: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    const PrefillSource = transport.NoopActivationStage;
    const PrefillTarget = transport.CudaActivationStage(@TypeOf(self));

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

    try cpu_stage0.prefillSlotLayerRange(slot_index, tokens, prefill_buffer, stage_adapters.localLayerOffset(self), source_embeddings_host);

    // Upload CPU source layer KV to GPU mirror buffers for cross-device sharing.
    try uploadCpuKvToMirrors(self, cpu_stage0, slot_index, 0, total_rows);

    const placement_plan = try stage_adapters.localTopologyPlacementPlan(self);
    const state_ownership_plan = stage_adapters.localTopologyStateOwnershipPlan(self);
    const allocator = stage_adapters.backendAllocator(self);
    var prefill_source = PrefillSource{};
    var prefill_target = PrefillTarget{ .backend = self };
    const boundary0 = try stage_adapters.localBoundaryRuntime(self, 0);

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
        const activation_metadata = try stage_adapters.buildPrefillActivationMetadata(
            self,
            boundary0.boundary_index,
            boundary0.dtype,
            boundary0.layout,
            .{ .cpu = {} },
            slot_index,
            pos_base,
            rows,
            batch_entries[0..],
        );
        try bridge.validateTensorFrameForPlanBoundary(&activation_metadata, try stage_adapters.localTopologyTensorFramePlanRef(self), boundary0.boundary_index);
        try bridge.validatePayloadBufferLength(&activation_metadata, chunk_bytes.len);
        const image = bridge.hostActivationByteImage(&activation_metadata, chunk_bytes);
        try executeLocalPrefillBoundaryChain(
            self,
            allocator,
            .prefill,
            PrefillSource,
            PrefillTarget,
            &prefill_source,
            &prefill_target,
            placement_plan,
            state_ownership_plan,
            &activation_metadata,
            &image,
            boundary0.staging,
            false,
            boundary0.local_device_peer_copy_available,
        );

        const active_rows_u32: u32 = @intCast(rows);
        const seq_len_u32: u32 = @intCast(pos_base + rows);
        const last_position = pos_base + rows - 1;
        const last_position_u32: u32 = @intCast(last_position);

        var final_hidden_rows = self.runtime_buffers.input_dev;
        var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (source_embeddings_host) |se_host| {
            // local-stage mode: upload source embeddings from CPU to deepstack_add_dev.
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
            dumpHiddenState(self, &self.runtime_buffers.input_dev, stage_adapters.localLayerOffset(self) + layer_idx, "post_layer", self.d_model, 1);
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
                    dumpHiddenState(self, &self.runtime_buffers.input_dev, stage_adapters.localLayerOffset(self) + layer_idx, "post_ple", self.d_model, 1);
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

pub fn executeLocalPrefillCudaCuda(
    self: anytype,
    stage1: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    if (comptime !@hasField(@TypeOf(stage1.*), "device")) return error.InvalidArgument;
    const PrefillSource = transport.CudaPeerActivationStage(@TypeOf(self), @TypeOf(stage1), .source_event_target_stream);
    const PrefillTarget = transport.CudaActivationStage(@TypeOf(stage1));

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

    const placement_plan = try stage_adapters.localTopologyPlacementPlan(self);
    const state_ownership_plan = stage_adapters.localTopologyStateOwnershipPlan(self);
    const allocator = stage_adapters.backendAllocator(self);
    var prefill_source = PrefillSource{ .backend = self, .target_backend = stage1 };
    var prefill_target = PrefillTarget{ .backend = stage1 };
    const boundary0 = try stage_adapters.localBoundaryRuntime(self, 0);

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
        var batch_entries: [1]bridge.TensorFrameBatchEntry = undefined;
        const activation_metadata = try stage_adapters.buildPrefillActivationMetadata(
            self,
            boundary0.boundary_index,
            boundary0.dtype,
            boundary0.layout,
            try stage_adapters.cudaPayloadLocationHint(self),
            slot_index,
            pos_base,
            rows,
            batch_entries[0..],
        );
        try bridge.validateTensorFrameForPlanBoundary(&activation_metadata, try stage_adapters.localTopologyTensorFramePlanRef(self), boundary0.boundary_index);
        try bridge.validatePayloadBufferLength(&activation_metadata, transfer_bytes);
        const image = bridge.deviceActivationByteImage(&activation_metadata);
        try executeLocalPrefillBoundaryChain(
            self,
            allocator,
            .prefill,
            PrefillSource,
            PrefillTarget,
            &prefill_source,
            &prefill_target,
            placement_plan,
            state_ownership_plan,
            &activation_metadata,
            &image,
            boundary0.staging,
            false,
            boundary0.local_device_peer_copy_available,
        );

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

pub fn executeLocalPrefillCpuCudaCuda(
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
    const CpuPrefillSource = transport.NoopActivationStage;
    const GpuPrefillSource = transport.CudaPeerActivationStage(@TypeOf(gpu_stage1), @TypeOf(self), .source_stream);
    const GpuPrefillTarget = transport.CudaActivationStage(@TypeOf(gpu_stage1));
    const FinalPrefillTarget = transport.CudaActivationStage(@TypeOf(self));

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

    try cpu_stage0.prefillSlotLayerRange(slot_index, tokens, prefill_buffer, stage_adapters.localLayerOffset(self), source_embeddings_host);

    const placement_plan = try stage_adapters.localTopologyPlacementPlan(self);
    const state_ownership_plan = stage_adapters.localTopologyStateOwnershipPlan(self);
    const allocator = stage_adapters.backendAllocator(self);
    const plan_ref = try stage_adapters.localTopologyTensorFramePlanRef(self);
    var cpu_prefill_source = CpuPrefillSource{};
    var gpu_prefill_source = GpuPrefillSource{ .backend = gpu_stage1, .target_backend = self };
    var gpu_prefill_target = GpuPrefillTarget{ .backend = gpu_stage1 };
    var final_prefill_target = FinalPrefillTarget{ .backend = self };
    const boundary0 = try stage_adapters.localBoundaryRuntime(self, 0);
    const boundary1 = try stage_adapters.localBoundaryRuntime(self, 1);

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
        const chunk_bytes = std.mem.sliceAsBytes(prefill_buffer[chunk_offset..][0..chunk_f32s]);
        var batch_entries01: [1]bridge.TensorFrameBatchEntry = undefined;
        const metadata01 = try stage_adapters.buildPrefillActivationMetadata(
            self,
            boundary0.boundary_index,
            boundary0.dtype,
            boundary0.layout,
            .{ .cpu = {} },
            slot_index,
            pos_base,
            rows,
            batch_entries01[0..],
        );
        try bridge.validateTensorFrameForPlanBoundary(&metadata01, plan_ref, boundary0.boundary_index);
        try bridge.validatePayloadBufferLength(&metadata01, chunk_bytes.len);
        const image01 = bridge.hostActivationByteImage(&metadata01, chunk_bytes);
        try executeLocalPrefillBoundaryChain(
            self,
            allocator,
            .prefill,
            CpuPrefillSource,
            GpuPrefillTarget,
            &cpu_prefill_source,
            &gpu_prefill_target,
            placement_plan,
            state_ownership_plan,
            &metadata01,
            &image01,
            boundary0.staging,
            false,
            boundary0.local_device_peer_copy_available,
        );

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
        var batch_entries12: [1]bridge.TensorFrameBatchEntry = undefined;
        const metadata12 = try stage_adapters.buildPrefillActivationMetadata(
            self,
            boundary1.boundary_index,
            boundary1.dtype,
            boundary1.layout,
            try stage_adapters.cudaPayloadLocationHint(gpu_stage1),
            slot_index,
            pos_base,
            rows,
            batch_entries12[0..],
        );
        try bridge.validateTensorFrameForPlanBoundary(&metadata12, plan_ref, boundary1.boundary_index);
        try bridge.validatePayloadBufferLength(&metadata12, transfer_bytes);
        const image12 = bridge.deviceActivationByteImage(&metadata12);
        try executeLocalPrefillBoundaryChain(
            self,
            allocator,
            .prefill,
            GpuPrefillSource,
            FinalPrefillTarget,
            &gpu_prefill_source,
            &final_prefill_target,
            placement_plan,
            state_ownership_plan,
            &metadata12,
            &image12,
            boundary1.staging,
            false,
            boundary1.local_device_peer_copy_available,
        );

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

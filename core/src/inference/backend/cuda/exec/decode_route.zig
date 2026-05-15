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

fn rejectUnsupportedStagedBatchedDecodeRoute(topology_tag: ?[]const u8) !void {
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
const prefill_route = @import("prefill_route.zig");
const kv_capacity = @import("kv_capacity.zig");
const resets = @import("resets.zig");
const stage_adapters = @import("stage_adapters.zig");
const resolveStagedPrefillChunkRows = prefill_route.resolveStagedPrefillChunkRows;
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const dumpHiddenState = common.dumpHiddenState;
const applyHostLogitsPostProcess = common.applyHostLogitsPostProcess;
const executeCpuStage0LayerRange = common.executeCpuStage0LayerRange;
const localActivationByteCountFor = common.localActivationByteCountFor;
const logDecodeInventoryOnce = common.logDecodeInventoryOnce;
const ensureKvCapacity = kv_capacity.ensureKvCapacity;
const resetShortConvStates = resets.resetShortConvStates;
const resetGatedDeltaStates = resets.resetGatedDeltaStates;
const resetAttentionCpuStates = resets.resetAttentionCpuStates;
const ensureGatedDeltaHostStageCapacity = resets.ensureGatedDeltaHostStageCapacity;

const DecodeBoundaryStageSide = enum {
    source,
    target,
};

pub fn executeDecodeWithLayerLimit(
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
    const local_pipeline_kind = if (layer_limit == self.block_runtime.blocks.len and compute_logits and !use_preloaded_input)
        try stage_adapters.localPipelinePlacementKind(self)
    else
        null;
    if (local_pipeline_kind) |placement_kind| {
        switch (placement_kind) {
            .cuda_cuda, .cpu_cuda, .cpu_cuda_cuda => {
                try executeSingleTokenLocalDecodePipeline(
                    self,
                    placement_kind,
                    token,
                    position,
                    slot_index,
                    logits_out_opt,
                    compute_logits,
                    download_logits,
                    ensure_kv_capacity,
                    trace_seq_len_u32,
                    trace_pos_offset,
                    hidden_override,
                    deepstack_layer_features_opt,
                    deepstack_feature_index_opt,
                );
                return;
            },
            .generic_local_chain => return error.InvalidTopologyConfig,
        }
    }
    if (comptime @hasDecl(SelfType, "executeDecodeWithLayerLimitTestHook")) {
        return self.executeDecodeWithLayerLimitTestHook(
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
        ensureKvCapacity(self, position + 1) catch |err| {
            log.warn("inference", "CUDA ensureKvCapacity failed in layer-range decode", .{
                .slot = slot_index,
                .position = position,
                .required_tokens = position + 1,
                .max_seq = self.max_seq_len,
                .kv_dtype = @tagName(self.kv_cache_dtype),
                .reason = @errorName(err),
            });
            return err;
        };
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

    var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
    if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
        if (use_preloaded_input) {
            // local-stage mode: input_dev has post-CPU-layer hidden states, not raw
            // embeddings. Look up the raw embedding on host and upload.
            if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
                per_layer_source_embeddings_opt = per_layer_branch_feature.capturePerLayerSourceEmbeddingsForLocalStage(self, token) catch |err| blk: {
                    log.warn("inference", "CUDA capturePerLayerSourceEmbeddingsForLocalStage failed", .{
                        .reason = @errorName(err),
                        .token = token,
                    });
                    break :blk null;
                };
            }
        } else {
            per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, 1);
        }
    }
    const gemma_token_id_single = [_]u32{token};

    // local-stage boundary: input_dev was populated by a host→device copy on
    // the default stream (cu_memcpy_htod). Synchronize compute_stream so kernels
    // on the named stream see the uploaded data before executing.
    if (use_preloaded_input) {
        if (self.compute_stream) |stream| {
            try self.device.synchronizeStream(stream);
        }
    }

    // Build BatchDecodeInfo for the single-token decode step. This routes
    // attention through the batched decode path (batched separate or flash
    // decode), which is the canonical single-token route.
    const slot_indices_single = [_]usize{slot_index};
    const positions_single = [_]usize{position};
    self.runtime_buffers.decode_seq_lens_host[0] = seq_len_u32;
    self.runtime_buffers.decode_positions_host[0] = position_u32;
    var proto_seq_lens_dev = try bufferSlice(&self.runtime_buffers.decode_seq_lens_dev, 0, @sizeOf(u32));
    var proto_positions_dev = try bufferSlice(&self.runtime_buffers.decode_positions_dev, 0, @sizeOf(u32));
    try proto_seq_lens_dev.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.decode_seq_lens_host[0..1]));
    try proto_positions_dev.upload(&self.device, std.mem.sliceAsBytes(self.runtime_buffers.decode_positions_host[0..1]));

    var slot_set_matches_cache = false;
    if (comptime @hasField(SelfType, "decode_ptr_tables_cached_rows") and @hasField(SelfType, "decode_ptr_tables_cached_slots")) {
        slot_set_matches_cache = self.decode_ptr_tables_cached_rows == 1 and
            self.decode_ptr_tables_cached_slots.len > 0 and
            self.decode_ptr_tables_cached_slots[0] == slot_index;
    }
    const pointer_tables_dirty = if (comptime @hasField(SelfType, "decode_ptr_tables_dirty"))
        self.decode_ptr_tables_dirty
    else
        true;
    const refresh_pointer_tables = pointer_tables_dirty or !slot_set_matches_cache;

    // Set batched attention max seq_len tier (power-of-2 ceiling).
    const model_max_u32: u32 = @intCast(self.max_seq_len);
    self.runtime_buffers.batched_attn_max_seq_len = @min(
        std.math.ceilPowerOfTwo(u32, seq_len_u32) catch model_max_u32,
        model_max_u32,
    );

    // Populate attention pointer tables from block_runtime (authoritative
    // after ensureKvCapacity which may have grown KV buffers).
    const attn_layers = self.block_runtime.attention_block_count;
    if (attn_layers > 0 and refresh_pointer_tables) {
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
    if (gd_layers > 0 and refresh_pointer_tables) {
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

    if (refresh_pointer_tables and comptime @hasField(SelfType, "decode_ptr_tables_cached_rows") and @hasField(SelfType, "decode_ptr_tables_cached_slots")) {
        if (self.decode_ptr_tables_cached_slots.len > 0) {
            self.decode_ptr_tables_cached_slots[0] = slot_index;
            self.decode_ptr_tables_cached_rows = 1;
        } else {
            self.decode_ptr_tables_cached_rows = 0;
        }
    }
    if (refresh_pointer_tables and comptime @hasField(SelfType, "decode_ptr_tables_dirty")) {
        self.decode_ptr_tables_dirty = false;
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

    // CUDA graph capture: record all GPU kernels (layer loop + final norm + lm_head)
    // into a graph, then replay with near-zero scheduling gaps between kernels.
    // Graph capture eliminates per-kernel launch overhead by recording the layer
    // sequence and replaying it as a single launch.  Re-captures every decode step
    // to pick up changed arguments (position, seq_len); graphExecUpdate patches
    // the existing exec in-place when topology matches (very fast).
    // Restricted to single-row decode: prefill has synchronous KV operations and
    // varying row counts that break graph topology stability.
    // local CUDA target stage excluded: receives activation via external transfer.
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
    const no_graph = @import("env_pkg").getenv("TALU_NO_GRAPH") != null;
    const graph_eligible = self.compute_stream != null and
        !trace.isEnabled() and deepstack_layer_features_opt == null and
        !event_timing_enabled and
        !no_graph;

    const graph_replay_eligible = graph_eligible and !refresh_pointer_tables and gd_layers == 0;
    compute: {
        if (graph_replay_eligible) {
            if (self.decode_graph_exec) |exec| {
                try self.device.graphLaunch(exec, self.compute_stream.?);
                self.loadKvSlot(slot_index);
                break :compute;
            }
        }

        var graph_capture_active = false;
        if (graph_replay_eligible) {
            if (self.device.streamBeginCapture(self.compute_stream.?)) {
                graph_capture_active = true;
            } else |_| {}
        }
        errdefer if (graph_capture_active) {
            _ = self.device.streamEndCapture(self.compute_stream.?) catch {};
        };

        var final_hidden = input_row;

        if (@import("env_pkg").getenv("TALU_DUMP_HIDDEN") != null) {
            log.warn("inference", "GPU_LOOP_ENTRY", .{
                .layer_limit = layer_limit,
                .split_layer = stage_adapters.localLayerOffset(self),
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
            final_hidden = self.tryExecuteLayerProgram(
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
            ) catch |err| {
                log.warn("inference", "CUDA decode layer dispatch failed", .{
                    .layer = layer_idx,
                    .slot = slot_index,
                    .position = position,
                    .seq_len = seq_len_u32,
                    .batched_attn_max_seq = self.runtime_buffers.batched_attn_max_seq_len,
                    .kv_dtype = @tagName(self.kv_cache_dtype),
                    .reason = @errorName(err),
                });
                return err;
            };
            if (layer.attention_binding != null) batch_info.attn_layer_index += 1;
            if (layer.gated_delta_binding != null) batch_info.gd_layer_index += 1;
            if (layer.shortconv_binding != null) batch_info.sc_layer_index += 1;
            dumpHiddenState(self, &self.runtime_buffers.input_dev, stage_adapters.localLayerOffset(self) + layer_idx, "post_layer", self.d_model, 1);
            if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
                if (per_layer_source_embeddings_opt) |*per_layer_source_embeddings| {
                    try per_layer_branch_feature.applyPerLayerBranch(
                        self,
                        layer_idx,
                        gemma_token_id_single[0..],
                        per_layer_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden = self.runtime_buffers.input_dev;
                    dumpHiddenState(self, &self.runtime_buffers.input_dev, stage_adapters.localLayerOffset(self) + layer_idx, "post_ple", self.d_model, 1);
                } else if (per_layer_branch_feature.hasStandaloneLayerScalars(self)) {
                    try per_layer_branch_feature.applyStandaloneLayerScalar(self, layer_idx, &self.runtime_buffers.input_dev, 1);
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
            // Finalize graph capture for non-logit stage (local CUDA source stage).
            // The copyFrom above uses async DtoD when a stream is set, so it is
            // captured into the graph alongside the layer kernels.
            if (graph_capture_active) {
                const new_graph = self.device.streamEndCapture(self.compute_stream.?) catch
                    return error.CudaGraphCaptureFailed;
                defer self.device.graphDestroy(new_graph);
                if (self.decode_graph_exec == null) {
                    self.decode_graph_exec = self.device.graphInstantiate(new_graph) catch
                        return error.CudaGraphInstantiateFailed;
                }
                try self.device.graphLaunch(self.decode_graph_exec.?, self.compute_stream);
            }
            break :compute;
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

        if (@import("env_pkg").getenv("TALU_CUDA_PROJECTION_DEBUG") != null) {
            const projection_kind = switch (self.runtime_buffers.projection_weight) {
                .dense_f32 => "dense_f32",
                .dense_u16 => "dense_u16",
                .gaffine_u4 => "gaffine_u4",
                .gaffine_u8 => "gaffine_u8",
                .fp8 => "fp8",
                .mxfp8 => "mxfp8",
                .nvfp4 => "nvfp4",
            };
            log.warn("inference", "CUDA projection weight kind", .{
                .mode = "decode",
                .kind = projection_kind,
            });
        }
        try engine_ops.linearForwardRows(self, &norm_out_row, 1, &self.runtime_buffers.projection_weight, &self.runtime_buffers.logits_dev);

        // End graph capture: instantiate once, then replay on steady-state tokens.
        if (graph_capture_active) {
            const new_graph = self.device.streamEndCapture(self.compute_stream.?) catch
                return error.CudaGraphCaptureFailed;
            defer self.device.graphDestroy(new_graph);

            if (self.decode_graph_exec == null) {
                self.decode_graph_exec = self.device.graphInstantiate(new_graph) catch
                    return error.CudaGraphInstantiateFailed;
            }
            try self.device.graphLaunch(self.decode_graph_exec.?, self.compute_stream);
        }
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
            .nvfp4 => "matmul_lm_head_nvfp4_host",
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

const BatchedDecodeOutputMode = enum {
    host_logits,
    device_only,
};

fn singleTokenOutputMode(download_logits: bool) BatchedDecodeOutputMode {
    return if (download_logits) .host_logits else .device_only;
}

fn copySingleBatchedDecodeLogitsToOutput(self: anytype, logits_out_opt: ?[]f32, download_logits: bool) !void {
    if (!download_logits) return;
    const logits_out = logits_out_opt orelse return error.InvalidArgument;
    if (logits_out.len != self.vocab_size) return error.InvalidArgument;
    const projected_vocab = self.runtime_buffers.projected_vocab;
    if (projected_vocab == logits_out.len) {
        @memcpy(logits_out, self.runtime_buffers.projected_logits_batch_host[0..projected_vocab]);
    } else {
        @memset(logits_out, -1.0e9);
        @memcpy(logits_out[0..projected_vocab], self.runtime_buffers.projected_logits_batch_host[0..projected_vocab]);
    }
}

fn decodeContext(
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
) stage_adapters.DecodeContext {
    return .{
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
}

fn stage1DeepstackFeatures(self: anytype, deepstack_layer_features_opt: ?[]const []const f32) ?[]const []const f32 {
    const deepstack_layer_features = deepstack_layer_features_opt orelse return null;
    const split_layer = stage_adapters.localLayerOffset(self);
    if (split_layer >= deepstack_layer_features.len) return null;
    return deepstack_layer_features[split_layer..];
}

fn executeSingleTokenLocalDecodePipeline(
    self: anytype,
    placement_kind: bridge.LocalPipelinePlacementKind,
    token: u32,
    position: usize,
    slot_index: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    ensure_kv_capacity: bool,
    trace_seq_len_u32: u32,
    trace_pos_offset: usize,
    hidden_override: ?[]const f32,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasDecl(SelfType, "localActivationByteCount")) return error.InvalidTopologyConfig;

    var ctx = decodeContext(
        token,
        position,
        slot_index,
        logits_out_opt,
        compute_logits,
        download_logits,
        ensure_kv_capacity,
        trace_seq_len_u32,
        trace_pos_offset,
    );
    const slot_indices = [_]usize{slot_index};
    const positions = [_]usize{position};
    const activation_bytes = try localActivationByteCountFor(self);

    switch (placement_kind) {
        .cuda_cuda => {
            if (comptime !@hasDecl(SelfType, "localCudaStage1")) return error.InvalidTopologyConfig;
            var stage1_backend = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
            try mirrorDecodeStageDescriptorsFromRoot(stage1_backend, self, &slot_indices);
            stage1_backend.activateKvSlot(slot_index);

            const Stage0 = stage_adapters.CudaDecodeSourceStage(
                @TypeOf(self),
                @TypeOf(stage1_backend),
                executeDecodeWithLayerLimit,
                .source_event_target_stream,
            );
            const Stage1 = stage_adapters.CudaDecodeTargetStage(@TypeOf(stage1_backend), executeDecodeWithLayerLimit);
            const boundary0_ids = try stage_adapters.localBoundaryStageIds(self, 0);
            var stage0 = Stage0{
                .backend = self,
                .target_backend = stage1_backend,
                .ctx = &ctx,
                .hidden_override = hidden_override,
                .deepstack_layer_features_opt = deepstack_layer_features_opt,
                .deepstack_feature_index_opt = deepstack_feature_index_opt,
            };
            var stage1 = Stage1{
                .backend = stage1_backend,
                .ctx = &ctx,
                .deepstack_layer_features_opt = stage1DeepstackFeatures(self, deepstack_layer_features_opt),
                .deepstack_feature_index_opt = deepstack_feature_index_opt,
            };
            var stages = [_]bridge.LocalStageChainStage{
                bridge.localStageAdapter(Stage0, boundary0_ids.source_stage_id, &stage0),
                bridge.localStageAdapter(Stage1, boundary0_ids.target_stage_id, &stage1),
            };
            const payload_specs = [_]stage_adapters.DecodeBoundaryPayloadSpec{.{
                .boundary_index = 0,
                .activation_byte_count = activation_bytes,
                .location_hint = try stage_adapters.cudaPayloadLocationHint(self),
                .image = .device,
            }};
            try stage_adapters.executeDecodeBoundaryPipeline(self, stages[0..], &slot_indices, &positions, payload_specs[0..]);
        },
        .cpu_cuda => {
            if (hidden_override != null or deepstack_layer_features_opt != null or deepstack_feature_index_opt != null) {
                return error.InvalidTopologyConfig;
            }
            if (comptime !@hasDecl(SelfType, "localCpuStage0") or !@hasDecl(SelfType, "localSplitLayer")) {
                return error.InvalidTopologyConfig;
            }
            const stage0_backend = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
            if (self.localSplitLayer() == 0) return error.InvalidTopologyConfig;
            self.activateKvSlot(slot_index);

            const Stage0 = stage_adapters.CpuDecodeSourceStage(@TypeOf(stage0_backend), @TypeOf(self));
            const Stage1 = stage_adapters.CudaDecodeTargetStage(@TypeOf(self), executeDecodeWithLayerLimit);
            const boundary0_ids = try stage_adapters.localBoundaryStageIds(self, 0);
            var stage0 = Stage0{ .backend = stage0_backend, .gpu_backend = self, .ctx = &ctx };
            var stage1 = Stage1{ .backend = self, .ctx = &ctx };
            var stages = [_]bridge.LocalStageChainStage{
                bridge.localStageAdapter(Stage0, boundary0_ids.source_stage_id, &stage0),
                bridge.localStageAdapter(Stage1, boundary0_ids.target_stage_id, &stage1),
            };
            const cpu_activation = stage0_backend.slotActivationBytes(slot_index);
            const payload_specs = [_]stage_adapters.DecodeBoundaryPayloadSpec{.{
                .boundary_index = 0,
                .activation_byte_count = activation_bytes,
                .location_hint = .{ .cpu = {} },
                .image = .{ .host_bytes = cpu_activation },
                .local_device_peer_copy_available = false,
            }};
            try stage_adapters.executeDecodeBoundaryPipeline(self, stages[0..], &slot_indices, &positions, payload_specs[0..]);
        },
        .cpu_cuda_cuda => {
            if (hidden_override != null or deepstack_layer_features_opt != null or deepstack_feature_index_opt != null) {
                return error.InvalidTopologyConfig;
            }
            if (comptime !@hasDecl(SelfType, "localCpuStage0") or
                !@hasDecl(SelfType, "localCudaStage1") or
                !@hasDecl(SelfType, "localSplitLayer") or
                !@hasDecl(SelfType, "localSplitLayerStage2"))
            {
                return error.InvalidTopologyConfig;
            }
            const stage0_backend = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
            var stage1_backend = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
            const split_layer = self.localSplitLayer();
            const split_layer_stage2 = self.localSplitLayerStage2();
            if (split_layer == 0 or split_layer_stage2 <= split_layer) return error.InvalidTopologyConfig;
            try mirrorDecodeStageDescriptorsFromRoot(stage1_backend, self, &slot_indices);
            stage1_backend.activateKvSlot(slot_index);
            self.activateKvSlot(slot_index);

            const Stage0 = stage_adapters.CpuDecodeSourceStage(@TypeOf(stage0_backend), @TypeOf(stage1_backend));
            const Stage1 = stage_adapters.CudaDecodeSourceStage(
                @TypeOf(stage1_backend),
                @TypeOf(self),
                executeDecodeWithLayerLimit,
                .source_stream,
            );
            const Stage2 = stage_adapters.CudaDecodeTargetStage(@TypeOf(self), executeDecodeWithLayerLimit);
            const boundary0_ids = try stage_adapters.localBoundaryStageIds(self, 0);
            const boundary1_ids = try stage_adapters.localBoundaryStageIds(self, 1);
            var stage0 = Stage0{ .backend = stage0_backend, .gpu_backend = stage1_backend, .ctx = &ctx };
            var stage1 = Stage1{ .backend = stage1_backend, .target_backend = self, .ctx = &ctx, .use_preloaded_input = true };
            var stage2 = Stage2{ .backend = self, .ctx = &ctx };
            var stages = [_]bridge.LocalStageChainStage{
                bridge.localStageAdapter(Stage0, boundary0_ids.source_stage_id, &stage0),
                bridge.localStageAdapter(Stage1, boundary0_ids.target_stage_id, &stage1),
                bridge.localStageAdapter(Stage2, boundary1_ids.target_stage_id, &stage2),
            };
            const cpu_activation = stage0_backend.slotActivationBytes(slot_index);
            const payload_specs = [_]stage_adapters.DecodeBoundaryPayloadSpec{
                .{
                    .boundary_index = 0,
                    .activation_byte_count = activation_bytes,
                    .location_hint = .{ .cpu = {} },
                    .image = .{ .host_bytes = cpu_activation },
                    .local_device_peer_copy_available = false,
                },
                .{
                    .boundary_index = 1,
                    .activation_byte_count = activation_bytes,
                    .location_hint = try stage_adapters.cudaPayloadLocationHint(stage1_backend),
                    .image = .device,
                },
            };
            try stage_adapters.executeDecodeBoundaryPipeline(self, stages[0..], &slot_indices, &positions, payload_specs[0..]);
        },
        .generic_local_chain => return error.InvalidTopologyConfig,
    }
}

const BatchedDecodeExecutionPlan = struct {
    allow_staged_internal_execution: bool = false,
    use_preloaded_input: bool = false,
    compute_logits: bool = true,
    emit_decode_summary: bool = true,
    summary_label_override: ?[]const u8 = null,
};

fn preserveBatchedDecodeStageFailure(
    root_backend: anytype,
    boundary_index: usize,
    boundary_dtype: bridge.BoundaryDType,
    boundary_layout: bridge.BoundaryLayout,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_indices: []const usize,
    positions: []const usize,
    active_side: DecodeBoundaryStageSide,
    source_error: anyerror,
) anyerror {
    const allocator = stage_adapters.backendAllocator(root_backend) orelse return source_error;
    var batch_entry_scratch = stage_adapters.DecodeBatchEntryScratch.init(allocator, slot_indices.len) catch return source_error;
    defer batch_entry_scratch.deinit();
    const metadata = stage_adapters.buildDecodeActivationMetadata(
        root_backend,
        boundary_index,
        boundary_dtype,
        boundary_layout,
        location_hint,
        slot_indices,
        positions,
        batch_entry_scratch.slice(slot_indices.len),
    ) catch return source_error;
    const placement_plan = stage_adapters.localTopologyPlacementPlan(root_backend) catch return source_error;
    return bridge.preserveLocalStageExecutionError(allocator, .{
        .placement_plan = placement_plan,
        .state_ownership_plan = stage_adapters.localTopologyStateOwnershipPlan(root_backend),
        .metadata = &metadata,
        .active_stage_id = switch (active_side) {
            .source => metadata.boundary.source_stage_id,
            .target => metadata.boundary.target_stage_id,
        },
        .source_error = source_error,
    });
}

fn elapsedNsSince(start_ns: i128) u64 {
    const elapsed_i128 = std.time.nanoTimestamp() - start_ns;
    return if (elapsed_i128 > 0) @intCast(elapsed_i128) else 0;
}

fn batchedDecodeModeLabel(
    placement_kind: bridge.LocalPipelinePlacementKind,
    output_mode: BatchedDecodeOutputMode,
) []const u8 {
    return switch (placement_kind) {
        .cuda_cuda => switch (output_mode) {
            .host_logits => "decode_pipeline2",
            .device_only => "decode_device_only_pipeline2",
        },
        .cpu_cuda => switch (output_mode) {
            .host_logits => "decode_cpu_gpu",
            .device_only => "decode_device_only_cpu_gpu",
        },
        .cpu_cuda_cuda => switch (output_mode) {
            .host_logits => "decode_cpu_gpu_gpu",
            .device_only => "decode_device_only_cpu_gpu_gpu",
        },
        .generic_local_chain => "decode_local_chain",
    };
}

fn preserveDecodeBoundaryFailure(
    root_backend: anytype,
    boundary: stage_adapters.LocalBoundaryRuntimeView,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_indices: []const usize,
    positions: []const usize,
    active_side: DecodeBoundaryStageSide,
    source_error: anyerror,
) anyerror {
    return preserveBatchedDecodeStageFailure(
        root_backend,
        boundary.boundary_index,
        boundary.dtype,
        boundary.layout,
        location_hint,
        slot_indices,
        positions,
        active_side,
        source_error,
    );
}

fn mirrorDecodeStageDescriptorsFromRoot(stage_backend: anytype, root_backend: anytype, slot_indices: []const usize) !void {
    const StageType = @TypeOf(stage_backend.*);
    if (comptime @hasField(StageType, "state_descriptor_count") and @hasDecl(StageType, "mirrorSlotStateBlocksFrom")) {
        if (stage_backend.state_descriptor_count > 0) {
            for (slot_indices) |slot_idx| {
                try stage_backend.mirrorSlotStateBlocksFrom(root_backend, slot_idx);
            }
        }
    }
}

fn computeBatchedDecodeStageOrPreserve(
    root_backend: anytype,
    stage_backend: anytype,
    boundary: stage_adapters.LocalBoundaryRuntimeView,
    location_hint: ?bridge.TensorFramePayloadLocationHint,
    slot_indices: []const usize,
    positions: []const usize,
    active_side: DecodeBoundaryStageSide,
    tokens: []const u32,
    output_mode: BatchedDecodeOutputMode,
    plan: BatchedDecodeExecutionPlan,
) !void {
    computeBatchedDecodeLogitsWithPlan(stage_backend, tokens, slot_indices, positions, output_mode, plan) catch |err| {
        return preserveDecodeBoundaryFailure(root_backend, boundary, location_hint, slot_indices, positions, active_side, err);
    };
}

fn copyBatchedDecodeHostLogitsFromStage(root_backend: anytype, source_backend: anytype, rows: usize) !void {
    const src_vocab = source_backend.runtime_buffers.projected_vocab;
    const dst_vocab = root_backend.runtime_buffers.projected_vocab;
    if (src_vocab == 0 or dst_vocab == 0) return;
    for (0..rows) |row| {
        const src_row_start = std.math.mul(usize, row, src_vocab) catch continue;
        const src_row_end = std.math.add(usize, src_row_start, src_vocab) catch continue;
        const dst_row_start = std.math.mul(usize, row, dst_vocab) catch continue;
        const dst_row_end = std.math.add(usize, dst_row_start, dst_vocab) catch continue;
        if (src_row_end > source_backend.runtime_buffers.projected_logits_batch_host.len or
            dst_row_end > root_backend.runtime_buffers.projected_logits_batch_host.len)
        {
            continue;
        }
        const src_row = source_backend.runtime_buffers.projected_logits_batch_host[src_row_start..src_row_end];
        const dst_row = root_backend.runtime_buffers.projected_logits_batch_host[dst_row_start..dst_row_end];
        const copy_len = @min(src_row.len, dst_row.len);
        @memset(dst_row, -1.0e9);
        @memcpy(dst_row[0..copy_len], src_row[0..copy_len]);
    }
}

fn prepareCpuBatchedDecodeSegments(
    root_backend: anytype,
    cpu_stage0: anytype,
    comptime activate_intermediate: bool,
    intermediate_backend: anytype,
    boundary0: stage_adapters.LocalBoundaryRuntimeView,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    split_layer: usize,
    row_bytes: usize,
    host_segments: [][]const u8,
) !void {
    for (0..tokens.len) |row_i| {
        const token = tokens[row_i];
        const slot_index = slot_indices[row_i];
        const position = positions[row_i];
        if (activate_intermediate) intermediate_backend.activateKvSlot(slot_index);
        root_backend.activateKvSlot(slot_index);
        executeCpuStage0LayerRange(
            cpu_stage0,
            token,
            position,
            slot_index,
            split_layer,
            true,
        ) catch |err| {
            const row_slot_indices = [_]usize{slot_index};
            const row_positions = [_]usize{position};
            return preserveDecodeBoundaryFailure(
                root_backend,
                boundary0,
                .{ .cpu = {} },
                &row_slot_indices,
                &row_positions,
                .source,
                err,
            );
        };
        const src_row = cpu_stage0.slotActivationBytes(slot_index);
        if (src_row.len < row_bytes) return error.InvalidTopologyConfig;
        host_segments[row_i] = src_row[0..row_bytes];
    }
}

fn computeBatchedDecodeLocalWithMode(
    self: anytype,
    placement_kind: bridge.LocalPipelinePlacementKind,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    output_mode: BatchedDecodeOutputMode,
) !void {
    const SelfType = @TypeOf(self.*);
    const row_bytes = std.math.mul(usize, self.d_model, @sizeOf(f32)) catch return error.InvalidArgument;

    switch (placement_kind) {
        .cuda_cuda => {
            if (comptime !@hasDecl(SelfType, "localCudaStage1")) return error.InvalidTopologyConfig;
            var stage1 = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
            const boundary0 = try stage_adapters.localBoundaryRuntime(self, 0);
            const decode_summary_enabled = @import("env_pkg").getenv("TALU_CUDA_DECODE_SUMMARY") != null;
            var decode_start_ns: i128 = 0;
            if (decode_summary_enabled and comptime @hasDecl(SelfType, "beginNvfp4RouteWindow") and
                @hasDecl(SelfType, "beginPhaseBudgetWindow"))
            {
                self.beginNvfp4RouteWindow();
                self.beginPhaseBudgetWindow();
                decode_start_ns = std.time.nanoTimestamp();
            }
            var stage0_compute_ns: u64 = 0;
            var stage0_to_stage1_transfer_ns: u64 = 0;
            var stage1_compute_ns: u64 = 0;
            var host_logits_copy_ns: u64 = 0;

            try mirrorDecodeStageDescriptorsFromRoot(stage1, self, slot_indices);
            const stage0_start_ns = std.time.nanoTimestamp();
            try computeBatchedDecodeStageOrPreserve(self, self, boundary0, stage_adapters.cudaPayloadLocationHint(self) catch null, slot_indices, positions, .source, tokens, .device_only, .{
                .allow_staged_internal_execution = true,
                .use_preloaded_input = false,
                .compute_logits = false,
                .emit_decode_summary = false,
            });
            if (decode_summary_enabled) try self.device.synchronize();
            stage0_compute_ns = elapsedNsSince(stage0_start_ns);

            const transfer_bytes = std.math.mul(usize, tokens.len, row_bytes) catch return error.InvalidArgument;
            const transfer_start_ns = std.time.nanoTimestamp();
            try stage_adapters.executeCudaDecodeActivationBoundary(
                self,
                self,
                stage1,
                0,
                slot_indices,
                positions,
                transfer_bytes,
                .source_event_target_stream,
            );
            if (decode_summary_enabled) {
                try self.device.synchronize();
                try stage1.device.synchronize();
            }
            stage0_to_stage1_transfer_ns = elapsedNsSince(transfer_start_ns);

            const stage1_start_ns = std.time.nanoTimestamp();
            try computeBatchedDecodeStageOrPreserve(self, stage1, boundary0, stage_adapters.cudaPayloadLocationHint(self) catch null, slot_indices, positions, .target, tokens, output_mode, .{
                .allow_staged_internal_execution = true,
                .use_preloaded_input = true,
                .compute_logits = true,
                .emit_decode_summary = false,
                .summary_label_override = batchedDecodeModeLabel(.cuda_cuda, output_mode),
            });
            if (decode_summary_enabled) try stage1.device.synchronize();
            stage1_compute_ns = elapsedNsSince(stage1_start_ns);

            if (output_mode == .host_logits) {
                const host_copy_start_ns = std.time.nanoTimestamp();
                try copyBatchedDecodeHostLogitsFromStage(self, stage1, tokens.len);
                host_logits_copy_ns = elapsedNsSince(host_copy_start_ns);
            }
            if (decode_summary_enabled and comptime @hasDecl(SelfType, "logNvfp4RouteSummaryImpl") and
                @hasDecl(SelfType, "logPhaseBudgetSummaryImpl"))
            {
                const decode_elapsed_ns = elapsedNsSince(decode_start_ns);
                const mode_label = batchedDecodeModeLabel(.cuda_cuda, output_mode);
                logDecodeInventoryOnce(self, mode_label, tokens.len, tokens.len);
                self.logNvfp4RouteSummaryImpl(mode_label, tokens.len);
                self.logPhaseBudgetSummaryImpl(mode_label, tokens.len, decode_elapsed_ns);
                const accounted_ns = saturatingU64FromU128(@as(u128, stage0_compute_ns) +
                    @as(u128, stage0_to_stage1_transfer_ns) +
                    @as(u128, stage1_compute_ns) +
                    @as(u128, host_logits_copy_ns));
                const residual_ns: u64 = if (decode_elapsed_ns >= accounted_ns) decode_elapsed_ns - accounted_ns else 0;
                const ms_divisor = 1_000_000.0;
                log.warn("inference", "CUDA decode overhead summary", .{
                    .mode = mode_label,
                    .tokens = tokens.len,
                    .elapsed_ms = @as(f64, @floatFromInt(decode_elapsed_ns)) / ms_divisor,
                    .stage0_compute_ms = @as(f64, @floatFromInt(stage0_compute_ns)) / ms_divisor,
                    .stage0_to_stage1_transfer_ms = @as(f64, @floatFromInt(stage0_to_stage1_transfer_ns)) / ms_divisor,
                    .stage1_compute_ms = @as(f64, @floatFromInt(stage1_compute_ns)) / ms_divisor,
                    .host_logits_copy_ms = @as(f64, @floatFromInt(host_logits_copy_ns)) / ms_divisor,
                    .residual_ms = @as(f64, @floatFromInt(residual_ns)) / ms_divisor,
                });
            }
        },
        .cpu_cuda => {
            if (comptime !@hasDecl(SelfType, "localCpuStage0") or !@hasDecl(SelfType, "localSplitLayer")) {
                return error.InvalidTopologyConfig;
            }
            const stage0 = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
            const split_layer = self.localSplitLayer();
            if (split_layer == 0) return error.InvalidTopologyConfig;
            try self.runtime_buffers.ensureRowCapacity(&self.device, tokens.len, self.fixed_alloc_mode);
            const boundary0 = try stage_adapters.localBoundaryRuntime(self, 0);
            var host_segment_scratch = try stage_adapters.HostSegmentScratch.init(stage_adapters.backendAllocator(self), tokens.len);
            defer host_segment_scratch.deinit();
            const host_segments = host_segment_scratch.slice(tokens.len);
            try prepareCpuBatchedDecodeSegments(self, stage0, false, self, boundary0, tokens, slot_indices, positions, split_layer, row_bytes, host_segments);

            const transfer_bytes = std.math.mul(usize, tokens.len, row_bytes) catch return error.InvalidArgument;
            try stage_adapters.executeCpuSegmentedDecodeActivationBoundary(self, stage0, self, 0, slot_indices, positions, host_segments, transfer_bytes);
            try computeBatchedDecodeStageOrPreserve(self, self, boundary0, .{ .cpu = {} }, slot_indices, positions, .target, tokens, output_mode, .{
                .allow_staged_internal_execution = true,
                .use_preloaded_input = true,
                .compute_logits = true,
                .emit_decode_summary = true,
                .summary_label_override = batchedDecodeModeLabel(.cpu_cuda, output_mode),
            });
        },
        .cpu_cuda_cuda => {
            if (comptime !@hasDecl(SelfType, "localCpuStage0") or
                !@hasDecl(SelfType, "localCudaStage1") or
                !@hasDecl(SelfType, "localSplitLayer") or
                !@hasDecl(SelfType, "localSplitLayerStage2"))
            {
                return error.InvalidTopologyConfig;
            }
            const stage0 = self.localCpuStage0() orelse return error.InvalidTopologyConfig;
            var stage1 = self.localCudaStage1() orelse return error.InvalidTopologyConfig;
            const split_layer = self.localSplitLayer();
            const split_layer_stage2 = self.localSplitLayerStage2();
            if (split_layer == 0 or split_layer_stage2 <= split_layer) return error.InvalidTopologyConfig;
            try mirrorDecodeStageDescriptorsFromRoot(stage1, self, slot_indices);
            try stage1.runtime_buffers.ensureRowCapacity(&stage1.device, tokens.len, stage1.fixed_alloc_mode);
            const boundary0 = try stage_adapters.localBoundaryRuntime(self, 0);
            const boundary1 = try stage_adapters.localBoundaryRuntime(self, 1);
            var host_segment_scratch = try stage_adapters.HostSegmentScratch.init(stage_adapters.backendAllocator(self), tokens.len);
            defer host_segment_scratch.deinit();
            const host_segments = host_segment_scratch.slice(tokens.len);
            try prepareCpuBatchedDecodeSegments(self, stage0, true, stage1, boundary0, tokens, slot_indices, positions, split_layer, row_bytes, host_segments);

            const transfer01_bytes = std.math.mul(usize, tokens.len, row_bytes) catch return error.InvalidArgument;
            try stage_adapters.executeCpuSegmentedDecodeActivationBoundary(self, stage0, stage1, 0, slot_indices, positions, host_segments, transfer01_bytes);
            try computeBatchedDecodeStageOrPreserve(self, stage1, boundary1, stage_adapters.cudaPayloadLocationHint(stage1) catch null, slot_indices, positions, .source, tokens, .device_only, .{
                .allow_staged_internal_execution = true,
                .use_preloaded_input = true,
                .compute_logits = false,
                .emit_decode_summary = false,
            });

            const transfer12_bytes = std.math.mul(usize, tokens.len, row_bytes) catch return error.InvalidArgument;
            try stage_adapters.executeCudaDecodeActivationBoundary(self, stage1, self, 1, slot_indices, positions, transfer12_bytes, .source_stream);
            try computeBatchedDecodeStageOrPreserve(self, self, boundary1, stage_adapters.cudaPayloadLocationHint(stage1) catch null, slot_indices, positions, .target, tokens, output_mode, .{
                .allow_staged_internal_execution = true,
                .use_preloaded_input = true,
                .compute_logits = true,
                .emit_decode_summary = true,
                .summary_label_override = batchedDecodeModeLabel(.cpu_cuda_cuda, output_mode),
            });
        },
        .generic_local_chain => return error.InvalidTopologyConfig,
    }
}

fn computeBatchedDecodeLogitsWithMode(
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    output_mode: BatchedDecodeOutputMode,
) !void {
    const SelfType = @TypeOf(self.*);
    if (try stage_adapters.localPipelinePlacementKind(self)) |placement_kind| {
        if (comptime @hasField(SelfType, "device")) {
            return computeBatchedDecodeLocalWithMode(self, placement_kind, tokens, slot_indices, positions, output_mode);
        }
        return error.InvalidTopologyConfig;
    }
    try rejectUnsupportedStagedBatchedDecodeRoute(topologyModeTag(self));
    if (comptime @hasDecl(SelfType, "executeDecodeWithLayerLimitTestHook")) {
        if (tokens.len != slot_indices.len or tokens.len != positions.len) return error.InvalidArgument;
        if (tokens.len == 0) return;
        if (tokens.len > self.max_batch_size) return error.InvalidArgument;
        for (tokens, slot_indices, positions) |token, slot_index, position| {
            if (comptime @hasDecl(SelfType, "slotIndexSupported")) {
                if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
            }
            if (comptime @hasField(SelfType, "max_seq_len")) {
                if (position >= self.max_seq_len) return error.InvalidArgument;
            }
            if (comptime @hasDecl(SelfType, "activateKvSlot")) {
                self.activateKvSlot(slot_index);
            }
            const logits_out_opt: ?[]f32 = switch (output_mode) {
                .host_logits => if (comptime @hasDecl(SelfType, "slotLogits"))
                    self.slotLogits(slot_index)
                else
                    return error.InvalidArgument,
                .device_only => null,
            };
            try self.executeDecodeWithLayerLimitTestHook(
                token,
                position,
                slot_index,
                logits_out_opt,
                self.block_runtime.blocks.len,
                true,
                output_mode == .host_logits,
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
    return computeBatchedDecodeLogitsWithPlan(self, tokens, slot_indices, positions, output_mode, .{});
}

fn computeBatchedDecodeLogitsWithPlan(
    self: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    output_mode: BatchedDecodeOutputMode,
    plan: BatchedDecodeExecutionPlan,
) !void {
    const SelfType = @TypeOf(self.*);
    if (!plan.compute_logits and output_mode == .host_logits) return error.InvalidArgument;

    const decode_summary_enabled = plan.emit_decode_summary and @import("env_pkg").getenv("TALU_CUDA_DECODE_SUMMARY") != null;
    var decode_start_ns: i128 = 0;
    if (decode_summary_enabled and comptime @hasDecl(SelfType, "beginNvfp4RouteWindow") and
        @hasDecl(SelfType, "beginPhaseBudgetWindow"))
    {
        self.beginNvfp4RouteWindow();
        self.beginPhaseBudgetWindow();
        decode_start_ns = std.time.nanoTimestamp();
    }
    const per_layer_branch_active = if (comptime @hasField(SelfType, "per_layer_branch_runtime"))
        self.per_layer_branch_runtime != null
    else
        false;
    const n_usize = tokens.len;
    if (n_usize == 0) return;
    if (!plan.allow_staged_internal_execution) try rejectUnsupportedStagedBatchedDecodeRoute(topologyModeTag(self));
    if (n_usize > self.max_batch_size) return error.InvalidArgument;
    const n: u32 = @intCast(n_usize);
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

    if (!plan.use_preloaded_input) {
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
                        const prefer_scalar_single_row = n_usize == 1 and embedding_lookup_u16_function != null;
                        if (!prefer_scalar_single_row) {
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

    const no_graph = @import("env_pkg").getenv("TALU_NO_GRAPH") != null;

    // CUDA graph: capture-once-replay-many. Three states:
    // 1. refresh_pointer_tables=true → run normally, no capture (batch changed).
    // 2. First steady-state step (no graph exec) → capture + instantiate + launch.
    // 3. Subsequent steady-state steps → graphLaunch only + host shadow bookkeeping.
    compute: {
        // State 3: steady-state replay — cached graph exec + stable batch.
        if (comptime @hasField(SelfType, "batched_decode_graph_exec")) {
            if (self.compute_stream != null and !no_graph and !trace.isEnabled() and !refresh_pointer_tables and
                !per_layer_branch_active and plan.compute_logits)
            {
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
            if (self.compute_stream != null and !no_graph and !trace.isEnabled() and !refresh_pointer_tables and
                !per_layer_branch_active and !event_timing_enabled and plan.compute_logits)
            {
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
        var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
            per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, n_usize);
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
            if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
                if (per_layer_source_embeddings_opt) |*per_layer_source_embeddings| {
                    try per_layer_branch_feature.applyPerLayerBranch(
                        self,
                        layer_idx,
                        tokens,
                        per_layer_source_embeddings,
                        &self.runtime_buffers.input_dev,
                    );
                    final_hidden = self.runtime_buffers.input_dev;
                } else if (per_layer_branch_feature.hasStandaloneLayerScalars(self)) {
                    try per_layer_branch_feature.applyStandaloneLayerScalar(self, layer_idx, &self.runtime_buffers.input_dev, n_usize);
                }
            }
            if (layer.attention_binding != null) batch_info.attn_layer_index += 1;
            if (layer.gated_delta_binding != null) batch_info.gd_layer_index += 1;
            if (layer.shortconv_binding != null) batch_info.sc_layer_index += 1;
        }

        if (plan.compute_logits) {
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
        }

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

    if (plan.compute_logits and output_mode == .host_logits) {
        const logits_batch_host = self.runtime_buffers.projected_logits_batch_host[0..logits_elems];
        try logits_batch.download(&self.device, std.mem.sliceAsBytes(logits_batch_host));

        applyHostLogitsPostProcess(
            logits_batch_host,
            self.loaded.config.logits_scaling,
            self.loaded.config.final_logit_softcapping,
        );
    }

    if (decode_summary_enabled and comptime @hasDecl(SelfType, "logNvfp4RouteSummaryImpl") and
        @hasDecl(SelfType, "logPhaseBudgetSummaryImpl"))
    {
        const decode_elapsed_ns: u64 = @intCast(std.time.nanoTimestamp() - decode_start_ns);
        const mode_label = plan.summary_label_override orelse switch (output_mode) {
            .host_logits => "decode",
            .device_only => if (plan.compute_logits) "decode_device_only" else "decode_layers_only",
        };
        logDecodeInventoryOnce(self, mode_label, n_usize, n_usize);
        self.logNvfp4RouteSummaryImpl(mode_label, n_usize);
        self.logPhaseBudgetSummaryImpl(mode_label, n_usize, decode_elapsed_ns);
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

test "rejectUnsupportedStagedBatchedDecodeRoute rejects staged batch route" {
    try std.testing.expectError(
        error.UnsupportedModel,
        rejectUnsupportedStagedBatchedDecodeRoute("cpu_gpu"),
    );
}

test "rejectUnsupportedStagedBatchedDecodeRoute allows single-device tag" {
    try rejectUnsupportedStagedBatchedDecodeRoute("single");
    try rejectUnsupportedStagedBatchedDecodeRoute(null);
}

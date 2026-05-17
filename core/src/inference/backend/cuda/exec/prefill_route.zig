//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

/// Resolve staged prefill chunk rows for a specific request length.
/// Keeps explicit env override behavior unchanged.
const common = @import("common.zig");
const buildAttentionKernelSet = common.buildAttentionKernelSet;
const prefillMath = common.prefillMath;
const prefillChunkContext = common.prefillChunkContext;
const prepareCudaPrefillBackend = common.prepareCudaPrefillBackend;
const populateCudaPrefillInputRows = common.populateCudaPrefillInputRows;
const executeGpuPrefillLayers = common.executeGpuPrefillLayers;
const projectFinalLogitsFromCudaStage = common.projectFinalLogitsFromCudaStage;

pub fn resolveStagedPrefillChunkRows(total_rows: usize, requested_cap: usize, env_override: bool) usize {
    const clamped = @max(@as(usize, 1), @min(total_rows, requested_cap));
    if (env_override) return clamped;
    // Empirical tuning on Blackwell NVFP4:
    // medium prefill lengths benefit from a slightly smaller staged chunk.
    if (total_rows >= 384 and total_rows <= 640 and clamped >= 254) return 254;
    return clamped;
}

fn validatePrefillRequest(
    self: anytype,
    tokens: []const u32,
    slot_index: usize,
    logits_out: []const f32,
) !void {
    if (tokens.len == 0) return error.InvalidArgument;
    if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
    if (logits_out.len != self.vocab_size) return error.InvalidArgument;
    if (tokens.len > self.max_seq_len) return error.InvalidArgument;
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
        !@hasDecl(SelfType, "executePrefillWithLayerLimitTestHook"))
    {
        try validatePrefillRequest(self, tokens, slot_index, logits_out);
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

    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);
    try validatePrefillRequest(self, tokens, slot_index, logits_out);
    if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;

    const total_rows = tokens.len;
    try prepareCudaPrefillBackend(self, total_rows);
    const math = try prefillMath(self);
    const attention_kernels = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;

    // Chunked prefill: process in chunks through all layers, building
    // KV cache incrementally. Keeps scratch buffer allocations bounded.
    var pos_base: usize = 0;
    while (pos_base < total_rows) {
        const rows = @min(total_rows - pos_base, self.prefill_chunk_rows_cap);
        const chunk = try prefillChunkContext(pos_base, rows);
        const chunk_tokens = tokens[pos_base .. pos_base + rows];
        try self.runtime_buffers.ensureRowCapacity(&self.device, rows, self.fixed_alloc_mode);
        try self.ensureLayerProgramSlotRowCapacity(rows, self.fixed_alloc_mode);
        try populateCudaPrefillInputRows(self, chunk_tokens, rows, math.row_bytes);

        var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
        if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
            per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, rows);
        }
        const final_hidden_rows = try executeGpuPrefillLayers(
            self,
            slot_index,
            chunk_tokens,
            math,
            chunk,
            attention_kernels,
            per_layer_source_embeddings_opt,
            per_layer_branch_feature.hasPerLayerBranchRuntime(self),
            null,
            layer_limit,
        );

        // Extract logits from the last row of the final chunk.
        if (pos_base + rows >= total_rows) {
            try projectFinalLogitsFromCudaStage(self, final_hidden_rows, rows, math.row_bytes, chunk.last_position, logits_out, "cuda_final_norm_host");
        }

        pos_base += rows;
    }
}

pub fn executePrefillLayerRange(
    self: anytype,
    slot_index: usize,
    tokens: []const u32,
    sequence_start: usize,
    layer_start: usize,
    layer_end: usize,
    use_preloaded_input: bool,
    compute_logits: bool,
    logits_out_opt: ?[]f32,
    source_embeddings_out: ?[]f32,
) !void {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "device")) return error.InvalidArgument;
    if (tokens.len == 0) return error.InvalidArgument;
    if (!self.slotIndexSupported(slot_index)) return error.InvalidArgument;
    if (layer_end <= layer_start) return error.InvalidArgument;
    if (comptime @hasField(SelfType, "stage_layer_start")) {
        if (layer_start != self.stage_layer_start) return error.InvalidArgument;
    }
    const layer_limit = layer_end - layer_start;
    if (layer_limit > self.block_runtime.blocks.len) return error.InvalidArgument;
    if (compute_logits and logits_out_opt == null) return error.InvalidArgument;
    if (logits_out_opt) |logits_out| {
        if (logits_out.len != self.vocab_size) return error.InvalidArgument;
    }

    const previous_launch_phase = self.device.setLaunchPhase(.prefill);
    defer _ = self.device.setLaunchPhase(previous_launch_phase);

    const required_tokens = std.math.add(usize, sequence_start, tokens.len) catch return error.InvalidArgument;
    if (sequence_start == 0) {
        try prepareCudaPrefillBackend(self, required_tokens);
    } else {
        try self.ensureKvCapacity(required_tokens);
    }

    const rows = tokens.len;
    try self.runtime_buffers.ensureRowCapacity(&self.device, rows, self.fixed_alloc_mode);
    try self.ensureLayerProgramSlotRowCapacity(rows, self.fixed_alloc_mode);
    const math = try prefillMath(self);
    const attention_kernels = buildAttentionKernelSet(self) catch return error.CudaKernelUnavailable;
    const chunk = try prefillChunkContext(sequence_start, rows);

    if (!use_preloaded_input) {
        try populateCudaPrefillInputRows(self, tokens, rows, math.row_bytes);
        if (source_embeddings_out) |out| {
            const hidden_count = std.math.mul(usize, rows, self.d_model) catch return error.InvalidArgument;
            if (out.len < hidden_count) return error.InvalidArgument;
            try self.runtime_buffers.input_dev.download(&self.device, std.mem.sliceAsBytes(out[0..hidden_count]));
        }
    } else if (source_embeddings_out != null) {
        return error.InvalidTopologyConfig;
    }

    var per_layer_source_embeddings_opt: ?compute.cuda.Buffer = null;
    if (per_layer_branch_feature.hasPerLayerBranchRuntime(self)) {
        per_layer_source_embeddings_opt = try per_layer_branch_feature.maybeCapturePerLayerSourceEmbeddings(self, rows);
    }
    const final_hidden_rows = try executeGpuPrefillLayers(
        self,
        slot_index,
        tokens,
        math,
        chunk,
        attention_kernels,
        per_layer_source_embeddings_opt,
        per_layer_branch_feature.hasPerLayerBranchRuntime(self),
        null,
        layer_limit,
    );

    if (compute_logits) {
        try projectFinalLogitsFromCudaStage(
            self,
            final_hidden_rows,
            rows,
            math.row_bytes,
            chunk.last_position,
            logits_out_opt.?,
            "cuda_final_norm_host",
        );
    }
}

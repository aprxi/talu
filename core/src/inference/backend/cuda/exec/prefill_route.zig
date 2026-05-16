//! Forward pass computation functions.
//!
//! Contains the main forward-pass entry points (single-token decode, batched
//! decode, prefill), KV capacity management, and recurrent state resets.
//! Functions use `self: anytype` to avoid circular imports with engine.zig.

const std = @import("std");
const compute = @import("compute_pkg");
const per_layer_branch_feature = @import("../per_layer_branch.zig");

fn topologyModeTag(self: anytype) ?[]const u8 {
    const SelfType = @TypeOf(self.*);
    if (comptime !@hasField(SelfType, "topology_mode")) return null;
    return @tagName(self.topology_mode);
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
const stage_adapters = @import("stage_adapters.zig");
const staged_prefill = @import("staged_prefill.zig");
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

    // Compatibility topology labels only reject staged mocks without bridge facts.
    if (comptime @hasField(SelfType, "block_runtime")) {
        if (layer_limit == self.block_runtime.blocks.len) {
            if (try stage_adapters.localPipelineFactsAvailable(self)) {
                try validatePrefillRequest(self, tokens, slot_index, logits_out);
                return staged_prefill.executeLocalPrefillPipeline(self, tokens, slot_index, logits_out);
            }
            try rejectUnsupportedStagedPrefillRoute(topologyModeTag(self));
        }
    }

    if (comptime @hasDecl(SelfType, "executePrefillWithLayerLimitTestHook")) {
        return self.executePrefillWithLayerLimitTestHook(tokens, slot_index, logits_out, layer_limit);
    }
    if (comptime @hasDecl(SelfType, "executeDecodeWithLayerLimitTestHook") and
        !@hasDecl(SelfType, "localCpuStage0"))
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

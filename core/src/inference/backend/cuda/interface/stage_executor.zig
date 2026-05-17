//! CUDA implementation of the backend stage executor interface.
//!
//! This interface executes only CUDA-owned layer ranges. It assumes any external
//! activation input has already been prepared by pipeline/transport code.

const std = @import("std");
const pipeline = @import("../../../pipeline/root.zig");
const cuda_stage_capabilities = @import("../stage_capabilities.zig");

pub const supports_local_stage_execution = true;

pub const DecodeContext = struct {
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

pub const StageLayerRange = struct {
    start: usize,
    end: usize,
};

pub fn backendKind() pipeline.HostBackendKind {
    return .cuda;
}

pub fn layerRange(backend: anytype) StageLayerRange {
    const start = stageLayerOffset(backend);
    return .{
        .start = start,
        .end = start + backend.block_runtime.blocks.len,
    };
}

pub fn maxBatchSize(backend: anytype) usize {
    return backend.max_batch_size;
}

pub fn prefillChunkRowsCap(backend: anytype) usize {
    return backend.prefill_chunk_rows_cap;
}

pub fn supportedBoundaryDTypes() []const pipeline.BoundaryDType {
    return cuda_stage_capabilities.supported_boundary_dtypes[0..];
}

pub fn validateEmptyInput(input: []const u8) !void {
    if (input.len != 0) return error.InvalidArgument;
}

pub fn decodeLayerLimit(layer_start: usize, layer_end: usize) !usize {
    if (layer_end < layer_start) return error.InvalidArgument;
    return layer_end - layer_start;
}

pub fn stageLayerOffset(backend: anytype) usize {
    const BackendType = @TypeOf(backend.*);
    if (comptime @hasField(BackendType, "stage_layer_start")) {
        return backend.stage_layer_start;
    }
    return 0;
}

pub fn cudaPayloadLocationHint(backend: anytype) !pipeline.TensorFramePayloadLocationHint {
    const BackendType = @TypeOf(backend.*);
    if (comptime !@hasField(BackendType, "device")) return error.InvalidTopologyConfig;
    const ordinal = backend.device.ordinal();
    return .{ .cuda = std.math.cast(u16, ordinal) orelse return error.InvalidTopologyConfig };
}

pub fn executeDecodeLayerRange(
    backend: anytype,
    ctx: anytype,
    layer_start: usize,
    layer_end: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    use_preloaded_input: bool,
) !void {
    if (comptime !hasDecl(@TypeOf(backend.*), "executeDecodeLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try backend.executeDecodeLayerRange(
        ctx.token,
        ctx.position,
        ctx.slot_index,
        logits_out_opt,
        layer_start,
        layer_end,
        compute_logits,
        download_logits,
        ctx.ensure_kv_capacity,
        use_preloaded_input,
    );
}

pub fn executePrefillLayerRange(
    backend: anytype,
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
    if (comptime !hasDecl(@TypeOf(backend.*), "executePrefillLayerRange")) {
        return error.InvalidTopologyConfig;
    }
    try backend.executePrefillLayerRange(
        slot_index,
        tokens,
        sequence_start,
        layer_start,
        layer_end,
        use_preloaded_input,
        compute_logits,
        logits_out_opt,
        source_embeddings_out,
    );
}

pub fn executeCudaDecodeLayerRange(
    comptime execute_decode_with_layer_limit: anytype,
    backend: anytype,
    ctx: *const DecodeContext,
    layer_start: usize,
    layer_end: usize,
    logits_out_opt: ?[]f32,
    compute_logits: bool,
    download_logits: bool,
    hidden_override: ?[]const f32,
    deepstack_layer_features_opt: ?[]const []const f32,
    deepstack_feature_index_opt: ?usize,
    use_preloaded_input: bool,
) !void {
    const local_layer_limit = try decodeLayerLimit(layer_start, layer_end);
    try execute_decode_with_layer_limit(
        backend,
        ctx.token,
        ctx.position,
        ctx.slot_index,
        logits_out_opt,
        local_layer_limit,
        compute_logits,
        download_logits,
        ctx.ensure_kv_capacity,
        ctx.trace_seq_len_u32,
        ctx.trace_pos_offset,
        hidden_override,
        deepstack_layer_features_opt,
        deepstack_feature_index_opt,
        use_preloaded_input,
    );
}

test "decodeLayerLimit rejects inverted ranges" {
    try std.testing.expectError(error.InvalidArgument, decodeLayerLimit(4, 3));
}

test "validateEmptyInput rejects route payloads" {
    try std.testing.expectError(error.InvalidArgument, validateEmptyInput(&.{1}));
}

fn hasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

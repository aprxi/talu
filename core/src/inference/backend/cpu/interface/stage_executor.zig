//! CPU implementation of the backend stage executor interface.

const bridge = @import("../../../bridge/root.zig");
const cpu_stage_capabilities = @import("../stage_capabilities.zig");

pub const supports_local_stage_execution = true;

pub const StageLayerRange = struct {
    start: usize,
    end: usize,
};

pub fn backendKind() bridge.HostBackendKind {
    return .cpu;
}

pub fn layerRange(backend: anytype) StageLayerRange {
    return .{
        .start = backend.layer_range.start,
        .end = backend.layer_range.end,
    };
}

pub fn maxBatchSize(backend: anytype) usize {
    return backend.max_batch_size;
}

pub fn prefillChunkRowsCap(backend: anytype) usize {
    _ = backend;
    return 1024;
}

pub fn supportedBoundaryDTypes() []const bridge.BoundaryDType {
    return cpu_stage_capabilities.supported_boundary_dtypes[0..];
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

pub fn prepareBatchedDecodeSegments(
    comptime preserve_decode_boundary_failure: anytype,
    root_backend: anytype,
    cpu_stage: anytype,
    activate_intermediate: bool,
    intermediate_backend: anytype,
    boundary: anytype,
    tokens: []const u32,
    slot_indices: []const usize,
    positions: []const usize,
    layer_start: usize,
    layer_end: usize,
    use_preloaded_input: bool,
    compute_logits: bool,
    row_bytes: usize,
    host_segments: [][]const u8,
) !void {
    if (comptime !hasDecl(@TypeOf(cpu_stage.*), "prepareBatchedDecodeSegments")) {
        return error.InvalidTopologyConfig;
    }
    try cpu_stage.prepareBatchedDecodeSegments(
        preserve_decode_boundary_failure,
        root_backend,
        activate_intermediate,
        intermediate_backend,
        boundary,
        tokens,
        slot_indices,
        positions,
        layer_start,
        layer_end,
        use_preloaded_input,
        compute_logits,
        row_bytes,
        host_segments,
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

fn hasDecl(comptime T: type, comptime name: []const u8) bool {
    return switch (@typeInfo(T)) {
        .@"struct", .@"enum", .@"union", .@"opaque" => @hasDecl(T, name),
        else => false,
    };
}

//! Metal backend stage executor interface placeholder.
//!
//! Metal does not yet expose local layer-range execution primitives. Keeping
//! the interface module present makes backend layout uniform without adding a
//! hidden fallback route.

const pipeline = @import("../../../pipeline/root.zig");

pub const supports_local_stage_execution = false;

pub fn backendKind() pipeline.HostBackendKind {
    return .metal;
}

pub fn supportedBoundaryDTypes() []const pipeline.BoundaryDType {
    return &.{.f32};
}

pub fn maxBatchSize(_: anytype) usize {
    return 0;
}

pub fn prefillChunkRowsCap(_: anytype) usize {
    return 0;
}

pub fn executeDecodeLayerRange(
    _: anytype,
    _: anytype,
    _: usize,
    _: usize,
    _: ?[]f32,
    _: bool,
    _: bool,
    _: bool,
) !void {
    return error.UnsupportedBackend;
}

pub fn executePrefillLayerRange(
    _: anytype,
    _: usize,
    _: []const u32,
    _: usize,
    _: usize,
    _: usize,
    _: bool,
    _: bool,
    _: ?[]f32,
    _: ?[]f32,
) !void {
    return error.UnsupportedBackend;
}

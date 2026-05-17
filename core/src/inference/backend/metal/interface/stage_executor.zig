//! Metal backend stage executor interface placeholder.
//!
//! Metal does not yet expose local layer-range execution primitives. Keeping
//! the interface module present makes backend layout uniform without adding a
//! hidden fallback route.

const bridge = @import("../../../bridge/root.zig");

pub const supports_local_stage_execution = false;

pub fn backendKind() bridge.HostBackendKind {
    return .metal;
}

pub fn supportedBoundaryDTypes() []const bridge.BoundaryDType {
    return &.{.f32};
}

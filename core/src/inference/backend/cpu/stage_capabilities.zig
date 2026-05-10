//! CPU staged-boundary capability declarations.

const pipeline = @import("../../bridge/pipeline.zig");

pub const supported_boundary_dtypes = [_]pipeline.BoundaryDType{
    .f32,
};

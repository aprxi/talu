//! CPU staged-boundary capability declarations.

const pipeline = @import("../../pipeline/pipeline.zig");

pub const supported_boundary_dtypes = [_]pipeline.BoundaryDType{
    .f32,
};

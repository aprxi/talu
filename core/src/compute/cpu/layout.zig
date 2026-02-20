//! Layout and indexing primitives for CPU compute.

pub const transform = @import("layout_transform.zig");
pub const transpose = @import("transpose.zig");
pub const masking = @import("masking.zig");
pub const broadcast = @import("broadcast.zig");

// Re-export transform helpers at namespace root to keep callers on
// `compute.cpu.layout.*` instead of deep leaf module paths.
pub const QkvViews = transform.QkvViews;
pub const fuseTwoProjectionWeights = transform.fuseTwoProjectionWeights;
pub const extractRowPrefixes = transform.extractRowPrefixes;
pub const splitLastDimContiguous = transform.splitLastDimContiguous;
pub const projectQkv = transform.projectQkv;

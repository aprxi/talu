//! Graph compatibility re-export for layer operation types.
//!
//! Source of truth lives in `core/src/models/layer_ops.zig`.

const layer_ops = @import("../models/layer_ops.zig");

pub const BufferId = layer_ops.BufferId;
pub const ResidualScale = layer_ops.ResidualScale;
pub const LayerOp = layer_ops.LayerOp;

//! Graph compatibility re-export for operation and architecture types.
//!
//! Source of truth lives in `core/src/models/op_types.zig`.

const op_types = @import("../models/op_types.zig");

pub const OpInput = op_types.OpInput;
pub const OpType = op_types.OpType;
pub const Op = op_types.Op;
pub const WeightLayout = op_types.WeightLayout;
pub const WeightTransform = op_types.WeightTransform;
pub const WeightSpec = op_types.WeightSpec;
pub const VariantAlias = op_types.VariantAlias;
pub const BlockVariant = op_types.BlockVariant;
pub const Architecture = op_types.Architecture;

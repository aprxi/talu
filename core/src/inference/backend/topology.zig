//! Shared inference backend topology primitives.
//!
//! Canonical ownership now lives in `core/src/models/op_types.zig`.
//! This module re-exports that contract for backend-local compatibility.

const model_types = @import("../../models/op_types.zig");

pub const BlockKind = model_types.BlockKind;
pub const FusedLayerKindId = model_types.FusedLayerKindId;
pub const fusedLayerKindId = model_types.fusedLayerKindId;

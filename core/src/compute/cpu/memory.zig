//! Memory movement primitives for CPU compute.

pub const copy = @import("tensor_copy.zig");
pub const gather = @import("tensor_gather.zig");
pub const padding = @import("padding.zig");
pub const slotted = @import("memory_slotted.zig");

// Re-export commonly-used helpers so callers import only this namespace.
pub const gatherRowsF32 = gather.gatherRowsF32;
pub const scatterAddRowsByPositions = gather.scatterAddRowsByPositions;
pub const collectPositionsU32 = gather.collectPositionsU32;
pub const scatterRowsByMatchedId = gather.scatterRowsByMatchedId;
pub const copy3DToSlotted4D = slotted.copy3DToSlotted4D;
pub const appendRowToSlotted4D = slotted.appendRowToSlotted4D;
pub const appendRowsToSlotted4D = slotted.appendRowsToSlotted4D;

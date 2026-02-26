//! Model plan subsystem root.
//!
//! Owns plan-facing opcode definitions and mapping from existing model metadata
//! (`OpType`, `LayerOp`) to runtime opcodes.

pub const opcode = @import("opcode.zig");
pub const opcode_map = @import("opcode_map.zig");
pub const compiler = @import("compiler.zig");

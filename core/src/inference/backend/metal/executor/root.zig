//! Metal backend executor module root.
//!
//! Groups Metal execution-time orchestration helpers.

pub const weights = @import("weights.zig");
pub const runtime = @import("runtime.zig");
pub const model = @import("model.zig");
pub const block = @import("block.zig");

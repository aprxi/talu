//! Legacy shim for the relocated agent C API module.
//!
//! New code should import `core/src/capi/agent/root.zig`.

pub const root = @import("agent/root.zig");

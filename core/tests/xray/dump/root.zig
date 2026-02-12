//! Integration tests for the xray/dump module.
//!
//! Tests types exported from core/src/xray/dump/root.zig.
//! These are dev-only tools for full tensor dumping during inference.

const std = @import("std");

pub const capture = @import("capture_test.zig");
pub const npz_writer = @import("npz_writer_test.zig");

test {
    std.testing.refAllDecls(@This());
}

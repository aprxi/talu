//! Integration tests for the validate/code module.
//!
//! Tests types exported from core/src/validate/code/root.zig.

const std = @import("std");

// CodeBlock (code block metadata)
pub const code_block = @import("code_block_test.zig");

// CodeBlockList (list of detected code blocks)
pub const code_block_list = @import("code_block_list_test.zig");

// FenceTracker (code fence state machine)
pub const fence_tracker = @import("fence_tracker_test.zig");

test {
    std.testing.refAllDecls(@This());
}

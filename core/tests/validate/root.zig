//! Validate Integration Tests
//!
//! Integration tests for the validate public API.
//! Tests types exported from core/src/validate/root.zig.

const std = @import("std");

// Validator (high-level schema validator)
pub const validator = @import("validator_test.zig");

// TokenMask (token bit vector for constrained sampling)
pub const token_mask = @import("token_mask_test.zig");

// SemanticValidator (post-parse validation for semantic constraints)
pub const semantic_validator = @import("semantic_validator_test.zig");

// Code validation module (CodeBlock, CodeBlockList, FenceTracker)
pub const code = @import("code/root.zig");

test {
    std.testing.refAllDecls(@This());
}

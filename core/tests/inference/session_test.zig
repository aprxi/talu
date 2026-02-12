//! Integration tests for inference.Session
//!
//! Tests the Session struct which provides the legacy inference session interface.
//! Note: Full inference tests require a loaded model.

const std = @import("std");
const main = @import("main");

const Session = main.inference.Session;
const InferenceConfig = main.inference.InferenceConfig;

// =============================================================================
// Configuration Tests
// =============================================================================

test "InferenceConfig has correct defaults" {
    const config = InferenceConfig{};

    try std.testing.expectEqual(@as(usize, 32), config.max_new_tokens);
    try std.testing.expectEqual(@as(?u32, null), config.bos_token_id);
}

test "InferenceConfig can be customized" {
    const config = InferenceConfig{
        .max_new_tokens = 500,
        .bos_token_id = 1,
    };

    try std.testing.expectEqual(@as(usize, 500), config.max_new_tokens);
    try std.testing.expectEqual(@as(?u32, 1), config.bos_token_id);
}

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Session type is accessible" {
    const T = Session;
    _ = T;
}

test "Session has expected structure" {
    const info = @typeInfo(Session);
    try std.testing.expect(info == .@"struct");
}

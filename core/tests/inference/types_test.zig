//! Integration tests for inference generation types.

const std = @import("std");
const main = @import("main");

const InferenceConfig = main.inference.InferenceConfig;
const InferenceState = main.inference.InferenceState;

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

test "InferenceState has expected structure" {
    const info = @typeInfo(InferenceState);
    try std.testing.expect(info == .@"struct");
}

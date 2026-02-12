//! Integration tests for inference.FinishReason
//!
//! Tests the FinishReason enum which indicates why generation stopped.

const std = @import("std");
const main = @import("main");
const FinishReason = main.inference.FinishReason;

// =============================================================================
// Enum Value Tests
// =============================================================================

test "FinishReason enum has expected variants" {
    // Verify all expected variants exist
    const eos = FinishReason.eos_token;
    const length = FinishReason.length;
    const stop = FinishReason.stop_sequence;

    try std.testing.expect(eos != length);
    try std.testing.expect(length != stop);
    try std.testing.expect(eos != stop);
}

test "FinishReason.toInt returns correct values for C-API" {
    // These values are part of the C-API contract
    try std.testing.expectEqual(@as(u8, 0), FinishReason.eos_token.toInt());
    try std.testing.expectEqual(@as(u8, 1), FinishReason.length.toInt());
    try std.testing.expectEqual(@as(u8, 2), FinishReason.stop_sequence.toInt());
}

test "FinishReason can be cast from integer" {
    // Test round-trip through integer representation
    const reasons = [_]FinishReason{ .eos_token, .length, .stop_sequence };

    for (reasons) |reason| {
        const int_val = reason.toInt();
        const back: FinishReason = @enumFromInt(int_val);
        try std.testing.expectEqual(reason, back);
    }
}

// =============================================================================
// Usage Pattern Tests
// =============================================================================

test "FinishReason can be used in switch statements" {
    const test_cases = [_]struct { reason: FinishReason, expected: []const u8 }{
        .{ .reason = .eos_token, .expected = "eos" },
        .{ .reason = .length, .expected = "length" },
        .{ .reason = .stop_sequence, .expected = "stop" },
    };

    for (test_cases) |tc| {
        const result = switch (tc.reason) {
            .eos_token => "eos",
            .length => "length",
            .stop_sequence => "stop",
        };
        try std.testing.expectEqualStrings(tc.expected, result);
    }
}

test "FinishReason default in InferenceState is eos_token" {
    const InferenceState = main.inference.InferenceState;

    // Create a minimal InferenceState to verify default
    var tokens = [_]u32{0};
    var logits = [_]f32{0.0};
    const state = InferenceState{
        .tokens = &tokens,
        .final_logits = &logits,
        .prompt_len = 0,
        .generated_len = 0,
        .prefill_ns = 0,
        .decode_ns = 0,
        // finish_reason defaults to .eos_token
    };

    try std.testing.expectEqual(FinishReason.eos_token, state.finish_reason);
}

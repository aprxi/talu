//! Integration tests for GenerationResult
//!
//! GenerationResult contains the output from engine.generate(),
//! including generated text, tokens, and timing statistics.

const std = @import("std");
const main = @import("main");
const GenerationResult = main.router.GenerationResult;

// =============================================================================
// Struct Layout Tests
// =============================================================================

test "GenerationResult has expected fields" {
    // Verify all fields exist with correct types via comptime reflection
    const fields = @typeInfo(GenerationResult).@"struct".fields;

    try std.testing.expectEqual(@as(usize, 8), fields.len);

    // Check field names exist using inline for (comptime iteration)
    var has_text = false;
    var has_tokens = false;
    var has_prompt_tokens = false;
    var has_generated_tokens = false;
    var has_prefill_ns = false;
    var has_decode_ns = false;
    var has_finish_reason = false;
    var has_tool_calls = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "text")) has_text = true;
        if (comptime std.mem.eql(u8, field.name, "tokens")) has_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "prompt_tokens")) has_prompt_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "generated_tokens")) has_generated_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "prefill_ns")) has_prefill_ns = true;
        if (comptime std.mem.eql(u8, field.name, "decode_ns")) has_decode_ns = true;
        if (comptime std.mem.eql(u8, field.name, "finish_reason")) has_finish_reason = true;
        if (comptime std.mem.eql(u8, field.name, "tool_calls")) has_tool_calls = true;
    }

    try std.testing.expect(has_text);
    try std.testing.expect(has_tokens);
    try std.testing.expect(has_prompt_tokens);
    try std.testing.expect(has_generated_tokens);
    try std.testing.expect(has_prefill_ns);
    try std.testing.expect(has_decode_ns);
    try std.testing.expect(has_finish_reason);
    try std.testing.expect(has_tool_calls);
}

test "GenerationResult can be constructed" {
    // Allocate test data
    const text = try std.testing.allocator.dupe(u8, "Hello, world!");
    const tokens = try std.testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4 });

    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 10,
        .generated_tokens = 4,
        .prefill_ns = 1_000_000, // 1ms
        .decode_ns = 2_000_000, // 2ms
    };

    try std.testing.expectEqualStrings("Hello, world!", result.text);
    try std.testing.expectEqual(@as(usize, 4), result.tokens.len);
    try std.testing.expectEqual(@as(usize, 10), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 4), result.generated_tokens);
    try std.testing.expectEqual(@as(u64, 1_000_000), result.prefill_ns);
    try std.testing.expectEqual(@as(u64, 2_000_000), result.decode_ns);

    // Clean up
    result.deinit(std.testing.allocator);
}

// =============================================================================
// Method Tests
// =============================================================================

test "GenerationResult.deinit frees memory" {
    // This test verifies deinit doesn't leak - allocator will catch leaks
    const text = try std.testing.allocator.dupe(u8, "Test response");
    const tokens = try std.testing.allocator.dupe(u32, &[_]u32{ 100, 200, 300 });

    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 5,
        .generated_tokens = 3,
        .prefill_ns = 0,
        .decode_ns = 0,
    };

    result.deinit(std.testing.allocator);
    // No leak = test passes
}

// =============================================================================
// Edge Case Tests
// =============================================================================

test "GenerationResult with empty text" {
    const text = try std.testing.allocator.dupe(u8, "");
    const tokens = try std.testing.allocator.dupe(u32, &[_]u32{});

    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 0,
        .generated_tokens = 0,
        .prefill_ns = 0,
        .decode_ns = 0,
    };

    try std.testing.expectEqual(@as(usize, 0), result.text.len);
    try std.testing.expectEqual(@as(usize, 0), result.tokens.len);

    result.deinit(std.testing.allocator);
}

test "GenerationResult timing values" {
    const text = try std.testing.allocator.dupe(u8, "x");
    const tokens = try std.testing.allocator.dupe(u32, &[_]u32{1});

    // Test with realistic timing values
    const result = GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = 100,
        .generated_tokens = 50,
        .prefill_ns = 500_000_000, // 500ms
        .decode_ns = 2_000_000_000, // 2s
    };

    // Calculate tokens/second
    const decode_seconds = @as(f64, @floatFromInt(result.decode_ns)) / 1_000_000_000.0;
    const tokens_per_second = @as(f64, @floatFromInt(result.generated_tokens)) / decode_seconds;

    try std.testing.expect(tokens_per_second > 0);

    result.deinit(std.testing.allocator);
}

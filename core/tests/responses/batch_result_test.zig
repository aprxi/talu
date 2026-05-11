//! Integration tests for BatchResult.
//!
//! BatchResult is the current response-generation completion result returned by
//! the batch wrapper path.

const std = @import("std");
const main = @import("main");
const BatchResult = main.responses.BatchResult;

test "BatchResult has expected fields" {
    const fields = @typeInfo(BatchResult).@"struct".fields;

    try std.testing.expectEqual(@as(usize, 8), fields.len);

    var has_prompt_tokens = false;
    var has_completion_tokens = false;
    var has_prefill_ns = false;
    var has_generation_ns = false;
    var has_ttft_ns = false;
    var has_finish_reason = false;
    var has_text = false;
    var has_tool_calls = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "prompt_tokens")) has_prompt_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "completion_tokens")) has_completion_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "prefill_ns")) has_prefill_ns = true;
        if (comptime std.mem.eql(u8, field.name, "generation_ns")) has_generation_ns = true;
        if (comptime std.mem.eql(u8, field.name, "ttft_ns")) has_ttft_ns = true;
        if (comptime std.mem.eql(u8, field.name, "finish_reason")) has_finish_reason = true;
        if (comptime std.mem.eql(u8, field.name, "text")) has_text = true;
        if (comptime std.mem.eql(u8, field.name, "tool_calls")) has_tool_calls = true;
    }

    try std.testing.expect(has_prompt_tokens);
    try std.testing.expect(has_completion_tokens);
    try std.testing.expect(has_prefill_ns);
    try std.testing.expect(has_generation_ns);
    try std.testing.expect(has_ttft_ns);
    try std.testing.expect(has_finish_reason);
    try std.testing.expect(has_text);
    try std.testing.expect(has_tool_calls);
}

test "BatchResult can be constructed without owned payloads" {
    var result = BatchResult{
        .prompt_tokens = 10,
        .completion_tokens = 4,
        .prefill_ns = 1_000_000,
        .generation_ns = 2_000_000,
        .ttft_ns = 500_000,
        .finish_reason = .length,
        .text = null,
        .tool_calls = null,
    };

    try std.testing.expectEqual(@as(usize, 10), result.prompt_tokens);
    try std.testing.expectEqual(@as(usize, 4), result.completion_tokens);
    try std.testing.expectEqual(@as(u64, 1_000_000), result.prefill_ns);
    try std.testing.expectEqual(@as(u64, 2_000_000), result.generation_ns);
    try std.testing.expectEqual(@as(u64, 500_000), result.ttft_ns);
    try std.testing.expect(result.text == null);
    try std.testing.expect(result.tool_calls == null);

    result.deinit();
}

test "BatchResult.deinit releases owned text allocated by the batch allocator" {
    const text = try std.heap.c_allocator.dupe(u8, "Hello, world!");

    var result = BatchResult{
        .prompt_tokens = 1,
        .completion_tokens = 2,
        .prefill_ns = 0,
        .generation_ns = 0,
        .ttft_ns = 0,
        .finish_reason = .eos_token,
        .text = text,
        .tool_calls = null,
    };

    result.deinit();
}

//! Integration tests for inference.Request
//!
//! Tests the Request struct which represents a generation request
//! managed by the scheduler.

const std = @import("std");
const main = @import("main");

const Request = main.inference.Request;
const RequestState = main.inference.RequestState;
const SamplingConfig = main.inference.SamplingConfig;

// =============================================================================
// Initialization Tests
// =============================================================================

test "Request struct has expected fields" {
    const allocator = std.testing.allocator;

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_ids = [_]u32{0};

    var request = Request{
        .id = 1,
        .state = .queued,
        .slot_idx = null,
        .prompt_tokens = &prompt,
        .max_tokens = 100,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &eos_ids,
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 0,
        .submit_time = 0,
    };

    try std.testing.expectEqual(@as(u64, 1), request.id);
    try std.testing.expectEqual(RequestState.queued, request.state);
    try std.testing.expectEqual(@as(?usize, null), request.slot_idx);
    try std.testing.expectEqual(@as(usize, 3), request.prompt_tokens.len);
    try std.testing.expectEqual(@as(usize, 100), request.max_tokens);

    request.deinit(allocator);
}

test "Request with generated tokens" {
    const allocator = std.testing.allocator;

    const prompt = [_]u32{1};
    const eos_ids = [_]u32{0};

    var request = Request{
        .id = 42,
        .state = .generating,
        .slot_idx = 5,
        .prompt_tokens = &prompt,
        .max_tokens = 50,
        .generated_tokens = .{},
        .token_position = 1,
        .eos_token_ids = &eos_ids,
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 10,
        .submit_time = 12345,
    };

    try request.generated_tokens.append(allocator, 100);
    try request.generated_tokens.append(allocator, 200);
    try request.generated_tokens.append(allocator, 300);

    try std.testing.expectEqual(@as(usize, 3), request.generated_tokens.items.len);
    try std.testing.expectEqual(@as(u32, 100), request.generated_tokens.items[0]);
    try std.testing.expectEqual(@as(u32, 200), request.generated_tokens.items[1]);
    try std.testing.expectEqual(@as(u32, 300), request.generated_tokens.items[2]);

    request.deinit(allocator);
}

test "Request.deinit frees error message" {
    const allocator = std.testing.allocator;

    const prompt = [_]u32{1};
    const eos_ids = [_]u32{0};

    var request = Request{
        .id = 1,
        .state = .failed,
        .slot_idx = null,
        .prompt_tokens = &prompt,
        .max_tokens = 10,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &eos_ids,
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = try allocator.dupe(u8, "Test error message"),
        .priority = 0,
        .submit_time = 0,
    };

    try std.testing.expectEqualStrings("Test error message", request.error_msg.?);

    request.deinit(allocator);
}

// =============================================================================
// State Tests
// =============================================================================

test "Request state transitions" {
    var request = Request{
        .id = 1,
        .state = .queued,
        .slot_idx = null,
        .prompt_tokens = &.{},
        .max_tokens = 10,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &.{},
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 0,
        .submit_time = 0,
    };

    try std.testing.expectEqual(RequestState.queued, request.state);

    request.slot_idx = 0;
    request.state = .pending_prefill;
    try std.testing.expectEqual(RequestState.pending_prefill, request.state);

    request.state = .generating;
    try std.testing.expectEqual(RequestState.generating, request.state);

    request.state = .completed;
    try std.testing.expectEqual(RequestState.completed, request.state);
}

test "Request priority ordering" {
    const high_priority = Request{
        .id = 1,
        .state = .queued,
        .slot_idx = null,
        .prompt_tokens = &.{},
        .max_tokens = 10,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &.{},
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 100,
        .submit_time = 1000,
    };

    const low_priority = Request{
        .id = 2,
        .state = .queued,
        .slot_idx = null,
        .prompt_tokens = &.{},
        .max_tokens = 10,
        .generated_tokens = .{},
        .token_position = 0,
        .eos_token_ids = &.{},
        .callback = null,
        .callback_data = null,
        .sampling_config = .{},
        .error_msg = null,
        .priority = 0,
        .submit_time = 500,
    };

    try std.testing.expect(high_priority.priority > low_priority.priority);
    try std.testing.expect(low_priority.submit_time < high_priority.submit_time);
}

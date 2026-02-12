//! Integration tests for router.SchedulerRequest
//!
//! SchedulerRequest represents a generation request managed by the Scheduler.
//! It's re-exported from the inference module as SchedulerRequest for use with LocalEngine.

const std = @import("std");
const main = @import("main");

const SchedulerRequest = main.router.SchedulerRequest;
const SchedulerRequestState = main.router.SchedulerRequestState;
const SamplingConfig = main.router.SamplingConfig;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "SchedulerRequest type is accessible from router" {
    const T = SchedulerRequest;
    _ = T;
}

test "SchedulerRequest is a struct" {
    const info = @typeInfo(SchedulerRequest);
    try std.testing.expect(info == .@"struct");
}

test "SchedulerRequest has expected fields" {
    const info = @typeInfo(SchedulerRequest);
    const fields = info.@"struct".fields;

    var has_id = false;
    var has_state = false;
    var has_slot_idx = false;
    var has_prompt_tokens = false;
    var has_max_tokens = false;
    var has_generated_tokens = false;

    inline for (fields) |field| {
        if (comptime std.mem.eql(u8, field.name, "id")) has_id = true;
        if (comptime std.mem.eql(u8, field.name, "state")) has_state = true;
        if (comptime std.mem.eql(u8, field.name, "slot_idx")) has_slot_idx = true;
        if (comptime std.mem.eql(u8, field.name, "prompt_tokens")) has_prompt_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "max_tokens")) has_max_tokens = true;
        if (comptime std.mem.eql(u8, field.name, "generated_tokens")) has_generated_tokens = true;
    }

    try std.testing.expect(has_id);
    try std.testing.expect(has_state);
    try std.testing.expect(has_slot_idx);
    try std.testing.expect(has_prompt_tokens);
    try std.testing.expect(has_max_tokens);
    try std.testing.expect(has_generated_tokens);
}

// =============================================================================
// Initialization Tests
// =============================================================================

test "SchedulerRequest struct can be initialized" {
    const allocator = std.testing.allocator;

    const prompt = [_]u32{ 1, 2, 3 };
    const eos_ids = [_]u32{0};

    var request = SchedulerRequest{
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
    try std.testing.expectEqual(SchedulerRequestState.queued, request.state);
    try std.testing.expectEqual(@as(?usize, null), request.slot_idx);
    try std.testing.expectEqual(@as(usize, 3), request.prompt_tokens.len);
    try std.testing.expectEqual(@as(usize, 100), request.max_tokens);

    request.deinit(allocator);
}

test "SchedulerRequest with generated tokens" {
    const allocator = std.testing.allocator;

    const prompt = [_]u32{1};
    const eos_ids = [_]u32{0};

    var request = SchedulerRequest{
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

test "SchedulerRequest.deinit frees error message" {
    const allocator = std.testing.allocator;

    const prompt = [_]u32{1};
    const eos_ids = [_]u32{0};

    var request = SchedulerRequest{
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

test "SchedulerRequest state transitions" {
    var request = SchedulerRequest{
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

    try std.testing.expectEqual(SchedulerRequestState.queued, request.state);

    request.slot_idx = 0;
    request.state = .pending_prefill;
    try std.testing.expectEqual(SchedulerRequestState.pending_prefill, request.state);

    request.state = .generating;
    try std.testing.expectEqual(SchedulerRequestState.generating, request.state);

    request.state = .completed;
    try std.testing.expectEqual(SchedulerRequestState.completed, request.state);
}

test "SchedulerRequest priority ordering" {
    const high_priority = SchedulerRequest{
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

    const low_priority = SchedulerRequest{
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

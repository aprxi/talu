//! Integration tests for inference.Scheduler
//!
//! Tests the Scheduler struct which manages continuous batching of requests.
//! Note: Full scheduling tests require a backend.

const std = @import("std");
const main = @import("main");

const Scheduler = main.inference.Scheduler;
const SchedulerConfig = main.inference.SchedulerConfig;
const RequestState = main.inference.RequestState;
const TokenEvent = main.inference.TokenEvent;
const SamplingConfig = main.inference.SamplingConfig;

// =============================================================================
// SchedulerConfig Tests
// =============================================================================

test "SchedulerConfig has correct defaults" {
    const config = SchedulerConfig{};

    try std.testing.expectEqual(@as(?usize, null), config.max_concurrent);
    try std.testing.expectEqual(@as(usize, 0), config.default_eos_ids.len);
    try std.testing.expectEqual(false, config.priority_scheduling);
}

test "SchedulerConfig custom values" {
    const eos_ids = [_]u32{ 0, 2 };
    const sampling = SamplingConfig{
        .strategy = .top_k,
        .top_k = 50,
        .temperature = 0.7,
    };

    const config = SchedulerConfig{
        .max_concurrent = 8,
        .default_eos_ids = &eos_ids,
        .default_sampling = sampling,
        .priority_scheduling = true,
    };

    try std.testing.expectEqual(@as(?usize, 8), config.max_concurrent);
    try std.testing.expectEqual(@as(usize, 2), config.default_eos_ids.len);
    try std.testing.expectEqual(@as(u32, 0), config.default_eos_ids[0]);
    try std.testing.expectEqual(@as(u32, 2), config.default_eos_ids[1]);
    try std.testing.expectEqual(true, config.priority_scheduling);
    try std.testing.expectEqual(@as(usize, 50), config.default_sampling.top_k);
}

// =============================================================================
// RequestState Tests
// =============================================================================

test "RequestState enum has all expected variants" {
    const states = [_]RequestState{
        .queued,
        .pending_prefill,
        .generating,
        .completed,
        .cancelled,
        .failed,
    };

    for (states, 0..) |s1, i| {
        for (states[i + 1 ..]) |s2| {
            try std.testing.expect(s1 != s2);
        }
    }
}

test "RequestState can be used in switch statements" {
    const test_cases = [_]struct { state: RequestState, is_active: bool }{
        .{ .state = .queued, .is_active = false },
        .{ .state = .pending_prefill, .is_active = true },
        .{ .state = .generating, .is_active = true },
        .{ .state = .completed, .is_active = false },
        .{ .state = .cancelled, .is_active = false },
        .{ .state = .failed, .is_active = false },
    };

    for (test_cases) |tc| {
        const is_active = switch (tc.state) {
            .pending_prefill, .generating => true,
            else => false,
        };
        try std.testing.expectEqual(tc.is_active, is_active);
    }
}

// =============================================================================
// TokenEvent Tests
// =============================================================================

test "TokenEvent fields are correctly initialized" {
    const event = TokenEvent{
        .request_id = 42,
        .token = 100,
        .is_final = false,
        .slot_idx = 3,
    };

    try std.testing.expectEqual(@as(u64, 42), event.request_id);
    try std.testing.expectEqual(@as(u32, 100), event.token);
    try std.testing.expectEqual(false, event.is_final);
    try std.testing.expectEqual(@as(usize, 3), event.slot_idx);
}

test "TokenEvent with is_final true" {
    const event = TokenEvent{
        .request_id = 1,
        .token = 2,
        .is_final = true,
        .slot_idx = 0,
    };

    try std.testing.expectEqual(true, event.is_final);
}

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Scheduler type is accessible" {
    const T = Scheduler;
    _ = T;
}

test "Scheduler has expected structure" {
    const info = @typeInfo(Scheduler);
    try std.testing.expect(info == .@"struct");
}

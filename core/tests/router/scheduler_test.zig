//! Integration tests for router.Scheduler
//!
//! Scheduler manages continuous batching of inference requests.
//! It's re-exported from the inference module for use with LocalEngine.
//!
//! Note: Full scheduling tests require a loaded model/backend.
//! These tests verify the type structure and exports.

const std = @import("std");
const main = @import("main");

const Scheduler = main.router.Scheduler;
const SchedulerConfig = main.router.SchedulerConfig;
const SchedulerRequestState = main.router.SchedulerRequestState;
const SchedulerTokenEvent = main.router.SchedulerTokenEvent;
const SamplingConfig = main.router.SamplingConfig;
const SamplingStrategy = main.router.SamplingStrategy;

// =============================================================================
// Type Verification Tests
// =============================================================================

test "Scheduler type is accessible from router" {
    const T = Scheduler;
    _ = T;
}

test "Scheduler is a struct" {
    const info = @typeInfo(Scheduler);
    try std.testing.expect(info == .@"struct");
}

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
// SchedulerRequestState Tests
// =============================================================================

test "SchedulerRequestState enum has all expected variants" {
    const states = [_]SchedulerRequestState{
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

test "SchedulerRequestState can be used in switch statements" {
    const test_cases = [_]struct { state: SchedulerRequestState, is_active: bool }{
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
// SchedulerTokenEvent Tests
// =============================================================================

test "SchedulerTokenEvent fields are correctly initialized" {
    const event = SchedulerTokenEvent{
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

test "SchedulerTokenEvent with is_final true" {
    const event = SchedulerTokenEvent{
        .request_id = 1,
        .token = 2,
        .is_final = true,
        .slot_idx = 0,
    };

    try std.testing.expectEqual(true, event.is_final);
}

// =============================================================================
// SamplingStrategy Tests
// =============================================================================

test "SamplingStrategy has expected variants" {
    const strategies = [_]SamplingStrategy{
        .greedy,
        .top_k,
        .top_p,
    };

    for (strategies, 0..) |s1, i| {
        for (strategies[i + 1 ..]) |s2| {
            try std.testing.expect(s1 != s2);
        }
    }
}

// =============================================================================
// SamplingConfig Tests
// =============================================================================

test "SamplingConfig has correct defaults" {
    const config = SamplingConfig{};

    try std.testing.expectEqual(SamplingStrategy.greedy, config.strategy);
    try std.testing.expectEqual(@as(f32, 1.0), config.temperature);
}

test "SamplingConfig custom values" {
    const config = SamplingConfig{
        .strategy = .top_p,
        .temperature = 0.8,
        .top_k = 40,
        .top_p = 0.95,
    };

    try std.testing.expectEqual(SamplingStrategy.top_p, config.strategy);
    try std.testing.expectEqual(@as(f32, 0.8), config.temperature);
    try std.testing.expectEqual(@as(usize, 40), config.top_k);
    try std.testing.expectEqual(@as(f32, 0.95), config.top_p);
}

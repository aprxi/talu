//! Inference scheduler module.
//!
//! Neutral contracts live under `scheduler/`; the current concrete
//! `GenericScheduler` implementation remains shared from the CPU backend.

const std = @import("std");

pub const contracts = @import("scheduler/contracts.zig");
const cpu_scheduler = @import("backend/cpu/scheduler.zig");

pub const Scheduler = cpu_scheduler.Scheduler;
pub const GenericScheduler = cpu_scheduler.GenericScheduler;

pub const SchedulerConfig = contracts.SchedulerConfig;
pub const RequestState = contracts.RequestState;
pub const TokenEvent = contracts.TokenEvent;
pub const Request = contracts.Request;
pub const FinishReason = contracts.FinishReason;
pub const TokenizerView = contracts.TokenizerView;
pub const SchedulerSingleDecodeRoute = contracts.SchedulerSingleDecodeRoute;
pub const SchedulerSingleDecodeRoutePlan = contracts.SchedulerSingleDecodeRoutePlan;
pub const SchedulerBatchedTopKRoutePlan = contracts.SchedulerBatchedTopKRoutePlan;

test "scheduler facade preserves neutral contracts and concrete scheduler exports" {
    try std.testing.expect(Request == contracts.Request);
    try std.testing.expect(RequestState == contracts.RequestState);
    try std.testing.expect(SchedulerConfig == contracts.SchedulerConfig);
    try std.testing.expect(TokenEvent == contracts.TokenEvent);
    try std.testing.expect(@hasDecl(@This(), "Scheduler"));
    try std.testing.expect(@hasDecl(@This(), "GenericScheduler"));
}

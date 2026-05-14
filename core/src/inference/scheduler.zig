//! Inference scheduler module.
//!
//! Backend-neutral scheduler contracts and implementation live under
//! `scheduler/`. Backend modules instantiate `GenericScheduler` with their
//! concrete backend type. The facade keeps `Scheduler` as the default CPU
//! scheduler alias for existing callers.

const std = @import("std");

pub const contracts = @import("scheduler/contracts.zig");
pub const generic = @import("scheduler/generic.zig");
const default_cpu_scheduler = @import("backend/cpu/scheduler.zig");

pub const Scheduler = default_cpu_scheduler.Scheduler;
pub const GenericScheduler = generic.GenericScheduler;

pub const SchedulerConfig = contracts.SchedulerConfig;
pub const RequestState = contracts.RequestState;
pub const TokenEvent = contracts.TokenEvent;
pub const DecodeRequest = contracts.DecodeRequest;
pub const DecodeResult = contracts.DecodeResult;
pub const PrefillBatchRequest = contracts.PrefillBatchRequest;
pub const Request = contracts.Request;
pub const FinishReason = contracts.FinishReason;
pub const TokenizerView = contracts.TokenizerView;
pub const SchedulerTopKCandidateRoutePlan = contracts.SchedulerTopKCandidateRoutePlan;
pub const SchedulerBatchedTopKRoutePlan = contracts.SchedulerBatchedTopKRoutePlan;

test "scheduler facade preserves neutral contracts and concrete scheduler exports" {
    try std.testing.expect(Request == contracts.Request);
    try std.testing.expect(RequestState == contracts.RequestState);
    try std.testing.expect(SchedulerConfig == contracts.SchedulerConfig);
    try std.testing.expect(TokenEvent == contracts.TokenEvent);
    try std.testing.expect(@hasDecl(@This(), "Scheduler"));
    try std.testing.expect(@hasDecl(@This(), "GenericScheduler"));
}

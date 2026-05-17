//! Inference scheduler module.
//!
//! Backend-neutral scheduler contracts, batching, and execution-target
//! initialization live under `scheduler/`. Backend modules instantiate
//! `GenericScheduler` with their concrete backend type. Response-serving code
//! usually instantiates it with `ExecutionTarget`, which can wrap one backend or
//! a local pipeline.

const std = @import("std");

pub const contracts = @import("contracts.zig");
pub const generic = @import("generic.zig");
pub const execution_target = @import("execution_target.zig");
const default_cpu_scheduler = @import("../backend/cpu/scheduler.zig");

pub const Scheduler = default_cpu_scheduler.Scheduler;
pub const GenericScheduler = generic.GenericScheduler;
pub const ExecutionTarget = execution_target.ExecutionTarget;
pub const ExecutionTargetInitOptions = execution_target.InitOptions;
pub const TargetSelection = execution_target.Selection;

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
pub const PoolingStrategy = execution_target.PoolingStrategy;
pub const has_metal = execution_target.has_metal;
pub const defaultModelLoadOptions = execution_target.defaultModelLoadOptions;
pub const effectiveLoadSelection = execution_target.effectiveLoadSelection;

test "scheduler facade preserves neutral contracts and concrete scheduler exports" {
    try std.testing.expect(Request == contracts.Request);
    try std.testing.expect(RequestState == contracts.RequestState);
    try std.testing.expect(SchedulerConfig == contracts.SchedulerConfig);
    try std.testing.expect(TokenEvent == contracts.TokenEvent);
    try std.testing.expect(@hasDecl(@This(), "Scheduler"));
    try std.testing.expect(@hasDecl(@This(), "GenericScheduler"));
    try std.testing.expect(@hasDecl(@This(), "ExecutionTarget"));
    try std.testing.expect(@hasDecl(@This(), "ExecutionTargetInitOptions"));
}

//! Inference scheduler module.
//!
//! Implementation currently lives in CPU backend path; this wrapper keeps the
//! public inference surface backend-neutral by module path.

const cpu_scheduler = @import("backend/cpu/scheduler.zig");

pub const Scheduler = cpu_scheduler.Scheduler;
pub const SchedulerConfig = cpu_scheduler.SchedulerConfig;
pub const RequestState = cpu_scheduler.RequestState;
pub const TokenEvent = cpu_scheduler.TokenEvent;
pub const Request = cpu_scheduler.Request;
pub const FinishReason = cpu_scheduler.FinishReason;
pub const GenericScheduler = cpu_scheduler.GenericScheduler;


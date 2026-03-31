//! Metal backend scheduler surface.
//!
//! Deliberately aliases CPU scheduler exports for now to keep backend module
//! contracts symmetric while scheduler policy remains shared.

const cpu_scheduler = @import("../cpu/scheduler.zig");

pub const RequestState = cpu_scheduler.RequestState;
pub const Request = cpu_scheduler.Request;
pub const FinishReason = cpu_scheduler.FinishReason;
pub const TokenEvent = cpu_scheduler.TokenEvent;
pub const SchedulerConfig = cpu_scheduler.SchedulerConfig;
pub const GenericScheduler = cpu_scheduler.GenericScheduler;
pub const Scheduler = cpu_scheduler.Scheduler;

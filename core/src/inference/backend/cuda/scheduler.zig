//! CUDA backend scheduler module.
//!
//! Scheduler is backend-agnostic host orchestration. CUDA backend uses the
//! shared production scheduler instantiated for `CudaBackend`.

const engine = @import("engine.zig");
const shared_scheduler = @import("../cpu/scheduler.zig");

pub const RequestState = shared_scheduler.RequestState;
pub const Request = shared_scheduler.Request;
pub const FinishReason = shared_scheduler.FinishReason;
pub const TokenEvent = shared_scheduler.TokenEvent;
pub const SchedulerConfig = shared_scheduler.SchedulerConfig;

pub const GenericScheduler = shared_scheduler.GenericScheduler;
pub const Scheduler = GenericScheduler(engine.CudaBackend);

//! CPU scheduler adapter.
//!
//! The scheduler implementation is backend-neutral and lives in
//! `inference/scheduler/generic.zig`. This module only binds it to the CPU
//! backend type and re-exports the shared scheduler contracts.

const engine = @import("engine.zig");
const contracts = @import("../../scheduler/contracts.zig");
const generic = @import("../../scheduler/generic.zig");

pub const RequestState = contracts.RequestState;
pub const Request = contracts.Request;
pub const FinishReason = contracts.FinishReason;
pub const TokenEvent = contracts.TokenEvent;
pub const DecodeRequest = contracts.DecodeRequest;
pub const DecodeResult = contracts.DecodeResult;
pub const PrefillBatchRequest = contracts.PrefillBatchRequest;
pub const TokenizerView = contracts.TokenizerView;
pub const SchedulerConfig = contracts.SchedulerConfig;
pub const SchedulerSingleDecodeRoute = contracts.SchedulerSingleDecodeRoute;
pub const SchedulerSingleDecodeRoutePlan = contracts.SchedulerSingleDecodeRoutePlan;
pub const SchedulerBatchedTopKRoutePlan = contracts.SchedulerBatchedTopKRoutePlan;

pub const GenericScheduler = generic.GenericScheduler;
pub const Scheduler = GenericScheduler(engine.FusedCpuBackend);

//! Response-serving inference boundary.
//!
//! Responses modules import inference through this boundary instead of directly
//! depending on inference internals.

const inference_abi = @import("inference_pkg");

pub const types = inference_abi.types;
pub const sampling = inference_abi.sampling;
pub const vision_types = inference_abi.vision_types;
pub const runtime_contract = inference_abi.runtime_contract;
pub const scheduler = inference_abi.scheduler;
pub const ExecutionTarget = inference_abi.ExecutionTarget;
pub const ExecutionTargetInitOptions = inference_abi.ExecutionTargetInitOptions;
pub const SchedulerConfig = inference_abi.SchedulerConfig;
pub const Request = inference_abi.Request;
pub const RequestState = inference_abi.RequestState;
pub const TokenEvent = inference_abi.TokenEvent;
pub const SamplingStrategy = inference_abi.SamplingStrategy;
pub const SamplingConfig = inference_abi.SamplingConfig;

pub const generation_config = @import("../models/config/generation.zig");

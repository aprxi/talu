//! Router-facing inference boundary.
//!
//! Router modules must import inference through this bridge instead of directly
//! importing inference internals.

const inference_abi = @import("inference_pkg");

pub const root = inference_abi.root;
pub const types = inference_abi.types;
pub const sampling = inference_abi.sampling;
pub const vision_types = inference_abi.vision_types;
pub const runtime_contract = inference_abi.runtime_contract;
pub const scheduler = inference_abi.scheduler;
pub const SchedulerConfig = inference_abi.SchedulerConfig;
pub const Request = inference_abi.Request;
pub const RequestState = inference_abi.RequestState;
pub const TokenEvent = inference_abi.TokenEvent;
pub const SamplingStrategy = inference_abi.SamplingStrategy;
pub const SamplingConfig = inference_abi.SamplingConfig;

pub const backend = inference_abi.backend;
pub const generation_config = @import("../config/generation.zig");
pub const preprocessor_config = inference_abi.preprocessor_config;

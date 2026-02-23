//! CUDA backend module root.
//!
//! This module exports the backend contract surface and CUDA-specific
//! orchestration modules. Contract-compatibility aliases remain explicit here
//! so `contract.zig` layout checks stay deterministic.

const engine_mod = @import("engine.zig");
const contract = @import("../contract.zig");
const compute = @import("../../../compute/root.zig");

pub const BackendType = engine_mod.CudaBackend;
pub const CudaBackend = engine_mod.CudaBackend;
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;

pub const engine = engine_mod;
pub const attention = @import("attention.zig");
pub const attention_policy = @import("attention_policy.zig");
pub const decode = @import("decode.zig");
pub const prefill = @import("prefill.zig");
pub const vision = @import("vision/root.zig");
pub const executor = @import("executor/root.zig");
pub const kernels = @import("kernels/root.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("sampling.zig");
pub const primitive_capabilities = compute.cuda.capabilities.support;

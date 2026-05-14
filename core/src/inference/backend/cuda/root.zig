//! CUDA backend module root.
//!
//! This module exports the backend contract surface. CUDA runtime internals are
//! imported by their owner modules rather than re-exported here.

const engine_mod = @import("engine.zig");
const contract = @import("../contract.zig");

pub const BackendType = engine_mod.CudaBackend;
pub const CudaBackend = engine_mod.CudaBackend;
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;

pub const engine = engine_mod;
pub const vision = @import("vision.zig");
pub const executor = @import("contract_executor.zig");
pub const kernels = @import("contract_kernels.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("../../sampling.zig");

//! CUDA backend module root.
//!
//! Phase 1 intentionally keeps CUDA module shape symmetric with CPU by
//! reusing CPU executor/kernel/vision/scheduler/sampling surfaces while
//! CUDA engine methods remain stubbed.

const engine_mod = @import("engine.zig");
const contract = @import("../contract.zig");
const compute = @import("../../../compute/root.zig");

pub const BackendType = engine_mod.CudaBackend;
pub const CudaBackend = engine_mod.CudaBackend;
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;

pub const engine = engine_mod;
pub const vision = @import("../cpu/vision/root.zig");
pub const executor = @import("../cpu/executor/root.zig");
pub const kernels = @import("../cpu/kernels/root.zig");
pub const scheduler = @import("../cpu/scheduler.zig");
pub const sampling = @import("../cpu/sampling.zig");
pub const primitive_capabilities = compute.cuda.capabilities.support;

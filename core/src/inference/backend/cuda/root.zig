//! CUDA backend module root.
//!
//! This module exports the backend contract surface. CUDA runtime internals are
//! imported by their owner modules rather than re-exported here.

const builtin = @import("builtin");

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
pub const interface = @import("interface/root.zig");
pub const stage_capabilities = @import("stage_capabilities.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("../../sampling/root.zig");

pub const testing = if (builtin.is_test) struct {
    pub const engine = engine_mod;
    pub const runtime = @import("runtime/root.zig");
    pub const weights = @import("weights/root.zig");
    pub const operators = @import("operators/root.zig");
    pub const exec = @import("exec/root.zig");
    pub const interface = @import("interface/root.zig");
} else struct {};

test "inference.backend.cuda fused projection wrapper tag policies" {
    try @import("operators/ffn/fused_gate_up.zig").testing.expectSplitWrapperTagPolicy();
    try @import("operators/qkv/fused.zig").testing.expectWrapperTagPolicy();
}

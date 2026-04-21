//! CUDA backend module root.
//!
//! This module exports the backend contract surface and CUDA-specific
//! orchestration modules. Contract aliasing remains explicit here
//! so `contract.zig` layout checks stay deterministic.

const engine_mod = @import("engine.zig");
const contract = @import("../contract.zig");
const compute = @import("compute_pkg");

pub const BackendType = engine_mod.CudaBackend;
pub const CudaBackend = engine_mod.CudaBackend;
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;

pub const engine = engine_mod;
pub const stage = @import("stage.zig");
pub const attention_path = @import("attention_path.zig");
pub const attention = attention_path;
pub const attention_policy = @import("attention_policy.zig");
pub const decode = @import("decode.zig");
pub const prefill = @import("prefill.zig");
pub const vision = @import("vision.zig");
pub const contract_executor = @import("contract_executor.zig");
pub const contract_kernels = @import("contract_kernels.zig");
pub const executor = contract_executor;
pub const kernels = contract_kernels;
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("sampling.zig");
pub const selftest = @import("selftest.zig");
pub const smoke_checks = selftest;

pub const runtime = @import("runtime/root.zig");
pub const exec = @import("exec/root.zig");
pub const program = @import("program/root.zig");
pub const operators = @import("operators/root.zig");
pub const weights = @import("weights/root.zig");
pub const engine_parts = @import("engine/root.zig");
pub const per_layer_branch = @import("per_layer_branch.zig");

pub const primitive_capabilities = compute.cuda.capabilities.support;

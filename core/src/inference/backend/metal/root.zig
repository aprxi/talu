//! Metal backend module root.
//!
//! Defines the backend module layout contract surface:
//! - `BackendType`
//! - `executor`
//! - `kernels`
//! - `engine`
//! - `vision`
//! - `scheduler`
//! - `sampling`

const engine_mod = @import("engine.zig");
const contract = @import("../contract.zig");
const compute = @import("../../../compute/root.zig");

pub const BackendType = engine_mod.MetalBackend;
pub const MetalBackend = engine_mod.MetalBackend;
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;

pub const engine = engine_mod;
pub const vision = @import("vision/root.zig");
pub const executor = @import("executor/root.zig");
pub const kernels = @import("kernels/root.zig");
pub const runtime_graph = @import("runtime_graph.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("sampling.zig");
pub const primitive_capabilities = compute.metal.capabilities.support;

pub const device = engine_mod.device;
pub const matmul = engine_mod.matmul;
pub const Graph = engine_mod.Graph;
pub const Forward = engine_mod.Forward;

pub const Device = engine_mod.Device;
pub const Buffer = engine_mod.Buffer;
pub const isAvailable = engine_mod.isAvailable;
pub const Cache = runtime_graph.Cache;
pub const WeightHandles = engine_mod.WeightHandles;

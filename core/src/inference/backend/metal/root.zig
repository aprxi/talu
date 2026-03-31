const engine_mod = @import("engine.zig");
const contract = @import("../contract.zig");

pub const BackendType = engine_mod.MetalBackend;
pub const MetalBackend = engine_mod.MetalBackend;
pub const DecodeRequest = contract.DecodeRequest;
pub const DecodeResult = contract.DecodeResult;

pub const engine = engine_mod;
pub const vision = @import("vision/root.zig");
pub const executor = @import("executor/root.zig");
pub const kernels = @import("kernels/root.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("sampling.zig");

pub const isAvailable = engine_mod.MetalBackend.isAvailable;

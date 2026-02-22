//! CUDA GPU compute primitives.
//!
//! Current phase includes runtime probing, device/buffer lifecycle,
//! cuBLAS matmul, and modular kernel runtime scaffolding.

pub const capabilities = @import("capabilities.zig");
pub const device = @import("device.zig");
pub const matmul = @import("matmul.zig");
pub const args = @import("args.zig");
pub const module = @import("module.zig");
pub const launch = @import("launch.zig");
pub const manifest = @import("manifest.zig");
pub const sideload = @import("sideload.zig");
pub const registry = @import("registry.zig");
pub const vector_add = @import("vector_add.zig");
pub const rmsnorm = @import("rmsnorm.zig");

pub const Device = device.Device;
pub const Buffer = device.Buffer;
pub const Probe = device.Probe;
pub const probeRuntime = device.probeRuntime;
pub const Blas = matmul.Blas;
pub const Module = module.Module;
pub const Function = module.Function;
pub const ArgPack = args.ArgPack;
pub const LaunchConfig = launch.LaunchConfig;
pub const Registry = registry.Registry;

test {
    _ = capabilities;
    _ = device;
    _ = matmul;
    _ = args;
    _ = module;
    _ = launch;
    _ = manifest;
    _ = sideload;
    _ = registry;
    _ = vector_add;
    _ = rmsnorm;
}

//! CPU backend module root.
//!
//! Defines the backend module layout contract surface:
//! - `BackendType`
//! - `executor`
//! - `kernels`
//! - `engine`
//! - `scheduler`
//! - `sampling`

const engine_mod = @import("engine.zig");

pub const BackendType = engine_mod.FusedCpuBackend;
pub const FusedCpuBackend = engine_mod.FusedCpuBackend;
pub const DecodeRequest = engine_mod.DecodeRequest;
pub const DecodeResult = engine_mod.DecodeResult;

pub const engine = engine_mod;
pub const executor = @import("executor/root.zig");
pub const kernels = @import("kernels/root.zig");
pub const scheduler = @import("scheduler.zig");
pub const sampling = @import("sampling.zig");

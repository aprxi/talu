//! Metal GPU Compute Primitives
//!
//! Low-level Metal/MLX kernel bindings for GPU-accelerated computation.
//! These are pure compute primitives with no orchestration logic.
//!
//! Main exports:
//! - `mlx` - MLX backend helpers (grouped-affine matmul, graph API wrappers)
//! - `matmul` - Metal matmul bindings
//! - `graph` - MLX lazy graph API
//! - `device` - Metal device abstraction

pub const mlx = @import("mlx.zig");
pub const matmul = @import("matmul.zig");
pub const graph = @import("graph.zig");
pub const device = @import("device.zig");

// Re-export commonly used types.
pub const Device = device.Device;
pub const Buffer = device.Buffer;
pub const isAvailable = device.isAvailable;
pub const Cache = graph.Cache;

// Re-export grouped-affine matmul functions (MLX backend).
pub const matmulGaffineU4 = mlx.matmulGaffineU4;

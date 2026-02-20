//! Compute subsystem boundary map.
//!
//! This is the single entry point for the compute module. All external code should
//! import from here.
//!
//! ## Public API
//!
//! - `device` - Device type definitions (CPU, CUDA, Metal)
//! - `parallel` - Thread pool and parallel execution (shared system primitive)
//! - `dlpack` - DLPack tensor exchange protocol types
//! - `mmap_policy` - file mapping policy helpers
//!
//! ## Internal API (for core/src/ only)
//!
//! - `metal` - Metal/MLX GPU compute primitives
//! - `cpu` - CPU-specific compute primitives shared by inference kernels

// =============================================================================
// Public API
// =============================================================================

/// Device type definitions (DLPack-compatible).
pub const device = @import("device.zig");

/// Thread pool and parallel execution.
pub const parallel = @import("../system/parallel.zig");

/// DLPack tensor exchange protocol types.
pub const dlpack = @import("dlpack.zig");

/// Memory mapping policy.
pub const mmap_policy = @import("mmap_policy.zig");

// Re-export commonly used types
pub const Device = device.Device;
pub const DeviceType = device.DeviceType;
pub const ThreadPool = parallel.ThreadPool;

// Re-export DLPack types for external interop
pub const DLDataType = dlpack.DLDataType;
pub const DLDevice = dlpack.DLDevice;
pub const DLTensor = dlpack.DLTensor;
pub const DLManagedTensor = dlpack.DLManagedTensor;

// Re-export behavioral types so check_coverage.sh --integration can verify test coverage
pub const MatmulScratch = cpu.matmul.MatmulScratch;
pub const TensorView = cpu.tensor_view.TensorView;

// =============================================================================
// Internal API (for core/src/ only)
// =============================================================================

/// Metal/MLX GPU compute primitives.
pub const metal = @import("metal/root.zig");

/// CPU compute primitives used by inference kernels.
pub const cpu = @import("cpu/root.zig");

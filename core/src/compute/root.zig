//! Compute Subsystem - tensor operations, SIMD, parallelization, and device management.
//!
//! This is the single entry point for the compute module. All external code should
//! import from here.
//!
//! ## Public API
//!
//! - `ops` - Core tensor operations (matmul, attention, activations, etc.)
//! - `simd` - SIMD vector primitives and architecture detection
//! - `quant` - Quantization utilities (Q4, Q8, grouped-affine)
//! - `device` - Device type definitions (CPU, CUDA, Metal)
//! - `parallel` - Thread pool and parallel execution
//! - `dlpack` - DLPack tensor exchange protocol types
//!
//! ## Internal API (for core/src/ only)
//!
//! - `metal` - Metal/MLX GPU compute primitives

// =============================================================================
// Public API
// =============================================================================

/// Core tensor operations.
pub const ops = @import("ops/root.zig");

/// SIMD vector primitives and architecture detection.
pub const simd = @import("simd/root.zig");

/// Quantization utilities.
pub const quant = @import("quant/root.zig");

/// Device type definitions (DLPack-compatible).
pub const device = @import("device.zig");

/// Thread pool and parallel execution.
pub const parallel = @import("parallel.zig");

/// DLPack tensor exchange protocol types.
pub const dlpack = @import("dlpack.zig");

/// Kernel interface and registry.
pub const kernel = @import("kernel.zig");
pub const registry = @import("registry.zig");

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
pub const MatmulScratch = ops.matmul.MatmulScratch;
pub const TensorView = ops.tensor_view.TensorView;

// =============================================================================
// Internal API (for core/src/ only)
// =============================================================================

/// Metal/MLX GPU compute primitives.
pub const metal = @import("metal/root.zig");

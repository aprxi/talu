//! Compute subsystem boundary map.
//!
//! This is the single entry point for the compute module. All external code should
//! import from here.
//!
//! ## Public API
//!
//! - `device` - Device type definitions (CPU, CUDA, Metal)
//! - `dtype` - canonical numeric and quantized data type definitions
//! - `tensor` - tensor descriptors, owned tensor storage, and DLPack interop
//! - `tensor_desc` - allocation-free tensor metadata and byte-span validation
//! - `capability` - static backend capability query vocabulary
//! - `copy_cast` - typed copy and cast contract validators
//! - `parallel` - Thread pool and parallel execution
//! - `dlpack` - DLPack tensor exchange protocol types
//! - `mmap_policy` - file mapping policy helpers
//!
//! ## Internal API (for core/src/ only)
//!
//! - `metal` - Metal/MLX GPU compute primitives
//! - `cuda` - CUDA compute primitives (stub)
//! - `cpu` - CPU-specific compute primitives consumed by runtime layers

// =============================================================================
// Public API
// =============================================================================

/// Device type definitions (DLPack-compatible).
pub const device = @import("device.zig");

/// Canonical numeric and quantized data type definitions.
pub const dtype = @import("dtype.zig");

/// Tensor descriptors, owned tensor storage, and DLPack interop.
pub const tensor = @import("tensor.zig");

/// Allocation-free tensor metadata and byte-span validation.
pub const tensor_desc = @import("tensor_desc.zig");

/// Static backend capability query vocabulary.
pub const capability = @import("capability.zig");

/// Typed copy and cast contract validators.
pub const copy_cast = @import("copy_cast.zig");

/// Thread pool and parallel execution.
pub const parallel = @import("parallel.zig");

/// DLPack tensor exchange protocol types.
pub const dlpack = @import("dlpack.zig");

/// Memory mapping policy.
pub const mmap_policy = @import("mmap_policy.zig");

// Re-export commonly used types
pub const Device = device.Device;
pub const DeviceType = device.DeviceType;
pub const DType = dtype.DType;
pub const Tensor = tensor.Tensor;
pub const OwnedTensor = tensor.OwnedTensor;
pub const TensorDescriptor = tensor_desc.TensorDescriptor;
pub const TensorLayout = tensor_desc.Layout;
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

/// CUDA GPU compute primitives.
pub const cuda = @import("cuda/root.zig");

/// CPU compute primitives used by runtime execution layers.
pub const cpu = @import("cpu/root.zig");

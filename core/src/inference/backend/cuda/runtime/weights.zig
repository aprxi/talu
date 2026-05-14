//! CUDA runtime aliases for compute-owned tensor and linear-weight handles.

const linear = @import("compute_pkg").cuda.linear;

pub const DenseU16Dtype = linear.DenseU16Dtype;
pub const EmbeddingLookupKind = linear.EmbeddingLookupKind;
pub const DeviceTensor = linear.DeviceTensor;
pub const missing_device_tensor = linear.missing_device_tensor;
pub const missing_host_tensor = linear.missing_host_tensor;
pub const EmbeddingLookup = linear.EmbeddingLookup;
pub const GaffineU4LinearWeight = linear.GaffineU4LinearWeight;
pub const GaffineU8LinearWeight = linear.GaffineU8LinearWeight;
pub const U16LinearWeight = linear.U16LinearWeight;
pub const Fp8LinearWeight = linear.Fp8LinearWeight;
pub const Mxfp8LinearWeight = linear.Mxfp8LinearWeight;
pub const Nvfp4LinearWeight = linear.Nvfp4LinearWeight;
pub const LinearWeight = linear.LinearWeight;

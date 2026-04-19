//! Tensor type aliases for C API modules.
//!
//! The active C API no longer exposes standalone tensor lifecycle functions.
//! DLPack interoperability is provided through `capi/dlpack.zig` buffer APIs.

const tensor_mod = @import("tensor_pkg");

pub const Tensor = tensor_mod.Tensor;
pub const DType = tensor_mod.DType;
pub const Device = tensor_mod.Device;
pub const DLManagedTensor = tensor_mod.DLManagedTensor;

//! Inference boundary root.
//!
//! This module exports the inference stack building blocks used across
//! conversion and routing boundaries:
//! - inference execution/runtime
//! - model architecture metadata/loading
//! - compute backends/kernels

pub const inference = @import("abi.zig");
pub const models = @import("models_pkg");
pub const compute = @import("compute_pkg");
pub const tensor = @import("compute_pkg").tensor;
pub const dtype = @import("compute_pkg").dtype;
pub const backend = @import("backend/root.zig");
pub const preprocessor_config = @import("config/preprocessor.zig");

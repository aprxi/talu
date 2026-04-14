//! Core Module Root
//!
//! This is the root.zig for the core module. All behavioral types that need
//! integration test coverage must be exported from here.
//!
//! The check_coverage.sh --integration script scans *root.zig files to detect
//! exported behavioral types.

const std = @import("std");

// =============================================================================
// Core Submodules
// =============================================================================

pub const tensor = @import("tensor_pkg");
pub const dtype = @import("dtype_pkg");
pub const compute = @import("compute_pkg");
pub const validate = @import("validate_pkg");
pub const db = @import("db/root.zig");
pub const image = @import("image_pkg");
pub const collab = @import("collab/root.zig");

// =============================================================================
// Behavioral Type Exports
// =============================================================================
// These types have pub fn methods and must be exported so check_coverage.sh --integration
// can verify they have integration test coverage.

pub const Tensor = tensor.Tensor;
pub const OwnedTensor = tensor.OwnedTensor;
pub const DType = dtype.DType;
pub const Device = compute.Device;
pub const DeviceType = compute.DeviceType;
pub const Image = image.Image;
pub const ModelBuffer = image.ModelBuffer;

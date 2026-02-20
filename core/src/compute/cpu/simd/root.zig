//! SIMD kernels entrypoint.
//!
//! This module groups SIMD-optimized kernels for compute ops.

pub const arch = @import("arch/root.zig");
pub const ssm_scan = @import("ssm_scan.zig");
pub const flash_attention = @import("flash_attention.zig");

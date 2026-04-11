//! CUDA integration tests for compute module surface.

pub const device = @import("device_test.zig");
pub const matmul = @import("matmul_test.zig");
pub const args = @import("args_test.zig");
pub const manifest = @import("manifest_test.zig");
pub const sideload = @import("sideload_test.zig");
pub const registry = @import("registry_test.zig");
pub const vector_add = @import("vector_add_test.zig");
pub const module = @import("module_test.zig");
pub const launch = @import("launch_test.zig");
pub const rmsnorm = @import("rmsnorm_test.zig");
pub const rope = @import("rope_test.zig");
pub const flash_prefill = @import("flash_prefill_test.zig");
pub const topk_rows = @import("topk_rows_test.zig");
pub const dequant_i32_scales = @import("dequant_i32_scales_test.zig");
pub const gaffine_u4_matvec = @import("gaffine_u4_matvec_test.zig");
pub const gaffine_quant_parity = @import("gaffine_quant_parity_test.zig");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}

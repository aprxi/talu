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

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}

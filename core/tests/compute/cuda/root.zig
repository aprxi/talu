//! CUDA integration tests for compute module surface.

pub const device = @import("device_test.zig");

const std = @import("std");
test {
    std.testing.refAllDecls(@This());
}

//! Integration tests for CUDA module loader wrappers.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "Module.load returns InvalidArgument for empty image" {
    if (cuda.probeRuntime() != .available) return error.SkipZigTest;

    var device = try cuda.Device.init();
    defer device.deinit();

    try std.testing.expectError(error.InvalidArgument, cuda.module.Module.load(&device, &.{}));
}

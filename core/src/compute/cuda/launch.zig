//! CUDA kernel launch wrapper.

const std = @import("std");
const device_mod = @import("device.zig");
const module_mod = @import("module.zig");
const args_mod = @import("args.zig");

pub const LaunchConfig = struct {
    grid_x: u32 = 1,
    grid_y: u32 = 1,
    grid_z: u32 = 1,
    block_x: u32 = 1,
    block_y: u32 = 1,
    block_z: u32 = 1,
    shared_mem_bytes: u32 = 0,

    pub fn validate(self: LaunchConfig) !void {
        if (self.grid_x == 0 or self.grid_y == 0 or self.grid_z == 0) return error.InvalidArgument;
        if (self.block_x == 0 or self.block_y == 0 or self.block_z == 0) return error.InvalidArgument;
    }
};

pub fn launch(
    device: *device_mod.Device,
    function: module_mod.Function,
    config: LaunchConfig,
    arg_pack: *const args_mod.ArgPack,
) !void {
    try config.validate();
    try device.launchKernel(
        function.handle,
        config.grid_x,
        config.grid_y,
        config.grid_z,
        config.block_x,
        config.block_y,
        config.block_z,
        config.shared_mem_bytes,
        arg_pack.asKernelParams(),
    );
}

test "LaunchConfig.validate accepts non-zero dimensions" {
    try (LaunchConfig{
        .grid_x = 1,
        .grid_y = 2,
        .grid_z = 3,
        .block_x = 64,
        .block_y = 1,
        .block_z = 1,
    }).validate();
}

test "LaunchConfig.validate rejects zero grid dimensions" {
    try std.testing.expectError(error.InvalidArgument, (LaunchConfig{
        .grid_x = 0,
        .block_x = 32,
    }).validate());
}

test "LaunchConfig.validate rejects zero block dimensions" {
    try std.testing.expectError(error.InvalidArgument, (LaunchConfig{
        .grid_x = 1,
        .block_x = 0,
    }).validate());
}

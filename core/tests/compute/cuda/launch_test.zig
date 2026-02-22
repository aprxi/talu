//! Integration tests for CUDA launch configuration helpers.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "LaunchConfig.validate accepts valid configuration" {
    try (cuda.launch.LaunchConfig{
        .grid_x = 1,
        .grid_y = 1,
        .grid_z = 1,
        .block_x = 256,
        .block_y = 1,
        .block_z = 1,
    }).validate();
}

test "LaunchConfig.validate rejects zero block_x" {
    try std.testing.expectError(
        error.InvalidArgument,
        (cuda.launch.LaunchConfig{
            .grid_x = 1,
            .block_x = 0,
        }).validate(),
    );
}

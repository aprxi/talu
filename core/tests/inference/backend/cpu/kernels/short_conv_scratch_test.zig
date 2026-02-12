//! Integration tests for inference.backend.cpu.kernels ShortConvScratch

const std = @import("std");
const main = @import("main");

const kernels = main.inference.backend.kernels;
const ShortConvScratch = kernels.ShortConvScratch;
const ShortConvConfig = kernels.ShortConvConfig;

test "ShortConvScratch type is accessible" {
    _ = ShortConvScratch;
}

test "ShortConvScratch is a struct" {
    const info = @typeInfo(ShortConvScratch);
    try std.testing.expect(info == .@"struct");
}

test "ShortConvScratch.init allocates buffer" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    // Verify buffer was allocated
    try std.testing.expect(scratch.buffer.len > 0);
}

test "ShortConvScratch getProjection returns correct size" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    // Projection is 3 * conv_dim (for B, C, x_proj)
    const proj = scratch.getProjection(3 * 768);
    try std.testing.expectEqual(@as(usize, 2304), proj.len);
}

test "ShortConvScratch getConvOutput returns correct size" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    const conv = scratch.getConvOutput(768);
    try std.testing.expectEqual(@as(usize, 768), conv.len);
}

test "ShortConvScratch getGatedOutput returns correct size" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    const gated = scratch.getGatedOutput(768);
    try std.testing.expectEqual(@as(usize, 768), gated.len);
}

test "ShortConvScratch buffer is zero-initialized" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 256,
        .d_conv = 4,
        .conv_dim = 128,
        .conv_dim_out = 256,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    // Verify buffer is zero-initialized
    for (scratch.buffer) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "ShortConvScratch offsets are non-overlapping" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 512,
        .d_conv = 3,
        .conv_dim = 256,
        .conv_dim_out = 512,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    // Offsets should be non-overlapping
    // proj_offset = 0
    // conv_offset = 3 * conv_dim = 768
    // gated_offset = conv_offset + conv_dim = 768 + 256 = 1024

    try std.testing.expectEqual(@as(usize, 0), scratch.proj_offset);
    try std.testing.expect(scratch.conv_offset > scratch.proj_offset);
    try std.testing.expect(scratch.gated_offset > scratch.conv_offset);
}

test "ShortConvScratch total buffer size" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 512,
        .d_conv = 3,
        .conv_dim = 256,
        .conv_dim_out = 512,
    };

    var scratch = try ShortConvScratch.init(allocator, config);
    defer scratch.deinit();

    // Total = 3*conv_dim + conv_dim + conv_dim = 5*conv_dim
    const expected_total = 5 * 256;
    try std.testing.expectEqual(expected_total, scratch.buffer.len);
}

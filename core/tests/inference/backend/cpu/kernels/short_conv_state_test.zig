//! Integration tests for inference.backend.cpu.kernels ShortConvState

const std = @import("std");
const main = @import("main");

const kernels = main.inference.backend.kernels;
const ShortConvState = kernels.ShortConvState;
const ShortConvConfig = kernels.ShortConvConfig;

test "ShortConvState type is accessible" {
    _ = ShortConvState;
}

test "ShortConvState is a struct" {
    const info = @typeInfo(ShortConvState);
    try std.testing.expect(info == .@"struct");
}

test "ShortConvState.init allocates state buffers" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var state = try ShortConvState.init(allocator, 1, config);
    defer state.deinit();

    // Verify dimensions
    try std.testing.expectEqual(@as(usize, 768), state.conv_dim);
    try std.testing.expectEqual(@as(usize, 3), state.d_conv);
    try std.testing.expectEqual(@as(usize, 1), state.batch_size);
}

test "ShortConvState.init zero-initializes state" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 256,
        .d_conv = 4,
        .conv_dim = 128,
        .conv_dim_out = 256,
    };

    var state = try ShortConvState.init(allocator, 2, config);
    defer state.deinit();

    // Verify all values are zero-initialized
    for (state.conv_state) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "ShortConvState conv_state size" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var state = try ShortConvState.init(allocator, 2, config);
    defer state.deinit();

    // Conv state should be batch * conv_dim * d_conv
    // = 2 * 768 * 3 = 4608
    const expected_size = 2 * 768 * 3;
    try std.testing.expectEqual(expected_size, state.conv_state.len);
}

test "ShortConvState.reset zeroes state" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    var state = try ShortConvState.init(allocator, 1, config);
    defer state.deinit();

    // Modify state
    state.conv_state[0] = 1.0;
    state.conv_state[100] = 2.5;

    // Reset
    state.reset();

    // Verify all zeroed
    for (state.conv_state) |v| {
        try std.testing.expectEqual(@as(f32, 0), v);
    }
}

test "ShortConvState multiple batches" {
    const allocator = std.testing.allocator;

    const config = ShortConvConfig{
        .d_model = 512,
        .d_conv = 4,
        .conv_dim = 256,
        .conv_dim_out = 512,
    };

    const batch_size: usize = 4;
    var state = try ShortConvState.init(allocator, batch_size, config);
    defer state.deinit();

    try std.testing.expectEqual(batch_size, state.batch_size);

    // State should be batch * conv_dim * d_conv
    const expected_size = batch_size * 256 * 4;
    try std.testing.expectEqual(expected_size, state.conv_state.len);
}

//! Integration tests for inference.backend.cpu.kernels MambaState

const std = @import("std");
const main = @import("main");

const kernels = main.inference.backend.kernels;
const MambaState = kernels.MambaState;
const MambaConfig = kernels.MambaConfig;

test "MambaState type is accessible" {
    _ = MambaState;
}

test "MambaState.init allocates state buffers" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var state = try MambaState.init(allocator, 1, config);
    defer state.deinit();

    // Verify dimensions
    const expected_d_inner = @as(usize, config.n_heads) * @as(usize, config.d_head);
    try std.testing.expectEqual(expected_d_inner, state.d_inner);
    try std.testing.expectEqual(@as(usize, config.d_conv), state.d_conv);
    try std.testing.expectEqual(@as(usize, config.n_heads), state.n_heads);
    try std.testing.expectEqual(@as(usize, config.d_head), state.d_head);
    try std.testing.expectEqual(@as(usize, config.d_state), state.d_state);
}

test "MambaState.reset zeroes state" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var state = try MambaState.init(allocator, 1, config);
    defer state.deinit();

    // Modify state
    state.conv_state[0] = 1.0;
    state.ssm_state[0] = 2.0;

    // Reset and verify
    state.reset();
    try std.testing.expectEqual(@as(f32, 0), state.conv_state[0]);
    try std.testing.expectEqual(@as(f32, 0), state.ssm_state[0]);
}

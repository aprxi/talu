//! Integration tests for inference.backend.cpu.kernels MambaScratch

const std = @import("std");
const main = @import("main");

const kernels = main.inference.backend.kernels;
const MambaScratch = kernels.MambaScratch;
const MambaConfig = kernels.MambaConfig;

test "MambaScratch type is accessible" {
    _ = MambaScratch;
}

test "MambaScratch.init allocates buffer" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var scratch = try MambaScratch.init(allocator, config);
    defer scratch.deinit();

    // Verify buffer was allocated
    try std.testing.expect(scratch.buffer.len > 0);
}

test "MambaScratch getters return valid slices" {
    const allocator = std.testing.allocator;

    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    var scratch = try MambaScratch.init(allocator, config);
    defer scratch.deinit();

    const d_inner: usize = @as(usize, config.n_heads) * @as(usize, config.d_head);
    const proj_len = 2 * d_inner + 2 * config.n_groups * config.d_state + config.n_heads;

    // Verify getters return correct sizes
    const proj = scratch.getProjection(proj_len);
    try std.testing.expectEqual(proj_len, proj.len);

    const conv = scratch.getConvOutput(d_inner);
    try std.testing.expectEqual(d_inner, conv.len);

    const ssm = scratch.getSsmOutput(d_inner);
    try std.testing.expectEqual(d_inner, ssm.len);

    const dt = scratch.getDt(config.n_heads);
    try std.testing.expectEqual(@as(usize, config.n_heads), dt.len);
}

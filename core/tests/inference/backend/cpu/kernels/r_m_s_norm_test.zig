//! Integration tests for RMSNorm kernel.
//!
//! Tests RMSNorm forward pass, traced forward, and describe functionality.

const std = @import("std");
const testing = std.testing;

const main = @import("main");
const norm_kernel = main.inference.backend.block_kernels.norm;
const tensor = main.core.tensor;

const RMSNorm = norm_kernel.RMSNorm;
const Tensor = tensor.Tensor;

test "RMSNorm: forward pass normalizes input" {
    const allocator = testing.allocator;

    // Create weight tensor [dim]
    const dim = 8;
    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    @memset(weight_owned.asSlice(f32), 1.0);

    const weight_tensor = weight_owned.toTensor();

    // Create RMSNorm config
    const norm = RMSNorm{
        .weight = &weight_tensor,
        .dim = dim,
        .eps = 1e-6,
        .weight_offset = 0.0,
    };

    // Create input tensor [1, 1, dim]
    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer input_owned.deinit();

    // Fill with values that have known RMS
    const input_data = input_owned.asSlice(f32);
    for (input_data, 0..) |*v, i| {
        v.* = @floatFromInt(i + 1);
    }

    // Create output tensor
    var output_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer output_owned.deinit();

    const input_tensor = input_owned.toTensor();
    var output_tensor = output_owned.toTensor();

    // Run forward pass
    norm.forward(&input_tensor, &output_tensor);

    // Output should be normalized
    const output_data = output_owned.asSlice(f32);

    // Verify output is not all zeros (normalization happened)
    var has_nonzero = false;
    for (output_data) |v| {
        if (v != 0.0) has_nonzero = true;
    }
    try testing.expect(has_nonzero);
}

test "RMSNorm: describe writes output" {
    const allocator = testing.allocator;

    const dim = 8;
    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    @memset(weight_owned.asSlice(f32), 1.0);

    const weight_tensor = weight_owned.toTensor();

    const norm = RMSNorm{
        .weight = &weight_tensor,
        .dim = dim,
        .eps = 1e-5,
        .weight_offset = 0.0,
    };

    var buffer: [512]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buffer);
    const writer = stream.writer();

    try norm.describe(writer, 0, false);

    const output = stream.getWritten();
    try testing.expect(output.len > 0);
    // Should contain RMSNorm-related info
    try testing.expect(std.mem.indexOf(u8, output, "RMSNorm") != null or
        std.mem.indexOf(u8, output, "dim") != null or
        std.mem.indexOf(u8, output, "8") != null);
}

test "RMSNorm: forwardTraced runs without error" {
    const allocator = testing.allocator;

    const dim = 8;
    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    @memset(weight_owned.asSlice(f32), 1.0);

    const weight_tensor = weight_owned.toTensor();

    const norm = RMSNorm{
        .weight = &weight_tensor,
        .dim = dim,
        .eps = 1e-6,
        .weight_offset = 0.0,
    };

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer input_owned.deinit();
    @memset(input_owned.asSlice(f32), 1.0);

    var output_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer output_owned.deinit();

    const input_tensor = input_owned.toTensor();
    var output_tensor = output_owned.toTensor();

    // forwardTraced should complete without error
    norm.forwardTraced(&input_tensor, &output_tensor);

    // Output should be valid
    const output_data = output_owned.asSlice(f32);
    for (output_data) |v| {
        try testing.expect(!std.math.isNan(v));
    }
}

test "RMSNorm: weight_offset affects output" {
    const allocator = testing.allocator;

    const dim = 4;
    var weight_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{dim});
    defer weight_owned.deinit();
    @memset(weight_owned.asSlice(f32), 0.5);

    const weight_tensor = weight_owned.toTensor();

    // Create two norms with different weight offsets
    const norm_no_offset = RMSNorm{
        .weight = &weight_tensor,
        .dim = dim,
        .eps = 1e-6,
        .weight_offset = 0.0,
    };

    const norm_with_offset = RMSNorm{
        .weight = &weight_tensor,
        .dim = dim,
        .eps = 1e-6,
        .weight_offset = 1.0, // Gemma-style (1+w) formulation
    };

    var input_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer input_owned.deinit();
    @memset(input_owned.asSlice(f32), 2.0);

    var output1_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer output1_owned.deinit();

    var output2_owned = try tensor.OwnedTensor.init(allocator, .f32, &[_]usize{ 1, 1, dim });
    defer output2_owned.deinit();

    const input_tensor = input_owned.toTensor();
    var output1_tensor = output1_owned.toTensor();
    var output2_tensor = output2_owned.toTensor();

    norm_no_offset.forward(&input_tensor, &output1_tensor);
    norm_with_offset.forward(&input_tensor, &output2_tensor);

    // Outputs should be different
    const out1 = output1_owned.asSlice(f32);
    const out2 = output2_owned.asSlice(f32);

    var different = false;
    for (out1, out2) |v1, v2| {
        if (@abs(v1 - v2) > 1e-6) different = true;
    }
    try testing.expect(different);
}

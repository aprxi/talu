//! Integration tests for inference.backend.cpu.kernels ShortConvKernel

const std = @import("std");
const main = @import("main");

const kernels = main.inference.backend.kernels;
const ShortConvKernel = kernels.ShortConvKernel;
const ShortConvConfig = kernels.ShortConvConfig;
const ShortConvWeights = kernels.ShortConvWeights;

test "ShortConvKernel type is accessible" {
    _ = ShortConvKernel;
}

test "ShortConvKernel is a struct" {
    const info = @typeInfo(ShortConvKernel);
    try std.testing.expect(info == .@"struct");
}

test "ShortConvKernel.init creates kernel with config" {
    // ShortConvKernel.init only stores references, doesn't allocate
    const config = ShortConvConfig{
        .d_model = 768,
        .d_conv = 3,
        .conv_dim = 768,
        .conv_dim_out = 768,
    };

    // Create minimal weight tensors (we just need the struct to exist)
    // In a real test, these would be properly allocated tensors
    var in_proj: main.Tensor = undefined;
    var conv1d_weight: main.Tensor = undefined;
    var out_proj: main.Tensor = undefined;

    const weights = ShortConvWeights{
        .in_proj = &in_proj,
        .conv1d_weight = &conv1d_weight,
        .out_proj = &out_proj,
    };

    // Use undefined for matmul functions since we're just testing init
    const kernel = ShortConvKernel.init(config, weights, undefined, undefined);

    // Verify config is stored correctly
    try std.testing.expectEqual(@as(u32, 768), kernel.config.d_model);
    try std.testing.expectEqual(@as(u32, 3), kernel.config.d_conv);
    try std.testing.expectEqual(@as(u32, 768), kernel.config.conv_dim);
}

test "ShortConvConfig has correct defaults" {
    const config = ShortConvConfig{
        .d_model = 512,
        .d_conv = 4,
        .conv_dim = 256,
        .conv_dim_out = 512,
    };

    // Check default value
    try std.testing.expectEqual(false, config.has_bias);
}

test "ShortConvConfig custom has_bias" {
    const config = ShortConvConfig{
        .d_model = 512,
        .d_conv = 4,
        .conv_dim = 256,
        .conv_dim_out = 512,
        .has_bias = true,
    };

    try std.testing.expectEqual(true, config.has_bias);
}

test "ShortConvWeights optional bias" {
    var in_proj: main.Tensor = undefined;
    var conv1d_weight: main.Tensor = undefined;
    var out_proj: main.Tensor = undefined;

    // Without bias
    const weights_no_bias = ShortConvWeights{
        .in_proj = &in_proj,
        .conv1d_weight = &conv1d_weight,
        .out_proj = &out_proj,
    };
    try std.testing.expectEqual(@as(?*const main.Tensor, null), weights_no_bias.conv1d_bias);

    // With bias
    var conv1d_bias: main.Tensor = undefined;
    const weights_with_bias = ShortConvWeights{
        .in_proj = &in_proj,
        .conv1d_weight = &conv1d_weight,
        .conv1d_bias = &conv1d_bias,
        .out_proj = &out_proj,
    };
    try std.testing.expect(weights_with_bias.conv1d_bias != null);
}

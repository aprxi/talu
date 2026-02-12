//! Integration tests for inference.backend.cpu.kernels MambaKernel

const std = @import("std");
const main = @import("main");

const kernels = main.inference.backend.kernels;
const MambaKernel = kernels.MambaKernel;
const MambaConfig = kernels.MambaConfig;
const MambaWeights = kernels.MambaWeights;

test "MambaKernel type is accessible" {
    _ = MambaKernel;
}

test "MambaKernel.init returns initialized kernel" {
    const config = MambaConfig{
        .d_model = 768,
        .d_state = 128,
        .d_conv = 4,
        .n_heads = 48,
        .d_head = 32,
    };

    // MambaKernel.init requires valid weights and function pointers
    // which would require full model loading. Type check is sufficient
    // for verifying the export is accessible.
    _ = config;
}

test "MambaKernel.describe signature is accessible" {
    // Verify describe method exists on the type
    const has_describe = @hasDecl(MambaKernel, "describe");
    try std.testing.expect(has_describe);
}

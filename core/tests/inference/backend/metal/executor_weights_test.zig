//! Integration checks for inference.backend.metal.executor.weights.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");

const has_metal = builtin.os.tag == .macos;
const weights = if (has_metal) main.inference.backend.metal.executor.weights else void;

test "weights module exposes lifecycle symbols" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(weights, "WeightHandles"));
    try std.testing.expect(@hasDecl(weights, "MLXError"));
    try std.testing.expect(@hasDecl(weights, "loadWeightsToGPU"));
    try std.testing.expect(@hasDecl(weights, "freeWeights"));
    try std.testing.expect(@hasDecl(weights, "createTestLoadedModel"));
    try std.testing.expect(@hasDecl(weights, "destroyTestLoadedModel"));
}

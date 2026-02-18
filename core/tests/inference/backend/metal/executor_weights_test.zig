//! Integration checks for inference.backend.metal.executor.weights.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const main = @import("main");

const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const weights = if (has_metal) main.inference.backend.metal.executor.weights else void;

test "weights module exposes lifecycle symbols" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(weights, "WeightHandles"));
    try std.testing.expect(@hasDecl(weights, "MLXError"));
    try std.testing.expect(@hasDecl(weights, "loadWeightsToGPU"));
    try std.testing.expect(@hasDecl(weights, "createFusedModel"));
    try std.testing.expect(@hasDecl(weights, "freeWeights"));
}

//! Integration checks for inference.backend.metal.executor.model.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const main = @import("main");

const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const Model = if (has_metal) main.inference.backend.metal.executor.model.Model else void;

test "Model.forward symbol exists" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(Model, "forward"));
}

test "Model.forwardWithEmbeddingOverride symbol exists" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(Model, "forwardWithEmbeddingOverride"));
}

test "Model.forwardFromGPUToken symbol exists" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(Model, "forwardFromGPUToken"));
}

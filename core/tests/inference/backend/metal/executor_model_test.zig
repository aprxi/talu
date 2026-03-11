//! Integration checks for inference.backend.metal.executor.model.

const std = @import("std");
const builtin = @import("builtin");
const main = @import("main");

const has_metal = builtin.os.tag == .macos;
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

//! Integration checks for inference.backend.metal.executor.block.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const main = @import("main");

const has_metal = build_options.enable_metal and builtin.os.tag == .macos;
const TransformerBlock = if (has_metal) main.inference.backend.metal.executor.block.TransformerBlock else void;

test "TransformerBlock.forward symbol exists" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(TransformerBlock, "forward"));
}

test "TransformerBlock.projectLogits symbol exists" {
    if (comptime !has_metal) return;
    try std.testing.expect(@hasDecl(TransformerBlock, "projectLogits"));
}

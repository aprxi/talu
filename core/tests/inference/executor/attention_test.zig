//! Integration tests for inference.executor.Attention
//!
//! Tests the Attention type from the executor module.

const std = @import("std");
const main = @import("main");

const Attention = main.inference.executor.Attention;

test "Attention type is accessible" {
    const T = Attention;
    _ = T;
}

test "Attention has expected structure" {
    const info = @typeInfo(Attention);
    try std.testing.expect(info == .@"struct");
}

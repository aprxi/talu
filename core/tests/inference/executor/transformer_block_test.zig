//! Integration tests for inference.executor.TransformerBlock
//!
//! Tests the TransformerBlock type from the executor module.

const std = @import("std");
const main = @import("main");

const TransformerBlock = main.inference.executor.TransformerBlock;

test "TransformerBlock type is accessible" {
    const T = TransformerBlock;
    _ = T;
}

test "TransformerBlock has expected structure" {
    const info = @typeInfo(TransformerBlock);
    try std.testing.expect(info == .@"struct");
}

//! Integration tests for inference.executor.FFNLayer
//!
//! Tests the FFNLayer type from the executor module.

const std = @import("std");
const main = @import("main");

const FFNLayer = main.inference.executor.FFNLayer;

test "FFNLayer type is accessible" {
    const T = FFNLayer;
    _ = T;
}

test "FFNLayer is a union" {
    const info = @typeInfo(FFNLayer);
    try std.testing.expect(info == .@"union");
}

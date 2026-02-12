//! Integration tests for inference.backend.cpu.kernels BatchedAttnTemp

const std = @import("std");
const main = @import("main");

const attention = main.inference.backend.kernels.attention;

test "attention module has BatchedAttnTemp" {
    const BatchedAttnTemp = attention.BatchedAttnTemp;
    _ = BatchedAttnTemp;
}

test "BatchedAttnTemp type is accessible" {
    const info = @typeInfo(attention.BatchedAttnTemp);
    try std.testing.expect(info == .@"struct");
}

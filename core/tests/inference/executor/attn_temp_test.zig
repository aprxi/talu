//! Integration tests for inference.executor.AttnTemp
//!
//! Tests the AttnTemp type from the executor module.

const std = @import("std");
const main = @import("main");

const AttnTemp = main.inference.executor.AttnTemp;

test "AttnTemp type is accessible" {
    const T = AttnTemp;
    _ = T;
}

test "AttnTemp initializes with empty slices" {
    const temp = AttnTemp{};
    try std.testing.expectEqual(@as(usize, 0), temp.q.len);
    try std.testing.expectEqual(@as(usize, 0), temp.k.len);
    try std.testing.expectEqual(@as(usize, 0), temp.v.len);
}

test "AttnTemp.deinit is safe on empty struct" {
    const allocator = std.testing.allocator;
    var temp = AttnTemp{};
    temp.deinit(allocator);
}

//! Integration tests for inference.backend.cpu.kernels AttnTemp

const std = @import("std");
const main = @import("main");

const attention = main.inference.backend.kernels.attention;

test "attention module is accessible" {
    _ = attention;
}

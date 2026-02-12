//! Integration tests for inference.backend.cpu.kernels AttnCache

const std = @import("std");
const main = @import("main");

const attention = main.inference.backend.kernels.attention;

test "attention module has expected exports" {
    _ = attention;
}

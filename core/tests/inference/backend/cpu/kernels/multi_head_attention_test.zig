//! Integration tests for inference.backend.cpu.kernels MultiHeadAttention

const std = @import("std");
const main = @import("main");

const attention = main.inference.backend.kernels.attention;

test "attention kernel module is accessible" {
    _ = attention;
}

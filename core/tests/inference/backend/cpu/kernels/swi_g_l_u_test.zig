//! Integration tests for inference.backend.cpu.kernels SwiGLU

const std = @import("std");
const main = @import("main");

const ffn = main.inference.backend.kernels.ffn;

test "ffn module with SwiGLU is accessible" {
    _ = ffn;
}

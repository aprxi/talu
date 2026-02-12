//! Integration tests for inference.backend.cpu.kernels FfnScratch

const std = @import("std");
const main = @import("main");

const ffn = main.inference.backend.kernels.ffn;

test "ffn module is accessible" {
    _ = ffn;
}

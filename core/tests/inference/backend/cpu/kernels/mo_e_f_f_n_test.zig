//! Integration tests for inference.backend.cpu.kernels MoEFFN

const std = @import("std");
const main = @import("main");

const moe = main.inference.backend.kernels.moe;

test "moe module is accessible" {
    _ = moe;
}

test "moe module has MoEFFN type" {
    const MoEFFN = moe.MoEFFN;
    _ = MoEFFN;
}

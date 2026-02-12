//! Integration tests for inference.backend.cpu.kernels MoEScratch

const std = @import("std");
const main = @import("main");

const moe = main.inference.backend.kernels.moe;

test "moe module has MoEScratch type" {
    const MoEScratch = moe.MoEScratch;
    _ = MoEScratch;
}

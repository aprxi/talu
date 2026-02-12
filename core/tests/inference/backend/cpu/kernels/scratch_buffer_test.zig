//! Integration tests for inference.backend.cpu.kernels ScratchBuffer

const std = @import("std");
const main = @import("main");

// ScratchBuffer is re-exported from block_kernels
const kernels = main.inference.backend.kernels;

test "kernels module is accessible" {
    _ = kernels;
}

test "kv_cache submodule is accessible" {
    _ = kernels.kv_cache;
}

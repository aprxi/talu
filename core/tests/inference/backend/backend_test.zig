//! Integration tests for inference.backend.Backend

const std = @import("std");
const main = @import("main");

const backend = main.inference.backend;

test "backend module is accessible" {
    _ = backend;
}

test "backend has cpu submodule" {
    _ = backend.cpu;
}

test "backend has kernels submodule" {
    _ = backend.kernels;
}

test "backend has block_kernels submodule" {
    _ = backend.block_kernels;
}

//! Integration tests for CUDA sideload cache helpers.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "sideload.artifactPath includes arch-specific file name" {
    const path = try cuda.sideload.artifactPath(std.testing.allocator, "/tmp/talu-cuda", "sm_89");
    defer std.testing.allocator.free(path);

    try std.testing.expect(std.mem.endsWith(u8, path, "kernels_sm_89.cubin"));
}

test "sideload.verifySha256 validates digest" {
    try cuda.sideload.verifySha256("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

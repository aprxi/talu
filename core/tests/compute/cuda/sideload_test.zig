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

test "sideload.loadOrFetchManifest uses cached file when present" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const cache_dir = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(cache_dir);

    const manifest_path = try cuda.sideload.manifestPath(allocator, cache_dir, "sm_89");
    defer allocator.free(manifest_path);

    const manifest_json =
        \\{
        \\  "schema_version": 1,
        \\  "kernel_abi_version": 1,
        \\  "arch": "sm_89",
        \\  "driver_min": "550.00",
        \\  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        \\  "kernels": [
        \\    {"op":"rmsnorm_f32","symbol":"talu_rmsnorm_f32_v1"}
        \\  ]
        \\}
    ;
    try cuda.sideload.writeCachedArtifact(manifest_path, manifest_json);

    const loaded = try cuda.sideload.loadOrFetchManifest(
        allocator,
        cache_dir,
        "sm_89",
        "https://invalid.local",
    );
    defer allocator.free(loaded);

    try std.testing.expectEqualStrings(manifest_json, loaded);
}

//! Integration tests for CUDA sideload manifest parsing.

const std = @import("std");
const main = @import("main");
const cuda = main.core.compute.cuda;

test "manifest.parse parses v1 schema" {
    const bytes =
        \\{
        \\  "schema_version": 1,
        \\  "kernel_abi_version": 1,
        \\  "arch": "sm_89",
        \\  "driver_min": "550.00",
        \\  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        \\  "kernels": [
        \\    {"op":"vector_add_f32","symbol":"talu_vector_add_f32"}
        \\  ]
        \\}
    ;

    var parsed = try cuda.manifest.parse(std.testing.allocator, bytes);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("talu_vector_add_f32", parsed.manifest.findSymbol("vector_add_f32").?);
}

test "manifest.validate rejects unsupported schema" {
    const manifest = cuda.manifest.Manifest{
        .schema_version = 2,
        .kernel_abi_version = 1,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .kernels = &.{.{ .op = "x", .symbol = "y" }},
    };
    try std.testing.expectError(error.UnsupportedManifestSchema, cuda.manifest.validate(manifest));
}

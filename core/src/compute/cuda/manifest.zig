//! Manifest schema for sideloaded CUDA kernel payloads.

const std = @import("std");

pub const KernelEntry = struct {
    op: []const u8,
    symbol: []const u8,
};

pub const Manifest = struct {
    schema_version: u32,
    kernel_abi_version: u32,
    arch: []const u8,
    driver_min: []const u8,
    sha256: []const u8,
    kernels: []const KernelEntry,

    pub fn findSymbol(self: *const Manifest, op: []const u8) ?[]const u8 {
        for (self.kernels) |kernel| {
            if (std.mem.eql(u8, kernel.op, op)) return kernel.symbol;
        }
        return null;
    }
};

pub const ParsedManifest = struct {
    arena: std.heap.ArenaAllocator,
    manifest: Manifest,

    pub fn deinit(self: *ParsedManifest) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !ParsedManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();

    const parsed = try std.json.parseFromSliceLeaky(Manifest, arena.allocator(), bytes, .{
        .ignore_unknown_fields = false,
    });
    try validate(parsed);

    return .{
        .arena = arena,
        .manifest = parsed,
    };
}

pub fn validate(manifest: Manifest) !void {
    if (manifest.schema_version != 1) return error.UnsupportedManifestSchema;
    if (manifest.kernel_abi_version == 0) return error.InvalidKernelAbiVersion;
    if (manifest.arch.len == 0) return error.InvalidArch;
    if (manifest.driver_min.len == 0) return error.InvalidDriverConstraint;
    if (manifest.sha256.len != 64) return error.InvalidSha256;
    if (!isHexDigest(manifest.sha256)) return error.InvalidSha256;
    if (manifest.kernels.len == 0) return error.InvalidKernelTable;

    for (manifest.kernels) |kernel| {
        if (kernel.op.len == 0 or kernel.symbol.len == 0) return error.InvalidKernelTable;
    }
}

fn isHexDigest(digest: []const u8) bool {
    for (digest) |ch| {
        if (!std.ascii.isHex(ch)) return false;
    }
    return true;
}

test "parse validates v1 manifest payload" {
    const json =
        \\{
        \\  "schema_version": 1,
        \\  "kernel_abi_version": 1,
        \\  "arch": "sm_89",
        \\  "driver_min": "550.00",
        \\  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        \\  "kernels": [
        \\    {"op":"rmsnorm_f32","symbol":"talu_rmsnorm_f32"}
        \\  ]
        \\}
    ;

    var parsed = try parse(std.testing.allocator, json);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u32, 1), parsed.manifest.schema_version);
    try std.testing.expectEqualStrings("talu_rmsnorm_f32", parsed.manifest.findSymbol("rmsnorm_f32").?);
}

test "validate rejects non-hex sha256" {
    const manifest = Manifest{
        .schema_version = 1,
        .kernel_abi_version = 1,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
        .kernels = &.{.{ .op = "x", .symbol = "y" }},
    };

    try std.testing.expectError(error.InvalidSha256, validate(manifest));
}

test "Manifest.findSymbol returns null when op is missing" {
    const manifest = Manifest{
        .schema_version = 1,
        .kernel_abi_version = 1,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .kernels = &.{.{ .op = "a", .symbol = "b" }},
    };
    try std.testing.expect(manifest.findSymbol("missing") == null);
}

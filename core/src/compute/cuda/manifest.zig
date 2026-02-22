//! Manifest schema for sideloaded CUDA kernel payloads.

const std = @import("std");

pub const schema_version: u32 = 1;
pub const kernel_abi_version: u32 = 1;

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
    if (manifest.schema_version != schema_version) return error.UnsupportedManifestSchema;
    if (manifest.kernel_abi_version == 0) return error.InvalidKernelAbiVersion;
    if (manifest.arch.len == 0) return error.InvalidArch;
    if (manifest.driver_min.len == 0) return error.InvalidDriverConstraint;
    if (manifest.sha256.len != 64) return error.InvalidSha256;
    if (!isHexDigest(manifest.sha256)) return error.InvalidSha256;
    if (manifest.kernels.len == 0) return error.InvalidKernelTable;

    for (manifest.kernels, 0..) |kernel, i| {
        if (kernel.op.len == 0 or kernel.symbol.len == 0) return error.InvalidKernelTable;
        if (!isKernelOpNameCanonical(kernel.op)) return error.InvalidKernelTable;
        if (!isKernelSymbolCanonical(kernel.symbol)) return error.InvalidKernelTable;

        var j: usize = 0;
        while (j < i) : (j += 1) {
            if (std.mem.eql(u8, kernel.op, manifest.kernels[j].op)) return error.InvalidKernelTable;
        }
    }
}

pub fn ensureCompatible(manifest: Manifest, expected_arch: []const u8, expected_kernel_abi_version: u32) !void {
    if (!std.mem.eql(u8, manifest.arch, expected_arch)) return error.CudaManifestArchMismatch;
    if (manifest.kernel_abi_version != expected_kernel_abi_version) return error.CudaManifestKernelAbiMismatch;
}

fn isHexDigest(digest: []const u8) bool {
    for (digest) |ch| {
        if (!std.ascii.isHex(ch)) return false;
    }
    return true;
}

fn isKernelOpNameCanonical(op_name: []const u8) bool {
    if (op_name.len == 0) return false;
    for (op_name) |ch| {
        if (!(std.ascii.isLower(ch) or std.ascii.isDigit(ch) or ch == '_')) return false;
    }
    return !hasVersionSuffix(op_name);
}

fn isKernelSymbolCanonical(symbol: []const u8) bool {
    if (!std.mem.startsWith(u8, symbol, "talu_")) return false;
    for (symbol) |ch| {
        if (!(std.ascii.isLower(ch) or std.ascii.isDigit(ch) or ch == '_')) return false;
    }
    return !hasVersionSuffix(symbol);
}

fn hasVersionSuffix(name: []const u8) bool {
    const marker = "_v";
    const at = std.mem.lastIndexOf(u8, name, marker) orelse return false;
    const digits = name[at + marker.len ..];
    if (digits.len == 0) return false;
    for (digits) |ch| {
        if (!std.ascii.isDigit(ch)) return false;
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

test "ensureCompatible rejects mismatched arch" {
    const manifest = Manifest{
        .schema_version = schema_version,
        .kernel_abi_version = kernel_abi_version,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .kernels = &.{.{ .op = "a", .symbol = "b" }},
    };

    try std.testing.expectError(
        error.CudaManifestArchMismatch,
        ensureCompatible(manifest, "sm_90", kernel_abi_version),
    );
}

test "ensureCompatible rejects mismatched kernel abi version" {
    const manifest = Manifest{
        .schema_version = schema_version,
        .kernel_abi_version = kernel_abi_version,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .kernels = &.{.{ .op = "a", .symbol = "b" }},
    };

    try std.testing.expectError(
        error.CudaManifestKernelAbiMismatch,
        ensureCompatible(manifest, "sm_89", kernel_abi_version + 1),
    );
}

test "validate rejects duplicate kernel op entries" {
    const manifest = Manifest{
        .schema_version = schema_version,
        .kernel_abi_version = kernel_abi_version,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .kernels = &.{
            .{ .op = "rmsnorm_f32", .symbol = "talu_rmsnorm_f32" },
            .{ .op = "rmsnorm_f32", .symbol = "talu_rmsnorm_f32_alt" },
        },
    };
    try std.testing.expectError(error.InvalidKernelTable, validate(manifest));
}

test "validate rejects symbol version suffix" {
    const manifest = Manifest{
        .schema_version = schema_version,
        .kernel_abi_version = kernel_abi_version,
        .arch = "sm_89",
        .driver_min = "550.00",
        .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        .kernels = &.{.{ .op = "rmsnorm_f32", .symbol = "talu_rmsnorm_f32_v2" }},
    };
    try std.testing.expectError(error.InvalidKernelTable, validate(manifest));
}

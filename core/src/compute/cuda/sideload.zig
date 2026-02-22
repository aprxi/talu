//! Sideloaded CUDA kernel payload cache and download helpers.

const std = @import("std");
const transport = @import("../../io/transport/root.zig");

pub const cache_dir_env = "TALU_CUDA_CACHE_DIR";

pub fn resolveCacheDir(allocator: std.mem.Allocator) ![]u8 {
    const override = std.process.getEnvVarOwned(allocator, cache_dir_env) catch null;
    if (override) |path| {
        if (path.len == 0) {
            allocator.free(path);
            return error.InvalidCacheDir;
        }
        return path;
    }

    const app_dir = try std.fs.getAppDataDir(allocator, "talu");
    defer allocator.free(app_dir);
    return std.fmt.allocPrint(allocator, "{s}{c}cuda", .{ app_dir, std.fs.path.sep });
}

pub fn artifactFileName(allocator: std.mem.Allocator, arch: []const u8) ![]u8 {
    if (arch.len == 0) return error.InvalidArgument;
    return std.fmt.allocPrint(allocator, "kernels_{s}.cubin", .{arch});
}

pub fn artifactPath(allocator: std.mem.Allocator, cache_dir: []const u8, arch: []const u8) ![]u8 {
    const file_name = try artifactFileName(allocator, arch);
    defer allocator.free(file_name);
    return std.fs.path.join(allocator, &.{ cache_dir, file_name });
}

pub fn ensureCacheDir(cache_dir: []const u8) !void {
    try std.fs.cwd().makePath(cache_dir);
}

pub fn readCachedArtifact(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(allocator, path, 256 * 1024 * 1024);
}

pub fn writeCachedArtifact(path: []const u8, bytes: []const u8) !void {
    const dir_path = std.fs.path.dirname(path) orelse return error.InvalidArgument;
    try ensureCacheDir(dir_path);

    const tmp_path = try std.fmt.allocPrint(std.heap.page_allocator, "{s}.tmp", .{path});
    defer std.heap.page_allocator.free(tmp_path);

    {
        const tmp_file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true, .read = false });
        defer tmp_file.close();
        try tmp_file.writeAll(bytes);
        try tmp_file.sync();
    }
    try std.fs.cwd().rename(tmp_path, path);
}

pub fn verifySha256(bytes: []const u8, expected_hex: []const u8) !void {
    if (expected_hex.len != 64) return error.InvalidSha256;
    const digest = sha256(bytes);
    const actual_hex = bytesToHex(&digest);
    if (!std.ascii.eqlIgnoreCase(actual_hex[0..], expected_hex)) return error.Sha256Mismatch;
}

pub fn loadOrFetchArtifact(
    allocator: std.mem.Allocator,
    cache_dir: []const u8,
    arch: []const u8,
    base_url: []const u8,
    expected_sha256: []const u8,
) ![]u8 {
    const path = try artifactPath(allocator, cache_dir, arch);
    defer allocator.free(path);

    const cached = readCachedArtifact(allocator, path) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    if (cached) |bytes| {
        verifySha256(bytes, expected_sha256) catch |err| {
            allocator.free(bytes);
            if (err != error.Sha256Mismatch) return err;
            std.fs.cwd().deleteFile(path) catch {};
            return loadOrFetchArtifact(allocator, cache_dir, arch, base_url, expected_sha256);
        };
        return bytes;
    }

    const file_name = try artifactFileName(allocator, arch);
    defer allocator.free(file_name);
    const url = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_url, file_name });
    defer allocator.free(url);

    transport.http.globalInit();
    defer transport.http.globalCleanup();

    const downloaded = try transport.http.fetch(allocator, url, .{});
    errdefer allocator.free(downloaded);
    try verifySha256(downloaded, expected_sha256);
    try writeCachedArtifact(path, downloaded);
    return downloaded;
}

fn sha256(bytes: []const u8) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return digest;
}

fn bytesToHex(digest: *const [32]u8) [64]u8 {
    const lut = "0123456789abcdef";
    var out: [64]u8 = undefined;
    var i: usize = 0;
    while (i < digest.len) : (i += 1) {
        const byte = digest[i];
        out[i * 2] = lut[byte >> 4];
        out[i * 2 + 1] = lut[byte & 0x0f];
    }
    return out;
}

test "artifactPath includes architecture suffix" {
    const path = try artifactPath(std.testing.allocator, "/tmp/talu-cuda", "sm_89");
    defer std.testing.allocator.free(path);

    try std.testing.expect(std.mem.endsWith(u8, path, "kernels_sm_89.cubin"));
}

test "verifySha256 accepts matching digest" {
    try verifySha256("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

test "verifySha256 rejects wrong digest" {
    try std.testing.expectError(
        error.Sha256Mismatch,
        verifySha256("abc", "0000000000000000000000000000000000000000000000000000000000000000"),
    );
}

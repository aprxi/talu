//! Model Path Resolution
//!
//! Resolves any model path to a Bundle. Handles:
//! - Direct paths to model directories
//! - HF cache format (models--org--name/snapshots/...)

const std = @import("std");
const Bundle = @import("bundle.zig").Bundle;
const cache = @import("cache.zig");


pub const ResolveOptions = struct {
    require_weights: bool = true,
};

pub const ResolveError = error{
    ConfigNotFound,
    WeightsNotFound,
    OutOfMemory,
    AccessDenied,
    Unexpected,
};

/// Resolve any model path to a Bundle.
///
/// Handles: direct paths, HF cache format.
/// When `options.require_weights` is false, returns a Bundle with `.none` weights
/// instead of failing when no weight files are found.
pub fn resolve(allocator: std.mem.Allocator, input_path: []const u8, options: ResolveOptions) ResolveError!Bundle {
    // Resolve HF cache format (has snapshots/) or direct directory
    const resolved_dir_path = resolveSnapshot(allocator, input_path) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.AccessDenied => return error.AccessDenied,
        error.Unexpected => return error.Unexpected,
        else => return error.WeightsNotFound,
    };

    // Find weights file (best-effort when not required)
    const weights_path_result = findWeightsFile(allocator, resolved_dir_path) catch |err| {
        allocator.free(resolved_dir_path);
        return err;
    };
    if (weights_path_result) |path| {
        return buildBundleFromDir(allocator, resolved_dir_path, path) catch |err| {
            allocator.free(resolved_dir_path);
            allocator.free(path);
            return err;
        };
    }

    // No weights found
    if (options.require_weights) {
        allocator.free(resolved_dir_path);
        return error.WeightsNotFound;
    }

    // Weight-optional: build bundle without weights
    return buildBundleFromDirNoWeights(allocator, resolved_dir_path) catch |err| {
        allocator.free(resolved_dir_path);
        return err;
    };
}

/// Resolve a directory with SafeTensors model.
/// Takes ownership of `dir` and `weights_path` (will be freed by Bundle.deinit).
fn buildBundleFromDir(allocator: std.mem.Allocator, dir: []const u8, weights_path: []const u8) ResolveError!Bundle {
    const config_path = std.fs.path.join(allocator, &.{ dir, "config.json" }) catch return error.OutOfMemory;
    errdefer allocator.free(config_path);

    std.fs.cwd().access(config_path, .{}) catch return error.ConfigNotFound;

    const tokenizer_path = std.fs.path.join(allocator, &.{ dir, "tokenizer.json" }) catch return error.OutOfMemory;
    errdefer allocator.free(tokenizer_path);

    // Check if tokenizer exists
    const tokenizer: Bundle.TokenizerSource = if (std.fs.cwd().access(tokenizer_path, .{})) |_|
        .{ .path = tokenizer_path }
    else |_| blk: {
        allocator.free(tokenizer_path);
        break :blk .{ .none = {} };
    };

    // Detect format (MLX vs standard SafeTensors)
    const format = detectModelFormat(dir);

    // For sharded weights, we need separate allocations for index_path and shard_dir
    // since Bundle.deinit() frees them independently
    const weights: Bundle.WeightsSource = if (std.mem.endsWith(u8, weights_path, ".index.json")) blk: {
        // Duplicate dir for shard_dir since we also use dir for Bundle.dir
        const shard_dir = allocator.dupe(u8, dir) catch return error.OutOfMemory;
        errdefer allocator.free(shard_dir);
        break :blk .{ .sharded = .{ .index_path = weights_path, .shard_dir = shard_dir } };
    } else .{ .single = weights_path };

    return Bundle{
        .allocator = allocator,
        .dir = dir,
        .config = .{ .path = config_path },
        .weights = weights,
        .tokenizer = tokenizer,
        .format = format,
    };
}

/// Build a Bundle from a directory without requiring weights or config.
/// Used for tokenizer-only resolution. Takes ownership of `dir`.
fn buildBundleFromDirNoWeights(allocator: std.mem.Allocator, dir: []const u8) ResolveError!Bundle {
    const tokenizer_path = std.fs.path.join(allocator, &.{ dir, "tokenizer.json" }) catch return error.OutOfMemory;
    errdefer allocator.free(tokenizer_path);

    const tokenizer: Bundle.TokenizerSource = if (std.fs.cwd().access(tokenizer_path, .{})) |_|
        .{ .path = tokenizer_path }
    else |_| blk: {
        allocator.free(tokenizer_path);
        break :blk .{ .none = {} };
    };

    // Config is best-effort: present if exists, placeholder if absent
    const config_path = std.fs.path.join(allocator, &.{ dir, "config.json" }) catch return error.OutOfMemory;
    const config: Bundle.ConfigSource = if (std.fs.cwd().access(config_path, .{})) |_|
        .{ .path = config_path }
    else |_| blk: {
        allocator.free(config_path);
        break :blk .{ .json = allocator.dupe(u8, "{}") catch return error.OutOfMemory };
    };

    const format = detectModelFormat(dir);

    return Bundle{
        .allocator = allocator,
        .dir = dir,
        .config = config,
        .weights = .{ .none = {} },
        .tokenizer = tokenizer,
        .format = format,
    };
}

/// Detect if a directory contains MLX-format model.
fn detectModelFormat(dir: []const u8) Bundle.Format {
    // MLX models typically have "MLX" in the directory name
    const basename = std.fs.path.basename(dir);
    if (std.mem.indexOf(u8, basename, "-MLX") != null or
        std.mem.indexOf(u8, basename, "_MLX") != null)
    {
        return .mlx;
    }
    return .safetensors;
}

/// Resolve HF cache snapshot directory.
/// If path has snapshots/, resolves to the best snapshot.
/// Otherwise returns path as-is.
pub fn resolveSnapshot(allocator: std.mem.Allocator, base_path: []const u8) ResolveError![]const u8 {
    const snapshots_path = try std.fs.path.join(allocator, &.{ base_path, "snapshots" });
    defer allocator.free(snapshots_path);

    var snapshots_dir = std.fs.cwd().openDir(snapshots_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return allocator.dupe(u8, base_path),
        error.AccessDenied => return error.AccessDenied,
        else => return error.Unexpected,
    };
    defer snapshots_dir.close();

    // Try refs/main first (HF cache stores current revision hash there)
    if (try readRefsMain(allocator, base_path)) |snapshot_hash| {
        const snapshot_path = try std.fs.path.join(allocator, &.{ base_path, "snapshots", snapshot_hash });
        allocator.free(snapshot_hash);

        if (isDirectory(snapshot_path)) {
            return snapshot_path;
        }
        allocator.free(snapshot_path);
    }

    // No refs/main found - scan snapshots directory (prefer 40-char hex hashes)
    var snapshot_iter = snapshots_dir.iterate();
    var fallback_snapshot_path: ?[]const u8 = null;

    while (snapshot_iter.next() catch |err| switch (err) {
        error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
        error.SystemResources, error.InvalidUtf8 => return error.Unexpected,
        else => return error.Unexpected,
    }) |entry| {
        if (entry.kind != .directory) continue;

        const snapshot_path = try std.fs.path.join(allocator, &.{ base_path, "snapshots", entry.name });

        // Prefer hex40 hashes (git commit format)
        if (isHex40(entry.name)) {
            if (fallback_snapshot_path) |p| allocator.free(p);
            return snapshot_path;
        }

        if (fallback_snapshot_path == null) {
            fallback_snapshot_path = snapshot_path;
        } else {
            allocator.free(snapshot_path);
        }
    }

    if (fallback_snapshot_path) |p| return p;
    return allocator.dupe(u8, base_path);
}

/// Read refs/main to get the preferred snapshot hash.
fn readRefsMain(allocator: std.mem.Allocator, base_path: []const u8) ResolveError!?[]const u8 {
    const refs_main_path = try std.fs.path.join(allocator, &.{ base_path, "refs", "main" });
    defer allocator.free(refs_main_path);

    const ref_contents = std.fs.cwd().readFileAlloc(allocator, refs_main_path, 128) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return null,
        error.AccessDenied => return error.AccessDenied,
        else => return error.Unexpected,
    };
    defer allocator.free(ref_contents);

    const hash_str = std.mem.trim(u8, ref_contents, " \t\r\n");
    if (!isHex40(hash_str)) return null;

    return allocator.dupe(u8, hash_str) catch return error.OutOfMemory;
}

/// Check if path is a valid directory.
fn isDirectory(path: []const u8) bool {
    var cwd_dir = std.fs.cwd().openDir(path, .{}) catch return false;
    cwd_dir.close();
    return true;
}

/// Check if string is a 40-character hex string (git hash).
fn isHex40(s: []const u8) bool {
    if (s.len != 40) return false;
    for (s) |c| {
        if (!std.ascii.isHex(c)) return false;
    }
    return true;
}

/// Find the weights file in a directory.
/// Returns the path or null if not found.
pub fn findWeightsFile(allocator: std.mem.Allocator, dir_path: []const u8) ResolveError!?[]const u8 {
    // Priority order: index file, then common names, then any safetensors
    const preferred_names = [_][]const u8{
        "model.safetensors.index.json",
        "model.safetensors",
        "weights.safetensors",
        "pytorch_model.safetensors",
    };

    for (preferred_names) |name| {
        const candidate_path = std.fs.path.join(allocator, &.{ dir_path, name }) catch return error.OutOfMemory;
        if (std.fs.cwd().access(candidate_path, .{})) |_| {
            return candidate_path;
        } else |err| {
            allocator.free(candidate_path);
            if (err == error.AccessDenied) return error.AccessDenied;
        }
    }

    // No standard filename found - scan directory for .safetensors files
    var dir_handle = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return null,
        error.AccessDenied => return error.AccessDenied,
        else => return error.Unexpected,
    };
    defer dir_handle.close();

    var dir_iter = dir_handle.iterate();
    while (dir_iter.next() catch |err| switch (err) {
        error.AccessDenied => return error.AccessDenied,
        else => return error.Unexpected,
    }) |entry| {
        if (entry.kind != .file) continue;

        // Skip sharded parts (model-00001-of-00003.safetensors)
        if (std.mem.indexOf(u8, entry.name, "-of-") != null) continue;

        if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const safetensors_path = std.fs.path.join(allocator, &.{ dir_path, entry.name }) catch return error.OutOfMemory;
            return safetensors_path;
        }
    }

    return null;
}

// =============================================================================
// Tests
// =============================================================================

test "isHex40: validates hex strings" {
    try std.testing.expect(isHex40("0123456789abcdef0123456789abcdef01234567"));
    try std.testing.expect(!isHex40("short"));
    try std.testing.expect(!isHex40("0123456789abcdef0123456789abcdef0123456g"));
}

test "detectModelFormat: detects MLX and safetensors formats" {
    try std.testing.expectEqual(Bundle.Format.mlx, detectModelFormat("/path/to/model-MLX-4bit"));
    try std.testing.expectEqual(Bundle.Format.mlx, detectModelFormat("/path/to/model_MLX"));
    try std.testing.expectEqual(Bundle.Format.safetensors, detectModelFormat("/path/to/model"));
}

test "findWeightsFile finds model.safetensors" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create model.safetensors
    const weights_filename = "model.safetensors";
    const weights_path = try std.fs.path.join(allocator, &.{ temp_dir_path, weights_filename });
    defer allocator.free(weights_path);

    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const found_path = try findWeightsFile(allocator, temp_dir_path);
    try std.testing.expect(found_path != null);
    defer allocator.free(found_path.?);

    try std.testing.expect(std.mem.endsWith(u8, found_path.?, weights_filename));
}

test "findWeightsFile finds model.safetensors.index.json" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create index file
    const index_filename = "model.safetensors.index.json";
    const index_path = try std.fs.path.join(allocator, &.{ temp_dir_path, index_filename });
    defer allocator.free(index_path);

    {
        var file = try std.fs.cwd().createFile(index_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const found_path = try findWeightsFile(allocator, temp_dir_path);
    try std.testing.expect(found_path != null);
    defer allocator.free(found_path.?);

    try std.testing.expect(std.mem.endsWith(u8, found_path.?, index_filename));
}

test "findWeightsFile finds weights.safetensors when model.safetensors missing" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create weights.safetensors
    const weights_filename = "weights.safetensors";
    const weights_path = try std.fs.path.join(allocator, &.{ temp_dir_path, weights_filename });
    defer allocator.free(weights_path);

    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const found_path = try findWeightsFile(allocator, temp_dir_path);
    try std.testing.expect(found_path != null);
    defer allocator.free(found_path.?);

    try std.testing.expect(std.mem.endsWith(u8, found_path.?, weights_filename));
}

test "findWeightsFile finds any .safetensors file" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create custom-named safetensors file
    const custom_filename = "custom_weights.safetensors";
    const custom_path = try std.fs.path.join(allocator, &.{ temp_dir_path, custom_filename });
    defer allocator.free(custom_path);

    {
        var file = try std.fs.cwd().createFile(custom_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const found_path = try findWeightsFile(allocator, temp_dir_path);
    try std.testing.expect(found_path != null);
    defer allocator.free(found_path.?);

    try std.testing.expect(std.mem.endsWith(u8, found_path.?, ".safetensors"));
}

test "findWeightsFile skips sharded parts (model-00001-of-00003.safetensors)" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create sharded files (should be skipped)
    const shard1_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model-00001-of-00003.safetensors" });
    defer allocator.free(shard1_path);
    {
        var file = try std.fs.cwd().createFile(shard1_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const shard2_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model-00002-of-00003.safetensors" });
    defer allocator.free(shard2_path);
    {
        var file = try std.fs.cwd().createFile(shard2_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    // Create a valid non-sharded file
    const model_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model.safetensors" });
    defer allocator.free(model_path);
    {
        var file = try std.fs.cwd().createFile(model_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const found_path = try findWeightsFile(allocator, temp_dir_path);
    try std.testing.expect(found_path != null);
    defer allocator.free(found_path.?);

    // Should find model.safetensors, not the sharded parts
    try std.testing.expect(std.mem.endsWith(u8, found_path.?, "model.safetensors"));
    try std.testing.expect(std.mem.indexOf(u8, found_path.?, "-of-") == null);
}

test "findWeightsFile returns null for directory without weights" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create some non-weights files
    const config_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const found_path = try findWeightsFile(allocator, temp_dir_path);
    try std.testing.expect(found_path == null);
}

test "resolveSnapshot returns base path when no snapshots directory" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    const resolved = try resolveSnapshot(allocator, temp_dir_path);
    defer allocator.free(resolved);

    try std.testing.expectEqualStrings(temp_dir_path, resolved);
}

test "resolveSnapshot finds snapshot from refs/main" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create snapshots directory
    const snapshots_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "snapshots" });
    defer allocator.free(snapshots_path);
    try std.fs.cwd().makePath(snapshots_path);

    // Create snapshot directory
    const snapshot_hash = "1234567890abcdef1234567890abcdef12345678";
    const snapshot_dir = try std.fs.path.join(allocator, &.{ snapshots_path, snapshot_hash });
    defer allocator.free(snapshot_dir);
    try std.fs.cwd().makePath(snapshot_dir);

    // Create refs/main pointing to this snapshot
    const refs_dir = try std.fs.path.join(allocator, &.{ temp_dir_path, "refs" });
    defer allocator.free(refs_dir);
    try std.fs.cwd().makePath(refs_dir);

    const refs_main_path = try std.fs.path.join(allocator, &.{ refs_dir, "main" });
    defer allocator.free(refs_main_path);
    {
        var file = try std.fs.cwd().createFile(refs_main_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll(snapshot_hash);
    }

    const resolved = try resolveSnapshot(allocator, temp_dir_path);
    defer allocator.free(resolved);

    try std.testing.expect(std.mem.endsWith(u8, resolved, snapshot_hash));
}

test "resolveSnapshot prefers hex40 hash over other names" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create snapshots directory
    const snapshots_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "snapshots" });
    defer allocator.free(snapshots_path);
    try std.fs.cwd().makePath(snapshots_path);

    // Create "main" snapshot (non-hex40)
    const main_dir = try std.fs.path.join(allocator, &.{ snapshots_path, "main" });
    defer allocator.free(main_dir);
    try std.fs.cwd().makePath(main_dir);

    // Create hex40 snapshot
    const snapshot_hash = "abcdef1234567890abcdef1234567890abcdef12";
    const hex40_dir = try std.fs.path.join(allocator, &.{ snapshots_path, snapshot_hash });
    defer allocator.free(hex40_dir);
    try std.fs.cwd().makePath(hex40_dir);

    const resolved = try resolveSnapshot(allocator, temp_dir_path);
    defer allocator.free(resolved);

    // Should prefer hex40 hash over "main"
    try std.testing.expect(std.mem.endsWith(u8, resolved, snapshot_hash));
}

test "resolveSnapshot falls back to first snapshot if no hex40" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create snapshots directory
    const snapshots_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "snapshots" });
    defer allocator.free(snapshots_path);
    try std.fs.cwd().makePath(snapshots_path);

    // Create "main" snapshot (only non-hex40)
    const main_dir = try std.fs.path.join(allocator, &.{ snapshots_path, "main" });
    defer allocator.free(main_dir);
    try std.fs.cwd().makePath(main_dir);

    const resolved = try resolveSnapshot(allocator, temp_dir_path);
    defer allocator.free(resolved);

    // Should fall back to "main" snapshot
    try std.testing.expect(std.mem.endsWith(u8, resolved, "main"));
}

test "resolve creates Bundle from direct model directory" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create required files
    const config_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const weights_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const tokenizer_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "tokenizer.json" });
    defer allocator.free(tokenizer_path);
    {
        var file = try std.fs.cwd().createFile(tokenizer_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    var bundle = try resolve(allocator, temp_dir_path, .{});
    defer bundle.deinit();

    try std.testing.expect(std.mem.endsWith(u8, bundle.dir, std.fs.path.basename(temp_dir_path)));
    try std.testing.expect(!bundle.isSharded());
}

test "resolve creates Bundle from HF cache format" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create HF cache structure
    const snapshot_hash = "abc123def456abc123def456abc123def456abc1";
    const snapshot_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "snapshots", snapshot_hash });
    defer allocator.free(snapshot_path);
    try std.fs.cwd().makePath(snapshot_path);

    // Create required files in snapshot
    const config_path = try std.fs.path.join(allocator, &.{ snapshot_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot_path, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    var bundle = try resolve(allocator, temp_dir_path, .{});
    defer bundle.deinit();

    try std.testing.expect(std.mem.indexOf(u8, bundle.dir, "snapshots") != null);
    try std.testing.expect(std.mem.indexOf(u8, bundle.dir, snapshot_hash) != null);
}

test "resolve returns error for directory without config" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create weights but no config
    const weights_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const result = resolve(allocator, temp_dir_path, .{});
    try std.testing.expectError(error.ConfigNotFound, result);
}

test "resolve returns error for directory without weights" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create config but no weights
    const config_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const result = resolve(allocator, temp_dir_path, .{});
    try std.testing.expectError(error.WeightsNotFound, result);
}

test "resolve handles MLX format detection" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create MLX-named directory
    const mlx_dir_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model-MLX-4bit" });
    defer allocator.free(mlx_dir_path);
    try std.fs.cwd().makePath(mlx_dir_path);

    const config_path = try std.fs.path.join(allocator, &.{ mlx_dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const weights_path = try std.fs.path.join(allocator, &.{ mlx_dir_path, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    var bundle = try resolve(allocator, mlx_dir_path, .{});
    defer bundle.deinit();

    try std.testing.expectEqual(Bundle.Format.mlx, bundle.format);
}

test "resolve handles sharded weights with index.json" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create required files
    const config_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const index_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "model.safetensors.index.json" });
    defer allocator.free(index_path);
    {
        var file = try std.fs.cwd().createFile(index_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    var bundle = try resolve(allocator, temp_dir_path, .{});
    defer bundle.deinit();

    try std.testing.expect(bundle.isSharded());
}

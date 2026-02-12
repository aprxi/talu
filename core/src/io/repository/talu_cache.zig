//! Talu Local Cache
//!
//! Scans ~/.cache/talu/models/ (or $TALU_HOME/models/) for converted models.
//! Models are stored as org/name subdirectories containing config.json + weights.

const std = @import("std");
const log = @import("../../log.zig");
const cache = @import("cache.zig");
const resolver = @import("resolver.zig");

const CachedModel = cache.CachedModel;
const CacheOrigin = cache.CacheOrigin;
const CacheError = cache.CacheError;
const ListOptions = cache.ListOptions;

/// Get the Talu home directory.
/// Resolution: $TALU_HOME > ~/.cache/talu
pub fn getTaluHome(allocator: std.mem.Allocator) CacheError![]const u8 {
    if (std.posix.getenv("TALU_HOME")) |talu_home| {
        return allocator.dupe(u8, talu_home) catch return CacheError.OutOfMemory;
    }
    const home_dir = std.posix.getenv("HOME") orelse return CacheError.NoHomeDir;
    return std.fs.path.join(allocator, &.{ home_dir, ".cache", "talu" }) catch return CacheError.OutOfMemory;
}

/// Get the Talu models directory ($TALU_HOME/models).
pub fn getTaluModelsDir(allocator: std.mem.Allocator) CacheError![]const u8 {
    const talu_home = try getTaluHome(allocator);
    defer allocator.free(talu_home);
    return std.fs.path.join(allocator, &.{ talu_home, "models" }) catch return CacheError.OutOfMemory;
}

/// Look up a single model in the Talu local cache by model ID.
///
/// Given "Org/Model", checks if $TALU_HOME/models/Org/Model exists and has weights.
/// Returns the directory path if found, null otherwise. Caller owns returned memory.
pub fn getTaluCachedPath(allocator: std.mem.Allocator, model_id: []const u8) !?[]const u8 {
    const models_dir = getTaluModelsDir(allocator) catch |err| switch (err) {
        CacheError.NoHomeDir, CacheError.OutOfMemory => return @as(?[]const u8, null),
        else => return @as(?[]const u8, null),
    };
    defer allocator.free(models_dir);

    const model_path = std.fs.path.join(allocator, &.{ models_dir, model_id }) catch return error.OutOfMemory;
    errdefer allocator.free(model_path);

    // Check directory exists
    std.fs.cwd().access(model_path, .{}) catch return null;

    // Check for weight files
    const weights = resolver.findWeightsFile(allocator, model_path) catch return null;
    if (weights) |path| {
        allocator.free(path);
        return model_path;
    }

    allocator.free(model_path);
    return null;
}

/// List converted models in the Talu local cache.
///
/// Scans $TALU_HOME/models/ for org/name subdirectories.
/// Returns empty slice if the directory doesn't exist.
/// Caller owns returned memory.
pub fn listTaluModels(allocator: std.mem.Allocator, options: ListOptions) CacheError![]CachedModel {
    const models_dir = getTaluModelsDir(allocator) catch |err| switch (err) {
        CacheError.NoHomeDir => return CacheError.NoHomeDir,
        CacheError.OutOfMemory => return CacheError.OutOfMemory,
        else => return &.{},
    };
    defer allocator.free(models_dir);

    // Open models directory; return empty if it doesn't exist
    var dir = std.fs.cwd().openDir(models_dir, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return allocator.alloc(CachedModel, 0) catch return CacheError.OutOfMemory,
        error.AccessDenied => return CacheError.AccessDenied,
        else => return allocator.alloc(CachedModel, 0) catch return CacheError.OutOfMemory,
    };
    defer dir.close();

    var cached_models = std.ArrayListUnmanaged(CachedModel){};
    errdefer {
        for (cached_models.items) |entry| {
            allocator.free(entry.model_id);
            allocator.free(entry.cache_dir);
        }
        cached_models.deinit(allocator);
    }

    // Scan org-level directories
    var org_iter = dir.iterate();
    while (org_iter.next() catch |err| switch (err) {
        else => return CacheError.Unexpected,
    }) |org_entry| {
        if (org_entry.kind != .directory) continue;

        // Open org subdirectory to scan model-level dirs
        var org_dir = dir.openDir(org_entry.name, .{ .iterate = true }) catch continue;
        defer org_dir.close();

        var model_iter = org_dir.iterate();
        while (model_iter.next() catch |err| switch (err) {
            else => return CacheError.Unexpected,
        }) |model_entry| {
            if (model_entry.kind != .directory) continue;

            // Build model path: models_dir/org/name
            const model_path = std.fs.path.join(allocator, &.{ models_dir, org_entry.name, model_entry.name }) catch return CacheError.OutOfMemory;
            errdefer allocator.free(model_path);

            if (options.require_weights) {
                const weights = resolver.findWeightsFile(allocator, model_path) catch |err| switch (err) {
                    error.OutOfMemory => return CacheError.OutOfMemory,
                    error.AccessDenied => return CacheError.AccessDenied,
                    else => {
                        allocator.free(model_path);
                        continue;
                    },
                };
                if (weights) |path| {
                    allocator.free(path);
                } else {
                    allocator.free(model_path);
                    continue;
                }
            }

            // Build model ID: org/name
            const model_id = std.fmt.allocPrint(allocator, "{s}/{s}", .{ org_entry.name, model_entry.name }) catch return CacheError.OutOfMemory;
            errdefer allocator.free(model_id);

            cached_models.append(allocator, .{
                .model_id = model_id,
                .cache_dir = model_path,
                .source = .managed,
            }) catch return CacheError.OutOfMemory;
        }
    }

    return cached_models.toOwnedSlice(allocator) catch return CacheError.OutOfMemory;
}

/// Check if a model directory exists in the Talu cache (regardless of weights).
pub fn taluModelDirExists(allocator: std.mem.Allocator, model_id: []const u8) CacheError!bool {
    const models_dir = getTaluModelsDir(allocator) catch |err| switch (err) {
        CacheError.NoHomeDir => return false,
        CacheError.OutOfMemory => return CacheError.OutOfMemory,
        else => return CacheError.Unexpected,
    };
    defer allocator.free(models_dir);

    const model_path = std.fs.path.join(allocator, &.{ models_dir, model_id }) catch return CacheError.OutOfMemory;
    defer allocator.free(model_path);

    std.fs.cwd().access(model_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        error.AccessDenied, error.PermissionDenied => return CacheError.AccessDenied,
        else => return CacheError.Unexpected,
    };
    return true;
}

/// Delete a converted model from the Talu local cache.
/// Returns true if anything was deleted, false if not present.
pub fn deleteTaluCachedModel(allocator: std.mem.Allocator, model_id: []const u8) CacheError!bool {
    const models_dir = getTaluModelsDir(allocator) catch |err| switch (err) {
        CacheError.NoHomeDir => return false,
        CacheError.OutOfMemory => return CacheError.OutOfMemory,
        else => return CacheError.Unexpected,
    };
    defer allocator.free(models_dir);

    const model_path = std.fs.path.join(allocator, &.{ models_dir, model_id }) catch return CacheError.OutOfMemory;
    defer allocator.free(model_path);

    std.fs.cwd().access(model_path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        error.AccessDenied, error.PermissionDenied => return CacheError.AccessDenied,
        else => return CacheError.Unexpected,
    };

    const parent_path = std.fs.path.dirname(model_path) orelse return CacheError.Unexpected;
    const base_name = std.fs.path.basename(model_path);

    var parent_dir = std.fs.cwd().openDir(parent_path, .{}) catch |err| switch (err) {
        error.AccessDenied, error.PermissionDenied => return CacheError.AccessDenied,
        else => return CacheError.Unexpected,
    };
    defer parent_dir.close();

    parent_dir.deleteTree(base_name) catch |err| switch (err) {
        error.AccessDenied, error.PermissionDenied => return CacheError.AccessDenied,
        error.ReadOnlyFileSystem => return CacheError.ReadOnlyFileSystem,
        else => return CacheError.Unexpected,
    };
    return true;
}

// =============================================================================
// Tests
// =============================================================================

const EnvFns = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

const Env = struct {
    fn setEnvVar(alloc: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        const key_z = try alloc.allocSentinel(u8, key.len, 0);
        defer alloc.free(key_z);
        @memcpy(key_z[0..key.len], key);

        const value_z = try alloc.allocSentinel(u8, value.len, 0);
        defer alloc.free(value_z);
        @memcpy(value_z[0..value.len], value);

        if (EnvFns.setenv(key_z.ptr, value_z.ptr, 1) != 0) return error.Unexpected;
    }

    fn unsetEnvVar(alloc: std.mem.Allocator, key: []const u8) !void {
        const key_z = try alloc.allocSentinel(u8, key.len, 0);
        defer alloc.free(key_z);
        @memcpy(key_z[0..key.len], key);

        _ = EnvFns.unsetenv(key_z.ptr);
    }
};

test "getTaluHome respects TALU_HOME env" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    try Env.setEnvVar(allocator, "TALU_HOME", "/custom/talu/path");
    const home = try getTaluHome(allocator);
    defer allocator.free(home);
    try std.testing.expectEqualStrings("/custom/talu/path", home);
}

test "getTaluHome defaults to HOME/.cache/talu" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    try Env.unsetEnvVar(allocator, "TALU_HOME");

    if (std.posix.getenv("HOME")) |home| {
        const talu_home = try getTaluHome(allocator);
        defer allocator.free(talu_home);

        const expected = try std.fs.path.join(allocator, &.{ home, ".cache", "talu" });
        defer allocator.free(expected);

        try std.testing.expectEqualStrings(expected, talu_home);
    }
}

test "getTaluModelsDir returns TALU_HOME/models" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    try Env.setEnvVar(allocator, "TALU_HOME", "/custom/talu/path");
    const models_dir = try getTaluModelsDir(allocator);
    defer allocator.free(models_dir);
    try std.testing.expectEqualStrings("/custom/talu/path/models", models_dir);
}

test "getTaluCachedPath returns path when model exists with weights" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_path);

    try Env.setEnvVar(allocator, "TALU_HOME", temp_path);

    // Create models/TestOrg/TestModel/ with a weights file
    const model_dir = try std.fs.path.join(allocator, &.{ temp_path, "models", "TestOrg", "TestModel" });
    defer allocator.free(model_dir);
    try std.fs.cwd().makePath(model_dir);

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("fake weights");
    }

    const cached_path = try getTaluCachedPath(allocator, "TestOrg/TestModel");
    try std.testing.expect(cached_path != null);
    defer allocator.free(cached_path.?);
    try std.testing.expectEqualStrings(model_dir, cached_path.?);
}

test "getTaluCachedPath returns null when model doesn't exist" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_path);

    try Env.setEnvVar(allocator, "TALU_HOME", temp_path);

    // Create empty models/ directory
    const models_dir = try std.fs.path.join(allocator, &.{ temp_path, "models" });
    defer allocator.free(models_dir);
    try std.fs.cwd().makePath(models_dir);

    const cached_path = try getTaluCachedPath(allocator, "NonExistent/Model");
    try std.testing.expect(cached_path == null);
}

test "getTaluCachedPath returns null when model exists but has no weights" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_path);

    try Env.setEnvVar(allocator, "TALU_HOME", temp_path);

    // Create models/TestOrg/NoWeights/ with only config.json
    const model_dir = try std.fs.path.join(allocator, &.{ temp_path, "models", "TestOrg", "NoWeights" });
    defer allocator.free(model_dir);
    try std.fs.cwd().makePath(model_dir);

    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const cached_path = try getTaluCachedPath(allocator, "TestOrg/NoWeights");
    try std.testing.expect(cached_path == null);
}

test "deleteTaluCachedModel removes model directory" {
    const allocator = std.testing.allocator;
    const models_dir = try getTaluModelsDir(allocator);
    defer allocator.free(models_dir);

    const model_id = "TestOrg/DeleteMe";
    const model_path = try std.fs.path.join(allocator, &.{ models_dir, model_id });
    defer allocator.free(model_path);

    std.fs.cwd().deleteTree(model_path) catch {};
    try std.fs.cwd().makePath(model_path);

    const deleted = try deleteTaluCachedModel(allocator, model_id);
    try std.testing.expect(deleted);

    const deleted_again = try deleteTaluCachedModel(allocator, model_id);
    try std.testing.expect(!deleted_again);
}

test "taluModelDirExists returns true when directory exists" {
    const allocator = std.testing.allocator;
    const models_dir = try getTaluModelsDir(allocator);
    defer allocator.free(models_dir);

    const model_id = "TestOrg/Exists";
    const model_path = try std.fs.path.join(allocator, &.{ models_dir, model_id });
    defer allocator.free(model_path);

    std.fs.cwd().deleteTree(model_path) catch {};
    try std.fs.cwd().makePath(model_path);

    const exists = try taluModelDirExists(allocator, model_id);
    try std.testing.expect(exists);

    std.fs.cwd().deleteTree(model_path) catch {};
    const exists_after = try taluModelDirExists(allocator, model_id);
    try std.testing.expect(!exists_after);
}

test "listTaluModels returns empty when directory doesn't exist" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_path);

    // Point TALU_HOME to empty temp dir (no models/ subdir)
    try Env.setEnvVar(allocator, "TALU_HOME", temp_path);

    const models = try listTaluModels(allocator, .{ .require_weights = false });
    defer allocator.free(models);

    try std.testing.expectEqual(@as(usize, 0), models.len);
}

test "listTaluModels finds models in org/name structure" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_path);

    try Env.setEnvVar(allocator, "TALU_HOME", temp_path);

    // Create models/TestOrg/TestModel-GAF4/ with a weights file
    const model_dir = try std.fs.path.join(allocator, &.{ temp_path, "models", "TestOrg", "TestModel-GAF4" });
    defer allocator.free(model_dir);
    try std.fs.cwd().makePath(model_dir);

    const weights_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("fake weights");
    }

    // List with require_weights = true
    const models = try listTaluModels(allocator, .{ .require_weights = true });
    defer {
        for (models) |m| {
            allocator.free(m.model_id);
            allocator.free(m.cache_dir);
        }
        allocator.free(models);
    }

    try std.testing.expectEqual(@as(usize, 1), models.len);
    try std.testing.expectEqualStrings("TestOrg/TestModel-GAF4", models[0].model_id);
    try std.testing.expectEqual(CacheOrigin.managed, models[0].source);
}

test "listTaluModels filters models without weights" {
    const allocator = std.testing.allocator;

    const old_talu_home = std.posix.getenv("TALU_HOME");
    defer {
        if (old_talu_home) |prev| {
            Env.setEnvVar(allocator, "TALU_HOME", std.mem.sliceTo(prev, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "TALU_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_path);

    try Env.setEnvVar(allocator, "TALU_HOME", temp_path);

    // Create models/TestOrg/NoWeights/ with only config.json
    const model_dir = try std.fs.path.join(allocator, &.{ temp_path, "models", "TestOrg", "NoWeights" });
    defer allocator.free(model_dir);
    try std.fs.cwd().makePath(model_dir);

    const config_path = try std.fs.path.join(allocator, &.{ model_dir, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    // With require_weights = true, should be empty
    const models = try listTaluModels(allocator, .{ .require_weights = true });
    defer allocator.free(models);
    try std.testing.expectEqual(@as(usize, 0), models.len);

    // With require_weights = false, should find it
    const all_models = try listTaluModels(allocator, .{ .require_weights = false });
    defer {
        for (all_models) |m| {
            allocator.free(m.model_id);
            allocator.free(m.cache_dir);
        }
        allocator.free(all_models);
    }
    try std.testing.expectEqual(@as(usize, 1), all_models.len);
}

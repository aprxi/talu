//! HF Cache Utilities
//!
//! Handles hub cache directory format and model ID parsing.

const std = @import("std");
const log = @import("../../log.zig");
const resolver = @import("resolver.zig");

/// Parse a model ID from a cache path.
/// e.g., "models--org--model-name" -> { .org = "org", .name = "model-name" }
pub const ModelId = struct {
    org: []const u8,
    name: []const u8,
};

pub const CacheError = error{
    OutOfMemory,
    NoHomeDir,
    NotFound,
    AccessDenied,
    ReadOnlyFileSystem,
    Unexpected,
};

pub const ListOptions = struct {
    /// If true, only include models/snapshots that contain loadable weights.
    require_weights: bool = true,
};

/// Where a cached model originated from.
pub const CacheOrigin = enum(u8) {
    /// HuggingFace Hub cache (~/.cache/huggingface/hub/)
    hub = 0,
    /// Talu managed cache (~/.cache/talu/models/)
    managed = 1,
};

pub const CachedModel = struct {
    /// "org/name"
    model_id: []const u8,
    /// Cache directory path
    cache_dir: []const u8,
    /// Where this model originated from
    source: CacheOrigin = .hub,
};

/// C-compatible cached model entry with null-terminated strings.
/// Used by the C API for FFI.
pub const CachedModelC = struct {
    model_id: [:0]const u8,
    cache_dir: [:0]const u8,
    source: CacheOrigin = .hub,
};

/// C-compatible list of cached models with null-terminated strings.
/// Used by the C API for FFI.
pub const CachedModelListC = struct {
    entries: []CachedModelC,

    const Self = @This();

    /// Convert a slice of CachedModel to C-compatible list with null-terminated strings.
    /// All strings are copied. Caller owns the result and must call deinit().
    pub fn fromModels(allocator: std.mem.Allocator, models: []const CachedModel) !Self {
        if (models.len == 0) {
            return Self{ .entries = &.{} };
        }

        const result = try allocator.alloc(CachedModelC, models.len);
        errdefer allocator.free(result);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |j| {
                allocator.free(result[j].model_id);
                allocator.free(result[j].cache_dir);
            }
        }

        for (models, 0..) |model, i| {
            const model_id = try allocator.allocSentinel(u8, model.model_id.len, 0);
            errdefer allocator.free(model_id);
            @memcpy(model_id, model.model_id);

            const cache_dir = try allocator.allocSentinel(u8, model.cache_dir.len, 0);
            @memcpy(cache_dir, model.cache_dir);

            result[i] = .{
                .model_id = model_id,
                .cache_dir = cache_dir,
                .source = model.source,
            };
            initialized = i + 1;
        }

        return Self{ .entries = result };
    }

    /// Free all strings and the entries array.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.entries) |entry| {
            allocator.free(entry.model_id);
            allocator.free(entry.cache_dir);
        }
        if (self.entries.len > 0) {
            allocator.free(self.entries);
        }
        self.entries = &.{};
    }

    /// Get the number of entries.
    pub fn count(self: *const Self) usize {
        return self.entries.len;
    }
};

/// C-compatible search result entry with null-terminated strings.
/// Used by the C API for FFI.
pub const SearchResultC = struct {
    model_id: [:0]const u8,
    downloads: i64,
    likes: i64,
    last_modified: [:0]const u8,
    pipeline_tag: [:0]const u8,
    params_total: i64,
};

/// C-compatible list of search results with null-terminated strings.
/// Used by the C API for FFI.
pub const SearchResultListC = struct {
    entries: []SearchResultC,

    const Self = @This();
    const hf = @import("../transport/hf.zig");

    /// Convert a slice of SearchResult to C-compatible list with null-terminated strings.
    /// All strings are copied. Caller owns the result and must call deinit().
    pub fn fromResults(allocator: std.mem.Allocator, results: []const hf.SearchResult) !Self {
        if (results.len == 0) {
            return Self{ .entries = &.{} };
        }

        const entries = try allocator.alloc(SearchResultC, results.len);
        errdefer allocator.free(entries);

        var initialized: usize = 0;
        errdefer {
            for (0..initialized) |j| {
                allocator.free(entries[j].model_id);
                allocator.free(entries[j].last_modified);
                allocator.free(entries[j].pipeline_tag);
            }
        }

        for (results, 0..) |r, i| {
            const model_id = try allocator.allocSentinel(u8, r.model_id.len, 0);
            errdefer allocator.free(model_id);
            @memcpy(model_id, r.model_id);

            const last_modified = try allocator.allocSentinel(u8, r.last_modified.len, 0);
            errdefer allocator.free(last_modified);
            @memcpy(last_modified, r.last_modified);

            const pipeline_tag = try allocator.allocSentinel(u8, r.pipeline_tag.len, 0);
            @memcpy(pipeline_tag, r.pipeline_tag);

            entries[i] = .{
                .model_id = model_id,
                .downloads = r.downloads,
                .likes = r.likes,
                .last_modified = last_modified,
                .pipeline_tag = pipeline_tag,
                .params_total = r.params_total,
            };
            initialized = i + 1;
        }

        return Self{ .entries = entries };
    }

    /// Free all strings and the entries array.
    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        for (self.entries) |entry| {
            allocator.free(entry.model_id);
            allocator.free(entry.last_modified);
            allocator.free(entry.pipeline_tag);
        }
        if (self.entries.len > 0) {
            allocator.free(self.entries);
        }
        self.entries = &.{};
    }

    pub fn count(self: *const Self) usize {
        return self.entries.len;
    }
};

pub const CachedSnapshot = struct {
    /// Snapshot hash directory name (often a git hash; this code also supports "main")
    revision: []const u8,
    /// Full path to snapshot directory
    snapshot_dir: []const u8,
    /// Whether this snapshot contains weights (safetensors)
    has_weights: bool,
};

fn parseModelId(path: []const u8) ?ModelId {
    const basename = std.fs.path.basename(path);
    if (!std.mem.startsWith(u8, basename, "models--")) return null;
    const id_suffix = basename["models--".len..];
    const sep = std.mem.indexOf(u8, id_suffix, "--") orelse return null;
    return .{ .org = id_suffix[0..sep], .name = id_suffix[sep + 2 ..] };
}

/// Check if a string looks like a model ID (org/model format).
pub fn isModelId(path: []const u8) bool {
    var slash_count: usize = 0;
    var last_slash_pos: usize = 0;

    for (path, 0..) |byte, byte_idx| {
        if (byte == '/') {
            slash_count += 1;
            last_slash_pos = byte_idx;
        }
    }

    // Must have exactly one slash, not at start or end
    if (slash_count != 1) return false;
    if (last_slash_pos == 0 or last_slash_pos == path.len - 1) return false;

    // Not a file path indicator
    if (path[0] == '.' or path[0] == '/' or path[0] == '~') return false;
    if (path.len > 1 and path[1] == ':') return false; // Windows path

    return true;
}

/// Get HF_HOME directory (defaults to ~/.cache/huggingface).
pub fn getHfHome(allocator: std.mem.Allocator) ![]const u8 {
    if (std.posix.getenv("HF_HOME")) |hf_home| {
        return allocator.dupe(u8, hf_home);
    }
    const home_dir = std.posix.getenv("HOME") orelse return error.NoHomeDir;
    return std.fs.path.join(allocator, &.{ home_dir, ".cache", "huggingface" });
}

/// Get HuggingFace token from environment or token file.
///
/// Resolution order:
/// 1. HF_TOKEN environment variable
/// 2. HF_HOME/token file (plaintext file containing only the token)
///
/// Returns null if no token is found.
/// Caller owns returned memory.
pub fn getHfToken(allocator: std.mem.Allocator) !?[]const u8 {
    // 1. Check HF_TOKEN environment variable first
    if (std.posix.getenv("HF_TOKEN")) |token| {
        log.info("fetch", "Using HF token from HF_TOKEN env", .{});
        return try allocator.dupe(u8, token);
    }

    // 2. Try to read token from HF_HOME/token file
    const hf_home = getHfHome(allocator) catch |err| switch (err) {
        error.NoHomeDir => {
            log.info("fetch", "No HF token (no HOME directory)", .{});
            return null;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer allocator.free(hf_home);

    const token_path = std.fs.path.join(allocator, &.{ hf_home, "token" }) catch return error.OutOfMemory;
    defer allocator.free(token_path);

    const file = std.fs.cwd().openFile(token_path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            log.info("fetch", "No HF token file found", .{ .checked = token_path });
            return null;
        },
        error.AccessDenied => {
            log.warn("fetch", "Cannot read token file (access denied)", .{ .path = token_path });
            return null;
        },
        else => {
            log.info("fetch", "No HF token file found", .{ .checked = token_path });
            return null;
        },
    };
    defer file.close();

    // Token files are small (typically ~40 chars for HF tokens)
    const content = file.readToEndAlloc(allocator, 1024) catch {
        log.warn("fetch", "Failed to read token file", .{ .path = token_path });
        return null;
    };

    // Trim whitespace (token files may have trailing newline)
    const trimmed = std.mem.trim(u8, content, &std.ascii.whitespace);
    if (trimmed.len == 0) {
        allocator.free(content);
        log.warn("fetch", "Token file is empty", .{ .path = token_path });
        return null;
    }

    log.info("fetch", "Using HF token from file", .{ .path = token_path });

    // If trimmed is the same as content, return as-is
    if (trimmed.ptr == content.ptr and trimmed.len == content.len) {
        return content;
    }

    // Otherwise, allocate a new slice with just the trimmed content
    const result = allocator.dupe(u8, trimmed) catch {
        allocator.free(content);
        return error.OutOfMemory;
    };
    allocator.free(content);
    return result;
}

/// Get the cache directory for a model (HF cache format).
/// Format: HF_HOME/hub/models--{org}--{model}
pub fn getModelCacheDir(allocator: std.mem.Allocator, model_id: []const u8) ![]const u8 {
    const hf_home = try getHfHome(allocator);
    defer allocator.free(hf_home);

    // Convert org/model to models--org--model format
    var cache_dir_builder = std.ArrayListUnmanaged(u8){};
    errdefer cache_dir_builder.deinit(allocator);

    try cache_dir_builder.appendSlice(allocator, "models--");
    for (model_id) |byte| {
        if (byte == '/') {
            try cache_dir_builder.appendSlice(allocator, "--");
        } else {
            try cache_dir_builder.append(allocator, byte);
        }
    }

    const cache_dir_path = try std.fs.path.join(allocator, &.{ hf_home, "hub", cache_dir_builder.items });
    cache_dir_builder.deinit(allocator);
    return cache_dir_path;
}

/// Get the path to a cached model if it exists.
/// When `require_weights` is true, only returns snapshots that contain weight files.
/// When false, returns any valid snapshot directory.
/// Returns null if not cached. Returns error for permission/access issues.
pub fn getCachedPath(allocator: std.mem.Allocator, model_id: []const u8, require_weights: bool) !?[]const u8 {
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshots_path = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots" });
    defer allocator.free(snapshots_path);

    var snapshots_dir = std.fs.cwd().openDir(snapshots_path, .{ .iterate = true }) catch |err| {
        return switch (err) {
            error.FileNotFound => null,
            error.AccessDenied, error.PermissionDenied => CacheError.AccessDenied,
            else => CacheError.Unexpected,
        };
    };
    defer snapshots_dir.close();

    // Check all snapshots for a valid one
    var snapshot_iter = snapshots_dir.iterate();
    while (try snapshot_iter.next()) |entry| {
        if (entry.kind != .directory) continue;

        const snapshot_path = try std.fs.path.join(allocator, &.{ snapshots_path, entry.name });
        errdefer allocator.free(snapshot_path);

        if (!require_weights) return snapshot_path;

        // Check if this snapshot has weights
        const weights = resolver.findWeightsFile(allocator, snapshot_path) catch |err| return err;
        if (weights) |path| {
            allocator.free(path);
            return snapshot_path;
        }

        allocator.free(snapshot_path);
    }

    return null;
}

/// Check if a model is already cached with valid weights.
pub fn isCached(allocator: std.mem.Allocator, model_id: []const u8) !bool {
    const path = try getCachedPath(allocator, model_id, true);
    if (path) |p| {
        allocator.free(p);
        return true;
    }
    return false;
}

/// Check if a model's cache directory exists (regardless of whether it has weights).
/// Use this for operations like `rm` where we want to delete even incomplete downloads.
pub fn modelCacheDirExists(allocator: std.mem.Allocator, model_id: []const u8) !bool {
    const cache_dir = getModelCacheDir(allocator, model_id) catch |err| switch (err) {
        error.NoHomeDir => return CacheError.NoHomeDir,
        error.OutOfMemory => return CacheError.OutOfMemory,
    };
    defer allocator.free(cache_dir);

    std.fs.cwd().access(cache_dir, .{}) catch |err| {
        return switch (err) {
            error.FileNotFound => false,
            error.AccessDenied, error.PermissionDenied => CacheError.AccessDenied,
            else => CacheError.Unexpected,
        };
    };
    return true;
}

/// List cached models present in the HF cache.
///
/// This is a cache-oriented operation (directory scan); it does not hit the network.
/// Caller owns returned memory.
pub fn listCachedModels(allocator: std.mem.Allocator, options: ListOptions) CacheError![]CachedModel {
    const hf_home = getHfHome(allocator) catch |err| switch (err) {
        error.NoHomeDir => return CacheError.NoHomeDir,
        error.OutOfMemory => return CacheError.OutOfMemory,
    };
    defer allocator.free(hf_home);

    const hub_dir_path = std.fs.path.join(allocator, &.{ hf_home, "hub" }) catch return CacheError.OutOfMemory;
    defer allocator.free(hub_dir_path);

    var hub_dir = std.fs.cwd().openDir(hub_dir_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return CacheError.NotFound,
        error.AccessDenied => return CacheError.AccessDenied,
        else => return CacheError.Unexpected,
    };
    defer hub_dir.close();

    var cached_models = std.ArrayListUnmanaged(CachedModel){};
    errdefer {
        for (cached_models.items) |entry| {
            allocator.free(entry.model_id);
            allocator.free(entry.cache_dir);
        }
        cached_models.deinit(allocator);
    }

    var dir_iter = hub_dir.iterate();
    while (dir_iter.next() catch |err| switch (err) {
        else => return CacheError.Unexpected,
    }) |entry| {
        if (entry.kind != .directory) continue;
        if (!std.mem.startsWith(u8, entry.name, "models--")) continue;

        const parsed_id = parseModelId(entry.name) orelse continue;
        const model_id = std.fmt.allocPrint(allocator, "{s}/{s}", .{ parsed_id.org, parsed_id.name }) catch return CacheError.OutOfMemory;
        errdefer allocator.free(model_id);

        const cache_dir = std.fs.path.join(allocator, &.{ hub_dir_path, entry.name }) catch return CacheError.OutOfMemory;
        errdefer allocator.free(cache_dir);

        if (options.require_weights) {
            // Only include models with at least one snapshot containing weights.
            if (getCachedPath(allocator, model_id, true) catch null) |cached_path| {
                allocator.free(cached_path);
            } else {
                allocator.free(model_id);
                allocator.free(cache_dir);
                continue;
            }
        }

        cached_models.append(allocator, .{ .model_id = model_id, .cache_dir = cache_dir }) catch return CacheError.OutOfMemory;
    }

    return cached_models.toOwnedSlice(allocator);
}

/// List cached snapshots for a model ID (org/name).
/// Caller owns returned memory.
pub fn listCachedSnapshots(allocator: std.mem.Allocator, model_id: []const u8, options: ListOptions) CacheError![]CachedSnapshot {
    const cache_dir = getModelCacheDir(allocator, model_id) catch |err| switch (err) {
        error.NoHomeDir => return CacheError.NoHomeDir,
        error.OutOfMemory => return CacheError.OutOfMemory,
    };
    defer allocator.free(cache_dir);

    const snapshots_path = std.fs.path.join(allocator, &.{ cache_dir, "snapshots" }) catch return CacheError.OutOfMemory;
    defer allocator.free(snapshots_path);

    var snapshots_dir = std.fs.cwd().openDir(snapshots_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return CacheError.NotFound,
        error.AccessDenied => return CacheError.AccessDenied,
        else => return CacheError.Unexpected,
    };
    defer snapshots_dir.close();

    var snapshot_entries = std.ArrayListUnmanaged(CachedSnapshot){};
    errdefer {
        for (snapshot_entries.items) |entry| {
            allocator.free(entry.revision);
            allocator.free(entry.snapshot_dir);
        }
        snapshot_entries.deinit(allocator);
    }

    var snapshot_iter = snapshots_dir.iterate();
    while (snapshot_iter.next() catch |err| switch (err) {
        else => return CacheError.Unexpected,
    }) |entry| {
        if (entry.kind != .directory) continue;

        const snapshot_dir = std.fs.path.join(allocator, &.{ snapshots_path, entry.name }) catch return CacheError.OutOfMemory;
        errdefer allocator.free(snapshot_dir);

        const has_weights = blk: {
            const weights = resolver.findWeightsFile(allocator, snapshot_dir) catch |err| switch (err) {
                error.OutOfMemory => return CacheError.OutOfMemory,
                error.AccessDenied => return CacheError.AccessDenied,
                error.Unexpected => return CacheError.Unexpected,
                else => return CacheError.Unexpected,
            };
            if (weights) |path| {
                allocator.free(path);
                break :blk true;
            }
            break :blk false;
        };

        if (options.require_weights and !has_weights) {
            allocator.free(snapshot_dir);
            continue;
        }

        const revision = allocator.dupe(u8, entry.name) catch return CacheError.OutOfMemory;
        errdefer allocator.free(revision);

        snapshot_entries.append(allocator, .{
            .revision = revision,
            .snapshot_dir = snapshot_dir,
            .has_weights = has_weights,
        }) catch return CacheError.OutOfMemory;
    }

    return snapshot_entries.toOwnedSlice(allocator);
}

fn deleteTreeIfExists(path: []const u8) CacheError!bool {
    std.fs.cwd().access(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        error.AccessDenied, error.PermissionDenied => return CacheError.AccessDenied,
        else => return CacheError.Unexpected,
    };

    const parent_path = std.fs.path.dirname(path) orelse return CacheError.Unexpected;
    const base_name = std.fs.path.basename(path);

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

/// Delete an entire cached model (all snapshots) from the HF cache.
/// Returns true if anything was deleted, false if not present.
pub fn deleteCachedModel(allocator: std.mem.Allocator, model_id: []const u8) CacheError!bool {
    const cache_dir = getModelCacheDir(allocator, model_id) catch |err| switch (err) {
        error.NoHomeDir => return CacheError.NoHomeDir,
        error.OutOfMemory => return CacheError.OutOfMemory,
    };
    defer allocator.free(cache_dir);
    return deleteTreeIfExists(cache_dir);
}

/// Delete a specific cached snapshot revision (hash) for a model ID.
/// Returns true if anything was deleted, false if not present.
pub fn deleteCachedSnapshot(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    revision: []const u8,
) CacheError!bool {
    const cache_dir = getModelCacheDir(allocator, model_id) catch |err| switch (err) {
        error.NoHomeDir => return CacheError.NoHomeDir,
        error.OutOfMemory => return CacheError.OutOfMemory,
    };
    defer allocator.free(cache_dir);

    const snapshot_dir = std.fs.path.join(allocator, &.{ cache_dir, "snapshots", revision }) catch return CacheError.OutOfMemory;
    defer allocator.free(snapshot_dir);

    return deleteTreeIfExists(snapshot_dir);
}

// =============================================================================
// Tests
// =============================================================================

test "isModelId" {
    try std.testing.expect(isModelId("org/model-name"));
    try std.testing.expect(isModelId("my-org/model-7b"));
    try std.testing.expect(!isModelId("local-model"));
    try std.testing.expect(!isModelId("/absolute/path"));
    try std.testing.expect(!isModelId("./relative/path"));
    try std.testing.expect(!isModelId("C:/windows/path"));
}

test "parseModelId" {
    const id = parseModelId("models--org--model-name");
    try std.testing.expect(id != null);
    try std.testing.expectEqualStrings("org", id.?.org);
    try std.testing.expectEqualStrings("model-name", id.?.name);

    try std.testing.expect(parseModelId("not-a-model-id") == null);
}

test "getModelCacheDir" {
    const allocator = std.testing.allocator;

    // This test depends on environment, just verify it doesn't crash
    if (getModelCacheDir(allocator, "org/model-name")) |cache_dir| {
        defer allocator.free(cache_dir);
        try std.testing.expect(std.mem.endsWith(u8, cache_dir, "models--org--model-name"));
    } else |_| {
        // OK if HOME is not set
    }
}

test "getHfHome with HF_HOME environment variable" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    // Test with custom HF_HOME
    try Env.setEnvVar(allocator, "HF_HOME", "/custom/hf/path");
    const hf_home = try getHfHome(allocator);
    defer allocator.free(hf_home);
    try std.testing.expectEqualStrings("/custom/hf/path", hf_home);
}

test "getHfHome defaults to HOME/.cache/huggingface" {
    const allocator = std.testing.allocator;

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

    const old_hf_home = std.posix.getenv("HF_HOME");
    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    // Unset HF_HOME to test default behavior
    try Env.unsetEnvVar(allocator, "HF_HOME");

    // Only test if HOME is set
    if (std.posix.getenv("HOME")) |home| {
        const hf_home = try getHfHome(allocator);
        defer allocator.free(hf_home);

        const expected_path = try std.fs.path.join(allocator, &.{ home, ".cache", "huggingface" });
        defer allocator.free(expected_path);

        try std.testing.expectEqualStrings(expected_path, hf_home);
    }
}

test "getCachedPath returns null for non-existent model" {
    const allocator = std.testing.allocator;

    const cached_path = try getCachedPath(allocator, "NonExistent/Model", true);
    try std.testing.expect(cached_path == null);
}

test "getCachedPath returns path for cached model with weights" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const model_id = "TestOrg/TestModel";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "abc123" });
    defer allocator.free(snapshot_dir);

    try std.fs.cwd().makePath(snapshot_dir);

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const cached_path = try getCachedPath(allocator, model_id, true);
    try std.testing.expect(cached_path != null);
    defer allocator.free(cached_path.?);

    try std.testing.expect(std.mem.endsWith(u8, cached_path.?, snapshot_dir));
}

test "getHfToken returns HF_TOKEN env var first" {
    const allocator = std.testing.allocator;

    const old_hf_token = std.posix.getenv("HF_TOKEN");
    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_token) |previous_value| {
            Env.setEnvVar(allocator, "HF_TOKEN", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_TOKEN") catch {};
        }
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    // Set HF_TOKEN env var
    try Env.setEnvVar(allocator, "HF_TOKEN", "hf_env_token_123");

    const token = try getHfToken(allocator);
    try std.testing.expect(token != null);
    defer allocator.free(token.?);
    try std.testing.expectEqualStrings("hf_env_token_123", token.?);
}

test "getHfToken reads from HF_HOME/token file" {
    const allocator = std.testing.allocator;

    const old_hf_token = std.posix.getenv("HF_TOKEN");
    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_token) |previous_value| {
            Env.setEnvVar(allocator, "HF_TOKEN", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_TOKEN") catch {};
        }
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    // Unset HF_TOKEN so we fall back to file
    try Env.unsetEnvVar(allocator, "HF_TOKEN");

    // Create temp dir for HF_HOME
    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create token file with content (including trailing newline like real files)
    const token_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "token" });
    defer allocator.free(token_path);
    {
        var file = try std.fs.cwd().createFile(token_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("hf_file_token_456\n");
    }

    const token = try getHfToken(allocator);
    try std.testing.expect(token != null);
    defer allocator.free(token.?);
    try std.testing.expectEqualStrings("hf_file_token_456", token.?);
}

test "getHfToken returns null when no token available" {
    const allocator = std.testing.allocator;

    const old_hf_token = std.posix.getenv("HF_TOKEN");
    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_token) |previous_value| {
            Env.setEnvVar(allocator, "HF_TOKEN", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_TOKEN") catch {};
        }
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    // Unset HF_TOKEN
    try Env.unsetEnvVar(allocator, "HF_TOKEN");

    // Create empty temp dir for HF_HOME (no token file)
    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const token = try getHfToken(allocator);
    try std.testing.expect(token == null);
}

test "getHfToken prefers HF_TOKEN env over file" {
    const allocator = std.testing.allocator;

    const old_hf_token = std.posix.getenv("HF_TOKEN");
    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_token) |previous_value| {
            Env.setEnvVar(allocator, "HF_TOKEN", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_TOKEN") catch {};
        }
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    // Set HF_TOKEN env var
    try Env.setEnvVar(allocator, "HF_TOKEN", "hf_env_token");

    // Also create token file with different content
    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const token_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "token" });
    defer allocator.free(token_path);
    {
        var file = try std.fs.cwd().createFile(token_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("hf_file_token");
    }

    // Should prefer env var over file
    const token = try getHfToken(allocator);
    try std.testing.expect(token != null);
    defer allocator.free(token.?);
    try std.testing.expectEqualStrings("hf_env_token", token.?);
}

test "isCached returns false for non-existent model" {
    const allocator = std.testing.allocator;

    const is_cached = try isCached(allocator, "NonExistent/Model");
    try std.testing.expect(!is_cached);
}

test "isCached returns true for cached model with weights" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const model_id = "CachedOrg/CachedModel";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "def456" });
    defer allocator.free(snapshot_dir);

    try std.fs.cwd().makePath(snapshot_dir);

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const is_cached = try isCached(allocator, model_id);
    try std.testing.expect(is_cached);
}

test "listCachedModels returns empty list when no models cached" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create hub directory but no models
    const hub_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "hub" });
    defer allocator.free(hub_path);
    try std.fs.cwd().makePath(hub_path);

    const models = try listCachedModels(allocator, .{ .require_weights = false });
    defer allocator.free(models);

    try std.testing.expectEqual(@as(usize, 0), models.len);
}

test "listCachedModels filters models without weights when require_weights is true" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create model without weights
    const model_id = "NoWeights/Model";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "xyz" });
    defer allocator.free(snapshot_dir);

    try std.fs.cwd().makePath(snapshot_dir);

    const models = try listCachedModels(allocator, .{ .require_weights = true });
    defer allocator.free(models);

    try std.testing.expectEqual(@as(usize, 0), models.len);
}

test "listCachedSnapshots returns snapshots for cached model" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const model_id = "SnapshotOrg/SnapshotModel";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    // Create two snapshots, one with weights, one without
    const snapshot1_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "snap1" });
    defer allocator.free(snapshot1_dir);
    try std.fs.cwd().makePath(snapshot1_dir);

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot1_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("");
    }

    const snapshot2_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "snap2" });
    defer allocator.free(snapshot2_dir);
    try std.fs.cwd().makePath(snapshot2_dir);

    // Test listing all snapshots
    const all_snapshots = try listCachedSnapshots(allocator, model_id, .{ .require_weights = false });
    defer {
        for (all_snapshots) |entry| {
            allocator.free(entry.revision);
            allocator.free(entry.snapshot_dir);
        }
        allocator.free(all_snapshots);
    }
    try std.testing.expectEqual(@as(usize, 2), all_snapshots.len);

    // Test listing only snapshots with weights
    const weighted_snapshots = try listCachedSnapshots(allocator, model_id, .{ .require_weights = true });
    defer {
        for (weighted_snapshots) |entry| {
            allocator.free(entry.revision);
            allocator.free(entry.snapshot_dir);
        }
        allocator.free(weighted_snapshots);
    }
    try std.testing.expectEqual(@as(usize, 1), weighted_snapshots.len);
    try std.testing.expectEqualStrings("snap1", weighted_snapshots[0].revision);
}

test "deleteCachedModel removes entire model directory" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const model_id = "RemoveOrg/RemoveModel";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "rev1" });
    defer allocator.free(snapshot_dir);
    try std.fs.cwd().makePath(snapshot_dir);

    // Verify directory exists
    std.fs.cwd().access(cache_dir, .{}) catch |err| {
        try std.testing.expect(err != error.FileNotFound);
    };

    // Delete the model
    const deleted = try deleteCachedModel(allocator, model_id);
    try std.testing.expect(deleted);

    // Verify directory is gone
    const access_result = std.fs.cwd().access(cache_dir, .{});
    try std.testing.expectError(error.FileNotFound, access_result);

    // Deleting non-existent model returns false
    const deleted_again = try deleteCachedModel(allocator, model_id);
    try std.testing.expect(!deleted_again);
}

test "deleteCachedSnapshot removes specific snapshot" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const model_id = "SnapRemoveOrg/SnapRemoveModel";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    // Create two snapshots
    const snapshot1_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "snap_a" });
    defer allocator.free(snapshot1_dir);
    try std.fs.cwd().makePath(snapshot1_dir);

    const snapshot2_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "snap_b" });
    defer allocator.free(snapshot2_dir);
    try std.fs.cwd().makePath(snapshot2_dir);

    // Delete one snapshot
    const deleted = try deleteCachedSnapshot(allocator, model_id, "snap_a");
    try std.testing.expect(deleted);

    // Verify snap_a is gone but snap_b remains
    const access_a = std.fs.cwd().access(snapshot1_dir, .{});
    try std.testing.expectError(error.FileNotFound, access_a);

    std.fs.cwd().access(snapshot2_dir, .{}) catch |err| {
        try std.testing.expect(err != error.FileNotFound);
    };

    // Deleting non-existent snapshot returns false
    const deleted_again = try deleteCachedSnapshot(allocator, model_id, "snap_a");
    try std.testing.expect(!deleted_again);
}

// =============================================================================
// Cache Size Operations
// =============================================================================

/// Get the size of a directory in bytes (recursive).
pub fn getDirSize(allocator: std.mem.Allocator, path: []const u8) !u64 {
    var total_bytes: u64 = 0;

    var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return 0,
        else => return err,
    };
    defer dir.close();

    var walker = dir.walk(allocator) catch return 0;
    defer walker.deinit();

    while (walker.next() catch null) |entry| {
        if (entry.kind == .file) {
            const stat = entry.dir.statFile(entry.basename) catch continue;
            total_bytes += stat.size;
        }
    }

    return total_bytes;
}

/// Get the size of a cached model in bytes.
pub fn getModelSize(allocator: std.mem.Allocator, model_id: []const u8) !u64 {
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);
    return getDirSize(allocator, cache_dir);
}

/// Get total size of the HF cache (hub directory) in bytes.
pub fn getTotalCacheSize(allocator: std.mem.Allocator) !u64 {
    const hf_home = try getHfHome(allocator);
    defer allocator.free(hf_home);

    const hub_path = try std.fs.path.join(allocator, &.{ hf_home, "hub" });
    defer allocator.free(hub_path);

    return getDirSize(allocator, hub_path);
}

/// Get the modification time of a cached model (Unix timestamp in seconds).
/// Returns the most recent mtime of any file in the model's cache directory.
/// Returns 0 if the model is not cached.
pub fn getModelMtime(allocator: std.mem.Allocator, model_id: []const u8) !i64 {
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);
    return getDirMtime(allocator, cache_dir);
}

/// Get the most recent modification time of any file in a directory (recursive).
/// Returns Unix timestamp in seconds, or 0 if directory doesn't exist.
pub fn getDirMtime(allocator: std.mem.Allocator, path: []const u8) !i64 {
    var max_mtime: i64 = 0;

    var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return 0,
        else => return err,
    };
    defer dir.close();

    var walker = dir.walk(allocator) catch return 0;
    defer walker.deinit();

    while (walker.next() catch null) |entry| {
        if (entry.kind == .file) {
            const stat = entry.dir.statFile(entry.basename) catch continue;
            // mtime is in nanoseconds, convert to seconds
            const mtime_sec: i64 = @intCast(@divTrunc(stat.mtime, std.time.ns_per_s));
            if (mtime_sec > max_mtime) {
                max_mtime = mtime_sec;
            }
        }
    }

    return max_mtime;
}

test "getDirSize returns 0 for non-existent directory" {
    const size = try getDirSize(std.testing.allocator, "/nonexistent/path/12345");
    try std.testing.expectEqual(@as(u64, 0), size);
}

test "getDirSize calculates size of files in directory" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create files with known sizes
    const file1_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "file1.txt" });
    defer allocator.free(file1_path);
    {
        var file = try std.fs.cwd().createFile(file1_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("12345"); // 5 bytes
    }

    const file2_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "file2.txt" });
    defer allocator.free(file2_path);
    {
        var file = try std.fs.cwd().createFile(file2_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("1234567890"); // 10 bytes
    }

    // Create subdirectory with file
    const subdir_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "subdir" });
    defer allocator.free(subdir_path);
    try std.fs.cwd().makePath(subdir_path);

    const file3_path = try std.fs.path.join(allocator, &.{ subdir_path, "file3.txt" });
    defer allocator.free(file3_path);
    {
        var file = try std.fs.cwd().createFile(file3_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("123"); // 3 bytes
    }

    const size = try getDirSize(allocator, temp_dir_path);
    try std.testing.expectEqual(@as(u64, 18), size); // 5 + 10 + 3 = 18
}

test "getModelSize returns size of cached model directory" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    const model_id = "SizeTest/Model";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "main" });
    defer allocator.free(snapshot_dir);
    try std.fs.cwd().makePath(snapshot_dir);

    // Create files with known sizes
    const config_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{\"test\": true}"); // 14 bytes
    }

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("0" ** 100); // 100 bytes
    }

    const size = try getModelSize(allocator, model_id);
    try std.testing.expectEqual(@as(u64, 114), size); // 14 + 100 = 114
}

test "getModelSize returns 0 for non-existent model" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create hub directory but no model
    const hub_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "hub" });
    defer allocator.free(hub_path);
    try std.fs.cwd().makePath(hub_path);

    const size = try getModelSize(allocator, "NonExistent/Model");
    try std.testing.expectEqual(@as(u64, 0), size);
}

test "getTotalCacheSize returns total size of hub directory" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create two models with files
    const model1_id = "TotalSize/Model1";
    const cache_dir1 = try getModelCacheDir(allocator, model1_id);
    defer allocator.free(cache_dir1);

    const snapshot_dir1 = try std.fs.path.join(allocator, &.{ cache_dir1, "snapshots", "main" });
    defer allocator.free(snapshot_dir1);
    try std.fs.cwd().makePath(snapshot_dir1);

    const file1_path = try std.fs.path.join(allocator, &.{ snapshot_dir1, "weights.bin" });
    defer allocator.free(file1_path);
    {
        var file = try std.fs.cwd().createFile(file1_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("0" ** 50); // 50 bytes
    }

    const model2_id = "TotalSize/Model2";
    const cache_dir2 = try getModelCacheDir(allocator, model2_id);
    defer allocator.free(cache_dir2);

    const snapshot_dir2 = try std.fs.path.join(allocator, &.{ cache_dir2, "snapshots", "main" });
    defer allocator.free(snapshot_dir2);
    try std.fs.cwd().makePath(snapshot_dir2);

    const file2_path = try std.fs.path.join(allocator, &.{ snapshot_dir2, "weights.bin" });
    defer allocator.free(file2_path);
    {
        var file = try std.fs.cwd().createFile(file2_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("0" ** 30); // 30 bytes
    }

    const total_size = try getTotalCacheSize(allocator);
    try std.testing.expectEqual(@as(u64, 80), total_size); // 50 + 30 = 80
}

test "getTotalCacheSize returns 0 for empty hub" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Don't create hub directory - getTotalCacheSize should return 0
    const total_size = try getTotalCacheSize(allocator);
    try std.testing.expectEqual(@as(u64, 0), total_size);
}

test "cache listing and removal (HF_HOME override)" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

            // Ignore errors from unsetenv for portability.
            _ = EnvFns.unsetenv(key_z.ptr);
        }
    };

    defer {
        if (old_hf_home) |previous_value| {
            // Restore previous value
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create HF cache structure with one model and one snapshot containing weights.
    const model_id = "Org/Model";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "abc" });
    defer allocator.free(snapshot_dir);

    try std.fs.cwd().makePath(snapshot_dir);

    const config_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    const weights_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        // Empty is fine; resolver only checks existence.
        try file.writeAll("");
    }

    const cached_models = try listCachedModels(allocator, .{ .require_weights = true });
    defer {
        for (cached_models) |entry| {
            allocator.free(entry.model_id);
            allocator.free(entry.cache_dir);
        }
        allocator.free(cached_models);
    }
    try std.testing.expectEqual(@as(usize, 1), cached_models.len);
    try std.testing.expectEqualStrings(model_id, cached_models[0].model_id);

    const cached_snapshots = try listCachedSnapshots(allocator, model_id, .{ .require_weights = true });
    defer {
        for (cached_snapshots) |entry| {
            allocator.free(entry.revision);
            allocator.free(entry.snapshot_dir);
        }
        allocator.free(cached_snapshots);
    }
    try std.testing.expectEqual(@as(usize, 1), cached_snapshots.len);
    try std.testing.expectEqualStrings("abc", cached_snapshots[0].revision);
    try std.testing.expect(cached_snapshots[0].has_weights);

    // Delete snapshot, then model directory.
    try std.testing.expect(try deleteCachedSnapshot(allocator, model_id, "abc"));
    const snapshots_after = try listCachedSnapshots(allocator, model_id, .{ .require_weights = false });
    defer {
        for (snapshots_after) |entry| {
            allocator.free(entry.revision);
            allocator.free(entry.snapshot_dir);
        }
        allocator.free(snapshots_after);
    }
    try std.testing.expectEqual(@as(usize, 0), snapshots_after.len);

    try std.testing.expect(try deleteCachedModel(allocator, model_id));
    const models_after = try listCachedModels(allocator, .{ .require_weights = false });
    defer {
        for (models_after) |entry| {
            allocator.free(entry.model_id);
            allocator.free(entry.cache_dir);
        }
        allocator.free(models_after);
    }
    try std.testing.expectEqual(@as(usize, 0), models_after.len);
}

test "getDirMtime returns 0 for non-existent directory" {
    const mtime = try getDirMtime(std.testing.allocator, "/nonexistent/path/12345");
    try std.testing.expectEqual(@as(i64, 0), mtime);
}

test "getDirMtime returns max mtime of files in directory" {
    const allocator = std.testing.allocator;

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Create a file
    const file1_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "file1.txt" });
    defer allocator.free(file1_path);
    {
        var file = try std.fs.cwd().createFile(file1_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("test content");
    }

    // Create subdirectory with file
    const subdir_path = try std.fs.path.join(allocator, &.{ temp_dir_path, "subdir" });
    defer allocator.free(subdir_path);
    try std.fs.cwd().makePath(subdir_path);

    const file2_path = try std.fs.path.join(allocator, &.{ subdir_path, "file2.txt" });
    defer allocator.free(file2_path);
    {
        var file = try std.fs.cwd().createFile(file2_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("more content");
    }

    const mtime = try getDirMtime(allocator, temp_dir_path);

    // mtime should be a recent timestamp (within last minute)
    const now = std.time.timestamp();
    try std.testing.expect(mtime > 0);
    try std.testing.expect(mtime <= now);
    try std.testing.expect(now - mtime < 60); // Created within last minute
}

test "getModelMtime returns mtime for cached model (HF_HOME override)" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Create HF cache structure with model files
    const model_id = "TestOrg/TestModel";
    const cache_dir = try getModelCacheDir(allocator, model_id);
    defer allocator.free(cache_dir);

    const snapshot_dir = try std.fs.path.join(allocator, &.{ cache_dir, "snapshots", "main" });
    defer allocator.free(snapshot_dir);

    try std.fs.cwd().makePath(snapshot_dir);

    // Create config file
    const config_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "config.json" });
    defer allocator.free(config_path);
    {
        var file = try std.fs.cwd().createFile(config_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("{}");
    }

    // Create weights file
    const weights_path = try std.fs.path.join(allocator, &.{ snapshot_dir, "model.safetensors" });
    defer allocator.free(weights_path);
    {
        var file = try std.fs.cwd().createFile(weights_path, .{ .truncate = true });
        defer file.close();
        try file.writeAll("fake weights");
    }

    // Get mtime
    const mtime = try getModelMtime(allocator, model_id);

    // mtime should be a recent timestamp
    const now = std.time.timestamp();
    try std.testing.expect(mtime > 0);
    try std.testing.expect(mtime <= now);
    try std.testing.expect(now - mtime < 60); // Created within last minute
}

test "getModelMtime returns 0 for uncached model" {
    const allocator = std.testing.allocator;

    const old_hf_home = std.posix.getenv("HF_HOME");
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

    defer {
        if (old_hf_home) |previous_value| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(previous_value, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const temp_dir_path = temp_dir.dir.realpathAlloc(allocator, ".") catch return error.OutOfMemory;
    defer allocator.free(temp_dir_path);

    // Point to empty temp dir (no cached models)
    try Env.setEnvVar(allocator, "HF_HOME", temp_dir_path);

    // Get mtime for non-existent model
    const mtime = try getModelMtime(allocator, "NonExistent/Model");
    try std.testing.expectEqual(@as(i64, 0), mtime);
}

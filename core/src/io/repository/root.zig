//! Model Repository
//!
//! Handles model storage resolution for **repository-based** models.
//!
//! ## Call Flow
//!
//! ```
//! capi/router.zig
//!     ↓
//! router/root.zig (classifyModel)
//!     │
//!     ├─ External API (openai::, vllm::, etc.) → error (not yet implemented)
//!     │
//!     └─ Native (repository-based) → router/local.zig (LocalEngine.init)
//!                                        ↓
//!                                    io/repository/root.zig (resolveModelPath) ← YOU ARE HERE
//!                                        ↓
//!                                    io/repository/scheme.zig (parse storage scheme)
//! ```
//!
//! ## Responsibility
//!
//! This module resolves WHERE a repository-based model is stored:
//!   - Local filesystem (paths)
//!   - HuggingFace Hub (model IDs)
//!
//! ## Model ID Resolution
//!
//! Use `resolveModelPath()` as the single entry point. It handles:
//!
//! | Input                          | Storage    | Action                        |
//! |--------------------------------|------------|-------------------------------|
//! | `/absolute/path`               | Local      | Resolve snapshots if needed   |
//! | `./relative/path`              | Local      | Resolve snapshots if needed   |
//! | `../parent/path`               | Local      | Resolve snapshots if needed   |
//! | `~/home/path`                  | Local      | Resolve snapshots if needed   |
//! | `models--org--name/snapshots/` | Local      | Resolve to best snapshot      |
//! | `org/model-name`               | HF Hub     | Cache lookup → download       |
//!
//! ## Internal Structure
//!
//! - `scheme.zig`: Parses URI schemes (local paths, model IDs)
//! - `cache.zig`: HF cache operations (~/.cache/huggingface/hub/)
//! - `resolver.zig`: Snapshot resolution, Bundle creation
//! - `bundle.zig`: Model file paths (config, weights, tokenizer)
//!
//! ## Usage
//!
//! ```zig
//! const repository = @import("io/repository/root.zig");
//!
//! // Resolve any model identifier to a local path
//! const path = try repository.resolveModelPath(allocator, "org/model-name", .{});
//! defer allocator.free(path);
//!
//! // Then load the model bundle
//! var bundle = try repository.resolve(allocator, path);
//! defer bundle.deinit();
//! ```

const std = @import("std");
const log = @import("../../log.zig");
const progress_api = @import("../../capi/progress.zig");

// Re-export core types
pub const Bundle = @import("bundle.zig").Bundle;
pub const resolver = @import("resolver.zig");
pub const cache = @import("cache.zig");
pub const talu_cache = @import("talu_cache.zig");
pub const scheme = @import("scheme.zig");
pub const source = @import("source.zig");

// Re-export scheme types
pub const Scheme = scheme.Scheme;
pub const Uri = scheme.Uri;
pub const ParseError = scheme.ParseError;
pub const parseUri = scheme.parse;

// Re-export source types
pub const ModelSource = source.ModelSource;
pub const SourceConfig = source.SourceConfig;
pub const SourceError = source.SourceError;

// Transport layer (for fetching from remote sources)
const transport = @import("../transport/root.zig");
pub const http = transport.http;
pub const hf = transport.hf;

// Re-export commonly used types
pub const DownloadConfig = hf.DownloadConfig;
pub const SearchConfig = hf.SearchConfig;
pub const ProgressCallback = http.ProgressCallback;
pub const FileStartCallback = http.FileStartCallback;
pub const CachedModel = cache.CachedModel;
pub const CachedModelC = cache.CachedModelC;
pub const CachedModelListC = cache.CachedModelListC;
pub const CachedSnapshot = cache.CachedSnapshot;
pub const CacheOrigin = cache.CacheOrigin;
pub const ListOptions = cache.ListOptions;

/// Re-export unified progress types for consumers
pub const ProgressContext = progress_api.ProgressContext;

/// Configuration for resolving model URIs.
pub const ResolutionConfig = struct {
    /// HF token (optional); defaults to HF_TOKEN env var.
    token: ?[]const u8 = null,
    /// If true, never attempt network access.
    offline: bool = false,
    /// If true, bypass cache and force download for remote schemes.
    force_download: bool = false,
    /// Custom HF endpoint URL (optional, overrides HF_ENDPOINT env var).
    /// Precedence: endpoint_url > HF_ENDPOINT env > default (huggingface.co)
    endpoint_url: ?[]const u8 = null,
    /// Progress context for download progress reporting.
    progress: ProgressContext = ProgressContext.NONE,
};

// Re-export cache size utilities
pub const getDirSize = cache.getDirSize;
pub const getDirMtime = cache.getDirMtime;
pub const getTotalCacheSize = cache.getTotalCacheSize;

/// Get the size of a cached model in bytes. Returns 0 if not cached.
pub fn getModelSize(allocator: std.mem.Allocator, model_id: []const u8) !u64 {
    const path = try getCachedPath(allocator, model_id) orelse return 0;
    defer allocator.free(path);
    return cache.getDirSize(allocator, path);
}

/// Get the modification time of a cached model (Unix timestamp in seconds).
/// Returns 0 if not cached.
pub fn getModelMtime(allocator: std.mem.Allocator, model_id: []const u8) !i64 {
    const path = try getCachedPath(allocator, model_id) orelse return 0;
    defer allocator.free(path);
    return cache.getDirMtime(allocator, path);
}

// Re-export token functions
pub const getHfToken = cache.getHfToken;

// Re-export search
pub const searchModels = hf.searchModels;
pub const searchModelsRich = hf.searchModelsRich;
pub const SearchResult = hf.SearchResult;
pub const SearchSort = hf.SearchSort;
pub const SearchDirection = hf.SearchDirection;
pub const SearchResultC = cache.SearchResultC;
pub const SearchResultListC = cache.SearchResultListC;

/// Resolve a local path or cached model to a Bundle.
///
/// Handles:
/// - Direct paths to model directories
/// - HF cache format (models--org--name/snapshots/...)
///
/// Returns error.NotFound if the path doesn't exist or is missing required files.
pub fn resolve(allocator: std.mem.Allocator, path: []const u8) !Bundle {
    return resolver.resolve(allocator, path);
}

/// Fetch a model from HF Hub (downloads if not cached).
/// Returns a Bundle ready for loading.
///
/// model_id: e.g., "org/model-name"
pub fn fetch(allocator: std.mem.Allocator, model_id: []const u8, config: DownloadConfig) !Bundle {
    return hf.fetchModel(allocator, model_id, config);
}

/// Fetch a model from HF Hub and return the path (caller frees).
/// Use fetch() for a higher-level API that returns a Bundle.
pub fn fetchModel(allocator: std.mem.Allocator, model_id: []const u8, config: DownloadConfig) ![]const u8 {
    return hf.fetchModel(allocator, model_id, config);
}

/// Check if a string looks like an HF model ID (org/model format).
pub fn isModelId(path: []const u8) bool {
    return cache.isModelId(path);
}

/// Get the local cache path for a model ID, or null if not cached.
/// Checks Talu managed cache first, then HuggingFace cache.
pub fn getCachedPath(allocator: std.mem.Allocator, model_id: []const u8) !?[]const u8 {
    if (try talu_cache.getTaluCachedPath(allocator, model_id)) |path| return path;
    return cache.getCachedPath(allocator, model_id);
}

/// List cached models from both Talu managed cache and HuggingFace cache.
/// Talu managed models are listed first, then HuggingFace cached models.
/// Caller owns returned memory; free strings and slice.
pub fn listCachedModels(allocator: std.mem.Allocator, options: ListOptions) ![]CachedModel {
    // 1. Scan Talu managed cache (source = .managed)
    const talu_models = talu_cache.listTaluModels(allocator, options) catch |err| switch (err) {
        cache.CacheError.NoHomeDir, cache.CacheError.NotFound => &.{},
        cache.CacheError.OutOfMemory => return error.OutOfMemory,
        else => &.{},
    };
    defer {
        if (talu_models.len > 0) {
            // We'll transfer ownership to the merged list; only free on error
        }
    }
    errdefer {
        for (talu_models) |m| {
            allocator.free(m.model_id);
            allocator.free(m.cache_dir);
        }
        if (talu_models.len > 0) allocator.free(talu_models);
    }

    // 2. Scan HuggingFace cache (source = .hub)
    const hub_models = cache.listCachedModels(allocator, options) catch |err| switch (err) {
        cache.CacheError.NoHomeDir, cache.CacheError.NotFound => &.{},
        cache.CacheError.OutOfMemory => return error.OutOfMemory,
        else => &.{},
    };
    errdefer {
        for (hub_models) |m| {
            allocator.free(m.model_id);
            allocator.free(m.cache_dir);
        }
        if (hub_models.len > 0) allocator.free(hub_models);
    }

    // 3. Merge: Talu first, then hub
    const total = talu_models.len + hub_models.len;
    if (total == 0) {
        if (talu_models.len > 0) allocator.free(talu_models);
        if (hub_models.len > 0) allocator.free(hub_models);
        return allocator.alloc(CachedModel, 0);
    }

    const merged = allocator.alloc(CachedModel, total) catch return error.OutOfMemory;
    @memcpy(merged[0..talu_models.len], talu_models);
    @memcpy(merged[talu_models.len..], hub_models);

    // Free the source slices (but not the strings inside — ownership transferred)
    if (talu_models.len > 0) allocator.free(talu_models);
    if (hub_models.len > 0) allocator.free(hub_models);

    return merged;
}

/// List cached snapshots for a given model ID (org/name).
/// Caller owns returned memory; free strings and slice.
pub fn listCachedSnapshots(allocator: std.mem.Allocator, model_id: []const u8, options: ListOptions) ![]CachedSnapshot {
    return cache.listCachedSnapshots(allocator, model_id, options);
}

/// Delete an entire cached model (all snapshots).
/// Returns true if anything was deleted.
pub fn deleteCachedModel(allocator: std.mem.Allocator, model_id: []const u8) !bool {
    const hf_deleted = cache.deleteCachedModel(allocator, model_id) catch |err| return err;
    const talu_deleted = talu_cache.deleteTaluCachedModel(allocator, model_id) catch |err| return err;
    const deleted = hf_deleted or talu_deleted;
    if (deleted) {
        log.info("fetch", "Deleted cached model", .{ .model_id = model_id, .hf = hf_deleted, .talu = talu_deleted });
    } else {
        log.info("fetch", "Model not found in cache", .{ .model_id = model_id });
    }
    return deleted;
}

/// Returns true if a model cache directory exists in either cache.
pub fn modelCacheDirExists(allocator: std.mem.Allocator, model_id: []const u8) !bool {
    const hf_exists = cache.modelCacheDirExists(allocator, model_id) catch |err| return err;
    const talu_exists = talu_cache.taluModelDirExists(allocator, model_id) catch |err| return err;
    return hf_exists or talu_exists;
}


/// Delete a specific cached snapshot revision for a model ID.
/// Returns true if anything was deleted.
pub fn deleteCachedSnapshot(allocator: std.mem.Allocator, model_id: []const u8, revision: []const u8) !bool {
    return cache.deleteCachedSnapshot(allocator, model_id, revision);
}

/// Initialize HTTP globally (call once at program start if using fetch)
pub const globalInit = http.globalInit;

/// Clean up HTTP globally (call once at program end)
pub const globalCleanup = http.globalCleanup;

// =============================================================================
// Unified Repository Operations
// =============================================================================

/// Check if a model exists (local cache OR remote source).
///
/// This is the user-facing "can I get this model?" check. It:
/// 1. Checks the local HF cache first (fast, no network)
/// 2. If not cached, queries the remote source (network request)
///
/// Use `isCached()` if you only want to check local cache without network.
///
/// Parameters:
/// - allocator: Memory allocator for temporary allocations
/// - model_id: Model identifier (e.g., "org/model-name")
/// - config: Source configuration (token for auth)
/// - src: Source to check (default: HuggingFace Hub)
///
/// Returns true if the model is available (cached or on source).
pub fn exists(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    config: SourceConfig,
    src: ?ModelSource,
) !bool {
    // 1. Check local cache first (no network)
    if (try cache.isCached(allocator, model_id)) {
        return true;
    }

    // 2. Check remote source
    const actual_source = src orelse ModelSource.default;

    // Initialize HTTP for network request
    transport.globalInit();
    defer transport.globalCleanup();

    return actual_source.exists(allocator, model_id, config) catch |err| {
        // Convert source errors to appropriate return values
        return switch (err) {
            SourceError.Unauthorized => error.Unauthorized,
            SourceError.RateLimited => error.RateLimited,
            SourceError.NotSupported => error.NotSupported,
            SourceError.OutOfMemory => error.OutOfMemory,
            else => false, // Network errors → model not accessible
        };
    };
}

/// Check if a model is in the local cache.
///
/// This is an explicit cache-only check. No network requests are made.
/// Use `exists()` if you want to check both cache and remote source.
///
/// Parameters:
/// - allocator: Memory allocator for temporary allocations
/// - model_id: Model identifier (e.g., "org/model-name")
///
/// Returns true if the model is cached locally with valid weights.
pub fn isCached(allocator: std.mem.Allocator, model_id: []const u8) !bool {
    if (try cache.isCached(allocator, model_id)) return true;
    if (try talu_cache.getTaluCachedPath(allocator, model_id) != null) return true;
    return false;
}

/// List files in a model repository.
///
/// Handles all supported schemes:
/// - `org/model` - List files (checks cache first, then HuggingFace Hub)
/// - `/path`, `./path`, `~/path` - List files in local directory
///
/// Parameters:
/// - allocator: Memory allocator (caller owns returned memory)
/// - model_path: Model path/URI (e.g., "org/model-name")
/// - config: Source configuration (token for auth, used for remote sources)
///
/// Returns owned slice of owned strings. Caller must free both.
pub fn listFiles(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    config: SourceConfig,
) ![][]const u8 {
    const uri = scheme.parse(model_path) catch |err| {
        return switch (err) {
            scheme.ParseError.UnknownScheme => error.InvalidFormat,
        };
    };

    switch (uri.scheme) {
        .local => {
            // Local path - list directory contents
            return listLocalFiles(allocator, uri.path);
        },
        .hub => {
            // HuggingFace Hub - check cache first, then list from API
            if (cache.getCachedPath(allocator, uri.path) catch null) |cached_path| {
                defer allocator.free(cached_path);
                return listLocalFiles(allocator, cached_path);
            }

            // Not cached - list from HuggingFace API
            transport.globalInit();
            defer transport.globalCleanup();

            return ModelSource.default.listFiles(allocator, uri.path, config) catch |err| {
                return switch (err) {
                    SourceError.Unauthorized => error.Unauthorized,
                    SourceError.RateLimited => error.RateLimited,
                    SourceError.NotSupported => error.NotSupported,
                    SourceError.OutOfMemory => error.OutOfMemory,
                    SourceError.ParseError => error.InvalidFormat,
                    else => error.NetworkError,
                };
            };
        },
    }
}

/// List files in a local directory.
fn listLocalFiles(allocator: std.mem.Allocator, dir_path: []const u8) ![][]const u8 {
    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch |err| {
        return switch (err) {
            error.FileNotFound => error.NotFound,
            error.AccessDenied => error.AccessDenied,
            else => error.NotFound,
        };
    };
    defer dir.close();

    var files = std.ArrayListUnmanaged([]const u8){};
    errdefer {
        for (files.items) |f| allocator.free(f);
        files.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind == .file) {
            const name = try allocator.dupe(u8, entry.name);
            try files.append(allocator, name);
        }
    }

    return files.toOwnedSlice(allocator);
}

// =============================================================================
// Unified Model Path Resolution
// =============================================================================

/// Resolve any model path/URI to a local directory path.
///
/// Handles all supported schemes:
/// - `/path`, `./path`, `../path`, `~/path` - Local filesystem
/// - `org/model` - HuggingFace Hub (cache first, then fetch)
/// - `models--org--name/...` - HF cache format
///
/// Config controls offline mode, forced downloads, and custom endpoint.
/// Returns an allocated path that the caller must free.
/// This is the SINGLE source of truth for model path resolution.
pub fn resolveModelPath(allocator: std.mem.Allocator, uri: []const u8, config: ResolutionConfig) ![]const u8 {
    const parsed = scheme.parse(uri) catch |err| {
        return switch (err) {
            scheme.ParseError.UnknownScheme => error.InvalidFormat,
        };
    };

    // Token resolution: config > HF_TOKEN env > HF_HOME/token file
    const owned_token = if (config.token != null) null else try cache.getHfToken(allocator);
    defer if (owned_token) |t| allocator.free(t);
    const token: ?[]const u8 = config.token orelse owned_token;

    switch (parsed.scheme) {
        .local => {
            // Local path - resolve snapshots if needed (handles cache format)
            const resolved = try resolver.resolveSnapshot(allocator, parsed.path);
            log.info("load", "Using local model", .{ .path = parsed.path });
            return resolved;
        },
        .hub => {
            if (!config.force_download) {
                // Check Talu local cache first ($TALU_HOME/models/org/model)
                if (try talu_cache.getTaluCachedPath(allocator, parsed.path)) |talu_path| {
                    log.info("load", "Using local model", .{ .model_id = parsed.path });
                    return talu_path;
                }

                // Then check HuggingFace cache
                if (try cache.getCachedPath(allocator, parsed.path)) |cached_path| {
                    log.info("load", "Using cached model", .{ .model_id = parsed.path });
                    return cached_path;
                }
            }

            if (config.offline) {
                return error.ModelNotCached;
            }

            // Not in any local cache - fetch from HuggingFace Hub
            transport.globalInit();
            defer transport.globalCleanup();

            return try transport.hf.fetchModel(allocator, parsed.path, .{
                .token = token,
                .force = config.force_download,
                .endpoint_url = config.endpoint_url,
                .progress = config.progress,
            });
        },
    }
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

test "resolveModelPath respects offline mode for uncached remote" {
    const allocator = std.testing.allocator;
    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const hf_home = try temp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(hf_home);

    const old = std.posix.getenv("HF_HOME");
    try Env.setEnvVar(allocator, "HF_HOME", hf_home);
    defer {
        if (old) |val| {
            Env.setEnvVar(allocator, "HF_HOME", std.mem.sliceTo(val, 0)) catch {};
        } else {
            Env.unsetEnvVar(allocator, "HF_HOME") catch {};
        }
    }

    try std.testing.expectError(
        error.ModelNotCached,
        resolveModelPath(allocator, "org/model-name", .{ .offline = true }),
    );
}

test "resolveModelPath returns local path without cache lookup" {
    const allocator = std.testing.allocator;
    var temp_dir = std.testing.tmpDir(.{});
    defer temp_dir.cleanup();

    const local_path = try temp_dir.dir.realpathAlloc(allocator, ".");
    defer allocator.free(local_path);

    const resolved = try resolveModelPath(allocator, local_path, .{});
    defer allocator.free(resolved);

    try std.testing.expectEqualStrings(local_path, resolved);
}

test "isModelId accepts valid HuggingFace model IDs" {
    // Standard org/model format
    try std.testing.expect(isModelId("org/model-name"));
    try std.testing.expect(isModelId("my-org/model-7b-hf"));
    try std.testing.expect(isModelId("some-org/model-2"));
    try std.testing.expect(isModelId("org-ai/model-7b-v0.1"));

    // Edge cases that should still be valid
    try std.testing.expect(isModelId("a/b")); // minimal valid ID
    try std.testing.expect(isModelId("org-name/model-name"));
    try std.testing.expect(isModelId("org_name/model_name"));
    try std.testing.expect(isModelId("Org123/Model456"));
}

test "isModelId rejects local file paths" {
    // Relative paths
    try std.testing.expect(!isModelId("./models/my-model"));
    try std.testing.expect(!isModelId("../models/my-model"));
    try std.testing.expect(!isModelId("models/subdir/model"));

    // Absolute paths
    try std.testing.expect(!isModelId("/home/user/models/my-model"));
    try std.testing.expect(!isModelId("/var/models/my-model"));

    // Home directory paths
    try std.testing.expect(!isModelId("~/models/my-model"));

    // Windows paths
    try std.testing.expect(!isModelId("C:/models/my-model"));
    try std.testing.expect(!isModelId("D:/models/my-model"));
}

test "isModelId rejects invalid formats" {
    // No slash
    try std.testing.expect(!isModelId("my-model"));
    try std.testing.expect(!isModelId("model-name"));

    // Multiple slashes (looks like a path)
    try std.testing.expect(!isModelId("org/model/extra"));
    try std.testing.expect(!isModelId("a/b/c"));

    // Slash at boundaries
    try std.testing.expect(!isModelId("/model"));
    try std.testing.expect(!isModelId("org/"));
    try std.testing.expect(!isModelId("/"));

    // Empty string
    try std.testing.expect(!isModelId(""));
}

//! C API for Model Repository Operations
//!
//! Unix-style API for managing model repositories:
//!
//! ## Unified Operations (scheme-agnostic)
//! - talu_repo_list() - List files in model (local, model ID)
//! - talu_repo_exists() - Check if model is available (cache OR source)
//! - talu_repo_resolve_path() - Resolve model URI to local path
//!
//! ## Cache Operations
//! - talu_repo_list_models() - List cached model IDs
//! - talu_repo_is_cached() - Check if model is in local cache
//! - talu_repo_get_cached_path() - Get path to cached model
//! - talu_repo_delete() - Delete model from cache
//! - talu_repo_size() - Get size of cached model
//!
//! ## Source Operations
//! - talu_repo_fetch() - Fetch model from source to cache
//! - talu_repo_search() - Search for models on source

const std = @import("std");
const repository = @import("../io/repository/root.zig");
const ffi = @import("../helpers/ffi.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const progress_api = @import("progress.zig");
const log = @import("../log.zig");

const allocator = std.heap.c_allocator;

/// Allocates a NUL-terminated copy of a byte slice.
/// Returns null on allocation failure (OOM).
fn allocZSlice(bytes: []const u8) ?[:0]u8 {
    // allocSentinel allocates len+1 bytes and sets the sentinel (0) at the end
    const buf = allocator.allocSentinel(u8, bytes.len, 0) catch return null;
    @memcpy(buf, bytes);
    return buf;
}

/// Helper to set error context with a consistent format.
/// ctx is the operation name (e.g., "fetch", "list").
fn setErr(comptime ctx: []const u8, err: anyerror) void {
    capi_error.setError(err, "{s}: {s}", .{ ctx, @errorName(err) });
}

// =============================================================================
// Types
// =============================================================================

/// Re-export types for C API consumers.
pub const CachedModelList = repository.CachedModelListC;
pub const CachedModelEntry = repository.CachedModelC;
pub const StringList = ffi.StringList;

/// Re-export unified progress types.
pub const CProgressCallback = progress_api.CProgressCallback;
pub const ProgressUpdate = progress_api.ProgressUpdate;
pub const ProgressAction = progress_api.ProgressAction;

pub const DownloadOptions = extern struct {
    token: ?[*:0]const u8 = null,
    progress_callback: ?CProgressCallback = null,
    user_data: ?*anyopaque = null,
    force: bool = false,
    /// Custom HF endpoint URL (optional, overrides HF_ENDPOINT env var)
    endpoint_url: ?[*:0]const u8 = null,
    /// Skip downloading weight files (.safetensors).
    skip_weights: bool = false,

    /// Get a ProgressContext from the options.
    pub fn progressContext(self: DownloadOptions) progress_api.ProgressContext {
        return progress_api.ProgressContext.init(self.progress_callback, self.user_data);
    }
};

// =============================================================================
// Cache Query Operations
// =============================================================================

/// Checks if a model is available in the local cache (with valid weights).
///
/// Returns 1 if cached, 0 if not cached or on error.
/// Use talu_get_last_error() to distinguish between "not cached" and error.
pub export fn talu_repo_is_cached(model_id: [*:0]const u8) callconv(.c) c_int {
    capi_error.clearError();
    const is_cached = repository.isCached(allocator, std.mem.span(model_id)) catch |e| {
        setErr("is_cached", e);
        return 0;
    };
    return if (is_cached) 1 else 0;
}

/// Checks if a model's cache directory exists (regardless of weights).
///
/// Use this for operations like delete where we want to remove even incomplete downloads.
/// Returns 1 if exists, 0 if not exists or on error.
/// Use talu_get_last_error() to distinguish between "not exists" and error.
pub export fn talu_repo_cache_dir_exists(model_id: [*:0]const u8) callconv(.c) c_int {
    capi_error.clearError();
    const exists = repository.modelCacheDirExists(allocator, std.mem.span(model_id)) catch |e| {
        setErr("cache_dir_exists", e);
        return 0;
    };
    return if (exists) 1 else 0;
}

/// Checks if a model exists (either in cache or at the remote source).
///
/// For HuggingFace models, this may make a network request if not cached.
/// Returns 1 if exists, 0 if not found or on error.
///
/// Parameters:
///   model_id: Model identifier (e.g., "org/model-name" or local path)
///   token: Optional HuggingFace API token for private models
pub export fn talu_repo_exists(model_id: [*:0]const u8, token: ?[*:0]const u8) callconv(.c) c_int {
    capi_error.clearError();
    const tok: ?[]const u8 = if (token) |t| std.mem.span(t) else null;
    const exists = repository.exists(allocator, std.mem.span(model_id), .{ .token = tok }, null) catch |e| {
        setErr("exists", e);
        return 0;
    };
    return if (exists) 1 else 0;
}

/// Resolves a model URI to a local filesystem path.
///
/// Handles multiple URI schemes:
///   - Local paths: Returns as-is if valid
///   - org/model: Treated as HuggingFace model ID (cache first, then fetch)
///
/// Parameters:
///   uri: Model URI or path
///   offline: If true, only check cache (no downloads)
///   token: Optional HuggingFace API token
///   endpoint_url: Optional custom HF endpoint URL (overrides HF_ENDPOINT env var)
///   out_path: Output pointer for resolved path (caller must free with talu_free_string)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_resolve_path(
    uri: [*:0]const u8,
    offline: bool,
    token: ?[*:0]const u8,
    endpoint_url: ?[*:0]const u8,
    require_weights: bool,
    out_path: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out_path.* = null;

    const tok: ?[]const u8 = if (token) |t| std.mem.span(t) else null;
    const endpoint: ?[]const u8 = if (endpoint_url) |e| std.mem.span(e) else null;
    const resolved_path = repository.resolveModelPath(
        allocator,
        std.mem.span(uri),
        .{ .token = tok, .offline = offline, .endpoint_url = endpoint, .require_weights = require_weights },
    ) catch |e| {
        setErr("resolve_path", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer allocator.free(resolved_path);

    out_path.* = (allocZSlice(resolved_path) orelse return @intFromEnum(error_codes.ErrorCode.out_of_memory)).ptr;
    return 0;
}

/// Gets the local filesystem path for a cached model.
///
/// Returns the path to the model's snapshot directory if cached.
/// If the model is not cached, out is set to null and returns 0 (not an error).
///
/// Parameters:
///   model_id: Model identifier (e.g., "org/model-name")
///   out: Output pointer for path (caller must free with talu_free_string)
///
/// Returns 0 on success (even if not cached), error code on failure.
pub export fn talu_repo_get_cached_path(model_id: [*:0]const u8, require_weights: bool, out: *?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const path = repository.cache.getCachedPath(allocator, std.mem.span(model_id), require_weights) catch |e| {
        setErr("get_cached_path", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    if (path) |p| {
        defer allocator.free(p);
        out.* = (allocZSlice(p) orelse return @intFromEnum(error_codes.ErrorCode.out_of_memory)).ptr;
    }
    return 0;
}

/// Gets the HuggingFace home directory path.
///
/// Returns the path to the HuggingFace cache root (typically ~/.cache/huggingface).
/// Respects the HF_HOME environment variable if set.
///
/// Parameters:
///   out: Output pointer for path (caller must free with talu_free_string)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_get_hf_home(out: *?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const path = repository.cache.getHfHome(allocator) catch |e| {
        setErr("get_hf_home", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer allocator.free(path);
    out.* = (allocZSlice(path) orelse return @intFromEnum(error_codes.ErrorCode.out_of_memory)).ptr;
    return 0;
}

/// Gets the Talu home directory path.
///
/// Returns the path to the Talu home directory (typically ~/.cache/talu).
/// Respects the TALU_HOME environment variable if set.
///
/// Parameters:
///   out: Output pointer for path (caller must free with talu_free_string)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_get_talu_home(out: *?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const path = repository.talu_cache.getTaluHome(allocator) catch |e| {
        setErr("get_talu_home", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer allocator.free(path);
    out.* = (allocZSlice(path) orelse return @intFromEnum(error_codes.ErrorCode.out_of_memory)).ptr;
    return 0;
}

/// Gets the cache directory for a specific model.
///
/// Returns the model-specific cache directory (e.g., ~/.cache/huggingface/hub/models--org--model-name).
/// This is the parent directory containing snapshots and refs.
///
/// Parameters:
///   model_id: Model identifier (e.g., "org/model-name")
///   out: Output pointer for path (caller must free with talu_free_string)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_get_cache_dir(model_id: [*:0]const u8, out: *?[*:0]u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const path = repository.cache.getModelCacheDir(allocator, std.mem.span(model_id)) catch |e| {
        setErr("get_cache_dir", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer allocator.free(path);
    out.* = (allocZSlice(path) orelse return @intFromEnum(error_codes.ErrorCode.out_of_memory)).ptr;
    return 0;
}

/// Checks if a string looks like a HuggingFace model ID.
///
/// Returns 1 if the path appears to be a model ID (e.g., "org/model"),
/// 0 if it looks like a local filesystem path.
pub export fn talu_repo_is_model_id(path: [*:0]const u8) callconv(.c) c_int {
    return if (repository.cache.isModelId(std.mem.span(path))) 1 else 0;
}

// =============================================================================
// List Cached Models
// =============================================================================

/// Lists all cached models that have complete weight files.
///
/// Lists cached models.
///
/// Parameters:
///   require_weights: If true, only include models with complete weight files
///   out: Output pointer for model list (caller must free with talu_repo_list_free)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_list_models(require_weights: bool, out: *?*CachedModelList) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;

    const models = repository.listCachedModels(allocator, .{ .require_weights = require_weights }) catch |e| {
        setErr("list_models", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer {
        for (models) |m| {
            allocator.free(m.model_id);
            allocator.free(m.cache_dir);
        }
        allocator.free(models);
    }

    const list = allocator.create(CachedModelList) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    list.* = repository.CachedModelListC.fromModels(allocator, models) catch {
        allocator.destroy(list);
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    out.* = list;
    return 0;
}

/// Gets the number of entries in a cached model list.
///
/// Parameters:
///   list: Model list from talu_repo_list_models
///
/// Returns the count, or 0 if list is null.
pub export fn talu_repo_list_count(list: ?*const CachedModelList) callconv(.c) usize {
    return if (list) |l| l.entries.len else 0;
}

/// Gets the model ID at the specified index in a cached model list.
///
/// The returned string is owned by the list and valid until talu_repo_list_free().
///
/// Parameters:
///   list: Model list from talu_repo_list_models
///   idx: Zero-based index
///   out: Output pointer for model ID string (do not free)
///
/// Returns 0 on success, error code on invalid handle or out-of-bounds index.
pub export fn talu_repo_list_get_id(list: ?*const CachedModelList, idx: usize, out: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const l = list orelse return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    if (idx >= l.entries.len) return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = l.entries[idx].model_id.ptr;
    return 0;
}

/// Gets the cache path at the specified index in a cached model list.
///
/// The returned string is owned by the list and valid until talu_repo_list_free().
///
/// Parameters:
///   list: Model list from talu_repo_list_models
///   idx: Zero-based index
///   out: Output pointer for cache path string (do not free)
///
/// Returns 0 on success, error code on invalid handle or out-of-bounds index.
pub export fn talu_repo_list_get_path(list: ?*const CachedModelList, idx: usize, out: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const l = list orelse return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    if (idx >= l.entries.len) return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = l.entries[idx].cache_dir.ptr;
    return 0;
}

/// Gets the cache origin (source) at the specified index in a cached model list.
///
/// Returns 0 for hub, 1 for local, or 255 on invalid handle/out-of-bounds.
///
/// Parameters:
///   list: Model list from talu_repo_list_models
///   idx: Zero-based index
///
/// Returns the source as a u8, or 255 on error.
pub export fn talu_repo_list_get_source(list: ?*const CachedModelList, idx: usize) callconv(.c) u8 {
    const l = list orelse return 255;
    if (idx >= l.entries.len) return 255;
    return @intFromEnum(l.entries[idx].source);
}

/// Frees a cached model list and all its entries.
///
/// Safe to call with null (no-op).
pub export fn talu_repo_list_free(list: ?*CachedModelList) callconv(.c) void {
    const l = list orelse return;
    l.deinit(allocator);
    allocator.destroy(l);
}

// =============================================================================
// Cache Management
// =============================================================================

/// Deletes a model from the local cache.
///
/// Removes the entire model cache directory including all snapshots.
///
/// Returns 1 if deleted, 0 if not found or on error.
pub export fn talu_repo_delete(model_id: [*:0]const u8) callconv(.c) c_int {
    capi_error.clearError();
    const deleted = repository.deleteCachedModel(allocator, std.mem.span(model_id)) catch |e| {
        setErr("delete", e);
        return 0;
    };
    return if (deleted) 1 else 0;
}

/// Gets the total size in bytes of a cached model.
///
/// Returns 0 if not cached or on error.
pub export fn talu_repo_size(model_id: [*:0]const u8) callconv(.c) u64 {
    return repository.getModelSize(allocator, std.mem.span(model_id)) catch |e| {
        setErr("size", e);
        return 0;
    };
}

/// Gets the total size in bytes of all cached models.
///
/// Returns 0 if cache is empty or on error.
pub export fn talu_repo_total_size() callconv(.c) u64 {
    return repository.getTotalCacheSize(allocator) catch |e| {
        setErr("total_size", e);
        return 0;
    };
}

/// Gets the modification time of a cached model.
///
/// Returns Unix timestamp in seconds, or 0 if not cached or on error.
pub export fn talu_repo_mtime(model_id: [*:0]const u8) callconv(.c) i64 {
    return repository.getModelMtime(allocator, std.mem.span(model_id)) catch |e| {
        setErr("mtime", e);
        return 0;
    };
}

// =============================================================================
// Remote Operations
// =============================================================================

/// Resolve HF token from explicit options or environment.
const TokenResult = struct { token: ?[]const u8, owned: ?[]const u8 };

fn resolveHfToken(options: ?*const DownloadOptions) TokenResult {
    const explicit: ?[]const u8 = if (options) |o| (if (o.token) |t| std.mem.span(t) else null) else null;
    if (explicit != null) return .{ .token = explicit, .owned = null };

    const owned = repository.cache.getHfToken(allocator) catch null;
    return .{ .token = owned, .owned = owned };
}

/// Execute the fetch operation after validation.
fn executeFetch(out: *?[*:0]u8, parsed_path: []const u8, options: ?*const DownloadOptions) i32 {
    const token_result = resolveHfToken(options);
    defer if (token_result.owned) |t| allocator.free(t);

    if (token_result.token != null) log.info("fetch", "HF token available", .{}) else log.info("fetch", "No HF token available", .{});

    var config = buildDownloadConfig(options);
    config.token = token_result.token;

    const path = repository.hf.fetchModel(allocator, parsed_path, config) catch |e| {
        setErr("fetch", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer allocator.free(path);

    out.* = (allocZSlice(path) orelse return @intFromEnum(error_codes.ErrorCode.out_of_memory)).ptr;
    return 0;
}

/// Downloads a model from HuggingFace Hub to the local cache.
///
/// If the model is already cached and force=false, returns the cached path.
/// Progress callbacks are invoked during download.
///
/// Parameters:
///   model_id: HuggingFace model ID (e.g., "org/model-name")
///   options: Download options (token, callbacks, force re-download)
///   out: Output pointer for local path (caller must free with talu_free_string)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_fetch(
    model_id: [*:0]const u8,
    options: ?*const DownloadOptions,
    out: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;

    const parsed = repository.scheme.parse(std.mem.span(model_id)) catch {
        setErr("fetch", error.InvalidFormat);
        return @intFromEnum(error_codes.ErrorCode.model_invalid_format);
    };
    if (parsed.scheme != .hub) {
        setErr("fetch", error.InvalidFormat);
        return @intFromEnum(error_codes.ErrorCode.model_invalid_format);
    }

    return executeFetch(out, parsed.path, options);
}

/// Build DownloadConfig from C API options.
fn buildDownloadConfig(options: ?*const DownloadOptions) repository.hf.DownloadConfig {
    const tok: ?[]const u8 = if (options) |o| (if (o.token) |t| std.mem.span(t) else null) else null;
    const endpoint: ?[]const u8 = if (options) |o| (if (o.endpoint_url) |e| std.mem.span(e) else null) else null;
    const progress = if (options) |o| o.progressContext() else progress_api.ProgressContext.NONE;

    return .{
        .token = tok,
        .progress = progress,
        .force = if (options) |o| o.force else false,
        .endpoint_url = endpoint,
        .skip_weights = if (options) |o| o.skip_weights else false,
    };
}

/// Fetch a single file from a model repository.
///
/// Downloads one file (e.g., "config.json") without fetching the full model.
///
/// Parameters:
///   model_id:  HuggingFace model ID (e.g., "org/model-name")
///   filename:  Name of file to fetch (e.g., "config.json")
///   options:   Download options (token, force). Progress callback ignored.
///   out:       Output pointer for file path. Caller must free with talu_text_free.
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_fetch_file(
    model_id: [*:0]const u8,
    filename: [*:0]const u8,
    options: ?*const DownloadOptions,
    out: *?[*:0]u8,
) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;

    const parsed = repository.scheme.parse(std.mem.span(model_id)) catch {
        setErr("fetch_file", error.InvalidFormat);
        return @intFromEnum(error_codes.ErrorCode.model_invalid_format);
    };

    if (parsed.scheme != .hub) {
        setErr("fetch_file", error.InvalidFormat);
        return @intFromEnum(error_codes.ErrorCode.model_invalid_format);
    }

    const token_result = resolveHfToken(options);
    defer if (token_result.owned) |t| allocator.free(t);

    var download_config = buildDownloadConfig(options);
    download_config.token = token_result.token;

    const file_path = repository.hf.fetchFile(
        allocator,
        parsed.path,
        std.mem.span(filename),
        download_config,
    ) catch |e| {
        setErr("fetch_file", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };

    const result = allocZSlice(file_path) orelse {
        allocator.free(file_path);
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    allocator.free(file_path);
    out.* = result.ptr;
    return 0;
}

/// Lists files in a model repository.
///
/// Works with local paths, HuggingFace model IDs, and cached models.
/// For remote models, may make a network request.
///
/// Parameters:
///   model_path: Model path, ID, or URI
///   token: Optional HuggingFace API token
///   out: Output pointer for file list (caller must free with talu_repo_string_list_free)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_list(
    model_path: [*:0]const u8,
    token: ?[*:0]const u8,
    out: *?*StringList,
) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;

    const tok: ?[]const u8 = if (token) |t| std.mem.span(t) else null;
    const files = repository.listFiles(allocator, std.mem.span(model_path), .{ .token = tok }) catch |e| {
        setErr("list", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer {
        for (files) |f| allocator.free(f);
        allocator.free(files);
    }

    const list = allocator.create(StringList) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    list.* = ffi.StringList.fromSlices(allocator, files) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    out.* = list;
    return 0;
}

/// Searches for models on HuggingFace Hub.
///
/// Requires network access to the HuggingFace API.
///
/// Parameters:
///   query: Search query string
///   limit: Maximum number of results to return
///   token: Optional HuggingFace API token
///   endpoint_url: Optional custom HF endpoint URL (overrides HF_ENDPOINT env var)
///   out: Output pointer for results (caller must free with talu_repo_string_list_free)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_search(
    query: [*:0]const u8,
    limit: usize,
    token: ?[*:0]const u8,
    endpoint_url: ?[*:0]const u8,
    out: *?*StringList,
) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;

    const tok: ?[]const u8 = if (token) |t| std.mem.span(t) else null;
    const endpoint: ?[]const u8 = if (endpoint_url) |e| std.mem.span(e) else null;
    const results = repository.searchModels(allocator, std.mem.span(query), .{
        .token = tok,
        .limit = limit,
        .endpoint_url = endpoint,
    }) catch |e| {
        setErr("search", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer {
        for (results) |r| allocator.free(r);
        allocator.free(results);
    }

    const list = allocator.create(StringList) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    list.* = ffi.StringList.fromSlices(allocator, results) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    out.* = list;
    return 0;
}

// =============================================================================
// Rich Search Operations
// =============================================================================

pub const SearchResultList = repository.SearchResultListC;

/// Searches for models on HuggingFace Hub with rich metadata.
///
/// Returns structured results including downloads, likes, last modified date,
/// and pipeline tag for each model.
///
/// Parameters:
///   query: Search query string
///   limit: Maximum number of results to return
///   token: Optional HuggingFace API token
///   endpoint_url: Optional custom HF endpoint URL (overrides HF_ENDPOINT env var)
///   filter: Optional pipeline filter (e.g., "text-generation"). Null defaults to "text-generation".
///   sort: Sort mode (0=trending, 1=downloads, 2=likes, 3=lastModified)
///   direction: Sort direction (0=descending, 1=ascending)
///   out: Output pointer for results (caller must free with talu_repo_search_result_free)
///
/// Returns 0 on success, error code on failure.
pub export fn talu_repo_search_rich(
    query: [*:0]const u8,
    limit: usize,
    token: ?[*:0]const u8,
    endpoint_url: ?[*:0]const u8,
    filter: ?[*:0]const u8,
    sort: u8,
    direction: u8,
    library: ?[*:0]const u8,
    out: *?*SearchResultList,
) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;

    const tok: ?[]const u8 = if (token) |t| std.mem.span(t) else null;
    const endpoint: ?[]const u8 = if (endpoint_url) |e| std.mem.span(e) else null;
    const filt: ?[]const u8 = if (filter) |f| std.mem.span(f) else null;
    const lib: ?[]const u8 = if (library) |l| std.mem.span(l) else null;

    const sort_enum: repository.SearchSort = @enumFromInt(@min(sort, 3));
    const dir_enum: repository.SearchDirection = @enumFromInt(@min(direction, 1));

    const results = repository.searchModelsRich(allocator, std.mem.span(query), .{
        .token = tok,
        .limit = limit,
        .endpoint_url = endpoint,
        .filter = filt,
        .library = lib,
        .sort = sort_enum,
        .direction = dir_enum,
    }) catch |e| {
        setErr("search_rich", e);
        return @intFromEnum(error_codes.errorToCode(e));
    };
    defer {
        for (results) |*r| r.deinit(allocator);
        allocator.free(results);
    }

    const list = allocator.create(SearchResultList) catch return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    list.* = repository.SearchResultListC.fromResults(allocator, results) catch {
        allocator.destroy(list);
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    out.* = list;
    return 0;
}

/// Gets the number of entries in a search result list.
pub export fn talu_repo_search_result_count(list: ?*const SearchResultList) callconv(.c) usize {
    return if (list) |l| l.entries.len else 0;
}

/// Gets the model ID at the specified index in a search result list.
///
/// The returned string is owned by the list and valid until talu_repo_search_result_free().
pub export fn talu_repo_search_result_get_id(list: ?*const SearchResultList, idx: usize, out: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const l = list orelse return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    if (idx >= l.entries.len) return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = l.entries[idx].model_id.ptr;
    return 0;
}

/// Gets the download count at the specified index.
pub export fn talu_repo_search_result_get_downloads(list: ?*const SearchResultList, idx: usize) callconv(.c) i64 {
    const l = list orelse return 0;
    if (idx >= l.entries.len) return 0;
    return l.entries[idx].downloads;
}

/// Gets the like count at the specified index.
pub export fn talu_repo_search_result_get_likes(list: ?*const SearchResultList, idx: usize) callconv(.c) i64 {
    const l = list orelse return 0;
    if (idx >= l.entries.len) return 0;
    return l.entries[idx].likes;
}

/// Gets the last modified date string at the specified index.
///
/// The returned string is owned by the list and valid until talu_repo_search_result_free().
pub export fn talu_repo_search_result_get_last_modified(list: ?*const SearchResultList, idx: usize, out: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const l = list orelse return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    if (idx >= l.entries.len) return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = l.entries[idx].last_modified.ptr;
    return 0;
}

/// Gets the total parameter count at the specified index.
/// Returns 0 if the list/index is invalid or the model has no safetensors metadata.
pub export fn talu_repo_search_result_get_params(list: ?*const SearchResultList, idx: usize) callconv(.c) i64 {
    const l = list orelse return 0;
    if (idx >= l.entries.len) return 0;
    return l.entries[idx].params_total;
}

/// Gets the pipeline tag string at the specified index.
///
/// The returned string is owned by the list and valid until talu_repo_search_result_free().
pub export fn talu_repo_search_result_get_pipeline_tag(list: ?*const SearchResultList, idx: usize, out: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const l = list orelse return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    if (idx >= l.entries.len) return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = l.entries[idx].pipeline_tag.ptr;
    return 0;
}

/// Frees a search result list and all its entries.
///
/// Safe to call with null (no-op).
pub export fn talu_repo_search_result_free(list: ?*SearchResultList) callconv(.c) void {
    const l = list orelse return;
    l.deinit(allocator);
    allocator.destroy(l);
}

// =============================================================================
// StringList Operations
// =============================================================================

/// Gets the number of strings in a string list.
///
/// Returns 0 if list is null.
pub export fn talu_repo_string_list_count(list: ?*const StringList) callconv(.c) usize {
    return if (list) |l| l.items.len else 0;
}

/// Gets the string at the specified index in a string list.
///
/// The returned string is owned by the list and valid until talu_repo_string_list_free().
///
/// Parameters:
///   list: String list
///   idx: Zero-based index
///   out: Output pointer for string (do not free)
///
/// Returns 0 on success, error code on invalid handle or out-of-bounds index.
pub export fn talu_repo_string_list_get(list: ?*const StringList, idx: usize, out: *?[*:0]const u8) callconv(.c) i32 {
    capi_error.clearError();
    out.* = null;
    const l = list orelse return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    if (idx >= l.items.len) return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    out.* = l.items[idx].ptr;
    return 0;
}

/// Frees a string list and all its entries.
///
/// Safe to call with null (no-op).
pub export fn talu_repo_string_list_free(list: ?*StringList) callconv(.c) void {
    const l = list orelse return;
    for (l.items) |item| allocator.free(item);
    allocator.free(l.items);
    allocator.destroy(l);
}

//! HF Hub Integration
//!
//! Downloads models from the hub and manages the local cache.

const std = @import("std");
const http = @import("http.zig");
const cache = @import("../repository/cache.zig");
const resolver = @import("../repository/resolver.zig");
const Bundle = @import("../repository/bundle.zig").Bundle;
const json_mod = @import("../json/root.zig");
const capi = @import("../../capi/error.zig");
const log = @import("../../log.zig");
const progress_api = @import("../../capi/progress.zig");

const ProgressContext = progress_api.ProgressContext;

/// Context for byte-level progress callback bridge.
/// Adapts http.ProgressCallback to the unified progress API.
const ByteProgressContext = struct {
    progress: ProgressContext,
    line_id: u8,
    file_name: []const u8,
    last_log_ms: i64,
    last_percent: i64,
};

/// Bridge callback from http.ProgressCallback to unified progress API.
fn byteProgressCallback(downloaded: u64, total: u64, user_data: ?*anyopaque) void {
    const ctx: *ByteProgressContext = @ptrCast(@alignCast(user_data orelse return));
    // Update the byte progress line (line_id=1)
    // We use updateLine with the current bytes downloaded
    ctx.progress.emit(.{
        .line_id = ctx.line_id,
        .action = .update,
        .current = downloaded,
        .total = total,
        .label = null,
        .message = null,
        .unit = null,
    });

    if (total == 0) return;
    const now_ms = std.time.milliTimestamp();
    const percent: i64 = @intCast((downloaded * 100) / total);
    const should_log = (now_ms - ctx.last_log_ms) >= 3000 or percent == 100;
    if (should_log and percent != ctx.last_percent) {
        ctx.last_log_ms = now_ms;
        ctx.last_percent = percent;
        var buf: [128]u8 = undefined;
        const downloaded_mb: u64 = downloaded / (1024 * 1024);
        const total_mb: u64 = total / (1024 * 1024);
        const msg = std.fmt.bufPrint(&buf, "Downloading model ({d} MB / {d} MB) {d}%", .{
            downloaded_mb,
            total_mb,
            percent,
        }) catch return;
        log.info("fetch", msg, .{ .file = ctx.file_name });
    }
}

/// Default HF Hub API endpoint (used when no override is provided)
const DEFAULT_HF_ENDPOINT = "https://huggingface.co";

/// Temp file scope - automatically cleans up on error, commits on success.
const TempDownload = struct {
    path: []const u8,
    allocator: std.mem.Allocator,
    committed: bool = false,

    fn init(allocator: std.mem.Allocator, dest_path: []const u8) error{OutOfMemory}!TempDownload {
        return .{
            .path = try std.fmt.allocPrint(allocator, "{s}.download", .{dest_path}),
            .allocator = allocator,
        };
    }

    fn commit(self: *TempDownload, dest_path: []const u8) DownloadError!void {
        std.fs.cwd().rename(self.path, dest_path) catch {
            return DownloadError.FileRenameFailed;
        };
        self.committed = true;
    }

    fn deinit(self: *TempDownload) void {
        if (!self.committed) {
            std.fs.cwd().deleteFile(self.path) catch {};
        }
        self.allocator.free(self.path);
    }
};

/// Map filesystem errors to DownloadError.
fn mapMakeDirError(err: anytype) DownloadError {
    return switch (err) {
        error.AccessDenied, error.PermissionDenied => DownloadError.PermissionDenied,
        error.ReadOnlyFileSystem => DownloadError.ReadOnlyFileSystem,
        error.NoSpaceLeft => DownloadError.NoSpaceLeft,
        error.DiskQuota => DownloadError.DiskQuota,
        else => DownloadError.Unexpected,
    };
}

/// Map file creation errors to DownloadError.
fn mapCreateFileError(err: anytype) DownloadError {
    return switch (err) {
        error.AccessDenied, error.PermissionDenied => DownloadError.PermissionDenied,
        error.NoSpaceLeft => DownloadError.NoSpaceLeft,
        else => DownloadError.Unexpected,
    };
}

const TempFileState = struct {
    file: std.fs.File,
    resume_from: u64,
};

fn openDownloadTempFile(temp_path: []const u8, allow_resume: bool) DownloadError!TempFileState {
    if (allow_resume) {
        if (std.fs.cwd().openFile(temp_path, .{ .mode = .read_write })) |existing| {
            const stat = existing.stat() catch |err| {
                existing.close();
                return switch (err) {
                    error.AccessDenied => DownloadError.PermissionDenied,
                    else => DownloadError.Unexpected,
                };
            };
            const resume_from = stat.size;
            existing.seekTo(resume_from) catch |err| {
                existing.close();
                return switch (err) {
                    error.AccessDenied => DownloadError.PermissionDenied,
                    else => DownloadError.Unexpected,
                };
            };
            return .{
                .file = existing,
                .resume_from = resume_from,
            };
        } else |err| switch (err) {
            error.FileNotFound => {},
            error.AccessDenied, error.PermissionDenied => return DownloadError.PermissionDenied,
            else => return DownloadError.Unexpected,
        }
    }

    const file = std.fs.cwd().createFile(temp_path, .{}) catch |err| {
        return mapCreateFileError(err);
    };
    return .{
        .file = file,
        .resume_from = 0,
    };
}

/// Download a file from URL to destination path with atomic temp-file handling.
/// Creates parent directories as needed. On success, file appears atomically at dest_path.
fn downloadFile(
    allocator: std.mem.Allocator,
    url: []const u8,
    dest_path: []const u8,
    token: ?[]const u8,
    progress_callback: ?http.ProgressCallback,
    progress_data: ?*anyopaque,
    allow_resume: bool,
) DownloadError!void {
    // Create parent directory if needed
    if (std.fs.path.dirname(dest_path)) |dir| {
        std.fs.cwd().makePath(dir) catch |err| {
            return mapMakeDirError(err);
        };
    }

    // Create temp file scope (auto-cleanup on error)
    var temp = TempDownload.init(allocator, dest_path) catch return DownloadError.OutOfMemory;
    defer temp.deinit();

    // Open the temp file for resume if available; otherwise create a fresh temp file.
    const temp_state = try openDownloadTempFile(temp.path, allow_resume);
    const resume_from = temp_state.resume_from;
    var file = temp_state.file;
    defer file.close();

    // Stream HTTP content to file
    http.downloadToFile(allocator, url, file, .{
        .token = token,
        .progress_callback = progress_callback,
        .progress_data = progress_data,
        .resume_from = resume_from,
    }) catch |err| {
        return err; // HttpError is a subset of DownloadError
    };

    // Atomically move temp file to final destination
    try temp.commit(dest_path);
}

pub const DownloadError = error{
    // Model/API errors
    InvalidModelId,
    ModelNotFound,
    ConfigNotFound,
    WeightsNotFound,
    ApiResponseParseError,
    // HTTP errors (from http.zig)
    Unauthorized,
    RateLimited,
    HttpError,
    ResponseTooLarge,
    NotFound,
    CurlInitFailed,
    CurlSetOptFailed,
    CurlPerformFailed,
    StreamWriteFailed,
    // Filesystem errors
    PermissionDenied, // EACCES - permission denied
    ReadOnlyFileSystem, // EROFS - read-only filesystem
    NoSpaceLeft, // ENOSPC - disk full
    DiskQuota, // EDQUOT - quota exceeded
    FileRenameFailed,
    // General
    OutOfMemory,
    Unexpected,
};

/// Download configuration
pub const DownloadConfig = struct {
    /// API token for private models (optional)
    token: ?[]const u8 = null,
    /// Unified progress context for progress reporting
    progress: ProgressContext = ProgressContext.NONE,
    /// Force re-download even if files exist
    force: bool = false,
    /// Custom HF endpoint URL (optional, overrides HF_ENDPOINT env var)
    /// Precedence: endpoint_url > HF_ENDPOINT env > DEFAULT_HF_ENDPOINT
    endpoint_url: ?[]const u8 = null,
};

/// Get the effective HF API base URL.
/// Precedence: config.endpoint_url > HF_ENDPOINT env > DEFAULT_HF_ENDPOINT
fn getEffectiveEndpoint(config_endpoint: ?[]const u8) []const u8 {
    // 1. Explicit config takes highest priority
    if (config_endpoint) |endpoint| {
        return endpoint;
    }
    // 2. Environment variable
    if (std.posix.getenv("HF_ENDPOINT")) |env_endpoint| {
        return std.mem.sliceTo(env_endpoint, 0);
    }
    // 3. Default
    return DEFAULT_HF_ENDPOINT;
}

/// JSON structure for HF API response (just what we need)
const HfModelInfo = struct {
    siblings: ?[]const Sibling = null,

    const Sibling = struct {
        rfilename: []const u8,
    };
};

/// Fetch the list of files in a model repository from the hub API.
/// Returns owned slice of owned strings. Caller must free both.
pub fn fetchFileList(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    download_config: DownloadConfig,
) ![][]const u8 {
    // Initialize curl (safe to call multiple times - reference counted)
    http.globalInit();

    const base_url = getEffectiveEndpoint(download_config.endpoint_url);
    const model_api_url = try std.fmt.allocPrint(allocator, "{s}/api/models/{s}", .{ base_url, model_id });
    defer allocator.free(model_api_url);

    const response_bytes = http.fetch(allocator, model_api_url, .{
        .token = download_config.token,
        .max_response_bytes = 10 * 1024 * 1024,
    }) catch |err| {
        capi.setContext("url={s}", .{model_api_url});
        return err;
    };
    defer allocator.free(response_bytes);

    // Parse JSON using std.json for robustness
    const parsed_info = json_mod.parseStruct(allocator, HfModelInfo, response_bytes, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
        .ignore_unknown_fields = true,
    }) catch |err| {
        capi.setContext("model_id={s}", .{model_id});
        return switch (err) {
            error.InputTooLarge => DownloadError.ApiResponseParseError,
            error.InputTooDeep => DownloadError.ApiResponseParseError,
            error.StringTooLong => DownloadError.ApiResponseParseError,
            error.InvalidJson => DownloadError.ApiResponseParseError,
            error.OutOfMemory => DownloadError.OutOfMemory,
        };
    };
    defer parsed_info.deinit();

    var repo_file_names = std.ArrayListUnmanaged([]const u8){};
    errdefer {
        for (repo_file_names.items) |name| allocator.free(name);
        repo_file_names.deinit(allocator);
    }

    if (parsed_info.value.siblings) |siblings| {
        for (siblings) |sibling| {
            const filename = sibling.rfilename;
            // Skip hidden files and directories
            if (!std.mem.startsWith(u8, filename, ".") and
                std.mem.indexOf(u8, filename, "/") == null)
            {
                try repo_file_names.append(allocator, try allocator.dupe(u8, filename));
            }
        }
    }

    return repo_file_names.toOwnedSlice(allocator);
}

/// Fetch a model from the hub and return the path to the downloaded model.
///
/// model_id: e.g., "org/model-name"
/// Returns: Owned path to the downloaded snapshot directory. Caller must free.
pub fn fetchModel(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    download_config: DownloadConfig,
) ![]const u8 {
    // Validate model ID (must contain /)
    if (std.mem.indexOf(u8, model_id, "/") == null) {
        return DownloadError.InvalidModelId;
    }

    // Check if already cached (unless force download)
    if (!download_config.force) {
        if (cache.getCachedPath(allocator, model_id) catch null) |cached_path| {
            log.info("fetch", "Model already cached", .{ .path = cached_path });
            return cached_path;
        }
    }

    // Fetch file list from API
    log.info("fetch", "Fetching file list", .{ .model_id = model_id });
    const repo_file_names = fetchFileList(allocator, model_id, download_config) catch |err| {
        // Map Unauthorized to ModelNotFound when no token is provided
        // HuggingFace returns 401 for non-existent public repos
        if (err == error.Unauthorized and download_config.token == null) {
            log.err("fetch", "Model not found (API returned 401 without token)", .{ .model_id = model_id }, @src());
            return DownloadError.ModelNotFound;
        }
        if (err == error.NotFound) {
            log.err("fetch", "Model not found on HuggingFace", .{ .model_id = model_id }, @src());
            return DownloadError.ModelNotFound;
        }
        return err;
    };
    defer {
        for (repo_file_names) |name| allocator.free(name);
        allocator.free(repo_file_names);
    }

    if (repo_file_names.len == 0) {
        capi.setContext("model_id={s}", .{model_id});
        return DownloadError.ModelNotFound;
    }

    log.info("fetch", "Found files to download", .{ .count = repo_file_names.len });

    // Create cache directory structure
    const cache_dir = cache.getModelCacheDir(allocator, model_id) catch return DownloadError.OutOfMemory;
    defer allocator.free(cache_dir);

    // Use a fixed snapshot hash for simplicity
    const snapshot_revision = "main";
    const snapshot_dir_path = std.fs.path.join(allocator, &.{ cache_dir, "snapshots", snapshot_revision }) catch return DownloadError.OutOfMemory;
    errdefer allocator.free(snapshot_dir_path);

    std.fs.cwd().makePath(snapshot_dir_path) catch |err| {
        return mapMakeDirError(err);
    };

    log.info("fetch", "Downloading model", .{ .model_id = model_id, .path = snapshot_dir_path });

    // Emit progress: start downloading files
    const progress = download_config.progress;
    progress.addLine(0, "Downloading", repo_file_names.len, null, "files");

    // Download all files from the repository
    var has_config = false;
    var files_downloaded: u64 = 0;

    const base_url = getEffectiveEndpoint(download_config.endpoint_url);

    var msg_buf: [256]u8 = undefined; // filled by @memcpy before use

    // Byte-level progress context (uses line_id=1)
    var byte_ctx = ByteProgressContext{
        .progress = progress,
        .line_id = 1,
        .file_name = "",
        .last_log_ms = 0,
        .last_percent = -1,
    };

    for (repo_file_names) |filename| {
        const file_url = std.fmt.allocPrint(allocator, "{s}/{s}/resolve/main/{s}", .{ base_url, model_id, filename }) catch continue;
        defer allocator.free(file_url);

        const destination_path = std.fs.path.join(allocator, &.{ snapshot_dir_path, filename }) catch continue;
        defer allocator.free(destination_path);

        // Skip files that are already present unless force-download is requested.
        if (!download_config.force) {
            if (std.fs.cwd().openFile(destination_path, .{})) |existing_file| {
                existing_file.close();

                // Remove stale partial if a completed file is present.
                if (std.fmt.allocPrint(allocator, "{s}.download", .{destination_path})) |stale_temp_path| {
                    defer allocator.free(stale_temp_path);
                    std.fs.cwd().deleteFile(stale_temp_path) catch {};
                } else |_| {}

                files_downloaded += 1;
                if (std.mem.eql(u8, filename, "config.json")) has_config = true;

                // Report cached progress so overall progress advances immediately.
                const prefix = "cached: ";
                @memcpy(msg_buf[0..prefix.len], prefix);
                const copy_len = @min(filename.len, msg_buf.len - prefix.len - 1);
                @memcpy(msg_buf[prefix.len .. prefix.len + copy_len], filename[0..copy_len]);
                msg_buf[prefix.len + copy_len] = 0;
                progress.updateLine(0, files_downloaded, @ptrCast(&msg_buf));
                continue;
            } else |_| {}
        }

        // Update file-level progress with current filename
        const copy_len = @min(filename.len, msg_buf.len - 1);
        @memcpy(msg_buf[0..copy_len], filename[0..copy_len]);
        msg_buf[copy_len] = 0;
        progress.updateLine(0, files_downloaded, @ptrCast(&msg_buf));

        // Add byte-level progress line for this file (indeterminate total until curl reports)
        progress.addLine(1, @ptrCast(&msg_buf), 0, null, "bytes");
        byte_ctx.file_name = filename;
        byte_ctx.last_log_ms = 0;
        byte_ctx.last_percent = -1;

        downloadFile(
            allocator,
            file_url,
            destination_path,
            download_config.token,
            byteProgressCallback,
            @ptrCast(&byte_ctx),
            !download_config.force,
        ) catch |err| {
            // Complete byte progress line on error
            progress.completeLine(1);
            // Only error for essential files, skip others silently
            if (std.mem.eql(u8, filename, "config.json")) {
                progress.completeLine(0);
                capi.setContext("file={s}, model_id={s}, err={s}", .{ filename, model_id, @errorName(err) });
                // Return filesystem errors directly, wrap HTTP errors as ConfigNotFound
                return switch (err) {
                    DownloadError.PermissionDenied,
                    DownloadError.ReadOnlyFileSystem,
                    DownloadError.NoSpaceLeft,
                    DownloadError.DiskQuota,
                    => err,
                    else => DownloadError.ConfigNotFound,
                };
            } else {
                continue;
            }
        };

        // Complete byte progress line for this file
        progress.completeLine(1);

        files_downloaded += 1;

        // Track what we've downloaded
        if (std.mem.eql(u8, filename, "config.json")) has_config = true;
    }

    // Emit progress: complete
    progress.completeLine(0);

    // Verify we have the minimum required files
    if (!has_config) {
        capi.setContext("model_id={s}", .{model_id});
        return DownloadError.ConfigNotFound;
    }
    // Note: missing weights is not a fatal error - the download still succeeds

    log.info("fetch", "Download complete", .{});

    return snapshot_dir_path;
}

/// Fetch a model and return a Bundle (convenience wrapper)
fn fetchModelBundle(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    download_config: DownloadConfig,
) !Bundle {
    const snapshot_path = try fetchModel(allocator, model_id, download_config);
    defer allocator.free(snapshot_path);
    return resolver.resolve(allocator, snapshot_path);
}

// =============================================================================
// Single File Download
// =============================================================================

/// Fetch a single file from a model repository.
///
/// Downloads one file (e.g., "config.json") without fetching the full model.
/// Uses the same cache directory layout as fetchModel (snapshots/main/).
///
/// model_id: e.g., "org/model-name"
/// filename: e.g., "config.json"
/// Returns: Owned path to the downloaded file. Caller must free.
pub fn fetchFile(
    allocator: std.mem.Allocator,
    model_id: []const u8,
    filename: []const u8,
    download_config: DownloadConfig,
) ![]const u8 {
    // Validate model ID (must contain /)
    if (std.mem.indexOf(u8, model_id, "/") == null) {
        return DownloadError.InvalidModelId;
    }

    const cache_dir = cache.getModelCacheDir(allocator, model_id) catch
        return DownloadError.OutOfMemory;
    defer allocator.free(cache_dir);

    const snapshot_revision = "main";
    const snapshot_dir_path = std.fs.path.join(
        allocator,
        &.{ cache_dir, "snapshots", snapshot_revision },
    ) catch return DownloadError.OutOfMemory;
    defer allocator.free(snapshot_dir_path);

    // Create snapshot directory
    std.fs.cwd().makePath(snapshot_dir_path) catch |err| {
        return mapMakeDirError(err);
    };

    const destination_path = std.fs.path.join(
        allocator,
        &.{ snapshot_dir_path, filename },
    ) catch return DownloadError.OutOfMemory;
    errdefer allocator.free(destination_path);

    // Check if file already exists (unless force download)
    if (!download_config.force) {
        if (std.fs.cwd().openFile(destination_path, .{})) |file| {
            file.close();
            log.info("fetch_file", "File already cached", .{
                .path = destination_path,
            });
            return destination_path;
        } else |_| {}
    }

    // Build direct file URL
    const base_url = getEffectiveEndpoint(download_config.endpoint_url);
    const file_url = std.fmt.allocPrint(
        allocator,
        "{s}/{s}/resolve/main/{s}",
        .{ base_url, model_id, filename },
    ) catch return DownloadError.OutOfMemory;
    defer allocator.free(file_url);

    log.info("fetch_file", "Downloading file", .{ .url = file_url });

    // Download the file (no progress callback for single file)
    downloadFile(
        allocator,
        file_url,
        destination_path,
        download_config.token,
        null,
        null,
        !download_config.force,
    ) catch |err| {
        capi.setContext("file={s}, model_id={s}", .{ filename, model_id });
        return err;
    };

    log.info("fetch_file", "Download complete", .{});
    return destination_path;
}

// =============================================================================
// Model Search
// =============================================================================

/// Sort field for search results.
pub const SearchSort = enum(u8) {
    /// Default HuggingFace trending ranking (omit sort param).
    trending = 0,
    /// Sort by total download count.
    downloads = 1,
    /// Sort by number of likes.
    likes = 2,
    /// Sort by last modification date.
    last_modified = 3,

    /// HF API query-parameter value, or null for the default (trending).
    pub fn toParam(self: SearchSort) ?[]const u8 {
        return switch (self) {
            .trending => null,
            .downloads => "downloads",
            .likes => "likes",
            .last_modified => "lastModified",
        };
    }
};

/// Sort direction for search results.
pub const SearchDirection = enum(u8) {
    descending = 0,
    ascending = 1,

    pub fn toParam(self: SearchDirection) []const u8 {
        return switch (self) {
            .descending => "-1",
            .ascending => "1",
        };
    }
};

/// Search configuration
pub const SearchConfig = struct {
    /// API token for authentication (optional)
    token: ?[]const u8 = null,
    /// Maximum number of results (default 10)
    limit: usize = 10,
    /// Filter by pipeline tag (e.g., "text-generation")
    filter: ?[]const u8 = "text-generation",
    /// Filter by library tag (e.g., "safetensors", "transformers", "mlx").
    /// Appended to filter as an AND condition.
    library: ?[]const u8 = null,
    /// Custom HF endpoint URL (optional, overrides HF_ENDPOINT env var)
    endpoint_url: ?[]const u8 = null,
    /// Sort field (default: trending)
    sort: SearchSort = .trending,
    /// Sort direction (default: descending)
    direction: SearchDirection = .descending,
};

/// A single search result with metadata from the HuggingFace API.
pub const SearchResult = struct {
    model_id: []const u8,
    downloads: i64,
    likes: i64,
    last_modified: []const u8,
    pipeline_tag: []const u8,
    /// Total parameter count from safetensors metadata (0 if unavailable).
    params_total: i64,

    pub fn deinit(self: *const SearchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_id);
        if (self.last_modified.len > 0) allocator.free(self.last_modified);
        if (self.pipeline_tag.len > 0) allocator.free(self.pipeline_tag);
    }
};

/// Parse search results JSON from HuggingFace API.
/// Returns owned slice of model IDs. Caller must free both slice and strings.
pub fn parseSearchResults(allocator: std.mem.Allocator, response_body: []const u8) ![][]const u8 {
    const parsed = json_mod.parseValue(allocator, response_body, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => DownloadError.ApiResponseParseError,
            error.InputTooDeep => DownloadError.ApiResponseParseError,
            error.StringTooLong => DownloadError.ApiResponseParseError,
            error.InvalidJson => DownloadError.ApiResponseParseError,
            error.OutOfMemory => DownloadError.OutOfMemory,
        };
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .array) {
        return DownloadError.ApiResponseParseError;
    }

    var model_ids = std.ArrayListUnmanaged([]const u8){};
    errdefer {
        for (model_ids.items) |id| allocator.free(id);
        model_ids.deinit(allocator);
    }

    for (root.array.items) |entry| {
        if (entry != .object) continue;

        // Use "modelId" field
        const model_id_value = entry.object.get("modelId") orelse continue;
        if (model_id_value != .string) continue;

        const model_id = model_id_value.string;
        // Only include if it looks like org/model format
        if (cache.isModelId(model_id)) {
            const id_copy = try allocator.dupe(u8, model_id);
            try model_ids.append(allocator, id_copy);
        }
    }

    return model_ids.toOwnedSlice(allocator);
}

/// Search for models on the HuggingFace Hub.
/// Returns owned slice of model IDs. Caller must free both slice and strings.
pub fn searchModels(
    allocator: std.mem.Allocator,
    query: []const u8,
    config: SearchConfig,
) ![][]const u8 {
    // Initialize curl (safe to call multiple times - reference counted)
    http.globalInit();

    const limit = if (config.limit == 0) 10 else config.limit;
    const base_url = getEffectiveEndpoint(config.endpoint_url);

    // Build search URL with percent-encoded query.
    var url_buf = std.ArrayListUnmanaged(u8){};
    defer url_buf.deinit(allocator);
    const writer = url_buf.writer(allocator);
    try writer.print("{s}/api/models?search=", .{base_url});
    try writePercentEncoded(writer, query);
    if (config.filter) |filter| {
        try writer.print("&filter={s}", .{filter});
    }
    try writer.print("&limit={d}", .{limit});
    const search_url = url_buf.items;

    const response_body = http.fetch(allocator, search_url, .{
        .token = config.token,
        .max_response_bytes = 10 * 1024 * 1024,
    }) catch |err| {
        return err;
    };
    defer allocator.free(response_body);

    return parseSearchResults(allocator, response_body);
}

/// Parse search results JSON into rich SearchResult structs.
/// Extracts modelId, downloads, likes, lastModified, and pipeline_tag.
/// Caller must free each result via SearchResult.deinit() and the slice itself.
pub fn parseSearchResultsRich(allocator: std.mem.Allocator, response_body: []const u8) ![]SearchResult {
    const parsed = json_mod.parseValue(allocator, response_body, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => DownloadError.ApiResponseParseError,
            error.InputTooDeep => DownloadError.ApiResponseParseError,
            error.StringTooLong => DownloadError.ApiResponseParseError,
            error.InvalidJson => DownloadError.ApiResponseParseError,
            error.OutOfMemory => DownloadError.OutOfMemory,
        };
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .array) {
        return DownloadError.ApiResponseParseError;
    }

    var results = std.ArrayListUnmanaged(SearchResult){};
    errdefer {
        for (results.items) |*r| r.deinit(allocator);
        results.deinit(allocator);
    }

    for (root.array.items) |entry| {
        if (entry != .object) continue;

        // The expand[] API returns "id"; the default API returns "modelId".
        const model_id_value = entry.object.get("id") orelse
            (entry.object.get("modelId") orelse continue);
        if (model_id_value != .string) continue;

        const model_id = model_id_value.string;
        if (!cache.isModelId(model_id)) continue;

        const id_copy = try allocator.dupe(u8, model_id);
        errdefer allocator.free(id_copy);

        // Extract integer fields with default 0.
        const downloads: i64 = blk: {
            const v = entry.object.get("downloads") orelse break :blk 0;
            if (v == .integer) break :blk v.integer;
            break :blk 0;
        };
        const likes: i64 = blk: {
            const v = entry.object.get("likes") orelse break :blk 0;
            if (v == .integer) break :blk v.integer;
            break :blk 0;
        };

        // Extract string fields with default "".
        const last_modified: []const u8 = blk: {
            const v = entry.object.get("lastModified") orelse break :blk &.{};
            if (v != .string) break :blk &.{};
            break :blk try allocator.dupe(u8, v.string);
        };
        errdefer if (last_modified.len > 0) allocator.free(last_modified);

        const pipeline_tag: []const u8 = blk: {
            const v = entry.object.get("pipeline_tag") orelse break :blk &.{};
            if (v != .string) break :blk &.{};
            break :blk try allocator.dupe(u8, v.string);
        };

        // Extract safetensors.total (parameter count) if available.
        const params_total: i64 = blk: {
            const st = entry.object.get("safetensors") orelse break :blk 0;
            if (st != .object) break :blk 0;
            const total = st.object.get("total") orelse break :blk 0;
            if (total == .integer) break :blk total.integer;
            break :blk 0;
        };

        try results.append(allocator, .{
            .model_id = id_copy,
            .downloads = downloads,
            .likes = likes,
            .last_modified = last_modified,
            .pipeline_tag = pipeline_tag,
            .params_total = params_total,
        });
    }

    return results.toOwnedSlice(allocator);
}

/// Search for models on the HuggingFace Hub with rich metadata.
/// Returns owned slice of SearchResult. Caller must free results and slice.
pub fn searchModelsRich(
    allocator: std.mem.Allocator,
    query: []const u8,
    config: SearchConfig,
) ![]SearchResult {
    http.globalInit();

    const limit = if (config.limit == 0) 10 else config.limit;
    const base_url = getEffectiveEndpoint(config.endpoint_url);

    // Build search URL with optional sort/direction params.
    var url_buf = std.ArrayListUnmanaged(u8){};
    defer url_buf.deinit(allocator);

    const writer = url_buf.writer(allocator);

    try writer.print("{s}/api/models?search=", .{base_url});
    try writePercentEncoded(writer, query);
    try writer.print("&limit={d}", .{limit});

    // Build filter tag list (task + library are AND-combined).
    if (config.filter) |filter| {
        try writer.print("&filter={s}", .{filter});
        if (config.library) |lib| try writer.print(",{s}", .{lib});
    } else if (config.library) |lib| {
        try writer.print("&filter={s}", .{lib});
    }

    if (config.sort.toParam()) |sort_param| {
        try writer.print("&sort={s}&direction={s}", .{ sort_param, config.direction.toParam() });
    }

    // Using expand[] switches the HF API to explicit field mode: only
    // expanded fields (plus _id, id, trendingScore) are returned. We must
    // expand every field we need. Brackets are percent-encoded for curl.
    try writer.writeAll("&expand%5B%5D=safetensors&expand%5B%5D=downloads&expand%5B%5D=likes&expand%5B%5D=lastModified&expand%5B%5D=pipeline_tag");

    const search_url = url_buf.items;

    const response_body = http.fetch(allocator, search_url, .{
        .token = config.token,
        .max_response_bytes = 10 * 1024 * 1024,
    }) catch |err| {
        return err;
    };
    defer allocator.free(response_body);

    return parseSearchResultsRich(allocator, response_body);
}

/// Percent-encode a query parameter value, writing the result to `writer`.
/// Encodes everything except unreserved characters (A-Z, a-z, 0-9, - _ . ~).
fn writePercentEncoded(writer: anytype, raw: []const u8) !void {
    for (raw) |c| {
        switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '-', '_', '.', '~' => try writer.writeByte(c),
            else => try writer.print("%{X:0>2}", .{c}),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

test "TempDownload: creates correct temp path" {
    const allocator = std.testing.allocator;
    var temp = try TempDownload.init(allocator, "/tmp/test_file.bin");
    defer temp.deinit();

    try std.testing.expectEqualStrings("/tmp/test_file.bin.download", temp.path);
    try std.testing.expect(!temp.committed);
}

test "TempDownload: cleanup on deinit without commit" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_hf_test_cleanup.bin";
    var temp = try TempDownload.init(allocator, test_path);

    // Save the path before deinit frees it
    var path_copy: [128]u8 = undefined;
    const path_len = temp.path.len;
    @memcpy(path_copy[0..path_len], temp.path);
    const saved_path = path_copy[0..path_len];

    // Create the temp file
    const file = std.fs.cwd().createFile(saved_path, .{}) catch |err| {
        temp.deinit();
        return err;
    };
    file.close();

    // deinit should delete the file since we didn't commit
    temp.deinit();

    // Verify file was deleted
    const open_result = std.fs.cwd().openFile(saved_path, .{});
    try std.testing.expectError(error.FileNotFound, open_result);
}

test "TempDownload: commit prevents cleanup deletion" {
    const allocator = std.testing.allocator;

    const test_path = "/tmp/talu_hf_test_commit.bin";
    const temp_path = "/tmp/talu_hf_test_commit.bin.download";

    // Clean up from previous test runs
    std.fs.cwd().deleteFile(test_path) catch {};
    std.fs.cwd().deleteFile(temp_path) catch {};

    var temp = try TempDownload.init(allocator, test_path);

    // Create the temp file with content
    const file = std.fs.cwd().createFile(temp.path, .{}) catch |err| {
        temp.deinit();
        return err;
    };
    try file.writeAll("test content");
    file.close();

    // Commit moves temp to final path
    try temp.commit(test_path);
    try std.testing.expect(temp.committed);

    // deinit should NOT delete because we committed
    temp.deinit();

    // Final file should exist
    const final_file = try std.fs.cwd().openFile(test_path, .{});
    defer final_file.close();
    var buf: [20]u8 = undefined;
    const len = try final_file.readAll(&buf);
    try std.testing.expectEqualStrings("test content", buf[0..len]);

    // Clean up
    std.fs.cwd().deleteFile(test_path) catch {};
}

test "mapMakeDirError: maps filesystem errors correctly" {
    // Test via the MakeDirError type from posix
    const MakeDirError = std.posix.MakeDirError;
    try std.testing.expectEqual(DownloadError.PermissionDenied, mapMakeDirError(@as(MakeDirError, error.AccessDenied)));
    try std.testing.expectEqual(DownloadError.PermissionDenied, mapMakeDirError(@as(MakeDirError, error.PermissionDenied)));
    try std.testing.expectEqual(DownloadError.ReadOnlyFileSystem, mapMakeDirError(@as(MakeDirError, error.ReadOnlyFileSystem)));
    try std.testing.expectEqual(DownloadError.NoSpaceLeft, mapMakeDirError(@as(MakeDirError, error.NoSpaceLeft)));
    try std.testing.expectEqual(DownloadError.DiskQuota, mapMakeDirError(@as(MakeDirError, error.DiskQuota)));
    try std.testing.expectEqual(DownloadError.Unexpected, mapMakeDirError(@as(MakeDirError, error.FileNotFound)));
}

test "mapCreateFileError: maps filesystem errors correctly" {
    // Test via the OpenError type from File
    const OpenError = std.fs.File.OpenError;
    try std.testing.expectEqual(DownloadError.PermissionDenied, mapCreateFileError(@as(OpenError, error.AccessDenied)));
    try std.testing.expectEqual(DownloadError.PermissionDenied, mapCreateFileError(@as(OpenError, error.PermissionDenied)));
    try std.testing.expectEqual(DownloadError.NoSpaceLeft, mapCreateFileError(@as(OpenError, error.NoSpaceLeft)));
    try std.testing.expectEqual(DownloadError.Unexpected, mapCreateFileError(@as(OpenError, error.FileNotFound)));
}

test "openDownloadTempFile resumes existing partial file" {
    const temp_path = "/tmp/talu_hf_resume_partial.download";
    std.fs.cwd().deleteFile(temp_path) catch {};
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    {
        var seed = try std.fs.cwd().createFile(temp_path, .{});
        defer seed.close();
        try seed.writeAll("abc");
    }

    var state = try openDownloadTempFile(temp_path, true);
    try std.testing.expectEqual(@as(u64, 3), state.resume_from);
    try state.file.writeAll("d");
    state.file.close();

    var file = try std.fs.cwd().openFile(temp_path, .{});
    defer file.close();
    var buf: [8]u8 = undefined;
    const n = try file.readAll(&buf);
    try std.testing.expectEqualStrings("abcd", buf[0..n]);
}

test "openDownloadTempFile truncates when resume is disabled" {
    const temp_path = "/tmp/talu_hf_resume_disabled.download";
    std.fs.cwd().deleteFile(temp_path) catch {};
    defer std.fs.cwd().deleteFile(temp_path) catch {};

    {
        var seed = try std.fs.cwd().createFile(temp_path, .{});
        defer seed.close();
        try seed.writeAll("abcdef");
    }

    var state = try openDownloadTempFile(temp_path, false);
    try std.testing.expectEqual(@as(u64, 0), state.resume_from);
    try state.file.writeAll("z");
    state.file.close();

    var file = try std.fs.cwd().openFile(temp_path, .{});
    defer file.close();
    var buf: [8]u8 = undefined;
    const n = try file.readAll(&buf);
    try std.testing.expectEqualStrings("z", buf[0..n]);
}

test "fetchModel: invalid model id" {
    const allocator = std.testing.allocator;
    const result = fetchModel(allocator, "invalid-no-slash", .{});
    try std.testing.expectError(DownloadError.InvalidModelId, result);
}

test "fetchFile: invalid model id" {
    const allocator = std.testing.allocator;
    const result = fetchFile(allocator, "invalid-no-slash", "config.json", .{});
    try std.testing.expectError(DownloadError.InvalidModelId, result);
}

test "fetchFile: signature verification" {
    const F = @TypeOf(fetchFile);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 4), info.params.len);
}

// -----------------------------------------------------------------------------
// Network-dependent function coverage
//
// The following functions perform HTTP requests and cannot be unit tested
// without network access. Their signatures are verified here; behavior is
// covered through:
//
// - fetchFileList: JSON parsing tested via HfModelInfo struct usage in
//   fetchFileList itself. Network behavior covered by integration tests.
//
// - searchModels: All parsing logic extracted to parseSearchResults() which
//   has comprehensive unit tests below. searchModels is a thin wrapper that
//   builds the URL and calls http.fetch + parseSearchResults.
//
// Integration tests in tests/io/ cover end-to-end behavior with real network.
// -----------------------------------------------------------------------------

test "fetchFileList: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires network access to HuggingFace API.
    // JSON parsing uses HfModelInfo struct; network tests in integration suite.
    const F = @TypeOf(fetchFileList);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 3), info.params.len);
}

test "searchModels: signature verification" {
    // Verify function signature compiles correctly.
    // Cannot unit test: requires network access to HuggingFace API.
    // All parsing logic tested via parseSearchResults tests below.
    const F = @TypeOf(searchModels);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 3), info.params.len);
}

test "parseSearchResults: parses valid response with multiple models" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "org-a/model-a", "downloads": 1000},
        \\  {"modelId": "org-b/model-b", "downloads": 500},
        \\  {"modelId": "org-c/model-c", "downloads": 200}
        \\]
    ;

    const results = try parseSearchResults(allocator, json);
    defer {
        for (results) |id| allocator.free(id);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 3), results.len);
    try std.testing.expectEqualStrings("org-a/model-a", results[0]);
    try std.testing.expectEqualStrings("org-b/model-b", results[1]);
    try std.testing.expectEqualStrings("org-c/model-c", results[2]);
}

test "parseSearchResults: returns empty slice for empty array" {
    const allocator = std.testing.allocator;

    const results = try parseSearchResults(allocator, "[]");
    defer allocator.free(results);

    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "parseSearchResults: filters out invalid model IDs" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "valid/model"},
        \\  {"modelId": "no-slash"},
        \\  {"modelId": "/leading-slash"},
        \\  {"modelId": "trailing-slash/"},
        \\  {"modelId": "too/many/slashes"},
        \\  {"modelId": "another/valid"}
        \\]
    ;

    const results = try parseSearchResults(allocator, json);
    defer {
        for (results) |id| allocator.free(id);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("valid/model", results[0]);
    try std.testing.expectEqualStrings("another/valid", results[1]);
}

test "parseSearchResults: skips entries without modelId field" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "org/model1"},
        \\  {"name": "missing-modelId"},
        \\  {"modelId": "org/model2"}
        \\]
    ;

    const results = try parseSearchResults(allocator, json);
    defer {
        for (results) |id| allocator.free(id);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("org/model1", results[0]);
    try std.testing.expectEqualStrings("org/model2", results[1]);
}

test "parseSearchResults: skips entries with non-string modelId" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "org/model1"},
        \\  {"modelId": 12345},
        \\  {"modelId": null},
        \\  {"modelId": "org/model2"}
        \\]
    ;

    const results = try parseSearchResults(allocator, json);
    defer {
        for (results) |id| allocator.free(id);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("org/model1", results[0]);
    try std.testing.expectEqualStrings("org/model2", results[1]);
}

test "parseSearchResults: returns error for non-array JSON" {
    const allocator = std.testing.allocator;

    const result = parseSearchResults(allocator, "{\"error\": \"not found\"}");
    try std.testing.expectError(DownloadError.ApiResponseParseError, result);
}

test "parseSearchResults: returns error for invalid JSON" {
    const allocator = std.testing.allocator;

    const result = parseSearchResults(allocator, "not valid json");
    try std.testing.expectError(DownloadError.ApiResponseParseError, result);
}

test "parseSearchResults: handles non-object array entries" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "org/model1"},
        \\  "string-entry",
        \\  123,
        \\  {"modelId": "org/model2"}
        \\]
    ;

    const results = try parseSearchResults(allocator, json);
    defer {
        for (results) |id| allocator.free(id);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("org/model1", results[0]);
    try std.testing.expectEqualStrings("org/model2", results[1]);
}

// -----------------------------------------------------------------------------
// parseSearchResultsRich tests
// -----------------------------------------------------------------------------

test "parseSearchResultsRich: parses full metadata" {
    const allocator = std.testing.allocator;

    // Uses "id" (expand[] API format) for first entry, "modelId" (default API) for second.
    const json =
        \\[
        \\  {
        \\    "id": "org-a/model-a",
        \\    "downloads": 12345,
        \\    "likes": 678,
        \\    "lastModified": "2025-01-15T10:30:00Z",
        \\    "pipeline_tag": "text-generation",
        \\    "safetensors": {"parameters": {"BF16": 751632384}, "total": 751632384}
        \\  },
        \\  {
        \\    "modelId": "org-b/model-b",
        \\    "downloads": 999,
        \\    "likes": 42,
        \\    "lastModified": "2024-12-01T00:00:00Z",
        \\    "pipeline_tag": "text2text-generation",
        \\    "safetensors": {"parameters": {"F16": 4022468096}, "total": 4022468096}
        \\  }
        \\]
    ;

    const results = try parseSearchResultsRich(allocator, json);
    defer {
        for (results) |*r| r.deinit(allocator);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);

    try std.testing.expectEqualStrings("org-a/model-a", results[0].model_id);
    try std.testing.expectEqual(@as(i64, 12345), results[0].downloads);
    try std.testing.expectEqual(@as(i64, 678), results[0].likes);
    try std.testing.expectEqualStrings("2025-01-15T10:30:00Z", results[0].last_modified);
    try std.testing.expectEqualStrings("text-generation", results[0].pipeline_tag);
    try std.testing.expectEqual(@as(i64, 751632384), results[0].params_total);

    try std.testing.expectEqualStrings("org-b/model-b", results[1].model_id);
    try std.testing.expectEqual(@as(i64, 999), results[1].downloads);
    try std.testing.expectEqual(@as(i64, 42), results[1].likes);
    try std.testing.expectEqualStrings("2024-12-01T00:00:00Z", results[1].last_modified);
    try std.testing.expectEqualStrings("text2text-generation", results[1].pipeline_tag);
    try std.testing.expectEqual(@as(i64, 4022468096), results[1].params_total);
}

test "parseSearchResultsRich: defaults for missing fields" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "org/minimal"}
        \\]
    ;

    const results = try parseSearchResultsRich(allocator, json);
    defer {
        for (results) |*r| r.deinit(allocator);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 1), results.len);
    try std.testing.expectEqualStrings("org/minimal", results[0].model_id);
    try std.testing.expectEqual(@as(i64, 0), results[0].downloads);
    try std.testing.expectEqual(@as(i64, 0), results[0].likes);
    try std.testing.expectEqual(@as(usize, 0), results[0].last_modified.len);
    try std.testing.expectEqual(@as(usize, 0), results[0].pipeline_tag.len);
    try std.testing.expectEqual(@as(i64, 0), results[0].params_total);
}

test "parseSearchResultsRich: returns error for non-array JSON" {
    const allocator = std.testing.allocator;
    const result = parseSearchResultsRich(allocator, "{\"error\": \"bad\"}");
    try std.testing.expectError(DownloadError.ApiResponseParseError, result);
}

test "parseSearchResultsRich: skips non-object entries" {
    const allocator = std.testing.allocator;

    const json =
        \\[
        \\  {"modelId": "org/model1", "downloads": 10, "likes": 5},
        \\  "string-entry",
        \\  123,
        \\  null,
        \\  {"modelId": "org/model2", "downloads": 20, "likes": 15}
        \\]
    ;

    const results = try parseSearchResultsRich(allocator, json);
    defer {
        for (results) |*r| r.deinit(allocator);
        allocator.free(results);
    }

    try std.testing.expectEqual(@as(usize, 2), results.len);
    try std.testing.expectEqualStrings("org/model1", results[0].model_id);
    try std.testing.expectEqual(@as(i64, 10), results[0].downloads);
    try std.testing.expectEqualStrings("org/model2", results[1].model_id);
    try std.testing.expectEqual(@as(i64, 20), results[1].downloads);
}

test "parseSearchResultsRich: returns empty for empty array" {
    const allocator = std.testing.allocator;
    const results = try parseSearchResultsRich(allocator, "[]");
    defer allocator.free(results);
    try std.testing.expectEqual(@as(usize, 0), results.len);
}

test "searchModelsRich: signature verification" {
    const F = @TypeOf(searchModelsRich);
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(@as(usize, 3), info.params.len);
}

test "SearchSort.toParam: maps correctly" {
    try std.testing.expectEqual(@as(?[]const u8, null), SearchSort.trending.toParam());
    try std.testing.expectEqualStrings("downloads", SearchSort.downloads.toParam().?);
    try std.testing.expectEqualStrings("likes", SearchSort.likes.toParam().?);
    try std.testing.expectEqualStrings("lastModified", SearchSort.last_modified.toParam().?);
}

test "SearchDirection.toParam: maps correctly" {
    try std.testing.expectEqualStrings("-1", SearchDirection.descending.toParam());
    try std.testing.expectEqualStrings("1", SearchDirection.ascending.toParam());
}

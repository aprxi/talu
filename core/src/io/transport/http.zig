//! HTTP Client
//!
//! Pure HTTP client using libcurl. No HF-specific logic.

const std = @import("std");
const c = @cImport({
    @cInclude("curl/curl.h");
});

const log = @import("../../log.zig");

/// Mozilla CA certificates bundle - embedded at compile time for portability
/// Downloaded from https://curl.se/ca/cacert.pem during build, provided via build.zig
const ca_bundle = @import("cacert").data;

pub const HttpError = error{
    CurlInitFailed,
    CurlSetOptFailed,
    CurlPerformFailed,
    HttpError,
    StreamWriteFailed,
    ResponseTooLarge,
    NotFound,
    Unauthorized,
    RateLimited,
    OutOfMemory,
    Cancelled,
};

/// Progress callback for download progress reporting
pub const ProgressCallback = *const fn (downloaded: u64, total: u64, user_data: ?*anyopaque) void;

/// File start callback - called when starting to download a new file
pub const FileStartCallback = *const fn (filename: []const u8, user_data: ?*anyopaque) void;

/// HTTP client configuration
pub const HttpConfig = struct {
    /// Bearer token for Authorization header (optional)
    token: ?[]const u8 = null,
    /// Progress callback (optional)
    progress_callback: ?ProgressCallback = null,
    progress_data: ?*anyopaque = null,
    /// User agent string
    user_agent: []const u8 = "talu/1.0",
    /// Maximum response body size (optional)
    max_response_bytes: ?usize = null,
    /// Resume offset in bytes for ranged downloads (0 = start from beginning)
    resume_from: u64 = 0,
    /// Cancel flag — set to true from another thread to abort the download
    cancel_flag: ?*const bool = null,
};

/// Writer context for curl write callback (to file)
const CurlFileWriteContext = struct {
    file: std.fs.File,
    bytes_written: u64,
};

/// Writer context for curl write callback (to memory)
const CurlMemoryWriteContext = struct {
    allocator: std.mem.Allocator,
    data: std.ArrayListUnmanaged(u8),
    max_bytes: ?usize = null,
    hit_limit: bool = false,
};

/// Progress context for curl progress callback
const CurlProgressContext = struct {
    callback: ?ProgressCallback,
    user_data: ?*anyopaque,
    resume_from: u64,
    cancel_flag: ?*const bool = null,
};

/// libcurl write callback - writes data to file
fn curlFileWriteCallback(data: [*c]u8, size: usize, nmemb: usize, user_data: *anyopaque) callconv(.c) usize {
    const file_ctx: *CurlFileWriteContext = @ptrCast(@alignCast(user_data));
    const total_size = size * nmemb;
    const bytes = data[0..total_size];

    file_ctx.file.writeAll(bytes) catch {
        return 0; // Signal error to curl
    };
    file_ctx.bytes_written += total_size;
    return total_size;
}

/// libcurl write callback - writes data to memory buffer
fn curlMemoryWriteCallback(data: [*c]u8, size: usize, nmemb: usize, user_data: *anyopaque) callconv(.c) usize {
    const memory_ctx: *CurlMemoryWriteContext = @ptrCast(@alignCast(user_data));
    const total_size = size * nmemb;
    const bytes = data[0..total_size];

    if (memory_ctx.max_bytes) |max_bytes| {
        if (memory_ctx.data.items.len + total_size > max_bytes) {
            memory_ctx.hit_limit = true;
            return 0; // Signal error to curl (size limit exceeded)
        }
    }

    memory_ctx.data.appendSlice(memory_ctx.allocator, bytes) catch {
        return 0; // Signal error to curl
    };
    return total_size;
}

/// libcurl progress callback
fn curlProgressCallback(
    user_data: *anyopaque,
    dltotal: c.curl_off_t,
    dlnow: c.curl_off_t,
    _: c.curl_off_t,
    _: c.curl_off_t,
) callconv(.c) c_int {
    const progress_ctx: *CurlProgressContext = @ptrCast(@alignCast(user_data));

    // Check cancel flag — abort if set.
    if (progress_ctx.cancel_flag) |flag| {
        if (@atomicLoad(bool, flag, .monotonic)) return 1;
    }

    if (progress_ctx.callback) |cb| {
        const downloaded_now: u64 = if (dlnow > 0) @intCast(dlnow) else 0;
        const total_now: u64 = if (dltotal > 0) @intCast(dltotal) else 0;
        const downloaded = downloaded_now + progress_ctx.resume_from;
        const total = if (total_now > 0)
            total_now + progress_ctx.resume_from
        else
            0;
        cb(
            downloaded,
            total,
            progress_ctx.user_data,
        );
    }
    return 0;
}

/// Configure SSL/TLS for curl using embedded Mozilla CA bundle
/// Works on all platforms with mbedTLS - no external certificate files needed
fn configureSsl(curl_handle: *c.CURL) HttpError!void {
    // Use CURLOPT_CAINFO_BLOB to load CA certs from embedded memory (curl 7.77.0+)
    var blob = c.curl_blob{
        .data = @ptrCast(@constCast(ca_bundle.ptr)),
        .len = ca_bundle.len,
        .flags = c.CURL_BLOB_NOCOPY,
    };
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_CAINFO_BLOB, &blob) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;
}

/// Map HTTP status code to error
fn httpError(code: c_long) HttpError {
    return switch (code) {
        404 => HttpError.NotFound,
        401 => HttpError.Unauthorized,
        429 => HttpError.RateLimited,
        else => HttpError.HttpError,
    };
}

/// Set common curl options (user agent, follow redirects, auth)
fn setCommonOptions(allocator: std.mem.Allocator, curl_handle: *c.CURL, config: HttpConfig) !?*c.struct_curl_slist {
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_FOLLOWLOCATION, @as(c_long, 1)) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;

    const user_agent_z = try allocator.dupeZ(u8, config.user_agent);
    defer allocator.free(user_agent_z);
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_USERAGENT, user_agent_z.ptr) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;

    if (config.token) |token| {
        const auth_header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{token});
        defer allocator.free(auth_header);
        const auth_header_z = try allocator.dupeZ(u8, auth_header);
        defer allocator.free(auth_header_z);
        const header_list = c.curl_slist_append(null, auth_header_z.ptr);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_HTTPHEADER, header_list) != c.CURLE_OK) {
            c.curl_slist_free_all(header_list);
            return HttpError.CurlSetOptFailed;
        }
        return header_list;
    }
    return null;
}

/// Fetch URL content into memory
pub fn fetch(allocator: std.mem.Allocator, url: []const u8, config: HttpConfig) ![]u8 {
    const curl_handle = c.curl_easy_init() orelse return HttpError.CurlInitFailed;
    defer c.curl_easy_cleanup(curl_handle);

    try configureSsl(curl_handle);

    const url_z = try allocator.dupeZ(u8, url);
    defer allocator.free(url_z);

    var response_buffer = CurlMemoryWriteContext{
        .allocator = allocator,
        .data = .{},
        .max_bytes = config.max_response_bytes,
    };
    errdefer response_buffer.data.deinit(allocator);

    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_URL, url_z.ptr) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEFUNCTION, @as(*const anyopaque, @ptrCast(&curlMemoryWriteCallback))) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEDATA, @as(*anyopaque, @ptrCast(&response_buffer))) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;

    if (config.max_response_bytes) |max_bytes| {
        const max_size: c.curl_off_t = @intCast(max_bytes);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_MAXFILESIZE_LARGE, max_size) != c.CURLE_OK)
            return HttpError.CurlSetOptFailed;
    }

    const header_list = try setCommonOptions(allocator, curl_handle, config);
    defer if (header_list) |h| c.curl_slist_free_all(h);

    const result = c.curl_easy_perform(curl_handle);
    if (result != c.CURLE_OK) {
        if (result == c.CURLE_FILESIZE_EXCEEDED or response_buffer.hit_limit) {
            return HttpError.ResponseTooLarge;
        }
        if (result == c.CURLE_WRITE_ERROR) {
            return HttpError.StreamWriteFailed;
        }
        log.err("http", "curl_easy_perform failed", .{ .code = result, .err_msg = c.curl_easy_strerror(result), .url = url }, @src());
        return HttpError.CurlPerformFailed;
    }

    var status_code: c_long = 0;
    _ = c.curl_easy_getinfo(curl_handle, c.CURLINFO_RESPONSE_CODE, &status_code);
    if (status_code >= 400) return httpError(status_code);

    return response_buffer.data.toOwnedSlice(allocator);
}

/// Stream URL content to an open file handle.
/// Caller is responsible for file creation, temp file management, and cleanup.
/// Returns StreamWriteFailed if writing to the file fails during transfer.
pub fn downloadToFile(
    allocator: std.mem.Allocator,
    url: []const u8,
    file: std.fs.File,
    config: HttpConfig,
) HttpError!void {
    const curl_handle = c.curl_easy_init() orelse return HttpError.CurlInitFailed;
    defer c.curl_easy_cleanup(curl_handle);

    var file_writer = CurlFileWriteContext{ .file = file, .bytes_written = 0 };
    var progress_ctx = CurlProgressContext{
        .callback = config.progress_callback,
        .user_data = config.progress_data,
        .resume_from = config.resume_from,
        .cancel_flag = config.cancel_flag,
    };

    try configureSsl(curl_handle);

    const url_z = try allocator.dupeZ(u8, url);
    defer allocator.free(url_z);

    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_URL, url_z.ptr) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEFUNCTION, @as(*const anyopaque, @ptrCast(&curlFileWriteCallback))) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;
    if (c.curl_easy_setopt(curl_handle, c.CURLOPT_WRITEDATA, @as(*anyopaque, @ptrCast(&file_writer))) != c.CURLE_OK)
        return HttpError.CurlSetOptFailed;

    if (config.resume_from > 0) {
        const resume_from: c.curl_off_t = @intCast(config.resume_from);
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_RESUME_FROM_LARGE, resume_from) != c.CURLE_OK)
            return HttpError.CurlSetOptFailed;
    }

    if (config.progress_callback != null or config.cancel_flag != null) {
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_XFERINFOFUNCTION, @as(*const anyopaque, @ptrCast(&curlProgressCallback))) != c.CURLE_OK)
            return HttpError.CurlSetOptFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_XFERINFODATA, @as(*anyopaque, @ptrCast(&progress_ctx))) != c.CURLE_OK)
            return HttpError.CurlSetOptFailed;
        if (c.curl_easy_setopt(curl_handle, c.CURLOPT_NOPROGRESS, @as(c_long, 0)) != c.CURLE_OK)
            return HttpError.CurlSetOptFailed;
    }

    const header_list = try setCommonOptions(allocator, curl_handle, config);
    defer if (header_list) |h| c.curl_slist_free_all(h);

    const result = c.curl_easy_perform(curl_handle);
    if (result != c.CURLE_OK) {
        // CURLE_ABORTED_BY_CALLBACK (42) — progress callback returned non-zero (cancel flag).
        if (result == c.CURLE_ABORTED_BY_CALLBACK) {
            return HttpError.Cancelled;
        }
        // CURLE_WRITE_ERROR (23) indicates the write callback returned an error
        if (result == c.CURLE_WRITE_ERROR) {
            return HttpError.StreamWriteFailed;
        }
        log.err("http", "curl download failed", .{ .code = result, .err_msg = c.curl_easy_strerror(result), .url = url }, @src());
        return HttpError.CurlPerformFailed;
    }

    var status_code: c_long = 0;
    _ = c.curl_easy_getinfo(curl_handle, c.CURLINFO_RESPONSE_CODE, &status_code);
    if (status_code >= 400) return httpError(status_code);
}

/// Initialize curl globally (call once at program start)
pub fn globalInit() void {
    _ = c.curl_global_init(c.CURL_GLOBAL_DEFAULT);
}

/// Clean up curl globally (call once at program end)
pub fn globalCleanup() void {
    c.curl_global_cleanup();
}

// =============================================================================
// Unit Tests
// =============================================================================

test "globalInit and globalCleanup are callable" {
    // These are thin wrappers around libcurl. Test that they don't crash.
    // Note: curl_global_init is reference-counted and safe to call multiple times.
    globalInit();
    globalCleanup();
}

test "globalInit can be called multiple times" {
    // curl_global_init uses reference counting, so multiple calls are safe
    globalInit();
    globalInit();
    globalCleanup();
    globalCleanup();
}

test "HttpError maps HTTP status codes correctly" {
    // Test the httpError helper function
    try std.testing.expectEqual(HttpError.NotFound, httpError(404));
    try std.testing.expectEqual(HttpError.Unauthorized, httpError(401));
    try std.testing.expectEqual(HttpError.RateLimited, httpError(429));
    try std.testing.expectEqual(HttpError.HttpError, httpError(500));
    try std.testing.expectEqual(HttpError.HttpError, httpError(503));
    try std.testing.expectEqual(HttpError.HttpError, httpError(400));
}

test "HttpConfig has sensible defaults" {
    const config = HttpConfig{};
    try std.testing.expect(config.token == null);
    try std.testing.expect(config.progress_callback == null);
    try std.testing.expect(config.progress_data == null);
    try std.testing.expectEqualStrings("talu/1.0", config.user_agent);
}

test "HttpConfig can be customized" {
    var user_data: u32 = 42;
    const config = HttpConfig{
        .token = "test_token",
        .progress_data = @ptrCast(&user_data),
        .user_agent = "custom/2.0",
    };
    try std.testing.expectEqualStrings("test_token", config.token.?);
    try std.testing.expectEqualStrings("custom/2.0", config.user_agent);
    try std.testing.expect(config.progress_data != null);
}

test "fetch returns error for invalid URL" {
    const allocator = std.testing.allocator;
    globalInit();
    defer globalCleanup();

    // Test with a URL that will fail DNS resolution
    const result = fetch(allocator, "http://invalid.invalid.invalid/", .{});
    try std.testing.expectError(HttpError.CurlPerformFailed, result);
}

test "downloadToFile returns error for invalid URL" {
    const allocator = std.testing.allocator;
    globalInit();
    defer globalCleanup();

    const test_path = "/tmp/talu_http_test_download_invalid.bin";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const file = try std.fs.cwd().createFile(test_path, .{});
    defer file.close();

    const result = downloadToFile(allocator, "http://invalid.invalid.invalid/file.bin", file, .{});
    try std.testing.expectError(HttpError.CurlPerformFailed, result);
}

test "CurlMemoryWriteContext accumulates data" {
    const allocator = std.testing.allocator;
    var ctx = CurlMemoryWriteContext{ .allocator = allocator, .data = .{} };
    defer ctx.data.deinit(allocator);

    // Simulate curl calling the write callback multiple times
    const chunk1 = "Hello, ";
    const chunk2 = "World!";

    const ret1 = curlMemoryWriteCallback(@constCast(chunk1.ptr), 1, chunk1.len, @ptrCast(&ctx));
    try std.testing.expectEqual(chunk1.len, ret1);

    const ret2 = curlMemoryWriteCallback(@constCast(chunk2.ptr), 1, chunk2.len, @ptrCast(&ctx));
    try std.testing.expectEqual(chunk2.len, ret2);

    try std.testing.expectEqualStrings("Hello, World!", ctx.data.items);
}

test "CurlFileWriteContext writes to file" {
    const test_path = "/tmp/talu_http_test_file_write.bin";
    defer std.fs.cwd().deleteFile(test_path) catch {};

    const file = try std.fs.cwd().createFile(test_path, .{});
    var ctx = CurlFileWriteContext{ .file = file, .bytes_written = 0 };

    const data = "Test file content";
    const ret = curlFileWriteCallback(@constCast(data.ptr), 1, data.len, @ptrCast(&ctx));

    try std.testing.expectEqual(data.len, ret);
    try std.testing.expectEqual(data.len, ctx.bytes_written);

    file.close();

    // Verify file content
    const read_file = try std.fs.cwd().openFile(test_path, .{});
    defer read_file.close();
    var buf: [100]u8 = undefined;
    const len = try read_file.readAll(&buf);
    try std.testing.expectEqualStrings(data, buf[0..len]);
}

test "curlProgressCallback invokes user callback" {
    const TestData = struct {
        var last_downloaded: u64 = 0;
        var last_total: u64 = 0;
        var call_count: u32 = 0;

        fn callback(downloaded: u64, total: u64, _: ?*anyopaque) void {
            last_downloaded = downloaded;
            last_total = total;
            call_count += 1;
        }
    };
    TestData.call_count = 0;

    var ctx = CurlProgressContext{
        .callback = TestData.callback,
        .user_data = null,
        .resume_from = 0,
    };

    const result = curlProgressCallback(@ptrCast(&ctx), 1000, 500, 0, 0);

    try std.testing.expectEqual(@as(c_int, 0), result);
    try std.testing.expectEqual(@as(u32, 1), TestData.call_count);
    try std.testing.expectEqual(@as(u64, 500), TestData.last_downloaded);
    try std.testing.expectEqual(@as(u64, 1000), TestData.last_total);
}

test "curlProgressCallback with null callback is no-op" {
    var ctx = CurlProgressContext{
        .callback = null,
        .user_data = null,
        .resume_from = 0,
    };

    // Should not crash when callback is null
    const result = curlProgressCallback(@ptrCast(&ctx), 1000, 500, 0, 0);
    try std.testing.expectEqual(@as(c_int, 0), result);
}

test "curlProgressCallback applies resume offset" {
    const TestData = struct {
        var last_downloaded: u64 = 0;
        var last_total: u64 = 0;
        var call_count: u32 = 0;

        fn callback(downloaded: u64, total: u64, _: ?*anyopaque) void {
            last_downloaded = downloaded;
            last_total = total;
            call_count += 1;
        }
    };
    TestData.call_count = 0;

    var ctx = CurlProgressContext{
        .callback = TestData.callback,
        .user_data = null,
        .resume_from = 128,
    };

    const result = curlProgressCallback(@ptrCast(&ctx), 1024, 256, 0, 0);

    try std.testing.expectEqual(@as(c_int, 0), result);
    try std.testing.expectEqual(@as(u32, 1), TestData.call_count);
    try std.testing.expectEqual(@as(u64, 384), TestData.last_downloaded);
    try std.testing.expectEqual(@as(u64, 1152), TestData.last_total);
}

test "curlProgressCallback aborts when cancel flag is set" {
    var cancelled: bool = true;

    var ctx = CurlProgressContext{
        .callback = null,
        .user_data = null,
        .resume_from = 0,
        .cancel_flag = &cancelled,
    };

    const result = curlProgressCallback(@ptrCast(&ctx), 1000, 500, 0, 0);
    try std.testing.expectEqual(@as(c_int, 1), result);
}

test "curlProgressCallback continues when cancel flag is false" {
    var cancelled: bool = false;

    var ctx = CurlProgressContext{
        .callback = null,
        .user_data = null,
        .resume_from = 0,
        .cancel_flag = &cancelled,
    };

    const result = curlProgressCallback(@ptrCast(&ctx), 1000, 500, 0, 0);
    try std.testing.expectEqual(@as(c_int, 0), result);
}

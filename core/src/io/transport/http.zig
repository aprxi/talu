//! HTTP Client
//!
//! Pure HTTP client using Zig stdlib HTTP. No HF-specific logic.

const std = @import("std");

/// Mozilla CA certificates bundle - embedded at compile time for portability.
/// Downloaded from https://curl.se/ca/cacert.pem during `make deps`, provided via build.zig.
const ca_bundle_pem = @import("cacert").data;

const Allocator = std.mem.Allocator;
const Bundle = std.crypto.Certificate.Bundle;
const HttpClient = std.http.Client;

pub const HttpError = error{
    NetworkFailed,
    TlsFailed,
    RequestFailed,
    HttpError,
    StreamWriteFailed,
    ResponseTooLarge,
    NotFound,
    Unauthorized,
    RateLimited,
    OutOfMemory,
    Cancelled,
};

/// Progress callback for download progress reporting.
pub const ProgressCallback = *const fn (downloaded: u64, total: u64, user_data: ?*anyopaque) void;

/// File start callback - called when starting to download a new file.
pub const FileStartCallback = *const fn (filename: []const u8, user_data: ?*anyopaque) void;

/// HTTP client configuration.
pub const HttpConfig = struct {
    /// Bearer token for Authorization header (optional).
    token: ?[]const u8 = null,
    /// Progress callback (optional).
    progress_callback: ?ProgressCallback = null,
    progress_data: ?*anyopaque = null,
    /// User agent string.
    user_agent: []const u8 = "talu/1.0",
    /// Maximum response body size (optional).
    max_response_bytes: ?usize = null,
    /// Resume offset in bytes for ranged downloads (0 = start from beginning).
    resume_from: u64 = 0,
    /// Cancel flag - set to true from another thread to abort the download.
    cancel_flag: ?*const bool = null,
    /// Maximum redirects followed by the underlying HTTP client.
    redirect_limit: u16 = 5,
};

const pem_decoder = std.base64.standard.decoderWithIgnore(" \t\r\n");

const PemLoadError = Allocator.Error ||
    std.base64.Error ||
    Bundle.ParseCertError ||
    error{ CertificateAuthorityBundleTooBig, MissingEndCertificateMarker };

fn addCertsFromPemBytes(bundle: *Bundle, allocator: Allocator, pem_bytes: []const u8) PemLoadError!void {
    const decoded_size_upper_bound = pem_bytes.len / 4 * 3;
    const needed_capacity = std.math.cast(u32, decoded_size_upper_bound + pem_bytes.len) orelse
        return error.CertificateAuthorityBundleTooBig;
    try bundle.bytes.ensureUnusedCapacity(allocator, needed_capacity);

    const end_reserved: u32 = @intCast(bundle.bytes.items.len + decoded_size_upper_bound);
    const encoded_bytes = bundle.bytes.allocatedSlice()[end_reserved..][0..pem_bytes.len];
    @memcpy(encoded_bytes, pem_bytes);

    const begin_marker = "-----BEGIN CERTIFICATE-----";
    const end_marker = "-----END CERTIFICATE-----";
    const now_sec = std.time.timestamp();

    var start_index: usize = 0;
    while (std.mem.indexOfPos(u8, encoded_bytes, start_index, begin_marker)) |begin_marker_start| {
        const cert_start = begin_marker_start + begin_marker.len;
        const cert_end = std.mem.indexOfPos(u8, encoded_bytes, cert_start, end_marker) orelse
            return error.MissingEndCertificateMarker;
        start_index = cert_end + end_marker.len;

        const encoded_cert = std.mem.trim(u8, encoded_bytes[cert_start..cert_end], " \t\r\n");
        const decoded_start: u32 = @intCast(bundle.bytes.items.len);
        const dest_buf = bundle.bytes.allocatedSlice()[decoded_start..];
        bundle.bytes.items.len += try pem_decoder.decode(dest_buf, encoded_cert);
        try bundle.parseCert(allocator, decoded_start, now_sec);
    }

    bundle.bytes.shrinkAndFree(allocator, bundle.bytes.items.len);
}

fn loadEmbeddedCertificateBundle(allocator: Allocator, bundle: *Bundle) HttpError!void {
    addCertsFromPemBytes(bundle, allocator, ca_bundle_pem) catch |err| {
        return switch (err) {
            error.OutOfMemory => HttpError.OutOfMemory,
            else => HttpError.TlsFailed,
        };
    };
    if (bundle.map.count() == 0) return HttpError.TlsFailed;
}

fn initClient(allocator: Allocator) HttpError!HttpClient {
    var client = HttpClient{ .allocator = allocator };
    errdefer client.deinit();

    if (!HttpClient.disable_tls) {
        try loadEmbeddedCertificateBundle(allocator, &client.ca_bundle);
        client.next_https_rescan_certs = false;
    }

    return client;
}

fn httpError(code: u16) HttpError {
    return switch (code) {
        404 => HttpError.NotFound,
        401 => HttpError.Unauthorized,
        429 => HttpError.RateLimited,
        else => HttpError.HttpError,
    };
}

fn redirectBehavior(limit: u16) HttpClient.Request.RedirectBehavior {
    if (limit == 0) return .not_allowed;
    return HttpClient.Request.RedirectBehavior.init(limit);
}

fn authorizationHeader(allocator: Allocator, token: ?[]const u8) HttpError!?[]u8 {
    const bearer = token orelse return null;
    return std.fmt.allocPrint(allocator, "Bearer {s}", .{bearer}) catch |err| {
        return switch (err) {
            error.OutOfMemory => HttpError.OutOfMemory,
        };
    };
}

fn requestHeaders(config: HttpConfig, authorization: ?[]const u8) HttpClient.Request.Headers {
    var headers = HttpClient.Request.Headers{
        .user_agent = .{ .override = config.user_agent },
        .accept_encoding = .{ .override = "identity" },
    };
    if (authorization) |value| {
        headers.authorization = .{ .override = value };
    }
    return headers;
}

fn resumeHeaders(
    config: HttpConfig,
    range_value: *[64]u8,
    extra_headers: *[1]std.http.Header,
) []const std.http.Header {
    if (config.resume_from == 0) return &.{};
    const value = std.fmt.bufPrint(range_value, "bytes={d}-", .{config.resume_from}) catch unreachable;
    extra_headers[0] = .{ .name = "range", .value = value };
    return extra_headers[0..1];
}

fn mapRequestError(err: anyerror) HttpError {
    return switch (err) {
        error.OutOfMemory => HttpError.OutOfMemory,
        error.CertificateBundleLoadFailure => HttpError.TlsFailed,
        error.UnsupportedUriScheme,
        error.UriMissingHost,
        error.UriHostTooLong,
        => HttpError.RequestFailed,
        else => HttpError.NetworkFailed,
    };
}

fn mapReceiveError(err: anyerror) HttpError {
    return switch (err) {
        error.OutOfMemory => HttpError.OutOfMemory,
        error.CertificateBundleLoadFailure => HttpError.TlsFailed,
        error.UnsupportedUriScheme,
        error.HttpHeadersInvalid,
        error.TooManyHttpRedirects,
        error.RedirectRequiresResend,
        error.HttpRedirectLocationMissing,
        error.HttpRedirectLocationOversize,
        error.HttpRedirectLocationInvalid,
        error.HttpContentEncodingUnsupported,
        error.HttpChunkInvalid,
        error.HttpChunkTruncated,
        error.HttpHeadersOversize,
        => HttpError.RequestFailed,
        else => HttpError.NetworkFailed,
    };
}

fn checkCancelled(config: HttpConfig) HttpError!void {
    if (config.cancel_flag) |flag| {
        if (@atomicLoad(bool, flag, .monotonic)) return HttpError.Cancelled;
    }
}

fn emitProgress(config: HttpConfig, downloaded: u64, total: u64) void {
    if (config.progress_callback) |callback| {
        callback(downloaded, total, config.progress_data);
    }
}

const MemorySink = struct {
    allocator: Allocator,
    data: *std.ArrayListUnmanaged(u8),

    fn writeAll(self: *MemorySink, bytes: []const u8) HttpError!void {
        self.data.appendSlice(self.allocator, bytes) catch |err| {
            return switch (err) {
                error.OutOfMemory => HttpError.OutOfMemory,
            };
        };
    }
};

const FileSink = struct {
    file: std.fs.File,

    fn writeAll(self: *FileSink, bytes: []const u8) HttpError!void {
        self.file.writeAll(bytes) catch return HttpError.StreamWriteFailed;
    }
};

fn streamResponseBody(
    response: *HttpClient.Response,
    sink: anytype,
    config: HttpConfig,
    progress_base: u64,
    progress_total: u64,
) HttpError!u64 {
    var transfer_buffer: [8192]u8 = undefined;
    var read_buffer: [8192]u8 = undefined;
    const expected_body_len = response.head.content_length;
    const reader = response.reader(&transfer_buffer);
    var transferred: u64 = 0;

    while (true) {
        try checkCancelled(config);

        var read_slice: []u8 = read_buffer[0..];
        if (expected_body_len) |expected_len| {
            if (transferred >= expected_len) break;
            const remaining = expected_len - transferred;
            const read_len = if (remaining > @as(u64, @intCast(read_buffer.len)))
                read_buffer.len
            else
                @as(usize, @intCast(remaining));
            read_slice = read_buffer[0..read_len];
        }

        const n = reader.readSliceShort(read_slice) catch |err| switch (err) {
            error.ReadFailed => return HttpError.NetworkFailed,
        };
        if (n == 0) break;

        const next_transferred = transferred + @as(u64, @intCast(n));
        if (config.max_response_bytes) |max_bytes| {
            if (next_transferred > @as(u64, @intCast(max_bytes))) {
                return HttpError.ResponseTooLarge;
            }
        }

        try sink.writeAll(read_buffer[0..n]);
        transferred = next_transferred;
        emitProgress(config, progress_base + transferred, progress_total);
    }

    return transferred;
}

fn progressTotal(content_length: ?u64, progress_base: u64) u64 {
    const remaining = content_length orelse return 0;
    return progress_base + remaining;
}

/// Fetch URL content into memory.
pub fn fetch(allocator: Allocator, url: []const u8, config: HttpConfig) ![]u8 {
    var client = try initClient(allocator);
    defer client.deinit();

    const uri = std.Uri.parse(url) catch return HttpError.RequestFailed;

    const authorization = try authorizationHeader(allocator, config.token);
    defer if (authorization) |value| allocator.free(value);

    var range_value: [64]u8 = undefined;
    var extra_headers_buf: [1]std.http.Header = undefined;
    const extra_headers = resumeHeaders(config, &range_value, &extra_headers_buf);

    var req = client.request(.GET, uri, .{
        .headers = requestHeaders(config, authorization),
        .extra_headers = extra_headers,
        .redirect_behavior = redirectBehavior(config.redirect_limit),
        .keep_alive = false,
    }) catch |err| return mapRequestError(err);
    defer req.deinit();

    req.sendBodiless() catch |err| return mapReceiveError(err);

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(&redirect_buffer) catch |err| return mapReceiveError(err);

    const status_code: u16 = @intFromEnum(response.head.status);
    if (status_code >= 400) return httpError(status_code);

    const progress_base = if (config.resume_from > 0 and status_code == 206) config.resume_from else 0;
    const total = progressTotal(response.head.content_length, progress_base);

    var response_buffer = std.ArrayListUnmanaged(u8){};
    errdefer response_buffer.deinit(allocator);

    var sink = MemorySink{ .allocator = allocator, .data = &response_buffer };
    _ = try streamResponseBody(&response, &sink, config, progress_base, total);

    return response_buffer.toOwnedSlice(allocator);
}

/// Stream URL content to an open file handle.
/// Caller is responsible for file creation, temp file management, and cleanup.
/// Returns StreamWriteFailed if writing to the file fails during transfer.
pub fn downloadToFile(
    allocator: Allocator,
    url: []const u8,
    file: std.fs.File,
    config: HttpConfig,
) HttpError!void {
    var client = try initClient(allocator);
    defer client.deinit();

    const uri = std.Uri.parse(url) catch return HttpError.RequestFailed;

    const authorization = try authorizationHeader(allocator, config.token);
    defer if (authorization) |value| allocator.free(value);

    var range_value: [64]u8 = undefined;
    var extra_headers_buf: [1]std.http.Header = undefined;
    const extra_headers = resumeHeaders(config, &range_value, &extra_headers_buf);

    var req = client.request(.GET, uri, .{
        .headers = requestHeaders(config, authorization),
        .extra_headers = extra_headers,
        .redirect_behavior = redirectBehavior(config.redirect_limit),
        .keep_alive = false,
    }) catch |err| return mapRequestError(err);
    defer req.deinit();

    req.sendBodiless() catch |err| return mapReceiveError(err);

    var redirect_buffer: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(&redirect_buffer) catch |err| return mapReceiveError(err);

    const status_code: u16 = @intFromEnum(response.head.status);
    if (status_code >= 400) return httpError(status_code);

    const progress_base: u64 = if (config.resume_from > 0 and status_code == 206) config.resume_from else 0;
    if (config.resume_from > 0 and status_code == 200) {
        file.seekTo(0) catch return HttpError.StreamWriteFailed;
        file.setEndPos(0) catch return HttpError.StreamWriteFailed;
    }
    const total = progressTotal(response.head.content_length, progress_base);

    var sink = FileSink{ .file = file };
    _ = try streamResponseBody(&response, &sink, config, progress_base, total);
}

// =============================================================================
// Unit Tests
// =============================================================================

const TestHandler = *const fn (stream: std.net.Stream, request_head: []const u8, user_data: ?*anyopaque) void;

const TestServer = struct {
    allocator: Allocator,
    server: std.net.Server,
    thread: std.Thread,
    stopping: std.atomic.Value(bool),
    request_limit: usize,
    handler: TestHandler,
    user_data: ?*anyopaque,
    url: []const u8,
    err: ?anyerror = null,

    fn start(
        allocator: Allocator,
        request_limit: usize,
        handler: TestHandler,
        user_data: ?*anyopaque,
    ) !*TestServer {
        var address = try std.net.Address.parseIp("127.0.0.1", 0);
        var listener = try address.listen(.{ .reuse_address = true });
        errdefer listener.deinit();

        const url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}", .{listener.listen_address.getPort()});
        errdefer allocator.free(url);

        const self = try allocator.create(TestServer);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .server = listener,
            .thread = undefined,
            .stopping = std.atomic.Value(bool).init(false),
            .request_limit = request_limit,
            .handler = handler,
            .user_data = user_data,
            .url = url,
        };
        self.thread = try std.Thread.spawn(.{}, run, .{self});
        return self;
    }

    fn deinit(self: *TestServer) void {
        self.stopping.store(true, .release);
        self.server.deinit();
        self.thread.join();
        self.allocator.free(self.url);
        self.allocator.destroy(self);
    }

    fn run(self: *TestServer) void {
        var handled: usize = 0;
        while (handled < self.request_limit) : (handled += 1) {
            const conn = self.server.accept() catch |err| {
                if (!self.stopping.load(.acquire)) self.err = err;
                return;
            };
            defer conn.stream.close();

            var head_buffer: [8192]u8 = undefined;
            const request_head = readRequestHead(conn.stream, &head_buffer) catch |err| {
                self.err = err;
                return;
            };
            self.handler(conn.stream, request_head, self.user_data);
        }
    }
};

fn readRequestHead(stream: std.net.Stream, buffer: []u8) ![]const u8 {
    var used: usize = 0;
    while (used < buffer.len) {
        const n = try stream.read(buffer[used..]);
        if (n == 0) return error.EndOfStream;
        used += n;
        if (std.mem.indexOf(u8, buffer[0..used], "\r\n\r\n") != null) {
            return buffer[0..used];
        }
    }
    return error.HttpHeadersOversize;
}

fn requestTarget(request_head: []const u8) []const u8 {
    const first_line_end = std.mem.indexOf(u8, request_head, "\r\n") orelse request_head.len;
    const first_line = request_head[0..first_line_end];
    const method_end = std.mem.indexOfScalar(u8, first_line, ' ') orelse return "";
    const target_end = std.mem.indexOfScalarPos(u8, first_line, method_end + 1, ' ') orelse return "";
    return first_line[method_end + 1 .. target_end];
}

fn requestHeader(request_head: []const u8, name: []const u8) ?[]const u8 {
    var lines = std.mem.splitSequence(u8, request_head, "\r\n");
    _ = lines.next();
    while (lines.next()) |line| {
        if (line.len == 0) return null;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const header_name = line[0..colon];
        const header_value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(header_name, name)) return header_value;
    }
    return null;
}

fn writeResponse(stream: std.net.Stream, status: u16, reason: []const u8, extra_headers: []const u8, body: []const u8) void {
    var header_buf: [256]u8 = undefined;
    const header = std.fmt.bufPrint(
        &header_buf,
        "HTTP/1.1 {d} {s}\r\ncontent-length: {d}\r\nconnection: close\r\n",
        .{ status, reason, body.len },
    ) catch return;
    stream.writeAll(header) catch return;
    stream.writeAll(extra_headers) catch return;
    stream.writeAll("\r\n") catch return;
    stream.writeAll(body) catch return;
}

fn successHandler(stream: std.net.Stream, _: []const u8, _: ?*anyopaque) void {
    writeResponse(stream, 200, "OK", "", "hello");
}

test "HttpError maps HTTP status codes correctly" {
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
    try std.testing.expectEqual(@as(u16, 5), config.redirect_limit);
}

test "HttpConfig can be customized" {
    var user_data: u32 = 42;
    const config = HttpConfig{
        .token = "test_token",
        .progress_data = @ptrCast(&user_data),
        .user_agent = "custom/2.0",
        .redirect_limit = 1,
    };
    try std.testing.expectEqualStrings("test_token", config.token.?);
    try std.testing.expectEqualStrings("custom/2.0", config.user_agent);
    try std.testing.expect(config.progress_data != null);
    try std.testing.expectEqual(@as(u16, 1), config.redirect_limit);
}

test "loadEmbeddedCertificateBundle parses embedded CA bundle" {
    const allocator = std.testing.allocator;
    var bundle = Bundle{};
    defer bundle.deinit(allocator);

    try loadEmbeddedCertificateBundle(allocator, &bundle);
    try std.testing.expect(bundle.map.count() > 0);
}

test "fetch downloads response body" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 1, successHandler, null);
    defer server.deinit();

    const body = try fetch(allocator, server.url, .{});
    defer allocator.free(body);

    try std.testing.expectEqualStrings("hello", body);
}

fn largeBodyHandler(stream: std.net.Stream, _: []const u8, _: ?*anyopaque) void {
    writeResponse(stream, 200, "OK", "", "abcdef");
}

test "fetch enforces max_response_bytes" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 1, largeBodyHandler, null);
    defer server.deinit();

    try std.testing.expectError(HttpError.ResponseTooLarge, fetch(allocator, server.url, .{
        .max_response_bytes = 3,
    }));
}

const StatusCtx = struct {
    status: u16,
    reason: []const u8,
};

fn statusHandler(stream: std.net.Stream, _: []const u8, user_data: ?*anyopaque) void {
    const ctx: *StatusCtx = @ptrCast(@alignCast(user_data.?));
    writeResponse(stream, ctx.status, ctx.reason, "", "error");
}

test "fetch maps HTTP status errors" {
    const allocator = std.testing.allocator;
    const cases = [_]struct {
        status: u16,
        reason: []const u8,
        expected: HttpError,
    }{
        .{ .status = 401, .reason = "Unauthorized", .expected = HttpError.Unauthorized },
        .{ .status = 404, .reason = "Not Found", .expected = HttpError.NotFound },
        .{ .status = 429, .reason = "Too Many Requests", .expected = HttpError.RateLimited },
        .{ .status = 500, .reason = "Server Error", .expected = HttpError.HttpError },
    };

    for (cases) |case| {
        var ctx = StatusCtx{ .status = case.status, .reason = case.reason };
        const server = try TestServer.start(allocator, 1, statusHandler, @ptrCast(&ctx));
        defer server.deinit();

        try std.testing.expectError(case.expected, fetch(allocator, server.url, .{}));
    }
}

fn redirectHandler(stream: std.net.Stream, request_head: []const u8, _: ?*anyopaque) void {
    if (std.mem.eql(u8, requestTarget(request_head), "/start")) {
        writeResponse(stream, 302, "Found", "location: /final\r\n", "");
        return;
    }
    writeResponse(stream, 200, "OK", "", "redirected");
}

test "fetch follows redirects" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 2, redirectHandler, null);
    defer server.deinit();
    const url = try std.fmt.allocPrint(allocator, "{s}/start", .{server.url});
    defer allocator.free(url);

    const body = try fetch(allocator, url, .{ .redirect_limit = 2 });
    defer allocator.free(body);

    try std.testing.expectEqualStrings("redirected", body);
}

fn redirectLoopHandler(stream: std.net.Stream, _: []const u8, _: ?*anyopaque) void {
    writeResponse(stream, 302, "Found", "location: /again\r\n", "");
}

test "fetch enforces redirect_limit" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 2, redirectLoopHandler, null);
    defer server.deinit();

    try std.testing.expectError(HttpError.RequestFailed, fetch(allocator, server.url, .{
        .redirect_limit = 1,
    }));
}

const HeaderCtx = struct {
    saw_auth: bool = false,
    saw_user_agent: bool = false,
    saw_identity_encoding: bool = false,
};

fn headerHandler(stream: std.net.Stream, request_head: []const u8, user_data: ?*anyopaque) void {
    const ctx: *HeaderCtx = @ptrCast(@alignCast(user_data.?));
    ctx.saw_auth = std.mem.eql(u8, requestHeader(request_head, "authorization") orelse "", "Bearer secret");
    ctx.saw_user_agent = std.mem.eql(u8, requestHeader(request_head, "user-agent") orelse "", "talu-test/1.0");
    ctx.saw_identity_encoding = std.mem.eql(u8, requestHeader(request_head, "accept-encoding") orelse "", "identity");
    writeResponse(stream, 200, "OK", "", "ok");
}

test "fetch sends bearer token user-agent and identity encoding headers" {
    const allocator = std.testing.allocator;
    var ctx = HeaderCtx{};
    const server = try TestServer.start(allocator, 1, headerHandler, @ptrCast(&ctx));
    var server_stopped = false;
    defer if (!server_stopped) server.deinit();

    const body = try fetch(allocator, server.url, .{
        .token = "secret",
        .user_agent = "talu-test/1.0",
    });
    defer allocator.free(body);

    try std.testing.expectEqualStrings("ok", body);
    server.deinit();
    server_stopped = true;
    try std.testing.expect(ctx.saw_auth);
    try std.testing.expect(ctx.saw_user_agent);
    try std.testing.expect(ctx.saw_identity_encoding);
}

const ProgressCtx = struct {
    calls: u32 = 0,
    last_downloaded: u64 = 0,
    last_total: u64 = 0,

    fn callback(downloaded: u64, total: u64, user_data: ?*anyopaque) void {
        const ctx: *ProgressCtx = @ptrCast(@alignCast(user_data.?));
        ctx.calls += 1;
        ctx.last_downloaded = downloaded;
        ctx.last_total = total;
    }
};

test "downloadToFile writes response body and reports progress totals" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 1, largeBodyHandler, null);
    defer server.deinit();

    const path = "/tmp/talu_http_download_success.bin";
    defer std.fs.cwd().deleteFile(path) catch {};
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var progress = ProgressCtx{};
    try downloadToFile(allocator, server.url, file, .{
        .progress_callback = ProgressCtx.callback,
        .progress_data = @ptrCast(&progress),
    });

    try std.testing.expect(progress.calls > 0);
    try std.testing.expectEqual(@as(u64, 6), progress.last_downloaded);
    try std.testing.expectEqual(@as(u64, 6), progress.last_total);

    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("abcdef", bytes);
}

fn streamingCancelHandler(stream: std.net.Stream, _: []const u8, _: ?*anyopaque) void {
    stream.writeAll("HTTP/1.1 200 OK\r\ncontent-length: 20000\r\nconnection: close\r\n\r\n") catch return;
    var chunk: [1024]u8 = undefined;
    @memset(chunk[0..], 'x');
    var remaining: usize = 20000;
    while (remaining > 0) {
        const n = @min(remaining, chunk.len);
        stream.writeAll(chunk[0..n]) catch return;
        remaining -= n;
    }
}

const CancelCtx = struct {
    cancel: bool = false,

    fn callback(_: u64, _: u64, user_data: ?*anyopaque) void {
        const ctx: *CancelCtx = @ptrCast(@alignCast(user_data.?));
        @atomicStore(bool, &ctx.cancel, true, .monotonic);
    }
};

test "downloadToFile returns Cancelled when cancel flag is set" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 1, streamingCancelHandler, null);
    defer server.deinit();

    const path = "/tmp/talu_http_download_cancel.bin";
    defer std.fs.cwd().deleteFile(path) catch {};
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var ctx = CancelCtx{};
    try std.testing.expectError(HttpError.Cancelled, downloadToFile(allocator, server.url, file, .{
        .progress_callback = CancelCtx.callback,
        .progress_data = @ptrCast(&ctx),
        .cancel_flag = &ctx.cancel,
    }));
}

test "downloadToFile maps write failures to StreamWriteFailed" {
    const allocator = std.testing.allocator;
    const server = try TestServer.start(allocator, 1, largeBodyHandler, null);
    defer server.deinit();

    const path = "/tmp/talu_http_download_readonly.bin";
    {
        const writable = try std.fs.cwd().createFile(path, .{});
        writable.close();
    }
    defer std.fs.cwd().deleteFile(path) catch {};

    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();

    try std.testing.expectError(HttpError.StreamWriteFailed, downloadToFile(allocator, server.url, file, .{}));
}

const ResumeCtx = struct {
    saw_range: bool = false,
};

fn resumePartialHandler(stream: std.net.Stream, request_head: []const u8, user_data: ?*anyopaque) void {
    const ctx: *ResumeCtx = @ptrCast(@alignCast(user_data.?));
    ctx.saw_range = std.mem.eql(u8, requestHeader(request_head, "range") orelse "", "bytes=5-");
    writeResponse(stream, 206, "Partial Content", "content-range: bytes 5-9/10\r\n", "world");
}

test "downloadToFile resumes with Range on 206 Partial Content" {
    const allocator = std.testing.allocator;
    var ctx = ResumeCtx{};
    const server = try TestServer.start(allocator, 1, resumePartialHandler, @ptrCast(&ctx));
    var server_stopped = false;
    defer if (!server_stopped) server.deinit();

    const path = "/tmp/talu_http_download_resume_206.bin";
    defer std.fs.cwd().deleteFile(path) catch {};
    {
        const existing = try std.fs.cwd().createFile(path, .{});
        defer existing.close();
        try existing.writeAll("hello");
    }
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_write });
    defer file.close();
    try file.seekTo(5);

    try downloadToFile(allocator, server.url, file, .{ .resume_from = 5 });

    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("helloworld", bytes);

    server.deinit();
    server_stopped = true;
    try std.testing.expect(ctx.saw_range);
}

fn resumeRestartHandler(stream: std.net.Stream, request_head: []const u8, user_data: ?*anyopaque) void {
    const ctx: *ResumeCtx = @ptrCast(@alignCast(user_data.?));
    ctx.saw_range = std.mem.eql(u8, requestHeader(request_head, "range") orelse "", "bytes=5-");
    writeResponse(stream, 200, "OK", "", "helloworld");
}

test "downloadToFile restarts from zero when resume receives 200 OK" {
    const allocator = std.testing.allocator;
    var ctx = ResumeCtx{};
    const server = try TestServer.start(allocator, 1, resumeRestartHandler, @ptrCast(&ctx));
    var server_stopped = false;
    defer if (!server_stopped) server.deinit();

    const path = "/tmp/talu_http_download_resume_200.bin";
    defer std.fs.cwd().deleteFile(path) catch {};
    {
        const existing = try std.fs.cwd().createFile(path, .{});
        defer existing.close();
        try existing.writeAll("hello");
    }
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_write });
    defer file.close();
    try file.seekTo(5);

    try downloadToFile(allocator, server.url, file, .{ .resume_from = 5 });

    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024);
    defer allocator.free(bytes);
    try std.testing.expectEqualStrings("helloworld", bytes);

    server.deinit();
    server_stopped = true;
    try std.testing.expect(ctx.saw_range);
}

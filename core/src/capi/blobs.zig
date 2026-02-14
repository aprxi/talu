//! C-API Blob Module - Raw content-addressable blob storage.
//!
//! This module provides C-compatible access to:
//! - Blob writes (`talu_blobs_put`)
//! - Blob streaming reads (`talu_blobs_open_stream`, `talu_blobs_stream_*`)

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const db_blob_store = @import("../db/blob/store.zig");

const allocator = std.heap.c_allocator;

/// Opaque handle for incremental blob reads.
/// Thread safety: NOT thread-safe.
pub const BlobStreamHandle = opaque {};

const BlobStreamState = struct {
    blob_store: db_blob_store.BlobStore,
    stream: db_blob_store.BlobReadStream,
};

fn validateDbPath(db_path: ?[*:0]const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is empty", .{});
        return null;
    }
    return slice;
}

fn validateRequiredArg(s: ?[*:0]const u8, comptime arg_name: []const u8) ?[]const u8 {
    const slice = std.mem.sliceTo(s orelse {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is null", .{});
        return null;
    }, 0);
    if (slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, arg_name ++ " is empty", .{});
        return null;
    }
    return slice;
}

fn blobErrorToCode(err: anyerror) error_codes.ErrorCode {
    return switch (err) {
        error.InvalidBlobRef,
        error.UnsupportedRefKind,
        error.InvalidMultipartManifest,
        error.InvalidChunkSize,
        => .invalid_argument,
        error.OutOfMemory => .out_of_memory,
        error.FileNotFound => .io_file_not_found,
        error.AccessDenied => .io_permission_denied,
        else => .storage_error,
    };
}

/// Store bytes in CAS blob storage and return a blob reference.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - bytes: Payload bytes (may be null when bytes_len=0)
///   - bytes_len: Payload byte length
///   - out_blob_ref: Output buffer for `sha256:<hex>` or `multi:<hex>`
///   - out_blob_ref_capacity: Output buffer capacity in bytes
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_put(
    db_path: ?[*:0]const u8,
    bytes: ?[*]const u8,
    bytes_len: usize,
    out_blob_ref: ?[*]u8,
    out_blob_ref_capacity: usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const out = out_blob_ref orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_blob_ref is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (out_blob_ref_capacity == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "out_blob_ref_capacity is zero", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    out[0] = 0;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    if (bytes == null and bytes_len > 0) {
        capi_error.setErrorWithCode(.invalid_argument, "bytes is null while bytes_len > 0", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    const payload: []const u8 = if (bytes_len == 0) &.{} else bytes.?[0..bytes_len];

    var blob_store = db_blob_store.BlobStore.init(allocator, db_path_slice) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to initialize blob store: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    defer blob_store.deinit();

    const blob_ref = blob_store.putAuto(payload) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to write blob: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    const blob_ref_slice = blob_ref.refSlice();

    if (out_blob_ref_capacity <= blob_ref_slice.len) {
        capi_error.setErrorWithCode(
            .resource_exhausted,
            "out_blob_ref buffer too small (need at least {d} bytes)",
            .{blob_ref_slice.len + 1},
        );
        return @intFromEnum(error_codes.ErrorCode.resource_exhausted);
    }

    @memcpy(out[0..blob_ref_slice.len], blob_ref_slice);
    out[blob_ref_slice.len] = 0;
    return 0;
}

/// Open a streaming reader for a blob reference.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - blob_ref: Blob reference (`sha256:<hex>` or `multi:<hex>`)
///   - out_stream: Output stream handle, close with talu_blobs_stream_close()
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_open_stream(
    db_path: ?[*:0]const u8,
    blob_ref: ?[*:0]const u8,
    out_stream: ?*?*BlobStreamHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_stream orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_stream is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const blob_ref_slice = validateRequiredArg(blob_ref, "blob_ref") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const state = allocator.create(BlobStreamState) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to allocate blob stream state: {s}", .{@errorName(err)});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(state);

    state.* = undefined;
    state.blob_store = db_blob_store.BlobStore.init(allocator, db_path_slice) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to initialize blob store: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    errdefer state.blob_store.deinit();

    state.stream = state.blob_store.openReadStream(allocator, blob_ref_slice) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to open blob stream: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    errdefer state.stream.deinit();

    out.* = @ptrCast(state);
    return 0;
}

/// Read bytes from a blob stream.
///
/// Parameters:
///   - stream_handle: Handle returned by talu_blobs_open_stream()
///   - out_buffer: Destination buffer
///   - out_buffer_len: Destination buffer size
///   - out_read_len: Number of bytes read (0 on EOF)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_stream_read(
    stream_handle: ?*BlobStreamHandle,
    out_buffer: ?[*]u8,
    out_buffer_len: usize,
    out_read_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const read_len_out = out_read_len orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_read_len is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    read_len_out.* = 0;

    const handle = stream_handle orelse {
        capi_error.setErrorWithCode(.invalid_handle, "stream_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };

    if (out_buffer_len > 0 and out_buffer == null) {
        capi_error.setErrorWithCode(.invalid_argument, "out_buffer is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    if (out_buffer_len == 0) return 0;

    const state: *BlobStreamState = @ptrCast(@alignCast(handle));
    const read_len = state.stream.read(out_buffer.?[0..out_buffer_len]) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to read blob stream: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    read_len_out.* = read_len;
    return 0;
}

/// Get total byte length for a blob stream.
pub export fn talu_blobs_stream_total_size(
    stream_handle: ?*BlobStreamHandle,
    out_total_size: ?*u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_total_size orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_total_size is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = 0;

    const handle = stream_handle orelse {
        capi_error.setErrorWithCode(.invalid_handle, "stream_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };

    const state: *BlobStreamState = @ptrCast(@alignCast(handle));
    out.* = state.stream.total_size;
    return 0;
}

/// Close a blob stream handle and release resources.
/// Safe to call with null.
pub export fn talu_blobs_stream_close(stream_handle: ?*BlobStreamHandle) callconv(.c) void {
    capi_error.clearError();
    const handle = stream_handle orelse return;
    const state: *BlobStreamState = @ptrCast(@alignCast(handle));
    state.stream.deinit();
    state.blob_store.deinit();
    allocator.destroy(state);
}

test "talu_blobs_put and talu_blobs_open_stream roundtrip bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const root_path_z = try std.testing.allocator.dupeZ(u8, root_path);
    defer std.testing.allocator.free(root_path_z);

    const payload = "blob-stream-roundtrip";

    var blob_ref_buf: [db_blob_store.ref_len + 1]u8 = undefined;
    const put_rc = talu_blobs_put(
        root_path_z.ptr,
        payload.ptr,
        payload.len,
        &blob_ref_buf,
        blob_ref_buf.len,
    );
    try std.testing.expectEqual(@as(i32, 0), put_rc);

    var stream_handle: ?*BlobStreamHandle = null;
    const open_rc = talu_blobs_open_stream(root_path_z.ptr, @ptrCast(&blob_ref_buf), &stream_handle);
    try std.testing.expectEqual(@as(i32, 0), open_rc);
    defer talu_blobs_stream_close(stream_handle);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    var buf: [5]u8 = undefined;
    while (true) {
        var read_len: usize = 0;
        const read_rc = talu_blobs_stream_read(stream_handle, &buf, buf.len, &read_len);
        try std.testing.expectEqual(@as(i32, 0), read_rc);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }

    try std.testing.expectEqualStrings(payload, out.items);
}

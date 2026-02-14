//! C-API Blob Module - Raw content-addressable blob storage.
//!
//! This module provides C-compatible access to:
//! - Blob writes (`talu_blobs_put`)
//! - Blob streaming reads (`talu_blobs_open_stream`, `talu_blobs_stream_*`)

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const capi_documents = @import("documents.zig");
const db_blob_gc = @import("../db/blob/gc.zig");
const db_blob_store = @import("../db/blob/store.zig");

const allocator = std.heap.c_allocator;

/// Opaque handle for incremental blob reads.
/// Thread safety: NOT thread-safe.
pub const BlobStreamHandle = opaque {};
/// Opaque handle for incremental blob writes.
/// Thread safety: NOT thread-safe.
pub const BlobWriteStreamHandle = opaque {};

const BlobStreamState = struct {
    blob_store: db_blob_store.BlobStore,
    stream: db_blob_store.BlobReadStream,
};

const BlobWriteStreamState = struct {
    blob_store: db_blob_store.BlobStore,
    stream: ?db_blob_store.BlobPutStream,
};

/// String list container for blob references.
pub const CStringList = capi_documents.CStringList;

/// C-compatible blob GC result statistics.
pub const BlobGcStats = extern struct {
    referenced_blob_count: usize,
    total_blob_files: usize,
    deleted_blob_files: usize,
    reclaimed_bytes: u64,
    invalid_reference_count: usize,
    skipped_invalid_entries: usize,
    skipped_recent_blob_files: usize,
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
        error.InvalidSeekOffset,
        => .invalid_argument,
        error.OutOfMemory => .out_of_memory,
        error.NoSpaceLeft,
        error.DiskQuota,
        error.FileTooBig,
        => .resource_exhausted,
        error.FileNotFound => .io_file_not_found,
        error.AccessDenied => .io_permission_denied,
        else => .storage_error,
    };
}

fn buildBlobRefList(digests: []const [db_blob_store.digest_hex_len]u8) !*CStringList {
    const list = allocator.create(CStringList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();
    const items = arena.alloc(?[*:0]const u8, digests.len) catch return error.OutOfMemory;
    for (digests, 0..) |digest, i| {
        const ref_buf = db_blob_store.buildSha256RefString(digest);
        items[i] = (arena.dupeZ(u8, ref_buf[0..db_blob_store.sha256_ref_len]) catch return error.OutOfMemory).ptr;
    }

    list.* = .{
        .items = if (digests.len > 0) items.ptr else null,
        .count = digests.len,
        ._arena = @ptrCast(arena_ptr),
    };
    return list;
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

/// Run blob mark-and-sweep garbage collection.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - min_blob_age_seconds: Grace period before unreferenced blobs are deleted
///   - out_stats: Output sweep statistics
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_gc(
    db_path: ?[*:0]const u8,
    min_blob_age_seconds: u64,
    out_stats: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();
    const out_ptr = out_stats orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_stats is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out: *BlobGcStats = @ptrCast(@alignCast(out_ptr));
    out.* = std.mem.zeroes(BlobGcStats);

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const stats = db_blob_gc.sweepUnreferencedBlobsWithOptions(allocator, db_path_slice, .{
        .min_blob_age_seconds = min_blob_age_seconds,
    }) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to sweep blobs: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };

    out.* = .{
        .referenced_blob_count = stats.referenced_blob_count,
        .total_blob_files = stats.total_blob_files,
        .deleted_blob_files = stats.deleted_blob_files,
        .reclaimed_bytes = stats.reclaimed_bytes,
        .invalid_reference_count = stats.invalid_reference_count,
        .skipped_invalid_entries = stats.skipped_invalid_entries,
        .skipped_recent_blob_files = stats.skipped_recent_blob_files,
    };
    return 0;
}

/// Check whether a blob reference resolves to stored bytes.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - blob_ref: Blob reference (`sha256:<hex>` or `multi:<hex>`)
///   - out_exists: Output boolean flag
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_exists(
    db_path: ?[*:0]const u8,
    blob_ref: ?[*:0]const u8,
    out_exists: ?*bool,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_exists orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_exists is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = false;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const blob_ref_slice = validateRequiredArg(blob_ref, "blob_ref") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    var blob_store = db_blob_store.BlobStore.init(allocator, db_path_slice) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to initialize blob store: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    defer blob_store.deinit();

    out.* = blob_store.exists(blob_ref_slice) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to check blob existence: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    return 0;
}

/// List stored physical blob references.
///
/// This returns `sha256:<hex>` references for stored shard files.
/// Caller must free with `talu_blobs_free_string_list()`.
pub export fn talu_blobs_list(
    db_path: ?[*:0]const u8,
    limit: usize,
    out_refs: ?*?*CStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_refs orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_refs is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var blob_store = db_blob_store.BlobStore.init(allocator, db_path_slice) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to initialize blob store: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    defer blob_store.deinit();

    const digests = blob_store.listSingleDigests(allocator, limit) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to list blobs: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    defer allocator.free(digests);

    out.* = buildBlobRefList(digests) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to build blob list: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    return 0;
}

pub export fn talu_blobs_free_string_list(list: ?*CStringList) callconv(.c) void {
    capi_error.clearError();
    const l = list orelse return;
    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
        l._arena = null;
    }
    allocator.destroy(l);
}

/// Open a streaming writer for blob uploads.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - out_stream: Output write stream handle, close with talu_blobs_write_stream_close()
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_open_write_stream(
    db_path: ?[*:0]const u8,
    out_stream: ?*?*BlobWriteStreamHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_stream orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_stream is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    const state = allocator.create(BlobWriteStreamState) catch |err| {
        capi_error.setErrorWithCode(.out_of_memory, "Failed to allocate blob write stream state: {s}", .{@errorName(err)});
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

    state.stream = db_blob_store.BlobPutStream.init(&state.blob_store, allocator);
    out.* = @ptrCast(state);
    return 0;
}

/// Write bytes into a blob write stream.
///
/// Parameters:
///   - stream_handle: Handle from talu_blobs_open_write_stream()
///   - bytes: Payload chunk pointer (may be null when bytes_len=0)
///   - bytes_len: Payload chunk length
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_write_stream_write(
    stream_handle: ?*BlobWriteStreamHandle,
    bytes: ?[*]const u8,
    bytes_len: usize,
) callconv(.c) i32 {
    capi_error.clearError();
    const handle = stream_handle orelse {
        capi_error.setErrorWithCode(.invalid_handle, "stream_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };
    if (bytes == null and bytes_len > 0) {
        capi_error.setErrorWithCode(.invalid_argument, "bytes is null while bytes_len > 0", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    if (bytes_len == 0) return 0;

    const state: *BlobWriteStreamState = @ptrCast(@alignCast(handle));
    const stream = if (state.stream) |*stream| stream else {
        capi_error.setErrorWithCode(.invalid_handle, "stream is already finalized", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };
    stream.write(bytes.?[0..bytes_len]) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to write blob stream chunk: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    return 0;
}

/// Finalize a blob write stream and return a blob reference.
///
/// Parameters:
///   - stream_handle: Handle from talu_blobs_open_write_stream()
///   - out_blob_ref: Output buffer for `sha256:<hex>` or `multi:<hex>`
///   - out_blob_ref_capacity: Output buffer capacity in bytes
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_write_stream_finish(
    stream_handle: ?*BlobWriteStreamHandle,
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

    const handle = stream_handle orelse {
        capi_error.setErrorWithCode(.invalid_handle, "stream_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };
    const state: *BlobWriteStreamState = @ptrCast(@alignCast(handle));
    const stream = if (state.stream) |*stream| stream else {
        capi_error.setErrorWithCode(.invalid_handle, "stream is already finalized", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };

    const blob_ref = stream.finish() catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to finalize blob stream: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
    stream.deinit();
    state.stream = null;

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

/// Seek a blob stream to an absolute byte offset.
///
/// Parameters:
///   - stream_handle: Handle returned by talu_blobs_open_stream()
///   - offset_bytes: Absolute offset from start of blob
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_blobs_stream_seek(
    stream_handle: ?*BlobStreamHandle,
    offset_bytes: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const handle = stream_handle orelse {
        capi_error.setErrorWithCode(.invalid_handle, "stream_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_handle);
    };

    const state: *BlobStreamState = @ptrCast(@alignCast(handle));
    state.stream.seek(offset_bytes) catch |err| {
        const code = blobErrorToCode(err);
        capi_error.setErrorWithCode(code, "Failed to seek blob stream: {s}", .{@errorName(err)});
        return @intFromEnum(code);
    };
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

/// Close a blob write stream handle and release resources.
/// Safe to call with null.
pub export fn talu_blobs_write_stream_close(stream_handle: ?*BlobWriteStreamHandle) callconv(.c) void {
    capi_error.clearError();
    const handle = stream_handle orelse return;
    const state: *BlobWriteStreamState = @ptrCast(@alignCast(handle));
    if (state.stream) |*stream| {
        stream.deinit();
        state.stream = null;
    }
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

test "talu_blobs_write_stream_write and finish roundtrip bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const root_path_z = try std.testing.allocator.dupeZ(u8, root_path);
    defer std.testing.allocator.free(root_path_z);

    var write_handle: ?*BlobWriteStreamHandle = null;
    const open_rc = talu_blobs_open_write_stream(root_path_z.ptr, &write_handle);
    try std.testing.expectEqual(@as(i32, 0), open_rc);
    defer talu_blobs_write_stream_close(write_handle);

    const chunk_a = "write-stream-";
    const chunk_b = "payload";
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_write_stream_write(write_handle, chunk_a.ptr, chunk_a.len),
    );
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_write_stream_write(write_handle, chunk_b.ptr, chunk_b.len),
    );

    var blob_ref_buf: [db_blob_store.ref_len + 1]u8 = undefined;
    const finish_rc = talu_blobs_write_stream_finish(
        write_handle,
        &blob_ref_buf,
        blob_ref_buf.len,
    );
    try std.testing.expectEqual(@as(i32, 0), finish_rc);

    var stream_handle: ?*BlobStreamHandle = null;
    const open_read_rc = talu_blobs_open_stream(root_path_z.ptr, @ptrCast(&blob_ref_buf), &stream_handle);
    try std.testing.expectEqual(@as(i32, 0), open_read_rc);
    defer talu_blobs_stream_close(stream_handle);

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    var buf: [8]u8 = undefined;
    while (true) {
        var read_len: usize = 0;
        const read_rc = talu_blobs_stream_read(stream_handle, &buf, buf.len, &read_len);
        try std.testing.expectEqual(@as(i32, 0), read_rc);
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }

    try std.testing.expectEqualStrings("write-stream-payload", out.items);
}

test "talu_blobs_exists reports true for stored blob and false for missing blob" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const root_path_z = try std.testing.allocator.dupeZ(u8, root_path);
    defer std.testing.allocator.free(root_path_z);

    const payload = "blob-exists-check";
    var blob_ref_buf: [db_blob_store.ref_len + 1]u8 = undefined;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_put(root_path_z.ptr, payload.ptr, payload.len, &blob_ref_buf, blob_ref_buf.len),
    );

    var exists = false;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_exists(root_path_z.ptr, @ptrCast(&blob_ref_buf), &exists),
    );
    try std.testing.expect(exists);

    const missing: [*:0]const u8 = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    exists = true;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_exists(root_path_z.ptr, missing, &exists),
    );
    try std.testing.expect(!exists);
}

test "talu_blobs_list returns blob refs and talu_blobs_free_string_list releases list" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const root_path_z = try std.testing.allocator.dupeZ(u8, root_path);
    defer std.testing.allocator.free(root_path_z);

    var ref_a: [db_blob_store.ref_len + 1]u8 = undefined;
    var ref_b: [db_blob_store.ref_len + 1]u8 = undefined;
    try std.testing.expectEqual(@as(i32, 0), talu_blobs_put(root_path_z.ptr, "a".ptr, 1, &ref_a, ref_a.len));
    try std.testing.expectEqual(@as(i32, 0), talu_blobs_put(root_path_z.ptr, "b".ptr, 1, &ref_b, ref_b.len));

    var out_list: ?*CStringList = null;
    try std.testing.expectEqual(@as(i32, 0), talu_blobs_list(root_path_z.ptr, 0, &out_list));
    defer talu_blobs_free_string_list(out_list);
    const list = out_list.?;
    try std.testing.expect(list.count >= 2);
}

test "talu_blobs_stream_seek repositions reads" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const root_path_z = try std.testing.allocator.dupeZ(u8, root_path);
    defer std.testing.allocator.free(root_path_z);

    const payload = "seekable-stream-payload";

    var blob_ref_buf: [db_blob_store.ref_len + 1]u8 = undefined;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_put(root_path_z.ptr, payload.ptr, payload.len, &blob_ref_buf, blob_ref_buf.len),
    );

    var stream_handle: ?*BlobStreamHandle = null;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_open_stream(root_path_z.ptr, @ptrCast(&blob_ref_buf), &stream_handle),
    );
    defer talu_blobs_stream_close(stream_handle);

    try std.testing.expectEqual(@as(i32, 0), talu_blobs_stream_seek(stream_handle, 4));

    var out = std.ArrayList(u8).empty;
    defer out.deinit(std.testing.allocator);
    var buf: [5]u8 = undefined;
    while (true) {
        var read_len: usize = 0;
        try std.testing.expectEqual(
            @as(i32, 0),
            talu_blobs_stream_read(stream_handle, &buf, buf.len, &read_len),
        );
        if (read_len == 0) break;
        try out.appendSlice(std.testing.allocator, buf[0..read_len]);
    }
    try std.testing.expectEqualStrings(payload[4..], out.items);
}

test "talu_blobs_gc deletes unreferenced blobs when grace period is zero" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const root_path_z = try std.testing.allocator.dupeZ(u8, root_path);
    defer std.testing.allocator.free(root_path_z);

    var blob_ref_buf: [db_blob_store.ref_len + 1]u8 = undefined;
    try std.testing.expectEqual(
        @as(i32, 0),
        talu_blobs_put(root_path_z.ptr, "orphan".ptr, "orphan".len, &blob_ref_buf, blob_ref_buf.len),
    );

    var stats = std.mem.zeroes(BlobGcStats);
    try std.testing.expectEqual(@as(i32, 0), talu_blobs_gc(root_path_z.ptr, 0, &stats));
    try std.testing.expectEqual(@as(usize, 1), stats.total_blob_files);
    try std.testing.expectEqual(@as(usize, 1), stats.deleted_blob_files);
}

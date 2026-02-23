//! Plane-specific DB C-API: Blob.

const blobs = @import("../blobs_impl.zig");

pub const BlobGcStats = blobs.BlobGcStats;
pub const BlobStreamHandle = blobs.BlobStreamHandle;
pub const BlobWriteStreamHandle = blobs.BlobWriteStreamHandle;
pub const CStringList = blobs.CStringList;

pub export fn talu_db_blob_put(
    db_path: ?[*:0]const u8,
    bytes: ?[*]const u8,
    bytes_len: usize,
    out_blob_ref: ?[*]u8,
    out_blob_ref_capacity: usize,
) callconv(.c) i32 {
    return blobs.talu_blobs_put(db_path, bytes, bytes_len, out_blob_ref, out_blob_ref_capacity);
}

pub export fn talu_db_blob_gc(
    db_path: ?[*:0]const u8,
    min_blob_age_seconds: u64,
    out_stats: ?*anyopaque,
) callconv(.c) i32 {
    return blobs.talu_blobs_gc(db_path, min_blob_age_seconds, out_stats);
}

pub export fn talu_db_blob_exists(
    db_path: ?[*:0]const u8,
    blob_ref: ?[*:0]const u8,
    out_exists: ?*bool,
) callconv(.c) i32 {
    return blobs.talu_blobs_exists(db_path, blob_ref, out_exists);
}

pub export fn talu_db_blob_list(
    db_path: ?[*:0]const u8,
    limit: usize,
    out_refs: ?*?*blobs.CStringList,
) callconv(.c) i32 {
    return blobs.talu_blobs_list(db_path, limit, out_refs);
}

pub export fn talu_db_blob_free_string_list(list: ?*blobs.CStringList) callconv(.c) void {
    blobs.talu_blobs_free_string_list(list);
}

pub export fn talu_db_blob_open_write_stream(
    db_path: ?[*:0]const u8,
    out_stream: ?*?*blobs.BlobWriteStreamHandle,
) callconv(.c) i32 {
    return blobs.talu_blobs_open_write_stream(db_path, out_stream);
}

pub export fn talu_db_blob_write_stream_write(
    stream_handle: ?*blobs.BlobWriteStreamHandle,
    bytes: ?[*]const u8,
    bytes_len: usize,
) callconv(.c) i32 {
    return blobs.talu_blobs_write_stream_write(stream_handle, bytes, bytes_len);
}

pub export fn talu_db_blob_write_stream_finish(
    stream_handle: ?*blobs.BlobWriteStreamHandle,
    out_blob_ref: ?[*]u8,
    out_blob_ref_capacity: usize,
) callconv(.c) i32 {
    return blobs.talu_blobs_write_stream_finish(stream_handle, out_blob_ref, out_blob_ref_capacity);
}

pub export fn talu_db_blob_write_stream_close(stream_handle: ?*blobs.BlobWriteStreamHandle) callconv(.c) void {
    blobs.talu_blobs_write_stream_close(stream_handle);
}

pub export fn talu_db_blob_open_stream(
    db_path: ?[*:0]const u8,
    blob_ref: ?[*:0]const u8,
    out_stream: ?*?*blobs.BlobStreamHandle,
) callconv(.c) i32 {
    return blobs.talu_blobs_open_stream(db_path, blob_ref, out_stream);
}

pub export fn talu_db_blob_stream_read(
    stream_handle: ?*blobs.BlobStreamHandle,
    out_buffer: ?[*]u8,
    out_buffer_len: usize,
    out_read_len: ?*usize,
) callconv(.c) i32 {
    return blobs.talu_blobs_stream_read(stream_handle, out_buffer, out_buffer_len, out_read_len);
}

pub export fn talu_db_blob_stream_total_size(
    stream_handle: ?*blobs.BlobStreamHandle,
    out_total_size: ?*u64,
) callconv(.c) i32 {
    return blobs.talu_blobs_stream_total_size(stream_handle, out_total_size);
}

pub export fn talu_db_blob_stream_seek(
    stream_handle: ?*blobs.BlobStreamHandle,
    offset_bytes: u64,
) callconv(.c) i32 {
    return blobs.talu_blobs_stream_seek(stream_handle, offset_bytes);
}

pub export fn talu_db_blob_stream_close(stream_handle: ?*blobs.BlobStreamHandle) callconv(.c) void {
    blobs.talu_blobs_stream_close(stream_handle);
}

//! Plane-specific DB C-API: Maintenance operations.

const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const legacy = @import("../db_impl.zig");

pub export fn talu_db_ops_simulate_crash(chat_handle: ?*anyopaque) callconv(.c) i32 {
    return legacy.talu_chat_simulate_crash(@ptrCast(chat_handle));
}

pub export fn talu_db_ops_set_storage_db(
    chat_handle: ?*anyopaque,
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    return legacy.talu_chat_set_storage_db(@ptrCast(chat_handle), db_path, session_id);
}

pub export fn talu_db_ops_set_max_segment_size(
    chat_handle: ?*anyopaque,
    max_bytes: u64,
) callconv(.c) i32 {
    return legacy.talu_chat_set_max_segment_size(@ptrCast(chat_handle), max_bytes);
}

pub export fn talu_db_ops_set_durability(
    chat_handle: ?*anyopaque,
    mode: u8,
) callconv(.c) i32 {
    return legacy.talu_chat_set_durability(@ptrCast(chat_handle), mode);
}

pub export fn talu_db_ops_vector_compact(
    handle: ?*legacy.VectorStoreHandle,
    dims: u32,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_compact(handle, dims, out_kept_count, out_removed_tombstones);
}

pub export fn talu_db_ops_vector_build_indexes_with_generation(
    handle: ?*legacy.VectorStoreHandle,
    expected_generation: u64,
    max_segments: usize,
    out_built_segments: *usize,
    out_failed_segments: *usize,
    out_pending_segments: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_build_indexes_with_generation(
        handle,
        expected_generation,
        max_segments,
        out_built_segments,
        out_failed_segments,
        out_pending_segments,
    );
}

pub export fn talu_db_ops_snapshot_create(
    _db_path: ?[*:0]const u8,
    _out_snapshot_id: ?[*]u8,
    _out_snapshot_id_capacity: usize,
) callconv(.c) i32 {
    _ = _db_path;
    _ = _out_snapshot_id;
    _ = _out_snapshot_id_capacity;
    capi_error.clearError();
    capi_error.setErrorWithCode(.invalid_argument, "snapshot controls are not implemented yet", .{});
    return @intFromEnum(error_codes.ErrorCode.invalid_argument);
}

pub export fn talu_db_ops_snapshot_release(
    _db_path: ?[*:0]const u8,
    _snapshot_id: ?[*:0]const u8,
) callconv(.c) i32 {
    _ = _db_path;
    _ = _snapshot_id;
    capi_error.clearError();
    capi_error.setErrorWithCode(.invalid_argument, "snapshot controls are not implemented yet", .{});
    return @intFromEnum(error_codes.ErrorCode.invalid_argument);
}

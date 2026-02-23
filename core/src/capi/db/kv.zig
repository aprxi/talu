//! Plane-specific DB C-API: KV.
//!
//! Current implementation forwards to repository pin metadata adapter.

const repo_meta = @import("../repo_meta.zig");

pub export fn talu_db_kv_init(
    db_path: ?[*:0]const u8,
    out_handle: ?*?*repo_meta.PinAdapterHandle,
) callconv(.c) i32 {
    return repo_meta.talu_repo_meta_init(db_path, out_handle);
}

pub export fn talu_db_kv_free(handle: ?*repo_meta.PinAdapterHandle) callconv(.c) void {
    repo_meta.talu_repo_meta_free(handle);
}

pub export fn talu_db_kv_put(
    handle: ?*repo_meta.PinAdapterHandle,
    key: ?[*:0]const u8,
) callconv(.c) i32 {
    return repo_meta.talu_repo_meta_pin(handle, key);
}

pub export fn talu_db_kv_delete(
    handle: ?*repo_meta.PinAdapterHandle,
    key: ?[*:0]const u8,
) callconv(.c) i32 {
    return repo_meta.talu_repo_meta_unpin(handle, key);
}

pub export fn talu_db_kv_update_size(
    handle: ?*repo_meta.PinAdapterHandle,
    key: ?[*:0]const u8,
    size_bytes: u64,
) callconv(.c) i32 {
    return repo_meta.talu_repo_meta_update_size(handle, key, size_bytes);
}

pub export fn talu_db_kv_clear_size(
    handle: ?*repo_meta.PinAdapterHandle,
    key: ?[*:0]const u8,
) callconv(.c) i32 {
    return repo_meta.talu_repo_meta_clear_size(handle, key);
}

pub export fn talu_db_kv_list(
    handle: ?*repo_meta.PinAdapterHandle,
    out_list: *repo_meta.CPinList,
) callconv(.c) i32 {
    return repo_meta.talu_repo_meta_list_pins(handle, out_list);
}

pub export fn talu_db_kv_free_list(list: *repo_meta.CPinList) callconv(.c) void {
    repo_meta.talu_repo_meta_free_list(list);
}

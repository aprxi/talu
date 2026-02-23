//! Plane-specific DB C-API: Vector.
//!
//! Canonical exports use `talu_db_vector_*` and forward to existing
//! vector store implementations.

const legacy = @import("../db_impl.zig");

pub export fn talu_db_vector_init(
    db_path: ?[*:0]const u8,
    out_handle: ?*?*legacy.VectorStoreHandle,
) callconv(.c) i32 {
    return legacy.talu_vector_store_init(db_path, out_handle);
}

pub export fn talu_db_vector_free(handle: ?*legacy.VectorStoreHandle) callconv(.c) void {
    legacy.talu_vector_store_free(handle);
}

pub export fn talu_db_vector_simulate_crash(handle: ?*legacy.VectorStoreHandle) callconv(.c) void {
    legacy.talu_vector_store_simulate_crash(handle);
}

pub export fn talu_db_vector_set_durability(
    handle: ?*legacy.VectorStoreHandle,
    mode: u8,
) callconv(.c) i32 {
    return legacy.talu_vector_store_set_durability(handle, mode);
}

pub export fn talu_db_vector_append(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
) callconv(.c) i32 {
    return legacy.talu_vector_store_append(handle, ids_ptr, vectors_ptr, count, dims);
}

pub export fn talu_db_vector_append_ex(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
    reject_existing: bool,
) callconv(.c) i32 {
    return legacy.talu_vector_store_append_ex(
        handle,
        ids_ptr,
        vectors_ptr,
        count,
        dims,
        normalize,
        reject_existing,
    );
}

pub export fn talu_db_vector_append_idempotent_ex(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
    reject_existing: bool,
    key_hash: u64,
    request_hash: u64,
) callconv(.c) i32 {
    return legacy.talu_vector_store_append_idempotent_ex(
        handle,
        ids_ptr,
        vectors_ptr,
        count,
        dims,
        normalize,
        reject_existing,
        key_hash,
        request_hash,
    );
}

pub export fn talu_db_vector_upsert(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
) callconv(.c) i32 {
    return legacy.talu_vector_store_upsert(handle, ids_ptr, vectors_ptr, count, dims);
}

pub export fn talu_db_vector_upsert_ex(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
) callconv(.c) i32 {
    return legacy.talu_vector_store_upsert_ex(handle, ids_ptr, vectors_ptr, count, dims, normalize);
}

pub export fn talu_db_vector_upsert_idempotent_ex(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
    key_hash: u64,
    request_hash: u64,
) callconv(.c) i32 {
    return legacy.talu_vector_store_upsert_idempotent_ex(
        handle,
        ids_ptr,
        vectors_ptr,
        count,
        dims,
        normalize,
        key_hash,
        request_hash,
    );
}

pub export fn talu_db_vector_delete(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    count: usize,
    out_deleted_count: *usize,
    out_not_found_count: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_delete(
        handle,
        ids_ptr,
        count,
        out_deleted_count,
        out_not_found_count,
    );
}

pub export fn talu_db_vector_delete_idempotent(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    count: usize,
    key_hash: u64,
    request_hash: u64,
    out_deleted_count: *usize,
    out_not_found_count: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_delete_idempotent(
        handle,
        ids_ptr,
        count,
        key_hash,
        request_hash,
        out_deleted_count,
        out_not_found_count,
    );
}

pub export fn talu_db_vector_fetch(
    handle: ?*legacy.VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    count: usize,
    include_values: bool,
    out_ids: *?[*]u64,
    out_vectors: *?[*]f32,
    out_found_count: *usize,
    out_dims: *u32,
    out_missing_ids: *?[*]u64,
    out_missing_count: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_fetch(
        handle,
        ids_ptr,
        count,
        include_values,
        out_ids,
        out_vectors,
        out_found_count,
        out_dims,
        out_missing_ids,
        out_missing_count,
    );
}

pub export fn talu_db_vector_free_fetch(
    ids_ptr: ?[*]u64,
    vectors_ptr: ?[*]f32,
    found_count: usize,
    dims: u32,
    missing_ids_ptr: ?[*]u64,
    missing_count: usize,
) callconv(.c) void {
    legacy.talu_vector_store_free_fetch(
        ids_ptr,
        vectors_ptr,
        found_count,
        dims,
        missing_ids_ptr,
        missing_count,
    );
}

pub export fn talu_db_vector_stats(
    handle: ?*legacy.VectorStoreHandle,
    out_visible_count: *usize,
    out_tombstone_count: *usize,
    out_segment_count: *usize,
    out_total_count: *usize,
    out_manifest_generation: *u64,
    out_index_ready_segments: *usize,
    out_index_pending_segments: *usize,
    out_index_failed_segments: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_stats(
        handle,
        out_visible_count,
        out_tombstone_count,
        out_segment_count,
        out_total_count,
        out_manifest_generation,
        out_index_ready_segments,
        out_index_pending_segments,
        out_index_failed_segments,
    );
}

pub export fn talu_db_vector_compact(
    handle: ?*legacy.VectorStoreHandle,
    dims: u32,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_compact(handle, dims, out_kept_count, out_removed_tombstones);
}

pub export fn talu_db_vector_compact_idempotent(
    handle: ?*legacy.VectorStoreHandle,
    dims: u32,
    key_hash: u64,
    request_hash: u64,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_compact_idempotent(
        handle,
        dims,
        key_hash,
        request_hash,
        out_kept_count,
        out_removed_tombstones,
    );
}

pub export fn talu_db_vector_compact_with_generation(
    handle: ?*legacy.VectorStoreHandle,
    dims: u32,
    expected_generation: u64,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_compact_with_generation(
        handle,
        dims,
        expected_generation,
        out_kept_count,
        out_removed_tombstones,
    );
}

pub export fn talu_db_vector_compact_expired_tombstones(
    handle: ?*legacy.VectorStoreHandle,
    dims: u32,
    now_ms: i64,
    max_age_ms: i64,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_compact_expired_tombstones(
        handle,
        dims,
        now_ms,
        max_age_ms,
        out_kept_count,
        out_removed_tombstones,
    );
}

pub export fn talu_db_vector_build_indexes_with_generation(
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

pub export fn talu_db_vector_changes(
    handle: ?*legacy.VectorStoreHandle,
    since: u64,
    limit: usize,
    out_seqs: *?[*]u64,
    out_ops: *?[*]u8,
    out_ids: *?[*]u64,
    out_timestamps: *?[*]i64,
    out_count: *usize,
    out_has_more: *bool,
    out_next_since: *u64,
) callconv(.c) i32 {
    return legacy.talu_vector_store_changes(
        handle,
        since,
        limit,
        out_seqs,
        out_ops,
        out_ids,
        out_timestamps,
        out_count,
        out_has_more,
        out_next_since,
    );
}

pub export fn talu_db_vector_free_changes(
    seqs_ptr: ?[*]u64,
    ops_ptr: ?[*]u8,
    ids_ptr: ?[*]u64,
    timestamps_ptr: ?[*]i64,
    count: usize,
) callconv(.c) void {
    legacy.talu_vector_store_free_changes(seqs_ptr, ops_ptr, ids_ptr, timestamps_ptr, count);
}

pub export fn talu_db_vector_load(
    handle: ?*legacy.VectorStoreHandle,
    out_ids: *?[*]u64,
    out_vectors: *?[*]f32,
    out_count: *usize,
    out_dims: *u32,
) callconv(.c) i32 {
    return legacy.talu_vector_store_load(handle, out_ids, out_vectors, out_count, out_dims);
}

pub export fn talu_db_vector_free_load(
    ids_ptr: ?[*]u64,
    vectors_ptr: ?[*]f32,
    count: usize,
    dims: u32,
) callconv(.c) void {
    legacy.talu_vector_store_free_load(ids_ptr, vectors_ptr, count, dims);
}

pub export fn talu_db_vector_search(
    handle: ?*legacy.VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    k: u32,
    out_ids: *?[*]u64,
    out_scores: *?[*]f32,
    out_count: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_search(
        handle,
        query_ptr,
        query_len,
        k,
        out_ids,
        out_scores,
        out_count,
    );
}

pub export fn talu_db_vector_free_search(
    ids_ptr: ?[*]u64,
    scores_ptr: ?[*]f32,
    count: usize,
) callconv(.c) void {
    legacy.talu_vector_store_free_search(ids_ptr, scores_ptr, count);
}

pub export fn talu_db_vector_search_batch_ex(
    handle: ?*legacy.VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    dims: u32,
    query_count: u32,
    k: u32,
    normalize_queries: bool,
    approximate: bool,
    out_ids: *?[*]u64,
    out_scores: *?[*]f32,
    out_count_per_query: *u32,
) callconv(.c) i32 {
    return legacy.talu_vector_store_search_batch_ex(
        handle,
        query_ptr,
        query_len,
        dims,
        query_count,
        k,
        normalize_queries,
        approximate,
        out_ids,
        out_scores,
        out_count_per_query,
    );
}

pub export fn talu_db_vector_search_batch(
    handle: ?*legacy.VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    dims: u32,
    query_count: u32,
    k: u32,
    out_ids: *?[*]u64,
    out_scores: *?[*]f32,
    out_count_per_query: *u32,
) callconv(.c) i32 {
    return legacy.talu_vector_store_search_batch(
        handle,
        query_ptr,
        query_len,
        dims,
        query_count,
        k,
        out_ids,
        out_scores,
        out_count_per_query,
    );
}

pub export fn talu_db_vector_scan_batch(
    handle: ?*legacy.VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    dims: u32,
    query_count: u32,
    out_ids: ?[*]u64,
    ids_len: usize,
    out_scores: ?[*]f32,
    scores_len: usize,
    out_total_rows: *usize,
) callconv(.c) i32 {
    return legacy.talu_vector_store_scan_batch(
        handle,
        query_ptr,
        query_len,
        dims,
        query_count,
        out_ids,
        ids_len,
        out_scores,
        scores_len,
        out_total_rows,
    );
}

pub export fn talu_db_vector_free_search_batch(
    ids_ptr: ?[*]u64,
    scores_ptr: ?[*]f32,
    count_per_query: u32,
    query_count: u32,
) callconv(.c) void {
    legacy.talu_vector_store_free_search_batch(ids_ptr, scores_ptr, count_per_query, query_count);
}

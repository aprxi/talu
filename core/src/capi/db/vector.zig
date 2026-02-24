//! DB C-API: Vector store operations.

const std = @import("std");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const db = @import("../../db/root.zig");
const tensor_mod = @import("../../tensor.zig");
const helpers = @import("helpers.zig");

const allocator = std.heap.c_allocator;
const setArgError = helpers.setArgError;

/// Opaque handle for TaluDB vector backend.
/// Thread safety: NOT thread-safe (single-writer semantics).
pub const VectorStoreHandle = opaque {};

/// Initialize a vector store.
///
/// The path must point to a directory (created if nonexistent).
/// Caller must free the returned handle with `talu_db_vector_free`.
pub export fn talu_db_vector_init(
    db_path: ?[*:0]const u8,
    out_handle: ?*?*VectorStoreHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_z = db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const db_path_slice = std.mem.sliceTo(db_path_z, 0);
    if (db_path_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const backend_ptr = db.vector.store.create(allocator, db_path_slice) catch |err| {
        capi_error.setError(err, "failed to initialize vector store", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out.* = @ptrCast(backend_ptr);
    return 0;
}

/// Free a vector store handle.
pub export fn talu_db_vector_free(handle: ?*VectorStoreHandle) callconv(.c) void {
    capi_error.clearError();
    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse return));
    db.vector.store.destroy(allocator, backend);
}

/// Simulate a process crash for a vector store (testing only).
///
/// Releases all file descriptors and locks WITHOUT flushing or deleting
/// the WAL file. After this call, the handle is invalid — call
/// `talu_db_vector_free` would double-free; instead, just discard
/// the handle pointer.
pub export fn talu_db_vector_simulate_crash(handle: ?*VectorStoreHandle) callconv(.c) void {
    capi_error.clearError();
    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse return));
    backend.simulateCrash();
    allocator.destroy(backend);
}

/// Set durability mode for a vector store.
///
/// Mode values:
/// - 0: Relaxed (fsync WAL on explicit flush only)
/// - 1: Strict (fsync WAL after every transaction)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_db_vector_set_durability(
    handle: ?*VectorStoreHandle,
    mode: u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "vector store handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const durability = std.meta.intToEnum(db.Durability, mode) catch {
        capi_error.setErrorWithCode(.invalid_argument, "invalid durability mode (expected 0 or 1)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    backend.fs_writer.durability = durability;
    return 0;
}

/// Append a batch of vectors to the store.
pub export fn talu_db_vector_append(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
) callconv(.c) i32 {
    return talu_db_vector_append_ex(handle, ids_ptr, vectors_ptr, count, dims, false, false);
}

/// Append a batch of vectors with explicit mutation options.
pub export fn talu_db_vector_append_ex(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
    reject_existing: bool,
) callconv(.c) i32 {
    return talu_db_vector_append_idempotent_ex(
        handle,
        ids_ptr,
        vectors_ptr,
        count,
        dims,
        normalize,
        reject_existing,
        0,
        0,
    );
}

/// Append vectors with durable idempotency semantics.
pub export fn talu_db_vector_append_idempotent_ex(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
    reject_existing: bool,
    key_hash: u64,
    request_hash: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const ids = ids_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "ids_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const vectors = vectors_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "vectors_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (count == 0) return 0;

    const total = count * @as(usize, dims);
    backend.appendBatchIdempotentWithOptions(ids[0..count], vectors[0..total], dims, .{
        .normalize = normalize,
        .reject_existing = reject_existing,
    }, key_hash, request_hash) catch |err| {
        switch (err) {
            error.IdempotencyConflict => capi_error.setError(err, "idempotency conflict", .{}),
            error.AlreadyExists => capi_error.setError(err, "already exists", .{}),
            else => capi_error.setError(err, "failed to append vectors", .{}),
        }
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Upsert a batch of vectors in the store.
pub export fn talu_db_vector_upsert(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
) callconv(.c) i32 {
    return talu_db_vector_upsert_ex(handle, ids_ptr, vectors_ptr, count, dims, false);
}

/// Upsert a batch of vectors with explicit mutation options.
pub export fn talu_db_vector_upsert_ex(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
) callconv(.c) i32 {
    return talu_db_vector_upsert_idempotent_ex(
        handle,
        ids_ptr,
        vectors_ptr,
        count,
        dims,
        normalize,
        0,
        0,
    );
}

/// Upsert vectors with durable idempotency semantics.
pub export fn talu_db_vector_upsert_idempotent_ex(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
    normalize: bool,
    key_hash: u64,
    request_hash: u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const ids = ids_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "ids_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const vectors = vectors_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "vectors_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (count == 0) return 0;

    const total = count * @as(usize, dims);
    backend.upsertBatchIdempotentWithOptions(ids[0..count], vectors[0..total], dims, .{
        .normalize = normalize,
        .reject_existing = false,
    }, key_hash, request_hash) catch |err| {
        switch (err) {
            error.IdempotencyConflict => capi_error.setError(err, "idempotency conflict", .{}),
            else => capi_error.setError(err, "failed to upsert vectors", .{}),
        }
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Delete vectors by ID with tombstone semantics.
pub export fn talu_db_vector_delete(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    count: usize,
    out_deleted_count: *usize,
    out_not_found_count: *usize,
) callconv(.c) i32 {
    return talu_db_vector_delete_idempotent(
        handle,
        ids_ptr,
        count,
        0,
        0,
        out_deleted_count,
        out_not_found_count,
    );
}

/// Delete vectors with durable idempotency semantics.
pub export fn talu_db_vector_delete_idempotent(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    count: usize,
    key_hash: u64,
    request_hash: u64,
    out_deleted_count: *usize,
    out_not_found_count: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_deleted_count.* = 0;
    out_not_found_count.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const ids = ids_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "ids_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const result = backend.deleteIdsIdempotent(ids[0..count], key_hash, request_hash) catch |err| {
        switch (err) {
            error.IdempotencyConflict => capi_error.setError(err, "idempotency conflict", .{}),
            else => capi_error.setError(err, "failed to delete vectors", .{}),
        }
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_deleted_count.* = result.deleted_count;
    out_not_found_count.* = result.not_found_count;
    return 0;
}

/// Fetch vectors by ID from visible state.
pub export fn talu_db_vector_fetch(
    handle: ?*VectorStoreHandle,
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
    capi_error.clearError();
    out_ids.* = null;
    out_vectors.* = null;
    out_found_count.* = 0;
    out_dims.* = 0;
    out_missing_ids.* = null;
    out_missing_count.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const ids = ids_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "ids_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const result = backend.fetchByIds(allocator, ids[0..count], include_values) catch |err| {
        capi_error.setError(err, "failed to fetch vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_found_count.* = result.ids.len;
    out_dims.* = result.dims;
    out_missing_count.* = result.missing_ids.len;
    out_ids.* = result.ids.ptr;
    out_vectors.* = result.vectors.ptr;
    out_missing_ids.* = result.missing_ids.ptr;
    return 0;
}

/// Free buffers returned by `talu_db_vector_fetch`.
pub export fn talu_db_vector_free_fetch(
    ids_ptr: ?[*]u64,
    vectors_ptr: ?[*]f32,
    found_count: usize,
    dims: u32,
    missing_ids_ptr: ?[*]u64,
    missing_count: usize,
) callconv(.c) void {
    capi_error.clearError();
    if (ids_ptr) |ptr| {
        allocator.free(ptr[0..found_count]);
    }
    if (vectors_ptr) |ptr| {
        const total = found_count * @as(usize, dims);
        allocator.free(ptr[0..total]);
    }
    if (missing_ids_ptr) |ptr| {
        allocator.free(ptr[0..missing_count]);
    }
}

/// Read vector mutation statistics.
pub export fn talu_db_vector_stats(
    handle: ?*VectorStoreHandle,
    out_visible_count: *usize,
    out_tombstone_count: *usize,
    out_segment_count: *usize,
    out_total_count: *usize,
    out_manifest_generation: *u64,
    out_index_ready_segments: *usize,
    out_index_pending_segments: *usize,
    out_index_failed_segments: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_visible_count.* = 0;
    out_tombstone_count.* = 0;
    out_segment_count.* = 0;
    out_total_count.* = 0;
    out_manifest_generation.* = 0;
    out_index_ready_segments.* = 0;
    out_index_pending_segments.* = 0;
    out_index_failed_segments.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const stats = backend.stats() catch |err| {
        capi_error.setError(err, "failed to read vector stats", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_visible_count.* = stats.visible_count;
    out_tombstone_count.* = stats.tombstone_count;
    out_segment_count.* = stats.segment_count;
    out_total_count.* = stats.total_count;
    out_manifest_generation.* = stats.manifest_generation;
    out_index_ready_segments.* = stats.index_ready_segments;
    out_index_pending_segments.* = stats.index_pending_segments;
    out_index_failed_segments.* = stats.index_failed_segments;
    return 0;
}

/// Compact vector storage from visible state.
pub export fn talu_db_vector_compact(
    handle: ?*VectorStoreHandle,
    dims: u32,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    return talu_db_vector_compact_idempotent(
        handle,
        dims,
        0,
        0,
        out_kept_count,
        out_removed_tombstones,
    );
}

/// Compact vectors with durable idempotency semantics.
pub export fn talu_db_vector_compact_idempotent(
    handle: ?*VectorStoreHandle,
    dims: u32,
    key_hash: u64,
    request_hash: u64,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_kept_count.* = 0;
    out_removed_tombstones.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = backend.compactIdempotent(dims, key_hash, request_hash) catch |err| {
        switch (err) {
            error.IdempotencyConflict => capi_error.setError(err, "idempotency conflict", .{}),
            else => capi_error.setError(err, "failed to compact vectors", .{}),
        }
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_kept_count.* = result.kept_count;
    out_removed_tombstones.* = result.removed_tombstones;
    return 0;
}

/// Compact vector storage if the current manifest generation matches expected_generation.
pub export fn talu_db_vector_compact_with_generation(
    handle: ?*VectorStoreHandle,
    dims: u32,
    expected_generation: u64,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_kept_count.* = 0;
    out_removed_tombstones.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = backend.compactWithExpectedGeneration(dims, expected_generation) catch |err| {
        switch (err) {
            error.ManifestGenerationConflict => capi_error.setErrorWithCode(
                .invalid_argument,
                "manifest generation conflict",
                .{},
            ),
            else => capi_error.setError(err, "failed to compact vectors with generation guard", .{}),
        }
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_kept_count.* = result.kept_count;
    out_removed_tombstones.* = result.removed_tombstones;
    return 0;
}

/// Compact vectors when tombstones older than TTL are present.
pub export fn talu_db_vector_compact_expired_tombstones(
    handle: ?*VectorStoreHandle,
    dims: u32,
    now_ms: i64,
    max_age_ms: i64,
    out_kept_count: *usize,
    out_removed_tombstones: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_kept_count.* = 0;
    out_removed_tombstones.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = backend.compactExpiredTombstones(dims, now_ms, max_age_ms) catch |err| {
        capi_error.setError(err, "failed to compact expired tombstones", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_kept_count.* = result.kept_count;
    out_removed_tombstones.* = result.removed_tombstones;
    return 0;
}

/// Build pending vector ANN indexes if the current manifest generation matches expected_generation.
pub export fn talu_db_vector_build_indexes_with_generation(
    handle: ?*VectorStoreHandle,
    expected_generation: u64,
    max_segments: usize,
    out_built_segments: *usize,
    out_failed_segments: *usize,
    out_pending_segments: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_built_segments.* = 0;
    out_failed_segments.* = 0;
    out_pending_segments.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = backend.buildPendingApproximateIndexesWithExpectedGeneration(expected_generation, max_segments) catch |err| {
        switch (err) {
            error.ManifestGenerationConflict => capi_error.setErrorWithCode(
                .invalid_argument,
                "manifest generation conflict",
                .{},
            ),
            else => capi_error.setError(err, "failed to build vector indexes with generation guard", .{}),
        }
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_built_segments.* = result.built_segments;
    out_failed_segments.* = result.failed_segments;
    out_pending_segments.* = result.pending_segments;
    return 0;
}

/// Read vector change events with cursor pagination.
pub export fn talu_db_vector_changes(
    handle: ?*VectorStoreHandle,
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
    capi_error.clearError();
    out_seqs.* = null;
    out_ops.* = null;
    out_ids.* = null;
    out_timestamps.* = null;
    out_count.* = 0;
    out_has_more.* = false;
    out_next_since.* = since;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    var result = backend.readChanges(allocator, since, limit) catch |err| {
        capi_error.setError(err, "failed to read vector changes", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer result.deinit(allocator);

    const seqs = allocator.alloc(u64, result.events.len) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate change seqs", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.free(seqs);
    const ops = allocator.alloc(u8, result.events.len) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate change ops", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.free(ops);
    const ids = allocator.alloc(u64, result.events.len) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate change ids", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.free(ids);
    const timestamps = allocator.alloc(i64, result.events.len) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate change timestamps", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.free(timestamps);

    for (result.events, 0..) |event, idx| {
        seqs[idx] = event.seq;
        ops[idx] = @intFromEnum(event.op);
        ids[idx] = event.id;
        timestamps[idx] = event.timestamp;
    }

    out_seqs.* = seqs.ptr;
    out_ops.* = ops.ptr;
    out_ids.* = ids.ptr;
    out_timestamps.* = timestamps.ptr;
    out_count.* = result.events.len;
    out_has_more.* = result.has_more;
    out_next_since.* = result.next_since;
    return 0;
}

/// Free buffers returned by `talu_db_vector_changes`.
pub export fn talu_db_vector_free_changes(
    seqs_ptr: ?[*]u64,
    ops_ptr: ?[*]u8,
    ids_ptr: ?[*]u64,
    timestamps_ptr: ?[*]i64,
    count: usize,
) callconv(.c) void {
    capi_error.clearError();
    if (seqs_ptr) |ptr| allocator.free(ptr[0..count]);
    if (ops_ptr) |ptr| allocator.free(ptr[0..count]);
    if (ids_ptr) |ptr| allocator.free(ptr[0..count]);
    if (timestamps_ptr) |ptr| allocator.free(ptr[0..count]);
}

/// Load all vectors from the store.
///
/// The caller owns the returned buffers and must free them with
/// `talu_db_vector_free_load`.
pub export fn talu_db_vector_load(
    handle: ?*VectorStoreHandle,
    out_ids: *?[*]u64,
    out_vectors: *?[*]f32,
    out_count: *usize,
    out_dims: *u32,
) callconv(.c) i32 {
    capi_error.clearError();
    out_ids.* = null;
    out_vectors.* = null;
    out_count.* = 0;
    out_dims.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const batch = backend.loadVectors(allocator) catch |err| {
        capi_error.setError(err, "failed to load vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_count.* = batch.ids.len;
    out_dims.* = batch.dims;
    out_ids.* = batch.ids.ptr;
    out_vectors.* = batch.vectors.ptr;
    return 0;
}

/// Free vector buffers returned by `talu_db_vector_load`.
pub export fn talu_db_vector_free_load(
    ids_ptr: ?[*]u64,
    vectors_ptr: ?[*]f32,
    count: usize,
    dims: u32,
) callconv(.c) void {
    capi_error.clearError();
    if (ids_ptr) |ptr| {
        allocator.free(ptr[0..count]);
    }
    if (vectors_ptr) |ptr| {
        const total = count * @as(usize, dims);
        allocator.free(ptr[0..total]);
    }
}

/// Search vectors using a dot-product scan.
/// Caller owns the returned buffers and must free them via
/// `talu_db_vector_free_search`.
pub export fn talu_db_vector_search(
    handle: ?*VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    k: u32,
    out_ids: *?[*]u64,
    out_scores: *?[*]f32,
    out_count: *usize,
) callconv(.c) i32 {
    capi_error.clearError();
    out_ids.* = null;
    out_scores.* = null;
    out_count.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const query = query_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "query_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const result = backend.search(allocator, query[0..query_len], k) catch |err| {
        capi_error.setError(err, "failed to search vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_ids.* = result.ids.ptr;
    out_scores.* = result.scores.ptr;
    out_count.* = result.ids.len;
    return 0;
}

/// Free buffers returned by `talu_db_vector_search`.
pub export fn talu_db_vector_free_search(
    ids_ptr: ?[*]u64,
    scores_ptr: ?[*]f32,
    count: usize,
) callconv(.c) void {
    capi_error.clearError();
    if (ids_ptr) |ptr| {
        allocator.free(ptr[0..count]);
    }
    if (scores_ptr) |ptr| {
        allocator.free(ptr[0..count]);
    }
}

/// Search vectors for multiple queries using a dot-product scan.
/// Caller owns the returned buffers and must free them via
/// `talu_db_vector_free_search_batch`.
pub export fn talu_db_vector_search_batch(
    handle: ?*VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    dims: u32,
    query_count: u32,
    k: u32,
    out_ids: *?[*]u64,
    out_scores: *?[*]f32,
    out_count_per_query: *u32,
) callconv(.c) i32 {
    return talu_db_vector_search_batch_ex(
        handle,
        query_ptr,
        query_len,
        dims,
        query_count,
        k,
        false,
        false,
        out_ids,
        out_scores,
        out_count_per_query,
    );
}

/// Search vectors for multiple queries using a dot-product scan with explicit options.
pub export fn talu_db_vector_search_batch_ex(
    handle: ?*VectorStoreHandle,
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
    capi_error.clearError();
    out_ids.* = null;
    out_scores.* = null;
    out_count_per_query.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const query = query_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "query_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (query_len != @as(usize, dims) * @as(usize, query_count)) {
        capi_error.setErrorWithCode(.invalid_argument, "query_len does not match dims * query_count", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const result = backend.searchBatchWithOptions(allocator, query[0..query_len], dims, query_count, k, .{
        .normalize_queries = normalize_queries,
        .approximate = approximate,
    }) catch |err| {
        capi_error.setError(err, "failed to search vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_ids.* = result.ids.ptr;
    out_scores.* = result.scores.ptr;
    out_count_per_query.* = result.count_per_query;
    return 0;
}

/// Free buffers returned by `talu_db_vector_search_batch`.
pub export fn talu_db_vector_free_search_batch(
    ids_ptr: ?[*]u64,
    scores_ptr: ?[*]f32,
    count_per_query: u32,
    query_count: u32,
) callconv(.c) void {
    capi_error.clearError();
    const total = @as(usize, count_per_query) * @as(usize, query_count);
    if (ids_ptr) |ptr| {
        allocator.free(ptr[0..total]);
    }
    if (scores_ptr) |ptr| {
        allocator.free(ptr[0..total]);
    }
}

/// Validate scan batch buffer sizes. Returns null on success, error code on failure.
fn validateScanBatchBuffers(ids_len: usize, scores_len: usize, total_rows: usize, query_count: u32) ?i32 {
    if (ids_len < total_rows) {
        capi_error.setErrorWithCode(.invalid_argument, "ids_len is too small", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    if (scores_len < total_rows * @as(usize, query_count)) {
        capi_error.setErrorWithCode(.invalid_argument, "scores_len is too small", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    return null;
}

/// Scan scores for multiple queries into caller-provided buffers.
pub export fn talu_db_vector_scan_batch(
    handle: ?*VectorStoreHandle,
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
    capi_error.clearError();
    out_total_rows.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const query = query_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "query_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (query_len != @as(usize, dims) * @as(usize, query_count)) {
        capi_error.setErrorWithCode(.invalid_argument, "query_len does not match dims * query_count", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const total_rows = backend.countEmbeddingRows(allocator, dims) catch |err| {
        capi_error.setError(err, "failed to count vector rows", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_total_rows.* = total_rows;

    if (out_ids == null or out_scores == null or ids_len == 0 or scores_len == 0) return 0;
    if (validateScanBatchBuffers(ids_len, scores_len, total_rows, query_count)) |code| return code;

    _ = backend.scanScoresBatchInto(allocator, query[0..query_len], dims, query_count, out_ids.?[0..ids_len], out_scores.?[0..scores_len]) catch |err| {
        capi_error.setError(err, "failed to scan vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

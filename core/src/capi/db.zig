//! C-API Storage Module - TaluDB storage backend and vector store.
//!
//! This module provides the C-compatible interface for:
//! - TaluDB persistence (`talu_chat_set_storage_db`)
//! - Vector store operations (`talu_vector_store_*`)
//! - Session listing and management (`talu_sessions_*`)

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const responses_mod = @import("responses.zig");
const responses_root = @import("../responses/root.zig");
const db = @import("../db/root.zig");
const tensor_mod = @import("../tensor.zig");
const log = @import("../log.zig");
const Conversation = responses_root.Conversation;

const allocator = std.heap.c_allocator;

/// Opaque handle for TaluDB vector backend.
/// Thread safety: NOT thread-safe (single-writer semantics).
pub const VectorStoreHandle = opaque {};

// =============================================================================
// C-ABI Storage Record
// =============================================================================

/// Item type discriminator for storage indexing.
/// Values match items.ItemType for direct casting.
pub const CItemType = enum(u8) {
    message = 0,
    function_call = 1,
    function_call_output = 2,
    reasoning = 3,
    item_reference = 4,
    unknown = 255,
};

/// Message role for storage indexing.
/// Values match items.MessageRole for direct casting.
/// 255 = N/A (for non-message items).
pub const CMessageRole = enum(u8) {
    system = 0,
    user = 1,
    assistant = 2,
    developer = 3,
    unknown_role = 254, // Forward compatibility
    not_applicable = 255, // Non-message items
};

/// C-ABI compatible storage record for external backends.
///
/// This struct is the "canonical" row format for storage. It contains:
///   - **Header fields**: Indexed columns for SQL WHERE clauses
///   - **Payload fields**: Full JSON for schema fidelity
///
/// Memory: All string pointers are borrowed from Zig buffers.
/// They are valid ONLY during the callback invocation.
pub const CStorageRecord = extern struct {
    /// Monotonic item identity (authoritative, never changes).
    /// Use as PRIMARY KEY in database.
    item_id: u64,

    /// Session identifier (null-terminated).
    /// Use as partition key for multi-tenant storage.
    /// May be null if no session_id was set.
    session_id: ?[*:0]const u8,

    /// Item type discriminator (for SQL filtering).
    /// Cast to CItemType enum for type-safe access.
    item_type: u8,

    /// Message role (for SQL filtering).
    /// 255 = not applicable (non-message items like function_call).
    /// Cast to CMessageRole enum for type-safe access.
    role: u8,

    /// Item status (ItemStatus enum as u8).
    status: u8,

    /// UI visibility flag.
    hidden: bool,

    /// Retention flag.
    pinned: bool,

    /// Structured output validation: JSON parsed successfully.
    json_valid: bool,

    /// Structured output validation: schema validation passed.
    schema_valid: bool,

    /// Structured output validation: output was repaired.
    repaired: bool,

    /// Parent item linkage (optional).
    parent_item_id: u64,

    /// Whether parent_item_id is present.
    has_parent: bool,

    /// Origin item linkage for forks.
    origin_item_id: u64,

    /// Whether origin_item_id/session_id is present.
    has_origin: bool,

    /// Origin session identifier (null-terminated, may be null).
    origin_session_id: ?[*:0]const u8,

    /// Finish reason (null-terminated, may be null).
    finish_reason: ?[*:0]const u8,

    /// Prefill time in nanoseconds.
    prefill_ns: u64,

    /// Generation time in nanoseconds.
    generation_ns: u64,

    /// Input token count (prompt tokens).
    input_tokens: u32,

    /// Output token count (completion tokens).
    output_tokens: u32,

    /// Expiration timestamp for retention (Unix ms). 0 = no expiry.
    ttl_ts: i64,

    /// Creation timestamp (Unix milliseconds).
    /// Set by Zig when item is finalized.
    created_at_ms: i64,

    /// Full Open Responses JSON (null-terminated).
    /// Serialized by Zig using SerializationDirection.response.
    /// Contains the complete, schema-compliant item object.
    /// Store in TEXT/JSONB column for full fidelity.
    content_json: ?[*:0]const u8,

    /// Developer metadata JSON (null-terminated).
    /// May be null if no metadata was set.
    /// Store in separate column for custom indexing.
    metadata_json: ?[*:0]const u8,

    /// Reserved for future expansion.
    _reserved: [13]u8 = [_]u8{0} ** 13,
};

// =============================================================================
// C-ABI Storage Events
// =============================================================================

/// Storage event type discriminator.
pub const CStorageEventType = enum(u8) {
    /// One or more items were finalized and should be stored.
    /// Use `items` pointer and `items_count` to access the batch.
    /// This enables transaction batching for parallel tool calls.
    put_items = 0,

    /// An item was deleted (soft-delete in storage).
    delete_item = 1,

    /// All items were cleared.
    clear_items = 2,

    /// Session metadata was set/updated.
    put_session = 3,

    /// Begin a fork transaction boundary.
    begin_fork = 4,

    /// End a fork transaction boundary.
    end_fork = 5,
};

/// C-ABI session record for external backends.
///
/// Contains session-level metadata that should be stored separately
/// from individual items. This ensures Zig is the single source of
/// truth for all session data (config, title, etc.).
pub const CSessionRecord = extern struct {
    /// Session identifier (null-terminated).
    session_id: ?[*:0]const u8,

    /// Model identifier (null-terminated, may be null).
    model: ?[*:0]const u8,

    /// Human-readable title (null-terminated, may be null).
    title: ?[*:0]const u8,

    /// System prompt (null-terminated, may be null).
    system_prompt: ?[*:0]const u8,

    /// GenerationConfig as JSON (null-terminated).
    /// Contains temperature, max_tokens, etc.
    config_json: ?[*:0]const u8,

    /// Session marker (null-terminated, may be null). E.g. "pinned", "archived", "deleted".
    marker: ?[*:0]const u8,

    /// Parent session identifier (null-terminated, may be null).
    parent_session_id: ?[*:0]const u8,

    /// Group identifier for multi-tenant session listing (null-terminated, may be null).
    group_id: ?[*:0]const u8,

    /// Latest item_id in the session (0 when no items yet).
    head_item_id: u64,

    /// Expiration timestamp for retention (Unix ms). 0 = no expiry.
    ttl_ts: i64,

    /// Session metadata as JSON (null-terminated, may be null).
    metadata_json: ?[*:0]const u8,

    /// Space-separated normalized tags extracted from metadata_json.tags.
    /// Pre-computed at write time for efficient search. Null for legacy sessions.
    tags_text: ?[*:0]const u8,

    /// Search snippet: text fragment around the matched search query in item content.
    /// Only populated when listing with a search_query that matched item content.
    /// Null when no search query or when matched by metadata only.
    search_snippet: ?[*:0]const u8,

    /// Source document ID for lineage tracking (null-terminated, may be null).
    /// Links this session to the prompt/persona document that spawned it.
    source_doc_id: ?[*:0]const u8,

    /// Creation timestamp (Unix milliseconds).
    created_at_ms: i64,

    /// Last update timestamp (Unix milliseconds).
    updated_at_ms: i64,
};

/// Session list container for C API.
/// Owns the memory for all contained records.
/// Thread safety: NOT thread-safe (caller must synchronize).
pub const CSessionList = extern struct {
    /// Array of session records.
    sessions: ?[*]CSessionRecord,

    /// Number of sessions in the array.
    count: usize,

    /// Internal: allocator used for cleanup.
    _allocator: ?*anyopaque,

    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

/// C-ABI storage event for external backends.
///
/// Exactly one of the payload fields is valid, based on event_type:
///   - put_items: `items` + `items_count` contain the batch to store
///   - delete_item: `deleted_item_id` and `deleted_at_ms` are valid
///   - clear_items: `cleared_at_ms` and `keep_context` are valid
///   - put_session: `session` contains the session metadata to store
///
/// The `put_items` event supports batching for efficient transaction handling.
/// Even single-item events use this interface (with items_count=1) to provide
/// a consistent API that enables future batching of parallel tool calls.
pub const CStorageEvent = extern struct {
    /// Event type discriminator.
    event_type: CStorageEventType,

    /// Padding for alignment.
    _pad: [7]u8 = [_]u8{0} ** 7,

    /// For put_items: pointer to array of records to store.
    /// Valid only when event_type == .put_items.
    items: ?[*]const CStorageRecord,

    /// For put_items: number of records in the items array.
    items_count: usize,

    /// For put_session: the session metadata to store.
    session: CSessionRecord,

    /// For delete_item: the item_id to delete.
    deleted_item_id: u64,

    /// For delete_item: deletion timestamp (Unix ms).
    deleted_at_ms: i64,

    /// For clear_items: clear timestamp (Unix ms).
    cleared_at_ms: i64,

    /// For clear_items: whether to keep context messages.
    /// If true, preserve system/developer messages.
    keep_context: bool,

    /// For begin_fork/end_fork: fork transaction identifier.
    fork_id: u64,

    /// For begin_fork/end_fork: session identifier (null-terminated).
    fork_session_id: ?[*:0]const u8,

    /// Reserved for future expansion.
    _reserved: [7]u8 = [_]u8{0} ** 7,
};

// =============================================================================
// TaluDB Backend Implementation
// =============================================================================

const responses = @import("../responses/root.zig");
const StorageBackend = responses.StorageBackend;
const StorageEvent = responses.StorageEvent;

/// Wrapper backend that owns a TaluDB chat backend for the C API.
pub const DbBackendWrapper = struct {
    backend: *db.table.sessions.TableAdapter,
    allocator: std.mem.Allocator,

    pub fn toStorageBackend(self: *DbBackendWrapper) StorageBackend {
        return StorageBackend{
            .ptr = self,
            .vtable = &vtable,
        };
    }

    const vtable = StorageBackend.VTable{
        .onEvent = onEventImpl,
        .loadAll = loadAllImpl,
        .deinit = deinitImpl,
    };

    fn onEventImpl(ptr: *anyopaque, event: *const StorageEvent) anyerror!void {
        const self: *DbBackendWrapper = @ptrCast(@alignCast(ptr));
        const storage = self.backend.backend();
        return storage.onEvent(event);
    }

    fn loadAllImpl(ptr: *anyopaque, alloc: std.mem.Allocator) anyerror![]responses.ItemRecord {
        const self: *DbBackendWrapper = @ptrCast(@alignCast(ptr));
        const storage = self.backend.backend();
        return storage.loadAll(alloc);
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *DbBackendWrapper = @ptrCast(@alignCast(ptr));
        const storage = self.backend.backend();
        storage.deinit();
        self.allocator.destroy(self.backend);
        self.allocator.destroy(self);
    }
};

// =============================================================================
// C-API Functions
// =============================================================================

const ChatHandle = responses_mod.ChatHandle;
const Chat = @import("../responses/root.zig").Chat;

/// Clear existing storage backend.
fn clearStorageBackend(storage_backend: *?responses_root.StorageBackend) void {
    if (storage_backend.*) |*sb| {
        sb.deinit();
        storage_backend.* = null;
    }
}

/// Create TaluDB backend and wrapper. Returns error code on failure (sets capi_error).
fn createDbBackend(db_path_slice: []const u8, session_slice: []const u8) ?*DbBackendWrapper {
    const backend_ptr = allocator.create(db.table.sessions.TableAdapter) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate TableAdapter", .{});
        return null;
    };

    backend_ptr.* = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_slice) catch |err| {
        allocator.destroy(backend_ptr);
        capi_error.setError(err, "failed to initialize TaluDB backend", .{});
        return null;
    };

    const wrapper = allocator.create(DbBackendWrapper) catch {
        const storage = backend_ptr.backend();
        storage.deinit();
        allocator.destroy(backend_ptr);
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate TaluDB wrapper", .{});
        return null;
    };

    wrapper.* = .{ .backend = backend_ptr, .allocator = allocator };
    return wrapper;
}

/// Set TaluDB storage backend on a Chat.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object
///   - db_path: Root path for TaluDB storage (null-terminated)
///   - session_id: Session identifier (null-terminated)
///
/// On success, loads any persisted items from TaluDB into the conversation.
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_chat_set_storage_db(
    chat_handle: ?*ChatHandle,
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const session_slice = std.mem.sliceTo(session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (db_path_slice.len == 0 or session_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path or session_id is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }
    clearStorageBackend(&chat.conv.storage_backend);
    const wrapper = createDbBackend(db_path_slice, session_slice) orelse {
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    chat.conv.storage_backend = wrapper.toStorageBackend();
    if (chat.conv.session_id) |old_sid| wrapper.allocator.free(old_sid);
    chat.conv.session_id = wrapper.allocator.dupe(u8, session_slice) catch {
        clearStorageBackend(&chat.conv.storage_backend);
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    chat.conv.loadFromStorageBackend() catch |err| {
        clearStorageBackend(&chat.conv.storage_backend);
        capi_error.setError(err, "failed to load TaluDB records", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Initialize a vector store in the given database root.
///
/// The caller owns the returned handle and must free it with
/// `talu_vector_store_free`.
pub export fn talu_vector_store_init(
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
pub export fn talu_vector_store_free(handle: ?*VectorStoreHandle) callconv(.c) void {
    capi_error.clearError();
    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse return));
    db.vector.store.destroy(allocator, backend);
}

/// Simulate a process crash for a vector store (testing only).
///
/// Releases all file descriptors and locks WITHOUT flushing or deleting
/// the WAL file. After this call, the handle is invalid — call
/// `talu_vector_store_free` would double-free; instead, just discard
/// the handle pointer.
pub export fn talu_vector_store_simulate_crash(handle: ?*VectorStoreHandle) callconv(.c) void {
    capi_error.clearError();
    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse return));
    backend.simulateCrash();
    allocator.destroy(backend);
}

/// Append a batch of vectors to the store.
pub export fn talu_vector_store_append(
    handle: ?*VectorStoreHandle,
    ids_ptr: ?[*]const u64,
    vectors_ptr: ?[*]const f32,
    count: usize,
    dims: u32,
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
    backend.appendBatch(ids[0..count], vectors[0..total], dims) catch |err| {
        capi_error.setError(err, "failed to append vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Load all vectors from the store.
///
/// The caller owns the returned buffers and must free them with
/// `talu_vector_store_free_load`.
pub export fn talu_vector_store_load(
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

/// Load all vectors and return them as a Tensor handle for DLPack export.
///
/// Caller owns the returned ids and tensor. Use `talu_vector_store_free_load`
/// to free ids, and `talu_tensor_free` (via ops bindings) to free the tensor.
pub export fn talu_vector_store_load_tensor(
    handle: ?*VectorStoreHandle,
    out_ids: ?*?[*]u64,
    out_tensor: ?*?*tensor_mod.Tensor,
    out_count: ?*usize,
    out_dims: ?*u32,
) callconv(.c) i32 {
    capi_error.clearError();
    const ids_out = out_ids orelse return setArgError("out_ids is null");
    const tensor_out = out_tensor orelse return setArgError("out_tensor is null");
    const count_out = out_count orelse return setArgError("out_count is null");
    const dims_out = out_dims orelse return setArgError("out_dims is null");
    ids_out.* = null;
    tensor_out.* = null;
    count_out.* = 0;
    dims_out.* = 0;

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const batch = backend.loadVectorsTensor(allocator) catch |err| {
        capi_error.setError(err, "failed to load vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    count_out.* = batch.ids.len;
    dims_out.* = batch.dims;
    ids_out.* = batch.ids.ptr;
    tensor_out.* = batch.tensor;
    return 0;
}

/// Search vectors using a dot-product scan.
/// Caller owns the returned buffers and must free them via
/// `talu_vector_store_free_search`.
pub export fn talu_vector_store_search(
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

/// Search vectors for multiple queries using a dot-product scan.
/// Caller owns the returned buffers and must free them via
/// `talu_vector_store_free_search_batch`.
pub export fn talu_vector_store_search_batch(
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

    const result = backend.searchBatch(allocator, query[0..query_len], dims, query_count, k) catch |err| {
        capi_error.setError(err, "failed to search vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out_ids.* = result.ids.ptr;
    out_scores.* = result.scores.ptr;
    out_count_per_query.* = result.count_per_query;
    return 0;
}

/// Free buffers returned by `talu_vector_store_search_batch`.
pub export fn talu_vector_store_free_search_batch(
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

/// Stream scores for all vectors to a caller-provided callback.
pub export fn talu_vector_store_scan(
    handle: ?*VectorStoreHandle,
    query_ptr: ?[*]const f32,
    query_len: usize,
    ctx: ?*anyopaque,
    callback: db.vector.store.ScoreCallbackC,
) callconv(.c) i32 {
    capi_error.clearError();

    const backend: *db.vector.store.VectorAdapter = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const query = query_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "query_ptr is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    backend.searchScoresC(allocator, query[0..query_len], ctx, callback) catch |err| {
        capi_error.setError(err, "failed to scan vectors", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
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
pub export fn talu_vector_store_scan_batch(
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

/// Free buffers returned by `talu_vector_store_search`.
pub export fn talu_vector_store_free_search(
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

/// Free vector buffers returned by `talu_vector_store_load`.
pub export fn talu_vector_store_free_load(
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

/// Emit a PutSession event to the storage callback.
///
/// Call this when session metadata changes (config, title, system_prompt).
/// The storage backend will receive a PutSession event with the provided
/// metadata, enabling external persistence of session data.
///
/// This ensures Zig is the single source of truth for all session data.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object
///   - model: Model identifier (or null)
///   - title: Human-readable session title (or null)
///   - system_prompt: System prompt text (or null)
///   - config_json: GenerationConfig as JSON (or null)
///   - marker: Session marker (or null). E.g. "pinned", "archived", "deleted"
///   - parent_session_id: Parent session identifier (or null)
///   - group_id: Group identifier for multi-tenant listing (or null)
///   - metadata_json: Session metadata as JSON (or null)
///   - source_doc_id: Source document ID for lineage tracking (or null)
///
/// Returns: 0 on success, negative error code on failure.
// lint:ignore capi-callconv - callconv(.c) on closing line
pub export fn talu_chat_notify_session_update(
    chat_handle: ?*ChatHandle,
    model: ?[*:0]const u8,
    title: ?[*:0]const u8,
    system_prompt: ?[*:0]const u8,
    config_json: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    parent_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    // Convert null-terminated strings to slices
    const model_slice: ?[]const u8 = if (model) |m| std.mem.span(m) else null;
    const title_slice: ?[]const u8 = if (title) |t| std.mem.span(t) else null;
    const system_slice: ?[]const u8 = if (system_prompt) |s| std.mem.span(s) else null;
    const config_slice: ?[]const u8 = if (config_json) |c| std.mem.span(c) else null;
    const marker_slice: ?[]const u8 = if (marker) |s| std.mem.span(s) else null;
    const parent_session_id_slice: ?[]const u8 = if (parent_session_id) |p| std.mem.span(p) else null;
    const group_id_slice: ?[]const u8 = if (group_id) |g| std.mem.span(g) else null;
    const metadata_json_slice: ?[]const u8 = if (metadata_json) |m| std.mem.span(m) else null;
    const source_doc_id_slice: ?[]const u8 = if (source_doc_id) |s| std.mem.span(s) else null;

    // Call the Conversation method to emit the event
    chat.conv.notifySessionUpdate(
        model_slice,
        title_slice,
        system_slice,
        config_slice,
        marker_slice,
        parent_session_id_slice,
        group_id_slice,
        metadata_json_slice,
        source_doc_id_slice,
    );

    return 0;
}

// =============================================================================
// Storage Management C API Functions
// =============================================================================

/// Internal wrapper for session list data.
const SessionListData = struct {
    list: *CSessionList,
    sessions_buf: []CSessionRecord,
    arena: *std.heap.ArenaAllocator,
};

// =============================================================================
// Internal helpers for capi glue (validation, conversions)
// =============================================================================

/// Validate and convert a C db_path to a slice. Sets error on failure.
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

/// Convert optional C string to slice (null-safe).
fn optSlice(s: ?[*:0]const u8) ?[]const u8 {
    return if (s) |p| std.mem.span(p) else null;
}

/// Build ScanParams from C arguments.
fn buildSessionScanParams(
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) db.table.sessions.TableAdapter.ScanParams {
    return db.table.sessions.TableAdapter.ScanParams.fromArgs(
        limit,
        before_updated_at_ms,
        optSlice(before_session_id),
        optSlice(group_id),
    );
}

/// Shared implementation for listing sessions with parameters.
fn listSessionsImpl(
    db_path_slice: []const u8,
    params: db.table.sessions.TableAdapter.ScanParams,
    out_sessions: *?*CSessionList,
) i32 {
    const records = db.table.sessions.listSessions(allocator, db_path_slice, params) catch |err| {
        capi_error.setError(err, "failed to scan sessions", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.sessions.freeScannedSessionRecords(allocator, records);

    const list = buildSessionList(records) catch |err| {
        capi_error.setError(err, "failed to build session list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_sessions.* = list;
    return 0;
}

/// Build a CSessionList from scanned session records.
/// Caller owns the returned list and must free it with talu_storage_free_sessions.
fn buildSessionList(records: []db.table.sessions.ScannedSessionRecord) !*CSessionList {
    const list = allocator.create(CSessionList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const sessions_buf = arena.alloc(CSessionRecord, records.len) catch return error.OutOfMemory;

    for (records, 0..) |record, i| {
        sessions_buf[i] = std.mem.zeroes(CSessionRecord);
        sessions_buf[i].session_id = if (record.session_id.len > 0)
            (arena.dupeZ(u8, record.session_id) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].model = if (record.model) |m|
            (arena.dupeZ(u8, m) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].title = if (record.title) |t|
            (arena.dupeZ(u8, t) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].system_prompt = if (record.system_prompt) |s|
            (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].config_json = if (record.config_json) |c|
            (arena.dupeZ(u8, c) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].marker = if (record.marker) |s|
            (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].parent_session_id = if (record.parent_session_id) |p|
            (arena.dupeZ(u8, p) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].group_id = if (record.group_id) |g|
            (arena.dupeZ(u8, g) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].head_item_id = record.head_item_id;
        sessions_buf[i].ttl_ts = record.ttl_ts;
        sessions_buf[i].metadata_json = if (record.metadata_json) |m|
            (arena.dupeZ(u8, m) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].tags_text = if (record.tags_text) |t|
            (arena.dupeZ(u8, t) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].search_snippet = if (record.search_snippet) |s|
            (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].source_doc_id = if (record.source_doc_id) |d|
            (arena.dupeZ(u8, d) catch return error.OutOfMemory).ptr
        else
            null;
        sessions_buf[i].created_at_ms = record.created_at_ms;
        sessions_buf[i].updated_at_ms = record.updated_at_ms;
    }

    list.* = .{
        .sessions = if (records.len > 0) sessions_buf.ptr else null,
        .count = records.len,
        ._allocator = @ptrCast(allocator.ptr),
        ._arena = @ptrCast(arena_ptr),
    };

    log.debug("capi", "buildSessionList", .{
        .list_ptr = @intFromPtr(list),
        .count = records.len,
        .arena_ptr = @intFromPtr(arena_ptr),
    }, @src());

    return list;
}

// =============================================================================
// Tag Internal Helpers
// =============================================================================

/// Validate a required C string parameter.
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

/// Set error and return invalid_argument code (for compact null validation).
fn setArgError(comptime msg: []const u8) i32 {
    capi_error.setErrorWithCode(.invalid_argument, msg, .{});
    return @intFromEnum(error_codes.ErrorCode.invalid_argument);
}

/// Shared implementation for listing tags.
fn listTagsImpl(db_path_slice: []const u8, group_id: ?[]const u8, out_tags: *?*CTagList) i32 {
    const records = db.table.tags.listTags(allocator, db_path_slice, group_id) catch |err| {
        capi_error.setError(err, "failed to scan tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.tags.freeTagRecords(allocator, records);

    const list = buildTagList(records) catch |err| {
        capi_error.setError(err, "failed to build tag list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_tags.* = list;
    return 0;
}

/// Shared implementation for getting a tag by ID.
fn getTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8, out_tag: *CTagRecord) i32 {
    var record = db.table.tags.getTag(allocator, db_path_slice, tag_id_slice) catch |err| {
        capi_error.setError(err, "tag not found", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    } orelse {
        capi_error.setErrorWithCode(.tag_not_found, "tag not found", .{});
        return @intFromEnum(error_codes.ErrorCode.tag_not_found);
    };
    defer record.deinit(allocator);
    populateTagRecord(out_tag, record);
    return 0;
}

/// Shared implementation for getting a tag by name.
fn getTagByNameImpl(db_path_slice: []const u8, name_slice: []const u8, group_id: ?[]const u8, out_tag: *CTagRecord) i32 {
    var record = db.table.tags.getTagByName(allocator, db_path_slice, name_slice, group_id) catch |err| {
        capi_error.setError(err, "tag not found", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    } orelse {
        capi_error.setErrorWithCode(.tag_not_found, "tag not found", .{});
        return @intFromEnum(error_codes.ErrorCode.tag_not_found);
    };
    defer record.deinit(allocator);
    populateTagRecord(out_tag, record);
    return 0;
}

/// Shared implementation for creating a tag.
fn createTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8, name_slice: []const u8, color: ?[]const u8, description: ?[]const u8, group_id: ?[]const u8) i32 {
    const now_ms = std.time.milliTimestamp();
    const record = db.table.tags.TagRecord{
        .tag_id = tag_id_slice,
        .name = name_slice,
        .color = color,
        .description = description,
        .group_id = group_id,
        .created_at_ms = now_ms,
        .updated_at_ms = now_ms,
    };
    db.table.tags.createTag(allocator, db_path_slice, record) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to create tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Shared implementation for updating a tag.
fn updateTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8, name: ?[]const u8, color: ?[]const u8, description: ?[]const u8) i32 {
    db.table.tags.updateTag(allocator, db_path_slice, tag_id_slice, name, color, description) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to update tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Shared implementation for deleting a tag.
fn deleteTagImpl(db_path_slice: []const u8, tag_id_slice: []const u8) i32 {
    db.table.tags.deleteTagAndAssociations(allocator, db_path_slice, tag_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to delete tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Shared implementation for adding a conversation tag.
fn addConversationTagImpl(db_path_slice: []const u8, session_id_slice: []const u8, tag_id_slice: []const u8) i32 {
    db.table.tags.addConversationTag(allocator, db_path_slice, session_id_slice, tag_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to add conversation tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Shared implementation for removing a conversation tag.
fn removeConversationTagImpl(db_path_slice: []const u8, session_id_slice: []const u8, tag_id_slice: []const u8) i32 {
    db.table.tags.removeConversationTag(allocator, db_path_slice, session_id_slice, tag_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to remove conversation tag", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Shared implementation for getting conversation tag IDs.
fn getConversationTagsImpl(db_path_slice: []const u8, session_id_slice: []const u8, out_tag_ids: *?*CStringList) i32 {
    const tag_ids = db.table.tags.getConversationTagIds(allocator, db_path_slice, session_id_slice) catch |err| {
        capi_error.setError(err, "failed to get conversation tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer {
        for (tag_ids) |id| allocator.free(id);
        allocator.free(tag_ids);
    }
    const list = buildStringList(tag_ids) catch |err| {
        capi_error.setError(err, "failed to build string list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_tag_ids.* = list;
    return 0;
}

/// Shared implementation for getting tag conversation IDs.
fn getTagConversationsImpl(db_path_slice: []const u8, tag_id_slice: []const u8, out_session_ids: *?*CStringList) i32 {
    const session_ids = db.table.tags.getTagConversationIds(allocator, db_path_slice, tag_id_slice) catch |err| {
        capi_error.setError(err, "failed to get tag conversations", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer {
        for (session_ids) |id| allocator.free(id);
        allocator.free(session_ids);
    }
    const list = buildStringList(session_ids) catch |err| {
        capi_error.setError(err, "failed to build string list", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out_session_ids.* = list;
    return 0;
}

/// List sessions in a TaluDB directory with cursor pagination and filtering.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - limit: Maximum number of results (0 = no limit)
///   - before_updated_at_ms: Cursor timestamp component (0 = no cursor)
///   - before_session_id: Cursor session ID component (null = no cursor;
///     Zig hashes internally — callers never compute hashes)
///   - group_id: Filter by group (null = no filter; uses scalar prune
///     + exact string verification for hash collision safety)
///   - search_query: Case-insensitive substring filter on session metadata
///     and item content (null = no filter)
///   - tags_filter: Space-separated tags for AND matching (null = no filter).
///     Matches sessions whose tags_text contains ALL specified tags.
///   - tags_filter_any: Space-separated tags for OR matching (null = no filter).
///     Matches sessions whose tags_text contains ANY of the specified tags.
///     Mutually exclusive with tags_filter (tags_filter takes precedence).
///   - out_sessions: Output parameter to receive session list handle
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_storage_free_sessions().
pub export fn talu_storage_list_sessions(
    db_path: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var params = buildSessionScanParams(limit, before_updated_at_ms, before_session_id, group_id);
    params.search_query = optSlice(search_query);
    params.tags_filter = optSlice(tags_filter);
    params.tags_filter_any = optSlice(tags_filter_any);
    if (params.search_query != null) params.max_scan = 5000;

    return listSessionsImpl(db_path_slice, params, out);
}

/// Extended session listing with all filter options.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - limit: Maximum number of results (0 = no limit)
///   - before_updated_at_ms: Cursor timestamp component (0 = no cursor)
///   - before_session_id: Cursor session ID component (null = no cursor)
///   - group_id: Filter by group (null = no filter)
///   - search_query: Case-insensitive substring filter (null = no filter)
///   - tags_filter: Space-separated tags for AND matching (null = no filter)
///   - tags_filter_any: Space-separated tags for OR matching (null = no filter)
///   - marker_filter: Exact marker match (null = no filter)
///   - marker_filter_any: Space-separated markers for OR matching (null = no filter)
///   - model_filter: Model filter with wildcard support (null = no filter)
///   - created_after_ms: Created after timestamp inclusive (0 = no filter)
///   - created_before_ms: Created before timestamp exclusive (0 = no filter)
///   - updated_after_ms: Updated after timestamp inclusive (0 = no filter)
///   - updated_before_ms: Updated before timestamp exclusive (0 = no filter)
///   - has_tags: 0 = no filter, 1 = must have tags, -1 = must not have tags
///   - source_doc_id: Filter by source document ID (null = no filter)
///   - out_sessions: Output parameter to receive session list handle
///
/// Returns: 0 on success, negative error code on failure.
// lint:ignore capi-callconv - callconv(.c) on closing line
pub export fn talu_storage_list_sessions_ex(
    db_path: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    search_query: ?[*:0]const u8,
    tags_filter: ?[*:0]const u8,
    tags_filter_any: ?[*:0]const u8,
    marker_filter: ?[*:0]const u8,
    marker_filter_any: ?[*:0]const u8,
    model_filter: ?[*:0]const u8,
    created_after_ms: i64,
    created_before_ms: i64,
    updated_after_ms: i64,
    updated_before_ms: i64,
    has_tags: i32,
    source_doc_id: ?[*:0]const u8,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    var params = buildSessionScanParams(limit, before_updated_at_ms, before_session_id, group_id);
    params.search_query = optSlice(search_query);
    params.tags_filter = optSlice(tags_filter);
    params.tags_filter_any = optSlice(tags_filter_any);
    params.marker_filter = optSlice(marker_filter);
    params.marker_filter_any = optSlice(marker_filter_any);
    params.model_filter = optSlice(model_filter);
    params.created_after_ms = if (created_after_ms != 0) created_after_ms else null;
    params.created_before_ms = if (created_before_ms != 0) created_before_ms else null;
    params.updated_after_ms = if (updated_after_ms != 0) updated_after_ms else null;
    params.updated_before_ms = if (updated_before_ms != 0) updated_before_ms else null;
    params.has_tags = if (has_tags == 1) true else if (has_tags == 0) false else null;
    params.source_doc_id = optSlice(source_doc_id);
    if (params.search_query != null) params.max_scan = 5000;

    return listSessionsImpl(db_path_slice, params, out);
}

/// List sessions filtered by source document (lineage query).
///
/// A convenience function for finding all conversations spawned from a specific
/// prompt document. Wraps talu_storage_list_sessions_ex with only the source_doc_id filter.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - source_doc_id: Document ID to filter by (null-terminated)
///   - limit: Maximum number of sessions to return
///   - before_updated_at_ms: Cursor timestamp for pagination (0 = start from newest)
///   - before_session_id: Cursor session_id for pagination (null = start from newest)
///   - out_sessions: Output parameter to receive session list handle
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_storage_list_sessions_by_source(
    db_path: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
    limit: u32,
    before_updated_at_ms: i64,
    before_session_id: ?[*:0]const u8,
    out_sessions: ?*?*CSessionList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_sessions orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_sessions is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const source_doc_slice = validateRequiredArg(source_doc_id, "source_doc_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);

    var params = buildSessionScanParams(limit, before_updated_at_ms, before_session_id, null);
    params.source_doc_id = source_doc_slice;

    return listSessionsImpl(db_path_slice, params, out);
}

/// Get detailed metadata for a specific session.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Session identifier to query (null-terminated)
///   - out_session: Output parameter to receive session record
///
/// Returns: 0 on success, storage_error if session not found or deleted.
///
/// Note: String fields in the output CSessionRecord use borrowed pointers
/// stored in a thread-local buffer. They are valid only until the next
/// talu_storage_get_session_info call. Bindings must copy them immediately.
pub export fn talu_storage_get_session_info(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    out_session: *CSessionRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_session.* = std.mem.zeroes(CSessionRecord);

    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const session_id_slice = std.mem.sliceTo(session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (db_path_slice.len == 0 or session_id_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path or session_id is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    var adapter = db.table.sessions.TableAdapter.initReadOnly(allocator, db_path_slice) catch |err| {
        capi_error.setError(err, "failed to open database for reading", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.deinitReadOnly();

    const record = adapter.lookupSession(allocator, session_id_slice) catch |err| {
        capi_error.setError(err, "session not found", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.sessions.freeScannedSessionRecord(allocator, @constCast(&record));

    populateSessionRecord(out_session, record);
    return 0;
}

/// Copy string to static buffer with null termination. Returns null if empty.
fn copyToStaticBuf(buf: []u8, src: []const u8) ?[*:0]const u8 {
    if (src.len == 0) return null;
    const len = @min(src.len, buf.len - 1);
    @memcpy(buf[0..len], src[0..len]);
    buf[len] = 0;
    return @ptrCast(buf.ptr);
}

/// Copy optional string to static buffer with null termination.
fn copyToStaticBufOpt(buf: []u8, src: ?[]const u8) ?[*:0]const u8 {
    const s = src orelse return null;
    return copyToStaticBuf(buf, s);
}

/// Copy ScannedSessionRecord fields to CSessionRecord using thread-local static buffers.
/// Output pointers are valid only until the next call on the same thread.
fn populateSessionRecord(out: *CSessionRecord, record: db.table.sessions.ScannedSessionRecord) void {
    // Thread-local buffers: overwritten on each call, never read before write.
    const S = struct {
        threadlocal var session_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var model_buf: [256]u8 = .{0} ** 256;
        threadlocal var title_buf: [1024]u8 = .{0} ** 1024;
        threadlocal var system_prompt_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var config_json_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var marker_buf: [64]u8 = .{0} ** 64;
        threadlocal var parent_session_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var group_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var metadata_json_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var tags_text_buf: [1024]u8 = .{0} ** 1024;
        threadlocal var search_snippet_buf: [4096]u8 = .{0} ** 4096;
        threadlocal var source_doc_id_buf: [256]u8 = .{0} ** 256;
    };
    out.session_id = copyToStaticBuf(&S.session_id_buf, record.session_id);
    out.model = copyToStaticBufOpt(&S.model_buf, record.model);
    out.title = copyToStaticBufOpt(&S.title_buf, record.title);
    out.system_prompt = copyToStaticBufOpt(&S.system_prompt_buf, record.system_prompt);
    out.config_json = copyToStaticBufOpt(&S.config_json_buf, record.config_json);
    out.marker = copyToStaticBufOpt(&S.marker_buf, record.marker);
    out.parent_session_id = copyToStaticBufOpt(&S.parent_session_id_buf, record.parent_session_id);
    out.group_id = copyToStaticBufOpt(&S.group_id_buf, record.group_id);
    out.head_item_id = record.head_item_id;
    out.ttl_ts = record.ttl_ts;
    out.metadata_json = copyToStaticBufOpt(&S.metadata_json_buf, record.metadata_json);
    out.tags_text = copyToStaticBufOpt(&S.tags_text_buf, record.tags_text);
    out.search_snippet = copyToStaticBufOpt(&S.search_snippet_buf, record.search_snippet);
    out.source_doc_id = copyToStaticBufOpt(&S.source_doc_id_buf, record.source_doc_id);
    out.created_at_ms = record.created_at_ms;
    out.updated_at_ms = record.updated_at_ms;
}

/// Update session metadata (read-modify-write).
///
/// Reads the current session head, merges non-null fields, and writes a
/// new session record. Never touches conversation items.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Session identifier to update (null-terminated)
///   - title: New title (null = no change)
///   - marker: New marker (null = no change). E.g. "pinned", "archived", "deleted"
///   - metadata_json: New metadata JSON (null = no change; replaces entire object)
///
/// Returns:
///   - 0 on success
///   - storage_error if session not found
///   - resource_busy if database is locked by another process
pub export fn talu_storage_update_session(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const session_id_slice = std.mem.sliceTo(session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (db_path_slice.len == 0 or session_id_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path or session_id is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const title_slice: ?[]const u8 = if (title) |t| std.mem.span(t) else null;
    const marker_slice: ?[]const u8 = if (marker) |s| std.mem.span(s) else null;
    const metadata_slice: ?[]const u8 = if (metadata_json) |m| std.mem.span(m) else null;

    var adapter = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to open database for writing", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.backend().deinit();

    adapter.updateSession(allocator, session_id_slice, title_slice, marker_slice, metadata_slice) catch |err| {
        capi_error.setError(err, "failed to update session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Extended session update with source document linking.
///
/// Same as talu_storage_update_session but allows setting source_doc_id
/// to link the session to a document (for lineage tracking).
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Session identifier to update (null-terminated)
///   - title: New title (null = no change)
///   - marker: New marker (null = no change)
///   - metadata_json: New metadata JSON (null = no change)
///   - source_doc_id: Document ID to link (null = no change)
///
/// Returns:
///   - 0 on success
///   - storage_error if session not found
///   - resource_busy if database is locked
pub export fn talu_storage_update_session_ex(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    title: ?[*:0]const u8,
    marker: ?[*:0]const u8,
    metadata_json: ?[*:0]const u8,
    source_doc_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const session_id_slice = std.mem.sliceTo(session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (db_path_slice.len == 0 or session_id_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path or session_id is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const title_slice: ?[]const u8 = if (title) |t| std.mem.span(t) else null;
    const marker_slice: ?[]const u8 = if (marker) |s| std.mem.span(s) else null;
    const metadata_slice: ?[]const u8 = if (metadata_json) |m| std.mem.span(m) else null;
    const source_doc_slice: ?[]const u8 = if (source_doc_id) |d| std.mem.span(d) else null;

    var adapter = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to open database for writing", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.backend().deinit();

    adapter.updateSessionEx(allocator, session_id_slice, title_slice, marker_slice, metadata_slice, source_doc_slice) catch |err| {
        capi_error.setError(err, "failed to update session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Fork a conversation at a specific item, creating a new session.
///
/// Clones items up to and including `target_item_id` into `new_session_id`.
/// The new session's `parent_session_id` points to `source_session_id`.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - source_session_id: Session to fork from (null-terminated)
///   - target_item_id: Item ID to fork at (inclusive)
///   - new_session_id: Session ID for the forked copy (null-terminated)
///
/// Returns:
///   - 0 on success
///   - item_not_found if target_item_id does not exist
///   - resource_busy if database is locked
pub export fn talu_storage_fork_session(
    db_path: ?[*:0]const u8,
    source_session_id: ?[*:0]const u8,
    target_item_id: u64,
    new_session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const source_slice = std.mem.sliceTo(source_session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "source_session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const new_slice = std.mem.sliceTo(new_session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "new_session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);

    db.table.sessions.forkSession(allocator, db_path_slice, source_slice, target_item_id, new_slice) catch |err| {
        capi_error.setError(err, "failed to fork session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Delete a session and all its items from TaluDB.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Session identifier to delete (null-terminated)
///
/// Returns:
///   - 0 on success
///   - storage_error if session not found
///   - resource_busy if database is locked by another process
pub export fn talu_storage_delete_session(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    const session_id_slice = std.mem.sliceTo(session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }, 0);
    if (db_path_slice.len == 0 or session_id_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path or session_id is empty", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    var adapter = db.table.sessions.TableAdapter.init(allocator, db_path_slice, session_id_slice) catch |err| {
        if (err == error.LockUnavailable) {
            capi_error.setErrorWithCode(.resource_busy, "Database is locked by another process", .{});
            return @intFromEnum(error_codes.ErrorCode.resource_busy);
        }
        capi_error.setError(err, "failed to open database for writing", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer adapter.backend().deinit();

    adapter.deleteSession(session_id_slice) catch |err| {
        capi_error.setError(err, "failed to delete session", .{});
        return @intFromEnum(error_codes.ErrorCode.storage_error);
    };
    return 0;
}

/// Load a full conversation (all items) from TaluDB.
///
/// Uses scalar-column scanning to efficiently locate items for the specific session
/// without decoding unrelated payloads.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Session identifier to load (null-terminated)
///
/// Returns: Handle to a new Conversation containing the loaded items, or null on error.
/// Caller owns the handle and must free it with talu_responses_free().
pub export fn talu_storage_load_conversation(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
) callconv(.c) ?*responses_mod.ResponsesHandle {
    capi_error.clearError();

    const db_path_slice = std.mem.sliceTo(db_path orelse {
        capi_error.setErrorWithCode(.invalid_argument, "db_path is null", .{});
        return null;
    }, 0);
    const session_id_slice = std.mem.sliceTo(session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "session_id is null", .{});
        return null;
    }, 0);
    if (db_path_slice.len == 0 or session_id_slice.len == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "db_path or session_id is empty", .{});
        return null;
    }

    const conv = db.table.sessions.loadConversation(allocator, db_path_slice, session_id_slice) catch |err| {
        capi_error.setError(err, "failed to load conversation", .{});
        return null;
    };
    return @ptrCast(conv);
}

/// Free a session list returned by talu_storage_list_sessions.
///
/// Parameters:
///   - sessions: Session list handle to free (may be null)
pub export fn talu_storage_free_sessions(
    sessions: ?*CSessionList,
) callconv(.c) void {
    capi_error.clearError();

    const list = sessions orelse return;

    log.debug("capi", "talu_storage_free_sessions", .{
        .list_ptr = @intFromPtr(list),
        .count = list.count,
        .arena_ptr = if (list._arena) |a| @intFromPtr(a) else 0,
    }, @src());

    // Free the arena (frees all strings allocated into it)
    if (list._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    // Free the list struct itself
    allocator.destroy(list);
}

/// Set the maximum segment size (in bytes) for the TaluDB storage backend.
///
/// When the active segment (`current.talu`) would exceed this size after a
/// flush, the writer automatically seals it as `seg-<uuid>.talu`, updates
/// `manifest.json`, and creates a fresh `current.talu`.
///
/// Must be called after `talu_chat_set_storage_db`. Only affects the
/// chat-handle's writer; has no effect on callback-based backends.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object (must have a TaluDB backend)
///   - max_bytes: Maximum segment size in bytes (0 = use default 64 MB)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_chat_set_max_segment_size(
    chat_handle: ?*ChatHandle,
    max_bytes: u64,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const sb = chat.conv.storage_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no storage backend set (call talu_chat_set_storage_db first)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    // Verify this is a TaluDB backend (not a callback backend).
    if (sb.vtable != &DbBackendWrapper.vtable) {
        capi_error.setErrorWithCode(.invalid_argument, "storage backend is not a TaluDB backend", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const wrapper: *DbBackendWrapper = @ptrCast(@alignCast(sb.ptr));
    const size = if (max_bytes == 0) db.writer.default_max_segment_size else max_bytes;
    wrapper.backend.fs_writer.max_segment_size = size;
    return 0;
}

/// Set the WAL durability mode for the TaluDB chat storage backend.
///
/// Controls whether each WAL write is followed by fsync:
///   - 0 (full): fsync after every write. Survives OS crash and power loss.
///   - 1 (async_os): skip fsync; OS page cache buffers writes. Survives
///     application crashes but NOT OS crash or power loss.
///
/// Default is `full` (0). Call after `talu_chat_set_storage_db`.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object (must have a TaluDB backend)
///   - mode: 0 = full (fsync), 1 = async_os (no fsync)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_chat_set_durability(
    chat_handle: ?*ChatHandle,
    mode: u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const durability = std.meta.intToEnum(db.Durability, mode) catch {
        capi_error.setErrorWithCode(.invalid_argument, "invalid durability mode (expected 0 or 1)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const sb = chat.conv.storage_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no storage backend set (call talu_chat_set_storage_db first)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (sb.vtable != &DbBackendWrapper.vtable) {
        capi_error.setErrorWithCode(.invalid_argument, "storage backend is not a TaluDB backend", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const wrapper: *DbBackendWrapper = @ptrCast(@alignCast(sb.ptr));
    wrapper.backend.fs_writer.durability = durability;
    return 0;
}

/// Set the WAL durability mode for a TaluDB vector store.
///
/// Controls whether each WAL write is followed by fsync:
///   - 0 (full): fsync after every write. Survives OS crash and power loss.
///   - 1 (async_os): skip fsync; OS page cache buffers writes. Survives
///     application crashes but NOT OS crash or power loss.
///
/// Default is `full` (0).
///
/// Parameters:
///   - handle: Opaque VectorStoreHandle
///   - mode: 0 = full (fsync), 1 = async_os (no fsync)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_vector_store_set_durability(
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

/// Simulate a process crash for testing purposes.
///
/// Releases all file descriptors and locks WITHOUT flushing pending data
/// or deleting the WAL file. This accurately simulates what the OS does
/// when a process dies: all locks are released, but files remain on disk.
/// The orphaned WAL file will be replayed by the next `Writer.open`.
///
/// After calling this, the chat handle is in an invalid state.
/// Call `talu_chat_free` to release the remaining memory.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object (must have a TaluDB backend)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_chat_simulate_crash(
    chat_handle: ?*ChatHandle,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const sb = chat.conv.storage_backend orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no storage backend set", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (sb.vtable != &DbBackendWrapper.vtable) {
        capi_error.setErrorWithCode(.invalid_argument, "storage backend is not a TaluDB backend", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const wrapper: *DbBackendWrapper = @ptrCast(@alignCast(sb.ptr));
    wrapper.backend.simulateCrash();
    wrapper.allocator.destroy(wrapper.backend);

    // Clear the storage backend so talu_chat_free doesn't double-free.
    chat.conv.storage_backend = null;

    wrapper.allocator.destroy(wrapper);
    return 0;
}

// =============================================================================
// Tags API C Exports
// =============================================================================

/// C-ABI tag record for external backends.
pub const CTagRecord = extern struct {
    /// Tag UUID (null-terminated).
    tag_id: ?[*:0]const u8,
    /// Tag name (null-terminated).
    name: ?[*:0]const u8,
    /// Hex color code (null-terminated, may be null).
    color: ?[*:0]const u8,
    /// Description text (null-terminated, may be null).
    description: ?[*:0]const u8,
    /// Group ID for multi-tenant isolation (null-terminated, may be null).
    group_id: ?[*:0]const u8,
    /// Creation timestamp (Unix milliseconds).
    created_at_ms: i64,
    /// Last update timestamp (Unix milliseconds).
    updated_at_ms: i64,
    /// Reserved for future expansion.
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

/// Tag list container for C API.
pub const CTagList = extern struct {
    /// Array of tag records.
    tags: ?[*]CTagRecord,
    /// Number of tags in the array.
    count: usize,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

/// Build a CTagList from scanned tag records.
fn buildTagList(records: []db.table.tags.TagRecord) !*CTagList {
    const list = allocator.create(CTagList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const tags_buf = arena.alloc(CTagRecord, records.len) catch return error.OutOfMemory;

    for (records, 0..) |record, i| {
        tags_buf[i] = std.mem.zeroes(CTagRecord);
        tags_buf[i].tag_id = (arena.dupeZ(u8, record.tag_id) catch return error.OutOfMemory).ptr;
        tags_buf[i].name = (arena.dupeZ(u8, record.name) catch return error.OutOfMemory).ptr;
        tags_buf[i].color = if (record.color) |c|
            (arena.dupeZ(u8, c) catch return error.OutOfMemory).ptr
        else
            null;
        tags_buf[i].description = if (record.description) |d|
            (arena.dupeZ(u8, d) catch return error.OutOfMemory).ptr
        else
            null;
        tags_buf[i].group_id = if (record.group_id) |g|
            (arena.dupeZ(u8, g) catch return error.OutOfMemory).ptr
        else
            null;
        tags_buf[i].created_at_ms = record.created_at_ms;
        tags_buf[i].updated_at_ms = record.updated_at_ms;
    }

    list.* = .{
        .tags = if (records.len > 0) tags_buf.ptr else null,
        .count = records.len,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

/// List all tags, optionally filtered by group.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - group_id: Filter by group (null = no filter)
///   - out_tags: Output parameter to receive tag list handle
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_storage_free_tags().
pub export fn talu_storage_list_tags(
    db_path: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    out_tags: ?*?*CTagList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tags orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tags is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return listTagsImpl(db_path_slice, optSlice(group_id), out);
}

/// Get a tag by ID.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - tag_id: Tag UUID to query (null-terminated)
///   - out_tag: Output parameter to receive tag record
///
/// Returns: 0 on success, storage_error if tag not found.
///
/// Note: String fields use borrowed pointers stored in a thread-local buffer.
/// They are valid only until the next talu_storage_get_tag call.
pub export fn talu_storage_get_tag(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_tag: *CTagRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_tag.* = std.mem.zeroes(CTagRecord);
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getTagImpl(db_path_slice, tag_id_slice, out_tag);
}

/// Get a tag by name within a group.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - name: Tag name to query (null-terminated)
///   - group_id: Group ID (null = no group filter)
///   - out_tag: Output parameter to receive tag record
///
/// Returns: 0 on success, storage_error if tag not found.
pub export fn talu_storage_get_tag_by_name(
    db_path: ?[*:0]const u8,
    name: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
    out_tag: *CTagRecord,
) callconv(.c) i32 {
    capi_error.clearError();
    out_tag.* = std.mem.zeroes(CTagRecord);
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const name_slice = validateRequiredArg(name, "name") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getTagByNameImpl(db_path_slice, name_slice, optSlice(group_id), out_tag);
}

/// Copy TagRecord fields to CTagRecord using thread-local static buffers.
fn populateTagRecord(out: *CTagRecord, record: db.table.tags.TagRecord) void {
    const S = struct {
        threadlocal var tag_id_buf: [256]u8 = .{0} ** 256;
        threadlocal var name_buf: [256]u8 = .{0} ** 256;
        threadlocal var color_buf: [32]u8 = .{0} ** 32;
        threadlocal var description_buf: [1024]u8 = .{0} ** 1024;
        threadlocal var group_id_buf: [256]u8 = .{0} ** 256;
    };
    out.tag_id = copyToStaticBuf(&S.tag_id_buf, record.tag_id);
    out.name = copyToStaticBuf(&S.name_buf, record.name);
    out.color = copyToStaticBufOpt(&S.color_buf, record.color);
    out.description = copyToStaticBufOpt(&S.description_buf, record.description);
    out.group_id = copyToStaticBufOpt(&S.group_id_buf, record.group_id);
    out.created_at_ms = record.created_at_ms;
    out.updated_at_ms = record.updated_at_ms;
}

/// Create a new tag.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - tag_id: Tag UUID (null-terminated)
///   - name: Tag name (null-terminated)
///   - color: Hex color code (null = no color)
///   - description: Description text (null = no description)
///   - group_id: Group ID (null = no group)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_storage_create_tag(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
    group_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const name_slice = validateRequiredArg(name, "name") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return createTagImpl(db_path_slice, tag_id_slice, name_slice, optSlice(color), optSlice(description), optSlice(group_id));
}

/// Update an existing tag.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - tag_id: Tag UUID to update (null-terminated)
///   - name: New name (null = no change)
///   - color: New color (null = no change)
///   - description: New description (null = no change)
///
/// Returns: 0 on success, storage_error if tag not found.
pub export fn talu_storage_update_tag(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    name: ?[*:0]const u8,
    color: ?[*:0]const u8,
    description: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return updateTagImpl(db_path_slice, tag_id_slice, optSlice(name), optSlice(color), optSlice(description));
}

/// Delete a tag.
///
/// Also removes the tag from all conversations.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - tag_id: Tag UUID to delete (null-terminated)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_storage_delete_tag(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return deleteTagImpl(db_path_slice, tag_id_slice);
}

/// Free a tag list returned by talu_storage_list_tags.
///
/// Parameters:
///   - tags: Tag list handle to free (may be null)
pub export fn talu_storage_free_tags(
    tags: ?*CTagList,
) callconv(.c) void {
    capi_error.clearError();

    const list = tags orelse return;

    if (list._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(list);
}

// =============================================================================
// Conversation-Tag Junction API
// =============================================================================

/// Add a tag to a conversation.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Conversation session ID (null-terminated)
///   - tag_id: Tag UUID to add (null-terminated)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_storage_add_conversation_tag(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return addConversationTagImpl(db_path_slice, session_id_slice, tag_id_slice);
}

/// Remove a tag from a conversation.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Conversation session ID (null-terminated)
///   - tag_id: Tag UUID to remove (null-terminated)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_storage_remove_conversation_tag(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return removeConversationTagImpl(db_path_slice, session_id_slice, tag_id_slice);
}

/// String list container for C API.
pub const CStringList = extern struct {
    /// Array of null-terminated string pointers.
    strings: ?[*]?[*:0]const u8,
    /// Number of strings in the array.
    count: usize,
    /// Internal: backing arena for string data.
    _arena: ?*anyopaque,
};

/// Build a CStringList from a slice of strings.
fn buildStringList(strings: [][]const u8) !*CStringList {
    const list = allocator.create(CStringList) catch return error.OutOfMemory;
    errdefer allocator.destroy(list);

    const arena_ptr = allocator.create(std.heap.ArenaAllocator) catch return error.OutOfMemory;
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    errdefer arena_ptr.deinit();

    const arena = arena_ptr.allocator();

    const strings_buf = arena.alloc(?[*:0]const u8, strings.len) catch return error.OutOfMemory;

    for (strings, 0..) |s, i| {
        strings_buf[i] = (arena.dupeZ(u8, s) catch return error.OutOfMemory).ptr;
    }

    list.* = .{
        .strings = if (strings.len > 0) strings_buf.ptr else null,
        .count = strings.len,
        ._arena = @ptrCast(arena_ptr),
    };

    return list;
}

/// Get all tag IDs for a conversation.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - session_id: Conversation session ID (null-terminated)
///   - out_tag_ids: Output parameter to receive tag ID list
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_storage_free_string_list().
pub export fn talu_storage_get_conversation_tags(
    db_path: ?[*:0]const u8,
    session_id: ?[*:0]const u8,
    out_tag_ids: ?*?*CStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_tag_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_tag_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const session_id_slice = validateRequiredArg(session_id, "session_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getConversationTagsImpl(db_path_slice, session_id_slice, out);
}

/// Get all conversation IDs that have a specific tag.
///
/// Parameters:
///   - db_path: Path to TaluDB storage directory (null-terminated)
///   - tag_id: Tag UUID to query (null-terminated)
///   - out_session_ids: Output parameter to receive session ID list
///
/// Returns: 0 on success, negative error code on failure.
/// On success, caller must free the handle with talu_storage_free_string_list().
pub export fn talu_storage_get_tag_conversations(
    db_path: ?[*:0]const u8,
    tag_id: ?[*:0]const u8,
    out_session_ids: ?*?*CStringList,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_session_ids orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_session_ids is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const tag_id_slice = validateRequiredArg(tag_id, "tag_id") orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    return getTagConversationsImpl(db_path_slice, tag_id_slice, out);
}

/// Free a string list returned by conversation tag functions.
///
/// Parameters:
///   - list: String list handle to free (may be null)
pub export fn talu_storage_free_string_list(
    list: ?*CStringList,
) callconv(.c) void {
    capi_error.clearError();

    const l = list orelse return;

    if (l._arena) |arena_ptr| {
        const arena: *std.heap.ArenaAllocator = @ptrCast(@alignCast(arena_ptr));
        arena.deinit();
        allocator.destroy(arena);
    }

    allocator.destroy(l);
}

// =============================================================================
// Document Tag Inheritance API
// =============================================================================

/// Copy tags from a document to a conversation.
///
/// This function looks up the document identified by `prompt_id` (if set on the chat),
/// retrieves its tags, and adds them to the conversation's session.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object (must have prompt_id set)
///   - db_path: Path to TaluDB storage directory (null-terminated)
///
/// Returns: 0 on success, negative error code on failure.
/// Returns invalid_argument if prompt_id is not set on the chat.
/// Returns invalid_argument if session_id is not set on the chat.
pub export fn talu_chat_inherit_tags(
    chat_handle: ?*ChatHandle,
    db_path: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();
    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse return setArgError("chat_handle is null")));
    const db_path_slice = validateDbPath(db_path) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    const prompt_id = chat.getPromptId() orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no prompt_id set (nothing to inherit)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const session_id = chat.session_id orelse chat.conv.session_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "no session_id set (cannot tag non-persisted session)", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const doc_tag_ids = db.table.document_tags.getDocumentTagIds(allocator, db_path_slice, prompt_id) catch |err| {
        capi_error.setError(err, "failed to get document tags", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer db.table.document_tags.freeStringSlice(allocator, @constCast(doc_tag_ids));
    for (doc_tag_ids) |tag_id| {
        const rc = addConversationTagImpl(db_path_slice, session_id, tag_id);
        if (rc != 0) log.debug("capi", "inherit_tags_failed", .{ .rc = rc }, @src());
    }
    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "CStorageRecord size and alignment" {
    // Verify struct is C-compatible
    try std.testing.expect(@sizeOf(CStorageRecord) > 0);
    try std.testing.expect(@alignOf(CStorageRecord) <= 8);
}

test "CStorageEvent size and alignment" {
    try std.testing.expect(@sizeOf(CStorageEvent) > 0);
    try std.testing.expect(@alignOf(CStorageEvent) <= 8);
}

test "CItemType values match ItemType" {
    const items = @import("../responses/items.zig");
    try std.testing.expectEqual(@intFromEnum(CItemType.message), @intFromEnum(items.ItemType.message));
    try std.testing.expectEqual(@intFromEnum(CItemType.function_call), @intFromEnum(items.ItemType.function_call));
    try std.testing.expectEqual(@intFromEnum(CItemType.reasoning), @intFromEnum(items.ItemType.reasoning));
}

test "CMessageRole values match MessageRole" {
    const items = @import("../responses/items.zig");
    try std.testing.expectEqual(@intFromEnum(CMessageRole.system), @intFromEnum(items.MessageRole.system));
    try std.testing.expectEqual(@intFromEnum(CMessageRole.user), @intFromEnum(items.MessageRole.user));
    try std.testing.expectEqual(@intFromEnum(CMessageRole.assistant), @intFromEnum(items.MessageRole.assistant));
}

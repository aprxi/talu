//! DB C-API: Operational and administrative functions.

const std = @import("std");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const responses_mod = @import("../responses.zig");
const responses_root = @import("../../responses/root.zig");
const db = @import("../../db/root.zig");
const log = @import("../../log.zig");
const vector = @import("vector.zig");

const allocator = std.heap.c_allocator;
const ChatHandle = responses_mod.ChatHandle;
const Chat = responses_root.Chat;
const Conversation = responses_root.Conversation;
const StorageBackend = responses_root.StorageBackend;
const StorageEvent = responses_root.StorageEvent;

// =============================================================================
// C-ABI Storage Event Types
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

    /// Search snippet: text fragment around the matched search query in item content.
    /// Only populated when listing with a search_query that matched item content.
    /// Null when no search query or when matched by metadata only.
    search_snippet: ?[*:0]const u8,

    /// Source document ID for lineage tracking (null-terminated, may be null).
    /// Links this session to the prompt/persona document that spawned it.
    source_doc_id: ?[*:0]const u8,

    /// Project identifier for multi-project session organization (null-terminated, may be null).
    project_id: ?[*:0]const u8,

    /// Creation timestamp (Unix milliseconds).
    created_at_ms: i64,

    /// Last updated timestamp (Unix milliseconds).
    updated_at_ms: i64,

    /// Reserved for future expansion.
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

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
// TaluDB Backend Wrapper
// =============================================================================

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

    fn loadAllImpl(ptr: *anyopaque, alloc: std.mem.Allocator) anyerror![]responses_root.ItemRecord {
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

// =============================================================================
// C-API Operational Functions
// =============================================================================

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
pub export fn talu_db_ops_set_storage_db(
    chat_handle: ?*anyopaque,
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

/// Set the maximum segment size for the TaluDB chat storage backend.
///
/// Controls WAL segment rollover. When a segment exceeds this size,
/// the next write will roll over to a new segment.
///
/// Parameters:
///   - chat_handle: Handle to the Chat object (must have a TaluDB backend)
///   - max_bytes: Maximum segment size in bytes (0 = use default)
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_db_ops_set_max_segment_size(
    chat_handle: ?*anyopaque,
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
pub export fn talu_db_ops_set_durability(
    chat_handle: ?*anyopaque,
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
pub export fn talu_db_ops_simulate_crash(
    chat_handle: ?*anyopaque,
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
// Vector Store Operations
// =============================================================================

/// Compact vector storage to remove tombstones.
///
/// Parameters:
///   - handle: Opaque VectorStoreHandle
///   - dims: Vector dimensionality
///   - out_kept_count: Output parameter for number of vectors kept
///   - out_removed_tombstones: Output parameter for number of tombstones removed
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_db_ops_vector_compact(
    handle: ?*vector.VectorStoreHandle,
    dims: u32,
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

    const result = backend.compactIdempotent(dims, 0, 0) catch |err| {
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

/// Build pending approximate indexes if the manifest generation matches expected_generation.
///
/// Parameters:
///   - handle: Opaque VectorStoreHandle
///   - expected_generation: Expected manifest generation (guards against concurrent modification)
///   - max_segments: Maximum number of segments to build (0 = no limit)
///   - out_built_segments: Output parameter for number of segments built
///   - out_failed_segments: Output parameter for number of segments that failed to build
///   - out_pending_segments: Output parameter for number of segments still pending
///
/// Returns: 0 on success, negative error code on failure.
pub export fn talu_db_ops_vector_build_indexes_with_generation(
    handle: ?*vector.VectorStoreHandle,
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

// =============================================================================
// Snapshot Operations (Not Implemented)
// =============================================================================

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
    const items = @import("../../responses/items.zig");
    try std.testing.expectEqual(@intFromEnum(CItemType.message), @intFromEnum(items.ItemType.message));
    try std.testing.expectEqual(@intFromEnum(CItemType.function_call), @intFromEnum(items.ItemType.function_call));
    try std.testing.expectEqual(@intFromEnum(CItemType.reasoning), @intFromEnum(items.ItemType.reasoning));
}

test "CMessageRole values match MessageRole" {
    const items = @import("../../responses/items.zig");
    try std.testing.expectEqual(@intFromEnum(CMessageRole.system), @intFromEnum(items.MessageRole.system));
    try std.testing.expectEqual(@intFromEnum(CMessageRole.user), @intFromEnum(items.MessageRole.user));
    try std.testing.expectEqual(@intFromEnum(CMessageRole.assistant), @intFromEnum(items.MessageRole.assistant));
}

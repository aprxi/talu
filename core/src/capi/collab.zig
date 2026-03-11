//! Collaboration C API.

const std = @import("std");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");
const db_helpers = @import("db/helpers.zig");
const collab_resource = @import("../collab/resource_store.zig");
const collab_types = @import("../collab/types.zig");
const kv_store = @import("../db/kv/store.zig");

const allocator = std.heap.c_allocator;

pub const CollabHandle = opaque {};

pub const CCollabSession = extern struct {
    namespace: ?[*:0]const u8,
    participant_id: ?[*:0]const u8,
    participant_kind: u8,
    _flags_reserved: [7]u8 = [_]u8{0} ** 7,
    status: ?[*:0]const u8,
    _reserved: [8]u8 = [_]u8{0} ** 8,
    _arena: ?*anyopaque,
};

pub const CCollabSummary = extern struct {
    namespace: ?[*:0]const u8,
    meta_json: ?[*:0]const u8,
    total_live_entries: usize,
    batched_pending: usize,
    ephemeral_live_entries: usize,
    watch_published: u64,
    _reserved: [8]u8 = [_]u8{0} ** 8,
    _arena: ?*anyopaque,
};

pub const CCollabValue = extern struct {
    data: ?[*]u8,
    len: usize,
    updated_at_ms: i64,
    found: bool,
    _reserved: [7]u8 = [_]u8{0} ** 7,
    _arena: ?*anyopaque,
};

pub const CCollabOpResult = extern struct {
    op_key: ?[*:0]const u8,
    accepted: bool,
    _flags_reserved: [7]u8 = [_]u8{0} ** 7,
    _arena: ?*anyopaque,
};

pub const CCollabHistoryEntry = extern struct {
    actor_id: ?[*:0]const u8,
    actor_seq: u64,
    op_id: ?[*:0]const u8,
    payload: ?[*]const u8,
    payload_len: usize,
    updated_at_ms: i64,
    key: ?[*:0]const u8,
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

pub const CCollabHistoryList = extern struct {
    items: ?[*]CCollabHistoryEntry,
    count: usize,
    _arena: ?*anyopaque,
};

pub const CCollabWatchEvent = extern struct {
    seq: u64,
    event_type: u8,
    durability_class: u8,
    has_durability: bool,
    ttl_ms: u64,
    has_ttl: bool,
    _flags_reserved: [6]u8 = [_]u8{0} ** 6,
    key: ?[*:0]const u8,
    value_len: usize,
    updated_at_ms: i64,
    _reserved: [8]u8 = [_]u8{0} ** 8,
};

pub const CCollabWatchBatch = extern struct {
    items: ?[*]CCollabWatchEvent,
    count: usize,
    lost: bool,
    _flags_reserved: [7]u8 = [_]u8{0} ** 7,
    _arena: ?*anyopaque,
};

pub const CCollabWatchWaitResult = extern struct {
    published_seq: u64,
    timed_out: bool,
    _flags_reserved: [7]u8 = [_]u8{0} ** 7,
};

fn toStore(handle: ?*CollabHandle) !*collab_resource.ResourceStore {
    return @ptrCast(@alignCast(handle orelse return error.InvalidHandle));
}

fn validateBytes(bytes: ?[*]const u8, bytes_len: usize) ?[]const u8 {
    if (bytes_len == 0) return &.{};
    const ptr = bytes orelse {
        capi_error.setErrorWithCode(.invalid_argument, "value is null but value_len > 0", .{});
        return null;
    };
    return ptr[0..bytes_len];
}

fn parseParticipantKind(raw: u8) ?collab_types.ParticipantKind {
    return switch (raw) {
        0 => .human,
        1 => .agent,
        2 => .external,
        3 => .system,
        else => null,
    };
}

fn buildArena() !*std.heap.ArenaAllocator {
    const arena_ptr = try allocator.create(std.heap.ArenaAllocator);
    errdefer allocator.destroy(arena_ptr);
    arena_ptr.* = std.heap.ArenaAllocator.init(allocator);
    return arena_ptr;
}

fn buildSession(info: collab_resource.SessionOpenInfo) !CCollabSession {
    var out = std.mem.zeroes(CCollabSession);
    const arena_ptr = try buildArena();
    errdefer {
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    const arena = arena_ptr.allocator();
    out.namespace = (try arena.dupeZ(u8, info.namespace)).ptr;
    out.participant_id = (try arena.dupeZ(u8, info.participant_id)).ptr;
    out.participant_kind = @intFromEnum(info.participant_kind);
    out.status = (try arena.dupeZ(u8, info.status)).ptr;
    out._arena = @ptrCast(arena_ptr);
    return out;
}

fn buildSummary(summary: collab_resource.ResourceSummary) !CCollabSummary {
    var out = std.mem.zeroes(CCollabSummary);
    const arena_ptr = try buildArena();
    errdefer {
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    const arena = arena_ptr.allocator();
    out.namespace = (try arena.dupeZ(u8, summary.namespace)).ptr;
    if (summary.meta_json) |meta_json| {
        out.meta_json = (try arena.dupeZ(u8, meta_json)).ptr;
    }
    out.total_live_entries = summary.stats.total_live_entries;
    out.batched_pending = summary.stats.batched_pending;
    out.ephemeral_live_entries = summary.stats.ephemeral_live_entries;
    out.watch_published = summary.stats.watch_published;
    out._arena = @ptrCast(arena_ptr);
    return out;
}

fn buildValue(value: collab_resource.BinaryValue) !CCollabValue {
    var out = std.mem.zeroes(CCollabValue);
    if (value.data) |data| {
        const arena_ptr = try buildArena();
        errdefer {
            arena_ptr.deinit();
            allocator.destroy(arena_ptr);
        }
        const arena = arena_ptr.allocator();
        const copy = try arena.dupe(u8, data);
        out.data = if (copy.len > 0) copy.ptr else null;
        out.len = copy.len;
        out.updated_at_ms = value.updated_at_ms orelse 0;
        out.found = true;
        out._arena = @ptrCast(arena_ptr);
    }
    return out;
}

fn buildOpResult(op_key: []const u8) !CCollabOpResult {
    var out = std.mem.zeroes(CCollabOpResult);
    const arena_ptr = try buildArena();
    errdefer {
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    const arena = arena_ptr.allocator();
    out.op_key = (try arena.dupeZ(u8, op_key)).ptr;
    out.accepted = true;
    out._arena = @ptrCast(arena_ptr);
    return out;
}

fn buildHistory(entries: []const collab_resource.OperationHistoryEntry) !CCollabHistoryList {
    const arena_ptr = try buildArena();
    errdefer {
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    const arena = arena_ptr.allocator();
    const items = try arena.alloc(CCollabHistoryEntry, entries.len);
    for (entries, 0..) |entry, idx| {
        items[idx] = std.mem.zeroes(CCollabHistoryEntry);
        items[idx].actor_id = (try arena.dupeZ(u8, entry.actor_id)).ptr;
        items[idx].actor_seq = entry.actor_seq;
        items[idx].op_id = (try arena.dupeZ(u8, entry.op_id)).ptr;
        const payload = try arena.dupe(u8, entry.payload);
        items[idx].payload = if (payload.len > 0) payload.ptr else null;
        items[idx].payload_len = payload.len;
        items[idx].updated_at_ms = entry.updated_at_ms;
        items[idx].key = (try arena.dupeZ(u8, entry.key)).ptr;
    }
    return .{
        .items = if (entries.len > 0) items.ptr else null,
        .count = entries.len,
        ._arena = @ptrCast(arena_ptr),
    };
}

fn buildWatchBatch(events: []const kv_store.WatchEvent, lost: bool) !CCollabWatchBatch {
    const arena_ptr = try buildArena();
    errdefer {
        arena_ptr.deinit();
        allocator.destroy(arena_ptr);
    }
    const arena = arena_ptr.allocator();
    const items = try arena.alloc(CCollabWatchEvent, events.len);
    for (events, 0..) |event, idx| {
        items[idx] = std.mem.zeroes(CCollabWatchEvent);
        items[idx].seq = event.seq;
        items[idx].event_type = @intFromEnum(event.event_type);
        if (event.durability) |durability| {
            items[idx].durability_class = @intFromEnum(durability);
            items[idx].has_durability = true;
        }
        if (event.ttl_ms) |ttl_ms| {
            items[idx].ttl_ms = ttl_ms;
            items[idx].has_ttl = true;
        }
        items[idx].key = (try arena.dupeZ(u8, event.key)).ptr;
        items[idx].value_len = event.value_len;
        items[idx].updated_at_ms = event.updated_at_ms;
    }
    return .{
        .items = if (events.len > 0) items.ptr else null,
        .count = events.len,
        .lost = lost,
        ._arena = @ptrCast(arena_ptr),
    };
}

fn freeArena(ptr: ?*anyopaque) void {
    const arena_ptr = @as(?*std.heap.ArenaAllocator, @ptrCast(@alignCast(ptr))) orelse return;
    arena_ptr.deinit();
    allocator.destroy(arena_ptr);
}

pub export fn talu_collab_init(
    db_path: ?[*:0]const u8,
    resource_kind: ?[*:0]const u8,
    resource_id: ?[*:0]const u8,
    out_handle: ?*?*CollabHandle,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const db_root = db_helpers.validateDbPath(db_path) orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const kind = db_helpers.validateRequiredArg(resource_kind, "resource_kind") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const id = db_helpers.validateRequiredArg(resource_id, "resource_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const store_ptr = allocator.create(collab_resource.ResourceStore) catch {
        capi_error.setErrorWithCode(.out_of_memory, "failed to allocate collab handle", .{});
        return @intFromEnum(error_codes.ErrorCode.out_of_memory);
    };
    errdefer allocator.destroy(store_ptr);

    store_ptr.* = collab_resource.ResourceStore.init(allocator, db_root, kind, id) catch |err| {
        capi_error.setError(err, "failed to initialize collab resource", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(store_ptr);
    return 0;
}

pub export fn talu_collab_free(handle: ?*CollabHandle) callconv(.c) void {
    capi_error.clearError();
    const store: *collab_resource.ResourceStore = @ptrCast(@alignCast(handle orelse return));
    store.deinit();
    allocator.destroy(store);
}

pub export fn talu_collab_open_session(
    handle: ?*CollabHandle,
    participant_id: ?[*:0]const u8,
    participant_kind: u8,
    role: ?[*:0]const u8,
    out_session: ?*CCollabSession,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_session orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_session is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabSession);

    const pid = db_helpers.validateRequiredArg(participant_id, "participant_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const parsed_kind = parseParticipantKind(participant_kind) orelse {
        capi_error.setErrorWithCode(.invalid_argument, "participant_kind is invalid", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const session = store.openSession(pid, parsed_kind, db_helpers.optSlice(role)) catch |err| {
        capi_error.setError(err, "failed to open collab session", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out.* = buildSession(session) catch |err| {
        capi_error.setError(err, "failed to build collab session output", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_get_summary(
    handle: ?*CollabHandle,
    out_summary: ?*CCollabSummary,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_summary orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_summary is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabSummary);

    var summary = store.getSummary(allocator) catch |err| {
        capi_error.setError(err, "failed to read collab summary", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer summary.deinit(allocator);

    out.* = buildSummary(summary) catch |err| {
        capi_error.setError(err, "failed to build collab summary output", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_get_snapshot(
    handle: ?*CollabHandle,
    out_value: ?*CCollabValue,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_value orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_value is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabValue);

    var value = store.getSnapshot(allocator) catch |err| {
        capi_error.setError(err, "failed to read collab snapshot", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer value.deinit(allocator);

    out.* = buildValue(value) catch |err| {
        capi_error.setError(err, "failed to build collab snapshot output", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_submit_op(
    handle: ?*CollabHandle,
    actor_id: ?[*:0]const u8,
    actor_seq: u64,
    op_id: ?[*:0]const u8,
    payload: ?[*]const u8,
    payload_len: usize,
    issued_at_ms: i64,
    has_issued_at_ms: bool,
    snapshot: ?[*]const u8,
    snapshot_len: usize,
    has_snapshot: bool,
    out_result: ?*CCollabOpResult,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_result orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_result is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabOpResult);

    const actor = db_helpers.validateRequiredArg(actor_id, "actor_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const op = db_helpers.validateRequiredArg(op_id, "op_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const payload_slice = validateBytes(payload, payload_len) orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    if (payload_slice.len == 0 or actor_seq == 0) {
        capi_error.setErrorWithCode(.invalid_argument, "invalid operation envelope", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }

    const snapshot_slice = if (has_snapshot)
        (validateBytes(snapshot, snapshot_len) orelse return @intFromEnum(error_codes.ErrorCode.invalid_argument))
    else
        null;

    const op_key = store.submitOperation(.{
        .actor_id = actor,
        .actor_seq = actor_seq,
        .op_id = op,
        .payload = payload_slice,
        .issued_at_ms = if (has_issued_at_ms) issued_at_ms else std.time.milliTimestamp(),
    }, snapshot_slice) catch |err| {
        capi_error.setError(err, "failed to submit collab operation", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(op_key);

    out.* = buildOpResult(op_key) catch |err| {
        capi_error.setError(err, "failed to build collab op result", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_get_history(
    handle: ?*CollabHandle,
    after_key: ?[*:0]const u8,
    limit: usize,
    out_history: ?*CCollabHistoryList,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_history orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_history is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabHistoryList);

    const history = store.getHistory(allocator, if (after_key) |value| std.mem.span(value) else null, limit) catch |err| {
        capi_error.setError(err, "failed to read collab history", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer collab_resource.freeOperationHistoryEntries(allocator, history);

    out.* = buildHistory(history) catch |err| {
        capi_error.setError(err, "failed to build collab history output", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_clear_snapshot(
    handle: ?*CollabHandle,
    actor_id: ?[*:0]const u8,
    actor_kind: u8,
    role: ?[*:0]const u8,
    op_kind: ?[*:0]const u8,
    out_result: ?*CCollabOpResult,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const actor = db_helpers.validateRequiredArg(actor_id, "actor_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const kind = parseParticipantKind(actor_kind) orelse {
        capi_error.setErrorWithCode(.invalid_argument, "invalid participant kind", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const op = db_helpers.validateRequiredArg(op_kind, "op_kind") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const out = out_result orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_result is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabOpResult);

    const op_key = store.clearSnapshot(
        actor,
        kind,
        if (role) |value| std.mem.span(value) else null,
        op,
    ) catch |err| {
        capi_error.setError(err, "failed to clear collab snapshot", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer allocator.free(op_key);

    out.* = buildOpResult(op_key) catch |err| {
        capi_error.setError(err, "failed to build collab op result", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_put_presence(
    handle: ?*CollabHandle,
    participant_id: ?[*:0]const u8,
    payload: ?[*]const u8,
    payload_len: usize,
    ttl_ms: u64,
    out_ttl_ms: ?*u64,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const pid = db_helpers.validateRequiredArg(participant_id, "participant_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const payload_slice = validateBytes(payload, payload_len) orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const effective_ttl = store.putPresence(pid, payload_slice, ttl_ms) catch |err| {
        capi_error.setError(err, "failed to write collab presence", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    if (out_ttl_ms) |ttl_out| ttl_out.* = effective_ttl;
    return 0;
}

pub export fn talu_collab_get_presence(
    handle: ?*CollabHandle,
    participant_id: ?[*:0]const u8,
    out_value: ?*CCollabValue,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_value orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_value is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabValue);

    const pid = db_helpers.validateRequiredArg(participant_id, "participant_id") orelse {
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    var value = store.getPresence(allocator, pid) catch |err| {
        capi_error.setError(err, "failed to read collab presence", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer value.deinit(allocator);

    out.* = buildValue(value) catch |err| {
        capi_error.setError(err, "failed to build collab presence output", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_watch_drain(
    handle: ?*CollabHandle,
    after_seq: u64,
    max_events: usize,
    out_batch: ?*CCollabWatchBatch,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_batch orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_batch is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabWatchBatch);

    var drained = store.watchDrain(allocator, after_seq, max_events) catch |err| {
        capi_error.setError(err, "failed to drain collab watch events", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    defer drained.deinit(allocator);

    out.* = buildWatchBatch(drained.events, drained.lost) catch |err| {
        capi_error.setError(err, "failed to build collab watch batch", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

pub export fn talu_collab_watch_wait(
    handle: ?*CollabHandle,
    after_seq: u64,
    timeout_ms: u64,
    out_result: ?*CCollabWatchWaitResult,
) callconv(.c) i32 {
    capi_error.clearError();
    const store = toStore(handle) catch |err| {
        capi_error.setError(err, "invalid collab handle", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    const out = out_result orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_result is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = std.mem.zeroes(CCollabWatchWaitResult);

    const result = store.watchWait(after_seq, timeout_ms) catch |err| {
        capi_error.setError(err, "failed to wait for collab watch event", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    out.published_seq = result.published_seq;
    out.timed_out = result.timed_out;
    return 0;
}

pub export fn talu_collab_free_session(session: ?*CCollabSession) callconv(.c) void {
    capi_error.clearError();
    freeArena((session orelse return)._arena);
    session.?._arena = null;
}

pub export fn talu_collab_free_summary(summary: ?*CCollabSummary) callconv(.c) void {
    capi_error.clearError();
    freeArena((summary orelse return)._arena);
    summary.?._arena = null;
}

pub export fn talu_collab_free_value(value: ?*CCollabValue) callconv(.c) void {
    capi_error.clearError();
    freeArena((value orelse return)._arena);
    value.?._arena = null;
}

pub export fn talu_collab_free_op_result(result: ?*CCollabOpResult) callconv(.c) void {
    capi_error.clearError();
    freeArena((result orelse return)._arena);
    result.?._arena = null;
}

pub export fn talu_collab_free_history(history: ?*CCollabHistoryList) callconv(.c) void {
    capi_error.clearError();
    freeArena((history orelse return)._arena);
    history.?._arena = null;
}

pub export fn talu_collab_free_watch_batch(batch: ?*CCollabWatchBatch) callconv(.c) void {
    capi_error.clearError();
    freeArena((batch orelse return)._arena);
    batch.?._arena = null;
}

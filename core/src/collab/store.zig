//! Collaboration storage adapter over DB KV.
//!
//! This module codifies how collaboration data maps onto KV durability lanes.
//! It keeps storage semantics in core so boundary layers only forward requests.

const std = @import("std");
const types = @import("types.zig");
const kv_store = @import("../db/kv/store.zig");

const Allocator = std.mem.Allocator;

/// Default TTL for ephemeral presence writes.
pub const default_presence_ttl_ms: u64 = 5_000;

pub const SessionStore = struct {
    allocator: Allocator,
    session_id: []u8,
    namespace: []u8,
    kv: kv_store.KVStore,

    pub fn init(allocator: Allocator, db_root: []const u8, session_id: []const u8) !SessionStore {
        if (session_id.len == 0) return error.InvalidArgument;

        const session_id_copy = try allocator.dupe(u8, session_id);
        errdefer allocator.free(session_id_copy);

        const namespace = try buildNamespace(allocator, session_id);
        errdefer allocator.free(namespace);

        var kv = try kv_store.KVStore.init(allocator, db_root, namespace);
        errdefer kv.deinit();

        return .{
            .allocator = allocator,
            .session_id = session_id_copy,
            .namespace = namespace,
            .kv = kv,
        };
    }

    pub fn deinit(self: *SessionStore) void {
        self.kv.deinit();
        self.allocator.free(self.namespace);
        self.allocator.free(self.session_id);
    }

    pub fn sessionId(self: *const SessionStore) []const u8 {
        return self.session_id;
    }

    pub fn namespaceId(self: *const SessionStore) []const u8 {
        return self.namespace;
    }

    /// Presence is always ephemeral and should self-expire quickly.
    pub fn putPresence(
        self: *SessionStore,
        participant_id: []const u8,
        payload: []const u8,
        ttl_ms: u64,
    ) !void {
        const key = try scopedKey(self.allocator, "presence", participant_id);
        defer self.allocator.free(key);

        const effective_ttl = if (ttl_ms == 0) default_presence_ttl_ms else ttl_ms;
        try self.kv.putWithOptions(key, payload, .{
            .durability = .ephemeral,
            .ttl_ms = effective_ttl,
        });
    }

    pub fn clearPresence(self: *SessionStore, participant_id: []const u8) !bool {
        const key = try scopedKey(self.allocator, "presence", participant_id);
        defer self.allocator.free(key);
        return self.kv.delete(key);
    }

    pub fn getPresenceCopy(self: *SessionStore, allocator: Allocator, participant_id: []const u8) !?[]u8 {
        const key = try scopedKey(self.allocator, "presence", participant_id);
        defer self.allocator.free(key);
        return self.kv.getCopy(allocator, key);
    }

    /// Recoverable, high-frequency participant state should use batched durability.
    pub fn putParticipantState(self: *SessionStore, participant_id: []const u8, payload: []const u8) !void {
        const key = try scopedKey(self.allocator, "participants", participant_id);
        defer self.allocator.free(key);
        try self.kv.putWithOptions(key, payload, .{ .durability = .batched });
    }

    pub fn getParticipantStateCopy(self: *SessionStore, allocator: Allocator, participant_id: []const u8) !?[]u8 {
        const key = try scopedKey(self.allocator, "participants", participant_id);
        defer self.allocator.free(key);
        return self.kv.getCopy(allocator, key);
    }

    /// Revision checkpoints and similar authoritative state are strong-durable.
    pub fn putCheckpoint(self: *SessionStore, name: []const u8, payload: []const u8) !void {
        const key = try scopedKey(self.allocator, "checkpoints", name);
        defer self.allocator.free(key);
        try self.kv.putWithOptions(key, payload, .{ .durability = .strong });
    }

    pub fn getCheckpointCopy(self: *SessionStore, allocator: Allocator, name: []const u8) !?[]u8 {
        const key = try scopedKey(self.allocator, "checkpoints", name);
        defer self.allocator.free(key);
        return self.kv.getCopy(allocator, key);
    }

    /// Persist one CRDT operation envelope as strong-durable session history.
    pub fn appendOperation(self: *SessionStore, op: types.OperationEnvelope) !void {
        const key = try op.key(self.allocator);
        defer self.allocator.free(key);
        try self.kv.putWithOptions(key, op.payload, .{ .durability = .strong });
    }

    pub fn watchDrain(self: *SessionStore, allocator: Allocator, after_seq: u64, max_count: usize) !kv_store.WatchDrainResult {
        return self.kv.watchDrain(allocator, after_seq, max_count);
    }

    pub fn flushBatched(self: *SessionStore) !void {
        try self.kv.flushBatched();
    }

    pub fn flush(self: *SessionStore) !void {
        try self.kv.flush();
    }

    pub fn stats(self: *SessionStore) kv_store.NamespaceStats {
        return self.kv.stats();
    }

    pub fn setNowMsForTesting(self: *SessionStore, now_ms: ?i64) void {
        self.kv.setNowMsForTesting(now_ms);
    }
};

fn buildNamespace(allocator: Allocator, session_id: []const u8) ![]u8 {
    const prefix = "collab-session-";
    var out = try allocator.alloc(u8, prefix.len + session_id.len * 2);
    @memcpy(out[0..prefix.len], prefix);
    const hex = std.fmt.hex_charset;
    for (session_id, 0..) |b, idx| {
        const off = prefix.len + idx * 2;
        out[off] = hex[b >> 4];
        out[off + 1] = hex[b & 0x0f];
    }
    return out;
}

fn scopedKey(allocator: Allocator, prefix: []const u8, id: []const u8) ![]u8 {
    if (id.len == 0) return error.InvalidArgument;
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, id });
}

test "SessionStore.putPresence uses ephemeral lane and default ttl" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try SessionStore.init(std.testing.allocator, root, "doc:1");
    defer store.deinit();

    store.setNowMsForTesting(1_000);
    try store.putPresence("human:1", "{\"cursor\":10}", 0);

    {
        const value = (try store.getPresenceCopy(std.testing.allocator, "human:1")).?;
        defer std.testing.allocator.free(value);
        try std.testing.expectEqualStrings("{\"cursor\":10}", value);
    }

    var events = try store.watchDrain(std.testing.allocator, 0, 16);
    defer events.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), events.events.len);
    try std.testing.expectEqual(kv_store.WatchEventType.put, events.events[0].event_type);
    try std.testing.expectEqual(@as(?kv_store.DurabilityClass, .ephemeral), events.events[0].durability);
    try std.testing.expectEqual(@as(?u64, default_presence_ttl_ms), events.events[0].ttl_ms);
    try std.testing.expectEqualStrings("presence/human:1", events.events[0].key);
}

test "SessionStore.putParticipantState queues batched writes and flushes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try SessionStore.init(std.testing.allocator, root, "doc:2");
    defer store.deinit();

    store.setNowMsForTesting(2_000);
    try store.putParticipantState("agent:7", "{\"role\":\"reviewer\"}");

    var stats = store.stats();
    try std.testing.expectEqual(@as(usize, 1), stats.batched_pending);

    try store.flushBatched();
    try store.flush();
    stats = store.stats();
    try std.testing.expectEqual(@as(usize, 0), stats.batched_pending);

    var reopened = try SessionStore.init(std.testing.allocator, root, "doc:2");
    defer reopened.deinit();
    const value = (try reopened.getParticipantStateCopy(std.testing.allocator, "agent:7")).?;
    defer std.testing.allocator.free(value);
    try std.testing.expectEqualStrings("{\"role\":\"reviewer\"}", value);
}

test "SessionStore.appendOperation persists strong operation envelope and emits watch event" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try SessionStore.init(std.testing.allocator, root, "doc:3");
    defer store.deinit();

    try store.appendOperation(.{
        .actor_id = "agent:alpha",
        .actor_seq = 1,
        .op_id = "op-0001",
        .payload = "{\"insert\":\"h\"}",
        .issued_at_ms = 3_000,
    });
    try store.flush();

    var events = try store.watchDrain(std.testing.allocator, 0, 16);
    defer events.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 1), events.events.len);
    try std.testing.expectEqual(kv_store.WatchEventType.put, events.events[0].event_type);
    try std.testing.expectEqual(@as(?kv_store.DurabilityClass, .strong), events.events[0].durability);
    try std.testing.expectEqualStrings("ops/agent:alpha:1:op-0001", events.events[0].key);

    var reopened = try SessionStore.init(std.testing.allocator, root, "doc:3");
    defer reopened.deinit();
    const value = (try reopened.kv.getCopy(std.testing.allocator, "ops/agent:alpha:1:op-0001")).?;
    defer std.testing.allocator.free(value);
    try std.testing.expectEqualStrings("{\"insert\":\"h\"}", value);
}

//! Resource-scoped collaboration storage on top of DB KV.
//!
//! This is the core owner for the current `/v1/collab/resources/*` semantics.

const std = @import("std");
const kv_store = @import("../db/kv/store.zig");
const types = @import("types.zig");
const session_store = @import("store.zig");

const Allocator = std.mem.Allocator;

pub const ResourceSummary = struct {
    namespace: []const u8,
    meta_json: ?[]u8,
    stats: kv_store.NamespaceStats,

    pub fn deinit(self: *ResourceSummary, allocator: Allocator) void {
        if (self.meta_json) |meta_json| allocator.free(meta_json);
    }
};

pub const BinaryValue = struct {
    data: ?[]u8,
    updated_at_ms: ?i64,

    pub fn deinit(self: *BinaryValue, allocator: Allocator) void {
        if (self.data) |data| allocator.free(data);
    }
};

pub const WatchWaitResult = struct {
    published_seq: u64,
    timed_out: bool,
};

pub const SessionOpenInfo = struct {
    namespace: []const u8,
    participant_id: []const u8,
    participant_kind: types.ParticipantKind,
    status: []const u8,
};

pub const OperationHistoryEntry = struct {
    actor_id: []u8,
    actor_seq: u64,
    op_id: []u8,
    payload: []u8,
    updated_at_ms: i64,
    key: []u8,

    pub fn deinit(self: *OperationHistoryEntry, allocator: Allocator) void {
        allocator.free(self.actor_id);
        allocator.free(self.op_id);
        allocator.free(self.payload);
        allocator.free(self.key);
    }
};

pub const OperationHistoryPage = struct {
    entries: []OperationHistoryEntry,

    pub fn deinit(self: *OperationHistoryPage, allocator: Allocator) void {
        freeOperationHistoryEntries(allocator, self.entries);
    }
};

pub fn freeOperationHistoryEntries(allocator: Allocator, entries: []OperationHistoryEntry) void {
    for (entries) |*entry| entry.deinit(allocator);
    allocator.free(entries);
}

pub const ResourceStore = struct {
    allocator: Allocator,
    resource_kind: []u8,
    resource_id: []u8,
    namespace: []u8,
    mutex: std.Thread.Mutex = .{},
    watch_cond: std.Thread.Condition = .{},
    kv: kv_store.KVStore,

    pub fn init(
        allocator: Allocator,
        db_root: []const u8,
        resource_kind: []const u8,
        resource_id: []const u8,
    ) !ResourceStore {
        try validateResourceComponent(resource_kind);
        try validateResourceId(resource_id);

        const resource_kind_copy = try allocator.dupe(u8, resource_kind);
        errdefer allocator.free(resource_kind_copy);

        const resource_id_copy = try allocator.dupe(u8, resource_id);
        errdefer allocator.free(resource_id_copy);

        const namespace = try namespaceForResourceAlloc(allocator, resource_kind, resource_id);
        errdefer allocator.free(namespace);

        var kv = try kv_store.KVStore.init(allocator, db_root, namespace);
        errdefer kv.deinit();

        return .{
            .allocator = allocator,
            .resource_kind = resource_kind_copy,
            .resource_id = resource_id_copy,
            .namespace = namespace,
            .kv = kv,
        };
    }

    pub fn deinit(self: *ResourceStore) void {
        self.kv.deinit();
        self.allocator.free(self.namespace);
        self.allocator.free(self.resource_id);
        self.allocator.free(self.resource_kind);
    }

    pub fn namespaceId(self: *const ResourceStore) []const u8 {
        return self.namespace;
    }

    pub fn openSession(
        self: *ResourceStore,
        participant_id: []const u8,
        participant_kind: types.ParticipantKind,
        role: ?[]const u8,
    ) !SessionOpenInfo {
        self.mutex.lock();
        defer self.mutex.unlock();
        try validateParticipantId(participant_id);
        if (role) |value| try validateRole(value);

        const meta_json = try jsonStringifyAlloc(self.allocator, .{
            .kind = self.resource_kind,
            .id = self.resource_id,
            .namespace = self.namespace,
            .created_at_ms = nowMs(),
        });
        defer self.allocator.free(meta_json);
        try self.kv.putWithOptions(metaKey, meta_json, .{ .durability = .strong });

        const participant_state = try jsonStringifyAlloc(self.allocator, .{
            .participant_id = participant_id,
            .participant_kind = @tagName(participant_kind),
            .role = role,
            .joined_at_ms = nowMs(),
        });
        defer self.allocator.free(participant_state);

        const participant_key = try scopedKey(self.allocator, "participants", participant_id);
        defer self.allocator.free(participant_key);
        try self.kv.putWithOptions(participant_key, participant_state, .{ .durability = .batched });

        const checkpoint_json = try jsonStringifyAlloc(self.allocator, .{
            .last_joined_participant = participant_id,
            .joined_at_ms = nowMs(),
        });
        defer self.allocator.free(checkpoint_json);
        try self.kv.putWithOptions(sessionKey, checkpoint_json, .{ .durability = .strong });
        self.watch_cond.broadcast();

        return .{
            .namespace = self.namespace,
            .participant_id = participant_id,
            .participant_kind = participant_kind,
            .status = "joined",
        };
    }

    pub fn getSummary(self: *ResourceStore, allocator: Allocator) !ResourceSummary {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .namespace = self.namespace,
            .meta_json = try self.kv.getCopy(allocator, metaKey),
            .stats = self.kv.stats(),
        };
    }

    pub fn getSnapshot(self: *ResourceStore, allocator: Allocator) !BinaryValue {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (try self.kv.getEntryCopy(allocator, snapshotKey)) |entry| {
            return .{
                .data = entry.value,
                .updated_at_ms = entry.updated_at_ms,
            };
        }
        return .{ .data = null, .updated_at_ms = null };
    }

    pub fn submitOperation(
        self: *ResourceStore,
        op: types.OperationEnvelope,
        snapshot: ?[]const u8,
    ) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const op_key = try op.key(self.allocator);
        errdefer self.allocator.free(op_key);

        try self.kv.putWithOptions(op_key, op.payload, .{ .durability = .strong });

        if (snapshot) |snapshot_bytes| {
            try self.kv.putWithOptions(snapshotKey, snapshot_bytes, .{ .durability = .strong });
        }

        const checkpoint_json = try jsonStringifyAlloc(self.allocator, .{
            .actor_id = op.actor_id,
            .actor_seq = op.actor_seq,
            .op_id = op.op_id,
            .issued_at_ms = op.issued_at_ms,
            .accepted_at_ms = nowMs(),
            .op_key = op_key,
        });
        defer self.allocator.free(checkpoint_json);
        try self.kv.putWithOptions(lastOpKey, checkpoint_json, .{ .durability = .strong });
        self.watch_cond.broadcast();

        return op_key;
    }

    /// Persist a synthetic snapshot update for resource owners that already
    /// materialized authoritative bytes outside the collab engine.
    pub fn syncSnapshot(
        self: *ResourceStore,
        actor_id: []const u8,
        actor_kind: types.ParticipantKind,
        role: ?[]const u8,
        op_kind: []const u8,
        snapshot: []const u8,
    ) ![]u8 {
        try validateParticipantId(actor_id);
        if (role) |value| try validateRole(value);
        try validateOpKind(op_kind);

        _ = try self.openSession(actor_id, actor_kind, role);

        const issued_at_ms = nowMs();
        const actor_seq = syntheticActorSeq();
        const op_id = try std.fmt.allocPrint(self.allocator, "{s}-{d}", .{ op_kind, actor_seq });
        defer self.allocator.free(op_id);

        const payload = try jsonStringifyAlloc(self.allocator, .{
            .type = op_kind,
            .resource_kind = self.resource_kind,
            .resource_id = self.resource_id,
            .snapshot_bytes = snapshot.len,
            .issued_at_ms = issued_at_ms,
        });
        defer self.allocator.free(payload);

        return try self.submitOperation(.{
            .actor_id = actor_id,
            .actor_seq = actor_seq,
            .op_id = op_id,
            .payload = payload,
            .issued_at_ms = issued_at_ms,
        }, snapshot);
    }

    pub fn clearSnapshot(
        self: *ResourceStore,
        actor_id: []const u8,
        actor_kind: types.ParticipantKind,
        role: ?[]const u8,
        op_kind: []const u8,
    ) ![]u8 {
        try validateParticipantId(actor_id);
        if (role) |value| try validateRole(value);
        try validateOpKind(op_kind);

        _ = try self.openSession(actor_id, actor_kind, role);

        self.mutex.lock();
        defer self.mutex.unlock();

        _ = try self.kv.delete(snapshotKey);

        const issued_at_ms = nowMs();
        const actor_seq = syntheticActorSeq();
        const op_id = try std.fmt.allocPrint(self.allocator, "{s}-{d}", .{ op_kind, actor_seq });
        defer self.allocator.free(op_id);

        const payload = try jsonStringifyAlloc(self.allocator, .{
            .type = op_kind,
            .resource_kind = self.resource_kind,
            .resource_id = self.resource_id,
            .issued_at_ms = issued_at_ms,
            .snapshot_cleared = true,
        });
        defer self.allocator.free(payload);

        const op = types.OperationEnvelope{
            .actor_id = actor_id,
            .actor_seq = actor_seq,
            .op_id = op_id,
            .payload = payload,
            .issued_at_ms = issued_at_ms,
        };
        const op_key = try op.key(self.allocator);
        errdefer self.allocator.free(op_key);

        try self.kv.putWithOptions(op_key, payload, .{ .durability = .strong });
        const checkpoint_json = try jsonStringifyAlloc(self.allocator, .{
            .actor_id = actor_id,
            .actor_seq = actor_seq,
            .op_id = op_id,
            .issued_at_ms = issued_at_ms,
            .accepted_at_ms = nowMs(),
            .op_key = op_key,
            .snapshot_cleared = true,
        });
        defer self.allocator.free(checkpoint_json);
        try self.kv.putWithOptions(lastOpKey, checkpoint_json, .{ .durability = .strong });
        self.watch_cond.broadcast();

        return op_key;
    }

    pub fn getHistory(
        self: *ResourceStore,
        allocator: Allocator,
        after_key: ?[]const u8,
        limit: usize,
    ) ![]OperationHistoryEntry {
        self.mutex.lock();
        defer self.mutex.unlock();
        const entries = try self.kv.listEntries(allocator);
        defer {
            for (entries) |*entry| entry.deinit(allocator);
            allocator.free(entries);
        }

        var history = std.ArrayListUnmanaged(OperationHistoryEntry){};
        errdefer freeOperationHistoryEntries(allocator, history.items);

        for (entries) |entry| {
            const parsed = parseOpKey(allocator, entry.key) catch continue;
            errdefer parsed.deinit(allocator);

            try history.append(allocator, .{
                .actor_id = parsed.actor_id,
                .actor_seq = parsed.actor_seq,
                .op_id = parsed.op_id,
                .payload = try allocator.dupe(u8, entry.value),
                .updated_at_ms = entry.updated_at_ms,
                .key = try allocator.dupe(u8, entry.key),
            });
        }

        std.mem.sort(OperationHistoryEntry, history.items, {}, lessThanHistoryEntry);

        var start_index: usize = 0;
        if (after_key) |cursor| {
            const idx = findHistoryIndex(history.items, cursor) orelse return error.InvalidArgument;
            start_index = idx + 1;
        }

        if (start_index > 0) {
            for (history.items[0..start_index]) |*entry| entry.deinit(allocator);
            std.mem.copyForwards(
                OperationHistoryEntry,
                history.items[0 .. history.items.len - start_index],
                history.items[start_index..],
            );
            history.items.len -= start_index;
        }

        if (limit > 0 and history.items.len > limit) {
            for (history.items[limit..]) |*entry| entry.deinit(allocator);
            history.items.len = limit;
        }

        return try history.toOwnedSlice(allocator);
    }

    pub fn putPresence(
        self: *ResourceStore,
        participant_id: []const u8,
        payload: []const u8,
        ttl_ms: u64,
    ) !u64 {
        self.mutex.lock();
        defer self.mutex.unlock();
        try validateParticipantId(participant_id);
        const effective_ttl = if (ttl_ms == 0) session_store.default_presence_ttl_ms else ttl_ms;
        const key = try scopedKey(self.allocator, "presence", participant_id);
        defer self.allocator.free(key);
        try self.kv.putWithOptions(key, payload, .{
            // Resource API presence must survive request-local handle churn.
            // SessionStore retains the pure ephemeral primitive for in-process owners.
            .durability = .strong,
            .ttl_ms = effective_ttl,
        });
        self.watch_cond.broadcast();
        return effective_ttl;
    }

    pub fn getPresence(self: *ResourceStore, allocator: Allocator, participant_id: []const u8) !BinaryValue {
        self.mutex.lock();
        defer self.mutex.unlock();
        try validateParticipantId(participant_id);
        const key = try scopedKey(self.allocator, "presence", participant_id);
        defer self.allocator.free(key);
        if (try self.kv.getEntryCopy(allocator, key)) |entry| {
            return .{
                .data = entry.value,
                .updated_at_ms = entry.updated_at_ms,
            };
        }
        return .{ .data = null, .updated_at_ms = null };
    }

    pub fn watchDrain(
        self: *ResourceStore,
        allocator: Allocator,
        after_seq: u64,
        max_count: usize,
    ) !kv_store.WatchDrainResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.kv.watchDrain(allocator, after_seq, max_count);
    }

    pub fn watchWait(
        self: *ResourceStore,
        after_seq: u64,
        timeout_ms: u64,
    ) !WatchWaitResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const deadline_ms = if (timeout_ms == 0) null else nowMs() + @as(i64, @intCast(timeout_ms));

        while (true) {
            const stats = self.kv.stats();
            if (stats.watch_published > after_seq) {
                return .{
                    .published_seq = stats.watch_published,
                    .timed_out = false,
                };
            }

            const wait_ns = computeWaitNsLocked(self, deadline_ms) orelse return .{
                .published_seq = stats.watch_published,
                .timed_out = true,
            };

            self.watch_cond.timedWait(&self.mutex, wait_ns) catch |err| switch (err) {
                error.Timeout => {},
            };
        }
    }

    pub fn flushBatched(self: *ResourceStore) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.kv.flushBatched();
    }

    pub fn setNowMsForTesting(self: *ResourceStore, now_ms: ?i64) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.kv.setNowMsForTesting(now_ms);
    }
};

const metaKey = "resource/meta";
const snapshotKey = "checkpoints/snapshot";
const sessionKey = "checkpoints/session";
const lastOpKey = "checkpoints/last_op";

pub fn namespaceForResourceAlloc(
    allocator: Allocator,
    resource_kind: []const u8,
    resource_id: []const u8,
) ![]u8 {
    try validateResourceComponent(resource_kind);
    try validateResourceId(resource_id);

    var normalized = std.ArrayListUnmanaged(u8){};
    defer normalized.deinit(allocator);

    for (resource_kind) |ch| {
        if (normalized.items.len == 32) break;
        const out: u8 = switch (ch) {
            'a'...'z', '0'...'9', '-', '_' => ch,
            'A'...'Z' => std.ascii.toLower(ch),
            else => '_',
        };
        try normalized.append(allocator, out);
    }

    const hash = fnv1a64(resource_kind, resource_id);
    return std.fmt.allocPrint(allocator, "collab-{s}-{x:0>16}", .{ normalized.items, hash });
}

fn validateResourceComponent(value: []const u8) !void {
    if (value.len == 0) return error.InvalidArgument;
    for (value) |ch| {
        if (ch == 0 or std.ascii.isControl(ch) or ch == '/') return error.InvalidArgument;
    }
}

fn validateResourceId(value: []const u8) !void {
    if (value.len == 0) return error.InvalidArgument;
    for (value) |ch| {
        if (ch == 0 or std.ascii.isControl(ch)) return error.InvalidArgument;
    }
}

fn validateParticipantId(value: []const u8) !void {
    if (value.len == 0) return error.InvalidArgument;
    for (value) |ch| {
        if (ch == 0 or std.ascii.isControl(ch) or ch == '/') return error.InvalidArgument;
    }
}

fn validateRole(value: []const u8) !void {
    for (value) |ch| {
        if (ch == 0 or std.ascii.isControl(ch)) return error.InvalidArgument;
    }
}

fn validateOpKind(value: []const u8) !void {
    if (value.len == 0) return error.InvalidArgument;
    for (value) |ch| {
        if (ch == 0 or std.ascii.isControl(ch) or ch == '/') return error.InvalidArgument;
    }
}

fn scopedKey(allocator: Allocator, prefix: []const u8, id: []const u8) ![]u8 {
    if (id.len == 0) return error.InvalidArgument;
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, id });
}

fn computeWaitNsLocked(self: *ResourceStore, deadline_ms: ?i64) ?u64 {
    const now_ms = nowMs();
    var wait_ms: ?u64 = null;

    if (deadline_ms) |deadline| {
        if (deadline <= now_ms) return null;
        wait_ms = @intCast(deadline - now_ms);
    }

    if (next_expiration_ms(self)) |expires_at_ms| {
        if (expires_at_ms <= now_ms) return 1;
        const until_expiry_ms: u64 = @intCast(expires_at_ms - now_ms);
        wait_ms = if (wait_ms) |current|
            @min(current, until_expiry_ms)
        else
            until_expiry_ms;
    }

    const effective_ms = wait_ms orelse return std.time.ns_per_s;
    return @max(@as(u64, 1), effective_ms) * std.time.ns_per_ms;
}

fn next_expiration_ms(self: *ResourceStore) ?i64 {
    var next: ?i64 = null;
    var iter = self.kv.values.iterator();
    while (iter.next()) |entry| {
        const expires_at_ms = entry.value_ptr.expires_at_ms orelse continue;
        if (next == null or expires_at_ms < next.?) {
            next = expires_at_ms;
        }
    }
    return next;
}

fn nowMs() i64 {
    return std.time.milliTimestamp();
}

fn syntheticActorSeq() u64 {
    const ts = std.time.nanoTimestamp();
    if (ts <= 0) return 1;
    return @intCast(ts);
}

fn fnv1a64(resource_kind: []const u8, resource_id: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (resource_kind) |ch| {
        hash ^= @as(u64, ch);
        hash *%= 0x100000001b3;
    }
    hash ^= @as(u64, '\n');
    hash *%= 0x100000001b3;
    for (resource_id) |ch| {
        hash ^= @as(u64, ch);
        hash *%= 0x100000001b3;
    }
    return hash;
}

fn jsonStringifyAlloc(allocator: Allocator, value: anytype) ![]u8 {
    var out: std.io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(value, .{}, &out.writer);
    return try allocator.dupe(u8, out.written());
}

const ParsedOpKey = struct {
    actor_id: []u8,
    actor_seq: u64,
    op_id: []u8,

    fn deinit(self: *const ParsedOpKey, allocator: Allocator) void {
        allocator.free(self.actor_id);
        allocator.free(self.op_id);
    }
};

fn parseOpKey(allocator: Allocator, key: []const u8) !ParsedOpKey {
    if (!std.mem.startsWith(u8, key, "ops/")) return error.InvalidArgument;
    const rest = key["ops/".len..];
    const last_colon = std.mem.lastIndexOfScalar(u8, rest, ':') orelse return error.InvalidArgument;
    const before_op = rest[0..last_colon];
    const op_id = rest[last_colon + 1 ..];
    const seq_colon = std.mem.lastIndexOfScalar(u8, before_op, ':') orelse return error.InvalidArgument;
    const actor_id = before_op[0..seq_colon];
    const actor_seq = try std.fmt.parseInt(u64, before_op[seq_colon + 1 ..], 10);
    return .{
        .actor_id = try allocator.dupe(u8, actor_id),
        .actor_seq = actor_seq,
        .op_id = try allocator.dupe(u8, op_id),
    };
}

fn lessThanHistoryEntry(_: void, lhs: OperationHistoryEntry, rhs: OperationHistoryEntry) bool {
    if (lhs.updated_at_ms != rhs.updated_at_ms) return lhs.updated_at_ms < rhs.updated_at_ms;
    const actor_cmp = std.mem.order(u8, lhs.actor_id, rhs.actor_id);
    if (actor_cmp != .eq) return actor_cmp == .lt;
    return lhs.actor_seq < rhs.actor_seq;
}

fn findHistoryIndex(entries: []const OperationHistoryEntry, key: []const u8) ?usize {
    for (entries, 0..) |entry, idx| {
        if (std.mem.eql(u8, entry.key, key)) return idx;
    }
    return null;
}

test "namespaceForResourceAlloc is stable and bounded" {
    const namespace = try namespaceForResourceAlloc(std.testing.allocator, "file_buffer", "/tmp/main.zig");
    defer std.testing.allocator.free(namespace);

    try std.testing.expect(std.mem.startsWith(u8, namespace, "collab-file_buffer-"));
    try std.testing.expect(namespace.len <= 64);
}

test "ResourceStore open session, submit op, and history roundtrip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try ResourceStore.init(std.testing.allocator, root, "text_document", "doc-1");
    defer store.deinit();
    store.setNowMsForTesting(1_000);

    const opened = try store.openSession("human:1", .human, "editor");
    try std.testing.expectEqualStrings("joined", opened.status);

    const op_key = try store.submitOperation(.{
        .actor_id = "human:1",
        .actor_seq = 1,
        .op_id = "op-1",
        .payload = "hello",
        .issued_at_ms = 1_001,
    }, "snapshot");
    defer std.testing.allocator.free(op_key);
    try std.testing.expectEqualStrings("ops/human:1:1:op-1", op_key);

    var snapshot = try store.getSnapshot(std.testing.allocator);
    defer snapshot.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("snapshot", snapshot.data.?);

    const history = try store.getHistory(std.testing.allocator, null, 100);
    defer freeOperationHistoryEntries(std.testing.allocator, history);
    try std.testing.expectEqual(@as(usize, 1), history.len);
    try std.testing.expectEqualStrings("human:1", history[0].actor_id);
    try std.testing.expectEqualStrings("op-1", history[0].op_id);
    try std.testing.expectEqualStrings("hello", history[0].payload);
}

test "ResourceStore presence roundtrips and summary returns stats" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try ResourceStore.init(std.testing.allocator, root, "file_buffer", "workdir:/a");
    defer store.deinit();
    store.setNowMsForTesting(2_000);

    const ttl = try store.putPresence("agent:7", "{\"cursor\":2}", 0);
    try std.testing.expectEqual(session_store.default_presence_ttl_ms, ttl);

    var presence = try store.getPresence(std.testing.allocator, "agent:7");
    defer presence.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("{\"cursor\":2}", presence.data.?);

    var summary = try store.getSummary(std.testing.allocator);
    defer summary.deinit(std.testing.allocator);
    try std.testing.expect(summary.stats.total_live_entries >= 1);
}

test "ResourceStore syncSnapshot persists snapshot and op history" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try ResourceStore.init(std.testing.allocator, root, "workdir_file", "notes/main.txt");
    defer store.deinit();
    store.setNowMsForTesting(3_000);

    const op_key = try store.syncSnapshot(
        "system:agent_fs",
        .system,
        "sync",
        "fs_write",
        "hello world",
    );
    defer std.testing.allocator.free(op_key);
    try std.testing.expect(std.mem.startsWith(u8, op_key, "ops/system:agent_fs:"));

    var snapshot = try store.getSnapshot(std.testing.allocator);
    defer snapshot.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("hello world", snapshot.data.?);

    const history = try store.getHistory(std.testing.allocator, null, 8);
    defer freeOperationHistoryEntries(std.testing.allocator, history);
    try std.testing.expectEqual(@as(usize, 1), history.len);
    try std.testing.expectEqualStrings("system:agent_fs", history[0].actor_id);
}

test "ResourceStore getHistory paginates forward from cursor key" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try ResourceStore.init(std.testing.allocator, root, "text_document", "doc-history");
    defer store.deinit();
    store.setNowMsForTesting(4_000);

    const first_key = try store.submitOperation(.{
        .actor_id = "human:1",
        .actor_seq = 1,
        .op_id = "op-1",
        .payload = "first",
        .issued_at_ms = 4_001,
    }, null);
    defer std.testing.allocator.free(first_key);

    const second_key = try store.submitOperation(.{
        .actor_id = "human:1",
        .actor_seq = 2,
        .op_id = "op-2",
        .payload = "second",
        .issued_at_ms = 4_002,
    }, null);
    defer std.testing.allocator.free(second_key);

    const third_key = try store.submitOperation(.{
        .actor_id = "human:1",
        .actor_seq = 3,
        .op_id = "op-3",
        .payload = "third",
        .issued_at_ms = 4_003,
    }, null);
    defer std.testing.allocator.free(third_key);

    const page = try store.getHistory(std.testing.allocator, first_key, 2);
    defer freeOperationHistoryEntries(std.testing.allocator, page);
    try std.testing.expectEqual(@as(usize, 2), page.len);
    try std.testing.expectEqualStrings(second_key, page[0].key);
    try std.testing.expectEqualStrings("ops/human:1:3:op-3", page[1].key);
}

test "ResourceStore clearSnapshot removes current snapshot and records op" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try ResourceStore.init(std.testing.allocator, root, "workdir_file", "notes.txt");
    defer store.deinit();
    store.setNowMsForTesting(5_000);

    const write_key = try store.syncSnapshot("system:agent_fs", .system, "sync", "fs_write", "hello");
    defer std.testing.allocator.free(write_key);
    const clear_key = try store.clearSnapshot("system:agent_fs", .system, "sync", "fs_delete");
    defer std.testing.allocator.free(clear_key);

    var snapshot = try store.getSnapshot(std.testing.allocator);
    defer snapshot.deinit(std.testing.allocator);
    try std.testing.expect(snapshot.data == null);

    const history = try store.getHistory(std.testing.allocator, null, 8);
    defer freeOperationHistoryEntries(std.testing.allocator, history);
    try std.testing.expectEqual(@as(usize, 2), history.len);
    const op_type = try historyOpType(std.testing.allocator, history[1].payload);
    defer std.testing.allocator.free(op_type);
    try std.testing.expectEqualStrings("fs_delete", op_type);
}

fn historyOpType(allocator: Allocator, payload: []const u8) ![]u8 {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, payload, .{});
    defer parsed.deinit();
    const op_type = parsed.value.object.get("type") orelse return error.InvalidArgument;
    return try allocator.dupe(u8, op_type.string);
}

test "ResourceStore watchWait wakes when a new watch event is published" {
    if (@import("builtin").single_threaded) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const allocator = std.heap.c_allocator;
    const root = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(root);

    var store = try ResourceStore.init(allocator, root, "text_document", "doc-watch");
    defer store.deinit();

    const WaitContext = struct {
        store: *ResourceStore,
        ready: *std.Thread.Semaphore,
        done: *std.Thread.Semaphore,
        result: *WatchWaitResult,
        err: *?anyerror,

        fn run(ctx: *@This()) void {
            ctx.ready.post();
            ctx.result.* = ctx.store.watchWait(0, 1_000) catch |err| {
                ctx.err.* = err;
                ctx.done.post();
                return;
            };
            ctx.done.post();
        }
    };

    var ready = std.Thread.Semaphore{};
    var done = std.Thread.Semaphore{};
    var result = WatchWaitResult{ .published_seq = 0, .timed_out = true };
    var err: ?anyerror = null;
    var ctx = WaitContext{
        .store = &store,
        .ready = &ready,
        .done = &done,
        .result = &result,
        .err = &err,
    };

    const thread = try std.Thread.spawn(.{}, WaitContext.run, .{&ctx});
    defer thread.join();

    ready.wait();
    const op_key = try store.submitOperation(.{
        .actor_id = "human:watcher",
        .actor_seq = 1,
        .op_id = "op-1",
        .payload = "hello",
        .issued_at_ms = 1,
    }, "snapshot");
    defer allocator.free(op_key);

    try done.timedWait(1 * std.time.ns_per_s);
    try std.testing.expect(err == null);
    try std.testing.expectEqual(false, result.timed_out);
    try std.testing.expect(result.published_seq >= 1);
}

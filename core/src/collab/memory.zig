//! Process-local in-memory storage for collaboration state.
//!
//! TODO: persistent collab storage should bind to `TALU_DB_HOST`. For now the
//! collab plane is intentionally process-local and in-memory only.

const std = @import("std");

const global_allocator = std.heap.c_allocator;

pub const DurabilityClass = enum(u8) {
    strong = 0,
    batched = 1,
    ephemeral = 2,
};

pub const WatchEventType = enum(u8) {
    put = 0,
    delete = 1,
};

pub const NamespaceStats = struct {
    total_live_entries: usize,
    batched_pending: usize,
    ephemeral_live_entries: usize,
    watch_published: u64,
};

pub const ValueCopy = struct {
    value: []u8,
    updated_at_ms: i64,

    pub fn deinit(self: *ValueCopy, allocator: std.mem.Allocator) void {
        allocator.free(self.value);
    }
};

pub const EntryCopy = struct {
    key: []u8,
    value: []u8,
    updated_at_ms: i64,

    pub fn deinit(self: *EntryCopy, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
        allocator.free(self.value);
    }
};

pub const WatchEvent = struct {
    seq: u64,
    event_type: WatchEventType,
    durability: ?DurabilityClass,
    ttl_ms: ?u64,
    key: []u8,
    value_len: usize,
    updated_at_ms: i64,

    pub fn deinit(self: *WatchEvent, allocator: std.mem.Allocator) void {
        allocator.free(self.key);
    }
};

pub const WatchDrainResult = struct {
    events: []WatchEvent,
    lost: bool,

    pub fn deinit(self: *WatchDrainResult, allocator: std.mem.Allocator) void {
        for (self.events) |*event| event.deinit(allocator);
        allocator.free(self.events);
    }
};

pub const WatchWaitResult = struct {
    published_seq: u64,
    timed_out: bool,
};

pub const PutOptions = struct {
    durability: DurabilityClass = .strong,
    ttl_ms: ?u64 = null,
};

const ValueEntry = struct {
    value: []u8,
    updated_at_ms: i64,
    durability: DurabilityClass,
    ttl_ms: ?u64,
    expires_at_ms: ?i64,
};

const OwnedWatchEvent = struct {
    seq: u64,
    event_type: WatchEventType,
    durability: ?DurabilityClass,
    ttl_ms: ?u64,
    key: []u8,
    value_len: usize,
    updated_at_ms: i64,
};

const SharedState = struct {
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    values: std.StringHashMapUnmanaged(ValueEntry) = .{},
    pending_batched: std.StringHashMapUnmanaged(void) = .{},
    watch_events: std.ArrayListUnmanaged(OwnedWatchEvent) = .{},
    watch_published: u64 = 0,
    now_ms_override: ?i64 = null,
};

const Registry = struct {
    mutex: std.Thread.Mutex = .{},
    states: std.StringHashMapUnmanaged(*SharedState) = .{},
};

var registry_state = Registry{};

pub const NamespaceStore = struct {
    allocator: std.mem.Allocator,
    namespace: []u8,
    shared: *SharedState,

    pub fn init(allocator: std.mem.Allocator, namespace: []const u8) !NamespaceStore {
        return .{
            .allocator = allocator,
            .namespace = try allocator.dupe(u8, namespace),
            .shared = try retainShared(namespace),
        };
    }

    pub fn deinit(self: *NamespaceStore) void {
        self.allocator.free(self.namespace);
    }

    pub fn namespaceId(self: *const NamespaceStore) []const u8 {
        return self.namespace;
    }

    pub fn lock(self: *NamespaceStore) void {
        self.shared.mutex.lock();
    }

    pub fn unlock(self: *NamespaceStore) void {
        self.shared.mutex.unlock();
    }

    pub fn putWithOptions(self: *NamespaceStore, key: []const u8, value: []const u8, options: PutOptions) !void {
        self.lock();
        defer self.unlock();
        try self.putWithOptionsLocked(key, value, options);
    }

    pub fn putWithOptionsLocked(self: *NamespaceStore, key: []const u8, value: []const u8, options: PutOptions) !void {
        try self.sweepExpiredLocked();
        const updated_at_ms = self.nowMsLocked();
        const expires_at_ms = if (options.ttl_ms) |ttl_ms|
            updated_at_ms + @as(i64, @intCast(ttl_ms))
        else
            null;

        const gop = try self.shared.values.getOrPut(global_allocator, key);
        if (!gop.found_existing) {
            gop.key_ptr.* = try global_allocator.dupe(u8, key);
            gop.value_ptr.* = .{
                .value = try global_allocator.dupe(u8, value),
                .updated_at_ms = updated_at_ms,
                .durability = options.durability,
                .ttl_ms = options.ttl_ms,
                .expires_at_ms = expires_at_ms,
            };
        } else {
            global_allocator.free(gop.value_ptr.value);
            gop.value_ptr.* = .{
                .value = try global_allocator.dupe(u8, value),
                .updated_at_ms = updated_at_ms,
                .durability = options.durability,
                .ttl_ms = options.ttl_ms,
                .expires_at_ms = expires_at_ms,
            };
        }

        if (options.durability == .batched) {
            _ = try self.shared.pending_batched.getOrPut(global_allocator, key);
        } else {
            _ = self.shared.pending_batched.remove(key);
        }

        try self.appendWatchEventLocked(.put, key, value.len, options.durability, options.ttl_ms, updated_at_ms);
    }

    pub fn getCopy(self: *NamespaceStore, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        self.lock();
        defer self.unlock();
        return self.getCopyLocked(allocator, key);
    }

    pub fn getCopyLocked(self: *NamespaceStore, allocator: std.mem.Allocator, key: []const u8) !?[]u8 {
        try self.sweepExpiredLocked();
        const entry = self.shared.values.get(key) orelse return null;
        return try allocator.dupe(u8, entry.value);
    }

    pub fn getEntryCopy(self: *NamespaceStore, allocator: std.mem.Allocator, key: []const u8) !?ValueCopy {
        self.lock();
        defer self.unlock();
        return self.getEntryCopyLocked(allocator, key);
    }

    pub fn getEntryCopyLocked(self: *NamespaceStore, allocator: std.mem.Allocator, key: []const u8) !?ValueCopy {
        try self.sweepExpiredLocked();
        const entry = self.shared.values.get(key) orelse return null;
        return .{
            .value = try allocator.dupe(u8, entry.value),
            .updated_at_ms = entry.updated_at_ms,
        };
    }

    pub fn delete(self: *NamespaceStore, key: []const u8) !bool {
        self.lock();
        defer self.unlock();
        return self.deleteLocked(key);
    }

    pub fn deleteLocked(self: *NamespaceStore, key: []const u8) !bool {
        try self.sweepExpiredLocked();
        const removed = self.shared.values.fetchRemove(key) orelse return false;
        _ = self.shared.pending_batched.remove(key);
        global_allocator.free(removed.key);
        global_allocator.free(removed.value.value);
        try self.appendWatchEventLocked(.delete, key, 0, removed.value.durability, null, self.nowMsLocked());
        return true;
    }

    pub fn listEntries(self: *NamespaceStore, allocator: std.mem.Allocator) ![]EntryCopy {
        self.lock();
        defer self.unlock();
        return self.listEntriesLocked(allocator);
    }

    pub fn listEntriesLocked(self: *NamespaceStore, allocator: std.mem.Allocator) ![]EntryCopy {
        try self.sweepExpiredLocked();

        var entries = try allocator.alloc(EntryCopy, self.shared.values.count());
        var count: usize = 0;
        errdefer {
            for (entries[0..count]) |*entry| entry.deinit(allocator);
            allocator.free(entries);
        }

        var iter = self.shared.values.iterator();
        while (iter.next()) |item| {
            entries[count] = .{
                .key = try allocator.dupe(u8, item.key_ptr.*),
                .value = try allocator.dupe(u8, item.value_ptr.value),
                .updated_at_ms = item.value_ptr.updated_at_ms,
            };
            count += 1;
        }
        return entries[0..count];
    }

    pub fn watchDrain(self: *NamespaceStore, allocator: std.mem.Allocator, after_seq: u64, max_count: usize) !WatchDrainResult {
        self.lock();
        defer self.unlock();
        return self.watchDrainLocked(allocator, after_seq, max_count);
    }

    pub fn watchDrainLocked(self: *NamespaceStore, allocator: std.mem.Allocator, after_seq: u64, max_count: usize) !WatchDrainResult {
        try self.sweepExpiredLocked();

        var start: usize = self.shared.watch_events.items.len;
        for (self.shared.watch_events.items, 0..) |event, idx| {
            if (event.seq > after_seq) {
                start = idx;
                break;
            }
        }

        if (start == self.shared.watch_events.items.len) {
            return .{ .events = try allocator.alloc(WatchEvent, 0), .lost = false };
        }

        const remaining = self.shared.watch_events.items.len - start;
        const take = if (max_count == 0) remaining else @min(remaining, max_count);
        var events = try allocator.alloc(WatchEvent, take);
        errdefer {
            for (events[0..take]) |*event| event.deinit(allocator);
            allocator.free(events);
        }

        for (self.shared.watch_events.items[start .. start + take], 0..) |event, idx| {
            events[idx] = .{
                .seq = event.seq,
                .event_type = event.event_type,
                .durability = event.durability,
                .ttl_ms = event.ttl_ms,
                .key = try allocator.dupe(u8, event.key),
                .value_len = event.value_len,
                .updated_at_ms = event.updated_at_ms,
            };
        }

        return .{ .events = events, .lost = false };
    }

    pub fn watchWait(self: *NamespaceStore, after_seq: u64, timeout_ms: u64) !WatchWaitResult {
        self.lock();
        defer self.unlock();

        const deadline_ms = if (timeout_ms == 0) null else self.nowMsLocked() + @as(i64, @intCast(timeout_ms));

        while (true) {
            try self.sweepExpiredLocked();
            if (self.shared.watch_published > after_seq) {
                return .{
                    .published_seq = self.shared.watch_published,
                    .timed_out = false,
                };
            }

            const wait_ns = self.computeWaitNsLocked(deadline_ms) orelse {
                return .{
                    .published_seq = self.shared.watch_published,
                    .timed_out = true,
                };
            };

            self.shared.cond.timedWait(&self.shared.mutex, wait_ns) catch |err| switch (err) {
                error.Timeout => {},
            };
        }
    }

    pub fn flushBatched(self: *NamespaceStore) !void {
        self.lock();
        defer self.unlock();
        self.shared.pending_batched.clearRetainingCapacity();
    }

    pub fn flush(self: *NamespaceStore) !void {
        _ = self;
    }

    pub fn stats(self: *NamespaceStore) NamespaceStats {
        self.lock();
        defer self.unlock();
        return self.statsLocked();
    }

    pub fn statsLocked(self: *NamespaceStore) NamespaceStats {
        const now_ms = self.nowMsLocked();
        var total_live_entries: usize = 0;
        var ephemeral_live_entries: usize = 0;

        var iter = self.shared.values.iterator();
        while (iter.next()) |item| {
            if (item.value_ptr.expires_at_ms) |expires_at_ms| {
                if (expires_at_ms <= now_ms) continue;
            }
            total_live_entries += 1;
            if (item.value_ptr.durability == .ephemeral) {
                ephemeral_live_entries += 1;
            }
        }

        return .{
            .total_live_entries = total_live_entries,
            .batched_pending = self.shared.pending_batched.count(),
            .ephemeral_live_entries = ephemeral_live_entries,
            .watch_published = self.shared.watch_published,
        };
    }

    pub fn setNowMsForTesting(self: *NamespaceStore, now_ms: ?i64) void {
        self.lock();
        defer self.unlock();
        self.shared.now_ms_override = now_ms;
        self.shared.cond.broadcast();
    }

    pub fn nowMsLocked(self: *NamespaceStore) i64 {
        return self.shared.now_ms_override orelse std.time.milliTimestamp();
    }

    fn sweepExpiredLocked(self: *NamespaceStore) !void {
        const now_ms = self.nowMsLocked();
        var expired_keys = std.ArrayListUnmanaged([]u8){};
        defer {
            for (expired_keys.items) |key| global_allocator.free(key);
            expired_keys.deinit(global_allocator);
        }

        var iter = self.shared.values.iterator();
        while (iter.next()) |item| {
            const expires_at_ms = item.value_ptr.expires_at_ms orelse continue;
            if (expires_at_ms <= now_ms) {
                try expired_keys.append(global_allocator, try global_allocator.dupe(u8, item.key_ptr.*));
            }
        }

        for (expired_keys.items) |key| {
            const removed = self.shared.values.fetchRemove(key) orelse continue;
            _ = self.shared.pending_batched.remove(key);
            global_allocator.free(removed.key);
            global_allocator.free(removed.value.value);
            try self.appendWatchEventLocked(.delete, key, 0, removed.value.durability, null, now_ms);
        }
    }

    fn appendWatchEventLocked(
        self: *NamespaceStore,
        event_type: WatchEventType,
        key: []const u8,
        value_len: usize,
        durability: ?DurabilityClass,
        ttl_ms: ?u64,
        updated_at_ms: i64,
    ) !void {
        self.shared.watch_published += 1;
        try self.shared.watch_events.append(global_allocator, .{
            .seq = self.shared.watch_published,
            .event_type = event_type,
            .durability = durability,
            .ttl_ms = ttl_ms,
            .key = try global_allocator.dupe(u8, key),
            .value_len = value_len,
            .updated_at_ms = updated_at_ms,
        });
        self.shared.cond.broadcast();
    }

    fn computeWaitNsLocked(self: *NamespaceStore, deadline_ms: ?i64) ?u64 {
        const now_ms = self.nowMsLocked();
        var wait_ms: ?u64 = null;

        if (deadline_ms) |deadline| {
            if (deadline <= now_ms) return null;
            wait_ms = @intCast(deadline - now_ms);
        }

        if (self.nextExpirationMsLocked()) |expires_at_ms| {
            if (expires_at_ms <= now_ms) return 1;
            const until_expiry_ms: u64 = @intCast(expires_at_ms - now_ms);
            wait_ms = if (wait_ms) |current| @min(current, until_expiry_ms) else until_expiry_ms;
        }

        const effective_ms = wait_ms orelse return std.time.ns_per_s;
        return @max(@as(u64, 1), effective_ms) * std.time.ns_per_ms;
    }

    fn nextExpirationMsLocked(self: *NamespaceStore) ?i64 {
        var next: ?i64 = null;
        var iter = self.shared.values.iterator();
        while (iter.next()) |item| {
            const expires_at_ms = item.value_ptr.expires_at_ms orelse continue;
            if (next == null or expires_at_ms < next.?) {
                next = expires_at_ms;
            }
        }
        return next;
    }
};

fn retainShared(namespace: []const u8) !*SharedState {
    registry_state.mutex.lock();
    defer registry_state.mutex.unlock();

    if (registry_state.states.get(namespace)) |existing| return existing;

    const key_copy = try global_allocator.dupe(u8, namespace);
    errdefer global_allocator.free(key_copy);

    const shared = try global_allocator.create(SharedState);
    errdefer global_allocator.destroy(shared);
    shared.* = .{};

    try registry_state.states.put(global_allocator, key_copy, shared);
    return shared;
}

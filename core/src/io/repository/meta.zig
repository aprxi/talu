//! Repository metadata domain store built on top of generic KV state storage.

const std = @import("std");
const kvbuf = @import("../kvbuf/root.zig");
const kv_store = @import("../../db/kv/store.zig");

const Allocator = std.mem.Allocator;
const KvBufWriter = kvbuf.KvBufWriter;
const KvBufReader = kvbuf.KvBufReader;
const RepoPinFieldIds = kvbuf.RepoPinFieldIds;

pub const PinRecord = struct {
    model_uri: []const u8,
    pinned_at_ms: i64,
    size_bytes: ?u64 = null,
    size_updated_at_ms: ?i64 = null,

    pub fn deinit(self: *PinRecord, allocator: Allocator) void {
        allocator.free(self.model_uri);
    }
};

const EncodedPinPayload = struct {
    model_uri: []const u8,
    pinned_at_ms: ?i64 = null,
    size_bytes: ?u64 = null,
    size_updated_at_ms: ?i64 = null,
};

pub const Store = struct {
    allocator: Allocator,
    kv: kv_store.KVStore,

    pub fn init(allocator: Allocator, db_root: []const u8) !Store {
        return .{
            .allocator = allocator,
            .kv = try kv_store.KVStore.init(allocator, db_root, "repo_meta"),
        };
    }

    pub fn deinit(self: *Store) void {
        self.kv.deinit();
    }

    pub fn pinModel(self: *Store, model_uri: []const u8) !bool {
        if (model_uri.len == 0) return error.InvalidArgument;
        if (self.kv.contains(model_uri)) return false;

        const now_ms = std.time.milliTimestamp();
        const payload = try encodePayload(self.allocator, .{
            .model_uri = model_uri,
            .pinned_at_ms = now_ms,
        });
        defer self.allocator.free(payload);

        try self.kv.put(model_uri, payload);
        return true;
    }

    pub fn unpinModel(self: *Store, model_uri: []const u8) !bool {
        if (model_uri.len == 0) return error.InvalidArgument;
        return self.kv.delete(model_uri);
    }

    pub fn upsertSizeBytes(self: *Store, model_uri: []const u8, size_bytes: u64) !void {
        if (model_uri.len == 0) return error.InvalidArgument;

        const existing_buf = (try self.kv.getCopy(self.allocator, model_uri)) orelse return error.ItemNotFound;
        defer self.allocator.free(existing_buf);

        var decoded = try decodePayload(existing_buf);
        decoded.size_bytes = size_bytes;
        decoded.size_updated_at_ms = std.time.milliTimestamp();

        const payload = try encodePayload(self.allocator, decoded);
        defer self.allocator.free(payload);
        try self.kv.put(model_uri, payload);
    }

    pub fn clearSizeBytes(self: *Store, model_uri: []const u8) !void {
        if (model_uri.len == 0) return error.InvalidArgument;

        const existing_buf = (try self.kv.getCopy(self.allocator, model_uri)) orelse return;
        defer self.allocator.free(existing_buf);

        var decoded = try decodePayload(existing_buf);
        if (decoded.size_bytes == null and decoded.size_updated_at_ms == null) return;

        decoded.size_bytes = null;
        decoded.size_updated_at_ms = null;
        const payload = try encodePayload(self.allocator, decoded);
        defer self.allocator.free(payload);
        try self.kv.put(model_uri, payload);
    }

    pub fn listPins(self: *Store, allocator: Allocator) ![]PinRecord {
        const entries = try self.kv.listEntries(allocator);
        defer kv_store.freeEntryRecords(allocator, entries);

        var out = try allocator.alloc(PinRecord, entries.len);
        errdefer {
            for (out) |*record| {
                if (record.model_uri.len > 0) allocator.free(record.model_uri);
            }
            allocator.free(out);
        }
        @memset(out, .{ .model_uri = "", .pinned_at_ms = 0, .size_bytes = null, .size_updated_at_ms = null });

        for (entries, 0..) |entry, idx| {
            const decoded = try decodePayload(entry.value);
            const chosen_uri = if (decoded.model_uri.len > 0) decoded.model_uri else entry.key;
            out[idx] = .{
                .model_uri = try allocator.dupe(u8, chosen_uri),
                .pinned_at_ms = decoded.pinned_at_ms orelse entry.updated_at_ms,
                .size_bytes = decoded.size_bytes,
                .size_updated_at_ms = decoded.size_updated_at_ms,
            };
        }

        std.sort.pdq(PinRecord, out, {}, lessThanPinRecord);
        return out;
    }

    pub fn compact(self: *Store) !void {
        try self.kv.compact();
    }

    pub fn flush(self: *Store) !void {
        try self.kv.flush();
    }
};

fn lessThanPinRecord(_: void, a: PinRecord, b: PinRecord) bool {
    if (a.pinned_at_ms != b.pinned_at_ms) return a.pinned_at_ms > b.pinned_at_ms;
    return std.mem.lessThan(u8, a.model_uri, b.model_uri);
}

pub fn freePinRecords(allocator: Allocator, records: []PinRecord) void {
    for (records) |*record| {
        record.deinit(allocator);
    }
    allocator.free(records);
}

fn encodePayload(allocator: Allocator, payload: EncodedPinPayload) ![]u8 {
    var writer = KvBufWriter.init();
    defer writer.deinit(allocator);

    try writer.addString(allocator, RepoPinFieldIds.model_uri, payload.model_uri);
    if (payload.pinned_at_ms) |pinned_at_ms| {
        try writer.addI64(allocator, RepoPinFieldIds.pinned_at_ms, pinned_at_ms);
    }
    if (payload.size_bytes) |size_bytes| {
        try writer.addU64(allocator, RepoPinFieldIds.size_bytes, size_bytes);
    }
    if (payload.size_updated_at_ms) |size_updated_at_ms| {
        try writer.addI64(allocator, RepoPinFieldIds.size_updated_at_ms, size_updated_at_ms);
    }
    return writer.finish(allocator);
}

fn decodePayload(payload: []const u8) !EncodedPinPayload {
    const reader = try KvBufReader.init(payload);
    const model_uri = reader.get(RepoPinFieldIds.model_uri) orelse return error.InvalidColumnData;

    return .{
        .model_uri = model_uri,
        .pinned_at_ms = reader.getI64(RepoPinFieldIds.pinned_at_ms),
        .size_bytes = reader.getU64(RepoPinFieldIds.size_bytes),
        .size_updated_at_ms = reader.getI64(RepoPinFieldIds.size_updated_at_ms),
    };
}

test "RepoMeta.Store pin/unpin replay across restart" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    {
        var store = try Store.init(std.testing.allocator, root);
        defer store.deinit();

        try std.testing.expect(try store.pinModel("Qwen/Qwen3-0.6B"));
        try std.testing.expect(!(try store.pinModel("Qwen/Qwen3-0.6B")));
        try std.testing.expect(try store.pinModel("openai/gpt-oss-20b"));
        try std.testing.expect(try store.unpinModel("Qwen/Qwen3-0.6B"));
        try std.testing.expect(!(try store.unpinModel("Qwen/Qwen3-0.6B")));
        try store.flush();
    }

    {
        var store = try Store.init(std.testing.allocator, root);
        defer store.deinit();

        const records = try store.listPins(std.testing.allocator);
        defer freePinRecords(std.testing.allocator, records);

        try std.testing.expectEqual(@as(usize, 1), records.len);
        try std.testing.expectEqualStrings("openai/gpt-oss-20b", records[0].model_uri);
    }
}

test "RepoMeta.Store upsert/clear size" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);

    var store = try Store.init(std.testing.allocator, root);
    defer store.deinit();

    try std.testing.expect(try store.pinModel("Qwen/Qwen3-0.6B"));
    try store.upsertSizeBytes("Qwen/Qwen3-0.6B", 1234);

    {
        const records = try store.listPins(std.testing.allocator);
        defer freePinRecords(std.testing.allocator, records);
        try std.testing.expectEqual(@as(?u64, 1234), records[0].size_bytes);
        try std.testing.expect(records[0].size_updated_at_ms != null);
    }

    try store.clearSizeBytes("Qwen/Qwen3-0.6B");

    {
        const records = try store.listPins(std.testing.allocator);
        defer freePinRecords(std.testing.allocator, records);
        try std.testing.expectEqual(@as(?u64, null), records[0].size_bytes);
        try std.testing.expectEqual(@as(?i64, null), records[0].size_updated_at_ms);
    }
}

//! Agent memory storage backed by DB documents + blob store.
//!
//! Memory files are markdown documents named `YYYYMMDD.md`.
//! Each file is persisted as a blob, with document metadata used for indexing.

const std = @import("std");
const Allocator = std.mem.Allocator;

const db_documents = @import("../../db/table/documents.zig");
const db_blob_store = @import("../../db/blob/store.zig");

const memory_doc_type = "agent_memory";
const memory_marker = "active";

pub const MemoryConfig = struct {
    db_path: []const u8,
    namespace: []const u8 = "default",
    owner_id: ?[]const u8 = null,
};

pub const MemoryError = error{
    InvalidNamespace,
    InvalidFilename,
    InvalidMemoryDocument,
};

pub const RecallHit = struct {
    filename: []const u8,
    snippet: []const u8,
    updated_at_ms: i64,

    pub fn deinit(self: *RecallHit, allocator: Allocator) void {
        allocator.free(self.filename);
        allocator.free(self.snippet);
    }
};

pub const MemoryStore = struct {
    allocator: Allocator,
    db_path: []const u8,
    namespace: []const u8,
    owner_id: ?[]const u8,

    pub fn init(allocator: Allocator, config: MemoryConfig) !MemoryStore {
        if (!isValidNamespace(config.namespace)) return MemoryError.InvalidNamespace;

        return .{
            .allocator = allocator,
            .db_path = try allocator.dupe(u8, config.db_path),
            .namespace = try allocator.dupe(u8, config.namespace),
            .owner_id = if (config.owner_id) |owner| try allocator.dupe(u8, owner) else null,
        };
    }

    pub fn deinit(self: *MemoryStore) void {
        self.allocator.free(self.db_path);
        self.allocator.free(self.namespace);
        if (self.owner_id) |owner| self.allocator.free(owner);
        self.* = undefined;
    }

    /// Persist markdown content for a specific daily file (`YYYYMMDD.md`).
    pub fn upsertMarkdown(self: *MemoryStore, filename: []const u8, markdown: []const u8) !void {
        if (!isDailyFilename(filename)) return MemoryError.InvalidFilename;

        const doc_id = try buildDocId(self.allocator, self.namespace, filename);
        defer self.allocator.free(doc_id);

        var blob_store = try db_blob_store.BlobStore.init(self.allocator, self.db_path);
        defer blob_store.deinit();
        const blob_ref = try blob_store.putAuto(markdown);

        const doc_json = try encodeMemoryDocJson(
            self.allocator,
            filename,
            blob_ref.refSlice(),
        );
        defer self.allocator.free(doc_json);

        const now_ms = std.time.milliTimestamp();
        var created_at_ms = now_ms;
        const existing = try db_documents.getDocument(self.allocator, self.db_path, doc_id);
        if (existing) |record| {
            created_at_ms = record.created_at_ms;
            var rec = record;
            rec.deinit(self.allocator);
        }

        const record = db_documents.DocumentRecord{
            .doc_id = doc_id,
            .doc_type = memory_doc_type,
            .title = filename,
            .tags_text = "memory markdown",
            .doc_json = doc_json,
            .parent_id = null,
            .marker = memory_marker,
            .group_id = self.namespace,
            .owner_id = self.owner_id,
            .created_at_ms = created_at_ms,
            .updated_at_ms = now_ms,
        };
        try db_documents.createDocument(self.allocator, self.db_path, record);
    }

    /// Append a memory entry to the day file computed from `epoch_ms`.
    pub fn appendDaily(self: *MemoryStore, epoch_ms: i64, entry: []const u8) !void {
        const day = buildDailyFilename(epoch_ms);
        const filename = day[0..];

        const current = try self.readMarkdown(filename);
        defer if (current) |buf| self.allocator.free(buf);

        const merged = if (current) |existing| blk: {
            const separator = if (existing.len == 0) "" else "\n\n";
            var out = std.ArrayList(u8).empty;
            errdefer out.deinit(self.allocator);
            try out.appendSlice(self.allocator, existing);
            try out.appendSlice(self.allocator, separator);
            try out.appendSlice(self.allocator, entry);
            break :blk try out.toOwnedSlice(self.allocator);
        } else try self.allocator.dupe(u8, entry);
        defer self.allocator.free(merged);

        try self.upsertMarkdown(filename, merged);
    }

    /// Load markdown content for a day file (`YYYYMMDD.md`).
    /// Caller owns the returned bytes and must free with the store allocator.
    pub fn readMarkdown(self: *MemoryStore, filename: []const u8) !?[]u8 {
        if (!isDailyFilename(filename)) return MemoryError.InvalidFilename;

        const doc_id = try buildDocId(self.allocator, self.namespace, filename);
        defer self.allocator.free(doc_id);

        const record_opt = try db_documents.getDocument(self.allocator, self.db_path, doc_id);
        if (record_opt == null) return null;
        var record = record_opt.?;
        defer record.deinit(self.allocator);

        const blob_ref = try parseBlobRefFromDocJson(self.allocator, record.doc_json);
        defer self.allocator.free(blob_ref);

        var blob_store = try db_blob_store.BlobStore.init(self.allocator, self.db_path);
        defer blob_store.deinit();
        return try blob_store.readAll(blob_ref, self.allocator);
    }

    /// Recall memory snippets by text query.
    ///
    /// Empty query returns newest entries up to `limit`.
    /// Caller owns returned slice and each hit payload.
    pub fn recall(self: *MemoryStore, query: []const u8, limit: usize) ![]RecallHit {
        if (limit == 0) return self.allocator.alloc(RecallHit, 0);

        const records = try db_documents.listDocuments(
            self.allocator,
            self.db_path,
            memory_doc_type,
            self.namespace,
            self.owner_id,
            memory_marker,
        );
        defer db_documents.freeDocumentRecords(self.allocator, records);

        std.mem.sort(db_documents.DocumentRecord, records, {}, lessByUpdatedAtDesc);

        var blob_store = try db_blob_store.BlobStore.init(self.allocator, self.db_path);
        defer blob_store.deinit();

        var hits = std.ArrayList(RecallHit).empty;
        errdefer {
            for (hits.items) |*hit| hit.deinit(self.allocator);
            hits.deinit(self.allocator);
        }

        for (records) |record| {
            if (hits.items.len >= limit) break;

            const blob_ref = parseBlobRefFromDocJson(self.allocator, record.doc_json) catch continue;
            defer self.allocator.free(blob_ref);

            const markdown = blob_store.readAll(blob_ref, self.allocator) catch continue;
            defer self.allocator.free(markdown);

            var match_pos: usize = 0;
            if (query.len > 0) {
                match_pos = textFindInsensitive(markdown, query) orelse continue;
            }

            const snippet = try buildSnippet(
                self.allocator,
                markdown,
                match_pos,
                if (query.len == 0) 0 else query.len,
            );
            errdefer self.allocator.free(snippet);

            try hits.append(self.allocator, .{
                .filename = try self.allocator.dupe(u8, record.title),
                .snippet = snippet,
                .updated_at_ms = record.updated_at_ms,
            });
        }

        return hits.toOwnedSlice(self.allocator);
    }

    /// Free a slice returned by `recall`.
    pub fn freeRecallHits(self: *MemoryStore, hits: []RecallHit) void {
        for (hits) |*hit| hit.deinit(self.allocator);
        self.allocator.free(hits);
    }
};

pub fn buildDailyFilename(epoch_ms: i64) [11]u8 {
    const secs_i64 = @divFloor(epoch_ms, std.time.ms_per_s);
    const secs_u64: u64 = @intCast(@max(@as(i64, 0), secs_i64));
    const epoch_secs = std.time.epoch.EpochSeconds{ .secs = secs_u64 };
    const epoch_day = epoch_secs.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();

    var out: [11]u8 = undefined;
    const day = month_day.day_index + 1;
    _ = std.fmt.bufPrint(&out, "{d:0>4}{d:0>2}{d:0>2}.md", .{
        year_day.year,
        month_day.month.numeric(),
        day,
    }) catch unreachable;
    return out;
}

pub fn isDailyFilename(filename: []const u8) bool {
    if (filename.len != 11) return false;
    if (!std.mem.endsWith(u8, filename, ".md")) return false;
    for (filename[0..8]) |c| {
        if (!std.ascii.isDigit(c)) return false;
    }
    return true;
}

fn lessByUpdatedAtDesc(_: void, lhs: db_documents.DocumentRecord, rhs: db_documents.DocumentRecord) bool {
    return lhs.updated_at_ms > rhs.updated_at_ms;
}

fn buildDocId(allocator: Allocator, namespace: []const u8, filename: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ namespace, filename });
}

fn encodeMemoryDocJson(allocator: Allocator, filename: []const u8, blob_ref: []const u8) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\"format\":\"markdown\",\"filename\":\"{s}\",\"blob_ref\":\"{s}\"}}",
        .{ filename, blob_ref },
    );
}

fn parseBlobRefFromDocJson(allocator: Allocator, doc_json: []const u8) ![]u8 {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, doc_json, .{}) catch {
        return MemoryError.InvalidMemoryDocument;
    };
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return MemoryError.InvalidMemoryDocument,
    };

    const blob_ref = switch (obj.get("blob_ref") orelse return MemoryError.InvalidMemoryDocument) {
        .string => |s| s,
        else => return MemoryError.InvalidMemoryDocument,
    };
    return allocator.dupe(u8, blob_ref);
}

fn isValidNamespace(namespace: []const u8) bool {
    if (namespace.len == 0) return false;
    for (namespace) |c| {
        if (std.ascii.isAlphanumeric(c)) continue;
        if (c == '_' or c == '-' or c == '.') continue;
        return false;
    }
    return true;
}

fn textFindInsensitive(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (haystack.len < needle.len) return null;

    for (0..haystack.len - needle.len + 1) |i| {
        var is_match = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                is_match = false;
                break;
            }
        }
        if (is_match) return i;
    }
    return null;
}

fn buildSnippet(allocator: Allocator, markdown: []const u8, match_pos: usize, query_len: usize) ![]u8 {
    if (markdown.len == 0) return allocator.dupe(u8, "");

    if (query_len == 0) {
        return allocator.dupe(u8, markdown[0..@min(@as(usize, 240), markdown.len)]);
    }

    const lead: usize = 40;
    const max_len: usize = 240;
    const start = if (match_pos > lead) match_pos - lead else 0;
    const end = @min(markdown.len, start + max_len);
    return allocator.dupe(u8, markdown[start..end]);
}

test "buildDailyFilename formats YYYYMMDD.md in UTC" {
    const filename = buildDailyFilename(1771632000000); // 2026-02-21T00:00:00Z
    try std.testing.expectEqualStrings("20260221.md", filename[0..]);
}

test "isDailyFilename validates strict daily format" {
    try std.testing.expect(isDailyFilename("20260221.md"));
    try std.testing.expect(!isDailyFilename("2026-02-21.md"));
    try std.testing.expect(!isDailyFilename("20260221.txt"));
    try std.testing.expect(!isDailyFilename("abc.md"));
}

test "MemoryStore upsert/read/append/recall roundtrip" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const db_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(db_path);

    var store = try MemoryStore.init(allocator, .{
        .db_path = db_path,
        .namespace = "agent",
    });
    defer store.deinit();

    try store.upsertMarkdown("20260221.md", "First memory line");

    const first = try store.readMarkdown("20260221.md");
    try std.testing.expect(first != null);
    defer allocator.free(first.?);
    try std.testing.expectEqualStrings("First memory line", first.?);

    try store.appendDaily(1771632000000, "Second memory line");

    const second = try store.readMarkdown("20260221.md");
    try std.testing.expect(second != null);
    defer allocator.free(second.?);
    try std.testing.expect(std.mem.indexOf(u8, second.?, "First memory line") != null);
    try std.testing.expect(std.mem.indexOf(u8, second.?, "Second memory line") != null);

    const hits = try store.recall("second", 5);
    defer store.freeRecallHits(hits);

    try std.testing.expectEqual(@as(usize, 1), hits.len);
    try std.testing.expectEqualStrings("20260221.md", hits[0].filename);
    try std.testing.expect(std.mem.indexOf(u8, hits[0].snippet, "Second") != null);
}

test "MemoryStore rejects invalid memory filename" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const db_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(db_path);

    var store = try MemoryStore.init(allocator, .{
        .db_path = db_path,
        .namespace = "agent",
    });
    defer store.deinit();

    try std.testing.expectError(
        MemoryError.InvalidFilename,
        store.upsertMarkdown("notes.md", "invalid"),
    );
}

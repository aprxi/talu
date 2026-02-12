//! StoreFS manifest handling for sealed segments.
//!
//! The manifest is the root of trust for immutable segments on disk.

const std = @import("std");
const json = @import("../io/json/root.zig");

const Allocator = std.mem.Allocator;

pub const SegmentEntry = struct {
    id: u128,
    path: []const u8,
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
};

pub const Manifest = struct {
    version: u32,
    segments: []SegmentEntry,
    last_compaction_ts: i64,

    /// Frees memory owned by the manifest.
    pub fn deinit(self: *Manifest, allocator: Allocator) void {
        for (self.segments) |segment| {
            allocator.free(segment.path);
        }
        allocator.free(self.segments);
        self.segments = &.{};
    }

    /// Loads a manifest from disk.
    pub fn load(allocator: Allocator, path: []const u8) !Manifest {
        const file_data = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
        defer allocator.free(file_data);

        var parsed = json.parseValue(allocator, file_data, .{ .max_size_bytes = 1 * 1024 * 1024 }) catch |err| {
            return switch (err) {
                error.InputTooLarge => error.InvalidManifest,
                error.InputTooDeep => error.InvalidManifest,
                error.StringTooLong => error.InvalidManifest,
                error.InvalidJson => error.InvalidManifest,
                error.OutOfMemory => error.OutOfMemory,
            };
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return error.InvalidManifest;

        const obj = root.object;
        const version_value = obj.get("version") orelse return error.InvalidManifest;
        const last_ts_value = obj.get("last_compaction_ts") orelse return error.InvalidManifest;
        const segments_value = obj.get("segments") orelse return error.InvalidManifest;

        const version = try parseU32(version_value);
        const last_compaction_ts = try parseI64(last_ts_value);
        const segments = try parseSegments(allocator, segments_value);

        return .{
            .version = version,
            .segments = segments,
            .last_compaction_ts = last_compaction_ts,
        };
    }

    /// Saves the manifest to disk via atomic rename.
    pub fn save(self: Manifest, path: []const u8) !void {
        var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        defer arena.deinit();

        const seg_json = try buildSegmentJson(&arena, self.segments);
        const json_doc = ManifestJson{
            .version = self.version,
            .segments = seg_json,
            .last_compaction_ts = self.last_compaction_ts,
        };

        const tmp_path = try std.fmt.allocPrint(arena.allocator(), "{s}.tmp", .{path});

        var tmp_file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
        errdefer std.fs.cwd().deleteFile(tmp_path) catch {};

        const json_bytes = try std.json.Stringify.valueAlloc(arena.allocator(), json_doc, .{ .whitespace = .indent_2 });
        try tmp_file.writeAll(json_bytes);
        try tmp_file.sync();
        tmp_file.close();

        try std.fs.cwd().rename(tmp_path, path);
    }
};

const SegmentJson = struct {
    id: []const u8,
    path: []const u8,
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
};

const ManifestJson = struct {
    version: u32,
    segments: []SegmentJson,
    last_compaction_ts: i64,
};

fn buildSegmentJson(arena: *std.heap.ArenaAllocator, segments: []const SegmentEntry) ![]SegmentJson {
    const alloc = arena.allocator();
    const out = try alloc.alloc(SegmentJson, segments.len); // lint:ignore errdefer-alloc - arena freed atomically
    for (segments, 0..) |segment, idx| {
        const id_str = try uuidToHexLower(alloc, segment.id);
        out[idx] = .{
            .id = id_str,
            .path = segment.path,
            .min_ts = segment.min_ts,
            .max_ts = segment.max_ts,
            .row_count = segment.row_count,
        };
    }
    return out;
}

fn parseSegments(allocator: Allocator, value: std.json.Value) ![]SegmentEntry {
    if (value != .array) return error.InvalidManifest;

    const arr = value.array.items;
    if (arr.len == 0) return allocator.alloc(SegmentEntry, 0);

    var segments = try allocator.alloc(SegmentEntry, arr.len);
    errdefer {
        for (segments) |segment| {
            allocator.free(segment.path);
        }
        allocator.free(segments);
    }

    for (arr, 0..) |item, idx| {
        if (item != .object) return error.InvalidManifest;
        const obj = item.object;

        const id_value = obj.get("id") orelse return error.InvalidManifest;
        const path_value = obj.get("path") orelse return error.InvalidManifest;
        const min_ts_value = obj.get("min_ts") orelse return error.InvalidManifest;
        const max_ts_value = obj.get("max_ts") orelse return error.InvalidManifest;
        const row_count_value = obj.get("row_count") orelse return error.InvalidManifest;

        if (path_value != .string) return error.InvalidManifest;

        const id_str = try parseString(id_value);
        const id = try parseUuid(id_str);
        const min_ts = try parseI64(min_ts_value);
        const max_ts = try parseI64(max_ts_value);
        const row_count = try parseU64(row_count_value);

        const path_copy = try allocator.dupe(u8, path_value.string);

        segments[idx] = .{
            .id = id,
            .path = path_copy,
            .min_ts = min_ts,
            .max_ts = max_ts,
            .row_count = row_count,
        };
    }

    return segments;
}

fn parseU32(value: std.json.Value) !u32 {
    if (value != .integer) return error.InvalidManifest;
    if (value.integer < 0 or value.integer > std.math.maxInt(u32)) return error.InvalidManifest;
    return @intCast(value.integer);
}

fn parseU64(value: std.json.Value) !u64 {
    if (value != .integer) return error.InvalidManifest;
    if (value.integer < 0) return error.InvalidManifest;
    return @intCast(value.integer);
}

fn parseI64(value: std.json.Value) !i64 {
    if (value != .integer) return error.InvalidManifest;
    return @intCast(value.integer);
}

fn parseString(value: std.json.Value) ![]const u8 {
    if (value != .string) return error.InvalidManifest;
    return value.string;
}

fn parseUuid(value: []const u8) !u128 {
    var buf: [32]u8 = undefined; // lint:ignore undefined-usage - filled char-by-char in loop below
    var count: usize = 0;

    for (value, 0..) |ch, idx| {
        if (ch == '-') {
            if (!(idx == 8 or idx == 13 or idx == 18 or idx == 23)) return error.InvalidManifest;
            continue;
        }
        if (!std.ascii.isHex(ch)) return error.InvalidManifest;
        if (count >= buf.len) return error.InvalidManifest;
        buf[count] = std.ascii.toLower(ch);
        count += 1;
    }

    if (count != buf.len) return error.InvalidManifest;

    return std.fmt.parseUnsigned(u128, buf[0..], 16) catch return error.InvalidManifest;
}

fn uuidToHexLower(allocator: Allocator, value: u128) ![]const u8 {
    var buf: [32]u8 = undefined;
    const slice = try std.fmt.bufPrint(&buf, "{x:0>32}", .{value});
    return allocator.dupe(u8, slice);
}

// =============================================================================
// Tests
// =============================================================================

test "Manifest.load parses manifest" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const manifest_json =
        "{" ++
        "\"version\":1," ++
        "\"last_compaction_ts\":123," ++
        "\"segments\":[{" ++
        "\"id\":\"0123456789abcdef0123456789abcdef\"," ++
        "\"path\":\"chat/seg-1.talu\"," ++
        "\"min_ts\":-5," ++
        "\"max_ts\":10," ++
        "\"row_count\":42" ++
        "}]" ++
        "}";

    try tmp.dir.writeFile(.{ .sub_path = "manifest.json", .data = manifest_json });

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "manifest.json");
    defer std.testing.allocator.free(path);

    var manifest = try Manifest.load(std.testing.allocator, path);
    defer manifest.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), manifest.version);
    try std.testing.expectEqual(@as(i64, 123), manifest.last_compaction_ts);
    try std.testing.expectEqual(@as(usize, 1), manifest.segments.len);
    try std.testing.expectEqual(@as(u64, 42), manifest.segments[0].row_count);
}

test "Manifest.save writes manifest" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);

    const path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "manifest.json" });
    defer std.testing.allocator.free(path);

    const segments = try std.testing.allocator.alloc(SegmentEntry, 1); // lint:ignore errdefer-alloc - freed via manifest.deinit()
    segments[0] = .{
        .id = 0x0123456789abcdef0123456789abcdef,
        .path = try std.testing.allocator.dupe(u8, "chat/seg-1.talu"),
        .min_ts = 0,
        .max_ts = 100,
        .row_count = 7,
    };

    var manifest = Manifest{
        .version = 1,
        .segments = segments,
        .last_compaction_ts = 99,
    };

    try manifest.save(path);

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "manifest.json", 1024);
    defer std.testing.allocator.free(data);

    try std.testing.expect(std.mem.indexOf(u8, data, "\"segments\"") != null);

    manifest.deinit(std.testing.allocator);
}

test "Manifest.deinit frees segments" {
    const segments = try std.testing.allocator.alloc(SegmentEntry, 1); // lint:ignore errdefer-alloc - freed via manifest.deinit()
    segments[0] = .{
        .id = 1,
        .path = try std.testing.allocator.dupe(u8, "chat/seg-1.talu"),
        .min_ts = 0,
        .max_ts = 0,
        .row_count = 0,
    };

    var manifest = Manifest{
        .version = 1,
        .segments = segments,
        .last_compaction_ts = 0,
    };

    manifest.deinit(std.testing.allocator);
}

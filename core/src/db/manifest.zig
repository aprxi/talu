//! StoreFS manifest handling for sealed segments.
//!
//! The manifest is the root of trust for immutable segments on disk.

const std = @import("std");
const json = @import("../io/json/root.zig");
const checksum = @import("checksum.zig");

const Allocator = std.mem.Allocator;
const max_manifest_bytes: usize = 64 * 1024 * 1024;

pub const SegmentEntry = struct {
    id: u128,
    path: []const u8,
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
    checksum_crc32c: u32 = 0,
    index: ?SegmentIndexMeta = null,
};

pub const SegmentIndexKind = enum {
    flat,
    ivf_flat,
};

pub const SegmentIndexState = enum {
    pending,
    ready,
    failed,
};

pub const SegmentIndexMeta = struct {
    kind: SegmentIndexKind,
    path: []const u8,
    checksum_crc32c: u32 = 0,
    state: SegmentIndexState = .ready,
};

pub const Manifest = struct {
    version: u32,
    generation: u64,
    segments: []SegmentEntry,
    last_compaction_ts: i64,

    /// Frees memory owned by the manifest.
    pub fn deinit(self: *Manifest, allocator: Allocator) void {
        for (self.segments) |segment| {
            allocator.free(segment.path);
            if (segment.index) |index_meta| {
                allocator.free(index_meta.path);
            }
        }
        allocator.free(self.segments);
        self.segments = &.{};
    }

    /// Loads a manifest from disk.
    pub fn load(allocator: Allocator, path: []const u8) !Manifest {
        const file_data = try std.fs.cwd().readFileAlloc(allocator, path, max_manifest_bytes);
        defer allocator.free(file_data);

        var parsed = json.parseValue(allocator, file_data, .{ .max_size_bytes = max_manifest_bytes }) catch |err| {
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
        const generation = if (obj.get("generation")) |generation_value|
            try parseU64(generation_value)
        else
            0;
        const last_ts_value = obj.get("last_compaction_ts") orelse return error.InvalidManifest;
        const segments_value = obj.get("segments") orelse return error.InvalidManifest;

        const version = try parseU32(version_value);
        const last_compaction_ts = try parseI64(last_ts_value);
        const segments = try parseSegments(allocator, segments_value);

        return .{
            .version = version,
            .generation = generation,
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
            .generation = self.generation,
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
        try syncParentDir(path);
    }

    /// Save manifest iff current on-disk generation equals expected_generation.
    /// On success, writes next generation and returns it.
    ///
    /// Note: caller should serialize manifest writers (for example via namespace lock).
    pub fn saveNextGeneration(self: Manifest, allocator: Allocator, path: []const u8, expected_generation: u64) !u64 {
        const current_generation = blk: {
            var current = Manifest.load(allocator, path) catch |err| switch (err) {
                error.FileNotFound => break :blk 0,
                else => return err,
            };
            defer current.deinit(allocator);
            break :blk current.generation;
        };

        if (current_generation != expected_generation) return error.ManifestGenerationConflict;
        if (expected_generation == std.math.maxInt(u64)) return error.ManifestGenerationOverflow;

        var next = self;
        next.generation = expected_generation + 1;
        try next.save(path);
        return next.generation;
    }
};

pub const ManifestPointer = struct {
    generation: u64,
    manifest_path: []u8,
    manifest_crc32c: u32,

    pub fn deinit(self: *ManifestPointer, allocator: Allocator) void {
        allocator.free(self.manifest_path);
        self.manifest_path = &[_]u8{};
    }
};

const ManifestPointerJson = struct {
    version: u32,
    generation: u64,
    manifest_path: []const u8,
    manifest_crc32c: u32,
};

const SegmentJson = struct {
    id: []const u8,
    path: []const u8,
    min_ts: i64,
    max_ts: i64,
    row_count: u64,
    checksum_crc32c: u32,
    index: ?SegmentIndexJson = null,
};

const SegmentIndexJson = struct {
    kind: []const u8,
    path: []const u8,
    checksum_crc32c: u32,
    state: ?[]const u8 = null,
};

const ManifestJson = struct {
    version: u32,
    generation: u64,
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
            .checksum_crc32c = segment.checksum_crc32c,
            .index = if (segment.index) |index_meta|
                .{
                    .kind = segmentIndexKindToString(index_meta.kind),
                    .path = index_meta.path,
                    .checksum_crc32c = index_meta.checksum_crc32c,
                    .state = segmentIndexStateToString(index_meta.state),
                }
            else
                null,
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
            if (segment.index) |index_meta| {
                allocator.free(index_meta.path);
            }
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
        const checksum_crc32c = if (obj.get("checksum_crc32c")) |checksum_value|
            try parseU32(checksum_value)
        else
            0;

        if (path_value != .string) return error.InvalidManifest;

        const id_str = try parseString(id_value);
        const id = try parseUuid(id_str);
        const min_ts = try parseI64(min_ts_value);
        const max_ts = try parseI64(max_ts_value);
        const row_count = try parseU64(row_count_value);
        const index_meta = if (obj.get("index")) |index_value|
            try parseSegmentIndex(allocator, index_value)
        else
            null;

        const path_copy = try allocator.dupe(u8, path_value.string);

        segments[idx] = .{
            .id = id,
            .path = path_copy,
            .min_ts = min_ts,
            .max_ts = max_ts,
            .row_count = row_count,
            .checksum_crc32c = checksum_crc32c,
            .index = index_meta,
        };
    }

    return segments;
}

fn parseSegmentIndex(allocator: Allocator, value: std.json.Value) !?SegmentIndexMeta {
    if (value == .null) return null;
    if (value != .object) return error.InvalidManifest;
    const obj = value.object;

    const kind_value = obj.get("kind") orelse return error.InvalidManifest;
    const path_value = obj.get("path") orelse return error.InvalidManifest;
    const checksum_value = obj.get("checksum_crc32c") orelse return error.InvalidManifest;

    const kind_str = try parseString(kind_value);
    const path_str = try parseString(path_value);
    const parsed_checksum = try parseU32(checksum_value);

    return .{
        .kind = try parseSegmentIndexKind(kind_str),
        .path = try allocator.dupe(u8, path_str),
        .checksum_crc32c = parsed_checksum,
        .state = if (obj.get("state")) |state_value|
            try parseSegmentIndexState(try parseString(state_value))
        else
            .ready,
    };
}

fn parseSegmentIndexKind(value: []const u8) !SegmentIndexKind {
    if (std.mem.eql(u8, value, "flat")) return .flat;
    if (std.mem.eql(u8, value, "ivf_flat")) return .ivf_flat;
    return error.InvalidManifest;
}

fn segmentIndexKindToString(kind: SegmentIndexKind) []const u8 {
    return switch (kind) {
        .flat => "flat",
        .ivf_flat => "ivf_flat",
    };
}

fn parseSegmentIndexState(value: []const u8) !SegmentIndexState {
    if (std.mem.eql(u8, value, "pending")) return .pending;
    if (std.mem.eql(u8, value, "ready")) return .ready;
    if (std.mem.eql(u8, value, "failed")) return .failed;
    return error.InvalidManifest;
}

fn segmentIndexStateToString(state: SegmentIndexState) []const u8 {
    return switch (state) {
        .pending => "pending",
        .ready => "ready",
        .failed => "failed",
    };
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

fn syncParentDir(path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse ".";
    var dir = try std.fs.cwd().openDir(parent, .{});
    defer dir.close();
    if (comptime @hasDecl(std.posix, "fsync")) {
        try std.posix.fsync(dir.fd);
    }
}

/// Loads a manifest pointer from disk.
/// Returns `error.FileNotFound` when the pointer does not exist.
pub fn loadPointer(allocator: Allocator, path: []const u8) !ManifestPointer {
    const bytes = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(ManifestPointerJson, allocator, bytes, .{});
    defer parsed.deinit();

    if (parsed.value.version != 1) return error.InvalidManifestPointer;
    return .{
        .generation = parsed.value.generation,
        .manifest_path = try allocator.dupe(u8, parsed.value.manifest_path),
        .manifest_crc32c = parsed.value.manifest_crc32c,
    };
}

/// Saves a manifest pointer via atomic rename.
pub fn savePointer(self: ManifestPointer, path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse ".";
    try std.fs.cwd().makePath(parent);

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const tmp_path = try std.fmt.allocPrint(arena.allocator(), "{s}.tmp", .{path});
    var tmp_file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
    errdefer std.fs.cwd().deleteFile(tmp_path) catch {};

    const json_doc = ManifestPointerJson{
        .version = 1,
        .generation = self.generation,
        .manifest_path = self.manifest_path,
        .manifest_crc32c = self.manifest_crc32c,
    };
    const json_bytes = try std.json.Stringify.valueAlloc(arena.allocator(), json_doc, .{ .whitespace = .indent_2 });
    try tmp_file.writeAll(json_bytes);
    try tmp_file.sync();
    tmp_file.close();

    try std.fs.cwd().rename(tmp_path, path);
    try syncParentDir(path);
}

/// Publish immutable manifest bytes, then atomically advance the pointer.
///
/// - `objects_root` is the immutable object namespace root.
/// - `pointer_path` is the atomically-swapped pointer file.
/// - `manifest_object_rel_path` is the object key under `objects_root`.
pub fn publishManifestObject(
    allocator: Allocator,
    objects_root: []const u8,
    pointer_path: []const u8,
    manifest_object_rel_path: []const u8,
    manifest_bytes: []const u8,
    expected_generation: u64,
) !ManifestPointer {
    const current_generation = blk: {
        var current = loadPointer(allocator, pointer_path) catch |err| switch (err) {
            error.FileNotFound => break :blk 0,
            else => return err,
        };
        defer current.deinit(allocator);
        break :blk current.generation;
    };
    if (current_generation != expected_generation) return error.ManifestGenerationConflict;
    if (expected_generation == std.math.maxInt(u64)) return error.ManifestGenerationOverflow;

    const object_path = try std.fs.path.join(allocator, &.{ objects_root, manifest_object_rel_path });
    defer allocator.free(object_path);
    const object_parent = std.fs.path.dirname(object_path) orelse ".";
    try std.fs.cwd().makePath(object_parent);

    var create = std.fs.cwd().createFile(object_path, .{ .read = true, .truncate = false, .exclusive = true }) catch |err| switch (err) {
        error.PathAlreadyExists => null,
        else => return err,
    };
    if (create) |*file| {
        defer file.close();
        try file.writeAll(manifest_bytes);
        try file.sync();
    } else {
        const existing = try std.fs.cwd().readFileAlloc(allocator, object_path, max_manifest_bytes);
        defer allocator.free(existing);
        if (!std.mem.eql(u8, existing, manifest_bytes)) return error.AlreadyExists;
    }

    const next_pointer = ManifestPointer{
        .generation = expected_generation + 1,
        .manifest_path = try allocator.dupe(u8, manifest_object_rel_path),
        .manifest_crc32c = checksum.crc32c(manifest_bytes),
    };
    errdefer allocator.free(next_pointer.manifest_path);
    try savePointer(next_pointer, pointer_path);
    return next_pointer;
}

/// Loads a manifest via pointer and verifies object checksum before parse.
pub fn loadFromPointer(allocator: Allocator, objects_root: []const u8, pointer_path: []const u8) !Manifest {
    var pointer = try loadPointer(allocator, pointer_path);
    defer pointer.deinit(allocator);

    const object_path = try std.fs.path.join(allocator, &.{ objects_root, pointer.manifest_path });
    defer allocator.free(object_path);

    const bytes = try std.fs.cwd().readFileAlloc(allocator, object_path, max_manifest_bytes);
    defer allocator.free(bytes);
    if (checksum.crc32c(bytes) != pointer.manifest_crc32c) return error.InvalidManifest;

    var parsed = json.parseValue(allocator, bytes, .{ .max_size_bytes = max_manifest_bytes }) catch |err| {
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
    const generation_value = obj.get("generation");
    const last_ts_value = obj.get("last_compaction_ts") orelse return error.InvalidManifest;
    const segments_value = obj.get("segments") orelse return error.InvalidManifest;

    return .{
        .version = try parseU32(version_value),
        .generation = if (generation_value) |v| try parseU64(v) else 0,
        .segments = try parseSegments(allocator, segments_value),
        .last_compaction_ts = try parseI64(last_ts_value),
    };
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
        "\"row_count\":42," ++
        "\"checksum_crc32c\":999" ++
        "}]" ++
        "}";

    try tmp.dir.writeFile(.{ .sub_path = "manifest.json", .data = manifest_json });

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "manifest.json");
    defer std.testing.allocator.free(path);

    var manifest = try Manifest.load(std.testing.allocator, path);
    defer manifest.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, 1), manifest.version);
    try std.testing.expectEqual(@as(u64, 0), manifest.generation);
    try std.testing.expectEqual(@as(i64, 123), manifest.last_compaction_ts);
    try std.testing.expectEqual(@as(usize, 1), manifest.segments.len);
    try std.testing.expectEqual(@as(u64, 42), manifest.segments[0].row_count);
    try std.testing.expectEqual(@as(u32, 999), manifest.segments[0].checksum_crc32c);
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
        .checksum_crc32c = 42,
    };

    var manifest = Manifest{
        .version = 1,
        .generation = 7,
        .segments = segments,
        .last_compaction_ts = 99,
    };

    try manifest.save(path);

    const data = try tmp.dir.readFileAlloc(std.testing.allocator, "manifest.json", 1024);
    defer std.testing.allocator.free(data);

    try std.testing.expect(std.mem.indexOf(u8, data, "\"segments\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, data, "\"checksum_crc32c\": 42") != null);

    manifest.deinit(std.testing.allocator);
}

test "Manifest.load and Manifest.save preserve segment index metadata" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "manifest-index.json" });
    defer std.testing.allocator.free(path);

    const segments = try std.testing.allocator.alloc(SegmentEntry, 1); // lint:ignore errdefer-alloc - freed via manifest.deinit()
    segments[0] = .{
        .id = 0x0fedcba9876543210fedcba987654321,
        .path = try std.testing.allocator.dupe(u8, "vector/seg-2.talu"),
        .min_ts = 10,
        .max_ts = 20,
        .row_count = 3,
        .checksum_crc32c = 777,
        .index = .{
            .kind = .ivf_flat,
            .path = try std.testing.allocator.dupe(u8, "vector/seg-2.ivf"),
            .checksum_crc32c = 1234,
            .state = .failed,
        },
    };

    var before = Manifest{
        .version = 1,
        .generation = 2,
        .segments = segments,
        .last_compaction_ts = 22,
    };
    defer before.deinit(std.testing.allocator);
    try before.save(path);

    var loaded = try Manifest.load(std.testing.allocator, path);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), loaded.segments.len);
    try std.testing.expect(loaded.segments[0].index != null);
    const index_meta = loaded.segments[0].index.?;
    try std.testing.expectEqual(SegmentIndexKind.ivf_flat, index_meta.kind);
    try std.testing.expectEqualStrings("vector/seg-2.ivf", index_meta.path);
    try std.testing.expectEqual(@as(u32, 1234), index_meta.checksum_crc32c);
    try std.testing.expectEqual(SegmentIndexState.failed, index_meta.state);
    try std.testing.expectEqual(@as(u32, 777), loaded.segments[0].checksum_crc32c);
}

test "Manifest.load defaults segment index state to ready when state field is missing" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const manifest_json =
        "{" ++
        "\"version\":1," ++
        "\"generation\":2," ++
        "\"last_compaction_ts\":0," ++
        "\"segments\":[{" ++
        "\"id\":\"0123456789abcdef0123456789abcdef\"," ++
        "\"path\":\"vector/seg-1.talu\"," ++
        "\"min_ts\":0," ++
        "\"max_ts\":0," ++
        "\"row_count\":1," ++
        "\"checksum_crc32c\":11," ++
        "\"index\":{" ++
        "\"kind\":\"ivf_flat\"," ++
        "\"path\":\"vector/seg-1.ivf\"," ++
        "\"checksum_crc32c\":22" ++
        "}" ++
        "}]" ++
        "}";

    try tmp.dir.writeFile(.{ .sub_path = "manifest-legacy-index.json", .data = manifest_json });
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "manifest-legacy-index.json");
    defer std.testing.allocator.free(path);

    var loaded = try Manifest.load(std.testing.allocator, path);
    defer loaded.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), loaded.segments.len);
    try std.testing.expect(loaded.segments[0].index != null);
    try std.testing.expectEqual(SegmentIndexState.ready, loaded.segments[0].index.?.state);
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
        .generation = 0,
        .segments = segments,
        .last_compaction_ts = 0,
    };

    manifest.deinit(std.testing.allocator);
}

test "Manifest.load accepts manifest larger than 1MB" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const padding_len = 1024 * 1024 + 64;
    const padding = try std.testing.allocator.alloc(u8, padding_len);
    defer std.testing.allocator.free(padding);
    @memset(padding, ' ');

    const manifest_json = try std.fmt.allocPrint(
        std.testing.allocator,
        "{{\"version\":1,\"last_compaction_ts\":0,\"segments\":[{s}]}}",
        .{padding},
    );
    defer std.testing.allocator.free(manifest_json);

    try tmp.dir.writeFile(.{ .sub_path = "manifest-large.json", .data = manifest_json });

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "manifest-large.json");
    defer std.testing.allocator.free(path);

    var manifest = try Manifest.load(std.testing.allocator, path);
    defer manifest.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), manifest.version);
    try std.testing.expectEqual(@as(u64, 0), manifest.generation);
    try std.testing.expectEqual(@as(usize, 0), manifest.segments.len);
}

test "Manifest.saveNextGeneration increments and enforces expected generation" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root_path = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root_path);
    const path = try std.fs.path.join(std.testing.allocator, &.{ root_path, "manifest.json" });
    defer std.testing.allocator.free(path);

    // First publish from empty state expects generation 0.
    var m1 = Manifest{
        .version = 1,
        .generation = 0,
        .segments = try std.testing.allocator.alloc(SegmentEntry, 0),
        .last_compaction_ts = 0,
    };
    defer m1.deinit(std.testing.allocator);
    const gen1 = try m1.saveNextGeneration(std.testing.allocator, path, 0);
    try std.testing.expectEqual(@as(u64, 1), gen1);

    // Second publish must use expected generation 1.
    var m2 = Manifest{
        .version = 1,
        .generation = 0,
        .segments = try std.testing.allocator.alloc(SegmentEntry, 0),
        .last_compaction_ts = 5,
    };
    defer m2.deinit(std.testing.allocator);
    const gen2 = try m2.saveNextGeneration(std.testing.allocator, path, 1);
    try std.testing.expectEqual(@as(u64, 2), gen2);

    try std.testing.expectError(
        error.ManifestGenerationConflict,
        m2.saveNextGeneration(std.testing.allocator, path, 1),
    );
}

test "Manifest.publishManifestObject and Manifest.loadFromPointer round-trip" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const objects_root = try std.fs.path.join(std.testing.allocator, &.{ root, "objects" });
    defer std.testing.allocator.free(objects_root);
    const pointer_path = try std.fs.path.join(std.testing.allocator, &.{ root, "manifest.current.json" });
    defer std.testing.allocator.free(pointer_path);

    const manifest_bytes =
        "{" ++
        "\"version\":1," ++
        "\"generation\":1," ++
        "\"last_compaction_ts\":0," ++
        "\"segments\":[]" ++
        "}";

    var pointer = try publishManifestObject(
        std.testing.allocator,
        objects_root,
        pointer_path,
        "manifest/000001.json",
        manifest_bytes,
        0,
    );
    defer pointer.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 1), pointer.generation);
    try std.testing.expectEqualStrings("manifest/000001.json", pointer.manifest_path);

    var loaded = try loadFromPointer(std.testing.allocator, objects_root, pointer_path);
    defer loaded.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 1), loaded.version);
    try std.testing.expectEqual(@as(u64, 1), loaded.generation);
}

test "Manifest.publishManifestObject enforces generation CAS" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const objects_root = try std.fs.path.join(std.testing.allocator, &.{ root, "objects" });
    defer std.testing.allocator.free(objects_root);
    const pointer_path = try std.fs.path.join(std.testing.allocator, &.{ root, "manifest.current.json" });
    defer std.testing.allocator.free(pointer_path);

    var first = try publishManifestObject(
        std.testing.allocator,
        objects_root,
        pointer_path,
        "manifest/000001.json",
        "{\"version\":1,\"generation\":1,\"last_compaction_ts\":0,\"segments\":[]}",
        0,
    );
    defer first.deinit(std.testing.allocator);

    try std.testing.expectError(
        error.ManifestGenerationConflict,
        publishManifestObject(
            std.testing.allocator,
            objects_root,
            pointer_path,
            "manifest/000002.json",
            "{\"version\":1,\"generation\":2,\"last_compaction_ts\":0,\"segments\":[]}",
            0,
        ),
    );
}

test "Manifest.loadFromPointer rejects checksum mismatches" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(root);
    const objects_root = try std.fs.path.join(std.testing.allocator, &.{ root, "objects" });
    defer std.testing.allocator.free(objects_root);
    const pointer_path = try std.fs.path.join(std.testing.allocator, &.{ root, "manifest.current.json" });
    defer std.testing.allocator.free(pointer_path);

    var pointer = try publishManifestObject(
        std.testing.allocator,
        objects_root,
        pointer_path,
        "manifest/000001.json",
        "{\"version\":1,\"generation\":1,\"last_compaction_ts\":0,\"segments\":[]}",
        0,
    );
    defer pointer.deinit(std.testing.allocator);

    // Corrupt immutable object bytes after publish.
    const object_path = try std.fs.path.join(std.testing.allocator, &.{ objects_root, "manifest/000001.json" });
    defer std.testing.allocator.free(object_path);
    var object_file = try std.fs.cwd().createFile(object_path, .{ .truncate = true });
    defer object_file.close();
    try object_file.writeAll("corrupt");

    try std.testing.expectError(
        error.InvalidManifest,
        loadFromPointer(std.testing.allocator, objects_root, pointer_path),
    );
}

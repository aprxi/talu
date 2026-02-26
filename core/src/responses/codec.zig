//! Serialization/deserialization for session records.
//!
//! MsgPack and KvBuf codecs, session hashing, ScannedSessionRecord type.
//! No dependencies on TableAdapter or scan logic.

const std = @import("std");
const kvbuf = @import("../io/kvbuf/root.zig");
const responses = @import("root.zig");

const Allocator = std.mem.Allocator;
const SessionRecord = responses.backend.SessionRecord;

// ============================================================================
// Hashing
// ============================================================================

pub fn computeSessionHash(session_id: []const u8) u64 {
    return std.hash.Wyhash.hash(0, session_id);
}

/// Compute group_id hash using Wyhash, consistent with session hashes.
pub const computeGroupHash = computeSessionHash;

pub fn computeOptionalHash(value: ?[]const u8) u64 {
    const slice = value orelse return 0;
    return computeSessionHash(slice);
}

/// Legacy SipHash 2-4 used by schema versions 2 and 4.
/// Only needed for backward-compat reads of pre-migration data.
pub fn computeLegacySipHash(data: []const u8) u64 {
    const siphash = std.hash.SipHash64(2, 4);
    const key: [16]u8 = .{
        0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b,
        0x0c, 0x0d, 0x0e, 0x0f,
    };
    var ctx = siphash.init(&key);
    ctx.update(data);
    var out: [8]u8 = undefined;
    ctx.final(&out);
    return std.mem.readInt(u64, &out, .little);
}

// ============================================================================
// ScannedSessionRecord
// ============================================================================

/// Session record returned by scanSessions. Owns allocated string fields.
pub const ScannedSessionRecord = struct {
    session_id: []const u8,
    model: ?[]const u8,
    title: ?[]const u8,
    system_prompt: ?[]const u8,
    config_json: ?[]const u8,
    marker: ?[]const u8,
    parent_session_id: ?[]const u8,
    group_id: ?[]const u8,
    head_item_id: u64,
    ttl_ts: i64,
    metadata_json: ?[]const u8,
    /// Text snippet around the matched search query in item content.
    /// Null when no search query or when matched by metadata only.
    search_snippet: ?[]const u8 = null,
    /// Source document ID for lineage tracking.
    /// Links this session to the prompt/persona document that spawned it.
    source_doc_id: ?[]const u8 = null,
    /// Project identifier for organizing sessions into projects.
    project_id: ?[]const u8 = null,
    created_at_ms: i64,
    updated_at_ms: i64,
};

/// Free a ScannedSessionRecord's allocated fields.
pub fn freeScannedSessionRecord(alloc: Allocator, record: *ScannedSessionRecord) void {
    if (record.session_id.len > 0) alloc.free(record.session_id);
    if (record.model) |m| alloc.free(m);
    if (record.title) |t| alloc.free(t);
    if (record.system_prompt) |s| alloc.free(s);
    if (record.config_json) |c| alloc.free(c);
    if (record.marker) |s| alloc.free(s);
    if (record.parent_session_id) |p| alloc.free(p);
    if (record.group_id) |g| alloc.free(g);
    if (record.metadata_json) |m| alloc.free(m);
    if (record.search_snippet) |s| alloc.free(s);
    if (record.source_doc_id) |d| alloc.free(d);
    if (record.project_id) |p| alloc.free(p);
    record.* = .{
        .session_id = "",
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .search_snippet = null,
        .source_doc_id = null,
        .project_id = null,
        .created_at_ms = 0,
        .updated_at_ms = 0,
    };
}

/// Free an array of ScannedSessionRecords.
pub fn freeScannedSessionRecords(alloc: Allocator, records: []ScannedSessionRecord) void {
    for (records) |*record| {
        freeScannedSessionRecord(alloc, record);
    }
    alloc.free(records);
}

// ============================================================================
// MsgPack Encoding
// ============================================================================

pub fn encodeSessionRecordMsgpack(allocator: Allocator, record: SessionRecord) ![]u8 {
    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);
    const writer = buffer.writer(allocator);

    const field_count: u32 = 14;
    try writeMsgpackMapHeader(writer, field_count);

    try writeMsgpackString(writer, "session_id");
    try writeMsgpackString(writer, record.session_id);

    try writeMsgpackString(writer, "model");
    try writeMsgpackOptionalString(writer, record.model);

    try writeMsgpackString(writer, "title");
    try writeMsgpackOptionalString(writer, record.title);

    try writeMsgpackString(writer, "system_prompt");
    try writeMsgpackOptionalString(writer, record.system_prompt);

    try writeMsgpackString(writer, "config_json");
    try writeMsgpackOptionalString(writer, record.config_json);

    try writeMsgpackString(writer, "marker");
    try writeMsgpackOptionalString(writer, record.marker);

    try writeMsgpackString(writer, "parent_session_id");
    try writeMsgpackOptionalString(writer, record.parent_session_id);

    try writeMsgpackString(writer, "group_id");
    try writeMsgpackOptionalString(writer, record.group_id);

    try writeMsgpackString(writer, "head_item_id");
    try writeMsgpackU64(writer, record.head_item_id);

    try writeMsgpackString(writer, "ttl_ts");
    try writeMsgpackI64(writer, record.ttl_ts);

    try writeMsgpackString(writer, "metadata_json");
    try writeMsgpackOptionalString(writer, record.metadata_json);

    try writeMsgpackString(writer, "created_at_ms");
    try writeMsgpackI64(writer, record.created_at_ms);

    try writeMsgpackString(writer, "updated_at_ms");
    try writeMsgpackI64(writer, record.updated_at_ms);

    try writeMsgpackString(writer, "project_id");
    try writeMsgpackOptionalString(writer, record.project_id);

    return buffer.toOwnedSlice(allocator);
}

/// Encode a SessionRecord as KvBuf (Schema 5).
pub fn encodeSessionRecordKvBuf(allocator: Allocator, record: SessionRecord) ![]u8 {
    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    const SFIds = kvbuf.SessionFieldIds;

    try w.addString(allocator, SFIds.session_id, record.session_id);
    try w.addOptionalString(allocator, SFIds.title, record.title);
    try w.addOptionalString(allocator, SFIds.model, record.model);
    try w.addOptionalString(allocator, SFIds.system_prompt, record.system_prompt);
    try w.addOptionalString(allocator, SFIds.config_json, record.config_json);
    try w.addOptionalString(allocator, SFIds.marker, record.marker);
    try w.addOptionalString(allocator, SFIds.parent_session_id, record.parent_session_id);
    try w.addOptionalString(allocator, SFIds.group_id, record.group_id);
    try w.addU64(allocator, SFIds.head_item_id, record.head_item_id);
    try w.addI64(allocator, SFIds.ttl_ts, record.ttl_ts);
    try w.addOptionalString(allocator, SFIds.metadata_json, record.metadata_json);
    try w.addI64(allocator, SFIds.created_at_ms, record.created_at_ms);
    try w.addI64(allocator, SFIds.updated_at_ms, record.updated_at_ms);
    try w.addOptionalString(allocator, SFIds.source_doc_id, record.source_doc_id);
    try w.addOptionalString(allocator, SFIds.project_id, record.project_id);

    return w.finish(allocator);
}

// ============================================================================
// MsgPack Decoding
// ============================================================================

/// Decode a MsgPack-encoded SessionRecord.
/// Caller owns the returned record and must free string fields with freeScannedSessionRecord.
pub fn decodeSessionRecordMsgpack(alloc: Allocator, payload: []const u8) !ScannedSessionRecord {
    var record = ScannedSessionRecord{
        .session_id = "",
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 0,
        .updated_at_ms = 0,
    };
    errdefer freeScannedSessionRecord(alloc, &record);

    var index: usize = 0;
    const map_count = try readMsgpackMapHeader(payload, &index);

    var i: u32 = 0;
    while (i < map_count) : (i += 1) {
        const key = try readMsgpackString(payload, &index);

        if (std.mem.eql(u8, key, "session_id")) {
            record.session_id = try readMsgpackStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "model")) {
            record.model = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "title")) {
            record.title = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "system_prompt")) {
            record.system_prompt = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "config_json")) {
            record.config_json = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "marker") or std.mem.eql(u8, key, "status")) {
            record.marker = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "parent_session_id")) {
            record.parent_session_id = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "group_id")) {
            record.group_id = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "head_item_id")) {
            record.head_item_id = try readMsgpackU64(payload, &index);
        } else if (std.mem.eql(u8, key, "ttl_ts")) {
            record.ttl_ts = try readMsgpackI64(payload, &index);
        } else if (std.mem.eql(u8, key, "metadata_json")) {
            record.metadata_json = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else if (std.mem.eql(u8, key, "created_at_ms")) {
            record.created_at_ms = try readMsgpackI64(payload, &index);
        } else if (std.mem.eql(u8, key, "updated_at_ms")) {
            record.updated_at_ms = try readMsgpackI64(payload, &index);
        } else if (std.mem.eql(u8, key, "project_id")) {
            record.project_id = try readMsgpackOptionalStringAlloc(alloc, payload, &index);
        } else {
            try skipMsgpackValue(payload, &index);
        }
    }

    return record;
}

/// Decode a KvBuf-encoded SessionRecord (Schema 5).
/// Caller owns the returned record and must free string fields with freeScannedSessionRecord.
pub fn decodeSessionRecordKvBuf(alloc: Allocator, payload: []const u8) !ScannedSessionRecord {
    const reader = kvbuf.KvBufReader.init(payload) catch return error.InvalidPayload;
    const SFIds = kvbuf.SessionFieldIds;

    var record = ScannedSessionRecord{
        .session_id = "",
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .source_doc_id = null,
        .created_at_ms = 0,
        .updated_at_ms = 0,
    };
    errdefer freeScannedSessionRecord(alloc, &record);

    // Required field
    const session_id_slice = reader.get(SFIds.session_id) orelse return error.InvalidPayload;
    record.session_id = try alloc.dupe(u8, session_id_slice);

    // Optional string fields
    if (reader.get(SFIds.title)) |s| record.title = try alloc.dupe(u8, s);
    if (reader.get(SFIds.model)) |s| record.model = try alloc.dupe(u8, s);
    if (reader.get(SFIds.system_prompt)) |s| record.system_prompt = try alloc.dupe(u8, s);
    if (reader.get(SFIds.config_json)) |s| record.config_json = try alloc.dupe(u8, s);
    if (reader.get(SFIds.marker)) |s| record.marker = try alloc.dupe(u8, s);
    if (reader.get(SFIds.parent_session_id)) |s| record.parent_session_id = try alloc.dupe(u8, s);
    if (reader.get(SFIds.group_id)) |s| record.group_id = try alloc.dupe(u8, s);
    if (reader.get(SFIds.metadata_json)) |s| record.metadata_json = try alloc.dupe(u8, s);
    if (reader.get(SFIds.source_doc_id)) |s| record.source_doc_id = try alloc.dupe(u8, s);
    if (reader.get(SFIds.project_id)) |s| record.project_id = try alloc.dupe(u8, s);

    // Integer fields
    record.head_item_id = reader.getU64(SFIds.head_item_id) orelse 0;
    record.ttl_ts = reader.getI64(SFIds.ttl_ts) orelse 0;
    record.created_at_ms = reader.getI64(SFIds.created_at_ms) orelse 0;
    record.updated_at_ms = reader.getI64(SFIds.updated_at_ms) orelse 0;

    return record;
}

// ============================================================================
// MsgPack Primitives
// ============================================================================

fn writeMsgpackMapHeader(writer: anytype, count: u32) !void {
    if (count <= 15) {
        try writer.writeByte(@as(u8, 0x80) | @as(u8, @intCast(count)));
        return;
    }
    if (count <= 0xFFFF) {
        try writer.writeByte(0xde);
        var buf: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf, @intCast(count), .big);
        try writer.writeAll(&buf);
        return;
    }
    try writer.writeByte(0xdf);
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, count, .big);
    try writer.writeAll(&buf);
}

fn writeMsgpackString(writer: anytype, value: []const u8) !void {
    const len = value.len;
    if (len <= 31) {
        try writer.writeByte(@as(u8, 0xa0) | @as(u8, @intCast(len)));
    } else if (len <= 0xFF) {
        try writer.writeByte(0xd9);
        try writer.writeByte(@intCast(len));
    } else if (len <= 0xFFFF) {
        try writer.writeByte(0xda);
        var buf16: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf16, @intCast(len), .big);
        try writer.writeAll(&buf16);
    } else {
        try writer.writeByte(0xdb);
        var buf32: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf32, @intCast(len), .big);
        try writer.writeAll(&buf32);
    }
    try writer.writeAll(value);
}

fn writeMsgpackOptionalString(writer: anytype, value: ?[]const u8) !void {
    if (value) |slice| {
        try writeMsgpackString(writer, slice);
    } else {
        try writer.writeByte(0xc0);
    }
}

fn writeMsgpackU64(writer: anytype, value: u64) !void {
    if (value <= 0x7f) {
        try writer.writeByte(@intCast(value));
    } else if (value <= 0xFF) {
        try writer.writeByte(0xcc);
        try writer.writeByte(@intCast(value));
    } else if (value <= 0xFFFF) {
        try writer.writeByte(0xcd);
        var buf16: [2]u8 = undefined;
        std.mem.writeInt(u16, &buf16, @intCast(value), .big);
        try writer.writeAll(&buf16);
    } else if (value <= 0xFFFF_FFFF) {
        try writer.writeByte(0xce);
        var buf32: [4]u8 = undefined;
        std.mem.writeInt(u32, &buf32, @intCast(value), .big);
        try writer.writeAll(&buf32);
    } else {
        try writer.writeByte(0xcf);
        var buf64: [8]u8 = undefined;
        std.mem.writeInt(u64, &buf64, value, .big);
        try writer.writeAll(&buf64);
    }
}

fn writeMsgpackI64(writer: anytype, value: i64) !void {
    if (value >= 0) {
        return writeMsgpackU64(writer, @intCast(value));
    }
    if (value >= -32) {
        const adjust: u8 = @intCast(value + 32);
        try writer.writeByte(@as(u8, 0xe0) + adjust);
        return;
    }
    if (value >= -128) {
        try writer.writeByte(0xd0);
        try writer.writeByte(@bitCast(@as(i8, @intCast(value))));
        return;
    }
    if (value >= -32_768) {
        try writer.writeByte(0xd1);
        var buf16: [2]u8 = undefined;
        std.mem.writeInt(i16, &buf16, @intCast(value), .big);
        try writer.writeAll(&buf16);
        return;
    }
    if (value >= -2_147_483_648) {
        try writer.writeByte(0xd2);
        var buf32: [4]u8 = undefined;
        std.mem.writeInt(i32, &buf32, @intCast(value), .big);
        try writer.writeAll(&buf32);
        return;
    }
    try writer.writeByte(0xd3);
    var buf64: [8]u8 = undefined;
    std.mem.writeInt(i64, &buf64, value, .big);
    try writer.writeAll(&buf64);
}

/// Read MsgPack map header, returns number of key-value pairs.
fn readMsgpackMapHeader(data: []const u8, index: *usize) !u32 {
    if (index.* >= data.len) return error.InvalidMsgpack;
    const byte = data[index.*];
    index.* += 1;

    if (byte >= 0x80 and byte <= 0x8f) {
        return @as(u32, byte & 0x0f);
    }
    if (byte == 0xde) {
        if (index.* + 2 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u16, data[index.*..][0..2], .big);
        index.* += 2;
        return @as(u32, value);
    }
    if (byte == 0xdf) {
        if (index.* + 4 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u32, data[index.*..][0..4], .big);
        index.* += 4;
        return value;
    }
    return error.InvalidMsgpack;
}

/// Read MsgPack string, returns a slice borrowing from input data.
fn readMsgpackString(data: []const u8, index: *usize) ![]const u8 {
    if (index.* >= data.len) return error.InvalidMsgpack;
    const byte = data[index.*];
    index.* += 1;

    var len: usize = 0;
    if (byte >= 0xa0 and byte <= 0xbf) {
        len = @as(usize, byte & 0x1f);
    } else if (byte == 0xd9) {
        if (index.* >= data.len) return error.InvalidMsgpack;
        len = @as(usize, data[index.*]);
        index.* += 1;
    } else if (byte == 0xda) {
        if (index.* + 2 > data.len) return error.InvalidMsgpack;
        len = @as(usize, std.mem.readInt(u16, data[index.*..][0..2], .big));
        index.* += 2;
    } else if (byte == 0xdb) {
        if (index.* + 4 > data.len) return error.InvalidMsgpack;
        len = @as(usize, std.mem.readInt(u32, data[index.*..][0..4], .big));
        index.* += 4;
    } else {
        return error.InvalidMsgpack;
    }

    if (index.* + len > data.len) return error.InvalidMsgpack;
    const slice = data[index.* .. index.* + len];
    index.* += len;
    return slice;
}

/// Read MsgPack string and allocate a copy. Caller owns the memory.
fn readMsgpackStringAlloc(alloc: Allocator, data: []const u8, index: *usize) ![]const u8 {
    const slice = try readMsgpackString(data, index);
    return try alloc.dupe(u8, slice);
}

/// Read optional MsgPack string, returns null if MsgPack nil.
fn readMsgpackOptionalStringAlloc(alloc: Allocator, data: []const u8, index: *usize) !?[]const u8 {
    if (index.* >= data.len) return error.InvalidMsgpack;
    if (data[index.*] == 0xc0) {
        index.* += 1;
        return null;
    }
    return try readMsgpackStringAlloc(alloc, data, index);
}

/// Read MsgPack unsigned integer (u64).
fn readMsgpackU64(data: []const u8, index: *usize) !u64 {
    if (index.* >= data.len) return error.InvalidMsgpack;
    const byte = data[index.*];
    index.* += 1;

    // Positive fixint (0x00-0x7f)
    if (byte <= 0x7f) {
        return @as(u64, byte);
    }
    // uint8 (0xcc)
    if (byte == 0xcc) {
        if (index.* >= data.len) return error.InvalidMsgpack;
        const value = data[index.*];
        index.* += 1;
        return @as(u64, value);
    }
    // uint16 (0xcd)
    if (byte == 0xcd) {
        if (index.* + 2 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u16, data[index.*..][0..2], .big);
        index.* += 2;
        return @as(u64, value);
    }
    // uint32 (0xce)
    if (byte == 0xce) {
        if (index.* + 4 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u32, data[index.*..][0..4], .big);
        index.* += 4;
        return @as(u64, value);
    }
    // uint64 (0xcf)
    if (byte == 0xcf) {
        if (index.* + 8 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u64, data[index.*..][0..8], .big);
        index.* += 8;
        return value;
    }
    return error.InvalidMsgpack;
}

/// Read MsgPack signed integer (i64).
fn readMsgpackI64(data: []const u8, index: *usize) !i64 {
    if (index.* >= data.len) return error.InvalidMsgpack;
    const byte = data[index.*];
    index.* += 1;

    // Positive fixint (0x00-0x7f)
    if (byte <= 0x7f) {
        return @as(i64, byte);
    }
    // Negative fixint (0xe0-0xff)
    if (byte >= 0xe0) {
        return @as(i64, @as(i8, @bitCast(byte)));
    }
    // int8 (0xd0)
    if (byte == 0xd0) {
        if (index.* >= data.len) return error.InvalidMsgpack;
        const value: i8 = @bitCast(data[index.*]);
        index.* += 1;
        return @as(i64, value);
    }
    // int16 (0xd1)
    if (byte == 0xd1) {
        if (index.* + 2 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(i16, data[index.*..][0..2], .big);
        index.* += 2;
        return @as(i64, value);
    }
    // int32 (0xd2)
    if (byte == 0xd2) {
        if (index.* + 4 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(i32, data[index.*..][0..4], .big);
        index.* += 4;
        return @as(i64, value);
    }
    // int64 (0xd3)
    if (byte == 0xd3) {
        if (index.* + 8 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(i64, data[index.*..][0..8], .big);
        index.* += 8;
        return value;
    }
    // Also handle unsigned types for positive values stored as unsigned
    if (byte == 0xcc) {
        if (index.* >= data.len) return error.InvalidMsgpack;
        const value = data[index.*];
        index.* += 1;
        return @as(i64, value);
    }
    if (byte == 0xcd) {
        if (index.* + 2 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u16, data[index.*..][0..2], .big);
        index.* += 2;
        return @as(i64, value);
    }
    if (byte == 0xce) {
        if (index.* + 4 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u32, data[index.*..][0..4], .big);
        index.* += 4;
        return @as(i64, value);
    }
    if (byte == 0xcf) {
        if (index.* + 8 > data.len) return error.InvalidMsgpack;
        const value = std.mem.readInt(u64, data[index.*..][0..8], .big);
        index.* += 8;
        // Check overflow
        if (value > @as(u64, @intCast(std.math.maxInt(i64)))) return error.InvalidMsgpack;
        return @as(i64, @intCast(value));
    }
    return error.InvalidMsgpack;
}

/// Skip over a MsgPack value (used to ignore unknown fields).
fn skipMsgpackValue(data: []const u8, index: *usize) !void {
    if (index.* >= data.len) return error.InvalidMsgpack;
    const byte = data[index.*];
    index.* += 1;

    // nil
    if (byte == 0xc0) return;
    // false, true
    if (byte == 0xc2 or byte == 0xc3) return;
    // positive fixint
    if (byte <= 0x7f) return;
    // negative fixint
    if (byte >= 0xe0) return;
    // fixstr
    if (byte >= 0xa0 and byte <= 0xbf) {
        const len = @as(usize, byte & 0x1f);
        if (index.* + len > data.len) return error.InvalidMsgpack;
        index.* += len;
        return;
    }
    // fixmap
    if (byte >= 0x80 and byte <= 0x8f) {
        const count = @as(usize, byte & 0x0f);
        var i: usize = 0;
        while (i < count * 2) : (i += 1) {
            try skipMsgpackValue(data, index);
        }
        return;
    }
    // fixarray
    if (byte >= 0x90 and byte <= 0x9f) {
        const count = @as(usize, byte & 0x0f);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            try skipMsgpackValue(data, index);
        }
        return;
    }
    // Sized types
    switch (byte) {
        0xc4 => { // bin8
            if (index.* >= data.len) return error.InvalidMsgpack;
            const len = @as(usize, data[index.*]);
            index.* += 1;
            if (index.* + len > data.len) return error.InvalidMsgpack;
            index.* += len;
        },
        0xc5 => { // bin16
            if (index.* + 2 > data.len) return error.InvalidMsgpack;
            const len = @as(usize, std.mem.readInt(u16, data[index.*..][0..2], .big));
            index.* += 2;
            if (index.* + len > data.len) return error.InvalidMsgpack;
            index.* += len;
        },
        0xc6 => { // bin32
            if (index.* + 4 > data.len) return error.InvalidMsgpack;
            const len = @as(usize, std.mem.readInt(u32, data[index.*..][0..4], .big));
            index.* += 4;
            if (index.* + len > data.len) return error.InvalidMsgpack;
            index.* += len;
        },
        0xca => index.* += 4, // float32
        0xcb => index.* += 8, // float64
        0xcc => index.* += 1, // uint8
        0xcd => index.* += 2, // uint16
        0xce => index.* += 4, // uint32
        0xcf => index.* += 8, // uint64
        0xd0 => index.* += 1, // int8
        0xd1 => index.* += 2, // int16
        0xd2 => index.* += 4, // int32
        0xd3 => index.* += 8, // int64
        0xd9 => { // str8
            if (index.* >= data.len) return error.InvalidMsgpack;
            const len = @as(usize, data[index.*]);
            index.* += 1;
            if (index.* + len > data.len) return error.InvalidMsgpack;
            index.* += len;
        },
        0xda => { // str16
            if (index.* + 2 > data.len) return error.InvalidMsgpack;
            const len = @as(usize, std.mem.readInt(u16, data[index.*..][0..2], .big));
            index.* += 2;
            if (index.* + len > data.len) return error.InvalidMsgpack;
            index.* += len;
        },
        0xdb => { // str32
            if (index.* + 4 > data.len) return error.InvalidMsgpack;
            const len = @as(usize, std.mem.readInt(u32, data[index.*..][0..4], .big));
            index.* += 4;
            if (index.* + len > data.len) return error.InvalidMsgpack;
            index.* += len;
        },
        0xdc => { // array16
            if (index.* + 2 > data.len) return error.InvalidMsgpack;
            const count = @as(usize, std.mem.readInt(u16, data[index.*..][0..2], .big));
            index.* += 2;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                try skipMsgpackValue(data, index);
            }
        },
        0xdd => { // array32
            if (index.* + 4 > data.len) return error.InvalidMsgpack;
            const count = @as(usize, std.mem.readInt(u32, data[index.*..][0..4], .big));
            index.* += 4;
            var i: usize = 0;
            while (i < count) : (i += 1) {
                try skipMsgpackValue(data, index);
            }
        },
        0xde => { // map16
            if (index.* + 2 > data.len) return error.InvalidMsgpack;
            const count = @as(usize, std.mem.readInt(u16, data[index.*..][0..2], .big));
            index.* += 2;
            var i: usize = 0;
            while (i < count * 2) : (i += 1) {
                try skipMsgpackValue(data, index);
            }
        },
        0xdf => { // map32
            if (index.* + 4 > data.len) return error.InvalidMsgpack;
            const count = @as(usize, std.mem.readInt(u32, data[index.*..][0..4], .big));
            index.* += 4;
            var i: usize = 0;
            while (i < count * 2) : (i += 1) {
                try skipMsgpackValue(data, index);
            }
        },
        else => return error.InvalidMsgpack,
    }
    if (index.* > data.len) return error.InvalidMsgpack;
}

// =============================================================================
// Tests
// =============================================================================

test "decodeSessionRecordMsgpack round-trips with encode" {
    // Encode a session record
    const original = SessionRecord{
        .session_id = "test-session-123",
        .model = "model-abc",
        .title = "Test Title",
        .system_prompt = null,
        .config_json = "{\"temp\":0.7}",
        .marker = "active",
        .parent_session_id = null,
        .group_id = "grp",
        .head_item_id = 42,
        .ttl_ts = 1000,
        .metadata_json = null,
        .created_at_ms = 123456,
        .updated_at_ms = 789012,
    };

    const encoded = try encodeSessionRecordMsgpack(std.testing.allocator, original);
    defer std.testing.allocator.free(encoded);

    // Decode it
    var decoded = try decodeSessionRecordMsgpack(std.testing.allocator, encoded);
    defer freeScannedSessionRecord(std.testing.allocator, &decoded);

    // Verify fields match
    try std.testing.expectEqualStrings("test-session-123", decoded.session_id);
    try std.testing.expectEqualStrings("model-abc", decoded.model.?);
    try std.testing.expectEqualStrings("Test Title", decoded.title.?);
    try std.testing.expect(decoded.system_prompt == null);
    try std.testing.expectEqualStrings("{\"temp\":0.7}", decoded.config_json.?);
    try std.testing.expectEqualStrings("active", decoded.marker.?);
    try std.testing.expect(decoded.parent_session_id == null);
    try std.testing.expectEqualStrings("grp", decoded.group_id.?);
    try std.testing.expectEqual(@as(u64, 42), decoded.head_item_id);
    try std.testing.expectEqual(@as(i64, 1000), decoded.ttl_ts);
    try std.testing.expect(decoded.metadata_json == null);
    try std.testing.expectEqual(@as(i64, 123456), decoded.created_at_ms);
    try std.testing.expectEqual(@as(i64, 789012), decoded.updated_at_ms);
}

test "encodeSessionRecordKvBuf round-trips with decode" {
    const alloc = std.testing.allocator;

    const original = SessionRecord{
        .session_id = "kvbuf-test-123",
        .model = "model-abc",
        .title = "Test Title",
        .system_prompt = "Be helpful",
        .config_json = "{\"temp\":0.7}",
        .marker = "active",
        .parent_session_id = "parent-1",
        .group_id = "grp",
        .head_item_id = 42,
        .ttl_ts = 1000,
        .metadata_json = "{\"custom\":1}",
        .created_at_ms = 123456,
        .updated_at_ms = 789012,
    };

    const encoded = try encodeSessionRecordKvBuf(alloc, original);
    defer alloc.free(encoded);

    // Verify it's KvBuf format
    try std.testing.expect(kvbuf.isKvBuf(encoded));

    // Decode it
    var decoded = try decodeSessionRecordKvBuf(alloc, encoded);
    defer freeScannedSessionRecord(alloc, &decoded);

    // Verify all fields
    try std.testing.expectEqualStrings("kvbuf-test-123", decoded.session_id);
    try std.testing.expectEqualStrings("model-abc", decoded.model.?);
    try std.testing.expectEqualStrings("Test Title", decoded.title.?);
    try std.testing.expectEqualStrings("Be helpful", decoded.system_prompt.?);
    try std.testing.expectEqualStrings("{\"temp\":0.7}", decoded.config_json.?);
    try std.testing.expectEqualStrings("active", decoded.marker.?);
    try std.testing.expectEqualStrings("parent-1", decoded.parent_session_id.?);
    try std.testing.expectEqualStrings("grp", decoded.group_id.?);
    try std.testing.expectEqual(@as(u64, 42), decoded.head_item_id);
    try std.testing.expectEqual(@as(i64, 1000), decoded.ttl_ts);
    try std.testing.expectEqualStrings("{\"custom\":1}", decoded.metadata_json.?);
    try std.testing.expectEqual(@as(i64, 123456), decoded.created_at_ms);
    try std.testing.expectEqual(@as(i64, 789012), decoded.updated_at_ms);
}

test "encodeSessionRecordKvBuf with null tags" {
    const alloc = std.testing.allocator;

    const original = SessionRecord{
        .session_id = "no-tags-test",
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 100,
        .updated_at_ms = 200,
    };

    const encoded = try encodeSessionRecordKvBuf(alloc, original);
    defer alloc.free(encoded);

    var decoded = try decodeSessionRecordKvBuf(alloc, encoded);
    defer freeScannedSessionRecord(alloc, &decoded);

    try std.testing.expectEqualStrings("no-tags-test", decoded.session_id);
}

// ============================================================================
// computeSessionHash unit tests
// ============================================================================

test "computeSessionHash is deterministic" {
    const h1 = computeSessionHash("test-session-123");
    const h2 = computeSessionHash("test-session-123");
    try std.testing.expectEqual(h1, h2);
}

test "computeSessionHash differs for different inputs" {
    const h1 = computeSessionHash("session-a");
    const h2 = computeSessionHash("session-b");
    try std.testing.expect(h1 != h2);
}

test "computeSessionHash handles empty string" {
    const h = computeSessionHash("");
    // Just verify it returns a value without crashing
    try std.testing.expect(h == computeSessionHash("") );
}

// ============================================================================
// computeOptionalHash unit tests
// ============================================================================

test "computeOptionalHash returns 0 for null" {
    try std.testing.expectEqual(@as(u64, 0), computeOptionalHash(null));
}

test "computeOptionalHash delegates to computeSessionHash for non-null" {
    const expected = computeSessionHash("my-session");
    try std.testing.expectEqual(expected, computeOptionalHash("my-session"));
}

// ============================================================================
// freeScannedSessionRecord unit tests
// ============================================================================

test "freeScannedSessionRecord frees all fields and zeroes struct" {
    const alloc = std.testing.allocator;

    var record = ScannedSessionRecord{
        .session_id = try alloc.dupe(u8, "sess-1"),
        .model = try alloc.dupe(u8, "model-x"),
        .title = try alloc.dupe(u8, "My Title"),
        .system_prompt = try alloc.dupe(u8, "Be helpful"),
        .config_json = try alloc.dupe(u8, "{}"),
        .marker = try alloc.dupe(u8, "active"),
        .parent_session_id = try alloc.dupe(u8, "parent-1"),
        .group_id = try alloc.dupe(u8, "grp-1"),
        .head_item_id = 42,
        .ttl_ts = 1000,
        .metadata_json = try alloc.dupe(u8, "{\"k\":1}"),
        .search_snippet = try alloc.dupe(u8, "snippet here"),
        .source_doc_id = try alloc.dupe(u8, "doc-1"),
        .created_at_ms = 100,
        .updated_at_ms = 200,
    };

    freeScannedSessionRecord(alloc, &record);

    // All fields should be zeroed/null
    try std.testing.expectEqualStrings("", record.session_id);
    try std.testing.expect(record.model == null);
    try std.testing.expect(record.title == null);
    try std.testing.expect(record.system_prompt == null);
    try std.testing.expect(record.config_json == null);
    try std.testing.expect(record.marker == null);
    try std.testing.expect(record.parent_session_id == null);
    try std.testing.expect(record.group_id == null);
    try std.testing.expectEqual(@as(u64, 0), record.head_item_id);
    try std.testing.expectEqual(@as(i64, 0), record.ttl_ts);
    try std.testing.expect(record.metadata_json == null);
    try std.testing.expect(record.search_snippet == null);
    try std.testing.expect(record.source_doc_id == null);
}

test "freeScannedSessionRecord handles all-null optional fields" {
    const alloc = std.testing.allocator;

    var record = ScannedSessionRecord{
        .session_id = try alloc.dupe(u8, "sess-2"),
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 0,
        .updated_at_ms = 0,
    };

    // Should not crash when optional fields are null
    freeScannedSessionRecord(alloc, &record);
    try std.testing.expectEqualStrings("", record.session_id);
}

// ============================================================================
// freeScannedSessionRecords unit tests
// ============================================================================

test "freeScannedSessionRecords frees array of records" {
    const alloc = std.testing.allocator;

    var records = try alloc.alloc(ScannedSessionRecord, 2);
    errdefer alloc.free(records);

    records[0] = .{
        .session_id = try alloc.dupe(u8, "sess-a"),
        .model = try alloc.dupe(u8, "model-1"),
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 100,
        .updated_at_ms = 200,
    };
    records[1] = .{
        .session_id = try alloc.dupe(u8, "sess-b"),
        .model = null,
        .title = try alloc.dupe(u8, "Title B"),
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 300,
        .updated_at_ms = 400,
    };

    // Should free both records and the slice without leaking
    freeScannedSessionRecords(alloc, records);
}

// ============================================================================
// encodeSessionRecordMsgpack unit tests
// ============================================================================

test "encodeSessionRecordMsgpack produces decodable output" {
    const alloc = std.testing.allocator;

    const record = SessionRecord{
        .session_id = "msgpack-test-1",
        .model = "test-model",
        .title = "Test Title",
        .system_prompt = null,
        .config_json = null,
        .marker = "active",
        .parent_session_id = null,
        .group_id = "grp-1",
        .head_item_id = 10,
        .ttl_ts = 500,
        .metadata_json = null,
        .created_at_ms = 1000,
        .updated_at_ms = 2000,
    };

    const encoded = try encodeSessionRecordMsgpack(alloc, record);
    defer alloc.free(encoded);

    // Verify the encoded bytes can be decoded back
    var decoded = try decodeSessionRecordMsgpack(alloc, encoded);
    defer freeScannedSessionRecord(alloc, &decoded);

    try std.testing.expectEqualStrings("msgpack-test-1", decoded.session_id);
    try std.testing.expectEqualStrings("test-model", decoded.model.?);
    try std.testing.expectEqualStrings("Test Title", decoded.title.?);
    try std.testing.expect(decoded.system_prompt == null);
    try std.testing.expect(decoded.config_json == null);
    try std.testing.expectEqualStrings("active", decoded.marker.?);
    try std.testing.expect(decoded.parent_session_id == null);
    try std.testing.expectEqualStrings("grp-1", decoded.group_id.?);
    try std.testing.expectEqual(@as(u64, 10), decoded.head_item_id);
    try std.testing.expectEqual(@as(i64, 500), decoded.ttl_ts);
    try std.testing.expect(decoded.metadata_json == null);
    try std.testing.expectEqual(@as(i64, 1000), decoded.created_at_ms);
    try std.testing.expectEqual(@as(i64, 2000), decoded.updated_at_ms);
}

test "encodeSessionRecordMsgpack all-null optional fields" {
    const alloc = std.testing.allocator;

    const record = SessionRecord{
        .session_id = "minimal-1",
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 0,
        .updated_at_ms = 0,
    };

    const encoded = try encodeSessionRecordMsgpack(alloc, record);
    defer alloc.free(encoded);

    var decoded = try decodeSessionRecordMsgpack(alloc, encoded);
    defer freeScannedSessionRecord(alloc, &decoded);

    try std.testing.expectEqualStrings("minimal-1", decoded.session_id);
    try std.testing.expect(decoded.model == null);
    try std.testing.expect(decoded.title == null);
}

// ============================================================================
// decodeSessionRecordKvBuf unit tests
// ============================================================================

test "decodeSessionRecordKvBuf decodes all fields" {
    const alloc = std.testing.allocator;

    const record = SessionRecord{
        .session_id = "kvbuf-direct-1",
        .model = "model-direct",
        .title = "Direct Title",
        .system_prompt = "Be concise",
        .config_json = "{\"temp\":0.5}",
        .marker = "pinned",
        .parent_session_id = "parent-direct",
        .group_id = "grp-direct",
        .head_item_id = 99,
        .ttl_ts = 7777,
        .metadata_json = "{\"key\":\"val\"}",
        .created_at_ms = 555,
        .updated_at_ms = 666,
    };

    const encoded = try encodeSessionRecordKvBuf(alloc, record);
    defer alloc.free(encoded);

    var decoded = try decodeSessionRecordKvBuf(alloc, encoded);
    defer freeScannedSessionRecord(alloc, &decoded);

    try std.testing.expectEqualStrings("kvbuf-direct-1", decoded.session_id);
    try std.testing.expectEqualStrings("model-direct", decoded.model.?);
    try std.testing.expectEqualStrings("Direct Title", decoded.title.?);
    try std.testing.expectEqualStrings("Be concise", decoded.system_prompt.?);
    try std.testing.expectEqualStrings("{\"temp\":0.5}", decoded.config_json.?);
    try std.testing.expectEqualStrings("pinned", decoded.marker.?);
    try std.testing.expectEqualStrings("parent-direct", decoded.parent_session_id.?);
    try std.testing.expectEqualStrings("grp-direct", decoded.group_id.?);
    try std.testing.expectEqual(@as(u64, 99), decoded.head_item_id);
    try std.testing.expectEqual(@as(i64, 7777), decoded.ttl_ts);
    try std.testing.expectEqualStrings("{\"key\":\"val\"}", decoded.metadata_json.?);
    try std.testing.expectEqual(@as(i64, 555), decoded.created_at_ms);
    try std.testing.expectEqual(@as(i64, 666), decoded.updated_at_ms);
}

test "decodeSessionRecordKvBuf minimal record with nulls" {
    const alloc = std.testing.allocator;

    const record = SessionRecord{
        .session_id = "kvbuf-minimal",
        .model = null,
        .title = null,
        .system_prompt = null,
        .config_json = null,
        .marker = null,
        .parent_session_id = null,
        .group_id = null,
        .head_item_id = 0,
        .ttl_ts = 0,
        .metadata_json = null,
        .created_at_ms = 0,
        .updated_at_ms = 0,
    };

    const encoded = try encodeSessionRecordKvBuf(alloc, record);
    defer alloc.free(encoded);

    var decoded = try decodeSessionRecordKvBuf(alloc, encoded);
    defer freeScannedSessionRecord(alloc, &decoded);

    try std.testing.expectEqualStrings("kvbuf-minimal", decoded.session_id);
    try std.testing.expect(decoded.model == null);
    try std.testing.expect(decoded.title == null);
}

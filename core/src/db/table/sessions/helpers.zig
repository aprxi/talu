//! Low-level I/O and JSON parsing utilities for session storage.
//!
//! Column readers, variable-length byte helpers, JSON field parsers.
//! No dependencies on TableAdapter or other session sub-modules.

const std = @import("std");
const block_reader = @import("../../block_reader.zig");
const types = @import("../../types.zig");
const responses = @import("../../../responses/root.zig");

const Allocator = std.mem.Allocator;
const ItemRecord = responses.ItemRecord;
const ItemStatus = responses.ItemStatus;
const ItemType = responses.ItemType;

pub fn findColumn(descs: []const types.ColumnDesc, column_id: u32) ?types.ColumnDesc {
    for (descs) |desc| {
        if (desc.column_id == column_id) return desc;
    }
    return null;
}

pub fn checkedRowCount(row_count: u32, data_len: usize, value_size: usize) !usize {
    const expected = @as(usize, row_count) * value_size;
    if (expected != data_len) return error.InvalidColumnData;
    return @as(usize, row_count);
}

pub fn readU64At(bytes: []const u8, row_idx: usize) !u64 {
    const start = row_idx * 8;
    const end = start + 8;
    if (end > bytes.len) return error.InvalidColumnData;
    return std.mem.readInt(u64, bytes[start..end][0..8], .little);
}

pub fn readI64At(bytes: []const u8, row_idx: usize) !i64 {
    const start = row_idx * 8;
    const end = start + 8;
    if (end > bytes.len) return error.InvalidColumnData;
    return std.mem.readInt(i64, bytes[start..end][0..8], .little);
}

pub const VarBytesBuffers = struct {
    data: []u8,
    offsets: []u32,
    lengths: []u32,

    pub fn deinit(self: *VarBytesBuffers, allocator: Allocator) void {
        allocator.free(self.data);
        allocator.free(self.offsets);
        allocator.free(self.lengths);
    }

    pub fn sliceForRow(self: VarBytesBuffers, row_idx: usize) ![]const u8 {
        if (row_idx >= self.offsets.len or row_idx >= self.lengths.len) return error.InvalidColumnData;
        const offset = self.offsets[row_idx];
        const length = self.lengths[row_idx];
        const start = @as(usize, offset);
        const end = start + @as(usize, length);
        if (end > self.data.len) return error.InvalidColumnData;
        return self.data[start..end];
    }
};

pub fn readVarBytesBuffers(
    file: std.fs.File,
    block_offset: u64,
    desc: types.ColumnDesc,
    row_count: u32,
    allocator: Allocator,
) !VarBytesBuffers {
    if (desc.offsets_off == 0 or desc.lengths_off == 0) return error.InvalidColumnLayout;

    const reader = block_reader.BlockReader.init(file, allocator);
    const data = try reader.readColumnData(block_offset, desc, allocator);
    errdefer allocator.free(data);

    const offsets = try readU32Array(file, block_offset + @as(u64, desc.offsets_off), row_count, allocator);
    errdefer allocator.free(offsets);

    const lengths = try readU32Array(file, block_offset + @as(u64, desc.lengths_off), row_count, allocator);
    errdefer allocator.free(lengths);

    return .{ .data = data, .offsets = offsets, .lengths = lengths };
}

pub fn readU32Array(file: std.fs.File, offset: u64, count: u32, allocator: Allocator) ![]u32 {
    const total_bytes = @as(usize, count) * @sizeOf(u32);
    const buffer = try allocator.alloc(u8, total_bytes);
    defer allocator.free(buffer);

    const read_len = try file.preadAll(buffer, offset);
    if (read_len != buffer.len) return error.UnexpectedEof;

    const values = try allocator.alloc(u32, count);
    var i: usize = 0;
    while (i < values.len) : (i += 1) {
        const start = i * 4;
        values[i] = std.mem.readInt(u32, buffer[start..][0..4], .little);
    }
    return values;
}

pub fn parseItemType(value: std.json.Value) ItemType {
    const obj = switch (value) {
        .object => |o| o,
        else => return ItemType.unknown,
    };
    const type_value = obj.get("type") orelse return ItemType.unknown;
    return switch (type_value) {
        .string => |s| ItemType.fromString(s),
        else => ItemType.unknown,
    };
}

pub fn parseStatus(value: std.json.Value) ?ItemStatus {
    const obj = switch (value) {
        .object => |o| o,
        else => return null,
    };
    const status_value = obj.get("status") orelse return null;
    return switch (status_value) {
        .string => |s| ItemStatus.fromString(s),
        else => null,
    };
}

pub const ParsedUsage = struct {
    input_tokens: u32 = 0,
    output_tokens: u32 = 0,
    prefill_ns: u64 = 0,
    generation_ns: u64 = 0,
    finish_reason: ?[]const u8 = null,
};

/// Parse _usage metadata from a record JSON object. Returns defaults if absent.
pub fn parseUsage(allocator: Allocator, record_value: std.json.Value) ParsedUsage {
    const obj = switch (record_value) {
        .object => |o| o,
        else => return .{},
    };
    const usage_value = obj.get("_usage") orelse return .{};
    const usage_obj = switch (usage_value) {
        .object => |o| o,
        else => return .{},
    };

    var result = ParsedUsage{};
    if (usage_obj.get("input_tokens")) |v| {
        result.input_tokens = switch (v) {
            .integer => |i| @intCast(@max(0, i)),
            else => 0,
        };
    }
    if (usage_obj.get("output_tokens")) |v| {
        result.output_tokens = switch (v) {
            .integer => |i| @intCast(@max(0, i)),
            else => 0,
        };
    }
    if (usage_obj.get("prefill_ns")) |v| {
        result.prefill_ns = switch (v) {
            .integer => |i| @intCast(@max(0, i)),
            else => 0,
        };
    }
    if (usage_obj.get("generation_ns")) |v| {
        result.generation_ns = switch (v) {
            .integer => |i| @intCast(@max(0, i)),
            else => 0,
        };
    }
    if (usage_obj.get("finish_reason")) |v| {
        result.finish_reason = switch (v) {
            .string => |s| allocator.dupe(u8, s) catch null,
            else => null,
        };
    }
    return result;
}

pub fn sortByItemId(_: void, a: ItemRecord, b: ItemRecord) bool {
    return a.item_id < b.item_id;
}

/// Parse generation metadata from a record JSON object. Returns null if absent.
/// The generation field contains model and sampling parameters as a JSON object.
pub fn parseGenerationJson(allocator: Allocator, record_value: std.json.Value) ?[]const u8 {
    const obj = switch (record_value) {
        .object => |o| o,
        else => return null,
    };
    const gen_value = obj.get("generation") orelse return null;
    // Re-serialize the generation object to a JSON string
    return std.json.Stringify.valueAlloc(allocator, gen_value, .{}) catch null;
}

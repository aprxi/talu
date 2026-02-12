//! Record Parser - JSON parsing for ItemRecord storage records.
//!
//! This module provides JSON-to-ItemRecord parsing for the storage restore API.
//! The inverse of record_serializer.zig - deserializes JSON back into ItemRecord types.
//!
//! # Usage
//!
//! ```zig
//! const variant = try parseItemVariantRecord(allocator, json_bytes, status);
//! defer freeItemVariantRecord(allocator, variant);
//! ```
//!
//! Thread safety: Functions are stateless and thread-safe.

const std = @import("std");
const json_mod = @import("../io/json/root.zig");
const backend = @import("backend.zig");

const ItemVariantRecord = backend.ItemVariantRecord;
const ItemContentPartRecord = backend.ItemContentPartRecord;
const ItemStatus = backend.ItemStatus;
const MessageRole = backend.MessageRole;
const ImageDetail = backend.ImageDetail;

// =============================================================================
// Public API
// =============================================================================

/// Parse an item variant from JSON bytes.
///
/// Returns an allocated ItemVariantRecord. Caller must free with freeItemVariantRecord.
pub fn parseItemVariantRecord(
    alloc: std.mem.Allocator,
    json_bytes: []const u8,
    status: ItemStatus,
) !ItemVariantRecord {
    const parsed = json_mod.parseValue(alloc, json_bytes, .{
        .max_size_bytes = 50 * 1024 * 1024,
        .max_value_bytes = 50 * 1024 * 1024,
        .max_string_bytes = 50 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidArgument,
            error.InputTooDeep => error.InvalidArgument,
            error.StringTooLong => error.InvalidArgument,
            error.InvalidJson => error.InvalidArgument,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return error.InvalidArgument,
    };
    const type_value = obj.get("type") orelse return error.InvalidArgument;
    const type_str = try expectString(type_value);

    if (std.mem.eql(u8, type_str, "message")) {
        const role_str = try expectString(obj.get("role") orelse return error.InvalidArgument);
        const role = MessageRole.fromString(role_str);
        const content_val = obj.get("content") orelse return error.InvalidArgument;
        const content = try parseContentPartsRecord(alloc, content_val);
        errdefer {
            for (content) |part| {
                freeContentPartRecord(alloc, part);
            }
            alloc.free(content);
        }
        return .{ .message = .{
            .role = role,
            .status = status,
            .content = content,
        } };
    }
    if (std.mem.eql(u8, type_str, "function_call")) {
        const call_id = try expectString(obj.get("call_id") orelse return error.InvalidArgument);
        const name = try expectString(obj.get("name") orelse return error.InvalidArgument);
        const args_val = obj.get("arguments") orelse return error.InvalidArgument;
        const call_id_copy = try alloc.dupe(u8, call_id);
        errdefer alloc.free(call_id_copy);
        const name_copy = try alloc.dupe(u8, name);
        errdefer alloc.free(name_copy);
        const args_copy = switch (args_val) {
            .string => |s| try alloc.dupe(u8, s),
            else => try encodeJsonValue(alloc, args_val),
        };
        errdefer alloc.free(args_copy);
        return .{ .function_call = .{
            .call_id = call_id_copy,
            .name = name_copy,
            .arguments = args_copy,
            .status = status,
        } };
    }
    if (std.mem.eql(u8, type_str, "function_call_output")) {
        const call_id = try expectString(obj.get("call_id") orelse return error.InvalidArgument);
        const output_val = obj.get("output") orelse return error.InvalidArgument;
        switch (output_val) {
            .array => {
                const output = try parseContentPartsRecord(alloc, output_val);
                errdefer {
                    for (output) |part| {
                        freeContentPartRecord(alloc, part);
                    }
                    alloc.free(output);
                }
                const call_id_copy = try alloc.dupe(u8, call_id);
                errdefer alloc.free(call_id_copy);
                return .{ .function_call_output = .{
                    .call_id = call_id_copy,
                    .output = output,
                    .status = status,
                } };
            },
            .string => |s| {
                const part = ItemContentPartRecord{ .text = .{ .text = try alloc.dupe(u8, s) } };
                errdefer freeContentPartRecord(alloc, part);
                const output = try alloc.alloc(ItemContentPartRecord, 1);
                errdefer alloc.free(output);
                output[0] = part;
                const call_id_copy = try alloc.dupe(u8, call_id);
                errdefer alloc.free(call_id_copy);
                return .{ .function_call_output = .{
                    .call_id = call_id_copy,
                    .output = output,
                    .status = status,
                } };
            },
            else => return error.InvalidArgument,
        }
    }
    if (std.mem.eql(u8, type_str, "reasoning")) {
        const summary_val = obj.get("summary") orelse return error.InvalidArgument;
        const summary = try parseContentPartsRecord(alloc, summary_val);
        errdefer {
            for (summary) |part| {
                freeContentPartRecord(alloc, part);
            }
            alloc.free(summary);
        }
        const content = if (obj.get("content")) |content_val| blk: {
            switch (content_val) {
                .array => |a| {
                    if (a.items.len == 0) break :blk try alloc.alloc(ItemContentPartRecord, 0);
                    break :blk try parseContentPartsRecord(alloc, content_val);
                },
                else => return error.InvalidArgument,
            }
        } else blk: {
            break :blk try alloc.alloc(ItemContentPartRecord, 0);
        };
        errdefer {
            for (content) |part| {
                freeContentPartRecord(alloc, part);
            }
            alloc.free(content);
        }
        const encrypted = if (obj.get("encrypted_content")) |val|
            try encodeJsonValue(alloc, val)
        else
            null;
        errdefer if (encrypted) |e| alloc.free(e);
        return .{ .reasoning = .{
            .content = content,
            .summary = summary,
            .encrypted_content = encrypted,
            .status = status,
        } };
    }
    if (std.mem.eql(u8, type_str, "item_reference")) {
        const id_val = obj.get("id") orelse return error.InvalidArgument;
        const id_str = try expectString(id_val);
        const id_copy = try alloc.dupe(u8, id_str);
        errdefer alloc.free(id_copy);
        return .{ .item_reference = .{
            .id = id_copy,
            .status = status,
        } };
    }

    const payload_value = obj.get("payload") orelse obj.get("data") orelse parsed.value;
    const payload = try encodeJsonValue(alloc, payload_value);
    errdefer alloc.free(payload);
    return .{ .unknown = .{
        .raw_type = try alloc.dupe(u8, type_str),
        .payload = payload,
    } };
}

/// Free an ItemContentPartRecord allocated by parseContentPartRecord.
pub fn freeContentPartRecord(alloc: std.mem.Allocator, record: ItemContentPartRecord) void {
    switch (record) {
        .input_text => |p| alloc.free(p.text),
        .input_image => |p| alloc.free(p.image_url),
        .input_audio => |p| alloc.free(p.audio_data),
        .input_video => |p| alloc.free(p.video_url),
        .input_file => |p| {
            if (p.filename) |f| alloc.free(f);
            if (p.file_data) |d| alloc.free(d);
            if (p.file_url) |u| alloc.free(u);
        },
        .output_text => |p| {
            alloc.free(p.text);
            if (p.logprobs_json) |l| alloc.free(l);
            if (p.annotations_json) |a| alloc.free(a);
            if (p.code_blocks_json) |c| alloc.free(c);
        },
        .refusal => |p| alloc.free(p.refusal),
        .text => |p| alloc.free(p.text),
        .reasoning_text => |p| alloc.free(p.text),
        .summary_text => |p| alloc.free(p.text),
        .unknown => |p| {
            if (p.raw_type.len > 0) alloc.free(p.raw_type);
            if (p.raw_data.len > 0) alloc.free(p.raw_data);
        },
    }
}

/// Convert ItemStatus from u8 value.
pub fn itemStatusFromU8(value: u8) ItemStatus {
    return switch (value) {
        0 => .in_progress,
        1 => .waiting,
        2 => .completed,
        3 => .incomplete,
        4 => .failed,
        else => .completed,
    };
}

// =============================================================================
// Internal Helpers
// =============================================================================

/// Extract string from JSON value.
fn expectString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |s| s,
        else => error.InvalidArgument,
    };
}

/// Re-encode a JSON value to a string.
fn encodeJsonValue(alloc: std.mem.Allocator, value: std.json.Value) ![]u8 {
    return std.json.Stringify.valueAlloc(alloc, value, .{});
}

/// Parse a single content part from JSON value.
fn parseContentPartRecord(
    alloc: std.mem.Allocator,
    value: std.json.Value,
) !ItemContentPartRecord {
    const obj = switch (value) {
        .object => |o| o,
        else => return error.InvalidArgument,
    };
    const type_value = obj.get("type") orelse return error.InvalidArgument;
    const type_str = try expectString(type_value);

    if (std.mem.eql(u8, type_str, "input_text")) {
        const text = try expectString(obj.get("text") orelse return error.InvalidArgument);
        return .{ .input_text = .{ .text = try alloc.dupe(u8, text) } };
    }
    if (std.mem.eql(u8, type_str, "output_text")) {
        const text = try expectString(obj.get("text") orelse return error.InvalidArgument);
        var record = ItemContentPartRecord{
            .output_text = .{
                .text = try alloc.dupe(u8, text),
                .logprobs_json = null,
                .annotations_json = null,
                .code_blocks_json = null,
            },
        };
        if (obj.get("logprobs")) |val| {
            record.output_text.logprobs_json = try encodeJsonValue(alloc, val);
        }
        if (obj.get("annotations")) |val| {
            record.output_text.annotations_json = try encodeJsonValue(alloc, val);
        }
        if (obj.get("code_blocks")) |val| {
            record.output_text.code_blocks_json = try encodeJsonValue(alloc, val);
        }
        return record;
    }
    if (std.mem.eql(u8, type_str, "input_image")) {
        const image_url = try expectString(obj.get("image_url") orelse return error.InvalidArgument);
        var detail: ImageDetail = .auto;
        if (obj.get("detail")) |val| {
            if (val == .string) {
                detail = ImageDetail.fromString(val.string) orelse .auto;
            }
        }
        return .{ .input_image = .{
            .image_url = try alloc.dupe(u8, image_url),
            .detail = detail,
        } };
    }
    if (std.mem.eql(u8, type_str, "input_audio")) {
        const audio_data = try expectString(obj.get("audio_data") orelse return error.InvalidArgument);
        return .{ .input_audio = .{ .audio_data = try alloc.dupe(u8, audio_data) } };
    }
    if (std.mem.eql(u8, type_str, "input_video")) {
        const video_url = try expectString(obj.get("video_url") orelse return error.InvalidArgument);
        return .{ .input_video = .{ .video_url = try alloc.dupe(u8, video_url) } };
    }
    if (std.mem.eql(u8, type_str, "input_file")) {
        return .{ .input_file = .{
            .filename = if (obj.get("filename")) |val| try alloc.dupe(u8, try expectString(val)) else null,
            .file_data = if (obj.get("file_data")) |val| try alloc.dupe(u8, try expectString(val)) else null,
            .file_url = if (obj.get("file_url")) |val| try alloc.dupe(u8, try expectString(val)) else null,
        } };
    }
    if (std.mem.eql(u8, type_str, "refusal")) {
        const refusal = try expectString(obj.get("refusal") orelse return error.InvalidArgument);
        return .{ .refusal = .{ .refusal = try alloc.dupe(u8, refusal) } };
    }
    if (std.mem.eql(u8, type_str, "text")) {
        const text = try expectString(obj.get("text") orelse return error.InvalidArgument);
        return .{ .text = .{ .text = try alloc.dupe(u8, text) } };
    }
    if (std.mem.eql(u8, type_str, "reasoning_text")) {
        const text = try expectString(obj.get("text") orelse return error.InvalidArgument);
        return .{ .reasoning_text = .{ .text = try alloc.dupe(u8, text) } };
    }
    if (std.mem.eql(u8, type_str, "summary_text")) {
        const text = try expectString(obj.get("text") orelse return error.InvalidArgument);
        return .{ .summary_text = .{ .text = try alloc.dupe(u8, text) } };
    }

    const payload_value = obj.get("payload") orelse obj.get("data") orelse value;
    const payload = try encodeJsonValue(alloc, payload_value);
    return .{ .unknown = .{
        .raw_type = try alloc.dupe(u8, type_str),
        .raw_data = payload,
    } };
}

/// Parse an array of content parts from JSON value.
fn parseContentPartsRecord(
    alloc: std.mem.Allocator,
    value: std.json.Value,
) ![]ItemContentPartRecord {
    const array = switch (value) {
        .array => |a| a,
        else => return error.InvalidArgument,
    };
    const result = try alloc.alloc(ItemContentPartRecord, array.items.len);
    var filled: usize = 0;
    errdefer {
        for (result[0..filled]) |record| {
            freeContentPartRecord(alloc, record);
        }
        alloc.free(result);
    }

    for (array.items, 0..) |item, idx| {
        result[idx] = try parseContentPartRecord(alloc, item);
        filled += 1;
    }

    return result;
}

// =============================================================================
// Tests
// =============================================================================

test "parseItemVariantRecord - basic message" {
    const allocator = std.testing.allocator;
    const json =
        \\{"type":"message","role":"user","content":[{"type":"input_text","text":"Hello"}]}
    ;

    const variant = try parseItemVariantRecord(allocator, json, .completed);
    defer {
        switch (variant) {
            .message => |m| {
                for (m.content) |part| {
                    freeContentPartRecord(allocator, part);
                }
                allocator.free(m.content);
            },
            else => {},
        }
    }

    try std.testing.expect(variant == .message);
    try std.testing.expectEqual(MessageRole.user, variant.message.role);
    try std.testing.expectEqual(@as(usize, 1), variant.message.content.len);
}

test "itemStatusFromU8" {
    try std.testing.expectEqual(ItemStatus.in_progress, itemStatusFromU8(0));
    try std.testing.expectEqual(ItemStatus.waiting, itemStatusFromU8(1));
    try std.testing.expectEqual(ItemStatus.completed, itemStatusFromU8(2));
    try std.testing.expectEqual(ItemStatus.incomplete, itemStatusFromU8(3));
    try std.testing.expectEqual(ItemStatus.failed, itemStatusFromU8(4));
    try std.testing.expectEqual(ItemStatus.completed, itemStatusFromU8(99)); // Default
}

test "freeContentPartRecord - all content types" {
    const allocator = std.testing.allocator;

    // Test input_text
    {
        const record = ItemContentPartRecord{ .input_text = .{ .text = try allocator.dupe(u8, "hello") } };
        freeContentPartRecord(allocator, record);
    }

    // Test input_image
    {
        const record = ItemContentPartRecord{ .input_image = .{ .image_url = try allocator.dupe(u8, "http://example.com/img.png"), .detail = .auto } };
        freeContentPartRecord(allocator, record);
    }

    // Test input_audio
    {
        const record = ItemContentPartRecord{ .input_audio = .{ .audio_data = try allocator.dupe(u8, "base64data") } };
        freeContentPartRecord(allocator, record);
    }

    // Test input_video
    {
        const record = ItemContentPartRecord{ .input_video = .{ .video_url = try allocator.dupe(u8, "http://example.com/video.mp4") } };
        freeContentPartRecord(allocator, record);
    }

    // Test input_file with all fields
    {
        const record = ItemContentPartRecord{ .input_file = .{
            .filename = try allocator.dupe(u8, "test.txt"),
            .file_data = try allocator.dupe(u8, "data"),
            .file_url = try allocator.dupe(u8, "http://example.com/file"),
        } };
        freeContentPartRecord(allocator, record);
    }

    // Test input_file with null fields
    {
        const record = ItemContentPartRecord{ .input_file = .{
            .filename = null,
            .file_data = null,
            .file_url = null,
        } };
        freeContentPartRecord(allocator, record);
    }

    // Test output_text with all fields
    {
        const record = ItemContentPartRecord{ .output_text = .{
            .text = try allocator.dupe(u8, "output"),
            .logprobs_json = try allocator.dupe(u8, "[]"),
            .annotations_json = try allocator.dupe(u8, "[]"),
        } };
        freeContentPartRecord(allocator, record);
    }

    // Test output_text with null optional fields
    {
        const record = ItemContentPartRecord{ .output_text = .{
            .text = try allocator.dupe(u8, "output"),
            .logprobs_json = null,
            .annotations_json = null,
        } };
        freeContentPartRecord(allocator, record);
    }

    // Test refusal
    {
        const record = ItemContentPartRecord{ .refusal = .{ .refusal = try allocator.dupe(u8, "I cannot help with that") } };
        freeContentPartRecord(allocator, record);
    }

    // Test text
    {
        const record = ItemContentPartRecord{ .text = .{ .text = try allocator.dupe(u8, "plain text") } };
        freeContentPartRecord(allocator, record);
    }

    // Test reasoning_text
    {
        const record = ItemContentPartRecord{ .reasoning_text = .{ .text = try allocator.dupe(u8, "reasoning") } };
        freeContentPartRecord(allocator, record);
    }

    // Test summary_text
    {
        const record = ItemContentPartRecord{ .summary_text = .{ .text = try allocator.dupe(u8, "summary") } };
        freeContentPartRecord(allocator, record);
    }

    // Test unknown with data
    {
        const record = ItemContentPartRecord{ .unknown = .{
            .raw_type = try allocator.dupe(u8, "custom_type"),
            .raw_data = try allocator.dupe(u8, "{\"key\":\"value\"}"),
        } };
        freeContentPartRecord(allocator, record);
    }

    // Test unknown with empty strings (no allocation to free)
    {
        const record = ItemContentPartRecord{ .unknown = .{
            .raw_type = "",
            .raw_data = "",
        } };
        freeContentPartRecord(allocator, record);
    }
}

//! Record Serialization - JSON serialization for ItemRecord storage records.
//!
//! This module provides ItemRecord-to-JSON serialization for the storage callback API.
//! ItemRecord is the portable snapshot type (with plain slices), distinct from Item
//! (which uses ArrayListUnmanaged for dynamic content).
//!
//! # Usage
//!
//! ```zig
//! const json = try serializeItemRecordToJsonZ(allocator, record);
//! defer allocator.free(json);
//! // json contains the full Open Responses JSON for this record
//! ```
//!
//! Thread safety: Functions are stateless and thread-safe.

const std = @import("std");
const backend = @import("backend.zig");
const kvbuf = @import("../io/kvbuf/root.zig");

const ItemRecord = backend.ItemRecord;
const ItemVariantRecord = backend.ItemVariantRecord;
const ItemContentPartRecord = backend.ItemContentPartRecord;
const ItemStatus = backend.ItemStatus;
const MessageRole = backend.MessageRole;

// =============================================================================
// Public API
// =============================================================================

/// Serialize an ItemRecord to a null-terminated JSON string.
///
/// Returns a sentinel-terminated slice suitable for C interop.
/// Caller owns returned memory.
pub fn serializeItemRecordToJsonZ(allocator: std.mem.Allocator, record: ItemRecord) ![:0]u8 {
    var json_bytes: std.ArrayListUnmanaged(u8) = .{};
    errdefer json_bytes.deinit(allocator);
    const writer = json_bytes.writer(allocator);

    try writeItemRecordJson(writer, record);

    return json_bytes.toOwnedSliceSentinel(allocator, 0);
}

/// Serialize an ItemRecord to a KvBuf binary blob for storage.
///
/// The blob contains three fields:
///   - FieldIds.session_id: The session_id string.
///   - FieldIds.record_json: The full record JSON (same as serializeItemRecordToJsonZ output).
///   - FieldIds.content_text: Pre-extracted plain text from all content parts (for search).
///
/// The content_text field enables zero-copy substring search without JSON parsing.
/// Caller owns returned memory.
pub fn serializeItemRecordToKvBuf(
    allocator: std.mem.Allocator,
    record: ItemRecord,
    session_id: []const u8,
) ![]u8 {
    var w = kvbuf.KvBufWriter.init();
    errdefer w.deinit(allocator);

    // Field 1: session_id
    try w.addString(allocator, kvbuf.FieldIds.session_id, session_id);

    // Field 2: record JSON
    const record_json_z = try serializeItemRecordToJsonZ(allocator, record);
    defer allocator.free(record_json_z);
    const record_json = std.mem.sliceTo(record_json_z, 0);
    try w.addString(allocator, kvbuf.FieldIds.record_json, record_json);

    // Field 3: content_text (pre-extracted for search)
    const content_text = extractContentText(allocator, record);
    defer if (content_text) |t| allocator.free(t);
    if (content_text) |text| {
        try w.addString(allocator, kvbuf.FieldIds.content_text, text);
    }

    const blob = try w.finish(allocator);
    w.deinit(allocator);
    return blob;
}

/// Extract plain text from all content parts in an ItemRecord.
///
/// Concatenates text fields from message content, function call outputs,
/// reasoning content/summary. Returns null if no text is found.
/// Caller owns the returned slice.
fn extractContentText(allocator: std.mem.Allocator, record: ItemRecord) ?[]u8 {
    var parts = std.ArrayList(u8).empty;

    const slices = switch (record.variant) {
        .message => |m| collectTextFromParts(&parts, allocator, m.content),
        .function_call_output => |f| collectTextFromParts(&parts, allocator, f.output),
        .reasoning => |r| {
            collectTextFromParts(&parts, allocator, r.summary);
            collectTextFromParts(&parts, allocator, r.content);
        },
        .function_call => |f| {
            appendText(&parts, allocator, f.arguments);
        },
        .unknown => |u| {
            appendText(&parts, allocator, u.payload);
        },
        .item_reference => {},
    };
    _ = slices;

    if (parts.items.len == 0) {
        parts.deinit(allocator);
        return null;
    }
    return parts.toOwnedSlice(allocator) catch {
        parts.deinit(allocator);
        return null;
    };
}

fn collectTextFromParts(
    result: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    parts: []const ItemContentPartRecord,
) void {
    for (parts) |part| {
        const text: ?[]const u8 = switch (part) {
            .input_text => |p| p.text,
            .output_text => |p| p.text,
            .text => |p| p.text,
            .reasoning_text => |p| p.text,
            .summary_text => |p| p.text,
            .refusal => |p| p.refusal,
            .input_image, .input_audio, .input_video, .input_file, .unknown => null,
        };
        if (text) |t| {
            appendText(result, allocator, t);
        }
    }
}

fn appendText(result: *std.ArrayList(u8), allocator: std.mem.Allocator, text: []const u8) void {
    if (text.len == 0) return;
    if (result.items.len > 0) {
        result.append(allocator, ' ') catch return;
    }
    result.appendSlice(allocator, text) catch return;
}

/// Extract the role from an ItemRecord as a u8 enum value.
///
/// Returns the MessageRole enum value for messages, or 255 (N/A) for other item types.
pub fn extractRoleFromRecord(record: ItemRecord) u8 {
    if (record.item_type != .message) {
        return 255; // N/A for non-message items
    }

    switch (record.variant) {
        .message => |m| return @intFromEnum(m.role),
        else => return 255,
    }
}

// =============================================================================
// Internal Serialization
// =============================================================================

/// Map ItemStatus to storage JSON string.
fn statusToString(status: ItemStatus) []const u8 {
    return switch (status) {
        .in_progress => "in_progress",
        .waiting => "waiting",
        .completed => "completed",
        .incomplete => "incomplete",
        .failed => "failed",
    };
}

/// Write a single ItemRecord as JSON.
fn writeItemRecordJson(writer: anytype, record: ItemRecord) !void {
    switch (record.variant) {
        .message => |m| {
            try writer.print("{{\"type\":\"message\",\"id\":\"msg_{d}\",\"role\":\"{s}\",\"status\":\"{s}\",\"content\":", .{
                record.item_id,
                m.role.toString(),
                statusToString(m.status),
            });
            try writeContentPartsRecordJson(writer, m.content);
            try writeGenerationField(writer, record);
            try writeUsageFields(writer, record);
            try writer.writeByte('}');
        },
        .function_call => |f| {
            try writer.print("{{\"type\":\"function_call\",\"id\":\"fc_{d}\",\"call_id\":\"{s}\",\"name\":\"{s}\",\"arguments\":{f},\"status\":\"{s}\"", .{
                record.item_id,
                f.call_id,
                f.name,
                std.json.fmt(f.arguments, .{}),
                statusToString(f.status),
            });
            try writeUsageFields(writer, record);
            try writer.writeByte('}');
        },
        .function_call_output => |f| {
            try writer.print("{{\"type\":\"function_call_output\",\"id\":\"fco_{d}\",\"call_id\":\"{s}\",\"output\":", .{
                record.item_id,
                f.call_id,
            });
            try writeContentPartsRecordJson(writer, f.output);
            try writer.print(",\"status\":\"{s}\"", .{statusToString(f.status)});
            try writeUsageFields(writer, record);
            try writer.writeByte('}');
        },
        .reasoning => |r| {
            try writer.print("{{\"type\":\"reasoning\",\"id\":\"rs_{d}\",\"summary\":", .{record.item_id});
            try writeContentPartsRecordJson(writer, r.summary);
            if (r.content.len > 0) {
                try writer.writeAll(",\"content\":");
                try writeContentPartsRecordJson(writer, r.content);
            }
            if (r.encrypted_content) |e| {
                try writer.print(",\"encrypted_content\":{f}", .{std.json.fmt(e, .{})});
            }
            try writeUsageFields(writer, record);
            try writer.writeByte('}');
        },
        .item_reference => |ref| {
            try writer.print("{{\"type\":\"item_reference\",\"id\":\"{s}\"", .{ref.id});
            try writeUsageFields(writer, record);
            try writer.writeByte('}');
        },
        .unknown => |u| {
            try writer.print("{{\"type\":\"{s}\",\"payload\":{s}", .{ u.raw_type, u.payload });
            try writeUsageFields(writer, record);
            try writer.writeByte('}');
        },
    }
}

/// Write _usage metadata fields if any are non-zero.
/// Emitted as: ,"_usage":{"input_tokens":N,"output_tokens":N,"prefill_ns":N,"generation_ns":N,"finish_reason":"..."}
fn writeUsageFields(writer: anytype, record: ItemRecord) !void {
    const has_tokens = record.input_tokens > 0 or record.output_tokens > 0;
    const has_timing = record.prefill_ns > 0 or record.generation_ns > 0;
    const has_reason = record.finish_reason != null;
    if (!has_tokens and !has_timing and !has_reason) return;

    try writer.writeAll(",\"_usage\":{");
    try writer.print("\"input_tokens\":{d},\"output_tokens\":{d}", .{ record.input_tokens, record.output_tokens });
    try writer.print(",\"prefill_ns\":{d},\"generation_ns\":{d}", .{ record.prefill_ns, record.generation_ns });
    if (record.finish_reason) |reason| {
        try writer.print(",\"finish_reason\":{f}", .{std.json.fmt(reason, .{})});
    }
    try writer.writeByte('}');
}

/// Write generation field if present.
/// Emitted as: ,"generation":{...} (raw JSON passthrough)
fn writeGenerationField(writer: anytype, record: ItemRecord) !void {
    if (record.generation_json) |gen| {
        try writer.writeAll(",\"generation\":");
        try writer.writeAll(gen);
    }
}

/// Write content parts array as JSON.
fn writeContentPartsRecordJson(writer: anytype, parts: []const ItemContentPartRecord) !void {
    try writer.writeByte('[');

    var first = true;
    for (parts) |part| {
        if (!first) try writer.writeByte(',');
        first = false;

        switch (part) {
            .input_text => |p| {
                try writer.print("{{\"type\":\"input_text\",\"text\":{f}}}", .{
                    std.json.fmt(p.text, .{}),
                });
            },
            .text => |p| {
                try writer.print("{{\"type\":\"text\",\"text\":{f}}}", .{
                    std.json.fmt(p.text, .{}),
                });
            },
            .reasoning_text => |p| {
                try writer.print("{{\"type\":\"reasoning_text\",\"text\":{f}}}", .{
                    std.json.fmt(p.text, .{}),
                });
            },
            .summary_text => |p| {
                try writer.print("{{\"type\":\"summary_text\",\"text\":{f}}}", .{
                    std.json.fmt(p.text, .{}),
                });
            },
            .output_text => |p| {
                try writer.print("{{\"type\":\"output_text\",\"text\":{f}", .{
                    std.json.fmt(p.text, .{}),
                });
                if (p.logprobs_json) |l| {
                    try writer.print(",\"logprobs\":{s}", .{l});
                }
                if (p.annotations_json) |a| {
                    try writer.print(",\"annotations\":{s}", .{a});
                }
                if (p.code_blocks_json) |c| {
                    try writer.print(",\"code_blocks\":{s}", .{c});
                }
                try writer.writeByte('}');
            },
            .refusal => |p| {
                try writer.print("{{\"type\":\"refusal\",\"refusal\":{f}}}", .{
                    std.json.fmt(p.refusal, .{}),
                });
            },
            .input_image => |p| {
                try writer.print("{{\"type\":\"input_image\",\"image_url\":{f}}}", .{
                    std.json.fmt(p.image_url, .{}),
                });
            },
            .input_audio => |p| {
                try writer.print("{{\"type\":\"input_audio\",\"audio_data\":{f}}}", .{
                    std.json.fmt(p.audio_data, .{}),
                });
            },
            .input_video => |p| {
                try writer.print("{{\"type\":\"input_video\",\"video_url\":{f}}}", .{
                    std.json.fmt(p.video_url, .{}),
                });
            },
            .input_file => |p| {
                try writer.writeAll("{\"type\":\"input_file\"");
                if (p.filename) |f| {
                    try writer.print(",\"filename\":{f}", .{std.json.fmt(f, .{})});
                }
                if (p.file_data) |d| {
                    try writer.print(",\"file_data\":{f}", .{std.json.fmt(d, .{})});
                }
                if (p.file_url) |u| {
                    try writer.print(",\"file_url\":{f}", .{std.json.fmt(u, .{})});
                }
                try writer.writeByte('}');
            },
            .unknown => |p| {
                try writer.print("{{\"type\":\"{s}\",\"data\":{s}}}", .{
                    p.raw_type,
                    p.raw_data,
                });
            },
        }
    }

    try writer.writeByte(']');
}

// =============================================================================
// Tests
// =============================================================================

test "serializeItemRecordToJsonZ - basic message" {
    const allocator = std.testing.allocator;

    var content = [_]ItemContentPartRecord{
        .{ .input_text = .{ .text = "Hello, world!" } },
    };

    const record = ItemRecord{
        .item_id = 42,
        .item_type = .message,
        .created_at_ms = 1234567890,
        .variant = .{
            .message = .{
                .role = .user,
                .status = .completed,
                .content = &content,
            },
        },
    };

    const json = try serializeItemRecordToJsonZ(allocator, record);
    defer allocator.free(json);

    // Verify it contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"message\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\":\"msg_42\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\":\"completed\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "Hello, world!") != null);
}

test "extractRoleFromRecord - message" {
    const record = ItemRecord{
        .item_id = 1,
        .item_type = .message,
        .created_at_ms = 0,
        .variant = .{
            .message = .{
                .role = .assistant,
                .status = .completed,
                .content = &.{},
            },
        },
    };

    try std.testing.expectEqual(@as(u8, 2), extractRoleFromRecord(record)); // assistant = 2
}

test "extractRoleFromRecord - non-message returns 255" {
    const record = ItemRecord{
        .item_id = 1,
        .item_type = .function_call,
        .created_at_ms = 0,
        .variant = .{
            .function_call = .{
                .call_id = "call_1",
                .name = "test",
                .arguments = "{}",
                .status = .completed,
            },
        },
    };

    try std.testing.expectEqual(@as(u8, 255), extractRoleFromRecord(record)); // N/A sentinel
}

test "serializeItemRecordToKvBuf - basic message" {
    const allocator = std.testing.allocator;

    var content = [_]ItemContentPartRecord{
        .{ .input_text = .{ .text = "Hello, world!" } },
    };

    const record = ItemRecord{
        .item_id = 42,
        .item_type = .message,
        .created_at_ms = 1234567890,
        .variant = .{
            .message = .{
                .role = .user,
                .status = .completed,
                .content = &content,
            },
        },
    };

    const blob = try serializeItemRecordToKvBuf(allocator, record, "test-session-1");
    defer allocator.free(blob);

    // Verify it's a KvBuf blob
    try std.testing.expect(kvbuf.isKvBuf(blob));

    // Read it back
    const reader = try kvbuf.KvBufReader.init(blob);

    // Check session_id
    const sid = reader.get(kvbuf.FieldIds.session_id) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("test-session-1", sid);

    // Check record_json contains expected content
    const rjson = reader.get(kvbuf.FieldIds.record_json) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, rjson, "Hello, world!") != null);

    // Check content_text is pre-extracted
    const ctext = reader.get(kvbuf.FieldIds.content_text) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("Hello, world!", ctext);
}

test "serializeItemRecordToKvBuf - function_call has no content_text" {
    const allocator = std.testing.allocator;

    const record = ItemRecord{
        .item_id = 1,
        .item_type = .function_call,
        .created_at_ms = 0,
        .variant = .{
            .function_call = .{
                .call_id = "call_1",
                .name = "test",
                .arguments = "{}",
                .status = .completed,
            },
        },
    };

    const blob = try serializeItemRecordToKvBuf(allocator, record, "sess-1");
    defer allocator.free(blob);

    const reader = try kvbuf.KvBufReader.init(blob);
    // function_call arguments "{}" is extracted as content_text
    const ctext = reader.get(kvbuf.FieldIds.content_text);
    try std.testing.expect(ctext != null);
}

test "extractContentText - message with multiple content parts" {
    const allocator = std.testing.allocator;

    var content = [_]ItemContentPartRecord{
        .{ .input_text = .{ .text = "Hello" } },
        .{ .output_text = .{ .text = "World", .logprobs_json = null, .annotations_json = null } },
    };

    const record = ItemRecord{
        .item_id = 1,
        .item_type = .message,
        .created_at_ms = 0,
        .variant = .{
            .message = .{
                .role = .user,
                .status = .completed,
                .content = &content,
            },
        },
    };

    const text = extractContentText(allocator, record) orelse return error.TestUnexpectedResult;
    defer allocator.free(text);
    try std.testing.expectEqualStrings("Hello World", text);
}

test "extractContentText - empty message returns null" {
    const record = ItemRecord{
        .item_id = 1,
        .item_type = .message,
        .created_at_ms = 0,
        .variant = .{
            .message = .{
                .role = .user,
                .status = .completed,
                .content = &.{},
            },
        },
    };

    const text = extractContentText(std.testing.allocator, record);
    try std.testing.expect(text == null);
}

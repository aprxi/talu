//! Storage Serialization - Canonical JSON serialization for storage backends.
//!
//! This module provides Item-to-JSON serialization specifically for storage.
//! It uses SerializationDirection.response to ensure full schema fidelity:
//!   - logprobs included in output_text
//!   - all annotations preserved
//!   - reasoning content included
//!
//! The output is 100% compatible with the Open Responses v2.3.0 schema.
//!
//! # Usage
//!
//! ```zig
//! const json = try serializeItemToJson(allocator, item);
//! defer allocator.free(json);
//! // json contains the full Open Responses JSON for this item
//! ```
//!
//! Thread safety: Functions are stateless and thread-safe.

const std = @import("std");
const items = @import("items.zig");

const Item = items.Item;
const ItemType = items.ItemType;
const ItemStatus = items.ItemStatus;
const MessageRole = items.MessageRole;
const ContentPart = items.ContentPart;
const ContentType = items.ContentType;

// =============================================================================
// Public API
// =============================================================================

/// Serialize an Item to Open Responses JSON (response direction).
///
/// Produces a complete, schema-compliant JSON object for storage.
/// The output includes all fields (logprobs, annotations, reasoning content).
///
/// Caller owns returned memory.
pub fn serializeItemToJson(allocator: std.mem.Allocator, item: *const Item) ![]u8 {
    var json_bytes: std.ArrayListUnmanaged(u8) = .{};
    errdefer json_bytes.deinit(allocator);
    const writer = json_bytes.writer(allocator);

    try writeItemJson(writer, item);

    return json_bytes.toOwnedSlice(allocator);
}

/// Serialize an Item to a null-terminated JSON string (for C API).
///
/// Returns a sentinel-terminated slice suitable for C interop.
/// Caller owns returned memory.
pub fn serializeItemToJsonZ(allocator: std.mem.Allocator, item: *const Item) ![:0]u8 {
    var json_bytes: std.ArrayListUnmanaged(u8) = .{};
    errdefer json_bytes.deinit(allocator);
    const writer = json_bytes.writer(allocator);

    try writeItemJson(writer, item);

    // Add null terminator
    return json_bytes.toOwnedSliceSentinel(allocator, 0);
}

/// Extract the role from an Item as a u8 enum value.
///
/// Returns the MessageRole enum value for messages, or 255 (N/A) for other item types.
pub fn extractRole(item: *const Item) u8 {
    return switch (item.data) {
        .message => |m| @intFromEnum(m.role),
        else => 255, // N/A sentinel for non-message items
    };
}

// =============================================================================
// Internal Serialization
// =============================================================================

/// Map ItemStatus to storage JSON string (full fidelity).
fn statusToString(status: ItemStatus) []const u8 {
    return switch (status) {
        .in_progress => "in_progress",
        .waiting => "waiting",
        .completed => "completed",
        .incomplete => "incomplete",
        .failed => "failed",
    };
}

/// Write a single item as JSON (response direction - full fidelity).
fn writeItemJson(writer: anytype, item: *const Item) !void {
    switch (item.data) {
        .message => |*m| {
            try writer.print("{{\"type\":\"message\",\"id\":\"{s}{d}\",\"role\":\"{s}\",\"status\":\"{s}\",\"content\":", .{
                "msg_",
                item.id,
                m.getRoleString(),
                statusToString(m.status),
            });
            try writeContentPartsJson(writer, m.content.items);
            if (item.generation_json) |gen| {
                try writer.writeAll(",\"generation\":");
                try writer.writeAll(gen);
            }
            try writer.writeByte('}');
        },
        .function_call => |*f| {
            try writer.print("{{\"type\":\"function_call\",\"id\":\"{s}{d}\",\"call_id\":\"{s}\",\"name\":\"{s}\",\"arguments\":{f},\"status\":\"{s}\"}}", .{
                "fc_",
                item.id,
                f.call_id,
                f.name,
                std.json.fmt(f.arguments.items, .{}),
                statusToString(f.status),
            });
        },
        .function_call_output => |*f| {
            try writer.print("{{\"type\":\"function_call_output\",\"id\":\"{s}{d}\",\"call_id\":\"{s}\",\"output\":", .{
                "fco_",
                item.id,
                f.call_id,
            });
            // Output can be string or array (union type)
            switch (f.output) {
                .text => |t| {
                    try writer.print("{f}", .{std.json.fmt(t.items, .{})});
                },
                .parts => |p| {
                    try writeContentPartsJson(writer, p.items);
                },
            }
            try writer.print(",\"status\":\"{s}\"}}", .{statusToString(f.status)});
        },
        .reasoning => |*r| {
            // Response direction: include all content (full fidelity for storage)
            try writer.print("{{\"type\":\"reasoning\",\"id\":\"{s}{d}\",\"summary\":", .{ "rs_", item.id });

            // Response direction: preserve ALL content types in summary
            try writeContentPartsJson(writer, r.summary.items);

            // Content: include for storage (response direction)
            if (r.content.items.len > 0) {
                try writer.writeAll(",\"content\":");
                try writeContentPartsJson(writer, r.content.items);
            }

            // Emit encrypted_content if present
            if (r.encrypted_content) |e| {
                try writer.print(",\"encrypted_content\":{f}", .{std.json.fmt(e, .{})});
            }
            try writer.writeByte('}');
        },
        .item_reference => |*ref| {
            try writer.print("{{\"type\":\"item_reference\",\"id\":\"{s}\"}}", .{ref.id});
        },
        .unknown => |*u| {
            try writer.print("{{\"type\":\"{s}\",\"payload\":{s}}}", .{ u.raw_type, u.payload });
        },
    }
}

/// Write content parts array as JSON (response direction - full fidelity).
fn writeContentPartsJson(writer: anytype, parts: []const ContentPart) !void {
    try writer.writeByte('[');

    var first = true;
    for (parts) |*part| {
        if (!first) try writer.writeByte(',');
        first = false;

        switch (part.variant) {
            .input_text => |v| {
                try writer.print("{{\"type\":\"input_text\",\"text\":{f}}}", .{
                    std.json.fmt(v.text.items, .{}),
                });
            },
            .text => |v| {
                try writer.print("{{\"type\":\"text\",\"text\":{f}}}", .{
                    std.json.fmt(v.text.items, .{}),
                });
            },
            .reasoning_text => |v| {
                try writer.print("{{\"type\":\"reasoning_text\",\"text\":{f}}}", .{
                    std.json.fmt(v.text.items, .{}),
                });
            },
            .summary_text => {
                try writer.print("{{\"type\":\"summary_text\",\"text\":{f}}}", .{
                    std.json.fmt(part.getData(), .{}),
                });
            },
            .output_text => |v| {
                try writer.print("{{\"type\":\"output_text\",\"text\":{f}", .{
                    std.json.fmt(v.text.items, .{}),
                });
                // Response direction: include logprobs if present
                if (v.logprobs_json) |l| {
                    try writer.print(",\"logprobs\":{s}", .{l});
                }
                // Response direction: include all annotations
                if (v.annotations_json) |a| {
                    try writer.print(",\"annotations\":{s}", .{a});
                }
                try writer.writeByte('}');
            },
            .refusal => |v| {
                try writer.print("{{\"type\":\"refusal\",\"refusal\":{f}}}", .{
                    std.json.fmt(v.refusal.items, .{}),
                });
            },
            .input_image => |v| {
                try writer.print("{{\"type\":\"input_image\",\"image_url\":{f}", .{
                    std.json.fmt(v.image_url.items, .{}),
                });
                if (v.detail != .auto) {
                    const detail_str = switch (v.detail) {
                        .low => "low",
                        .high => "high",
                        .auto => "auto",
                    };
                    try writer.print(",\"detail\":\"{s}\"", .{detail_str});
                }
                try writer.writeByte('}');
            },
            .input_audio => |v| {
                try writer.print("{{\"type\":\"input_audio\",\"audio_data\":{f}}}", .{
                    std.json.fmt(v.audio_data.items, .{}),
                });
            },
            .input_video => |v| {
                try writer.print("{{\"type\":\"input_video\",\"video_url\":{f}}}", .{
                    std.json.fmt(v.video_url.items, .{}),
                });
            },
            .input_file => |v| {
                try writer.writeAll("{\"type\":\"input_file\"");
                if (v.filename) |f| {
                    try writer.print(",\"filename\":{f}", .{std.json.fmt(f.items, .{})});
                }
                if (v.file_data) |d| {
                    try writer.print(",\"file_data\":{f}", .{std.json.fmt(d.items, .{})});
                }
                if (v.file_url) |u| {
                    try writer.print(",\"file_url\":{f}", .{std.json.fmt(u.items, .{})});
                }
                try writer.writeByte('}');
            },
            .unknown => |v| {
                // For storage, preserve unknown content types as-is
                try writer.print("{{\"type\":\"{s}\",\"data\":{s}}}", .{ v.raw_type, v.raw_data });
            },
        }
    }

    try writer.writeByte(']');
}

// =============================================================================
// Tests
// =============================================================================

test "serializeItemToJson - basic message" {
    const allocator = std.testing.allocator;

    // Create a simple message item
    var content_list = std.ArrayListUnmanaged(ContentPart){};
    defer content_list.deinit(allocator);

    var text_buf = std.ArrayListUnmanaged(u8){};
    defer text_buf.deinit(allocator);
    try text_buf.appendSlice(allocator, "Hello, world!");

    try content_list.append(allocator, .{
        .variant = .{ .input_text = .{ .text = text_buf } },
    });

    const item = Item{
        .id = 42,
        .created_at_ms = 1234567890,
        .data = .{
            .message = .{
                .role = .user,
                .status = .completed,
                .content = content_list,
            },
        },
    };

    const json = try serializeItemToJson(allocator, &item);
    defer allocator.free(json);

    // Verify it contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"message\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"id\":\"msg_42\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\":\"completed\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "Hello, world!") != null);
}

test "extractRole - message" {
    const item = Item{
        .id = 1,
        .created_at_ms = 0,
        .data = .{
            .message = .{
                .role = .assistant,
                .status = .completed,
                .content = .{},
            },
        },
    };

    try std.testing.expectEqual(@as(u8, 2), extractRole(&item)); // assistant = 2
}

test "extractRole - non-message returns 255" {
    const item = Item{
        .id = 1,
        .created_at_ms = 0,
        .data = .{
            .function_call = .{
                .call_id = "call_1",
                .name = "test",
                .arguments = .{},
                .status = .completed,
            },
        },
    };

    try std.testing.expectEqual(@as(u8, 255), extractRole(&item)); // N/A sentinel
}

test "serializeItemToJsonZ - returns null-terminated string" {
    const allocator = std.testing.allocator;

    // Create a simple message item
    var content_list = std.ArrayListUnmanaged(ContentPart){};
    defer content_list.deinit(allocator);

    var text_buf = std.ArrayListUnmanaged(u8){};
    defer text_buf.deinit(allocator);
    try text_buf.appendSlice(allocator, "Test message");

    try content_list.append(allocator, .{
        .variant = .{ .input_text = .{ .text = text_buf } },
    });

    const item = Item{
        .id = 1,
        .created_at_ms = 1234567890,
        .data = .{
            .message = .{
                .role = .user,
                .status = .completed,
                .content = content_list,
            },
        },
    };

    const json_z = try serializeItemToJsonZ(allocator, &item);
    defer allocator.free(json_z);

    // Verify null termination
    try std.testing.expectEqual(@as(u8, 0), json_z[json_z.len]);

    // Verify content is valid JSON with expected fields
    try std.testing.expect(std.mem.indexOf(u8, json_z, "\"type\":\"message\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_z, "Test message") != null);
}

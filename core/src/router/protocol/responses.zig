//! Responses Protocol Adapter (Input Parsing)
//!
//! Parses the OpenResponses `input` field (string or ItemParam array) into
//! a Conversation's Item-based format.
//!
//! Supports:
//!   - String shorthand: treated as a user message
//!   - Array of ItemParam objects with `type` discriminator:
//!     - "message" (user/system/developer/assistant)
//!     - "function_call" (call_id, name, arguments)
//!     - "function_call_output" (call_id, output)
//!     - "item_reference" (id)
//!
//! This module is the ONLY place that knows about Responses input format
//! parsing. All format conversion logic lives in the protocol module.

const std = @import("std");
const io = @import("../../io/root.zig");
const responses_mod = @import("../../responses/root.zig");
const Conversation = responses_mod.Conversation;
const ContentType = responses_mod.ContentType;
const MessageRole = responses_mod.MessageRole;

/// Parse OpenResponses input JSON into a Conversation.
///
/// The input can be:
///   - A JSON string: treated as a user message
///   - A JSON array of ItemParam objects
///
/// Does NOT clear existing items — appends to the conversation.
pub fn parse(conv: *Conversation, json: []const u8) !void {
    const parsed = io.json.parseValue(conv.allocator, json, .{
        .max_size_bytes = 10 * 1024 * 1024,
        .max_value_bytes = 10 * 1024 * 1024,
        .max_string_bytes = 1 * 1024 * 1024,
    }) catch |err| {
        return switch (err) {
            error.InputTooLarge => error.InvalidJson,
            error.InputTooDeep => error.InvalidJson,
            error.StringTooLong => error.InvalidJson,
            error.InvalidJson => error.InvalidJson,
            error.OutOfMemory => error.OutOfMemory,
        };
    };
    defer parsed.deinit();

    switch (parsed.value) {
        .string => |s| {
            _ = try conv.appendUserMessage(s);
        },
        .array => |arr| {
            for (arr.items) |item_value| {
                try parseItem(conv, item_value);
            }
        },
        else => return error.InvalidJson,
    }
}

fn parseItem(conv: *Conversation, value: std.json.Value) !void {
    const obj = switch (value) {
        .object => |o| o,
        else => return error.InvalidJson,
    };

    const type_str = switch (obj.get("type") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };

    if (std.mem.eql(u8, type_str, "message")) {
        try parseMessage(conv, obj);
    } else if (std.mem.eql(u8, type_str, "function_call")) {
        try parseFunctionCall(conv, obj);
    } else if (std.mem.eql(u8, type_str, "function_call_output")) {
        try parseFunctionCallOutput(conv, obj);
    } else if (std.mem.eql(u8, type_str, "item_reference")) {
        try parseItemReference(conv, obj);
    } else {
        // Unknown type — skip rather than fail, for forward compatibility
    }
}

fn parseMessage(conv: *Conversation, obj: std.json.ObjectMap) !void {
    const role_str = switch (obj.get("role") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };

    const role: MessageRole = if (std.mem.eql(u8, role_str, "user"))
        .user
    else if (std.mem.eql(u8, role_str, "system"))
        .system
    else if (std.mem.eql(u8, role_str, "developer"))
        .developer
    else if (std.mem.eql(u8, role_str, "assistant"))
        .assistant
    else
        return error.InvalidJson;

    const content_value = obj.get("content") orelse {
        // No content — create empty message (valid for assistant)
        if (role == .assistant) {
            const msg = try conv.appendAssistantMessage();
            conv.finalizeItem(msg);
        }
        return;
    };

    switch (content_value) {
        .string => |s| {
            switch (role) {
                .system => _ = try conv.appendSystemMessage(s),
                .developer => _ = try conv.appendDeveloperMessage(s),
                .user => _ = try conv.appendUserMessage(s),
                .assistant => {
                    const msg = try conv.appendAssistantMessage();
                    try conv.appendTextContent(msg, s);
                    conv.finalizeItem(msg);
                },
                else => return error.InvalidJson,
            }
        },
        .array => |parts_array| {
            const msg = try conv.appendEmptyMessage(role);
            errdefer _ = conv.deleteItem(conv.len() - 1);

            for (parts_array.items) |part_value| {
                const part_obj = switch (part_value) {
                    .object => |o| o,
                    else => return error.InvalidJson,
                };

                const part_type = switch (part_obj.get("type") orelse return error.InvalidJson) {
                    .string => |s| s,
                    else => return error.InvalidJson,
                };

                if (std.mem.eql(u8, part_type, "input_text")) {
                    const text = switch (part_obj.get("text") orelse return error.InvalidJson) {
                        .string => |s| s,
                        else => return error.InvalidJson,
                    };
                    const part = try conv.addContentPart(msg, ContentType.input_text);
                    try part.appendData(conv.allocator, text);
                } else if (std.mem.eql(u8, part_type, "output_text")) {
                    const text = switch (part_obj.get("text") orelse return error.InvalidJson) {
                        .string => |s| s,
                        else => return error.InvalidJson,
                    };
                    const part = try conv.addContentPart(msg, ContentType.output_text);
                    try part.appendData(conv.allocator, text);
                } else if (std.mem.eql(u8, part_type, "input_image")) {
                    const image_url = switch (part_obj.get("image_url") orelse return error.InvalidJson) {
                        .string => |s| s,
                        else => return error.InvalidJson,
                    };
                    const part = try conv.addContentPart(msg, ContentType.input_image);
                    try part.appendData(conv.allocator, image_url);
                } else if (std.mem.eql(u8, part_type, "input_file")) {
                    const part = try conv.addContentPart(msg, ContentType.input_file);
                    if (part_obj.get("file_data")) |fd_val| {
                        const file_data = switch (fd_val) {
                            .string => |s| s,
                            else => return error.InvalidJson,
                        };
                        try part.appendData(conv.allocator, file_data);
                    }
                }
                // Unknown part types are silently skipped for forward compatibility.
            }
            conv.finalizeItem(msg);
        },
        else => return error.InvalidJson,
    }
}

fn parseFunctionCall(conv: *Conversation, obj: std.json.ObjectMap) !void {
    const call_id = switch (obj.get("call_id") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };
    const name = switch (obj.get("name") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };
    const arguments = switch (obj.get("arguments") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };

    const fc = try conv.appendFunctionCall(call_id, name);
    try conv.setFunctionCallArguments(fc, arguments);
}

fn parseFunctionCallOutput(conv: *Conversation, obj: std.json.ObjectMap) !void {
    const call_id = switch (obj.get("call_id") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };

    // Output can be a string or array of parts. For now, support string.
    const output_value = obj.get("output") orelse return error.InvalidJson;
    const output = switch (output_value) {
        .string => |s| s,
        else => return error.InvalidJson,
    };

    _ = try conv.appendFunctionCallOutput(call_id, output);
}

fn parseItemReference(conv: *Conversation, obj: std.json.ObjectMap) !void {
    const id = switch (obj.get("id") orelse return error.InvalidJson) {
        .string => |s| s,
        else => return error.InvalidJson,
    };

    _ = try conv.appendItemReference(id);
}

// =============================================================================
// Tests
// =============================================================================

test "parse string input" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    try parse(conv, "\"Hello, world!\"");
    try std.testing.expectEqual(@as(usize, 1), conv.len());
}

test "parse array with message items" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    const json =
        \\[
        \\  {"type": "message", "role": "system", "content": "You are helpful."},
        \\  {"type": "message", "role": "user", "content": "Hi!"},
        \\  {"type": "message", "role": "assistant", "content": "Hello!"}
        \\]
    ;

    try parse(conv, json);
    try std.testing.expectEqual(@as(usize, 3), conv.len());
}

test "parse function_call and function_call_output items" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    const json =
        \\[
        \\  {"type": "message", "role": "user", "content": "What is the weather?"},
        \\  {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"location\":\"NYC\"}"},
        \\  {"type": "function_call_output", "call_id": "call_1", "output": "72°F, sunny"}
        \\]
    ;

    try parse(conv, json);
    try std.testing.expectEqual(@as(usize, 3), conv.len());
}

test "parse item_reference" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    const json =
        \\[{"type": "item_reference", "id": "msg_abc123"}]
    ;

    try parse(conv, json);
    try std.testing.expectEqual(@as(usize, 1), conv.len());
}

test "parse multimodal content array" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    const json =
        \\[{"type": "message", "role": "user", "content": [
        \\  {"type": "input_text", "text": "What is this?"}
        \\]}]
    ;

    try parse(conv, json);
    try std.testing.expectEqual(@as(usize, 1), conv.len());
}

test "parse appends to existing conversation" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    _ = try conv.appendSystemMessage("system prompt");
    try parse(conv, "\"user message\"");
    try std.testing.expectEqual(@as(usize, 2), conv.len());
}

test "parse unknown type is skipped" {
    const alloc = std.testing.allocator;
    const conv = try Conversation.init(alloc);
    defer conv.deinit();

    const json =
        \\[
        \\  {"type": "unknown_future_type", "data": "whatever"},
        \\  {"type": "message", "role": "user", "content": "Hi"}
        \\]
    ;

    try parse(conv, json);
    try std.testing.expectEqual(@as(usize, 1), conv.len());
}

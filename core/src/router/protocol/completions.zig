//! Completions Protocol Adapter
//!
//! Transforms Item-based Conversation data into the legacy Chat Completions
//! format used by OpenAI v1, Anthropic, Ollama, vLLM, etc.
//!
//! This module is the ONLY place that knows about Completions format conversion.
//! The responses module (`core/src/responses/`) should be pure Item-based.
//!
//! # Folding Algorithm (Backward-Attach)
//!
//! The typical item sequence in Item-based format is:
//!   assistant message → function_call → function_call_output
//!
//! Tool calls must be attached to the PRECEDING assistant message's tool_calls
//! array, not the next one. This requires buffering assistant messages until
//! we know if function_calls follow.
//!
//! Algorithm:
//! 1. When we see an assistant message, DON'T emit it yet - buffer it
//! 2. When we see function_call items, collect them
//! 3. When we see something else (non-assistant, non-function_call), flush:
//!    - Emit the buffered assistant WITH any collected tool_calls
//!    - Then emit the current item
//! 4. If no assistant was buffered when we see function_calls, they become
//!    orphaned and we synthesize an assistant for them only when needed
//!
//! This prevents invalid consecutive assistant messages and correctly associates
//! tool_calls with the assistant message that generated them.
//!
//! # Folding Rules
//!
//! 1. Messages: Direct map. Developer role maps to system.
//! 2. Item References: Dereference - look up target item and inject content.
//! 3. Function Calls: Fold backward into previous assistant's tool_calls array.
//! 4. Function Outputs: Map to role="tool" messages with tool_call_id.
//! 5. Reasoning: Configurable - strip or emit summary as system message.

const std = @import("std");
const io = @import("../../io/root.zig");
const responses_mod = @import("../../responses/root.zig");
const Conversation = responses_mod.Conversation;
const Item = responses_mod.Item;
const ContentPart = responses_mod.ContentPart;
const MessageData = responses_mod.MessageData;

// =============================================================================
// Public API
// =============================================================================

/// Options for Completions projection.
pub const Options = struct {
    pub const ImageContentType = enum {
        image_url,
        image,
    };

    /// Include reasoning in output (emit summary as system message).
    include_reasoning: bool = false,

    /// Dereference item_reference items (look up and inline content).
    dereference_references: bool = true,

    /// Content "type" value for image parts.
    /// - `image_url`: OpenAI Chat Completions style.
    /// - `image`: HF chat-template style (e.g., templates that emit `<image>`).
    image_content_type: ImageContentType = .image_url,
};

/// Serialize a Conversation to Chat Completions JSON format.
///
/// This performs the "Folding" transform from Item-based format to legacy format.
/// Caller owns the returned memory.
///
/// Example output:
/// ```json
/// [
///   {"role": "system", "content": "You are helpful."},
///   {"role": "user", "content": "Hello!"},
///   {"role": "assistant", "content": "Hi there!", "tool_calls": [...]},
///   {"role": "tool", "tool_call_id": "call_123", "content": "result"}
/// ]
/// ```
pub fn serialize(
    allocator: std.mem.Allocator,
    conv: *const Conversation,
    opts: Options,
) ![]u8 {
    var json_buffer = std.ArrayListUnmanaged(u8){};
    errdefer json_buffer.deinit(allocator);
    var writer = json_buffer.writer(allocator);

    try writer.writeByte('[');

    // Buffered assistant message (waiting to see if function_calls follow)
    var pending_assistant: ?PendingAssistant = null;
    defer if (pending_assistant) |*pa| pa.deinit(allocator);

    // Collected function_calls to attach to the pending assistant
    var pending_tool_calls: std.ArrayListUnmanaged(PendingToolCall) = .{};
    defer {
        for (pending_tool_calls.items) |tc| {
            allocator.free(tc.id);
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        pending_tool_calls.deinit(allocator);
    }

    var first = true;
    var i: usize = 0;

    while (i < conv.items_list.items.len) : (i += 1) {
        const item = &conv.items_list.items[i];

        switch (item.data.getType()) {
            .message => {
                const msg = switch (item.data) {
                    .message => |*m| m,
                    else => unreachable,
                };

                if (msg.role == .assistant) {
                    // Flush any previous pending assistant (with its tool_calls)
                    if (pending_assistant) |*pa| {
                        try flushPendingAssistant(allocator, writer, pa, pending_tool_calls.items, &first);
                        pa.deinit(allocator);
                        pending_assistant = null;

                        for (pending_tool_calls.items) |tc| {
                            allocator.free(tc.id);
                            allocator.free(tc.name);
                            allocator.free(tc.arguments);
                        }
                        pending_tool_calls.clearRetainingCapacity();
                    } else if (pending_tool_calls.items.len > 0) {
                        // Orphaned tool_calls with no assistant - emit synthetic
                        if (!first) try writer.writeByte(',');
                        first = false;
                        try writeToolCallsMessage(writer, pending_tool_calls.items);

                        for (pending_tool_calls.items) |tc| {
                            allocator.free(tc.id);
                            allocator.free(tc.name);
                            allocator.free(tc.arguments);
                        }
                        pending_tool_calls.clearRetainingCapacity();
                    }

                    // Buffer this assistant message (don't emit yet)
                    pending_assistant = try PendingAssistant.capture(allocator, msg);
                } else {
                    // Non-assistant message: flush pending state first
                    if (pending_assistant) |*pa| {
                        try flushPendingAssistant(allocator, writer, pa, pending_tool_calls.items, &first);
                        pa.deinit(allocator);
                        pending_assistant = null;

                        for (pending_tool_calls.items) |tc| {
                            allocator.free(tc.id);
                            allocator.free(tc.name);
                            allocator.free(tc.arguments);
                        }
                        pending_tool_calls.clearRetainingCapacity();
                    } else if (pending_tool_calls.items.len > 0) {
                        // Orphaned tool_calls - emit synthetic assistant
                        if (!first) try writer.writeByte(',');
                        first = false;
                        try writeToolCallsMessage(writer, pending_tool_calls.items);

                        for (pending_tool_calls.items) |tc| {
                            allocator.free(tc.id);
                            allocator.free(tc.name);
                            allocator.free(tc.arguments);
                        }
                        pending_tool_calls.clearRetainingCapacity();
                    }

                    // Emit this non-assistant message
                    if (!first) try writer.writeByte(',');
                    first = false;

                    const role_str = switch (msg.role) {
                        .system => "system",
                        .user => "user",
                        .assistant => "assistant",
                        .developer => "system", // Developer maps to system
                        .unknown => "user", // Fallback for forward compat
                    };

                    try writer.print("{{\"role\":\"{s}\",\"content\":", .{role_str});
                    try writeMessageContent(writer, msg, opts);
                    try writer.writeByte('}');
                }
            },

            .function_call => {
                const fc = switch (item.data) {
                    .function_call => |*f| f,
                    else => unreachable,
                };

                // Collect tool calls - they'll be attached to the pending assistant
                try pending_tool_calls.append(allocator, .{
                    .id = try allocator.dupe(u8, fc.call_id),
                    .name = try allocator.dupe(u8, fc.name),
                    .arguments = try allocator.dupe(u8, fc.arguments.items),
                });
            },

            .function_call_output => {
                const fco = switch (item.data) {
                    .function_call_output => |*f| f,
                    else => unreachable,
                };

                // Before tool output, flush pending assistant with its tool_calls
                if (pending_assistant) |*pa| {
                    try flushPendingAssistant(allocator, writer, pa, pending_tool_calls.items, &first);
                    pa.deinit(allocator);
                    pending_assistant = null;

                    for (pending_tool_calls.items) |tc| {
                        allocator.free(tc.id);
                        allocator.free(tc.name);
                        allocator.free(tc.arguments);
                    }
                    pending_tool_calls.clearRetainingCapacity();
                } else if (pending_tool_calls.items.len > 0) {
                    // Orphaned tool_calls (no assistant) - emit synthetic
                    if (!first) try writer.writeByte(',');
                    first = false;
                    try writeToolCallsMessage(writer, pending_tool_calls.items);

                    for (pending_tool_calls.items) |tc| {
                        allocator.free(tc.id);
                        allocator.free(tc.name);
                        allocator.free(tc.arguments);
                    }
                    pending_tool_calls.clearRetainingCapacity();
                }

                // Tool output -> role="tool" message
                if (!first) try writer.writeByte(',');
                first = false;

                try writer.print("{{\"role\":\"tool\",\"tool_call_id\":\"{s}\",\"content\":", .{fco.call_id});

                // Get output text
                const output_text = fco.getOutputText();
                try writer.print("{f}", .{std.json.fmt(output_text, .{})});
                try writer.writeByte('}');
            },

            .reasoning => {
                // Reasoning: optionally include as system message or skip.
                //
                // IMPORTANT (Schema Constraint per Addendum #2):
                // - As Output (ReasoningBody): can contain content + summary
                // - As Input (ReasoningItemParam): content MUST be NULL, only summary allowed
                //
                // For Completions projection (legacy input format), we ONLY emit summary
                // as a system message. The content field is intentionally NOT emitted.
                if (opts.include_reasoning) {
                    const rd = switch (item.data) {
                        .reasoning => |*r| r,
                        else => unreachable,
                    };

                    // Emit only summary (content is intentionally excluded per schema)
                    const summary = rd.getSummaryText();
                    if (summary.len > 0) {
                        if (!first) try writer.writeByte(',');
                        first = false;

                        try writer.writeAll("{\"role\":\"system\",\"content\":");
                        try writer.print("{f}", .{std.json.fmt(summary, .{})});
                        try writer.writeByte('}');
                    }
                }
                // Otherwise skip reasoning items
            },

            .item_reference => {
                // Item references: dereference if enabled
                if (opts.dereference_references) {
                    const ref = switch (item.data) {
                        .item_reference => |*r| r,
                        else => unreachable,
                    };

                    // Parse the ID to look up
                    const target_id = parseItemId(ref.id);
                    if (target_id) |tid| {
                        if (conv.findById(tid)) |target| {
                            // Recursively emit this item
                            // (Skip to avoid infinite loops with self-references)
                            if (target.id != item.id) {
                                // For simplicity, just emit target message content inline
                                if (target.asMessage()) |target_msg| {
                                    if (!first) try writer.writeByte(',');
                                    first = false;

                                    const role_str = switch (target_msg.role) {
                                        .system => "system",
                                        .user => "user",
                                        .assistant => "assistant",
                                        .developer => "system",
                                        .unknown => "user",
                                    };

                                    try writer.print("{{\"role\":\"{s}\",\"content\":\"", .{role_str});
                                    for (target_msg.content.items) |*part| {
                                        switch (part.variant) {
                                            .input_text, .output_text, .text => {
                                                try writeJsonEscaped(writer, part.getData());
                                            },
                                            else => {},
                                        }
                                    }
                                    try writer.writeAll("\"}");
                                }
                            }
                        }
                    }
                }
                // Otherwise skip reference items
            },

            .unknown => {
                // Skip unknown items in legacy format
            },
        }
    }

    // Flush any remaining pending assistant (with tool_calls if any)
    if (pending_assistant) |*pa| {
        try flushPendingAssistant(allocator, writer, pa, pending_tool_calls.items, &first);
        // Note: pa.deinit handled by defer
    } else if (pending_tool_calls.items.len > 0) {
        // Orphaned tool_calls at end - emit synthetic assistant
        if (!first) try writer.writeByte(',');
        try writeToolCallsMessage(writer, pending_tool_calls.items);
    }

    try writer.writeByte(']');

    return json_buffer.toOwnedSlice(allocator);
}

// =============================================================================
// Internal Types
// =============================================================================

/// Buffered assistant message for backward-attach folding.
const PendingAssistant = struct {
    content: []u8,
    is_multimodal: bool,

    fn capture(allocator: std.mem.Allocator, msg: *const MessageData) !PendingAssistant {
        // Serialize content to buffer
        var buf = std.ArrayListUnmanaged(u8){};
        errdefer buf.deinit(allocator);
        const buf_writer = buf.writer(allocator);

        // Check if multimodal
        var has_non_text = false;
        for (msg.content.items) |*part| {
            switch (part.variant) {
                .input_text, .output_text, .text => {},
                else => {
                    has_non_text = true;
                    break;
                },
            }
        }

        if (has_non_text) {
            try writeContentPartsCompletions(buf_writer, msg.content.items, .{});
        } else {
            try buf_writer.writeByte('"');
            for (msg.content.items) |*part| {
                switch (part.variant) {
                    .input_text, .output_text, .text => {
                        try writeJsonEscaped(buf_writer, part.getData());
                    },
                    else => {},
                }
            }
            try buf_writer.writeByte('"');
        }

        return .{
            .content = try buf.toOwnedSlice(allocator),
            .is_multimodal = has_non_text,
        };
    }

    fn deinit(self: *PendingAssistant, allocator: std.mem.Allocator) void {
        allocator.free(self.content);
    }
};

/// Pending tool call for folding.
const PendingToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8,
};

// =============================================================================
// Internal Helpers
// =============================================================================

/// Flush a pending assistant message with optional tool_calls.
fn flushPendingAssistant(
    allocator: std.mem.Allocator,
    writer: anytype,
    pa: *const PendingAssistant,
    tool_calls: []const PendingToolCall,
    first: *bool,
) !void {
    _ = allocator;

    if (!first.*) try writer.writeByte(',');
    first.* = false;

    try writer.writeAll("{\"role\":\"assistant\",\"content\":");
    try writer.writeAll(pa.content);

    if (tool_calls.len > 0) {
        try writer.writeAll(",\"tool_calls\":[");
        for (tool_calls, 0..) |tc, j| {
            if (j > 0) try writer.writeByte(',');
            try writer.print("{{\"id\":\"{s}\",\"type\":\"function\",\"function\":{{\"name\":\"{s}\",\"arguments\":", .{
                tc.id,
                tc.name,
            });
            try writer.print("{f}", .{std.json.fmt(tc.arguments, .{})});
            try writer.writeAll("}}");
        }
        try writer.writeByte(']');
    }

    try writer.writeByte('}');
}

/// Write message content (shared by assistant and non-assistant paths).
fn writeMessageContent(writer: anytype, msg: *const MessageData, opts: Options) !void {
    // Check if multimodal
    var has_non_text = false;
    for (msg.content.items) |*part| {
        switch (part.variant) {
            .input_text, .output_text, .text => {},
            else => {
                has_non_text = true;
                break;
            },
        }
    }

    if (has_non_text) {
        try writeContentPartsCompletions(writer, msg.content.items, opts);
    } else {
        try writer.writeByte('"');
        for (msg.content.items) |*part| {
            switch (part.variant) {
                .input_text, .output_text, .text => {
                    try writeJsonEscaped(writer, part.getData());
                },
                else => {},
            }
        }
        try writer.writeByte('"');
    }
}

/// Write a synthetic assistant message with tool_calls (no content).
fn writeToolCallsMessage(writer: anytype, tool_calls: []const PendingToolCall) !void {
    try writer.writeAll("{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[");
    for (tool_calls, 0..) |tc, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.print("{{\"id\":\"{s}\",\"type\":\"function\",\"function\":{{\"name\":\"{s}\",\"arguments\":", .{
            tc.id,
            tc.name,
        });
        try writer.print("{f}", .{std.json.fmt(tc.arguments, .{})});
        try writer.writeAll("}}");
    }
    try writer.writeAll("]}");
}

/// Write content parts in Completions format (for multimodal).
fn writeContentPartsCompletions(writer: anytype, parts: []const ContentPart, opts: Options) !void {
    try writer.writeByte('[');

    for (parts, 0..) |*part, j| {
        if (j > 0) try writer.writeByte(',');

        switch (part.variant) {
            .input_text, .output_text, .text => {
                try writer.writeAll("{\"type\":\"text\",\"text\":");
                try writer.print("{f}", .{std.json.fmt(part.getData(), .{})});
                try writer.writeByte('}');
            },
            .input_image => |img| {
                switch (opts.image_content_type) {
                    .image_url => {
                        try writer.writeAll("{\"type\":\"image_url\",\"image_url\":{\"url\":");
                        try writer.print("{f}", .{std.json.fmt(part.getData(), .{})});
                        try writer.print(",\"detail\":\"{s}\"}}}}", .{img.detail.toString()});
                    },
                    .image => {
                        try writer.writeAll("{\"type\":\"image\",\"image\":");
                        try writer.print("{f}", .{std.json.fmt(part.getData(), .{})});
                        try writer.writeAll(",\"image_url\":{\"url\":");
                        try writer.print("{f}", .{std.json.fmt(part.getData(), .{})});
                        try writer.print(",\"detail\":\"{s}\"}}", .{img.detail.toString()});
                        try writer.writeByte('}');
                    },
                }
            },
            else => {
                // Other types: emit as generic data
                try writer.print("{{\"type\":\"{s}\",\"data\":", .{part.getContentType().toString()});
                try writer.print("{f}", .{std.json.fmt(part.getData(), .{})});
                try writer.writeByte('}');
            },
        }
    }

    try writer.writeByte(']');
}

/// Write JSON-escaped string content (without surrounding quotes).
fn writeJsonEscaped(writer: anytype, s: []const u8) !void {
    for (s) |c| {
        switch (c) {
            '"' => try writer.writeAll("\\\""),
            '\\' => try writer.writeAll("\\\\"),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            // Other control characters (excluding \n=0x0a, \r=0x0d, \t=0x09)
            0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => |ctrl| try writer.print("\\u{x:0>4}", .{ctrl}),
            else => try writer.writeByte(c),
        }
    }
}

/// Parse an item ID string like "msg_123" to u64.
fn parseItemId(id: []const u8) ?u64 {
    // Find the underscore separator
    const sep = std.mem.indexOf(u8, id, "_") orelse return null;
    const num_part = id[sep + 1 ..];
    return std.fmt.parseInt(u64, num_part, 10) catch null;
}

// =============================================================================
// Tests
// =============================================================================

test "serialize empty conversation" {
    const allocator = std.testing.allocator;

    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const json = try serialize(allocator, conv, .{});
    defer allocator.free(json);

    try std.testing.expectEqualStrings("[]", json);
}

test "serialize simple messages" {
    const allocator = std.testing.allocator;

    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    // Add system message
    _ = try conv.appendSystemMessage("You are helpful.");

    // Add user message
    _ = try conv.appendUserMessage("Hello!");

    // Add assistant message
    const asst = try conv.appendAssistantMessage();
    try conv.appendTextContent(asst, "Hi there!");
    conv.finalizeItem(asst);

    const json = try serialize(allocator, conv, .{});
    defer allocator.free(json);

    // Should have 3 messages
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"system\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"user\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"assistant\"") != null);
}

test "serialize function calls (backward folding)" {
    const allocator = std.testing.allocator;

    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    // Add user message
    _ = try conv.appendUserMessage("What's the weather?");

    // Add assistant message (will have tool_calls attached)
    const asst = try conv.appendAssistantMessage();
    try conv.appendTextContent(asst, "Let me check...");
    conv.finalizeItem(asst);

    // Add function call (should fold backward to assistant)
    const fc = try conv.appendFunctionCall("call_123", "get_weather");
    try conv.setFunctionCallArguments(fc, "{\"city\":\"NYC\"}");

    // Add function output
    _ = try conv.appendFunctionCallOutput("call_123", "Sunny, 72F");

    const json = try serialize(allocator, conv, .{});
    defer allocator.free(json);

    // Assistant should have tool_calls
    try std.testing.expect(std.mem.indexOf(u8, json, "\"tool_calls\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"role\":\"tool\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"tool_call_id\":\"call_123\"") != null);
}

test "serialize multimodal image content supports template image type option" {
    const allocator = std.testing.allocator;

    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const user_msg = try conv.appendEmptyMessage(.user);
    const image_part = try conv.addContentPart(user_msg, .input_image);
    try image_part.appendData(allocator, "data:image/png;base64,AAAA");
    conv.finalizeItem(user_msg);

    const json_default = try serialize(allocator, conv, .{});
    defer allocator.free(json_default);
    try std.testing.expect(std.mem.indexOf(u8, json_default, "\"type\":\"image_url\"") != null);

    const json_template = try serialize(allocator, conv, .{ .image_content_type = .image });
    defer allocator.free(json_template);
    try std.testing.expect(std.mem.indexOf(u8, json_template, "\"type\":\"image\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_template, "\"image_url\"") != null);
}

// =============================================================================
// Parse (Unfolding)
// =============================================================================

/// Parse Chat Completions JSON format into a Conversation.
///
/// This performs the "Unfolding" transform from legacy format to Item-based format.
/// Clears any existing items in the conversation first.
///
/// Supports both formats:
/// - Legacy: `[{"role": "user", "content": "Hello"}]`
/// - Multimodal: `[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]`
pub fn parse(
    conv: *Conversation,
    json: []const u8,
) !void {
    // Clear existing items
    conv.clear();

    // Parse JSON
    const parsed = io.json.parseValue(conv.allocator, json, .{
        .max_size_bytes = 50 * 1024 * 1024,
        .max_value_bytes = 50 * 1024 * 1024,
        .max_string_bytes = 50 * 1024 * 1024,
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

    const array = switch (parsed.value) {
        .array => |arr| arr,
        else => return error.InvalidJson,
    };

    // Create items from parsed messages
    for (array.items) |item_value| {
        const obj = switch (item_value) {
            .object => |o| o,
            else => return error.InvalidJson,
        };

        // Get role
        const role_str = switch (obj.get("role") orelse return error.MissingRole) {
            .string => |s| s,
            else => return error.InvalidRole,
        };

        // Check if this is a tool/function_call_output message
        if (std.mem.eql(u8, role_str, "tool")) {
            // Parse tool output message
            const call_id = switch (obj.get("tool_call_id") orelse return error.MissingToolCallId) {
                .string => |s| s,
                else => return error.InvalidToolCallId,
            };
            const content = switch (obj.get("content") orelse return error.MissingContent) {
                .string => |s| s,
                else => return error.InvalidContent,
            };
            _ = try conv.appendFunctionCallOutput(call_id, content);
            continue;
        }

        // Create appropriate message item based on role
        const role: responses_mod.MessageRole = if (std.mem.eql(u8, role_str, "system"))
            .system
        else if (std.mem.eql(u8, role_str, "developer"))
            .developer
        else if (std.mem.eql(u8, role_str, "user"))
            .user
        else if (std.mem.eql(u8, role_str, "assistant"))
            .assistant
        else
            return error.InvalidRole;

        // Get content - supports both string (legacy) and array (new) formats
        const content_value = obj.get("content");
        if (content_value) |cv| switch (cv) {
            .string => |s| {
                // Legacy format: content is a string - create message with text
                switch (role) {
                    .system => _ = try conv.appendSystemMessage(s),
                    .developer => _ = try conv.appendDeveloperMessage(s),
                    .user => _ = try conv.appendUserMessage(s),
                    .assistant => {
                        const msg = try conv.appendAssistantMessage();
                        try conv.appendTextContent(msg, s);
                        conv.finalizeItem(msg);
                    },
                    else => return error.InvalidRole,
                }
            },
            .array => |parts_array| {
                // Multimodal format: content is array of parts
                const msg = try conv.appendEmptyMessage(role);
                errdefer _ = conv.deleteItem(conv.len() - 1);

                for (parts_array.items) |part_value| {
                    const part_obj = switch (part_value) {
                        .object => |o| o,
                        else => return error.InvalidContent,
                    };

                    // Get part type
                    const type_str = switch (part_obj.get("type") orelse return error.InvalidContent) {
                        .string => |s| s,
                        else => return error.InvalidContent,
                    };

                    if (std.mem.eql(u8, type_str, "text")) {
                        const text = switch (part_obj.get("text") orelse return error.InvalidContent) {
                            .string => |s| s,
                            else => return error.InvalidContent,
                        };
                        try conv.appendTextContent(msg, text);
                    }
                    // Other content types (image_url, etc.) could be handled here
                }
                conv.finalizeItem(msg);
            },
            .null => {
                // Null content is valid for assistant messages with tool_calls
                if (role == .assistant) {
                    const msg = try conv.appendAssistantMessage();
                    conv.finalizeItem(msg);
                } else {
                    return error.InvalidContent;
                }
            },
            else => return error.InvalidContent,
        } else if (role == .assistant) {
            // No content key but might have tool_calls
            const msg = try conv.appendAssistantMessage();
            conv.finalizeItem(msg);
        } else {
            return error.MissingContent;
        }

        // Handle tool_calls on assistant messages
        if (role == .assistant) {
            if (obj.get("tool_calls")) |tool_calls_value| {
                switch (tool_calls_value) {
                    .array => |calls_array| {
                        for (calls_array.items) |call_value| {
                            const call_obj = switch (call_value) {
                                .object => |o| o,
                                else => continue,
                            };

                            // Get call id
                            const call_id = switch (call_obj.get("id") orelse continue) {
                                .string => |s| s,
                                else => continue,
                            };

                            // Get function info
                            const func_obj = switch (call_obj.get("function") orelse continue) {
                                .object => |o| o,
                                else => continue,
                            };

                            const func_name = switch (func_obj.get("name") orelse continue) {
                                .string => |s| s,
                                else => continue,
                            };

                            const func_args = switch (func_obj.get("arguments") orelse continue) {
                                .string => |s| s,
                                else => continue,
                            };

                            // Create function call item
                            const fc = try conv.appendFunctionCall(call_id, func_name);
                            try conv.setFunctionCallArguments(fc, func_args);
                        }
                    },
                    else => {},
                }
            }
        }
    }
}

test "parse simple messages" {
    const allocator = std.testing.allocator;

    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    const json =
        \\[{"role":"system","content":"You are helpful."},
        \\{"role":"user","content":"Hello!"},
        \\{"role":"assistant","content":"Hi there!"}]
    ;

    try parse(conv, json);

    try std.testing.expectEqual(@as(usize, 3), conv.len());
}

test "parse and serialize roundtrip" {
    const allocator = std.testing.allocator;

    const conv = try Conversation.init(allocator);
    defer conv.deinit();

    // Create conversation
    _ = try conv.appendSystemMessage("Be helpful.");
    _ = try conv.appendUserMessage("Test");
    const asst = try conv.appendAssistantMessage();
    try conv.appendTextContent(asst, "Response");
    conv.finalizeItem(asst);

    // Serialize
    const json1 = try serialize(allocator, conv, .{});
    defer allocator.free(json1);

    // Parse into new conversation
    const conv2 = try Conversation.init(allocator);
    defer conv2.deinit();
    try parse(conv2, json1);

    // Serialize again
    const json2 = try serialize(allocator, conv2, .{});
    defer allocator.free(json2);

    // Should match
    try std.testing.expectEqualStrings(json1, json2);
}

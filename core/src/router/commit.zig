//! Post-generation commit logic.
//!
//! After any engine (local or HTTP) produces a generation result, this
//! module commits the result to the chat's Conversation:
//!   - Parses reasoning tags (<think>...</think>) into ReasoningItems
//!   - Checks tool calls against the firewall policy
//!   - Creates AssistantMessage or FunctionCallItem(s)
//!   - Calls finalizeItem (which triggers storage persistence)
//!
//! This is the SINGLE path for all post-generation state changes.
//! Engines produce raw results; this module interprets them.

const std = @import("std");
const responses_mod = @import("../responses/root.zig");
const Chat = responses_mod.Chat;
const reasoning_parser_mod = responses_mod.reasoning_parser;
const firewall = @import("firewall.zig");
const log = @import("../log.zig");

/// Backend-agnostic tool call input for commit.
pub const ToolCallInput = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8,
};

/// Backend-agnostic generation result for commit.
///
/// Both `local.GenerationResult` and `http_engine.GenerationResult` can be
/// projected into this without conversion — callers fill in the fields they have.
pub const CommitParams = struct {
    /// The raw generated text (may contain <think> tags).
    text: []const u8,

    /// Tool calls from the model (empty slice if none).
    /// For local: populated from grammar-constrained JSON parsing.
    /// For HTTP: populated from API response.
    tool_calls: []const ToolCallInput = &.{},

    /// Prompt token count.
    prompt_tokens: usize = 0,

    /// Completion token count.
    completion_tokens: usize = 0,

    /// Prefill latency in nanoseconds.
    prefill_ns: u64 = 0,

    /// Generation latency in nanoseconds.
    generation_ns: u64 = 0,

    /// Finish reason as string (e.g., "stop", "length", "tool_calls").
    finish_reason: [:0]const u8 = "stop",

    /// Reasoning tag name (e.g., "think").
    /// When non-null, the parser looks for `<tag>...</tag>` markers and
    /// separates reasoning from response content into distinct items.
    /// Null uses the default "think" tag.
    reasoning_tag: ?[]const u8 = null,

    /// Generation parameters JSON (model, temperature, top_p, etc.).
    /// Set on assistant message items for tracking what settings produced them.
    generation_json: ?[]const u8 = null,
};

/// Commit a generation result to chat history.
///
/// This is the SINGLE function that both backends (local and HTTP) and
/// all API surfaces (non-streaming generate, streaming iterator) call
/// after generation completes.
///
/// It handles:
///   1. Tool calls: firewall check + FunctionCallItem creation
///   2. Reasoning: <think> tag parsing + ReasoningItem creation
///   3. Normal text: AssistantMessage creation
///   4. finalizeItem (triggers storage persistence)
pub fn commitGenerationResult(
    allocator: std.mem.Allocator,
    chat: *Chat,
    params: CommitParams,
) !void {
    // --- Tool call path ---
    if (params.tool_calls.len > 0) {
        for (params.tool_calls) |tc| {
            const policy_denied = try firewall.checkFirewall(
                allocator,
                chat,
                tc.name,
                tc.arguments,
            );

            const item = try chat.conv.appendFunctionCall(tc.id, tc.name);
            try chat.conv.setFunctionCallArguments(item, tc.arguments);

            item.input_tokens = @intCast(params.prompt_tokens);
            item.output_tokens = @intCast(params.completion_tokens);
            item.prefill_ns = params.prefill_ns;
            item.generation_ns = params.generation_ns;
            item.finish_reason = params.finish_reason;

            chat.conv.finalizeItem(item);

            // Override status after finalize: finalizeItem sets .completed,
            // but policy denial requires .failed for the caller.
            if (policy_denied) {
                item.data.function_call.status = .failed;
            }
        }
        return;
    }

    // --- Text path: parse reasoning tags, create items ---
    var parser = try reasoning_parser_mod.ReasoningParser.init(
        allocator,
        params.reasoning_tag,
    );
    defer parser.deinit();

    try parser.processChunk(params.text);
    const parsed = try parser.finalize();

    if (parsed.reasoning) |reasoning_text| {
        // Create ReasoningItem
        const reasoning_item = try chat.conv.appendReasoning();
        try chat.conv.addReasoningContent(reasoning_item, reasoning_text);
        reasoning_item.input_tokens = @intCast(params.prompt_tokens);

        if (parsed.response) |response_text| {
            // Both reasoning and response
            chat.conv.finalizeItem(reasoning_item);

            const response_item = try chat.conv.appendAssistantMessage();
            try chat.conv.appendTextContent(response_item, response_text);
            response_item.output_tokens = @intCast(params.completion_tokens);
            response_item.prefill_ns = params.prefill_ns;
            response_item.generation_ns = params.generation_ns;
            response_item.finish_reason = params.finish_reason;
            response_item.generation_json = if (params.generation_json) |g|
                try allocator.dupe(u8, g)
            else
                null;
            chat.conv.finalizeItem(response_item);
        } else {
            // Thinking-only
            reasoning_item.output_tokens = @intCast(params.completion_tokens);
            reasoning_item.generation_ns = params.generation_ns;
            reasoning_item.finish_reason = params.finish_reason;
            chat.conv.finalizeItem(reasoning_item);
        }
    } else {
        // No reasoning: single AssistantMessage
        const item = try chat.conv.appendAssistantMessage();
        try chat.conv.appendTextContent(item, params.text);
        item.input_tokens = @intCast(params.prompt_tokens);
        item.output_tokens = @intCast(params.completion_tokens);
        item.prefill_ns = params.prefill_ns;
        item.generation_ns = params.generation_ns;
        item.finish_reason = params.finish_reason;
        item.generation_json = if (params.generation_json) |g|
            try allocator.dupe(u8, g)
        else
            null;
        chat.conv.finalizeItem(item);
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "commitGenerationResult: plain text creates AssistantMessage" {
    const allocator = std.testing.allocator;

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    try commitGenerationResult(allocator, &chat, .{
        .text = "Hello, world!",
        .prompt_tokens = 10,
        .completion_tokens = 5,
        .prefill_ns = 1000,
        .generation_ns = 2000,
        .finish_reason = "stop",
    });

    // Should have created one assistant message
    try std.testing.expectEqual(@as(usize, 1), chat.conv.len());
    const item = chat.conv.getItem(0).?;
    const msg = item.asMessage().?;
    try std.testing.expectEqual(responses_mod.MessageRole.assistant, msg.role);
    try std.testing.expectEqualStrings("Hello, world!", msg.getFirstText());
    try std.testing.expectEqual(@as(u32, 10), item.input_tokens);
    try std.testing.expectEqual(@as(u32, 5), item.output_tokens);
    try std.testing.expectEqual(@as(u64, 1000), item.prefill_ns);
    try std.testing.expectEqual(@as(u64, 2000), item.generation_ns);
    try std.testing.expectEqualStrings("stop", item.finish_reason.?);
}

test "commitGenerationResult: reasoning tags create ReasoningItem + AssistantMessage" {
    const allocator = std.testing.allocator;

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    try commitGenerationResult(allocator, &chat, .{
        .text = "<think>Let me reason about this.</think>The answer is 42.",
        .prompt_tokens = 10,
        .completion_tokens = 20,
        .prefill_ns = 500,
        .generation_ns = 3000,
        .finish_reason = "stop",
    });

    // Should have created two items: ReasoningItem + AssistantMessage
    try std.testing.expectEqual(@as(usize, 2), chat.conv.len());

    // First item: ReasoningItem
    const reasoning_item = chat.conv.getItem(0).?;
    try std.testing.expect(reasoning_item.data == .reasoning);
    try std.testing.expectEqual(@as(u32, 10), reasoning_item.input_tokens);

    // Second item: AssistantMessage
    const msg_item = chat.conv.getItem(1).?;
    const msg = msg_item.asMessage().?;
    try std.testing.expectEqual(responses_mod.MessageRole.assistant, msg.role);
    try std.testing.expectEqualStrings("The answer is 42.", msg.getFirstText());
    try std.testing.expectEqual(@as(u32, 20), msg_item.output_tokens);
    try std.testing.expectEqual(@as(u64, 3000), msg_item.generation_ns);
}

test "commitGenerationResult: thinking-only creates single ReasoningItem" {
    const allocator = std.testing.allocator;

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    try commitGenerationResult(allocator, &chat, .{
        .text = "<think>Just reasoning, no response.</think>",
        .prompt_tokens = 5,
        .completion_tokens = 10,
        .generation_ns = 1500,
        .finish_reason = "stop",
    });

    // Should have created one ReasoningItem (no AssistantMessage)
    try std.testing.expectEqual(@as(usize, 1), chat.conv.len());
    const item = chat.conv.getItem(0).?;
    try std.testing.expect(item.data == .reasoning);
    try std.testing.expectEqual(@as(u32, 10), item.output_tokens);
    try std.testing.expectEqual(@as(u64, 1500), item.generation_ns);
}

test "commitGenerationResult: tool calls create FunctionCallItems" {
    const allocator = std.testing.allocator;

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    const tool_calls = [_]ToolCallInput{
        .{
            .id = "call_abc123",
            .name = "get_weather",
            .arguments = "{\"location\":\"Paris\"}",
        },
    };

    try commitGenerationResult(allocator, &chat, .{
        .text = "",
        .tool_calls = &tool_calls,
        .prompt_tokens = 10,
        .completion_tokens = 15,
        .generation_ns = 2000,
        .finish_reason = "tool_calls",
    });

    // Should have created one FunctionCallItem
    try std.testing.expectEqual(@as(usize, 1), chat.conv.len());
    const item = chat.conv.getItem(0).?;
    const fc = item.asFunctionCall().?;
    try std.testing.expectEqualStrings("call_abc123", fc.call_id);
    try std.testing.expectEqualStrings("get_weather", fc.name);
    try std.testing.expectEqualStrings("{\"location\":\"Paris\"}", fc.arguments.items);
    try std.testing.expectEqual(@as(u32, 10), item.input_tokens);
    try std.testing.expectEqual(@as(u32, 15), item.output_tokens);
}

test "commitGenerationResult: tool calls with policy denial set status=failed" {
    const allocator = std.testing.allocator;

    const policy_mod = @import("../policy/evaluate.zig");

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    // Set up a deny-all policy
    var policy = policy_mod.Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &.{},
        ._pattern_buf = &.{},
        .allocator = allocator,
    };
    chat.policy = &policy;

    const tool_calls = [_]ToolCallInput{
        .{
            .id = "call_denied",
            .name = "dangerous_tool",
            .arguments = "{\"cmd\":\"rm -rf /\"}",
        },
    };

    try commitGenerationResult(allocator, &chat, .{
        .text = "",
        .tool_calls = &tool_calls,
        .prompt_tokens = 5,
        .completion_tokens = 8,
        .finish_reason = "tool_calls",
    });

    try std.testing.expectEqual(@as(usize, 1), chat.conv.len());
    const item = chat.conv.getItem(0).?;
    const fc = item.asFunctionCall().?;
    try std.testing.expectEqual(responses_mod.ItemStatus.failed, fc.status);
}

test "commitGenerationResult: multiple tool calls creates multiple items" {
    const allocator = std.testing.allocator;

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    const tool_calls = [_]ToolCallInput{
        .{
            .id = "call_1",
            .name = "search",
            .arguments = "{\"query\":\"zig\"}",
        },
        .{
            .id = "call_2",
            .name = "get_weather",
            .arguments = "{\"location\":\"Tokyo\"}",
        },
    };

    try commitGenerationResult(allocator, &chat, .{
        .text = "",
        .tool_calls = &tool_calls,
        .prompt_tokens = 10,
        .completion_tokens = 20,
        .finish_reason = "tool_calls",
    });

    try std.testing.expectEqual(@as(usize, 2), chat.conv.len());

    const fc0 = chat.conv.getItem(0).?.asFunctionCall().?;
    try std.testing.expectEqualStrings("call_1", fc0.call_id);
    try std.testing.expectEqualStrings("search", fc0.name);

    const fc1 = chat.conv.getItem(1).?.asFunctionCall().?;
    try std.testing.expectEqualStrings("call_2", fc1.call_id);
    try std.testing.expectEqualStrings("get_weather", fc1.name);
}

test "commitGenerationResult: custom reasoning tag" {
    const allocator = std.testing.allocator;

    var chat = try Chat.init(allocator);
    defer chat.deinit();

    try commitGenerationResult(allocator, &chat, .{
        .text = "<thought>Deep reasoning here.</thought>Final answer.",
        .prompt_tokens = 8,
        .completion_tokens = 12,
        .finish_reason = "stop",
        .reasoning_tag = "thought",
    });

    // Should have created ReasoningItem + AssistantMessage
    try std.testing.expectEqual(@as(usize, 2), chat.conv.len());

    const reasoning_item = chat.conv.getItem(0).?;
    try std.testing.expect(reasoning_item.data == .reasoning);

    const msg_item = chat.conv.getItem(1).?;
    const msg = msg_item.asMessage().?;
    try std.testing.expectEqualStrings("Final answer.", msg.getFirstText());
}

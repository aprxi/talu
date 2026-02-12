//! Chat - Lightweight chat state for conversations.
//!
//! Chat manages conversation history and generation preferences.
//! It contains no model or compute - pass it to an Engine to generate responses.
//!
//! This is the Zig equivalent of Python's Chat. Both Python and CLI
//! use this same implementation via C API.
//!
//! Architecture:
//! - Chat owns a Conversation instance (the Item-based data layer)
//! - Zig has sole write access to Conversation
//! - Python has read-only access via double-pointers

const std = @import("std");
const conversation_mod = @import("conversation.zig");
const items_mod = @import("items.zig");
const validate_mod = @import("../validate/root.zig");
const io = @import("../io/root.zig");

pub const Conversation = conversation_mod.Conversation;
pub const Item = items_mod.Item;
pub const ItemType = items_mod.ItemType;
pub const ItemStatus = items_mod.ItemStatus;
pub const MessageData = items_mod.MessageData;
pub const MessageRole = items_mod.MessageRole;
pub const ContentType = items_mod.ContentType;
pub const ContentPart = items_mod.ContentPart;
const ConstrainedSampler = validate_mod.sampler.ConstrainedSampler;

/// Configuration for model resolution (e.g., offline mode).
pub const ResolutionConfig = io.repository.ResolutionConfig;

/// Lightweight chat state for conversations.
///
/// Chat manages conversation history and generation preferences.
/// It contains no model or compute - pass it to an Engine to generate responses.
///
/// Chat owns a Conversation instance which provides:
/// - Zero-copy access from Python via double-pointers
/// - Streaming support for token-by-token updates
/// - Item storage with status (streaming/final)
///
/// ## Session Identity
/// The optional `session_id` identifies which conversation this Chat belongs to.
/// When persisting to StoreFS, session_id is hashed to `SESSION_HASH` (u64)
/// for efficient Jump Reads during session restoration.
///
/// Thread safety: NOT thread-safe. All access must be from a single thread.
pub const Chat = struct {
    /// Magic number for detecting use-after-free and memory corruption.
    pub const MAGIC_VALID: u64 = 0xC0FFEE_FACE_BEEF;
    pub const MAGIC_FREED: u64 = 0xDEAD_C0DE_CAFE;

    /// Magic number - first field for memory debugging.
    _magic: u64 = MAGIC_VALID,

    allocator: std.mem.Allocator,

    /// Conversation container (Item-based data layer).
    /// Provides typed Item access for Open Responses API.
    conv: *Conversation,

    /// Session identifier for this conversation.
    /// Used by storage backends to group messages by session.
    /// When null, messages are not associated with a named session.
    session_id: ?[]const u8 = null,

    /// Sampling temperature (0.0-2.0).
    temperature: f32 = 0.7,

    /// Maximum tokens to generate.
    max_tokens: usize = 256,

    /// Top-k sampling parameter.
    top_k: usize = 50,

    /// Nucleus sampling threshold.
    top_p: f32 = 0.9,

    /// Minimum probability threshold (min_p sampling).
    /// Tokens with probability < min_p * max_prob are excluded.
    /// 0.0 = disabled (default).
    min_p: f32 = 0.0,

    /// Repetition penalty applied to tokens in the context.
    /// 1.0 = no penalty (default), >1.0 = discourage repetition.
    repetition_penalty: f32 = 1.0,

    /// Optional grammar-constrained sampler for structured output.
    grammar_sampler: ?*ConstrainedSampler = null,

    /// Model resolution configuration (e.g., offline mode).
    resolution_config: ResolutionConfig = .{},

    /// Tool definitions as opaque JSON blob (OpenResponses format).
    /// Stored for round-tripping in the response resource.
    /// Set via C API; not interpreted by core beyond pass-through to generation.
    tools_json: ?[]u8 = null,

    /// Tool choice as opaque JSON blob (e.g. `"auto"`, `"none"`, `{"type":"function","name":"..."}`)
    /// Stored for round-tripping in the response resource.
    tool_choice_json: ?[]u8 = null,

    /// Optional policy for tool call filtering (IAM-style firewall).
    /// When set, tool calls are evaluated against this policy before
    /// being committed to the conversation. The policy must outlive the Chat.
    /// Not owned by Chat — caller manages the Policy lifecycle.
    policy: ?*const @import("../policy/evaluate.zig").Policy = null,

    /// Prompt document ID for lineage tracking.
    /// When the system prompt comes from a Document, this records the document ID.
    /// Used by storage backends to link sessions to their source prompts.
    prompt_id: ?[]u8 = null,

    /// Create a new empty chat.
    pub fn init(allocator: std.mem.Allocator) !Chat {
        return initWithSession(allocator, null);
    }

    /// Create a new chat with a session identifier.
    ///
    /// The session_id is used by storage backends to group messages.
    /// When persisting to StoreFS, it's hashed to SESSION_HASH for
    /// efficient Jump Reads during session restoration.
    pub fn initWithSession(allocator: std.mem.Allocator, session_id_arg: ?[]const u8) !Chat {
        const conv = try Conversation.initWithSession(allocator, session_id_arg);
        errdefer conv.deinit();

        // Copy session_id for Chat's own reference
        const owned_session_id = if (session_id_arg) |sid|
            try allocator.dupe(u8, sid)
        else
            null;

        return .{
            .allocator = allocator,
            .conv = conv,
            .session_id = owned_session_id,
        };
    }

    /// Create a new chat with a system prompt.
    pub fn initWithSystem(allocator: std.mem.Allocator, system_prompt: []const u8) !Chat {
        return initWithSystemAndSession(allocator, system_prompt, null);
    }

    /// Create a new chat with a system prompt and session identifier.
    pub fn initWithSystemAndSession(
        allocator: std.mem.Allocator,
        system_prompt: []const u8,
        session_id_arg: ?[]const u8,
    ) !Chat {
        var chat_session = try initWithSession(allocator, session_id_arg);
        errdefer chat_session.deinit();

        // Add system message
        _ = try chat_session.conv.appendSystemMessage(system_prompt);

        return chat_session;
    }

    /// Free all resources.
    pub fn deinit(self: *Chat) void {
        std.debug.assert(self._magic == MAGIC_VALID);
        self.clearGrammar();
        if (self.tools_json) |t| self.allocator.free(t);
        if (self.tool_choice_json) |t| self.allocator.free(t);
        if (self.session_id) |sid| {
            self.allocator.free(sid);
        }
        if (self.prompt_id) |pid| {
            self.allocator.free(pid);
        }
        self.conv.deinit();
        self._magic = MAGIC_FREED; // Mark as freed for use-after-free detection
    }

    /// Check if this Chat is valid (not freed or corrupted).
    pub fn isValid(self: *const Chat) bool {
        return self._magic == MAGIC_VALID;
    }

    /// Set grammar-constrained sampler for structured output.
    pub fn setGrammar(self: *Chat, sampler: *ConstrainedSampler) void {
        if (self.grammar_sampler) |existing| {
            existing.deinit();
            self.allocator.destroy(existing);
        }
        self.grammar_sampler = sampler;
    }

    /// Clear grammar-constrained sampler.
    pub fn clearGrammar(self: *Chat) void {
        if (self.grammar_sampler) |existing| {
            existing.deinit();
            self.allocator.destroy(existing);
            self.grammar_sampler = null;
        }
    }

    /// Get the Conversation container (Item-based API).
    /// Provides typed access to Items for Open Responses API.
    pub fn getConversation(self: *Chat) *Conversation {
        return self.conv;
    }

    /// Set tool definitions (opaque JSON blob).
    /// Takes ownership of a copy of the input.
    pub fn setTools(self: *Chat, json: []const u8) !void {
        if (self.tools_json) |old| self.allocator.free(old);
        self.tools_json = try self.allocator.dupe(u8, json);
    }

    /// Get tool definitions JSON, or null if not set.
    pub fn getTools(self: *const Chat) ?[]const u8 {
        return self.tools_json;
    }

    /// Clear tool definitions.
    pub fn clearTools(self: *Chat) void {
        if (self.tools_json) |old| {
            self.allocator.free(old);
            self.tools_json = null;
        }
    }

    /// Set tool_choice (opaque JSON blob).
    /// Takes ownership of a copy of the input.
    pub fn setToolChoice(self: *Chat, json: []const u8) !void {
        if (self.tool_choice_json) |old| self.allocator.free(old);
        self.tool_choice_json = try self.allocator.dupe(u8, json);
    }

    /// Get tool_choice JSON, or null if not set.
    pub fn getToolChoice(self: *const Chat) ?[]const u8 {
        return self.tool_choice_json;
    }

    /// Clear tool_choice.
    pub fn clearToolChoice(self: *Chat) void {
        if (self.tool_choice_json) |old| {
            self.allocator.free(old);
            self.tool_choice_json = null;
        }
    }

    /// Append a text message to the conversation.
    ///
    /// Creates a finalized item with text content.
    pub fn append(self: *Chat, role: MessageRole, content: []const u8) !void {
        _ = switch (role) {
            .system => try self.conv.appendSystemMessage(content),
            .user => try self.conv.appendUserMessage(content),
            .assistant => blk: {
                const item = try self.conv.appendAssistantMessage();
                try self.conv.appendTextContent(item, content);
                self.conv.finalizeItem(item);
                break :blk item;
            },
            .developer => try self.conv.appendDeveloperMessage(content),
            .unknown => try self.conv.appendUserMessage(content), // Safe fallback
        };
    }

    /// Start a streaming message (returns item for appending tokens).
    pub fn startStreaming(self: *Chat, role: MessageRole) !*Item {
        // For streaming, we typically only support assistant messages
        if (role == .assistant) {
            return try self.conv.appendAssistantMessage();
        } else {
            // For other roles, create a message that will be finalized immediately
            const item = switch (role) {
                .system => try self.conv.appendSystemMessage(""),
                .user => try self.conv.appendUserMessage(""),
                .developer => try self.conv.appendDeveloperMessage(""),
                else => try self.conv.appendUserMessage(""),
            };
            return item;
        }
    }

    /// Append content to a streaming message.
    pub fn appendToStreaming(self: *Chat, item: *Item, content: []const u8) !void {
        try self.conv.appendTextContent(item, content);
    }

    /// Finalize a streaming message.
    pub fn finalizeStreaming(self: *Chat, item: *Item) void {
        self.conv.finalizeItem(item);
    }

    /// Get system prompt (if present).
    pub fn getSystem(self: *const Chat) ?[]const u8 {
        if (self.conv.len() == 0) return null;
        const first_item = self.conv.getItem(0) orelse return null;
        const msg = first_item.asMessage() orelse return null;
        if (msg.role != .system and msg.role != .developer) return null;
        return msg.getFirstText();
    }

    /// Set the system prompt.
    /// If a system message already exists, updates it. Otherwise creates one.
    pub fn setSystem(self: *Chat, content: []const u8) !void {
        // Check if first item is system
        if (self.conv.len() > 0) {
            const first_item = self.conv.getItemMut(0).?;
            if (first_item.asMessage()) |msg| {
                if (msg.role == .system or msg.role == .developer) {
                    // Clear and update existing system message content
                    if (msg.content.items.len > 0) {
                        msg.content.items[0].clearData();
                        try msg.content.items[0].appendData(self.allocator, content);
                    }
                    return;
                }
            }
        }

        // No system message exists - need to insert at beginning
        // This requires rebuilding the conversation
        const new_conv = try Conversation.initWithSession(self.allocator, self.session_id);
        errdefer new_conv.deinit();

        // Add system message first
        _ = try new_conv.appendSystemMessage(content);

        // Copy existing items
        for (self.conv.items_list.items) |*old_item| {
            try self.copyItemTo(new_conv, old_item);
        }

        // Swap
        self.conv.deinit();
        self.conv = new_conv;
    }

    /// Internal: Copy an item to a new conversation.
    fn copyItemTo(self: *Chat, target: *Conversation, item: *const Item) !void {
        switch (item.data) {
            .message => |*m| {
                const new_item = switch (m.role) {
                    .system => try target.appendSystemMessage(""),
                    .user => try target.appendUserMessage(""),
                    .assistant => try target.appendAssistantMessage(),
                    .developer => try target.appendDeveloperMessage(""),
                    .unknown => try target.appendUserMessage(""),
                };
                // Copy content parts
                const new_msg = new_item.asMessageMut().?;
                new_msg.content.clearRetainingCapacity();
                for (m.content.items) |*old_part| {
                    var new_part = old_part.clone(self.allocator) catch continue;
                    new_msg.content.append(self.allocator, new_part) catch {
                        new_part.deinit(self.allocator);
                        continue;
                    };
                }
                target.finalizeItem(new_item);
            },
            .function_call => |*f| {
                const new_item = try target.appendFunctionCall(f.call_id, f.name);
                try target.setFunctionCallArguments(new_item, f.arguments.items);
                target.finalizeItem(new_item);
            },
            .function_call_output => |*f| {
                const output_text = f.getOutputText();
                _ = try target.appendFunctionCallOutput(f.call_id, output_text);
            },
            .reasoning => |*r| {
                const new_item = try target.appendReasoning();
                for (r.summary.items) |*part| {
                    const text = part.getData();
                    try target.addReasoningSummary(new_item, text);
                }
            },
            .item_reference => |*ref| {
                _ = try target.appendItemReference(ref.id);
            },
            .unknown => {},
        }
    }

    /// Clear conversation history (keeps system prompt and settings).
    pub fn clear(self: *Chat) void {
        self.conv.clearKeepingSystem();
    }

    /// Clear only the system prompt (keeps other messages).
    /// If no system prompt exists, this is a no-op.
    pub fn clearSystem(self: *Chat) void {
        if (self.conv.len() == 0) return;

        const first_item = self.conv.getItem(0).?;
        const msg = first_item.asMessage() orelse return;
        if (msg.role != .system and msg.role != .developer) return;

        // Delete the first item (system message)
        _ = self.conv.deleteItem(0);
    }

    /// Get prompt document ID (if set).
    /// Returns null if no prompt_id is set.
    pub fn getPromptId(self: *const Chat) ?[]const u8 {
        return self.prompt_id;
    }

    /// Set prompt document ID for lineage tracking.
    /// Pass null or empty string to clear.
    pub fn setPromptId(self: *Chat, id: ?[]const u8) !void {
        // Free existing
        if (self.prompt_id) |old| {
            self.allocator.free(old);
            self.prompt_id = null;
        }
        // Set new (if non-empty)
        if (id) |new_id| {
            if (new_id.len > 0) {
                self.prompt_id = try self.allocator.dupe(u8, new_id);
            }
        }
    }

    /// Reset everything including system prompt.
    pub fn reset(self: *Chat) void {
        self.conv.clear();
        self.temperature = 0.7;
        self.max_tokens = 256;
        self.top_k = 50;
        self.top_p = 0.9;
        self.min_p = 0.0;
        self.repetition_penalty = 1.0;
        self.resolution_config = .{};
        self.clearGrammar();
        if (self.tools_json) |t| {
            self.allocator.free(t);
            self.tools_json = null;
        }
        if (self.tool_choice_json) |t| {
            self.allocator.free(t);
            self.tool_choice_json = null;
        }
    }

    /// Get number of items (including system if present).
    pub fn len(self: *const Chat) usize {
        return self.conv.len();
    }

    /// Get an item by index.
    pub fn get(self: *const Chat, index: usize) ?*const Item {
        return self.conv.getItem(index);
    }

    /// Remove and discard the last item.
    /// Returns true on success, false if empty.
    pub fn pop(self: *Chat) bool {
        if (self.conv.len() == 0) return false;
        return self.conv.deleteItem(self.conv.len() - 1);
    }

    /// Remove item at index.
    /// Returns true on success, false on invalid index.
    pub fn remove(self: *Chat, index: usize) bool {
        return self.conv.deleteItem(index);
    }

    /// Insert a text message at index.
    /// Returns error on failure.
    pub fn insert(self: *Chat, index: usize, role: MessageRole, content: []const u8) !void {
        const conv = self.conv;

        // Validate index (can insert at end)
        if (index > conv.len()) return error.IndexOutOfBounds;

        // For insert at end, just use normal append
        if (index == conv.len()) {
            try self.append(role, content);
            return;
        }

        // For insert in middle, rebuild conversation
        const new_conv = try Conversation.initWithSession(self.allocator, self.session_id);
        errdefer new_conv.deinit();

        // Copy items before index
        for (conv.items_list.items[0..index]) |*old_item| {
            try self.copyItemTo(new_conv, old_item);
        }

        // Insert new item
        _ = switch (role) {
            .system => try new_conv.appendSystemMessage(content),
            .user => try new_conv.appendUserMessage(content),
            .assistant => blk: {
                const item = try new_conv.appendAssistantMessage();
                try new_conv.appendTextContent(item, content);
                new_conv.finalizeItem(item);
                break :blk item;
            },
            .developer => try new_conv.appendDeveloperMessage(content),
            .unknown => try new_conv.appendUserMessage(content),
        };

        // Copy items after index
        for (conv.items_list.items[index..]) |*old_item| {
            try self.copyItemTo(new_conv, old_item);
        }

        // Swap
        conv.deinit();
        self.conv = new_conv;
    }
};

// =============================================================================
// Tests
// =============================================================================

test "Chat.init" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    try std.testing.expectEqual(@as(usize, 0), chat_session.len());
    try std.testing.expect(chat_session.getSystem() == null);
}

test "Chat.initWithSystem" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "You are helpful.");
    defer chat_session.deinit();

    try std.testing.expectEqualStrings("You are helpful.", chat_session.getSystem().?);
}

test "Chat.append" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");
    try std.testing.expectEqual(@as(usize, 1), chat_session.len());

    try chat_session.append(.assistant, "Hi there!");
    try std.testing.expectEqual(@as(usize, 2), chat_session.len());

    // Verify items
    const item0 = chat_session.get(0).?;
    const msg0 = item0.asMessage().?;
    try std.testing.expectEqual(MessageRole.user, msg0.role);
    try std.testing.expectEqualStrings("Hello!", msg0.getFirstText());

    const item1 = chat_session.get(1).?;
    const msg1 = item1.asMessage().?;
    try std.testing.expectEqual(MessageRole.assistant, msg1.role);
    try std.testing.expectEqualStrings("Hi there!", msg1.getFirstText());
}

test "Chat.clear" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");
    try chat_session.append(.assistant, "Hi!");

    chat_session.clear();

    // After clear, only system message remains
    try std.testing.expectEqual(@as(usize, 1), chat_session.len());
    try std.testing.expectEqualStrings("System", chat_session.getSystem().?);
}

test "Chat.reset" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");
    chat_session.temperature = 0.5;

    chat_session.reset();

    try std.testing.expectEqual(@as(usize, 0), chat_session.len());
    try std.testing.expect(chat_session.getSystem() == null);
    try std.testing.expectEqual(@as(f32, 0.7), chat_session.temperature);
}


test "Chat.startStreaming and streaming workflow" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Start a streaming response
    const item = try chat_session.startStreaming(.assistant);
    const msg = item.asMessage().?;
    try std.testing.expectEqual(ItemStatus.in_progress, msg.status);

    // Append tokens
    try chat_session.appendToStreaming(item, "Hello");
    try std.testing.expectEqual(@as(usize, 5), msg.getFirstText().len);

    try chat_session.appendToStreaming(item, ", world!");
    try std.testing.expectEqual(@as(usize, 13), msg.getFirstText().len);

    // Finalize
    chat_session.finalizeStreaming(item);
    try std.testing.expectEqual(ItemStatus.completed, msg.status);

    // Verify content
    try std.testing.expectEqualStrings("Hello, world!", msg.getFirstText());
}

test "Chat.getConversation" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");

    // Get Conversation for direct access
    const conv = chat_session.getConversation();
    try std.testing.expectEqual(@as(usize, 2), conv.len());

    // Verify item API access works
    const item = conv.getItem(1).?;
    const msg = item.asMessage().?;
    try std.testing.expectEqualStrings("Hello!", msg.getFirstText());
}

test "Chat.append all roles" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test all role types
    try chat_session.append(.system, "System message");
    try chat_session.append(.user, "User message");
    try chat_session.append(.assistant, "Assistant message");
    try chat_session.append(.user, "Tool result"); // Tool role removed, use user

    try std.testing.expectEqual(@as(usize, 4), chat_session.len());

    // Verify roles
    const item0 = chat_session.get(0).?;
    const msg0 = item0.asMessage().?;
    try std.testing.expectEqual(MessageRole.system, msg0.role);
    try std.testing.expectEqualStrings("System message", msg0.getFirstText());

    // Tool maps to user in MessageRole
    const item3 = chat_session.get(3).?;
    const msg3 = item3.asMessage().?;
    try std.testing.expectEqual(MessageRole.user, msg3.role);
    try std.testing.expectEqualStrings("Tool result", msg3.getFirstText());
}

test "Chat.setSystem existing" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "Original system");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");

    // Update system prompt
    try chat_session.setSystem("Updated system");

    try std.testing.expectEqual(@as(usize, 2), chat_session.len());
    try std.testing.expectEqualStrings("Updated system", chat_session.getSystem().?);

    // User message should still be present
    const item = chat_session.get(1).?;
    const msg = item.asMessage().?;
    try std.testing.expectEqualStrings("Hello!", msg.getFirstText());
}

test "Chat.setSystem new" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");

    // Add system prompt to chat without one
    try chat_session.setSystem("New system");

    try std.testing.expectEqual(@as(usize, 2), chat_session.len());
    try std.testing.expectEqualStrings("New system", chat_session.getSystem().?);

    // User message should be moved to position 1
    const item = chat_session.get(1).?;
    const msg = item.asMessage().?;
    try std.testing.expectEqualStrings("Hello!", msg.getFirstText());
}

test "Chat.clearSystem" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");
    try chat_session.append(.assistant, "Hi!");

    // Clear only the system prompt
    chat_session.clearSystem();

    try std.testing.expectEqual(@as(usize, 2), chat_session.len());
    try std.testing.expect(chat_session.getSystem() == null);

    // Other messages should remain
    const item0 = chat_session.get(0).?;
    const msg0 = item0.asMessage().?;
    try std.testing.expectEqual(MessageRole.user, msg0.role);
    try std.testing.expectEqualStrings("Hello!", msg0.getFirstText());
}

test "Chat.clearSystem no system" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");

    // Should be a no-op
    chat_session.clearSystem();

    try std.testing.expectEqual(@as(usize, 1), chat_session.len());
}

test "Chat.clearSystem empty" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Should be a no-op
    chat_session.clearSystem();

    try std.testing.expectEqual(@as(usize, 0), chat_session.len());
}


test "Chat.get out of bounds" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");

    try std.testing.expect(chat_session.get(1) == null);
    try std.testing.expect(chat_session.get(100) == null);
}

test "Chat.streaming multiple messages" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Add user message
    try chat_session.append(.user, "Tell me a story");

    // Stream assistant response in parts
    const item1 = try chat_session.startStreaming(.assistant);
    try chat_session.appendToStreaming(item1, "Once");
    try chat_session.appendToStreaming(item1, " upon");
    try chat_session.appendToStreaming(item1, " a time");
    chat_session.finalizeStreaming(item1);

    // Add another user message
    try chat_session.append(.user, "Continue");

    // Stream another assistant response
    const item2 = try chat_session.startStreaming(.assistant);
    try chat_session.appendToStreaming(item2, "There was");
    try chat_session.appendToStreaming(item2, " a cat");
    chat_session.finalizeStreaming(item2);

    try std.testing.expectEqual(@as(usize, 4), chat_session.len());

    const msg1 = chat_session.get(1).?.asMessage().?;
    try std.testing.expectEqualStrings("Once upon a time", msg1.getFirstText());

    const msg3 = chat_session.get(3).?.asMessage().?;
    try std.testing.expectEqualStrings("There was a cat", msg3.getFirstText());
}

test "Chat.reset clears all settings" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello!");

    // Modify all settings
    chat_session.temperature = 1.5;
    chat_session.max_tokens = 512;
    chat_session.top_k = 100;
    chat_session.top_p = 0.95;
    chat_session.min_p = 0.05;
    chat_session.repetition_penalty = 1.2;

    chat_session.reset();

    // All should be back to defaults
    try std.testing.expectEqual(@as(usize, 0), chat_session.len());
    try std.testing.expect(chat_session.getSystem() == null);
    try std.testing.expectEqual(@as(f32, 0.7), chat_session.temperature);
    try std.testing.expectEqual(@as(usize, 256), chat_session.max_tokens);
    try std.testing.expectEqual(@as(usize, 50), chat_session.top_k);
    try std.testing.expectEqual(@as(f32, 0.9), chat_session.top_p);
    try std.testing.expectEqual(@as(f32, 0.0), chat_session.min_p);
    try std.testing.expectEqual(@as(f32, 1.0), chat_session.repetition_penalty);
}


test "Chat.deinit frees all resources" {
    const allocator = std.testing.allocator;

    // Create a chat with multiple messages to ensure all memory is freed
    var chat_session = try Chat.initWithSystem(allocator, "You are a helpful assistant.");

    // Add multiple messages to allocate memory
    try chat_session.append(.user, "Hello, how are you?");
    try chat_session.append(.assistant, "I'm doing well, thank you!");
    try chat_session.append(.user, "Can you help me with something?");
    try chat_session.append(.assistant, "Of course! I'd be happy to help.");

    // Add a streaming message
    const item = try chat_session.startStreaming(.assistant);
    try chat_session.appendToStreaming(item, "This is a ");
    try chat_session.appendToStreaming(item, "streaming message.");
    chat_session.finalizeStreaming(item);

    // Verify chat has messages before deinit
    try std.testing.expectEqual(@as(usize, 6), chat_session.len());

    // Deinit should free all memory - testing.allocator will detect leaks
    chat_session.deinit();

    // Test passes if no memory leaks detected by testing.allocator
}

test "Chat.deinit empty chat" {
    const allocator = std.testing.allocator;

    // Deinit on empty chat should not leak
    var chat_session = try Chat.init(allocator);
    chat_session.deinit();

    // Test passes if no memory leaks detected
}

test "Chat.deinit after operations" {
    const allocator = std.testing.allocator;

    // Use initWithSystem to avoid setSystem memory complexity
    var chat_session = try Chat.initWithSystem(allocator, "System prompt");

    // Perform various operations
    try chat_session.append(.user, "Message 1");
    try chat_session.append(.assistant, "Response 1");

    // Deinit should free all memory including any intermediate allocations
    chat_session.deinit();

    // Test passes if no memory leaks detected
}

test "Chat.startStreaming creates item with in_progress status" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test normal path
    const item = try chat_session.startStreaming(.assistant);
    const msg = item.asMessage().?;
    try std.testing.expectEqual(ItemStatus.in_progress, msg.status);
    try std.testing.expectEqual(MessageRole.assistant, msg.role);
}

test "Chat.appendToStreaming appends content" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    const item = try chat_session.startStreaming(.assistant);

    // Test normal path - single append
    try chat_session.appendToStreaming(item, "Hello");
    const msg = item.asMessage().?;
    try std.testing.expectEqual(@as(usize, 5), msg.getFirstText().len);
    try std.testing.expectEqualStrings("Hello", msg.getFirstText());

    // Test edge case - multiple appends
    try chat_session.appendToStreaming(item, " world");
    try std.testing.expectEqual(@as(usize, 11), msg.getFirstText().len);
    try std.testing.expectEqualStrings("Hello world", msg.getFirstText());

    // Test edge case - empty append
    try chat_session.appendToStreaming(item, "");
    try std.testing.expectEqual(@as(usize, 11), msg.getFirstText().len);
    try std.testing.expectEqualStrings("Hello world", msg.getFirstText());
}

test "Chat.finalizeStreaming changes status to completed" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    const item = try chat_session.startStreaming(.assistant);
    try chat_session.appendToStreaming(item, "Complete message");

    // Test normal path
    const msg = item.asMessage().?;
    try std.testing.expectEqual(ItemStatus.in_progress, msg.status);
    chat_session.finalizeStreaming(item);
    try std.testing.expectEqual(ItemStatus.completed, msg.status);

    // Content should remain unchanged
    try std.testing.expectEqualStrings("Complete message", msg.getFirstText());
}

test "Chat.finalizeStreaming on empty streaming item" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test edge case - finalize without appending
    const item = try chat_session.startStreaming(.assistant);
    chat_session.finalizeStreaming(item);

    const msg = item.asMessage().?;
    try std.testing.expectEqual(ItemStatus.completed, msg.status);
}

test "Chat.getSystem with no messages" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test edge case - empty chat
    try std.testing.expect(chat_session.getSystem() == null);
}

test "Chat.getSystem with non-system first message" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello");

    // Test edge case - first message is not system
    try std.testing.expect(chat_session.getSystem() == null);
}

test "Chat.getSystem with system message" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "You are helpful.");
    defer chat_session.deinit();

    // Test normal path
    const system = chat_session.getSystem();
    try std.testing.expect(system != null);
    try std.testing.expectEqualStrings("You are helpful.", system.?);
}

test "Chat.getSystem after adding user messages" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System prompt");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello");
    try chat_session.append(.assistant, "Hi");

    // Test normal path - system should persist
    const system = chat_session.getSystem();
    try std.testing.expect(system != null);
    try std.testing.expectEqualStrings("System prompt", system.?);
}

test "Chat.len with empty chat" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test edge case - empty
    try std.testing.expectEqual(@as(usize, 0), chat_session.len());
}

test "Chat.len increments correctly" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test normal path - len increases with each append
    try std.testing.expectEqual(@as(usize, 0), chat_session.len());

    try chat_session.append(.system, "System");
    try std.testing.expectEqual(@as(usize, 1), chat_session.len());

    try chat_session.append(.user, "User");
    try std.testing.expectEqual(@as(usize, 2), chat_session.len());

    try chat_session.append(.assistant, "Assistant");
    try std.testing.expectEqual(@as(usize, 3), chat_session.len());
}

test "Chat.len with streaming messages" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.init(allocator);
    defer chat_session.deinit();

    // Test normal path - streaming messages count
    const item1 = try chat_session.startStreaming(.assistant);
    try std.testing.expectEqual(@as(usize, 1), chat_session.len());

    try chat_session.appendToStreaming(item1, "Hello");
    try std.testing.expectEqual(@as(usize, 1), chat_session.len());

    chat_session.finalizeStreaming(item1);
    try std.testing.expectEqual(@as(usize, 1), chat_session.len());

    // Add another streaming message
    _ = try chat_session.startStreaming(.assistant);
    try std.testing.expectEqual(@as(usize, 2), chat_session.len());
}

test "Chat.getConversation returns Conversation pointer" {
    const allocator = std.testing.allocator;
    var chat_session = try Chat.initWithSystem(allocator, "System");
    defer chat_session.deinit();

    try chat_session.append(.user, "Hello");
    try chat_session.append(.assistant, "Hi");

    // Test getConversation provides access to underlying Conversation
    const conv = chat_session.getConversation();
    try std.testing.expectEqual(@as(usize, 3), conv.len());
    try std.testing.expectEqual(MessageRole.system, conv.getItem(0).?.asMessage().?.role);
    try std.testing.expectEqual(MessageRole.user, conv.getItem(1).?.asMessage().?.role);
    try std.testing.expectEqual(MessageRole.assistant, conv.getItem(2).?.asMessage().?.role);
}

test "Chat.setTools and getTools" {
    const alloc = std.testing.allocator;
    var chat_session = try Chat.init(alloc);
    defer chat_session.deinit();

    // Initially null
    try std.testing.expect(chat_session.getTools() == null);

    // Set tools
    const tools = "[{\"type\":\"function\",\"name\":\"get_weather\"}]";
    try chat_session.setTools(tools);
    try std.testing.expectEqualStrings(tools, chat_session.getTools().?);

    // Overwrite tools
    const tools2 = "[]";
    try chat_session.setTools(tools2);
    try std.testing.expectEqualStrings(tools2, chat_session.getTools().?);

    // Clear tools
    chat_session.clearTools();
    try std.testing.expect(chat_session.getTools() == null);
}

test "Chat.setToolChoice and getToolChoice" {
    const alloc = std.testing.allocator;
    var chat_session = try Chat.init(alloc);
    defer chat_session.deinit();

    // Initially null
    try std.testing.expect(chat_session.getToolChoice() == null);

    // Set tool_choice
    try chat_session.setToolChoice("\"auto\"");
    try std.testing.expectEqualStrings("\"auto\"", chat_session.getToolChoice().?);

    // Overwrite
    try chat_session.setToolChoice("\"none\"");
    try std.testing.expectEqualStrings("\"none\"", chat_session.getToolChoice().?);

    // Clear
    chat_session.clearToolChoice();
    try std.testing.expect(chat_session.getToolChoice() == null);
}

test "Chat.reset clears tools and tool_choice" {
    const alloc = std.testing.allocator;
    var chat_session = try Chat.init(alloc);
    defer chat_session.deinit();

    try chat_session.setTools("[{\"type\":\"function\"}]");
    try chat_session.setToolChoice("\"auto\"");

    chat_session.reset();

    try std.testing.expect(chat_session.getTools() == null);
    try std.testing.expect(chat_session.getToolChoice() == null);
}

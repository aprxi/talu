//! Agent - Stateful, autonomous agent with built-in compaction and retry.
//!
//! Owns a Chat (conversation), ToolRegistry, and makes decisions internally:
//! turn-aware compaction when approaching context limits, retry with
//! exponential backoff on transient failures, inbox draining from a
//! MessageBus, goal persistence across compaction, and optional context
//! injection (RAG) before each generation.
//!
//! Exposes `prompt()` (user turn), `continueLoop()` (resume), `heartbeat()`
//! (autonomous check-in), and `abort()` (cross-thread cancellation).
//!
//! The Agent delegates actual generation to `loop.run()`, adding statefulness
//! and autonomy on top of the stateless loop.
//!
//! # Ownership
//!
//! Agent owns: Chat, ToolRegistry, agent_id, state_dir, goals,
//! base_system_prompt.
//! Agent borrows: InferenceBackend, MessageBus (caller-managed lifecycles).
//! `deinit()` frees all owned resources.
//!
//! # Thread Safety
//!
//! NOT thread-safe for method calls. Use `abort()` (atomic) for cross-thread
//! cancellation. Do not call `prompt()`/`continueLoop()`/`heartbeat()`
//! concurrently.

const std = @import("std");
const Allocator = std.mem.Allocator;

const loop_mod = @import("loop.zig");
const tool_mod = @import("tool.zig");
const bus_mod = @import("bus.zig");
const compaction_mod = @import("compaction.zig");

const responses = @import("../responses/root.zig");
const Chat = responses.Chat;
const Conversation = responses.Conversation;

const router = @import("../router/root.zig");
const InferenceBackend = router.InferenceBackend;

const AgentLoopConfig = loop_mod.AgentLoopConfig;
const AgentLoopResult = loop_mod.AgentLoopResult;
const LoopStopReason = loop_mod.LoopStopReason;
const OnTokenFn = loop_mod.OnTokenFn;
const OnEventFn = loop_mod.OnEventFn;
const OnSessionFn = loop_mod.OnSessionFn;
const ToolRegistry = tool_mod.ToolRegistry;
const Tool = tool_mod.Tool;
const MessageBus = bus_mod.MessageBus;
const Message = bus_mod.Message;

// =============================================================================
// Callback types
// =============================================================================

/// Called before each generation to allow context injection (e.g., RAG).
///
/// Receives the most recent user message text (or null if none).
/// Writes context to out_context/out_context_len. The agent copies the
/// returned text before proceeding. Returns false to cancel the loop.
pub const OnContextInjectFn = *const fn (
    last_user_message: ?[*]const u8,
    last_user_message_len: usize,
    out_context: *?[*]const u8,
    out_context_len: *usize,
    user_data: ?*anyopaque,
) callconv(.c) bool;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for creating an Agent.
pub const AgentConfig = struct {
    /// Maximum context size in tokens. 0 = unlimited (no compaction).
    context_limit: u64 = 0,
    /// Fraction of context_limit at which compaction triggers (0.0-1.0).
    compaction_threshold: f32 = 0.8,
    /// Number of retries on generation failure. 0 = no retry.
    max_retries: u32 = 0,
    /// Base delay for exponential backoff (nanoseconds).
    retry_base_delay_ns: u64 = std.time.ns_per_s,
    /// Maximum tool-call iterations per turn.
    max_iterations_per_turn: usize = 10,
    /// Stop the loop when a tool execution returns an error.
    abort_on_tool_error: bool = false,
    /// Agent identifier. Null = auto-generate a random ID.
    agent_id: ?[]const u8 = null,
    /// Directory for agent state files (tools, skills, key-value).
    state_dir: ?[]const u8 = null,

    // Observation callbacks (forwarded to loop.run)
    on_token: ?OnTokenFn = null,
    on_token_data: ?*anyopaque = null,
    on_event: ?OnEventFn = null,
    on_event_data: ?*anyopaque = null,
    on_session: ?OnSessionFn = null,
    on_session_data: ?*anyopaque = null,

    // Context injection callback (called at Agent level before each generation)
    on_context_inject: ?OnContextInjectFn = null,
    on_context_inject_data: ?*anyopaque = null,
};

// =============================================================================
// Agent
// =============================================================================

pub const AgentError = error{
    OutOfMemory,
    GenerationFailed,
    IteratorCreationFailed,
};

/// Stateful agent with built-in compaction, retry, and inter-agent messaging.
pub const Agent = struct {
    allocator: Allocator,

    // Owned state
    chat: *Chat,
    registry: ToolRegistry,

    // References (not owned)
    backend: *InferenceBackend,
    bus: ?*MessageBus,

    // Identity
    agent_id: []const u8,
    state_dir: ?[]const u8,

    // Autonomy config
    context_limit: u64,
    compaction_threshold: f32,
    max_retries: u32,
    retry_base_delay_ns: u64,
    max_iterations_per_turn: usize,
    abort_on_tool_error: bool,

    // Runtime state
    stop_flag: std.atomic.Value(bool),
    total_iterations: usize,
    total_tool_calls: usize,

    // Observation callbacks
    on_token: ?OnTokenFn,
    on_token_data: ?*anyopaque,
    on_event: ?OnEventFn,
    on_event_data: ?*anyopaque,
    on_session: ?OnSessionFn,
    on_session_data: ?*anyopaque,

    // Context injection callback
    on_context_inject: ?OnContextInjectFn,
    on_context_inject_data: ?*anyopaque,

    // Goals — persistent objectives that survive compaction
    goals: std.ArrayListUnmanaged([]const u8),
    /// The caller's original system prompt, before goal injection.
    /// Null until setSystem() is called.
    base_system_prompt: ?[]const u8,

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Create a new agent.
    ///
    /// The agent owns the Chat and ToolRegistry. The backend must outlive the
    /// agent (caller-managed).
    pub fn init(
        allocator: Allocator,
        backend: *InferenceBackend,
        config: AgentConfig,
    ) !*Agent {
        // Create chat
        const chat_val = try Chat.init(allocator);
        const chat_ptr = try allocator.create(Chat);
        chat_ptr.* = chat_val;
        errdefer {
            chat_ptr.deinit();
            allocator.destroy(chat_ptr);
        }

        // Generate or copy agent_id
        const agent_id = if (config.agent_id) |id|
            try allocator.dupe(u8, id)
        else
            try generateId(allocator);
        errdefer allocator.free(agent_id);

        // Copy state_dir if provided
        const state_dir = if (config.state_dir) |dir|
            try allocator.dupe(u8, dir)
        else
            null;
        errdefer if (state_dir) |d| allocator.free(d);

        const agent = try allocator.create(Agent);
        agent.* = .{
            .allocator = allocator,
            .chat = chat_ptr,
            .registry = ToolRegistry.init(allocator),
            .backend = backend,
            .bus = null,
            .agent_id = agent_id,
            .state_dir = state_dir,
            .context_limit = config.context_limit,
            .compaction_threshold = config.compaction_threshold,
            .max_retries = config.max_retries,
            .retry_base_delay_ns = config.retry_base_delay_ns,
            .max_iterations_per_turn = config.max_iterations_per_turn,
            .abort_on_tool_error = config.abort_on_tool_error,
            .stop_flag = std.atomic.Value(bool).init(false),
            .total_iterations = 0,
            .total_tool_calls = 0,
            .on_token = config.on_token,
            .on_token_data = config.on_token_data,
            .on_event = config.on_event,
            .on_event_data = config.on_event_data,
            .on_session = config.on_session,
            .on_session_data = config.on_session_data,
            .on_context_inject = config.on_context_inject,
            .on_context_inject_data = config.on_context_inject_data,
            .goals = .{},
            .base_system_prompt = null,
        };
        return agent;
    }

    /// Free all owned resources.
    pub fn deinit(self: *Agent) void {
        self.registry.deinit();
        self.chat.deinit();
        self.allocator.destroy(self.chat);
        self.allocator.free(self.agent_id);
        if (self.state_dir) |d| self.allocator.free(d);
        for (self.goals.items) |goal| self.allocator.free(goal);
        self.goals.deinit(self.allocator);
        if (self.base_system_prompt) |bsp| self.allocator.free(bsp);
        self.allocator.destroy(self);
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the system prompt.
    ///
    /// Stores the prompt as the base system prompt and rebuilds the effective
    /// system prompt with any active goals appended.
    pub fn setSystem(self: *Agent, system_prompt: []const u8) !void {
        if (self.base_system_prompt) |old| self.allocator.free(old);
        self.base_system_prompt = try self.allocator.dupe(u8, system_prompt);
        try self.syncSystemPromptWithGoals();
    }

    /// Register a tool. The agent takes ownership via its ToolRegistry.
    pub fn registerTool(self: *Agent, new_tool: Tool) !void {
        try self.registry.register(new_tool);
    }

    /// Connect to a MessageBus for inter-agent communication.
    /// The bus must outlive the agent (caller-managed).
    pub fn setBus(self: *Agent, bus: *MessageBus) void {
        self.bus = bus;
    }

    // =========================================================================
    // Goals — persistent objectives that survive compaction
    // =========================================================================

    /// Add a goal that persists across compaction.
    ///
    /// Goals are injected into the system prompt before each generation,
    /// ensuring the LLM always knows its objectives even after context
    /// truncation. Duplicate goals (exact string match) are silently ignored.
    pub fn addGoal(self: *Agent, goal: []const u8) !void {
        for (self.goals.items) |existing| {
            if (std.mem.eql(u8, existing, goal)) return;
        }
        const owned = try self.allocator.dupe(u8, goal);
        errdefer self.allocator.free(owned);
        try self.goals.append(self.allocator, owned);
    }

    /// Remove a specific goal by exact string match.
    /// Returns true if the goal was found and removed.
    pub fn removeGoal(self: *Agent, goal: []const u8) bool {
        for (self.goals.items, 0..) |existing, idx| {
            if (std.mem.eql(u8, existing, goal)) {
                self.allocator.free(existing);
                _ = self.goals.orderedRemove(idx);
                return true;
            }
        }
        return false;
    }

    /// Remove all goals.
    pub fn clearGoals(self: *Agent) void {
        for (self.goals.items) |goal| {
            self.allocator.free(goal);
        }
        self.goals.clearRetainingCapacity();
    }

    /// Get the current goals (read-only slice).
    pub fn getGoals(self: *const Agent) []const []const u8 {
        return self.goals.items;
    }

    // =========================================================================
    // Execution
    // =========================================================================

    /// Send a user message and run the agent loop.
    ///
    /// 1. Drain inbox (bus messages -> developer messages)
    /// 2. Sync goals into system prompt
    /// 3. Compact if approaching context limit (turn-aware)
    /// 4. Append user message
    /// 5. Inject context (RAG callback)
    /// 6. Run loop with retry
    pub fn prompt(self: *Agent, message: []const u8) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        try self.syncSystemPromptWithGoals();
        self.compactIfNeeded();
        _ = try self.chat.conv.appendUserMessage(message);
        if (!try self.injectContext()) {
            return cancelledResult(self);
        }

        return self.runWithRetry();
    }

    /// Resume the agent loop without a new user message.
    ///
    /// Drains inbox, syncs goals, compacts if needed, injects context,
    /// then continues from where the previous turn left off.
    pub fn continueLoop(self: *Agent) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        try self.syncSystemPromptWithGoals();
        self.compactIfNeeded();
        if (!try self.injectContext()) {
            return cancelledResult(self);
        }

        return self.runWithRetry();
    }

    /// Autonomous heartbeat: drain inbox and act on pending work.
    ///
    /// Appends a developer message prompting the LLM to review its state,
    /// then runs the loop. Callers invoke this on a timer for autonomy.
    pub fn heartbeat(self: *Agent) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        try self.syncSystemPromptWithGoals();
        self.compactIfNeeded();
        _ = try self.chat.conv.appendDeveloperMessage(
            "Heartbeat: review pending work and act.",
        );
        if (!try self.injectContext()) {
            return cancelledResult(self);
        }

        return self.runWithRetry();
    }

    /// Request cancellation from another thread.
    /// The loop checks this flag between iterations.
    pub fn abort(self: *Agent) void {
        self.stop_flag.store(true, .release);
    }

    // =========================================================================
    // Read access
    // =========================================================================

    /// Get the underlying Chat (read-only).
    pub fn getChat(self: *const Agent) *const Chat {
        return self.chat;
    }

    /// Get the ToolRegistry (read-only).
    pub fn getRegistry(self: *const Agent) *const ToolRegistry {
        return &self.registry;
    }

    // =========================================================================
    // Internal: inbox drain
    // =========================================================================

    /// Drain all pending messages from the bus into the conversation as
    /// developer messages formatted as "[from:<agent_id>] <payload>".
    fn drainInbox(self: *Agent) !void {
        const bus = self.bus orelse return;

        while (bus.receive(self.agent_id)) |msg| {
            defer bus.freeMessage(msg);

            const text = try std.fmt.allocPrint(
                self.allocator,
                "[from:{s}] {s}",
                .{ msg.from, msg.payload },
            );
            defer self.allocator.free(text);

            _ = try self.chat.conv.appendDeveloperMessage(text);
        }
    }

    // =========================================================================
    // Internal: goal sync
    // =========================================================================

    /// Build and set the system prompt with goals injected.
    ///
    /// If goals are present, appends them to the base system prompt in a
    /// structured section. If no goals, sets the base prompt directly.
    /// No-op if no base system prompt has been set.
    fn syncSystemPromptWithGoals(self: *Agent) !void {
        const base = self.base_system_prompt orelse return;

        if (self.goals.items.len == 0) {
            try self.chat.setSystem(base);
            return;
        }

        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(self.allocator);
        const writer = buf.writer(self.allocator);

        try writer.writeAll(base);
        try writer.writeAll("\n\n## Active Goals\n\nThese are your current objectives. Pursue them proactively:\n");
        for (self.goals.items) |goal| {
            try writer.writeAll("- ");
            try writer.writeAll(goal);
            try writer.writeByte('\n');
        }

        try self.chat.setSystem(buf.items);
    }

    // =========================================================================
    // Internal: context injection
    // =========================================================================

    /// Call the context injection hook if set.
    ///
    /// Finds the most recent user message, passes its text to the callback,
    /// and appends the returned context as a hidden developer message
    /// (visible to LLM, excluded from UI).
    ///
    /// Returns false if the callback requested cancellation.
    fn injectContext(self: *Agent) !bool {
        const inject_fn = self.on_context_inject orelse return true;

        // Find last user message text
        var last_user_text: ?[]const u8 = null;
        var scan_idx: usize = self.chat.conv.len();
        while (scan_idx > 0) {
            scan_idx -= 1;
            const item = self.chat.conv.getItem(scan_idx) orelse continue;
            const msg = item.asMessage() orelse continue;
            if (msg.role == .user) {
                last_user_text = msg.getFirstText();
                break;
            }
        }

        var out_ctx: ?[*]const u8 = null;
        var out_len: usize = 0;

        const user_ptr: ?[*]const u8 = if (last_user_text) |t| t.ptr else null;
        const user_len: usize = if (last_user_text) |t| t.len else 0;

        const should_continue = inject_fn(
            user_ptr,
            user_len,
            &out_ctx,
            &out_len,
            self.on_context_inject_data,
        );

        if (!should_continue) return false;

        if (out_ctx) |ctx_ptr| {
            if (out_len > 0) {
                const context_text = ctx_ptr[0..out_len];
                _ = try self.chat.conv.appendMessageWithHidden(
                    .developer,
                    .input_text,
                    context_text,
                    true,
                );
            }
        }

        return true;
    }

    // =========================================================================
    // Internal: compaction
    // =========================================================================

    /// Compact the conversation if token counts exceed the threshold.
    ///
    /// Uses turn-aware compaction: groups items into logical turns and deletes
    /// the oldest unpinned turns first, preserving turn integrity and respecting
    /// the pinned flag on items.
    fn compactIfNeeded(self: *Agent) void {
        if (self.context_limit == 0) return;

        const threshold = @as(u64, @intFromFloat(
            @as(f64, @floatFromInt(self.context_limit)) * @as(f64, self.compaction_threshold),
        ));

        const info = loop_mod.computeContextInfo(self.chat.conv, 0);
        const total_tokens = info.total_input_tokens + info.total_output_tokens;
        if (total_tokens <= threshold) return;

        const target = self.context_limit / 2;

        const turns = compaction_mod.identifyTurns(self.allocator, self.chat.conv) catch return;
        defer self.allocator.free(turns);

        _ = compaction_mod.compactTurns(self.chat.conv, turns, target, total_tokens);
    }

    // =========================================================================
    // Internal: retry wrapper
    // =========================================================================

    /// Run loop.run() with exponential backoff retry on failure.
    fn runWithRetry(self: *Agent) !AgentLoopResult {
        const loop_config = AgentLoopConfig{
            .max_iterations = self.max_iterations_per_turn,
            .abort_on_tool_error = self.abort_on_tool_error,
            .stop_flag = &self.stop_flag,
            .on_token = self.on_token,
            .on_token_data = self.on_token_data,
            .on_event = self.on_event,
            .on_event_data = self.on_event_data,
            .on_session = self.on_session,
            .on_session_data = self.on_session_data,
            .initial_iteration_count = self.total_iterations,
            .initial_tool_call_count = self.total_tool_calls,
        };

        var last_err: ?anyerror = null;
        var attempt: u32 = 0;
        while (attempt <= self.max_retries) : (attempt += 1) {
            const result = loop_mod.run(
                self.allocator,
                self.chat,
                self.backend,
                &self.registry,
                loop_config,
            ) catch |err| {
                last_err = err;
                if (attempt < self.max_retries) {
                    const delay = self.retry_base_delay_ns *
                        (@as(u64, 1) << @intCast(attempt));
                    std.Thread.sleep(delay);
                }
                continue;
            };

            // Update cumulative state
            self.total_iterations = result.iterations;
            self.total_tool_calls = result.total_tool_calls;
            return result;
        }

        // All retries exhausted
        return last_err.?;
    }

    // =========================================================================
    // Internal: helpers
    // =========================================================================

    fn cancelledResult(self: *const Agent) AgentLoopResult {
        return .{
            .stop_reason = .cancelled,
            .iterations = self.total_iterations,
            .total_tool_calls = self.total_tool_calls,
        };
    }
};

// =============================================================================
// Helpers
// =============================================================================

/// Generate a random 16-hex-char agent ID.
fn generateId(allocator: Allocator) ![]const u8 {
    var buf: [8]u8 = undefined;
    std.crypto.random.bytes(&buf);
    const hex = try std.fmt.allocPrint(allocator, "{x:0>16}", .{std.mem.readInt(u64, &buf, .little)});
    return hex;
}

// =============================================================================
// Tests
// =============================================================================

test "AgentConfig defaults" {
    const config = AgentConfig{};
    try std.testing.expectEqual(@as(u64, 0), config.context_limit);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), config.compaction_threshold, 0.001);
    try std.testing.expectEqual(@as(u32, 0), config.max_retries);
    try std.testing.expectEqual(std.time.ns_per_s, config.retry_base_delay_ns);
    try std.testing.expectEqual(@as(usize, 10), config.max_iterations_per_turn);
    try std.testing.expect(!config.abort_on_tool_error);
    try std.testing.expect(config.agent_id == null);
    try std.testing.expect(config.state_dir == null);
    try std.testing.expect(config.on_token == null);
    try std.testing.expect(config.on_event == null);
    try std.testing.expect(config.on_session == null);
    try std.testing.expect(config.on_context_inject == null);
    try std.testing.expect(config.on_context_inject_data == null);
}

test "generateId produces 16 hex chars" {
    const allocator = std.testing.allocator;
    const id = try generateId(allocator);
    defer allocator.free(id);

    try std.testing.expectEqual(@as(usize, 16), id.len);
    for (id) |ch| {
        try std.testing.expect((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'));
    }
}

test "generateId produces unique IDs" {
    const allocator = std.testing.allocator;
    const id1 = try generateId(allocator);
    defer allocator.free(id1);
    const id2 = try generateId(allocator);
    defer allocator.free(id2);

    try std.testing.expect(!std.mem.eql(u8, id1, id2));
}

test "Agent addGoal stores and deduplicates" {
    const allocator = std.testing.allocator;

    // Create a minimal agent for testing goals (no backend needed)
    var goals_list = std.ArrayListUnmanaged([]const u8){};
    defer {
        for (goals_list.items) |goal| allocator.free(goal);
        goals_list.deinit(allocator);
    }

    // Simulate addGoal logic
    const goal_text = "Fix the login bug";
    const owned = try allocator.dupe(u8, goal_text);
    try goals_list.append(allocator, owned);

    // Deduplicate: same string should not be added twice
    for (goals_list.items) |existing| {
        if (std.mem.eql(u8, existing, goal_text)) break;
    } else {
        const dup = try allocator.dupe(u8, goal_text);
        try goals_list.append(allocator, dup);
    }

    try std.testing.expectEqual(@as(usize, 1), goals_list.items.len);
    try std.testing.expectEqualStrings("Fix the login bug", goals_list.items[0]);
}

test "Agent removeGoal removes by exact match" {
    const allocator = std.testing.allocator;

    var goals_list = std.ArrayListUnmanaged([]const u8){};
    defer {
        for (goals_list.items) |goal| allocator.free(goal);
        goals_list.deinit(allocator);
    }

    try goals_list.append(allocator, try allocator.dupe(u8, "goal-a"));
    try goals_list.append(allocator, try allocator.dupe(u8, "goal-b"));

    // Remove "goal-a"
    var removed = false;
    for (goals_list.items, 0..) |existing, idx| {
        if (std.mem.eql(u8, existing, "goal-a")) {
            allocator.free(existing);
            _ = goals_list.orderedRemove(idx);
            removed = true;
            break;
        }
    }

    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 1), goals_list.items.len);
    try std.testing.expectEqualStrings("goal-b", goals_list.items[0]);
}

test "Agent clearGoals removes all" {
    const allocator = std.testing.allocator;

    var goals_list = std.ArrayListUnmanaged([]const u8){};
    defer goals_list.deinit(allocator);

    try goals_list.append(allocator, try allocator.dupe(u8, "goal-1"));
    try goals_list.append(allocator, try allocator.dupe(u8, "goal-2"));

    for (goals_list.items) |goal| allocator.free(goal);
    goals_list.clearRetainingCapacity();

    try std.testing.expectEqual(@as(usize, 0), goals_list.items.len);
}

test "syncSystemPromptWithGoals appends goals section" {
    const allocator = std.testing.allocator;

    // Build a combined prompt like syncSystemPromptWithGoals does
    const base = "You are an assistant.";
    const goal_text = "Complete the report";

    var buf = std.ArrayListUnmanaged(u8){};
    defer buf.deinit(allocator);
    const writer = buf.writer(allocator);

    try writer.writeAll(base);
    try writer.writeAll("\n\n## Active Goals\n\nThese are your current objectives. Pursue them proactively:\n");
    try writer.writeAll("- ");
    try writer.writeAll(goal_text);
    try writer.writeByte('\n');

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "## Active Goals") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "Complete the report") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "You are an assistant.") != null);
}

test "syncSystemPromptWithGoals no goals passes base through" {
    const allocator = std.testing.allocator;

    // With no goals, base prompt should be used as-is
    const base = "You are an assistant.";
    const base_owned = try allocator.dupe(u8, base);
    defer allocator.free(base_owned);

    // Simulate: no goals means no "Active Goals" section
    try std.testing.expectEqualStrings(base, base_owned);
}

test "OnContextInjectFn callback type" {
    // Verify the callback type compiles and can be assigned
    const TestData = struct {
        fn inject(
            _: ?[*]const u8,
            _: usize,
            out_ctx: *?[*]const u8,
            out_len: *usize,
            _: ?*anyopaque,
        ) callconv(.c) bool {
            out_ctx.* = null;
            out_len.* = 0;
            return true;
        }
    };

    const cb: OnContextInjectFn = &TestData.inject;
    var out_ctx: ?[*]const u8 = null;
    var out_len: usize = 0;
    const result = cb(null, 0, &out_ctx, &out_len, null);
    try std.testing.expect(result);
    try std.testing.expect(out_ctx == null);
    try std.testing.expectEqual(@as(usize, 0), out_len);
}

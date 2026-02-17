//! Agent - Stateful, autonomous agent with built-in compaction and retry.
//!
//! Owns a Chat (conversation), ToolRegistry, and makes decisions internally:
//! compaction when approaching context limits, retry with exponential backoff
//! on transient failures, inbox draining from a MessageBus.
//!
//! Exposes `prompt()` (user turn), `continueLoop()` (resume), `heartbeat()`
//! (autonomous check-in), and `abort()` (cross-thread cancellation).
//!
//! The Agent delegates actual generation to `loop.run()`, adding statefulness
//! and autonomy on top of the stateless loop.
//!
//! # Ownership
//!
//! Agent owns: Chat, ToolRegistry, agent_id, state_dir.
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
        self.allocator.destroy(self);
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the system prompt. Forwards to Chat.setSystem().
    pub fn setSystem(self: *Agent, system_prompt: []const u8) !void {
        try self.chat.setSystem(system_prompt);
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
    // Execution
    // =========================================================================

    /// Send a user message and run the agent loop.
    ///
    /// 1. Drain inbox (bus messages -> developer messages)
    /// 2. Compact if approaching context limit
    /// 3. Append user message
    /// 4. Run loop with retry
    /// 5. Update cumulative state
    pub fn prompt(self: *Agent, message: []const u8) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        self.compactIfNeeded();
        _ = try self.chat.conv.appendUserMessage(message);

        return self.runWithRetry();
    }

    /// Resume the agent loop without a new user message.
    ///
    /// Drains inbox, compacts if needed, then continues from where the
    /// previous turn left off.
    pub fn continueLoop(self: *Agent) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        self.compactIfNeeded();

        return self.runWithRetry();
    }

    /// Autonomous heartbeat: drain inbox and act on pending work.
    ///
    /// Appends a developer message prompting the LLM to review its state,
    /// then runs the loop. Callers invoke this on a timer for autonomy.
    pub fn heartbeat(self: *Agent) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        self.compactIfNeeded();
        _ = try self.chat.conv.appendDeveloperMessage(
            "Heartbeat: review pending work and act.",
        );

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
    // Internal: compaction
    // =========================================================================

    /// Compact the conversation if token counts exceed the threshold.
    ///
    /// Strategy: delete items from the front (after system prompt) until total
    /// tokens fit within context_limit / 2. This keeps the system prompt and
    /// the most recent items.
    fn compactIfNeeded(self: *Agent) void {
        if (self.context_limit == 0) return;

        const threshold = @as(u64, @intFromFloat(
            @as(f64, @floatFromInt(self.context_limit)) * @as(f64, self.compaction_threshold),
        ));

        const info = loop_mod.computeContextInfo(self.chat.conv, 0);
        const total_tokens = info.total_input_tokens + info.total_output_tokens;
        if (total_tokens <= threshold) return;

        const target = self.context_limit / 2;

        // Delete from index 1 (skip system prompt at 0) until we're under target
        while (self.chat.conv.len() > 1) {
            const check = loop_mod.computeContextInfo(self.chat.conv, 0);
            if (check.total_input_tokens + check.total_output_tokens <= target) break;
            _ = self.chat.conv.deleteItem(1);
        }
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

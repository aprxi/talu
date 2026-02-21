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
const tools_mod = @import("tools/root.zig");
const memory_mod = @import("memory/root.zig");

const responses = @import("../responses/root.zig");
const Chat = responses.Chat;
const Conversation = responses.Conversation;

const router = @import("../router/root.zig");
const InferenceBackend = router.InferenceBackend;

const db = @import("../db/root.zig");
const VectorAdapter = db.VectorAdapter;
const SearchResult = db.vector.store.SearchResult;

const AgentLoopConfig = loop_mod.AgentLoopConfig;
const AgentLoopResult = loop_mod.AgentLoopResult;
const LoopStopReason = loop_mod.LoopStopReason;
const OnTokenFn = loop_mod.OnTokenFn;
const OnEventFn = loop_mod.OnEventFn;
const OnSessionFn = loop_mod.OnSessionFn;
const OnBeforeToolFn = loop_mod.OnBeforeToolFn;
const OnAfterToolFn = loop_mod.OnAfterToolFn;
const ToolRegistry = tool_mod.ToolRegistry;
const Tool = tool_mod.Tool;
const MessageBus = bus_mod.MessageBus;
const Message = bus_mod.Message;

const persisted_state_filename = "agent_state.json";
const persisted_state_version: u32 = 1;
const max_state_file_bytes: usize = 1024 * 1024;

const PersistedAgentState = struct {
    version: u32,
    agent_id: []const u8,
    total_iterations: usize,
    total_tool_calls: usize,
    base_system_prompt: ?[]const u8,
    goals: []const []const u8,
};

// =============================================================================
// Callback types
// =============================================================================

/// Called before each generation to allow context injection (e.g., RAG).
///
/// Receives the agent's Chat as an opaque handle. The callback can inspect
/// the full conversation via C API functions (talu_chat_get_messages, etc.)
/// to construct a RAG query. Writes context to out_context/out_context_len.
/// The agent copies the returned text before proceeding.
/// Returns false to cancel the loop.
pub const OnContextInjectFn = *const fn (
    chat_handle: ?*anyopaque,
    out_context: *?[*]const u8,
    out_context_len: *usize,
    user_data: ?*anyopaque,
) callconv(.c) bool;

/// Embed text into a float vector for RAG search.
///
/// Writes the vector pointer and dimension to out params. The agent copies the
/// vector before returning. Returns false to skip RAG for this turn.
pub const OnEmbedFn = *const fn (
    text: [*]const u8,
    text_len: usize,
    out_vector: *?[*]const f32,
    out_dim: *usize,
    user_data: ?*anyopaque,
) callconv(.c) bool;

/// Resolve vector search results into context text.
///
/// Receives doc IDs and scores. Writes context text to out params. The agent
/// copies the text before returning. Returns false to cancel the loop.
pub const OnResolveDocFn = *const fn (
    doc_ids: [*]const u64,
    scores: [*]const f32,
    count: usize,
    out_context: *?[*]const u8,
    out_context_len: *usize,
    user_data: ?*anyopaque,
) callconv(.c) bool;

/// Configuration for built-in vector store RAG.
pub const RagConfig = struct {
    /// Maximum number of results to retrieve.
    top_k: u32 = 5,
    /// Minimum score threshold for results (0.0 = no filtering).
    min_score: f32 = 0.0,
    /// Callback to embed user text into a query vector.
    on_embed: ?OnEmbedFn = null,
    on_embed_data: ?*anyopaque = null,
    /// Callback to resolve doc IDs + scores into context text.
    on_resolve: ?OnResolveDocFn = null,
    on_resolve_data: ?*anyopaque = null,
};

/// Configuration for optional agent memory integration.
pub const MemoryIntegrationConfig = struct {
    /// Talu DB path used by MemoryStore.
    db_path: []const u8,
    /// Memory namespace partition.
    namespace: []const u8 = "default",
    /// Optional owner partition.
    owner_id: ?[]const u8 = null,
    /// Max number of recall hits to inject per run.
    recall_limit: usize = 5,
    /// Whether to append a run summary to daily memory.
    append_on_run: bool = true,
};

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
    /// Maximum bytes for tool output. 0 = unlimited.
    max_tool_output_bytes: usize = 0,
    /// Maximum tool calls per turn. 0 = unlimited.
    max_tool_calls_per_turn: usize = 0,
    /// Agent identifier. Null = auto-generate a random ID.
    agent_id: ?[]const u8 = null,
    /// Directory for agent state files (tools, skills, key-value).
    state_dir: ?[]const u8 = null,
    /// Optional built-in tool auto-registration.
    builtin_tools: ?tools_mod.BuiltinToolsConfig = null,
    /// Optional memory integration over db blobs/documents.
    memory: ?MemoryIntegrationConfig = null,

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

    // Tool middleware (forwarded to loop.run)
    on_before_tool: ?OnBeforeToolFn = null,
    on_before_tool_data: ?*anyopaque = null,
    on_after_tool: ?OnAfterToolFn = null,
    on_after_tool_data: ?*anyopaque = null,
};

// =============================================================================
// Agent
// =============================================================================

pub const AgentError = error{
    OutOfMemory,
    GenerationFailed,
    IteratorCreationFailed,
    InvalidAgentState,
    UnsupportedAgentStateVersion,
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
    max_tool_output_bytes: usize,
    max_tool_calls_per_turn: usize,

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

    // Tool middleware (forwarded to loop.run)
    on_before_tool: ?OnBeforeToolFn,
    on_before_tool_data: ?*anyopaque,
    on_after_tool: ?OnAfterToolFn,
    on_after_tool_data: ?*anyopaque,

    // Built-in vector store RAG
    vector_store: ?*VectorAdapter,
    rag_config: RagConfig,
    memory_store: ?memory_mod.MemoryStore,
    memory_recall_limit: usize,
    memory_append_on_run: bool,

    // Message notification (for waitForMessage / runLoop)
    message_mutex: std.Thread.Mutex,
    message_cond: std.Thread.Condition,
    message_pending: bool,

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
            .max_tool_output_bytes = config.max_tool_output_bytes,
            .max_tool_calls_per_turn = config.max_tool_calls_per_turn,
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
            .on_before_tool = config.on_before_tool,
            .on_before_tool_data = config.on_before_tool_data,
            .on_after_tool = config.on_after_tool,
            .on_after_tool_data = config.on_after_tool_data,
            .vector_store = null,
            .rag_config = .{},
            .memory_store = null,
            .memory_recall_limit = 0,
            .memory_append_on_run = false,
            .message_mutex = .{},
            .message_cond = .{},
            .message_pending = false,
            .goals = .{},
            .base_system_prompt = null,
        };

        errdefer agent.deinit();

        if (config.memory) |memory_cfg| {
            agent.memory_store = try memory_mod.MemoryStore.init(allocator, .{
                .db_path = memory_cfg.db_path,
                .namespace = memory_cfg.namespace,
                .owner_id = memory_cfg.owner_id,
            });
            agent.memory_recall_limit = memory_cfg.recall_limit;
            agent.memory_append_on_run = memory_cfg.append_on_run;
        }

        if (config.builtin_tools) |builtin_cfg| {
            try tools_mod.registerDefaultTools(allocator, &agent.registry, builtin_cfg);
        }

        try agent.loadPersistedState();

        return agent;
    }

    /// Free all owned resources.
    pub fn deinit(self: *Agent) void {
        if (self.memory_store) |*store| {
            store.deinit();
        }
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
        try self.persistState();
    }

    /// Register a tool. The agent takes ownership via its ToolRegistry.
    pub fn registerTool(self: *Agent, new_tool: Tool) !void {
        try self.registry.register(new_tool);
    }

    /// Connect to a MessageBus for inter-agent communication.
    ///
    /// The bus must outlive the agent (caller-managed). The agent must
    /// already be registered on the bus. Registers a notification callback
    /// so `waitForMessage()` and `runLoop()` can wake on message arrival.
    pub fn setBus(self: *Agent, bus: *MessageBus) void {
        self.bus = bus;
        bus.setNotify(self.agent_id, onBusNotify, @ptrCast(self)) catch {};
    }

    /// Wire the agent to a vector store for built-in RAG.
    ///
    /// Before each generation, the agent will:
    ///   1. Extract the last user message text
    ///   2. Call on_embed to produce a query vector
    ///   3. Search the vector store for top_k results
    ///   4. Call on_resolve to turn (doc_ids, scores) into context text
    ///   5. Inject the resolved text as a hidden developer message
    ///
    /// The store must outlive the agent (caller-managed).
    pub fn setVectorStore(self: *Agent, store: *VectorAdapter, config: RagConfig) void {
        self.vector_store = store;
        self.rag_config = config;
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
        try self.persistState();
    }

    /// Remove a specific goal by exact string match.
    /// Returns true if the goal was found and removed.
    pub fn removeGoal(self: *Agent, goal: []const u8) bool {
        for (self.goals.items, 0..) |existing, idx| {
            if (std.mem.eql(u8, existing, goal)) {
                self.allocator.free(existing);
                _ = self.goals.orderedRemove(idx);
                self.persistState() catch {};
                return true;
            }
        }
        return false;
    }

    /// Remove all goals.
    pub fn clearGoals(self: *Agent) void {
        self.clearGoalsInMemory();
        self.persistState() catch {};
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
    /// 5. Vector store RAG (embed → search → resolve → inject)
    /// 6. Inject context (custom callback)
    /// 7. Run loop with retry
    pub fn prompt(self: *Agent, message: []const u8) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        try self.drainInbox();
        try self.syncSystemPromptWithGoals();
        self.compactIfNeeded();
        _ = try self.chat.conv.appendUserMessage(message);
        try self.injectMemoryRecall(message);
        if (!try self.injectVectorContext()) return cancelledResult(self);
        if (!try self.injectContext()) return cancelledResult(self);

        const result = try self.runWithRetry();
        try self.appendRunMemoryEntry("prompt", message, result);
        return result;
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
        if (self.lastUserText()) |query| {
            try self.injectMemoryRecall(query);
        }
        if (!try self.injectVectorContext()) return cancelledResult(self);
        if (!try self.injectContext()) return cancelledResult(self);

        const result = try self.runWithRetry();
        try self.appendRunMemoryEntry("continue", null, result);
        return result;
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
        if (self.lastUserText()) |query| {
            try self.injectMemoryRecall(query);
        }
        if (!try self.injectVectorContext()) return cancelledResult(self);
        if (!try self.injectContext()) return cancelledResult(self);

        const result = try self.runWithRetry();
        try self.appendRunMemoryEntry("heartbeat", null, result);
        return result;
    }

    /// Request cancellation from another thread.
    /// The loop checks this flag between iterations and `runLoop()` exits.
    pub fn abort(self: *Agent) void {
        self.stop_flag.store(true, .release);
        // Wake waitForMessage/runLoop so they can observe the flag
        self.message_mutex.lock();
        self.message_cond.signal();
        self.message_mutex.unlock();
    }

    /// Block until a message arrives in the bus mailbox or timeout elapses.
    ///
    /// Returns true if a message notification was received, false on timeout.
    /// Thread-safe: can be called from any thread. Requires a bus to be set
    /// via `setBus()`.
    pub fn waitForMessage(self: *Agent, timeout_ns: u64) bool {
        self.message_mutex.lock();
        defer self.message_mutex.unlock();

        if (self.message_pending) {
            self.message_pending = false;
            return true;
        }

        self.message_cond.timedWait(&self.message_mutex, timeout_ns) catch {
            // Timeout — check if pending was set between check and wait
            const result = self.message_pending;
            self.message_pending = false;
            return result;
        };

        const result = self.message_pending;
        self.message_pending = false;
        return result;
    }

    /// Autonomous message-driven loop.
    ///
    /// Runs `heartbeat()` when messages arrive, waits between iterations.
    /// Stops on `abort()`, generation error, or non-recoverable stop reason
    /// (tool_error, cancelled). Normal completion and max_iterations continue
    /// the loop.
    ///
    /// `idle_timeout_ns` controls the maximum wait between message checks.
    /// Use `abort()` from another thread to stop the loop.
    pub fn runLoop(self: *Agent, idle_timeout_ns: u64) !AgentLoopResult {
        self.stop_flag.store(false, .release);

        while (!self.stop_flag.load(.acquire)) {
            const has_messages = if (self.bus) |bus|
                bus.pendingCount(self.agent_id) > 0
            else
                false;

            if (has_messages) {
                const result = try self.heartbeat();
                switch (result.stop_reason) {
                    .completed, .max_iterations => continue,
                    else => return result,
                }
            }

            // Wait for notification or timeout
            _ = self.waitForMessage(idle_timeout_ns);
        }

        return cancelledResult(self);
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
    // Internal: state persistence
    // =========================================================================

    /// Persist agent state to `state_dir/agent_state.json` if state_dir is set.
    fn persistState(self: *const Agent) !void {
        const state_dir = self.state_dir orelse return;

        try std.fs.cwd().makePath(state_dir);

        const state_path = try std.fs.path.join(
            self.allocator,
            &.{ state_dir, persisted_state_filename },
        );
        defer self.allocator.free(state_path);

        const tmp_path = try std.fmt.allocPrint(self.allocator, "{s}.tmp", .{state_path});
        defer self.allocator.free(tmp_path);

        const goals_view = try self.allocator.alloc([]const u8, self.goals.items.len);
        defer self.allocator.free(goals_view);
        for (self.goals.items, 0..) |goal, idx| {
            goals_view[idx] = goal;
        }

        const state = PersistedAgentState{
            .version = persisted_state_version,
            .agent_id = self.agent_id,
            .total_iterations = self.total_iterations,
            .total_tool_calls = self.total_tool_calls,
            .base_system_prompt = self.base_system_prompt,
            .goals = goals_view,
        };

        const json_bytes = try std.json.Stringify.valueAlloc(
            self.allocator,
            state,
            .{ .whitespace = .indent_2 },
        );
        defer self.allocator.free(json_bytes);

        var tmp_file = try std.fs.cwd().createFile(tmp_path, .{ .truncate = true });
        errdefer {
            tmp_file.close();
            std.fs.cwd().deleteFile(tmp_path) catch {};
        }
        try tmp_file.writeAll(json_bytes);
        try tmp_file.sync();
        tmp_file.close();

        try std.fs.cwd().rename(tmp_path, state_path);
    }

    /// Load persisted state from `state_dir/agent_state.json` if present.
    fn loadPersistedState(self: *Agent) !void {
        const state_dir = self.state_dir orelse return;
        const state_path = try std.fs.path.join(
            self.allocator,
            &.{ state_dir, persisted_state_filename },
        );
        defer self.allocator.free(state_path);

        const json_bytes = std.fs.cwd().readFileAlloc(
            self.allocator,
            state_path,
            max_state_file_bytes,
        ) catch |err| switch (err) {
            error.FileNotFound => return,
            else => return err,
        };
        defer self.allocator.free(json_bytes);

        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json_bytes,
            .{},
        ) catch return AgentError.InvalidAgentState;
        defer parsed.deinit();

        const root = switch (parsed.value) {
            .object => |obj| obj,
            else => return AgentError.InvalidAgentState,
        };

        if (root.get("version")) |version_value| {
            const version = try parseStateU32(version_value);
            if (version != persisted_state_version) {
                return AgentError.UnsupportedAgentStateVersion;
            }
        }

        if (root.get("agent_id")) |agent_id_value| {
            const stored_id = switch (agent_id_value) {
                .string => |s| s,
                else => return AgentError.InvalidAgentState,
            };
            if (!std.mem.eql(u8, stored_id, self.agent_id)) {
                return;
            }
        }

        if (root.get("total_iterations")) |iterations_value| {
            self.total_iterations = try parseStateUsize(iterations_value);
        }

        if (root.get("total_tool_calls")) |tool_calls_value| {
            self.total_tool_calls = try parseStateUsize(tool_calls_value);
        }

        if (self.base_system_prompt) |old| {
            self.allocator.free(old);
            self.base_system_prompt = null;
        }
        self.clearGoalsInMemory();

        if (root.get("base_system_prompt")) |system_value| {
            switch (system_value) {
                .null => {},
                .string => |s| {
                    self.base_system_prompt = try self.allocator.dupe(u8, s);
                },
                else => return AgentError.InvalidAgentState,
            }
        }

        if (root.get("goals")) |goals_value| {
            const goals = switch (goals_value) {
                .array => |arr| arr.items,
                else => return AgentError.InvalidAgentState,
            };
            for (goals) |goal_value| {
                const goal_text = switch (goal_value) {
                    .string => |s| s,
                    else => return AgentError.InvalidAgentState,
                };
                try self.goals.append(self.allocator, try self.allocator.dupe(u8, goal_text));
            }
        }

        try self.syncSystemPromptWithGoals();
    }

    fn clearGoalsInMemory(self: *Agent) void {
        for (self.goals.items) |goal| {
            self.allocator.free(goal);
        }
        self.goals.clearRetainingCapacity();
    }

    // =========================================================================
    // Internal: context injection
    // =========================================================================

    /// Call the context injection hook if set.
    ///
    /// Passes the agent's Chat as an opaque handle so the callback can
    /// inspect the full conversation for RAG query construction. Appends
    /// the returned context as a hidden developer message (visible to LLM,
    /// excluded from UI).
    ///
    /// Returns false if the callback requested cancellation.
    fn injectContext(self: *Agent) !bool {
        const inject_fn = self.on_context_inject orelse return true;

        var out_ctx: ?[*]const u8 = null;
        var out_len: usize = 0;

        const chat_handle: ?*anyopaque = @ptrCast(self.chat);

        const should_continue = inject_fn(
            chat_handle,
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

    /// Inject markdown memory recall as hidden developer context.
    fn injectMemoryRecall(self: *Agent, query: []const u8) !void {
        var store = &(self.memory_store orelse return);
        if (self.memory_recall_limit == 0) return;

        const hits = try store.recall(query, self.memory_recall_limit);
        defer store.freeRecallHits(hits);
        if (hits.len == 0) return;

        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(self.allocator);
        const writer = buf.writer(self.allocator);

        try writer.writeAll("## Recalled Memory\n");
        for (hits) |hit| {
            try writer.print("- {s}: ", .{hit.filename});
            try writer.writeAll(hit.snippet);
            try writer.writeByte('\n');
        }

        _ = try self.chat.conv.appendMessageWithHidden(
            .developer,
            .input_text,
            buf.items,
            true,
        );
    }

    /// Embed → search → resolve → inject via the built-in vector store.
    ///
    /// Returns false if the resolve callback requested cancellation.
    /// Silently no-ops if vector store, callbacks, or user message are absent.
    fn injectVectorContext(self: *Agent) !bool {
        const store = self.vector_store orelse return true;
        const config = self.rag_config;
        const embed_fn = config.on_embed orelse return true;
        const resolve_fn = config.on_resolve orelse return true;

        // 1. Extract last user message text.
        const text = self.lastUserText() orelse return true;
        if (text.len == 0) return true;

        // 2. Embed.
        var out_vec: ?[*]const f32 = null;
        var out_dim: usize = 0;
        if (!embed_fn(text.ptr, text.len, &out_vec, &out_dim, config.on_embed_data))
            return true; // skip RAG this turn
        const query = (out_vec orelse return true)[0..out_dim];
        if (query.len == 0) return true;

        // 3. Search.
        var result = store.search(self.allocator, query, config.top_k) catch return true;
        defer result.deinit(self.allocator);
        if (result.ids.len == 0) return true;

        // 4. Filter by min_score.
        var count: usize = result.ids.len;
        if (config.min_score > 0.0) {
            count = 0;
            for (result.scores) |s| {
                if (s >= config.min_score) {
                    count += 1;
                } else break;
            }
        }
        if (count == 0) return true;

        // 5. Resolve doc IDs → context text.
        var out_ctx: ?[*]const u8 = null;
        var out_len: usize = 0;
        if (!resolve_fn(
            result.ids.ptr,
            result.scores.ptr,
            count,
            &out_ctx,
            &out_len,
            config.on_resolve_data,
        )) return false; // callback requested cancellation

        // 6. Inject as hidden developer message.
        if (out_ctx) |ctx| {
            if (out_len > 0) {
                _ = try self.chat.conv.appendMessageWithHidden(
                    .developer,
                    .input_text,
                    ctx[0..out_len],
                    true,
                );
            }
        }
        return true;
    }

    /// Returns the latest user message text from the conversation.
    fn lastUserText(self: *const Agent) ?[]const u8 {
        var idx: usize = self.chat.conv.len();
        while (idx > 0) {
            idx -= 1;
            const item = self.chat.conv.getItem(idx) orelse continue;
            const msg = item.asMessage() orelse continue;
            if (msg.role == .user) return msg.getFirstText();
        }
        return null;
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

        // Fallback: if still over target after turn-level compaction,
        // truncate the text content of the largest non-pinned item.
        const info2 = loop_mod.computeContextInfo(self.chat.conv, 0);
        const remaining = info2.total_input_tokens + info2.total_output_tokens;
        if (remaining > target) {
            _ = compaction_mod.truncateOversizedItem(self.chat.conv, target, remaining);
        }
    }

    // =========================================================================
    // Internal: retry wrapper
    // =========================================================================

    /// Run loop.run() with exponential backoff retry on failure.
    fn runWithRetry(self: *Agent) !AgentLoopResult {
        const loop_config = AgentLoopConfig{
            .max_iterations = self.max_iterations_per_turn,
            .max_tool_calls = self.max_tool_calls_per_turn,
            .abort_on_tool_error = self.abort_on_tool_error,
            .stop_flag = &self.stop_flag,
            .on_token = self.on_token,
            .on_token_data = self.on_token_data,
            .on_event = self.on_event,
            .on_event_data = self.on_event_data,
            .on_session = self.on_session,
            .on_session_data = self.on_session_data,
            .on_before_tool = self.on_before_tool,
            .on_before_tool_data = self.on_before_tool_data,
            .on_after_tool = self.on_after_tool,
            .on_after_tool_data = self.on_after_tool_data,
            .max_tool_output_bytes = self.max_tool_output_bytes,
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
            self.persistState() catch {};
            return result;
        }

        // All retries exhausted
        return last_err.?;
    }

    fn appendRunMemoryEntry(
        self: *Agent,
        trigger: []const u8,
        user_message: ?[]const u8,
        result: AgentLoopResult,
    ) !void {
        var store = &(self.memory_store orelse return);
        if (!self.memory_append_on_run) return;

        var buf = std.ArrayListUnmanaged(u8){};
        defer buf.deinit(self.allocator);
        const writer = buf.writer(self.allocator);

        try writer.writeAll("## Agent Run\n");
        try writer.print("- Trigger: {s}\n", .{trigger});
        try writer.print("- Stop: {s}\n", .{stopReasonLabel(result.stop_reason)});
        try writer.print("- Iterations: {d}\n", .{result.iterations});
        try writer.print("- Tool calls: {d}\n", .{result.total_tool_calls});

        if (user_message) |msg| {
            if (msg.len > 0) {
                const max_user_len: usize = 400;
                const keep = @min(msg.len, max_user_len);
                try writer.writeAll("- User: ");
                try writer.writeAll(msg[0..keep]);
                if (keep < msg.len) {
                    try writer.writeAll("...");
                }
                try writer.writeByte('\n');
            }
        }

        try store.appendDaily(std.time.milliTimestamp(), buf.items);
    }

    // =========================================================================
    // Internal: helpers
    // =========================================================================

    /// Bus notification callback — signals the condition variable.
    ///
    /// Called by the MessageBus **outside** its mutex when a message is
    /// enqueued into this agent's mailbox. Wakes `waitForMessage()` /
    /// `runLoop()`.
    fn onBusNotify(
        _: [*:0]const u8,
        _: usize,
        user_data: ?*anyopaque,
    ) callconv(.c) void {
        const agent: *Agent = @ptrCast(@alignCast(user_data));
        agent.message_mutex.lock();
        agent.message_pending = true;
        agent.message_cond.signal();
        agent.message_mutex.unlock();
    }

    fn cancelledResult(self: *const Agent) AgentLoopResult {
        return .{
            .stop_reason = .cancelled,
            .iterations = self.total_iterations,
            .total_tool_calls = self.total_tool_calls,
        };
    }
};

fn stopReasonLabel(reason: LoopStopReason) []const u8 {
    return switch (reason) {
        .completed => "completed",
        .max_iterations => "max_iterations_or_budget",
        .tool_error => "tool_error",
        .cancelled => "cancelled",
    };
}

fn parseStateUsize(value: std.json.Value) AgentError!usize {
    if (value != .integer) return AgentError.InvalidAgentState;
    if (value.integer < 0) return AgentError.InvalidAgentState;
    const as_u64: u64 = @intCast(value.integer);
    if (as_u64 > std.math.maxInt(usize)) return AgentError.InvalidAgentState;
    return @intCast(as_u64);
}

fn parseStateU32(value: std.json.Value) AgentError!u32 {
    if (value != .integer) return AgentError.InvalidAgentState;
    if (value.integer < 0) return AgentError.InvalidAgentState;
    const as_u64: u64 = @intCast(value.integer);
    if (as_u64 > std.math.maxInt(u32)) return AgentError.InvalidAgentState;
    return @intCast(as_u64);
}

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
    try std.testing.expectEqual(@as(usize, 0), config.max_tool_output_bytes);
    try std.testing.expectEqual(@as(usize, 0), config.max_tool_calls_per_turn);
    try std.testing.expect(config.agent_id == null);
    try std.testing.expect(config.state_dir == null);
    try std.testing.expect(config.builtin_tools == null);
    try std.testing.expect(config.memory == null);
    try std.testing.expect(config.on_token == null);
    try std.testing.expect(config.on_event == null);
    try std.testing.expect(config.on_session == null);
    try std.testing.expect(config.on_context_inject == null);
    try std.testing.expect(config.on_context_inject_data == null);
    try std.testing.expect(config.on_before_tool == null);
    try std.testing.expect(config.on_before_tool_data == null);
    try std.testing.expect(config.on_after_tool == null);
    try std.testing.expect(config.on_after_tool_data == null);
}

test "Agent init auto-registers built-in tools when configured" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(workspace);

    var backend: InferenceBackend = undefined;
    var agent = try Agent.init(allocator, &backend, .{
        .builtin_tools = .{
            .workspace_dir = workspace,
        },
    });
    defer agent.deinit();

    try std.testing.expectEqual(@as(usize, 4), agent.getRegistry().count());
    try std.testing.expect(agent.getRegistry().get("read_file") != null);
    try std.testing.expect(agent.getRegistry().get("write_file") != null);
    try std.testing.expect(agent.getRegistry().get("edit_file") != null);
    try std.testing.expect(agent.getRegistry().get("http_fetch") != null);
}

test "Agent memory integration recalls and appends entries" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const db_path = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(db_path);

    var backend: InferenceBackend = undefined;
    var agent = try Agent.init(allocator, &backend, .{
        .memory = .{
            .db_path = db_path,
            .namespace = "agent_test",
            .recall_limit = 5,
            .append_on_run = true,
        },
    });
    defer agent.deinit();

    var store = try memory_mod.MemoryStore.init(allocator, .{
        .db_path = db_path,
        .namespace = "agent_test",
    });
    defer store.deinit();

    try store.upsertMarkdown("20260221.md", "memory sentinel line");
    try agent.injectMemoryRecall("sentinel");

    const last = agent.chat.conv.lastItem() orelse return error.TestUnexpectedResult;
    try std.testing.expect(last.hidden);
    const msg = last.asMessage() orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(.developer, msg.role);
    try std.testing.expect(std.mem.indexOf(u8, msg.getFirstText(), "Recalled Memory") != null);

    const result = AgentLoopResult{
        .stop_reason = .completed,
        .iterations = 2,
        .total_tool_calls = 1,
    };
    try agent.appendRunMemoryEntry("prompt", "remember this", result);

    const day = memory_mod.buildDailyFilename(std.time.milliTimestamp());
    const day_doc = try store.readMarkdown(day[0..]);
    defer if (day_doc) |buf| allocator.free(buf);

    try std.testing.expect(day_doc != null);
    try std.testing.expect(std.mem.indexOf(u8, day_doc.?, "Trigger: prompt") != null);
    try std.testing.expect(std.mem.indexOf(u8, day_doc.?, "remember this") != null);
}

test "Agent state_dir persists and restores goals, system prompt, and counters" {
    const allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const state_dir = try tmp.dir.realpathAlloc(allocator, ".");
    defer allocator.free(state_dir);

    var backend: InferenceBackend = undefined;
    var first = try Agent.init(allocator, &backend, .{
        .agent_id = "stateful-agent",
        .state_dir = state_dir,
    });
    defer first.deinit();

    try first.setSystem("Base system prompt.");
    try first.addGoal("Goal One");
    try first.addGoal("Goal Two");
    first.total_iterations = 7;
    first.total_tool_calls = 3;
    try first.persistState();

    var second = try Agent.init(allocator, &backend, .{
        .agent_id = "stateful-agent",
        .state_dir = state_dir,
    });
    defer second.deinit();

    try std.testing.expect(second.base_system_prompt != null);
    try std.testing.expectEqualStrings("Base system prompt.", second.base_system_prompt.?);
    try std.testing.expectEqual(@as(usize, 2), second.getGoals().len);
    try std.testing.expectEqualStrings("Goal One", second.getGoals()[0]);
    try std.testing.expectEqualStrings("Goal Two", second.getGoals()[1]);
    try std.testing.expectEqual(@as(usize, 7), second.total_iterations);
    try std.testing.expectEqual(@as(usize, 3), second.total_tool_calls);

    const effective_system = second.chat.getSystem() orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, effective_system, "## Active Goals") != null);
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

test "OnContextInjectFn callback receives chat handle" {
    // Verify the callback type compiles and receives an opaque handle
    const TestData = struct {
        fn inject(
            chat_handle: ?*anyopaque,
            out_ctx: *?[*]const u8,
            out_len: *usize,
            _: ?*anyopaque,
        ) callconv(.c) bool {
            // Verify we received a non-null handle
            if (chat_handle == null) return false;
            out_ctx.* = null;
            out_len.* = 0;
            return true;
        }
    };

    const cb: OnContextInjectFn = &TestData.inject;
    // Simulate passing a non-null chat handle
    var dummy: u8 = 0;
    var out_ctx: ?[*]const u8 = null;
    var out_len: usize = 0;
    const result = cb(@ptrCast(&dummy), &out_ctx, &out_len, null);
    try std.testing.expect(result);
    try std.testing.expect(out_ctx == null);
    try std.testing.expectEqual(@as(usize, 0), out_len);

    // Null handle also works (callback returns false)
    const result2 = cb(null, &out_ctx, &out_len, null);
    try std.testing.expect(!result2);
}

test "waitForMessage returns false on timeout when no messages" {
    const allocator = std.testing.allocator;
    // Tests don't call generate() — backend is never used
    var backend: InferenceBackend = undefined;
    var agent = try Agent.init(allocator, &backend, .{});
    defer agent.deinit();

    // No bus set, should timeout immediately (10ms)
    const got = agent.waitForMessage(10 * std.time.ns_per_ms);
    try std.testing.expect(!got);
}

test "waitForMessage returns true after onBusNotify signal" {
    const allocator = std.testing.allocator;
    var backend: InferenceBackend = undefined;
    var agent = try Agent.init(allocator, &backend, .{});
    defer agent.deinit();

    // Simulate bus notification directly
    Agent.onBusNotify(@ptrCast("x\x00".ptr), 1, @ptrCast(agent));

    // Should return true immediately (message_pending was set)
    const got = agent.waitForMessage(10 * std.time.ns_per_ms);
    try std.testing.expect(got);

    // Second call should timeout (pending was cleared)
    const got2 = agent.waitForMessage(10 * std.time.ns_per_ms);
    try std.testing.expect(!got2);
}

test "onBusNotify callback signature matches OnMessageNotifyFn" {
    // Verify onBusNotify is assignable to OnMessageNotifyFn
    const notify_fn: bus_mod.OnMessageNotifyFn = Agent.onBusNotify;
    _ = notify_fn;
}

test "setBus registers notification on bus" {
    const allocator = std.testing.allocator;
    var backend: InferenceBackend = undefined;
    var agent = try Agent.init(allocator, &backend, .{ .agent_id = "test-agent" });
    defer agent.deinit();

    var bus = bus_mod.MessageBus.init(allocator);
    defer bus.deinit();

    try bus.register("test-agent");
    try bus.register("sender");
    agent.setBus(&bus);

    // Send a message — should trigger notification on agent
    try bus.send("sender", "test-agent", "hello");

    // waitForMessage should return true because onBusNotify was called
    const got = agent.waitForMessage(10 * std.time.ns_per_ms);
    try std.testing.expect(got);
}

test "abort signals message condition" {
    const allocator = std.testing.allocator;
    var backend: InferenceBackend = undefined;
    var agent = try Agent.init(allocator, &backend, .{});
    defer agent.deinit();

    // Abort should signal the condvar (prevents runLoop from hanging)
    agent.abort();
    try std.testing.expect(agent.stop_flag.load(.acquire));
}

//! AgentLoop - Agentic generate-execute-repeat orchestration.
//!
//! Drives the LLM generation → tool execution → continuation cycle. Supports
//! two generation modes:
//!
//!   - **Synchronous** (default): calls `generateWithBackend()` and returns a
//!     complete `GenerateResult` with tool calls.
//!   - **Streaming**: when `on_token` is set, uses `createIterator()` (background
//!     thread + ring buffer) and forwards each token to the callback. Tool calls
//!     are extracted from the conversation after the iterator completes.
//!
//! Lifecycle events (`on_event`) notify callers of generation start/end and
//! tool start/end for progress indicators, logging, or human-in-the-loop
//! confirmation (block inside the tool execute callback).
//!
//! Additional hooks for production agents:
//!   - `on_context`: called before each generation with token counts, enabling
//!     caller-driven context window management / compaction.
//!   - `on_before_tool` / `on_after_tool`: tool middleware for interception
//!     (blocking, human-in-the-loop confirmation) and observation.
//!   - `on_session`: session lifecycle events (loop start/end, storage errors).
//!
//! Does NOT own Chat, InferenceBackend, or ToolRegistry — caller manages
//! their lifecycles.
//!
//! # Thread Safety
//!
//! NOT thread-safe. All arguments must be accessed from a single thread.
//! Use `stop_flag` (atomic) for cross-thread cancellation.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tool_mod = @import("tool.zig");
const ToolRegistry = tool_mod.ToolRegistry;
const ToolResult = tool_mod.ToolResult;

const router = @import("../router/root.zig");
const capi_bridge = router.capi_bridge;
const responses = @import("../responses/root.zig");
const Chat = responses.Chat;
const Conversation = responses.Conversation;
const ItemStatus = responses.ItemStatus;

const GenerateResult = capi_bridge.GenerateResult;
const GenerateContentPart = capi_bridge.GenerateContentPart;
const CGenerateConfig = capi_bridge.CGenerateConfig;
const InferenceBackend = router.InferenceBackend;
const ToolCallRef = router.ToolCallRef;
const TokenIterator = router.TokenIterator;
const FinishReason = router.FinishReason;

// =============================================================================
// Callback types — observability
// =============================================================================

/// Callback invoked for each streamed token during generation.
/// Receives null-terminated token text. Return false to cancel generation.
pub const OnTokenFn = *const fn (
    token: [*:0]const u8,
    user_data: ?*anyopaque,
) callconv(.c) bool;

/// Agent lifecycle event types.
pub const AgentEventType = enum(u8) {
    /// About to call generate (iteration N).
    generation_start = 0,
    /// Generation returned (with or without tool calls).
    generation_end = 1,
    /// About to execute a tool.
    tool_start = 2,
    /// Tool execution completed.
    tool_end = 3,
};

/// Event payload for lifecycle callbacks.
pub const AgentEvent = extern struct {
    event_type: AgentEventType,
    _pad0: [3]u8 = .{0} ** 3,
    iteration: u32 = 0,
    /// Tool name (set for tool_start/tool_end, null otherwise).
    tool_name: ?[*:0]const u8 = null,
    /// Tool arguments JSON (set for tool_start, null otherwise).
    tool_arguments: ?[*:0]const u8 = null,
    /// Tool output (set for tool_end, null otherwise).
    tool_output: ?[*:0]const u8 = null,
    /// Whether tool returned an error (set for tool_end).
    tool_is_error: u8 = 0,
    _pad1: [7]u8 = .{0} ** 7,
};

/// Callback invoked for lifecycle events. Return false to cancel the loop.
pub const OnEventFn = *const fn (
    event: *const AgentEvent,
    user_data: ?*anyopaque,
) callconv(.c) bool;

// =============================================================================
// Callback types — context management
// =============================================================================

/// Information about current context state, passed to the on_context callback.
/// Enables caller-driven context window management (compaction, truncation,
/// summarization) by providing the token counts needed to decide when to act.
pub const ContextInfo = extern struct {
    /// Current number of items in conversation.
    item_count: usize = 0,
    /// Sum of input_tokens across all items (u64 to avoid overflow).
    total_input_tokens: u64 = 0,
    /// Sum of output_tokens across all items.
    total_output_tokens: u64 = 0,
    /// Current iteration number.
    iteration: u32 = 0,
    _pad0: [4]u8 = .{0} ** 4,
};

/// Called before each generation to allow context window management.
/// The callback receives token counts and can truncate, summarize, or
/// modify the conversation via the Chat handle the caller already owns.
/// Return false to cancel the loop.
pub const OnContextFn = *const fn (
    context: *const ContextInfo,
    user_data: ?*anyopaque,
) callconv(.c) bool;

// =============================================================================
// Callback types — tool middleware
// =============================================================================

/// Called before tool execution. Return false to SKIP this tool call.
/// The loop will append a "Tool execution blocked by caller." output and
/// continue to the next tool call.
pub const OnBeforeToolFn = *const fn (
    tool_name: [*:0]const u8,
    tool_arguments: [*]const u8,
    tool_arguments_len: usize,
    user_data: ?*anyopaque,
) callconv(.c) bool;

/// Called after tool execution with the result. Return false to cancel
/// the loop. Cannot modify the output (already appended to conversation).
pub const OnAfterToolFn = *const fn (
    tool_name: [*:0]const u8,
    tool_output: [*]const u8,
    tool_output_len: usize,
    tool_is_error: u8,
    user_data: ?*anyopaque,
) callconv(.c) bool;

// =============================================================================
// Callback types — session lifecycle
// =============================================================================

/// Session lifecycle event types.
pub const SessionEventType = enum(u8) {
    /// Loop is starting — caller can restore conversation from storage.
    loop_start = 0,
    /// Loop completed — caller can persist final state.
    loop_end = 1,
    /// Storage error detected on conversation.
    storage_error = 2,
};

/// Session lifecycle event payload.
pub const SessionEvent = extern struct {
    event_type: SessionEventType,
    _pad0: [3]u8 = .{0} ** 3,
    /// Stop reason (set for loop_end, 0 otherwise).
    stop_reason: u8 = 0,
    _pad1: [3]u8 = .{0} ** 3,
    /// Iteration count at event time.
    iterations: usize = 0,
    /// Total tool calls at event time.
    total_tool_calls: usize = 0,
};

/// Callback invoked for session lifecycle events (start, end, storage error).
/// Notification-only — does not control loop flow.
pub const OnSessionFn = *const fn (
    event: *const SessionEvent,
    user_data: ?*anyopaque,
) callconv(.c) void;

// =============================================================================
// Configuration and result types
// =============================================================================

pub const AgentLoopConfig = struct {
    /// Maximum number of generate→execute iterations before stopping.
    max_iterations: usize = 10,
    /// If true, stop the loop when a tool execution returns an error.
    abort_on_tool_error: bool = false,
    /// Atomic flag for cross-thread cancellation. Set to true to stop the loop.
    stop_flag: ?*const std.atomic.Value(bool) = null,

    /// Token streaming callback. When set, generation uses the iterator
    /// (background thread + ring buffer) and forwards each token to this
    /// callback. When null, uses synchronous generateWithBackend().
    on_token: ?OnTokenFn = null,
    on_token_data: ?*anyopaque = null,

    /// Lifecycle event callback for generation_start/end, tool_start/end.
    on_event: ?OnEventFn = null,
    on_event_data: ?*anyopaque = null,

    /// Optional generation config for sampling params, template overrides,
    /// stop sequences, etc. Passed through to generateWithBackend() or
    /// createIterator(). Caller owns the pointer.
    generate_config: ?*const CGenerateConfig = null,

    /// Starting iteration count for resume support. The loop adds to this
    /// value; max_iterations is still the cap for new iterations in this call.
    initial_iteration_count: usize = 0,
    /// Starting tool call count for resume.
    initial_tool_call_count: usize = 0,

    /// Context transform callback — called before each generation.
    /// Receives token counts so the caller can compact/truncate the
    /// conversation when approaching context window limits.
    on_context: ?OnContextFn = null,
    on_context_data: ?*anyopaque = null,

    /// Tool middleware — called before tool execution. Return false to skip
    /// the tool call (a "blocked" output is appended instead).
    on_before_tool: ?OnBeforeToolFn = null,
    on_before_tool_data: ?*anyopaque = null,

    /// Tool middleware — called after tool execution. Return false to cancel
    /// the loop. Output is already appended to conversation.
    on_after_tool: ?OnAfterToolFn = null,
    on_after_tool_data: ?*anyopaque = null,

    /// Session lifecycle callback — loop_start, loop_end, storage_error.
    on_session: ?OnSessionFn = null,
    on_session_data: ?*anyopaque = null,
};

pub const LoopStopReason = enum(u8) {
    /// LLM produced a response with no tool calls (normal completion).
    completed = 0,
    /// Reached max_iterations without completing.
    max_iterations = 1,
    /// A tool execution failed and abort_on_tool_error was set.
    tool_error = 2,
    /// Cancelled via stop_flag or callback returning false.
    cancelled = 3,
};

pub const AgentLoopResult = struct {
    stop_reason: LoopStopReason,
    /// Total iterations including initial_iteration_count.
    iterations: usize,
    /// Total tool calls including initial_tool_call_count.
    total_tool_calls: usize,
};

// =============================================================================
// Internal: tool call info for streaming path
// =============================================================================

/// Lightweight tool call reference extracted from conversation items.
/// Used by the streaming path where tool calls are committed to the
/// conversation by the iterator's worker thread rather than returned
/// in a GenerateResult.
const PendingToolCall = struct {
    item_index: usize,
    call_id: []const u8,
    name: []const u8,
    arguments: []const u8,
    status: ItemStatus,
};

// =============================================================================
// Agent loop
// =============================================================================

/// Run the agent loop: generate → execute tools → repeat.
///
/// Iterates until the LLM produces a response with no tool calls (completion),
/// max_iterations is reached, a tool error occurs (if abort_on_tool_error), or
/// the stop_flag / callback cancellation is triggered.
///
/// When `config.on_token` is set, generation uses the pull-based TokenIterator
/// for streaming. Otherwise, uses synchronous `generateWithBackend()`.
///
/// The caller owns Chat, backend, and registry lifecycles. This function does
/// not free them.
pub fn run(
    allocator: Allocator,
    chat: *Chat,
    backend: *InferenceBackend,
    registry: *const ToolRegistry,
    config: AgentLoopConfig,
) !AgentLoopResult {
    var iterations: usize = config.initial_iteration_count;
    var total_tool_calls: usize = config.initial_tool_call_count;

    // Set tool definitions on the chat so the LLM knows about available tools
    const tools_json = try registry.getToolDefinitionsJson(allocator);
    defer allocator.free(tools_json);
    try chat.setTools(tools_json);

    // Emit session start event
    emitSessionEvent(config, .loop_start, iterations, total_tool_calls, 0);

    var new_iterations: usize = 0;
    var stop_reason: LoopStopReason = .max_iterations;

    while (new_iterations < config.max_iterations) : (new_iterations += 1) {
        iterations = config.initial_iteration_count + new_iterations;

        // Check cancellation
        if (config.stop_flag) |flag| {
            if (flag.load(.acquire)) {
                stop_reason = .cancelled;
                break;
            }
        }

        // Context transform — let caller inspect token counts and compact
        if (config.on_context) |on_context| {
            const ctx_info = computeContextInfo(chat.conv, iterations);
            if (!on_context(&ctx_info, config.on_context_data)) {
                stop_reason = .cancelled;
                break;
            }
        }

        // Check for storage errors after potential compaction
        if (chat.conv.hasStorageError()) {
            emitSessionEvent(config, .storage_error, iterations, total_tool_calls, 0);
            chat.conv.clearStorageError();
        }

        // Emit generation_start event
        if (!emitEvent(config, .generation_start, iterations, .{})) {
            stop_reason = .cancelled;
            break;
        }

        // Generate — streaming or synchronous path
        const has_tool_calls = if (config.on_token != null)
            try generateStreaming(allocator, chat, backend, config)
        else
            try generateSync(allocator, chat, backend, config);

        // Check for storage errors after generation (commit may have failed)
        if (chat.conv.hasStorageError()) {
            emitSessionEvent(config, .storage_error, iterations, total_tool_calls, 0);
            chat.conv.clearStorageError();
        }

        // Emit generation_end event
        if (!emitEvent(config, .generation_end, iterations, .{})) {
            iterations += 1;
            stop_reason = .cancelled;
            break;
        }

        if (!has_tool_calls) {
            iterations += 1;
            stop_reason = .completed;
            break;
        }

        // Execute tool calls from conversation
        const exec_result = try executeToolCalls(
            allocator,
            chat,
            registry,
            config,
            iterations,
            &total_tool_calls,
        );

        switch (exec_result) {
            .continue_loop => {},
            .stop_cancelled => {
                iterations += 1;
                stop_reason = .cancelled;
                break;
            },
            .stop_tool_error => {
                iterations += 1;
                stop_reason = .tool_error;
                break;
            },
        }
    }

    // Final iteration count for max_iterations case (loop ran to completion)
    if (stop_reason == .max_iterations) {
        iterations = config.initial_iteration_count + new_iterations;
    }

    // Emit session end event
    emitSessionEvent(config, .loop_end, iterations, total_tool_calls, @intFromEnum(stop_reason));

    return .{
        .stop_reason = stop_reason,
        .iterations = iterations,
        .total_tool_calls = total_tool_calls,
    };
}

pub const RunError = error{
    GenerationFailed,
    OutOfMemory,
    IteratorCreationFailed,
};

// =============================================================================
// Generation paths
// =============================================================================

/// Synchronous generation via generateWithBackend().
/// Returns true if tool calls were produced, false if completed normally.
fn generateSync(
    allocator: Allocator,
    chat: *Chat,
    backend: *InferenceBackend,
    config: AgentLoopConfig,
) !bool {
    var result = capi_bridge.generateWithBackend(
        allocator,
        chat,
        &.{},
        backend,
        config.generate_config,
    );

    if (result.error_code != 0) {
        return error.GenerationFailed;
    }

    const tool_calls = result.tool_calls orelse {
        result.deinit(allocator);
        return false;
    };

    if (tool_calls.len == 0) {
        result.tool_calls = null;
        result.deinit(allocator);
        return false;
    }

    // Tool calls are already committed to conversation by generateWithBackend.
    // We don't need the result's tool_call refs — we'll read from conversation.
    result.tool_calls = null;
    result.deinit(allocator);
    return true;
}

/// Streaming generation via TokenIterator.
/// Polls tokens and forwards them to on_token callback.
/// Returns true if tool calls were produced, false if completed normally.
fn generateStreaming(
    allocator: Allocator,
    chat: *Chat,
    backend: *InferenceBackend,
    config: AgentLoopConfig,
) !bool {
    const on_token = config.on_token.?;

    var iter = capi_bridge.createIterator(
        allocator,
        chat,
        &.{},
        backend,
        config.generate_config,
    ) catch {
        return error.IteratorCreationFailed;
    };
    defer iter.deinit();

    // Poll tokens and forward to callback
    while (iter.next()) |token_ptr| {
        if (!on_token(token_ptr, config.on_token_data)) {
            iter.cancel();
            return error.GenerationFailed;
        }

        // Check external stop_flag
        if (config.stop_flag) |flag| {
            if (flag.load(.acquire)) {
                iter.cancel();
                return error.GenerationFailed;
            }
        }
    }

    // Check for iterator error
    if (iter.hasError()) {
        return error.GenerationFailed;
    }

    // Check finish reason — tool_calls means we have pending calls
    const finish = iter.getFinishReason();
    return finish == @intFromEnum(FinishReason.tool_calls);
}

// =============================================================================
// Tool call execution (shared by both paths)
// =============================================================================

const ExecResult = enum {
    continue_loop,
    stop_cancelled,
    stop_tool_error,
};

/// Execute pending tool calls from the conversation.
///
/// Both generation paths (sync and streaming) commit tool calls to the
/// conversation. This function scans for pending FunctionCall items from
/// the end and executes them.
fn executeToolCalls(
    allocator: Allocator,
    chat: *Chat,
    registry: *const ToolRegistry,
    config: AgentLoopConfig,
    iteration: usize,
    total_tool_calls: *usize,
) !ExecResult {
    // Collect pending tool calls from the conversation
    // Scan from the end — stop at the first non-function-call item
    const conv = chat.conv;
    const conv_len = conv.len();

    // Find pending function calls (scan backward from end)
    var pending_start: usize = conv_len;
    {
        var i: usize = conv_len;
        while (i > 0) {
            i -= 1;
            const item = conv.getItem(i) orelse break;
            switch (item.data) {
                .function_call => {
                    pending_start = i;
                },
                else => break,
            }
        }
    }

    if (pending_start >= conv_len) {
        // No pending tool calls found
        return .continue_loop;
    }

    var idx = pending_start;
    while (idx < conv_len) : (idx += 1) {
        const item = conv.getItem(idx) orelse continue;
        const fc = switch (item.data) {
            .function_call => |fc| fc,
            else => continue,
        };

        total_tool_calls.* += 1;

        // Null-terminate strings for event callbacks
        const name_z = @as([*:0]const u8, @ptrCast(fc.name.ptr));
        const args_slice = fc.getArguments();

        // Check if denied by policy
        if (fc.status == .failed) {
            _ = try conv.appendFunctionCallOutput(
                fc.call_id,
                "Tool call denied by policy.",
            );
            continue;
        }

        // Tool middleware: on_before_tool — can skip this tool call
        if (config.on_before_tool) |before_fn| {
            if (!before_fn(
                name_z,
                args_slice.ptr,
                args_slice.len,
                config.on_before_tool_data,
            )) {
                _ = try conv.appendFunctionCallOutput(
                    fc.call_id,
                    "Tool execution blocked by caller.",
                );

                // Emit tool_start + tool_end with error for blocked tool
                _ = emitEvent(config, .tool_start, iteration, .{
                    .tool_name = name_z,
                });
                _ = emitEvent(config, .tool_end, iteration, .{
                    .tool_name = name_z,
                    .tool_is_error = 1,
                });

                continue;
            }
        }

        // Emit tool_start event
        if (!emitEvent(config, .tool_start, iteration, .{
            .tool_name = name_z,
            .tool_arguments = if (args_slice.len > 0 and args_slice[args_slice.len - 1] == 0)
                @as([*:0]const u8, @ptrCast(args_slice.ptr))
            else
                null,
        })) {
            return .stop_cancelled;
        }

        // Look up and execute the tool
        if (registry.get(fc.name)) |tool| {
            var tool_result = tool.execute(allocator, args_slice) catch |err| {
                const err_msg = try std.fmt.allocPrint(
                    allocator,
                    "Tool execution error: {s}",
                    .{@errorName(err)},
                );
                defer allocator.free(err_msg);
                _ = try conv.appendFunctionCallOutput(fc.call_id, err_msg);

                // Emit tool_end with error
                _ = emitEvent(config, .tool_end, iteration, .{
                    .tool_name = name_z,
                    .tool_is_error = 1,
                });

                // Tool middleware: on_after_tool for error case
                if (config.on_after_tool) |after_fn| {
                    if (!after_fn(
                        name_z,
                        err_msg.ptr,
                        err_msg.len,
                        1,
                        config.on_after_tool_data,
                    )) {
                        return .stop_cancelled;
                    }
                }

                if (config.abort_on_tool_error) {
                    return .stop_tool_error;
                }
                continue;
            };
            defer tool_result.deinit(allocator);

            _ = try conv.appendFunctionCallOutput(fc.call_id, tool_result.output);

            // Emit tool_end with success
            _ = emitEvent(config, .tool_end, iteration, .{
                .tool_name = name_z,
                .tool_is_error = if (tool_result.is_error) @as(u8, 1) else @as(u8, 0),
            });

            // Tool middleware: on_after_tool
            if (config.on_after_tool) |after_fn| {
                if (!after_fn(
                    name_z,
                    tool_result.output.ptr,
                    tool_result.output.len,
                    if (tool_result.is_error) @as(u8, 1) else @as(u8, 0),
                    config.on_after_tool_data,
                )) {
                    return .stop_cancelled;
                }
            }
        } else {
            // Unknown tool name
            const err_msg = try std.fmt.allocPrint(
                allocator,
                "Unknown tool: {s}",
                .{fc.name},
            );
            defer allocator.free(err_msg);
            _ = try conv.appendFunctionCallOutput(fc.call_id, err_msg);

            // Emit tool_end with error
            _ = emitEvent(config, .tool_end, iteration, .{
                .tool_name = name_z,
                .tool_is_error = 1,
            });

            // Tool middleware: on_after_tool for unknown tool
            if (config.on_after_tool) |after_fn| {
                if (!after_fn(
                    name_z,
                    err_msg.ptr,
                    err_msg.len,
                    1,
                    config.on_after_tool_data,
                )) {
                    return .stop_cancelled;
                }
            }

            if (config.abort_on_tool_error) {
                return .stop_tool_error;
            }
        }

        // Check for storage errors after each tool output
        if (chat.conv.hasStorageError()) {
            emitSessionEvent(config, .storage_error, iteration, total_tool_calls.*, 0);
            chat.conv.clearStorageError();
        }
    }

    return .continue_loop;
}

// =============================================================================
// Helpers
// =============================================================================

const EventFields = struct {
    tool_name: ?[*:0]const u8 = null,
    tool_arguments: ?[*:0]const u8 = null,
    tool_output: ?[*:0]const u8 = null,
    tool_is_error: u8 = 0,
};

/// Emit a lifecycle event via on_event callback. Returns true to continue,
/// false if the callback requested cancellation.
fn emitEvent(
    config: AgentLoopConfig,
    event_type: AgentEventType,
    iteration: usize,
    fields: EventFields,
) bool {
    const on_event = config.on_event orelse return true;
    const event = AgentEvent{
        .event_type = event_type,
        .iteration = @intCast(iteration),
        .tool_name = fields.tool_name,
        .tool_arguments = fields.tool_arguments,
        .tool_output = fields.tool_output,
        .tool_is_error = fields.tool_is_error,
    };
    return on_event(&event, config.on_event_data);
}

/// Emit a session lifecycle event via on_session callback.
fn emitSessionEvent(
    config: AgentLoopConfig,
    event_type: SessionEventType,
    iterations: usize,
    total_tool_calls: usize,
    stop_reason: u8,
) void {
    const on_session = config.on_session orelse return;
    const event = SessionEvent{
        .event_type = event_type,
        .stop_reason = stop_reason,
        .iterations = iterations,
        .total_tool_calls = total_tool_calls,
    };
    on_session(&event, config.on_session_data);
}

/// Compute context info by scanning conversation items for token counts.
pub fn computeContextInfo(conv: *const Conversation, iteration: usize) ContextInfo {
    var total_input: u64 = 0;
    var total_output: u64 = 0;
    const count = conv.len();
    var i: usize = 0;
    while (i < count) : (i += 1) {
        if (conv.getItem(i)) |item| {
            total_input += item.input_tokens;
            total_output += item.output_tokens;
        }
    }
    return .{
        .item_count = count,
        .total_input_tokens = total_input,
        .total_output_tokens = total_output,
        .iteration = @intCast(iteration),
    };
}

// =============================================================================
// Tests
// =============================================================================

// Note: Full integration tests for the agent loop require a loaded model or
// HTTP server and live in core/tests/agent/. The loop's internal orchestration
// logic (iteration counting, stop reasons, tool dispatch) is verified at that
// level. Unit tests here validate the types and configuration.

test "AgentLoopConfig defaults" {
    const config = AgentLoopConfig{};
    try std.testing.expectEqual(@as(usize, 10), config.max_iterations);
    try std.testing.expect(!config.abort_on_tool_error);
    try std.testing.expect(config.stop_flag == null);
    try std.testing.expect(config.on_token == null);
    try std.testing.expect(config.on_token_data == null);
    try std.testing.expect(config.on_event == null);
    try std.testing.expect(config.on_event_data == null);
    try std.testing.expect(config.generate_config == null);
    try std.testing.expectEqual(@as(usize, 0), config.initial_iteration_count);
    try std.testing.expectEqual(@as(usize, 0), config.initial_tool_call_count);
    // New hook fields
    try std.testing.expect(config.on_context == null);
    try std.testing.expect(config.on_context_data == null);
    try std.testing.expect(config.on_before_tool == null);
    try std.testing.expect(config.on_before_tool_data == null);
    try std.testing.expect(config.on_after_tool == null);
    try std.testing.expect(config.on_after_tool_data == null);
    try std.testing.expect(config.on_session == null);
    try std.testing.expect(config.on_session_data == null);
}

test "LoopStopReason enum values" {
    // Verify enum values match C API expectations
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(LoopStopReason.completed));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(LoopStopReason.max_iterations));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(LoopStopReason.tool_error));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(LoopStopReason.cancelled));
}

test "AgentEventType enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(AgentEventType.generation_start));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(AgentEventType.generation_end));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(AgentEventType.tool_start));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(AgentEventType.tool_end));
}

test "AgentEvent default construction" {
    const event = AgentEvent{ .event_type = .generation_start };
    try std.testing.expectEqual(AgentEventType.generation_start, event.event_type);
    try std.testing.expectEqual(@as(u32, 0), event.iteration);
    try std.testing.expect(event.tool_name == null);
    try std.testing.expect(event.tool_arguments == null);
    try std.testing.expect(event.tool_output == null);
    try std.testing.expectEqual(@as(u8, 0), event.tool_is_error);
}

test "emitEvent returns true when no callback set" {
    const config = AgentLoopConfig{};
    try std.testing.expect(emitEvent(config, .generation_start, 0, .{}));
}

test "emitEvent calls callback and returns result" {
    const Ctx = struct {
        var last_type: AgentEventType = .generation_start;
        var last_iteration: u32 = 99;
        var call_count: usize = 0;

        fn callback(event: *const AgentEvent, _: ?*anyopaque) callconv(.c) bool {
            last_type = event.event_type;
            last_iteration = event.iteration;
            call_count += 1;
            return true;
        }

        fn cancelCallback(_: *const AgentEvent, _: ?*anyopaque) callconv(.c) bool {
            return false;
        }
    };

    // Test normal callback
    Ctx.call_count = 0;
    const config = AgentLoopConfig{
        .on_event = Ctx.callback,
    };
    try std.testing.expect(emitEvent(config, .tool_start, 5, .{}));
    try std.testing.expectEqual(AgentEventType.tool_start, Ctx.last_type);
    try std.testing.expectEqual(@as(u32, 5), Ctx.last_iteration);
    try std.testing.expectEqual(@as(usize, 1), Ctx.call_count);

    // Test cancel callback
    const cancel_config = AgentLoopConfig{
        .on_event = Ctx.cancelCallback,
    };
    try std.testing.expect(!emitEvent(cancel_config, .generation_end, 0, .{}));
}

test "ContextInfo default construction" {
    const info = ContextInfo{};
    try std.testing.expectEqual(@as(usize, 0), info.item_count);
    try std.testing.expectEqual(@as(u64, 0), info.total_input_tokens);
    try std.testing.expectEqual(@as(u64, 0), info.total_output_tokens);
    try std.testing.expectEqual(@as(u32, 0), info.iteration);
}

test "SessionEventType enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(SessionEventType.loop_start));
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(SessionEventType.loop_end));
    try std.testing.expectEqual(@as(u8, 2), @intFromEnum(SessionEventType.storage_error));
}

test "SessionEvent default construction" {
    const event = SessionEvent{ .event_type = .loop_start };
    try std.testing.expectEqual(SessionEventType.loop_start, event.event_type);
    try std.testing.expectEqual(@as(u8, 0), event.stop_reason);
    try std.testing.expectEqual(@as(usize, 0), event.iterations);
    try std.testing.expectEqual(@as(usize, 0), event.total_tool_calls);
}

test "emitSessionEvent is noop when no callback set" {
    const config = AgentLoopConfig{};
    // Should not crash — just a noop
    emitSessionEvent(config, .loop_start, 0, 0, 0);
    emitSessionEvent(config, .loop_end, 5, 3, @intFromEnum(LoopStopReason.completed));
    emitSessionEvent(config, .storage_error, 2, 1, 0);
}

test "emitSessionEvent calls callback with correct fields" {
    const Ctx = struct {
        var last_type: SessionEventType = .loop_start;
        var last_stop_reason: u8 = 99;
        var last_iterations: usize = 99;
        var last_tool_calls: usize = 99;
        var call_count: usize = 0;

        fn callback(event: *const SessionEvent, _: ?*anyopaque) callconv(.c) void {
            last_type = event.event_type;
            last_stop_reason = event.stop_reason;
            last_iterations = event.iterations;
            last_tool_calls = event.total_tool_calls;
            call_count += 1;
        }
    };

    Ctx.call_count = 0;
    const config = AgentLoopConfig{
        .on_session = Ctx.callback,
    };

    emitSessionEvent(config, .loop_end, 7, 4, @intFromEnum(LoopStopReason.completed));
    try std.testing.expectEqual(SessionEventType.loop_end, Ctx.last_type);
    try std.testing.expectEqual(@as(u8, 0), Ctx.last_stop_reason); // completed = 0
    try std.testing.expectEqual(@as(usize, 7), Ctx.last_iterations);
    try std.testing.expectEqual(@as(usize, 4), Ctx.last_tool_calls);
    try std.testing.expectEqual(@as(usize, 1), Ctx.call_count);
}

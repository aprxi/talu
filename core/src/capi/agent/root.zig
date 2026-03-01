//! C API for the agent framework.
//!
//! Three levels:
//!   - Tool registry: create/free/register tools.
//!   - Agent loop (low-level): stateless generate-execute-repeat via talu_agent_run.
//!   - Agent (high-level): stateful agent with compaction, retry, inbox via
//!     talu_agent_create/prompt/heartbeat/abort.
//!   - Message bus: inter-agent communication via talu_agent_bus_*.
//!
//! See core/src/agent/ for the implementation.

const std = @import("std");
const agent_mod = @import("../../agent/root.zig");
const agent_capi_bridge = @import("../../agent/capi_bridge.zig");
const chat_mod = @import("../../responses/chat.zig");
const router_mod = @import("../../router/root.zig");
const db_mod = @import("../../db/root.zig");
const capi_error = @import("../error.zig");
const error_codes = @import("../error_codes.zig");
const fs_api = @import("fs.zig");
const shell_api = @import("shell.zig");

const ToolRegistry = agent_mod.ToolRegistry;
const Agent = agent_mod.Agent;
const AgentLoopResult = agent_mod.AgentLoopResult;
const MessageBus = agent_mod.MessageBus;
const BusError = agent_mod.BusError;
const OnTokenFn = agent_mod.OnTokenFn;
const OnEventFn = agent_mod.OnEventFn;
const OnContextFn = agent_mod.OnContextFn;
const OnBeforeToolFn = agent_mod.OnBeforeToolFn;
const OnAfterToolFn = agent_mod.OnAfterToolFn;
const OnSessionFn = agent_mod.OnSessionFn;
const OnContextInjectFn = agent_mod.OnContextInjectFn;
const OnEmbedFn = agent_mod.OnEmbedFn;
const OnResolveDocFn = agent_mod.OnResolveDocFn;
const RagConfig = agent_mod.RagConfig;
const OnMessageNotifyFn = agent_mod.OnMessageNotifyFn;
const VectorAdapter = db_mod.VectorAdapter;
const VectorStoreHandle = @import("../db/vector.zig").VectorStoreHandle;
const Chat = chat_mod.Chat;
const InferenceBackend = router_mod.InferenceBackend;
const CGenerateConfig = router_mod.capi_bridge.CGenerateConfig;

const allocator = std.heap.c_allocator;

// =============================================================================
// Opaque handles
// =============================================================================

/// Opaque handle for a ToolRegistry object.
pub const TaluToolRegistry = opaque {};

/// Opaque handle for a stateful Agent.
pub const TaluAgent = opaque {};

/// Opaque handle for a MessageBus.
pub const TaluAgentBus = opaque {};

/// Opaque handle for workspace-scoped filesystem operations.
pub const TaluFs = fs_api.TaluFs;
pub const TaluFsStat = fs_api.TaluFsStat;

pub export fn talu_fs_create(
    workspace_dir: ?[*:0]const u8,
    out_handle: ?*?*TaluFs,
) callconv(.c) i32 {
    return fs_api.talu_fs_create(workspace_dir, out_handle);
}

pub export fn talu_fs_free(handle: ?*TaluFs) callconv(.c) void {
    fs_api.talu_fs_free(handle);
}

pub export fn talu_fs_read(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    max_bytes: usize,
    out_content: ?*?[*]const u8,
    out_content_len: ?*usize,
    out_size: ?*u64,
    out_truncated: ?*bool,
) callconv(.c) i32 {
    return fs_api.talu_fs_read(handle, path, max_bytes, out_content, out_content_len, out_size, out_truncated);
}

pub export fn talu_fs_write(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    content: ?[*]const u8,
    content_len: usize,
    mkdir: bool,
    out_bytes_written: ?*usize,
) callconv(.c) i32 {
    return fs_api.talu_fs_write(handle, path, content, content_len, mkdir, out_bytes_written);
}

pub export fn talu_fs_edit(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    old_text: ?[*]const u8,
    old_len: usize,
    new_text: ?[*]const u8,
    new_len: usize,
    replace_all: bool,
    out_replacements: ?*usize,
) callconv(.c) i32 {
    return fs_api.talu_fs_edit(handle, path, old_text, old_len, new_text, new_len, replace_all, out_replacements);
}

pub export fn talu_fs_stat(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    out_stat: ?*TaluFsStat,
) callconv(.c) i32 {
    return fs_api.talu_fs_stat(handle, path, out_stat);
}

pub export fn talu_fs_list(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    glob: ?[*:0]const u8,
    recursive: bool,
    limit: usize,
    out_json: ?*?[*]const u8,
    out_json_len: ?*usize,
) callconv(.c) i32 {
    return fs_api.talu_fs_list(handle, path, glob, recursive, limit, out_json, out_json_len);
}

pub export fn talu_fs_remove(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    recursive: bool,
) callconv(.c) i32 {
    return fs_api.talu_fs_remove(handle, path, recursive);
}

pub export fn talu_fs_mkdir(
    handle: ?*TaluFs,
    path: ?[*:0]const u8,
    recursive: bool,
) callconv(.c) i32 {
    return fs_api.talu_fs_mkdir(handle, path, recursive);
}

pub export fn talu_fs_rename(
    handle: ?*TaluFs,
    from: ?[*:0]const u8,
    to: ?[*:0]const u8,
) callconv(.c) i32 {
    return fs_api.talu_fs_rename(handle, from, to);
}

pub export fn talu_fs_free_string(ptr: ?[*]const u8, len: usize) callconv(.c) void {
    fs_api.talu_fs_free_string(ptr, len);
}

// =============================================================================
// Shell execution
// =============================================================================

pub export fn talu_shell_exec(
    command: ?[*:0]const u8,
    out_stdout: ?*?[*]const u8,
    out_stdout_len: ?*usize,
    out_stderr: ?*?[*]const u8,
    out_stderr_len: ?*usize,
    out_exit_code: ?*i32,
) callconv(.c) i32 {
    return shell_api.talu_shell_exec(command, out_stdout, out_stdout_len, out_stderr, out_stderr_len, out_exit_code);
}

pub export fn talu_shell_check_command(
    command: ?[*:0]const u8,
    out_allowed: ?*bool,
    out_reason: ?*?[*]const u8,
    out_reason_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_check_command(command, out_allowed, out_reason, out_reason_len);
}

pub export fn talu_shell_default_policy_json(
    out_json: ?*?[*]const u8,
    out_json_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_default_policy_json(out_json, out_json_len);
}

pub export fn talu_shell_normalize_command(
    command: ?[*:0]const u8,
    out_normalized: ?*?[*]const u8,
    out_normalized_len: ?*usize,
) callconv(.c) i32 {
    return shell_api.talu_shell_normalize_command(command, out_normalized, out_normalized_len);
}

pub export fn talu_shell_free_string(ptr: ?[*]const u8, len: usize) callconv(.c) void {
    shell_api.talu_shell_free_string(ptr, len);
}

// =============================================================================
// C callback tool wrapper
// =============================================================================

/// C function pointer type for tool execution.
/// Returns 0 on success, non-zero on error.
pub const CToolExecuteFn = agent_capi_bridge.CToolExecuteFn;

// =============================================================================
// Registry lifecycle
// =============================================================================

/// Create a new tool registry. Caller owns the returned handle.
///
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_registry_create(
    out_registry: ?*?*TaluToolRegistry,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_registry orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_registry is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const registry = allocator.create(ToolRegistry) catch |err| {
        capi_error.setError(err, "failed to allocate tool registry", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    registry.* = ToolRegistry.init(allocator);
    out.* = @ptrCast(registry);
    return 0;
}

/// Free a tool registry handle. Calls deinit on all registered tools.
pub export fn talu_agent_registry_free(
    handle: ?*TaluToolRegistry,
) callconv(.c) void {
    const registry: *ToolRegistry = @ptrCast(@alignCast(handle orelse return));
    registry.deinit();
    allocator.destroy(registry);
}

// =============================================================================
// Tool registration
// =============================================================================

/// Register a tool by providing a name, description, parameters schema, and
/// a C callback for execution.
///
/// The execute_fn callback receives arguments JSON and must set out_ptr/out_len
/// to the output string (allocated with talu_alloc_string or malloc). The agent
/// framework will free the output after consuming it.
///
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_registry_add(
    handle: ?*TaluToolRegistry,
    name_ptr: ?[*:0]const u8,
    description_ptr: ?[*:0]const u8,
    parameters_schema_ptr: ?[*:0]const u8,
    execute_fn: ?CToolExecuteFn,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();
    const registry: *ToolRegistry = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "registry handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const callback_args = agent_capi_bridge.parseCallbackToolArgs(name_ptr, description_ptr, parameters_schema_ptr, execute_fn) catch |err| {
        switch (err) {
            error.NullName => capi_error.setErrorWithCode(.invalid_argument, "name is null", .{}),
            error.NullDescription => capi_error.setErrorWithCode(.invalid_argument, "description is null", .{}),
            error.NullSchema => capi_error.setErrorWithCode(.invalid_argument, "parameters_schema is null", .{}),
            error.NullExecuteFn => capi_error.setErrorWithCode(.invalid_argument, "execute_fn is null", .{}),
        }
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    agent_capi_bridge.registerCallbackTool(allocator, registry, callback_args.name, callback_args.description, callback_args.schema, callback_args.execute_fn, user_data) catch |err| {
        capi_error.setError(err, "failed to register tool", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Get the number of registered tools.
pub export fn talu_agent_registry_count(
    handle: ?*const TaluToolRegistry,
) callconv(.c) usize {
    const registry: *const ToolRegistry = @ptrCast(@alignCast(handle orelse return 0));
    return registry.count();
}

// =============================================================================
// Agent loop
// =============================================================================

/// C-compatible agent loop configuration.
///
/// All new fields default to null/zero when the struct is zeroed, preserving
/// backward compatibility: zeroed config = synchronous generation, no callbacks.
pub const CAgentLoopConfig = extern struct {
    max_iterations: usize = 10,
    /// Maximum tool calls before the loop stops. 0 = unlimited.
    max_tool_calls: usize = 0,
    abort_on_tool_error: u8 = 0,
    _pad0: [7]u8 = .{0} ** 7,
    stop_flag: ?*const std.atomic.Value(bool) = null,

    /// Token streaming callback (null = synchronous generation).
    /// When set, generation uses the TokenIterator and forwards each token.
    on_token: ?OnTokenFn = null,
    on_token_data: ?*anyopaque = null,

    /// Lifecycle event callback for generation/tool start/end.
    on_event: ?OnEventFn = null,
    on_event_data: ?*anyopaque = null,

    /// Generation config (sampling params, stop sequences, etc.).
    /// Passed through to generateWithBackend() or createIterator().
    generate_config: ?*const CGenerateConfig = null,

    /// Starting iteration count for resume support.
    initial_iteration_count: usize = 0,
    /// Starting tool call count for resume.
    initial_tool_call_count: usize = 0,

    /// Context transform callback (compaction hook).
    /// Called before each generation with token counts.
    on_context: ?OnContextFn = null,
    on_context_data: ?*anyopaque = null,

    /// Tool middleware — called before tool execution.
    /// Return false to skip (block) the tool call.
    on_before_tool: ?OnBeforeToolFn = null,
    on_before_tool_data: ?*anyopaque = null,

    /// Tool middleware — called after tool execution.
    /// Return false to cancel the loop.
    on_after_tool: ?OnAfterToolFn = null,
    on_after_tool_data: ?*anyopaque = null,

    /// Session lifecycle callback (loop_start, loop_end, storage_error).
    on_session: ?OnSessionFn = null,
    on_session_data: ?*anyopaque = null,

    /// Maximum bytes for tool output. 0 = unlimited.
    max_tool_output_bytes: usize = 0,
};

/// C-compatible agent loop result.
pub const CAgentLoopResult = extern struct {
    /// 0=completed, 1=max_iterations, 2=tool_error, 3=cancelled
    stop_reason: u8 = 0,
    _padding: [7]u8 = .{0} ** 7,
    iterations: usize = 0,
    total_tool_calls: usize = 0,
};

/// Run the agent loop: generate → execute tools → repeat.
///
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_run(
    chat_handle: ?*@import("../responses.zig").ChatHandle,
    backend_handle: ?*@import("../router.zig").TaluInferenceBackend,
    registry_handle: ?*TaluToolRegistry,
    config_ptr: ?*const CAgentLoopConfig,
    out_result: ?*CAgentLoopResult,
) callconv(.c) i32 {
    capi_error.clearError();

    const chat: *Chat = @ptrCast(@alignCast(chat_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "chat handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const backend: *InferenceBackend = @ptrCast(@alignCast(backend_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const registry: *const ToolRegistry = @ptrCast(@alignCast(registry_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "registry handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = agent_capi_bridge.runLoopWithConfig(
        allocator,
        chat,
        backend,
        registry,
        config_ptr,
    ) catch |err| {
        capi_error.setError(err, "agent loop failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    writeLoopResult(out_result, result);
    return 0;
}

// =============================================================================
// Tests
// =============================================================================

test "talu_agent_registry_create and free" {
    var out: ?*TaluToolRegistry = null;
    const rc = talu_agent_registry_create(&out);
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(out != null);
    talu_agent_registry_free(out);
}

test "talu_agent_registry_create null out returns error" {
    const rc = talu_agent_registry_create(null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_registry_free null is safe" {
    talu_agent_registry_free(null);
}

test "talu_agent_registry_add null handle returns error" {
    const rc = talu_agent_registry_add(null, "test", "desc", "{}", null, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_registry_count returns 0 for empty registry" {
    var out: ?*TaluToolRegistry = null;
    _ = talu_agent_registry_create(&out);
    defer talu_agent_registry_free(out);

    try std.testing.expectEqual(@as(usize, 0), talu_agent_registry_count(@ptrCast(out)));
}

test "talu_agent_registry_count null returns 0" {
    try std.testing.expectEqual(@as(usize, 0), talu_agent_registry_count(null));
}

test "CAgentLoopConfig zeroed is valid" {
    const config = std.mem.zeroes(CAgentLoopConfig);
    try std.testing.expectEqual(@as(usize, 0), config.max_iterations);
    try std.testing.expectEqual(@as(usize, 0), config.max_tool_calls);
    try std.testing.expectEqual(@as(u8, 0), config.abort_on_tool_error);
    try std.testing.expect(config.stop_flag == null);
    try std.testing.expect(config.on_token == null);
    try std.testing.expect(config.on_token_data == null);
    try std.testing.expect(config.on_event == null);
    try std.testing.expect(config.on_event_data == null);
    try std.testing.expect(config.generate_config == null);
    try std.testing.expectEqual(@as(usize, 0), config.initial_iteration_count);
    try std.testing.expectEqual(@as(usize, 0), config.initial_tool_call_count);
    try std.testing.expect(config.on_context == null);
    try std.testing.expect(config.on_context_data == null);
    try std.testing.expect(config.on_before_tool == null);
    try std.testing.expect(config.on_before_tool_data == null);
    try std.testing.expect(config.on_after_tool == null);
    try std.testing.expect(config.on_after_tool_data == null);
    try std.testing.expect(config.on_session == null);
    try std.testing.expect(config.on_session_data == null);
    try std.testing.expectEqual(@as(usize, 0), config.max_tool_output_bytes);
}

test "CAgentLoopResult zeroed is valid" {
    const result = std.mem.zeroes(CAgentLoopResult);
    try std.testing.expectEqual(@as(u8, 0), result.stop_reason);
    try std.testing.expectEqual(@as(usize, 0), result.iterations);
    try std.testing.expectEqual(@as(usize, 0), result.total_tool_calls);
}

// =============================================================================
// Stateful Agent — C API
// =============================================================================

/// C-compatible configuration for creating a stateful Agent.
///
/// Zeroed config is valid: 0 context_limit = unlimited, 0 max_retries = no retry,
/// 10 max_iterations, null agent_id = auto-generate.
pub const CAgentCreateConfig = extern struct {
    /// Maximum context size in tokens. 0 = unlimited.
    context_limit: u64 = 0,
    /// Compaction threshold as percentage (0-100). Default 80.
    compaction_threshold_pct: u32 = 80,
    /// Number of retries on generation failure. 0 = no retry.
    max_retries: u32 = 0,
    /// Base delay for exponential backoff in milliseconds.
    retry_base_delay_ms: u32 = 1000,
    _pad0: [4]u8 = .{0} ** 4,
    /// Maximum tool-call iterations per turn.
    max_iterations_per_turn: usize = 10,
    /// Maximum tool calls per turn. 0 = unlimited.
    max_tool_calls_per_turn: usize = 0,
    /// Stop loop when a tool execution returns an error.
    abort_on_tool_error: u8 = 0,
    _pad1: [7]u8 = .{0} ** 7,
    /// Directory for agent state files. Null = no state dir.
    state_dir: ?[*:0]const u8 = null,
    /// Agent identifier. Null = auto-generate.
    agent_id: ?[*:0]const u8 = null,

    // Observation callbacks
    on_token: ?OnTokenFn = null,
    on_token_data: ?*anyopaque = null,
    on_event: ?OnEventFn = null,
    on_event_data: ?*anyopaque = null,
    on_session: ?OnSessionFn = null,
    on_session_data: ?*anyopaque = null,

    // Context injection callback (RAG hook)
    on_context_inject: ?OnContextInjectFn = null,
    on_context_inject_data: ?*anyopaque = null,

    // Tool middleware callbacks
    on_before_tool: ?OnBeforeToolFn = null,
    on_before_tool_data: ?*anyopaque = null,
    on_after_tool: ?OnAfterToolFn = null,
    on_after_tool_data: ?*anyopaque = null,

    /// Maximum bytes for tool output. 0 = unlimited.
    max_tool_output_bytes: usize = 0,

    /// Built-in tool auto-registration. Null workspace disables built-ins.
    builtin_workspace_dir: ?[*:0]const u8 = null,
    /// Max bytes read by built-in file tools. 0 uses core default.
    builtin_file_max_read_bytes: usize = 0,
    /// Max bytes returned by built-in http tool. 0 uses core default.
    builtin_http_max_response_bytes: usize = 0,

    /// Memory integration over db blobs/documents. Null db path disables memory.
    memory_db_path: ?[*:0]const u8 = null,
    memory_namespace: ?[*:0]const u8 = null,
    memory_owner_id: ?[*:0]const u8 = null,
    memory_recall_limit: usize = 0,
    memory_append_on_run: u8 = 1,
    _pad2: [7]u8 = .{0} ** 7,
};

/// Create a stateful agent. Caller owns the returned handle.
///
/// The backend must outlive the agent (caller-managed lifecycle).
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_create(
    backend_handle: ?*@import("../router.zig").TaluInferenceBackend,
    config_ptr: ?*const CAgentCreateConfig,
    out_agent: ?*?*TaluAgent,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_agent orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_agent is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const backend: *InferenceBackend = @ptrCast(@alignCast(backend_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "backend handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const agent_ptr = agent_capi_bridge.createAgent(allocator, backend, config_ptr) catch |err| {
        capi_error.setError(err, "failed to create agent", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    out.* = @ptrCast(agent_ptr);
    return 0;
}

/// Free a stateful agent and all its owned resources.
pub export fn talu_agent_free(
    handle: ?*TaluAgent,
) callconv(.c) void {
    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse return));
    agent_ptr.deinit();
}

/// Set the system prompt on the agent's chat.
pub export fn talu_agent_set_system(
    handle: ?*TaluAgent,
    system_prompt: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const prompt_z = system_prompt orelse {
        capi_error.setErrorWithCode(.invalid_argument, "system_prompt is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    agent_ptr.setSystem(std.mem.span(prompt_z)) catch |err| {
        capi_error.setError(err, "failed to set system prompt", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Register a tool on the agent via C callback.
pub export fn talu_agent_register_tool(
    handle: ?*TaluAgent,
    name_ptr: ?[*:0]const u8,
    description_ptr: ?[*:0]const u8,
    parameters_schema_ptr: ?[*:0]const u8,
    execute_fn: ?CToolExecuteFn,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();
    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const callback_args = agent_capi_bridge.parseCallbackToolArgs(
        name_ptr,
        description_ptr,
        parameters_schema_ptr,
        execute_fn,
    ) catch |err| {
        switch (err) {
            error.NullName => capi_error.setErrorWithCode(.invalid_argument, "name is null", .{}),
            error.NullDescription => capi_error.setErrorWithCode(.invalid_argument, "description is null", .{}),
            error.NullSchema => capi_error.setErrorWithCode(.invalid_argument, "parameters_schema is null", .{}),
            error.NullExecuteFn => capi_error.setErrorWithCode(.invalid_argument, "execute_fn is null", .{}),
        }
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    agent_capi_bridge.registerCallbackToolOnAgent(
        allocator,
        agent_ptr,
        callback_args.name,
        callback_args.description,
        callback_args.schema,
        callback_args.execute_fn,
        user_data,
    ) catch |err| {
        capi_error.setError(err, "failed to register tool", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    return 0;
}

/// Connect the agent to a message bus for inter-agent communication.
pub export fn talu_agent_set_bus(
    handle: ?*TaluAgent,
    bus_handle: ?*TaluAgentBus,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(bus_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    agent_ptr.setBus(bus_ptr);
    return 0;
}

/// Send a user message and run the agent loop.
pub export fn talu_agent_prompt(
    handle: ?*TaluAgent,
    message: ?[*:0]const u8,
    out_result: ?*CAgentLoopResult,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const msg_z = message orelse {
        capi_error.setErrorWithCode(.invalid_argument, "message is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const result = agent_ptr.prompt(std.mem.span(msg_z)) catch |err| {
        capi_error.setError(err, "agent prompt failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    writeLoopResult(out_result, result);
    return 0;
}

/// Resume the agent loop without a new user message.
pub export fn talu_agent_continue(
    handle: ?*TaluAgent,
    out_result: ?*CAgentLoopResult,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = agent_ptr.continueLoop() catch |err| {
        capi_error.setError(err, "agent continue failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    writeLoopResult(out_result, result);
    return 0;
}

/// Autonomous heartbeat: drain inbox and act on pending work.
pub export fn talu_agent_heartbeat(
    handle: ?*TaluAgent,
    out_result: ?*CAgentLoopResult,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const result = agent_ptr.heartbeat() catch |err| {
        capi_error.setError(err, "agent heartbeat failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    writeLoopResult(out_result, result);
    return 0;
}

/// Request cancellation from another thread.
pub export fn talu_agent_abort(
    handle: ?*TaluAgent,
) callconv(.c) void {
    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse return));
    agent_ptr.abort();
}

/// Get the agent's underlying Chat handle (for use with other chat C APIs).
pub export fn talu_agent_get_chat(
    handle: ?*const TaluAgent,
) callconv(.c) ?*const @import("../responses.zig").ChatHandle {
    const agent_ptr: *const Agent = @ptrCast(@alignCast(handle orelse return null));
    return @ptrCast(agent_ptr.getChat());
}

/// Get the agent's identifier string (null-terminated, borrowed from agent).
pub export fn talu_agent_get_id(
    handle: ?*const TaluAgent,
    out_ptr: ?*?[*]const u8,
    out_len: ?*usize,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *const Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    if (out_ptr) |p| p.* = agent_ptr.agent_id.ptr;
    if (out_len) |l| l.* = agent_ptr.agent_id.len;
    return 0;
}

// =============================================================================
// Agent goals — C API
// =============================================================================

/// Add a persistent goal to the agent.
///
/// Goals are injected into the system prompt before each generation, ensuring
/// the LLM always knows its objectives even after context compaction.
/// Duplicate goals (exact string match) are silently ignored.
pub export fn talu_agent_add_goal(
    handle: ?*TaluAgent,
    goal: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const goal_z = goal orelse {
        capi_error.setErrorWithCode(.invalid_argument, "goal is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    agent_ptr.addGoal(std.mem.span(goal_z)) catch |err| {
        capi_error.setError(err, "failed to add goal", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Remove a specific goal by exact string match.
///
/// Returns 0 if the goal was found and removed, non-zero if not found.
pub export fn talu_agent_remove_goal(
    handle: ?*TaluAgent,
    goal: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const goal_z = goal orelse {
        capi_error.setErrorWithCode(.invalid_argument, "goal is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    if (agent_ptr.removeGoal(std.mem.span(goal_z))) {
        return 0;
    }
    return 1;
}

/// Remove all goals from the agent.
pub export fn talu_agent_clear_goals(
    handle: ?*TaluAgent,
) callconv(.c) void {
    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse return));
    agent_ptr.clearGoals();
}

/// Get the number of active goals.
pub export fn talu_agent_goal_count(
    handle: ?*const TaluAgent,
) callconv(.c) usize {
    const agent_ptr: *const Agent = @ptrCast(@alignCast(handle orelse return 0));
    return agent_ptr.getGoals().len;
}

// =============================================================================
// Agent context injection — C API
// =============================================================================

/// Set the context injection callback on an existing agent.
///
/// The callback is called before each generation with the agent's Chat as
/// an opaque handle. The callback can inspect the full conversation via
/// C API functions (talu_chat_get_messages, etc.) to construct a RAG query.
/// It can return additional context to inject as a hidden developer message
/// (visible to LLM, hidden from UI).
pub export fn talu_agent_set_context_inject(
    handle: ?*TaluAgent,
    inject_fn: ?OnContextInjectFn,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    agent_ptr.on_context_inject = inject_fn;
    agent_ptr.on_context_inject_data = user_data;
    return 0;
}

// =============================================================================
// Agent tool middleware — C API setters
// =============================================================================

/// Set the before-tool callback on an existing agent.
///
/// The callback is called before each tool execution. Return values:
///   0 (allow): proceed with execution.
///   1 (deny): skip tool, use deny reason if provided.
///   2 (cancel): stop the entire loop.
pub export fn talu_agent_set_before_tool(
    handle: ?*TaluAgent,
    before_fn: ?OnBeforeToolFn,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    agent_ptr.on_before_tool = before_fn;
    agent_ptr.on_before_tool_data = user_data;
    return 0;
}

/// Set the after-tool callback on an existing agent.
///
/// The callback is called after each tool execution. Return false to cancel
/// the loop.
pub export fn talu_agent_set_after_tool(
    handle: ?*TaluAgent,
    after_fn: ?OnAfterToolFn,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    agent_ptr.on_after_tool = after_fn;
    agent_ptr.on_after_tool_data = user_data;
    return 0;
}

// =============================================================================
// Bus notification — C API
// =============================================================================

/// Set a notification callback on a bus mailbox.
///
/// The callback fires (outside the bus lock) whenever a message is enqueued
/// for the given agent. Pass null to clear the notification.
pub export fn talu_agent_bus_set_notify(
    handle: ?*TaluAgentBus,
    agent_id: ?[*:0]const u8,
    notify_fn: ?OnMessageNotifyFn,
    user_data: ?*anyopaque,
) callconv(.c) i32 {
    capi_error.clearError();

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const id_z = agent_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    bus_ptr.setNotify(std.mem.span(id_z), notify_fn, user_data) catch |err| {
        capi_error.setError(err, "failed to set bus notify", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

// =============================================================================
// Agent active receiver — C API
// =============================================================================

/// Block until a message arrives or timeout elapses.
///
/// timeout_ms: maximum wait in milliseconds (0 = check immediately).
/// out_received: set to 1 if a message arrived, 0 on timeout.
/// Thread-safe: can be called from any thread.
pub export fn talu_agent_wait_for_message(
    handle: ?*TaluAgent,
    timeout_ms: u64,
    out_received: ?*u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const timeout_ns = timeout_ms * std.time.ns_per_ms;
    const received = agent_ptr.waitForMessage(timeout_ns);

    if (out_received) |out| {
        out.* = if (received) 1 else 0;
    }

    return 0;
}

/// Run an autonomous message-driven loop.
///
/// Runs heartbeat when messages arrive, waits between iterations.
/// Stops on abort(), generation error, or non-recoverable stop reason.
/// idle_timeout_ms controls max wait between message checks.
pub export fn talu_agent_run_loop(
    handle: ?*TaluAgent,
    idle_timeout_ms: u64,
    out_result: ?*CAgentLoopResult,
) callconv(.c) i32 {
    capi_error.clearError();

    const agent_ptr: *Agent = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const idle_timeout_ns = idle_timeout_ms * std.time.ns_per_ms;
    const result = agent_ptr.runLoop(idle_timeout_ns) catch |err| {
        capi_error.setError(err, "agent run loop failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    writeLoopResult(out_result, result);
    return 0;
}

/// Write an AgentLoopResult to the C output struct.
fn writeLoopResult(out: ?*CAgentLoopResult, result: AgentLoopResult) void {
    const out_ptr = out orelse return;
    out_ptr.* = std.mem.zeroes(CAgentLoopResult);
    out_ptr.stop_reason = @intFromEnum(result.stop_reason);
    out_ptr.iterations = result.iterations;
    out_ptr.total_tool_calls = result.total_tool_calls;
}

// =============================================================================
// Vector Store RAG — C API
// =============================================================================

/// C-compatible configuration for built-in vector store RAG.
///
/// All-zeroed is a valid (no-op) config: top_k=0, no callbacks.
pub const CRagConfig = extern struct {
    /// Maximum number of results to retrieve. 0 = use default (5).
    top_k: u32 = 5,
    /// Minimum score threshold. 0.0 = no filtering.
    min_score: f32 = 0.0,
    /// Callback to embed text into a query vector.
    on_embed: ?OnEmbedFn = null,
    on_embed_data: ?*anyopaque = null,
    /// Callback to resolve doc IDs + scores into context text.
    on_resolve: ?OnResolveDocFn = null,
    on_resolve_data: ?*anyopaque = null,
};

/// Wire a vector store to the agent for built-in RAG.
///
/// Before each generation, the agent embeds the last user message, searches the
/// store, resolves results via the callback, and injects context.
///
/// Both agent_handle and store_handle must be non-null.
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_set_vector_store(
    agent_handle: ?*TaluAgent,
    store_handle: ?*VectorStoreHandle,
    config_ptr: ?*const CRagConfig,
) callconv(.c) i32 {
    capi_error.clearError();
    const agent: *Agent = @ptrCast(@alignCast(agent_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));
    const store: *VectorAdapter = @ptrCast(@alignCast(store_handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "store handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const config = if (config_ptr) |c| RagConfig{
        .top_k = if (c.top_k == 0) 5 else c.top_k,
        .min_score = c.min_score,
        .on_embed = c.on_embed,
        .on_embed_data = c.on_embed_data,
        .on_resolve = c.on_resolve,
        .on_resolve_data = c.on_resolve_data,
    } else RagConfig{};

    agent.setVectorStore(store, config);
    return 0;
}

// =============================================================================
// MessageBus — C API
// =============================================================================

/// Create a new message bus.
///
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_bus_create(
    out_bus: ?*?*TaluAgentBus,
) callconv(.c) i32 {
    capi_error.clearError();
    const out = out_bus orelse {
        capi_error.setErrorWithCode(.invalid_argument, "out_bus is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    out.* = null;

    const bus_ptr = allocator.create(MessageBus) catch |err| {
        capi_error.setError(err, "failed to allocate message bus", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    bus_ptr.* = MessageBus.init(allocator);
    out.* = @ptrCast(bus_ptr);
    return 0;
}

/// Free a message bus and all its resources.
pub export fn talu_agent_bus_free(
    handle: ?*TaluAgentBus,
) callconv(.c) void {
    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse return));
    bus_ptr.deinit();
    allocator.destroy(bus_ptr);
}

/// Register a local agent on the bus, creating its mailbox.
pub export fn talu_agent_bus_register(
    handle: ?*TaluAgentBus,
    agent_id: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const id_z = agent_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    bus_ptr.register(std.mem.span(id_z)) catch |err| {
        capi_error.setError(err, "failed to register agent on bus", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Unregister a local agent and free its mailbox.
pub export fn talu_agent_bus_unregister(
    handle: ?*TaluAgentBus,
    agent_id: ?[*:0]const u8,
) callconv(.c) void {
    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse return));
    const id_z = agent_id orelse return;
    bus_ptr.unregister(std.mem.span(id_z));
}

/// Register a remote peer agent reachable at the given URL.
/// transport: 0 = HTTP.
pub export fn talu_agent_bus_add_peer(
    handle: ?*TaluAgentBus,
    agent_id: ?[*:0]const u8,
    url: ?[*:0]const u8,
    transport: u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const id_z = agent_id orelse {
        capi_error.setErrorWithCode(.invalid_argument, "agent_id is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const url_z = url orelse {
        capi_error.setErrorWithCode(.invalid_argument, "url is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const peer_transport: agent_mod.PeerTransport = std.meta.intToEnum(
        agent_mod.PeerTransport,
        transport,
    ) catch {
        capi_error.setErrorWithCode(.invalid_argument, "invalid transport value", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    bus_ptr.addPeer(std.mem.span(id_z), std.mem.span(url_z), peer_transport) catch |err| {
        capi_error.setError(err, "failed to add peer", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Remove a remote peer. Silently ignores unknown peer IDs.
pub export fn talu_agent_bus_remove_peer(
    handle: ?*TaluAgentBus,
    agent_id: ?[*:0]const u8,
) callconv(.c) void {
    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse return));
    const id_z = agent_id orelse return;
    bus_ptr.removePeer(std.mem.span(id_z));
}

/// Send a message from one agent to another.
/// Use "*" as `to` for broadcast.
pub export fn talu_agent_bus_send(
    handle: ?*TaluAgentBus,
    from: ?[*:0]const u8,
    to: ?[*:0]const u8,
    payload: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const from_z = from orelse {
        capi_error.setErrorWithCode(.invalid_argument, "from is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const to_z = to orelse {
        capi_error.setErrorWithCode(.invalid_argument, "to is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const payload_z = payload orelse {
        capi_error.setErrorWithCode(.invalid_argument, "payload is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    bus_ptr.send(std.mem.span(from_z), std.mem.span(to_z), std.mem.span(payload_z)) catch |err| {
        capi_error.setError(err, "failed to send message", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Deliver an incoming message into a local mailbox.
/// Called by the host's HTTP server when a remote message arrives.
pub export fn talu_agent_bus_deliver(
    handle: ?*TaluAgentBus,
    from: ?[*:0]const u8,
    to: ?[*:0]const u8,
    payload: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const from_z = from orelse {
        capi_error.setErrorWithCode(.invalid_argument, "from is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const to_z = to orelse {
        capi_error.setErrorWithCode(.invalid_argument, "to is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const payload_z = payload orelse {
        capi_error.setErrorWithCode(.invalid_argument, "payload is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    bus_ptr.deliver(std.mem.span(from_z), std.mem.span(to_z), std.mem.span(payload_z)) catch |err| {
        capi_error.setError(err, "failed to deliver message", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Broadcast a message from one agent to all others.
pub export fn talu_agent_bus_broadcast(
    handle: ?*TaluAgentBus,
    from: ?[*:0]const u8,
    payload: ?[*:0]const u8,
) callconv(.c) i32 {
    capi_error.clearError();

    const bus_ptr: *MessageBus = @ptrCast(@alignCast(handle orelse {
        capi_error.setErrorWithCode(.invalid_argument, "bus handle is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    }));

    const from_z = from orelse {
        capi_error.setErrorWithCode(.invalid_argument, "from is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const payload_z = payload orelse {
        capi_error.setErrorWithCode(.invalid_argument, "payload is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    bus_ptr.broadcast(std.mem.span(from_z), std.mem.span(payload_z)) catch |err| {
        capi_error.setError(err, "failed to broadcast message", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    return 0;
}

/// Number of messages waiting in an agent's mailbox.
/// Returns 0 for unknown agents.
pub export fn talu_agent_bus_pending(
    handle: ?*const TaluAgentBus,
    agent_id: ?[*:0]const u8,
) callconv(.c) usize {
    const bus_ptr: *MessageBus = @ptrCast(@alignCast(@constCast(handle orelse return 0)));
    const id_z = agent_id orelse return 0;
    return bus_ptr.pendingCount(std.mem.span(id_z));
}

// =============================================================================
// Agent + Bus tests
// =============================================================================

test "CAgentCreateConfig zeroed is valid" {
    const config = std.mem.zeroes(CAgentCreateConfig);
    try std.testing.expectEqual(@as(u64, 0), config.context_limit);
    try std.testing.expectEqual(@as(u32, 0), config.compaction_threshold_pct);
    try std.testing.expectEqual(@as(u32, 0), config.max_retries);
    try std.testing.expectEqual(@as(u32, 0), config.retry_base_delay_ms);
    try std.testing.expectEqual(@as(usize, 0), config.max_iterations_per_turn);
    try std.testing.expectEqual(@as(usize, 0), config.max_tool_calls_per_turn);
    try std.testing.expectEqual(@as(u8, 0), config.abort_on_tool_error);
    try std.testing.expect(config.state_dir == null);
    try std.testing.expect(config.agent_id == null);
    try std.testing.expect(config.on_token == null);
    try std.testing.expect(config.on_event == null);
    try std.testing.expect(config.on_session == null);
    try std.testing.expect(config.on_context_inject == null);
    try std.testing.expect(config.on_context_inject_data == null);
    try std.testing.expect(config.on_before_tool == null);
    try std.testing.expect(config.on_after_tool == null);
    try std.testing.expectEqual(@as(usize, 0), config.max_tool_output_bytes);
    try std.testing.expect(config.builtin_workspace_dir == null);
    try std.testing.expectEqual(@as(usize, 0), config.builtin_file_max_read_bytes);
    try std.testing.expectEqual(@as(usize, 0), config.builtin_http_max_response_bytes);
    try std.testing.expect(config.memory_db_path == null);
    try std.testing.expect(config.memory_namespace == null);
    try std.testing.expect(config.memory_owner_id == null);
    try std.testing.expectEqual(@as(usize, 0), config.memory_recall_limit);
    try std.testing.expectEqual(@as(u8, 0), config.memory_append_on_run);
}

test "talu_agent_create null backend returns error" {
    var out: ?*TaluAgent = null;
    const config = std.mem.zeroes(CAgentCreateConfig);
    const rc = talu_agent_create(null, &config, &out);
    try std.testing.expect(rc != 0);
    try std.testing.expect(out == null);
}

test "talu_agent_create null out returns error" {
    const config = std.mem.zeroes(CAgentCreateConfig);
    const rc = talu_agent_create(null, &config, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_create maps tool and memory config into core AgentConfig" {
    const test_allocator = std.testing.allocator;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const workspace = try tmp.dir.realpathAlloc(test_allocator, ".");
    defer test_allocator.free(workspace);
    const workspace_z = try test_allocator.dupeZ(u8, workspace);
    defer test_allocator.free(workspace_z);

    var backend: InferenceBackend = undefined;
    var out: ?*TaluAgent = null;

    var config = std.mem.zeroes(CAgentCreateConfig);
    config.agent_id = "agent-capi-map";
    config.max_tool_calls_per_turn = 9;
    config.builtin_workspace_dir = workspace_z;
    config.memory_db_path = workspace_z;
    config.memory_namespace = "agent_capi_map";
    config.memory_recall_limit = 4;
    config.memory_append_on_run = 1;

    const rc = talu_agent_create(@ptrCast(&backend), &config, &out);
    try std.testing.expectEqual(@as(i32, 0), rc);
    defer talu_agent_free(out);

    const agent_ptr: *Agent = @ptrCast(@alignCast(out.?));
    try std.testing.expectEqual(@as(usize, 9), agent_ptr.max_tool_calls_per_turn);
    try std.testing.expectEqual(@as(usize, 4), agent_ptr.getRegistry().count());
    try std.testing.expect(agent_ptr.memory_store != null);
    try std.testing.expectEqual(@as(usize, 4), agent_ptr.memory_recall_limit);
    try std.testing.expect(agent_ptr.memory_append_on_run);
}

test "talu_agent_free null is safe" {
    talu_agent_free(null);
}

test "talu_agent_set_system null handle returns error" {
    const rc = talu_agent_set_system(null, "test");
    try std.testing.expect(rc != 0);
}

test "talu_agent_abort null is safe" {
    talu_agent_abort(null);
}

test "talu_agent_get_chat null returns null" {
    const result = talu_agent_get_chat(null);
    try std.testing.expect(result == null);
}

test "talu_agent_bus_create and free" {
    var out: ?*TaluAgentBus = null;
    const rc = talu_agent_bus_create(&out);
    try std.testing.expectEqual(@as(i32, 0), rc);
    try std.testing.expect(out != null);
    talu_agent_bus_free(out);
}

test "talu_agent_bus_create null out returns error" {
    const rc = talu_agent_bus_create(null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_bus_free null is safe" {
    talu_agent_bus_free(null);
}

test "talu_agent_bus_register and send" {
    var bus_out: ?*TaluAgentBus = null;
    _ = talu_agent_bus_create(&bus_out);
    defer talu_agent_bus_free(bus_out);

    try std.testing.expectEqual(@as(i32, 0), talu_agent_bus_register(bus_out, "alice"));
    try std.testing.expectEqual(@as(i32, 0), talu_agent_bus_register(bus_out, "bob"));

    try std.testing.expectEqual(@as(i32, 0), talu_agent_bus_send(bus_out, "alice", "bob", "hello"));
    try std.testing.expectEqual(@as(usize, 1), talu_agent_bus_pending(@ptrCast(bus_out), "bob"));
}

test "talu_agent_bus_pending null returns 0" {
    try std.testing.expectEqual(@as(usize, 0), talu_agent_bus_pending(null, "x"));
}

test "talu_agent_bus_unregister null is safe" {
    talu_agent_bus_unregister(null, "x");
}

test "talu_agent_bus_remove_peer null is safe" {
    talu_agent_bus_remove_peer(null, "x");
}

// =============================================================================
// Goal + context injection C API tests
// =============================================================================

test "talu_agent_add_goal null handle returns error" {
    const rc = talu_agent_add_goal(null, "some goal");
    try std.testing.expect(rc != 0);
}

test "talu_agent_add_goal null goal returns error" {
    const rc = talu_agent_add_goal(@ptrFromInt(0x1), null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_remove_goal null handle returns error" {
    const rc = talu_agent_remove_goal(null, "some goal");
    try std.testing.expect(rc != 0);
}

test "talu_agent_clear_goals null is safe" {
    talu_agent_clear_goals(null);
}

test "talu_agent_goal_count null returns 0" {
    try std.testing.expectEqual(@as(usize, 0), talu_agent_goal_count(null));
}

test "talu_agent_set_context_inject null handle returns error" {
    const rc = talu_agent_set_context_inject(null, null, null);
    try std.testing.expect(rc != 0);
}

// =============================================================================
// Tool middleware + active receiver C API tests
// =============================================================================

test "CAgentCreateConfig zeroed has null tool middleware" {
    const config = std.mem.zeroes(CAgentCreateConfig);
    try std.testing.expect(config.on_before_tool == null);
    try std.testing.expect(config.on_before_tool_data == null);
    try std.testing.expect(config.on_after_tool == null);
    try std.testing.expect(config.on_after_tool_data == null);
}

test "talu_agent_set_before_tool null handle returns error" {
    const rc = talu_agent_set_before_tool(null, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_set_after_tool null handle returns error" {
    const rc = talu_agent_set_after_tool(null, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_bus_set_notify null handle returns error" {
    const rc = talu_agent_bus_set_notify(null, "agent", null, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_bus_set_notify null agent_id returns error" {
    var bus_out: ?*TaluAgentBus = null;
    _ = talu_agent_bus_create(&bus_out);
    defer talu_agent_bus_free(bus_out);

    const rc = talu_agent_bus_set_notify(bus_out, null, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_bus_set_notify on registered agent succeeds" {
    var bus_out: ?*TaluAgentBus = null;
    _ = talu_agent_bus_create(&bus_out);
    defer talu_agent_bus_free(bus_out);

    _ = talu_agent_bus_register(bus_out, "alice");

    const DummyNotify = struct {
        fn cb(_: [*:0]const u8, _: usize, _: ?*anyopaque) callconv(.c) void {}
    };
    const rc = talu_agent_bus_set_notify(bus_out, "alice", DummyNotify.cb, null);
    try std.testing.expectEqual(@as(i32, 0), rc);
}

test "talu_agent_wait_for_message null handle returns error" {
    const rc = talu_agent_wait_for_message(null, 0, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_run_loop null handle returns error" {
    const rc = talu_agent_run_loop(null, 0, null);
    try std.testing.expect(rc != 0);
}

// =============================================================================
// Vector Store RAG C API tests
// =============================================================================

test "CRagConfig zeroed is valid" {
    const config = std.mem.zeroes(CRagConfig);
    try std.testing.expectEqual(@as(u32, 0), config.top_k);
    try std.testing.expectEqual(@as(f32, 0.0), config.min_score);
    try std.testing.expect(config.on_embed == null);
    try std.testing.expect(config.on_resolve == null);
}

test "talu_agent_set_vector_store null agent returns error" {
    const rc = talu_agent_set_vector_store(null, null, null);
    try std.testing.expect(rc != 0);
}

test "talu_agent_set_vector_store null store returns error" {
    // Use a dummy non-null value for agent handle to reach the store check.
    var dummy: u8 = 0;
    const fake_agent: *TaluAgent = @ptrCast(&dummy);
    const rc = talu_agent_set_vector_store(fake_agent, null, null);
    try std.testing.expect(rc != 0);
}

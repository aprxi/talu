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
const agent_mod = @import("../agent/root.zig");
const chat_mod = @import("../responses/chat.zig");
const router_mod = @import("../router/root.zig");
const capi_error = @import("error.zig");
const error_codes = @import("error_codes.zig");

const ToolRegistry = agent_mod.ToolRegistry;
const Tool = agent_mod.Tool;
const ToolResult = agent_mod.ToolResult;
const Agent = agent_mod.Agent;
const AgentConfig = agent_mod.AgentConfig;
const AgentLoopResult = agent_mod.AgentLoopResult;
const MessageBus = agent_mod.MessageBus;
const BusError = agent_mod.BusError;
const OnTokenFn = agent_mod.OnTokenFn;
const OnEventFn = agent_mod.OnEventFn;
const OnContextFn = agent_mod.OnContextFn;
const OnBeforeToolFn = agent_mod.OnBeforeToolFn;
const OnAfterToolFn = agent_mod.OnAfterToolFn;
const OnSessionFn = agent_mod.OnSessionFn;
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

// =============================================================================
// C callback tool wrapper
// =============================================================================

/// C function pointer type for tool execution.
/// Returns 0 on success, non-zero on error.
pub const CToolExecuteFn = *const fn (
    user_data: ?*anyopaque,
    args_ptr: [*]const u8,
    args_len: usize,
    out_ptr: *?[*]u8,
    out_len: *usize,
    out_is_error: *u8,
) callconv(.c) i32;

/// Wraps a C callback as a Tool vtable implementation.
const CCallbackTool = struct {
    tool_name: [:0]u8,
    tool_description: [:0]u8,
    tool_schema: [:0]u8,
    execute_fn: CToolExecuteFn,
    user_data: ?*anyopaque,

    fn name(ctx: *anyopaque) []const u8 {
        const self: *CCallbackTool = @ptrCast(@alignCast(ctx));
        return self.tool_name;
    }

    fn description(ctx: *anyopaque) []const u8 {
        const self: *CCallbackTool = @ptrCast(@alignCast(ctx));
        return self.tool_description;
    }

    fn parametersSchema(ctx: *anyopaque) []const u8 {
        const self: *CCallbackTool = @ptrCast(@alignCast(ctx));
        return self.tool_schema;
    }

    fn execute(ctx: *anyopaque, alloc: std.mem.Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *CCallbackTool = @ptrCast(@alignCast(ctx));

        var out_ptr: ?[*]u8 = null;
        var out_len: usize = 0;
        var out_is_error: u8 = 0;

        const rc = self.execute_fn(
            self.user_data,
            arguments_json.ptr,
            arguments_json.len,
            &out_ptr,
            &out_len,
            &out_is_error,
        );

        if (rc != 0) {
            return error.ToolExecutionFailed;
        }

        // Copy output from C-allocated buffer to Zig allocator
        const c_output = if (out_ptr) |ptr| ptr[0..out_len] else "";
        const output = try alloc.dupe(u8, c_output);

        // Free C-allocated output
        if (out_ptr) |ptr| {
            std.heap.c_allocator.free(ptr[0..out_len]);
        }

        return ToolResult{
            .output = output,
            .is_error = out_is_error != 0,
        };
    }

    fn deinitFn(ctx: *anyopaque) void {
        const self: *CCallbackTool = @ptrCast(@alignCast(ctx));
        allocator.free(self.tool_name);
        allocator.free(self.tool_description);
        allocator.free(self.tool_schema);
        allocator.destroy(self);
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = execute,
        .deinit = deinitFn,
    };
};

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

    const name_z = name_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "name is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const desc_z = description_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "description is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const schema_z = parameters_schema_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "parameters_schema is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const exec_fn = execute_fn orelse {
        capi_error.setErrorWithCode(.invalid_argument, "execute_fn is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    // Create the CCallbackTool
    const callback_tool = allocator.create(CCallbackTool) catch |err| {
        capi_error.setError(err, "failed to allocate tool", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    errdefer allocator.destroy(callback_tool);

    const name_slice = std.mem.span(name_z);
    const desc_slice = std.mem.span(desc_z);
    const schema_slice = std.mem.span(schema_z);

    callback_tool.* = .{
        .tool_name = allocator.dupeZ(u8, name_slice) catch |err| {
            capi_error.setError(err, "failed to allocate tool name", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        },
        .tool_description = allocator.dupeZ(u8, desc_slice) catch |err| {
            allocator.free(callback_tool.tool_name);
            capi_error.setError(err, "failed to allocate tool description", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        },
        .tool_schema = allocator.dupeZ(u8, schema_slice) catch |err| {
            allocator.free(callback_tool.tool_name);
            allocator.free(callback_tool.tool_description);
            capi_error.setError(err, "failed to allocate tool schema", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        },
        .execute_fn = exec_fn,
        .user_data = user_data,
    };

    const tool = Tool.init(callback_tool, &CCallbackTool.vtable);

    registry.register(tool) catch |err| {
        tool.deinit();
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
    chat_handle: ?*@import("responses.zig").ChatHandle,
    backend_handle: ?*@import("router.zig").TaluInferenceBackend,
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

    // Build config from C struct
    var loop_config = agent_mod.AgentLoopConfig{};
    if (config_ptr) |cfg| {
        loop_config.max_iterations = cfg.max_iterations;
        loop_config.abort_on_tool_error = cfg.abort_on_tool_error != 0;
        loop_config.stop_flag = cfg.stop_flag;
        loop_config.on_token = cfg.on_token;
        loop_config.on_token_data = cfg.on_token_data;
        loop_config.on_event = cfg.on_event;
        loop_config.on_event_data = cfg.on_event_data;
        loop_config.generate_config = cfg.generate_config;
        loop_config.initial_iteration_count = cfg.initial_iteration_count;
        loop_config.initial_tool_call_count = cfg.initial_tool_call_count;
        loop_config.on_context = cfg.on_context;
        loop_config.on_context_data = cfg.on_context_data;
        loop_config.on_before_tool = cfg.on_before_tool;
        loop_config.on_before_tool_data = cfg.on_before_tool_data;
        loop_config.on_after_tool = cfg.on_after_tool;
        loop_config.on_after_tool_data = cfg.on_after_tool_data;
        loop_config.on_session = cfg.on_session;
        loop_config.on_session_data = cfg.on_session_data;
    }

    const result = agent_mod.run(allocator, chat, backend, registry, loop_config) catch |err| {
        capi_error.setError(err, "agent loop failed", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };

    // Write result to output struct
    if (out_result) |out| {
        out.* = std.mem.zeroes(CAgentLoopResult);
        out.stop_reason = @intFromEnum(result.stop_reason);
        out.iterations = result.iterations;
        out.total_tool_calls = result.total_tool_calls;
    }

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
};

/// Create a stateful agent. Caller owns the returned handle.
///
/// The backend must outlive the agent (caller-managed lifecycle).
/// Returns 0 on success, error code on failure.
pub export fn talu_agent_create(
    backend_handle: ?*@import("router.zig").TaluInferenceBackend,
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

    // Build AgentConfig from C struct
    var agent_config = AgentConfig{};
    if (config_ptr) |cfg| {
        agent_config.context_limit = cfg.context_limit;
        agent_config.compaction_threshold = @as(f32, @floatFromInt(cfg.compaction_threshold_pct)) / 100.0;
        agent_config.max_retries = cfg.max_retries;
        agent_config.retry_base_delay_ns = @as(u64, cfg.retry_base_delay_ms) * std.time.ns_per_ms;
        agent_config.max_iterations_per_turn = cfg.max_iterations_per_turn;
        agent_config.abort_on_tool_error = cfg.abort_on_tool_error != 0;
        agent_config.on_token = cfg.on_token;
        agent_config.on_token_data = cfg.on_token_data;
        agent_config.on_event = cfg.on_event;
        agent_config.on_event_data = cfg.on_event_data;
        agent_config.on_session = cfg.on_session;
        agent_config.on_session_data = cfg.on_session_data;

        if (cfg.agent_id) |id_z| {
            agent_config.agent_id = std.mem.span(id_z);
        }
        if (cfg.state_dir) |dir_z| {
            agent_config.state_dir = std.mem.span(dir_z);
        }
    }

    const agent_ptr = Agent.init(allocator, backend, agent_config) catch |err| {
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

    const name_z = name_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "name is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const desc_z = description_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "description is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const schema_z = parameters_schema_ptr orelse {
        capi_error.setErrorWithCode(.invalid_argument, "parameters_schema is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };
    const exec_fn = execute_fn orelse {
        capi_error.setErrorWithCode(.invalid_argument, "execute_fn is null", .{});
        return @intFromEnum(error_codes.ErrorCode.invalid_argument);
    };

    const callback_tool = allocator.create(CCallbackTool) catch |err| {
        capi_error.setError(err, "failed to allocate tool", .{});
        return @intFromEnum(error_codes.errorToCode(err));
    };
    errdefer allocator.destroy(callback_tool);

    callback_tool.* = .{
        .tool_name = allocator.dupeZ(u8, std.mem.span(name_z)) catch |err| {
            capi_error.setError(err, "failed to allocate tool name", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        },
        .tool_description = allocator.dupeZ(u8, std.mem.span(desc_z)) catch |err| {
            allocator.free(callback_tool.tool_name);
            capi_error.setError(err, "failed to allocate tool description", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        },
        .tool_schema = allocator.dupeZ(u8, std.mem.span(schema_z)) catch |err| {
            allocator.free(callback_tool.tool_name);
            allocator.free(callback_tool.tool_description);
            capi_error.setError(err, "failed to allocate tool schema", .{});
            return @intFromEnum(error_codes.errorToCode(err));
        },
        .execute_fn = exec_fn,
        .user_data = user_data,
    };

    const tool_iface = Tool.init(callback_tool, &CCallbackTool.vtable);
    agent_ptr.registerTool(tool_iface) catch |err| {
        tool_iface.deinit();
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
) callconv(.c) ?*const @import("responses.zig").ChatHandle {
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

/// Write an AgentLoopResult to the C output struct.
fn writeLoopResult(out: ?*CAgentLoopResult, result: AgentLoopResult) void {
    const out_ptr = out orelse return;
    out_ptr.* = std.mem.zeroes(CAgentLoopResult);
    out_ptr.stop_reason = @intFromEnum(result.stop_reason);
    out_ptr.iterations = result.iterations;
    out_ptr.total_tool_calls = result.total_tool_calls;
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
    try std.testing.expectEqual(@as(u8, 0), config.abort_on_tool_error);
    try std.testing.expect(config.state_dir == null);
    try std.testing.expect(config.agent_id == null);
    try std.testing.expect(config.on_token == null);
    try std.testing.expect(config.on_event == null);
    try std.testing.expect(config.on_session == null);
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

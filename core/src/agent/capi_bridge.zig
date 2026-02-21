//! Internal bridge helpers used by C API glue.
//!
//! Keeps conversion and adapter logic in core modules so `src/capi/` exports
//! remain thin boundary functions.

const std = @import("std");
const Allocator = std.mem.Allocator;

const agent_mod = @import("agent.zig");
const loop_mod = @import("loop.zig");
const tool_mod = @import("tool.zig");
const responses = @import("../responses/root.zig");
const router = @import("../router/root.zig");

const Tool = tool_mod.Tool;
const ToolRegistry = tool_mod.ToolRegistry;
const ToolResult = tool_mod.ToolResult;
const Agent = agent_mod.Agent;
const Chat = responses.Chat;
const InferenceBackend = router.InferenceBackend;

/// C callback function pointer type used by callback-backed tools.
pub const CToolExecuteFn = *const fn (
    user_data: ?*anyopaque,
    args_ptr: [*]const u8,
    args_len: usize,
    out_ptr: *?[*]u8,
    out_len: *usize,
    out_is_error: *u8,
) callconv(.c) i32;

pub const CallbackToolArgs = struct {
    name: []const u8,
    description: []const u8,
    schema: []const u8,
    execute_fn: CToolExecuteFn,
};

pub const CallbackToolArgsError = error{
    NullName,
    NullDescription,
    NullSchema,
    NullExecuteFn,
};

const CallbackTool = struct {
    allocator: Allocator,
    tool_name: [:0]u8,
    tool_description: [:0]u8,
    tool_schema: [:0]u8,
    execute_fn: CToolExecuteFn,
    user_data: ?*anyopaque,

    fn name(ctx: *anyopaque) []const u8 {
        const self: *CallbackTool = @ptrCast(@alignCast(ctx));
        return self.tool_name;
    }

    fn description(ctx: *anyopaque) []const u8 {
        const self: *CallbackTool = @ptrCast(@alignCast(ctx));
        return self.tool_description;
    }

    fn parametersSchema(ctx: *anyopaque) []const u8 {
        const self: *CallbackTool = @ptrCast(@alignCast(ctx));
        return self.tool_schema;
    }

    fn execute(ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *CallbackTool = @ptrCast(@alignCast(ctx));

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
        if (rc != 0) return error.ToolExecutionFailed;

        const c_output = if (out_ptr) |ptr| ptr[0..out_len] else "";
        const output = try allocator.dupe(u8, c_output);

        if (out_ptr) |ptr| {
            std.heap.c_allocator.free(ptr[0..out_len]);
        }

        return .{
            .output = output,
            .is_error = out_is_error != 0,
        };
    }

    fn deinitFn(ctx: *anyopaque) void {
        const self: *CallbackTool = @ptrCast(@alignCast(ctx));
        self.allocator.free(self.tool_name);
        self.allocator.free(self.tool_description);
        self.allocator.free(self.tool_schema);
        self.allocator.destroy(self);
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = execute,
        .deinit = deinitFn,
    };
};

/// Register a callback-backed tool in a registry.
pub fn registerCallbackTool(
    allocator: Allocator,
    registry: *ToolRegistry,
    name: []const u8,
    description: []const u8,
    schema: []const u8,
    execute_fn: CToolExecuteFn,
    user_data: ?*anyopaque,
) !void {
    const tool = try makeCallbackTool(
        allocator,
        name,
        description,
        schema,
        execute_fn,
        user_data,
    );
    errdefer tool.deinit();
    try registry.register(tool);
}

/// Parse callback tool arguments from C ABI pointers.
pub fn parseCallbackToolArgs(
    name_ptr: ?[*:0]const u8,
    description_ptr: ?[*:0]const u8,
    schema_ptr: ?[*:0]const u8,
    execute_fn: ?CToolExecuteFn,
) CallbackToolArgsError!CallbackToolArgs {
    const name_z = name_ptr orelse return error.NullName;
    const desc_z = description_ptr orelse return error.NullDescription;
    const schema_z = schema_ptr orelse return error.NullSchema;
    const exec_fn = execute_fn orelse return error.NullExecuteFn;
    return .{
        .name = std.mem.span(name_z),
        .description = std.mem.span(desc_z),
        .schema = std.mem.span(schema_z),
        .execute_fn = exec_fn,
    };
}

/// Register a callback-backed tool on an agent.
pub fn registerCallbackToolOnAgent(
    allocator: Allocator,
    agent: *Agent,
    name: []const u8,
    description: []const u8,
    schema: []const u8,
    execute_fn: CToolExecuteFn,
    user_data: ?*anyopaque,
) !void {
    const tool = try makeCallbackTool(
        allocator,
        name,
        description,
        schema,
        execute_fn,
        user_data,
    );
    errdefer tool.deinit();
    try agent.registerTool(tool);
}

fn makeCallbackTool(
    allocator: Allocator,
    name: []const u8,
    description: []const u8,
    schema: []const u8,
    execute_fn: CToolExecuteFn,
    user_data: ?*anyopaque,
) !Tool {
    const callback_tool = try allocator.create(CallbackTool);
    errdefer allocator.destroy(callback_tool);

    const name_z = try allocator.dupeZ(u8, name);
    errdefer allocator.free(name_z);
    const desc_z = try allocator.dupeZ(u8, description);
    errdefer allocator.free(desc_z);
    const schema_z = try allocator.dupeZ(u8, schema);
    errdefer allocator.free(schema_z);

    callback_tool.* = .{
        .allocator = allocator,
        .tool_name = name_z,
        .tool_description = desc_z,
        .tool_schema = schema_z,
        .execute_fn = execute_fn,
        .user_data = user_data,
    };
    return Tool.init(callback_tool, &CallbackTool.vtable);
}

/// Map C loop config struct fields into AgentLoopConfig.
pub fn mapLoopConfig(config_ptr: anytype) loop_mod.AgentLoopConfig {
    var loop_config = loop_mod.AgentLoopConfig{};
    if (config_ptr) |cfg| {
        loop_config.max_iterations = cfg.max_iterations;
        loop_config.max_tool_calls = cfg.max_tool_calls;
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
        loop_config.max_tool_output_bytes = cfg.max_tool_output_bytes;
    }
    return loop_config;
}

/// Execute the stateless agent loop using C-originated config.
pub fn runLoopWithConfig(
    allocator: Allocator,
    chat: *Chat,
    backend: *InferenceBackend,
    registry: *const ToolRegistry,
    config_ptr: anytype,
) !loop_mod.AgentLoopResult {
    return loop_mod.run(allocator, chat, backend, registry, mapLoopConfig(config_ptr));
}

/// Map C create config struct fields into AgentConfig.
pub fn mapAgentConfig(config_ptr: anytype) agent_mod.AgentConfig {
    var agent_config = agent_mod.AgentConfig{};
    if (config_ptr) |cfg| {
        agent_config.context_limit = cfg.context_limit;
        agent_config.compaction_threshold = @as(f32, @floatFromInt(cfg.compaction_threshold_pct)) / 100.0;
        agent_config.max_retries = cfg.max_retries;
        agent_config.retry_base_delay_ns = @as(u64, cfg.retry_base_delay_ms) * std.time.ns_per_ms;
        agent_config.max_iterations_per_turn = cfg.max_iterations_per_turn;
        agent_config.max_tool_calls_per_turn = cfg.max_tool_calls_per_turn;
        agent_config.abort_on_tool_error = cfg.abort_on_tool_error != 0;
        agent_config.on_token = cfg.on_token;
        agent_config.on_token_data = cfg.on_token_data;
        agent_config.on_event = cfg.on_event;
        agent_config.on_event_data = cfg.on_event_data;
        agent_config.on_session = cfg.on_session;
        agent_config.on_session_data = cfg.on_session_data;
        agent_config.on_context_inject = cfg.on_context_inject;
        agent_config.on_context_inject_data = cfg.on_context_inject_data;
        agent_config.on_before_tool = cfg.on_before_tool;
        agent_config.on_before_tool_data = cfg.on_before_tool_data;
        agent_config.on_after_tool = cfg.on_after_tool;
        agent_config.on_after_tool_data = cfg.on_after_tool_data;
        agent_config.max_tool_output_bytes = cfg.max_tool_output_bytes;

        agent_config.agent_id = spanOptionalZ(cfg.agent_id);
        agent_config.state_dir = spanOptionalZ(cfg.state_dir);

        if (spanOptionalZ(cfg.builtin_workspace_dir)) |workspace| {
            agent_config.builtin_tools = .{
                .workspace_dir = workspace,
                .file_max_read_bytes = cfg.builtin_file_max_read_bytes,
                .http_max_response_bytes = cfg.builtin_http_max_response_bytes,
            };
        }

        if (spanOptionalZ(cfg.memory_db_path)) |db_path| {
            agent_config.memory = .{
                .db_path = db_path,
                .namespace = spanOptionalZ(cfg.memory_namespace) orelse "default",
                .owner_id = spanOptionalZ(cfg.memory_owner_id),
                .recall_limit = cfg.memory_recall_limit,
                .append_on_run = cfg.memory_append_on_run != 0,
            };
        }
    }
    return agent_config;
}

/// Create a stateful agent using C-originated config.
pub fn createAgent(
    allocator: Allocator,
    backend: *InferenceBackend,
    config_ptr: anytype,
) !*Agent {
    return Agent.init(allocator, backend, mapAgentConfig(config_ptr));
}

fn spanOptionalZ(z: ?[*:0]const u8) ?[]const u8 {
    return if (z) |s| std.mem.span(s) else null;
}

test "mapLoopConfig copies optional fields" {
    const DummyLoopConfig = struct {
        max_iterations: usize = 8,
        max_tool_calls: usize = 3,
        abort_on_tool_error: u8 = 1,
        stop_flag: ?*const std.atomic.Value(bool) = null,
        on_token: ?loop_mod.OnTokenFn = null,
        on_token_data: ?*anyopaque = null,
        on_event: ?loop_mod.OnEventFn = null,
        on_event_data: ?*anyopaque = null,
        generate_config: ?*const router.capi_bridge.CGenerateConfig = null,
        initial_iteration_count: usize = 2,
        initial_tool_call_count: usize = 1,
        on_context: ?loop_mod.OnContextFn = null,
        on_context_data: ?*anyopaque = null,
        on_before_tool: ?loop_mod.OnBeforeToolFn = null,
        on_before_tool_data: ?*anyopaque = null,
        on_after_tool: ?loop_mod.OnAfterToolFn = null,
        on_after_tool_data: ?*anyopaque = null,
        on_session: ?loop_mod.OnSessionFn = null,
        on_session_data: ?*anyopaque = null,
        max_tool_output_bytes: usize = 128,
    };

    var src = DummyLoopConfig{};
    const mapped = mapLoopConfig(@as(?*const DummyLoopConfig, &src));
    try std.testing.expectEqual(@as(usize, 8), mapped.max_iterations);
    try std.testing.expectEqual(@as(usize, 3), mapped.max_tool_calls);
    try std.testing.expect(mapped.abort_on_tool_error);
    try std.testing.expectEqual(@as(usize, 2), mapped.initial_iteration_count);
    try std.testing.expectEqual(@as(usize, 1), mapped.initial_tool_call_count);
    try std.testing.expectEqual(@as(usize, 128), mapped.max_tool_output_bytes);
}

test "mapAgentConfig maps builtin and memory fields" {
    const DummyCreateConfig = struct {
        context_limit: u64 = 1024,
        compaction_threshold_pct: u32 = 75,
        max_retries: u32 = 1,
        retry_base_delay_ms: u32 = 50,
        max_iterations_per_turn: usize = 12,
        max_tool_calls_per_turn: usize = 7,
        abort_on_tool_error: u8 = 1,
        state_dir: ?[*:0]const u8 = "state",
        agent_id: ?[*:0]const u8 = "agent-id",
        on_token: ?loop_mod.OnTokenFn = null,
        on_token_data: ?*anyopaque = null,
        on_event: ?loop_mod.OnEventFn = null,
        on_event_data: ?*anyopaque = null,
        on_session: ?loop_mod.OnSessionFn = null,
        on_session_data: ?*anyopaque = null,
        on_context_inject: ?agent_mod.OnContextInjectFn = null,
        on_context_inject_data: ?*anyopaque = null,
        on_before_tool: ?loop_mod.OnBeforeToolFn = null,
        on_before_tool_data: ?*anyopaque = null,
        on_after_tool: ?loop_mod.OnAfterToolFn = null,
        on_after_tool_data: ?*anyopaque = null,
        max_tool_output_bytes: usize = 222,
        builtin_workspace_dir: ?[*:0]const u8 = "/tmp",
        builtin_file_max_read_bytes: usize = 64,
        builtin_http_max_response_bytes: usize = 128,
        memory_db_path: ?[*:0]const u8 = "/tmp/db",
        memory_namespace: ?[*:0]const u8 = "mem",
        memory_owner_id: ?[*:0]const u8 = "owner",
        memory_recall_limit: usize = 5,
        memory_append_on_run: u8 = 1,
    };

    var src = DummyCreateConfig{};
    const mapped = mapAgentConfig(@as(?*const DummyCreateConfig, &src));
    try std.testing.expectEqual(@as(u64, 1024), mapped.context_limit);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), mapped.compaction_threshold, 0.0001);
    try std.testing.expectEqual(@as(usize, 7), mapped.max_tool_calls_per_turn);
    try std.testing.expectEqualStrings("agent-id", mapped.agent_id.?);
    try std.testing.expectEqualStrings("state", mapped.state_dir.?);
    try std.testing.expect(mapped.builtin_tools != null);
    try std.testing.expectEqualStrings("/tmp", mapped.builtin_tools.?.workspace_dir);
    try std.testing.expect(mapped.memory != null);
    try std.testing.expectEqualStrings("/tmp/db", mapped.memory.?.db_path);
    try std.testing.expectEqualStrings("mem", mapped.memory.?.namespace);
    try std.testing.expectEqualStrings("owner", mapped.memory.?.owner_id.?);
    try std.testing.expectEqual(@as(usize, 5), mapped.memory.?.recall_limit);
    try std.testing.expect(mapped.memory.?.append_on_run);
}

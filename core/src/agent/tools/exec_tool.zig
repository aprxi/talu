//! Built-in `execute_command` tool for agent shell execution.
//!
//! Validates commands against the safety whitelist before execution.
//! Denied commands return a tool error with the denial reason.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Tool = @import("../tool.zig").Tool;
const ToolResult = @import("../tool.zig").ToolResult;
const shell = @import("../shell/root.zig");

pub const ExecCommandTool = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) !*ExecCommandTool {
        const self = try allocator.create(ExecCommandTool);
        self.* = .{ .allocator = allocator };
        return self;
    }

    pub fn deinit(self: *ExecCommandTool) void {
        self.allocator.destroy(self);
    }

    pub fn asTool(self: *ExecCommandTool) Tool {
        return Tool.init(self, &vtable);
    }

    fn name(_: *anyopaque) []const u8 {
        return "execute_command";
    }

    fn description(_: *anyopaque) []const u8 {
        return "Execute a shell command. The command is validated against a safety whitelist before execution.";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return
        \\{"type":"object","properties":{"command":{"type":"string","description":"Shell command to execute."}},"required":["command"],"additionalProperties":false}
        ;
    }

    fn executeFn(ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *ExecCommandTool = @ptrCast(@alignCast(ctx));
        return self.execute(allocator, arguments_json);
    }

    fn execute(_: *ExecCommandTool, allocator: Allocator, arguments_json: []const u8) !ToolResult {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, arguments_json, .{}) catch {
            return toolError(allocator, "Invalid JSON arguments.");
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return toolError(allocator, "Arguments must be a JSON object."),
        };

        const cmd_value = obj.get("command") orelse return toolError(allocator, "Missing required field: command.");
        const command = switch (cmd_value) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'command' must be a string."),
        };

        if (command.len == 0) {
            return toolError(allocator, "Field 'command' must be non-empty.");
        }

        // Safety check
        const check = shell.safety.checkCommand(command);
        if (!check.allowed) {
            return toolErrorFmt(allocator, "Command denied: {s}", .{check.reason orelse "blocked by safety policy"});
        }

        // Execute
        var result = shell.exec.exec(allocator, command) catch |err| {
            return toolErrorFmt(allocator, "Command execution failed: {s}", .{@errorName(err)});
        };
        defer result.deinit(allocator);

        return toolSuccessJson(allocator, .{
            .stdout = result.stdout,
            .stderr = result.stderr,
            .exit_code = result.exit_code,
        });
    }

    const vtable = Tool.VTable{
        .name = name,
        .description = description,
        .parametersSchema = parametersSchema,
        .execute = executeFn,
        .deinit = deinitOpaque,
    };

    fn deinitOpaque(ctx: *anyopaque) void {
        const self: *ExecCommandTool = @ptrCast(@alignCast(ctx));
        self.deinit();
    }
};

fn toolSuccessJson(allocator: Allocator, value: anytype) !ToolResult {
    const output = try std.fmt.allocPrint(allocator, "{f}", .{std.json.fmt(value, .{})});
    return .{
        .output = output,
        .is_error = false,
    };
}

fn toolError(allocator: Allocator, message: []const u8) !ToolResult {
    return .{
        .output = try allocator.dupe(u8, message),
        .is_error = true,
    };
}

fn toolErrorFmt(allocator: Allocator, comptime fmt: []const u8, args: anytype) !ToolResult {
    return .{
        .output = try std.fmt.allocPrint(allocator, fmt, args),
        .is_error = true,
    };
}

test "ExecCommandTool blocks non-whitelisted command" {
    const allocator = std.testing.allocator;

    var tool_impl = try ExecCommandTool.init(allocator);
    defer tool_impl.deinit();

    const args =
        \\{"command":"rm -rf /"}
    ;
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Command denied") != null);
}

test "ExecCommandTool executes whitelisted command" {
    const allocator = std.testing.allocator;

    var tool_impl = try ExecCommandTool.init(allocator);
    defer tool_impl.deinit();

    const args =
        \\{"command":"echo hello"}
    ;
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(!result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "hello") != null);
}

test "ExecCommandTool rejects empty command" {
    const allocator = std.testing.allocator;

    var tool_impl = try ExecCommandTool.init(allocator);
    defer tool_impl.deinit();

    const args =
        \\{"command":""}
    ;
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "non-empty") != null);
}

test "ExecCommandTool rejects missing command field" {
    const allocator = std.testing.allocator;

    var tool_impl = try ExecCommandTool.init(allocator);
    defer tool_impl.deinit();

    const args =
        \\{"cmd":"echo hi"}
    ;
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "Missing required field") != null);
}

test "ExecCommandTool blocks find -exec" {
    const allocator = std.testing.allocator;

    var tool_impl = try ExecCommandTool.init(allocator);
    defer tool_impl.deinit();

    const args =
        \\{"command":"find . -exec rm {} ;"}
    ;
    var result = try tool_impl.asTool().execute(allocator, args);
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "denied") != null);
}

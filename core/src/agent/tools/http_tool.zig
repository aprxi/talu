//! Built-in HTTP fetch tool for agent execution.
//!
//! Tool:
//! - `http_fetch`
//!
//! This uses the core libcurl transport wrapper.

const std = @import("std");
const Allocator = std.mem.Allocator;
const transport = @import("../../io/transport/root.zig");
const Tool = @import("../tool.zig").Tool;
const ToolResult = @import("../tool.zig").ToolResult;

const default_max_bytes: usize = 1024 * 1024;

pub const HttpFetchTool = struct {
    allocator: Allocator,
    max_response_bytes: usize,

    pub fn init(allocator: Allocator, max_response_bytes: usize) !*HttpFetchTool {
        const self = try allocator.create(HttpFetchTool);
        self.* = .{
            .allocator = allocator,
            .max_response_bytes = if (max_response_bytes == 0) default_max_bytes else max_response_bytes,
        };
        return self;
    }

    pub fn deinit(self: *HttpFetchTool) void {
        self.allocator.destroy(self);
    }

    pub fn asTool(self: *HttpFetchTool) Tool {
        return Tool.init(self, &vtable);
    }

    fn name(_: *anyopaque) []const u8 {
        return "http_fetch";
    }

    fn description(_: *anyopaque) []const u8 {
        return "Fetch UTF-8 text from an HTTP/HTTPS URL.";
    }

    fn parametersSchema(_: *anyopaque) []const u8 {
        return 
        \\{"type":"object","properties":{"url":{"type":"string","description":"HTTP or HTTPS URL."},"bearer_token":{"type":"string","description":"Optional bearer token."},"max_bytes":{"type":"integer","description":"Optional response size limit in bytes."}},"required":["url"],"additionalProperties":false}
        ;
    }

    fn executeFn(ctx: *anyopaque, allocator: Allocator, arguments_json: []const u8) anyerror!ToolResult {
        const self: *HttpFetchTool = @ptrCast(@alignCast(ctx));
        return self.execute(allocator, arguments_json);
    }

    fn execute(self: *HttpFetchTool, allocator: Allocator, arguments_json: []const u8) !ToolResult {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, arguments_json, .{}) catch {
            return toolError(allocator, "Invalid JSON arguments.");
        };
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |o| o,
            else => return toolError(allocator, "Arguments must be a JSON object."),
        };

        const url_value = obj.get("url") orelse return toolError(allocator, "Missing required field: url.");
        const url = switch (url_value) {
            .string => |s| s,
            else => return toolError(allocator, "Field 'url' must be a string."),
        };
        if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) {
            return toolError(allocator, "http_fetch only supports http:// and https:// URLs.");
        }

        var token: ?[]const u8 = null;
        if (obj.get("bearer_token")) |token_value| {
            token = switch (token_value) {
                .string => |s| s,
                else => return toolError(allocator, "Field 'bearer_token' must be a string."),
            };
        }

        var max_bytes = self.max_response_bytes;
        if (obj.get("max_bytes")) |max_value| {
            switch (max_value) {
                .integer => |n| {
                    if (n <= 0) return toolError(allocator, "Field 'max_bytes' must be > 0.");
                    max_bytes = @intCast(n);
                },
                else => return toolError(allocator, "Field 'max_bytes' must be an integer."),
            }
        }

        transport.globalInit();
        defer transport.globalCleanup();

        const body = transport.http.fetch(allocator, url, .{
            .token = token,
            .max_response_bytes = max_bytes,
            .user_agent = "talu-agent/1.0",
        }) catch |err| {
            return toolErrorFmt(allocator, "http_fetch failed: {s}", .{@errorName(err)});
        };
        defer allocator.free(body);

        if (!std.unicode.utf8ValidateSlice(body)) {
            return toolError(allocator, "http_fetch response is not valid UTF-8 text.");
        }

        return toolSuccessJson(allocator, .{
            .url = url,
            .bytes = body.len,
            .body = body,
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
        const self: *HttpFetchTool = @ptrCast(@alignCast(ctx));
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

test "HttpFetchTool.execute validates scheme" {
    const allocator = std.testing.allocator;
    var tool = try HttpFetchTool.init(allocator, 0);
    defer tool.deinit();

    var result = try tool.asTool().execute(allocator, "{\"url\":\"file:///etc/passwd\"}");
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "http:// and https://") != null);
}

test "HttpFetchTool.execute validates argument shape" {
    const allocator = std.testing.allocator;
    var tool = try HttpFetchTool.init(allocator, 0);
    defer tool.deinit();

    var result = try tool.asTool().execute(allocator, "[]");
    defer result.deinit(allocator);

    try std.testing.expect(result.is_error);
    try std.testing.expect(std.mem.indexOf(u8, result.output, "JSON object") != null);
}

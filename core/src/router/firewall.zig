//! Tool call firewall helpers.
//!
//! Centralizes policy evaluation for tool calls across engines.

const std = @import("std");
const io = @import("../io/root.zig");
const log = @import("../log.zig");
const responses_mod = @import("../responses/root.zig");
const policy_mod = @import("../policy/evaluate.zig");

const Chat = responses_mod.Chat;
const Policy = policy_mod.Policy;

/// Evaluate the tool call firewall.
/// Returns `true` if BLOCKED, `false` if ALLOWED.
pub fn checkFirewall(
    allocator: std.mem.Allocator,
    chat: *Chat,
    tool_name: []const u8,
    tool_args: []const u8,
) !bool {
    const policy = chat.policy orelse {
        log.debug("policy", "No policy attached to chat", .{}, @src());
        return false;
    };

    const action = extractPolicyAction(allocator, tool_name, tool_args) catch null;
    defer if (action) |a| allocator.free(a);

    if (action) |a| {
        const effect = policy.evaluate(a);
        if (effect == .deny) {
            if (policy.mode == .audit) {
                log.info("policy", "Audit: would deny tool call", .{});
                return false;
            }
            log.info("policy", "Policy DENIED tool call", .{});
            return true;
        }
        log.info("policy", "Policy ALLOWED tool call", .{});
        return false;
    }

    log.info("policy", "Could not extract action, defaulting to allow", .{});
    return false;
}

/// Extract the action string for policy evaluation from a parsed tool call.
///
/// For `execute_command`: extracts the `"command"` field from the arguments JSON.
/// For other tools: returns the tool name as the action (future: name + args).
///
/// Caller owns returned memory.
fn extractPolicyAction(
    alloc: std.mem.Allocator,
    tool_name: []const u8,
    arguments: []const u8,
) ![]u8 {
    // For execute_command, extract the "command" field value.
    if (std.mem.eql(u8, tool_name, "execute_command")) {
        if (io.json.extractStringField(alloc, arguments, "command", .{
            .max_size_bytes = 64_000,
        }) catch null) |cmd| {
            defer alloc.free(cmd);
            return try normalizeCommand(alloc, cmd);
        }
    }
    // Fallback: use the tool name itself as the action.
    return try alloc.dupe(u8, tool_name);
}

/// Replace the leading executable's absolute path with its basename.
/// "/bin/ls -la" → "ls -la", "/usr/bin/env FOO=1 git show" → "env FOO=1 git show".
/// Non-absolute commands pass through unchanged.
fn normalizeCommand(alloc: std.mem.Allocator, cmd: []const u8) ![]u8 {
    // Find the first space (end of first token).
    const first_space = std.mem.indexOfScalar(u8, cmd, ' ');
    const exe = if (first_space) |s| cmd[0..s] else cmd;

    // Only normalize if the executable starts with '/'.
    if (exe.len > 0 and exe[0] == '/') {
        // basename: everything after the last '/'.
        const last_slash = std.mem.lastIndexOfScalar(u8, exe, '/') orelse 0;
        const basename = exe[last_slash + 1 ..];
        if (basename.len == 0) return try alloc.dupe(u8, cmd);

        // Reconstruct: basename + rest of command.
        if (first_space) |s| {
            const rest = cmd[s..]; // includes the leading space
            const result = try alloc.alloc(u8, basename.len + rest.len);
            @memcpy(result[0..basename.len], basename);
            @memcpy(result[basename.len..], rest);
            return result;
        } else {
            return try alloc.dupe(u8, basename);
        }
    }

    return try alloc.dupe(u8, cmd);
}

test "checkFirewall returns blocked for denied action" {
    const alloc = std.testing.allocator;

    var chat = try responses_mod.Chat.init(alloc);
    defer chat.deinit();

    var policy = Policy{
        .default_effect = .allow,
        .mode = .enforce,
        .statements = &[_]policy_mod.Statement{
            .{ .effect = .deny, .action_pattern = "rm *" },
        },
        ._pattern_buf = &.{},
        .allocator = alloc,
    };

    chat.policy = &policy;
    const blocked = try checkFirewall(alloc, &chat, "execute_command", "{\"command\": \"rm -rf /\"}");
    try std.testing.expect(blocked);
}

test "checkFirewall returns allowed for permitted action" {
    const alloc = std.testing.allocator;

    var chat = try responses_mod.Chat.init(alloc);
    defer chat.deinit();

    var policy = Policy{
        .default_effect = .deny,
        .mode = .enforce,
        .statements = &[_]policy_mod.Statement{
            .{ .effect = .allow, .action_pattern = "ls *" },
        },
        ._pattern_buf = &.{},
        .allocator = alloc,
    };

    chat.policy = &policy;
    const blocked = try checkFirewall(alloc, &chat, "execute_command", "{\"command\": \"ls -la\"}");
    try std.testing.expect(!blocked);
}

test "checkFirewall audit mode logs but allows" {
    const alloc = std.testing.allocator;

    var chat = try responses_mod.Chat.init(alloc);
    defer chat.deinit();

    var policy = Policy{
        .default_effect = .allow,
        .mode = .audit,
        .statements = &[_]policy_mod.Statement{
            .{ .effect = .deny, .action_pattern = "rm *" },
        },
        ._pattern_buf = &.{},
        .allocator = alloc,
    };

    chat.policy = &policy;
    const blocked = try checkFirewall(alloc, &chat, "execute_command", "{\"command\": \"rm -rf /\"}");
    try std.testing.expect(!blocked);
}

test "checkFirewall no policy returns allowed" {
    const alloc = std.testing.allocator;

    var chat = try responses_mod.Chat.init(alloc);
    defer chat.deinit();

    const blocked = try checkFirewall(alloc, &chat, "execute_command", "{\"command\": \"ls\"}");
    try std.testing.expect(!blocked);
}

test "extractPolicyAction execute_command" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": \"ls -la\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("ls -la", action);
}

test "extractPolicyAction absolute path normalized" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": \"/bin/ls -la\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("ls -la", action);
}

test "extractPolicyAction usr bin path normalized" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": \"/usr/bin/git show HEAD\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("git show HEAD", action);
}

test "extractPolicyAction bare absolute no args" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": \"/bin/ls\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("ls", action);
}

test "extractPolicyAction fallback to tool name" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "get_weather", "{\"location\": \"Paris\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("get_weather", action);
}

test "extractPolicyAction handles escaped quotes" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": \"echo \\\"hello\\\"\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("echo \"hello\"", action);
}

test "extractPolicyAction handles unicode escapes" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": \"echo \\u0048ello\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("echo Hello", action);
}

test "extractPolicyAction malformed JSON falls back to tool name" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"command\": ");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("execute_command", action);
}

test "extractPolicyAction ignores nested command fields" {
    const alloc = std.testing.allocator;

    const action = try extractPolicyAction(alloc, "execute_command", "{\"meta\": {\"command\": \"rm\"}, \"command\": \"ls\"}");
    defer alloc.free(action);
    try std.testing.expectEqualStrings("ls", action);
}
